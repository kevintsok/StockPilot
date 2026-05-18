#!/usr/bin/env python3
"""
BC + RL Fine-tuning.

This implements the fine-tuning step after Behavior Cloning pretraining:
1. Load BC pretrained model (mimics expert strategy)
2. Fine-tune with RL using Sharpe-based reward
3. Goal: improve upon the BC policy through environment interaction

The RL approach uses policy gradient with advantage estimation.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .bc_pretrain_trainer import BCNetwork, ExpertTrajectoryCollector, prepare_data_with_predictions
from ..config import DATA_DIR
from ..data.storage import list_symbols


class AShareConstraints:
    """A-share trading constraints."""
    commission: float = 0.0003
    stamp_tax: float = 0.001
    slippage: float = 0.0001
    max_position_pct: float = 0.15


class TradingEnvironment:
    """Simple trading environment for RL fine-tuning."""

    def __init__(
        self,
        sequences: List[np.ndarray],
        start_idx: int = 60,
        constraints: AShareConstraints = None,
    ):
        self.sequences = sequences
        self.start_idx = start_idx
        self.constraints = constraints or AShareConstraints()
        self.reset()

    def reset(self, sym_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        self.current_sym = sym_idx
        self.current_idx = self.start_idx
        self.position = 0.0
        self.portfolio_value = 1.0
        self.entry_price = 0.0
        return self._get_state()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return (new_state, reward, done)."""
        action = float(np.clip(action, 0.0, self.constraints.max_position_pct))

        seq = self.sequences[self.current_sym]
        current_price = seq[self.current_idx, 3]

        # Compute return (this is the return from current to next timestep)
        # This is what we earn while holding a position from t to t+1
        if self.current_idx < len(seq) - 1:
            next_price = seq[self.current_idx + 1, 3]
            actual_return = (next_price - current_price) / current_price if current_price > 0 else 0.0
        else:
            actual_return = 0.0

        # Trading logic
        is_new = (action > 0.01 and self.position < 0.01)
        is_close = (action < 0.01 and self.position > 0.01)

        if is_new:
            cost = action * (1.0 + self.constraints.commission + self.constraints.slippage)
            self.portfolio_value *= (1.0 - cost)
            self.position = action
            self.entry_price = current_price
        elif is_close:
            rev = self.position * (1.0 - self.constraints.commission - self.constraints.stamp_tax - self.constraints.slippage)
            self.portfolio_value *= (1.0 + rev)
            self.position = 0.0
            self.entry_price = 0.0
        elif self.position > 0.01:
            delta = abs(action - self.position)
            if delta > 0.001:
                cost = delta * (self.constraints.commission + self.constraints.slippage)
                self.portfolio_value *= (1.0 - cost)
            self.position = action

        # Apply return to portfolio
        if self.position > 0.01:
            self.portfolio_value *= (1.0 + actual_return * self.position)

        self.current_idx += 1

        # Compute reward: Sharpe-like (position * return)
        position_return = actual_return * self.position
        reward = position_return

        done = self.current_idx >= len(seq) - 1

        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:
        """Get state vector."""
        seq = self.sequences[self.current_sym]
        t = self.current_idx

        recent = seq[max(0, t-20):t+1, :]

        if len(recent) >= 20:
            ma20 = np.mean(recent[-20:, 3])
            price_level = recent[-1, 3] / ma20 if ma20 > 0 else 1.0
        else:
            price_level = 1.0

        if len(recent) >= 2:
            ret_1d = np.log(recent[-1, 3] / recent[-2, 3]) if recent[-2, 3] > 0 else 0.0
        else:
            ret_1d = 0.0

        if len(recent) >= 5:
            ret_5d = np.log(recent[-1, 3] / recent[-5, 3]) if recent[-5, 3] > 0 else 0.0
        else:
            ret_5d = 0.0

        unrealized_pnl = 0.0
        if self.position > 0.01 and self.entry_price > 0:
            unrealized_pnl = (recent[-1, 3] - self.entry_price) / self.entry_price

        state = np.array([
            price_level - 1.0,
            ret_1d,
            ret_5d,
            self.position / 0.15,
            unrealized_pnl,
            0.0,
        ], dtype=np.float32)

        return state


def finetune_bc_with_rl(
    bc_checkpoint: str,
    num_epochs: int = 30,
    lr: float = 1e-5,
    gamma: float = 0.99,
    batch_size: int = 8,
    episode_length: int = 50,
):
    """
    Fine-tune BC pretrained model with RL.

    Uses policy gradient with advantage estimation.
    """
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("BC + RL Fine-tuning")
    print("Loading BC pretrained model...")
    print("=" * 60)

    # Load BC pretrained model
    bc_model = BCNetwork(input_dim=6, hidden_dim=128).to(device)
    bc_model.load_state_dict(torch.load(bc_checkpoint, map_location=device))
    print(f"Loaded BC model from {bc_checkpoint}")

    # Freeze early layers, only train action head
    for name, param in bc_model.named_parameters():
        if 'action_head' not in name:
            param.requires_grad = False

    # Count trainable params
    trainable = sum(p.numel() for p in bc_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}")

    optimizer = torch.optim.Adam(
        [p for p in bc_model.parameters() if p.requires_grad],
        lr=lr
    )

    # Load data
    all_symbols = list_symbols(base_dir=DATA_DIR)
    train_symbols = all_symbols[:100]
    val_symbols = all_symbols[100:130]

    print("\nLoading training data...")
    train_sequences, train_preds = prepare_data_with_predictions(
        train_symbols, "2020-01-01", "2024-01-01"
    )

    print("\nLoading validation data...")
    val_sequences, val_preds = prepare_data_with_predictions(
        val_symbols, "2024-01-01", "2025-12-31"
    )

    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")

    if len(train_sequences) < 10:
        print("Not enough data!")
        return None, None

    constraints = AShareConstraints()

    print(f"\nFine-tuning: {num_epochs} epochs, batch={batch_size}, episode={episode_length}")

    best_val_sharpe = -float('inf')
    save_path = Path(DATA_DIR).parent / "models" / "bc_rl_finetuned.pt"

    for epoch in tqdm(range(num_epochs)):
        # Sample batch
        sampled_idxs = random.sample(range(len(train_sequences)), min(batch_size, len(train_sequences)))
        epoch_rewards = []
        epoch_sharpes = []

        for sym_idx in sampled_idxs:
            env = TradingEnvironment(train_sequences, start_idx=60, constraints=constraints)
            state = env.reset(sym_idx=sym_idx)

            rewards = []
            actions = []
            states = []

            for step in range(episode_length):
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    action_prob = bc_model(state_t).item()

                # Add exploration noise
                action = action_prob + np.random.normal(0, 0.02)
                action = float(np.clip(action, 0.0, constraints.max_position_pct))

                next_state, reward, done = env.step(action)

                rewards.append(reward)
                actions.append(action)
                states.append(state)

                state = next_state
                if done:
                    break

            # Compute Sharpe ratio for this episode
            if len(rewards) > 5:
                rewards_arr = np.array(rewards)
                sharpe = np.mean(rewards_arr) / (np.std(rewards_arr) + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0.0

            epoch_rewards.extend(rewards)
            epoch_sharpes.append(sharpe)

            # Policy gradient update
            if len(rewards) > 5:
                # Compute discounted rewards
                discounted_rewards = []
                cumulative = 0.0
                for r in reversed(rewards):
                    cumulative = r + gamma * cumulative
                    discounted_rewards.insert(0, cumulative)

                # Normalize advantages
                adv_tensor = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
                adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

                # Compute policy gradient loss
                policy_loss = 0.0
                for step, (s, adv) in enumerate(zip(states, adv_tensor)):
                    s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                    action_pred = bc_model(s_t)  # Model predicts action

                    # Loss = -adv * log_prob(action)
                    # For sigmoid, use binary cross entropy style
                    target_action = actions[step] / constraints.max_position_pct  # Normalize to [0, 1]
                    target_tensor = torch.tensor([target_action], dtype=torch.float32).to(device)

                    # Use BCE-like loss
                    action_pred_clamped = action_pred.clamp(1e-6, 1-1e-6)
                    loss = -adv * torch.log(action_pred_clamped).squeeze()
                    policy_loss = policy_loss + loss

                optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(bc_model.parameters(), 1.0)
                optimizer.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            val_sharpes = []

            for sym_idx in range(min(10, len(val_sequences))):
                env = TradingEnvironment(val_sequences, start_idx=60, constraints=constraints)
                state = env.reset(sym_idx=sym_idx)

                rewards = []
                done = False
                step = 0

                while not done and step < 100:
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        action_prob = bc_model(state_t).item()

                    action = float(np.clip(action_prob, 0.0, constraints.max_position_pct))

                    next_state, reward, done = env.step(action)
                    rewards.append(reward)
                    state = next_state
                    step += 1

                if len(rewards) > 5:
                    rewards_arr = np.array(rewards)
                    sharpe = np.mean(rewards_arr) / (np.std(rewards_arr) + 1e-8) * np.sqrt(252)
                    val_sharpes.append(sharpe)

            avg_val_sharpe = np.mean(val_sharpes) if val_sharpes else 0.0
            avg_train_sharpe = np.mean(epoch_sharpes)

            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Sharpe: {avg_train_sharpe:.4f}, Val Sharpe: {avg_val_sharpe:.4f}")
            print(f"  Avg reward: {np.mean(epoch_rewards):.6f}")

            if avg_val_sharpe > best_val_sharpe:
                best_val_sharpe = avg_val_sharpe
                torch.save(bc_model.state_dict(), save_path)
                print(f"  -> Saved best (sharpe={avg_val_sharpe:.4f})")

    print(f"\nFine-tuning complete! Best val Sharpe: {best_val_sharpe:.4f}")
    return bc_model, save_path


def backtest_bc_rl(checkpoint_path: str = None):
    """Backtest the BC+RL fine-tuned model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not checkpoint_path or not Path(checkpoint_path).exists():
        print("No checkpoint found")
        return

    # Load model
    model = BCNetwork(input_dim=6, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # Load test data
    all_symbols = list_symbols(base_dir=DATA_DIR)
    test_symbols = all_symbols[200:250]

    test_sequences, test_preds = prepare_data_with_predictions(
        test_symbols, "2024-01-01", "2025-12-31"
    )

    print(f"\nTest sequences: {len(test_sequences)}")

    constraints = AShareConstraints()
    portfolio_values = []
    all_actions = []
    trade_count = 0

    for seq in test_sequences:
        env = TradingEnvironment([seq], start_idx=60, constraints=constraints)
        state = env.reset()

        done = False
        step = 0

        while not done and step < 200:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(state_t).item()

            # Trading threshold
            trade_threshold = 0.05
            if action > trade_threshold and env.position < 0.01:
                trade_count += 1

            next_state, _, done = env.step(action)
            state = next_state
            step += 1

        portfolio_values.append(env.portfolio_value)
        all_actions.append(env.position)

    returns = [v - 1.0 for v in portfolio_values]
    print("\n" + "=" * 50)
    print("BC+RL Fine-tuned Backtest Results")
    print("=" * 50)
    print(f"Total Return:     {np.mean(returns)*100:.2f}%")
    print(f"Std of Return:    {np.std(returns)*100:.2f}%")
    print(f"Sharpe Ratio:    {np.mean(returns)/(np.std(returns)+1e-8)*np.sqrt(252):.4f}")
    print(f"Trade Count:     {trade_count}")
    print(f"Avg Position:    {np.mean(all_actions):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["finetune", "backtest"], default="finetune")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "finetune":
        bc_ckpt = args.checkpoint or str(Path(DATA_DIR).parent / "models" / "bc_pretrain.pt")
        finetune_bc_with_rl(bc_ckpt)
    else:
        backtest_bc_rl(args.checkpoint)
