#!/usr/bin/env python3
"""
RRL (Recurrent Reinforcement Learning) with Direct Sharpe Optimization.

Key insight from RRL literature (Moody et al.):
- Instead of maximizing cumulative returns (which is noisy),
- Directly maximize the differential Sharpe ratio
- This is more robust to noise in stock returns

The differential Sharpe ratio:
- Uses exponential moving average of returns
- Computes Sharpe ratio over rolling window
- Gradient-based optimization of network weights

Architecture:
- Simple RNN/LSTM network
- Input: price features + portfolio state
- Output: position size
- Loss: negative Sharpe ratio
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..config import DATA_DIR
from ..data.storage import list_symbols, load_stock_history
from ..predict.data import compute_technical_indicators, PRICE_FEATURE_COLUMNS


@dataclass
class AShareConstraints:
    """A-share trading constraints."""
    commission: float = 0.0003
    stamp_tax: float = 0.001
    slippage: float = 0.0001
    max_position_pct: float = 0.15
    cash_reserve: float = 0.05


class RRLNetwork(nn.Module):
    """
    Simple RNN network for position sizing.

    Input: [price_features; portfolio_state; prev_action]
    Output: position size (0 to max_position)

    The network learns to predict optimal position sizing
    that maximizes Sharpe ratio.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, input_dim)
        Returns: (batch, 1) position size, hidden state
        """
        if hidden is None:
            rnn_out, hidden = self.rnn(x)
        else:
            rnn_out, hidden = self.rnn(x, hidden)

        # Use last output
        last_out = rnn_out[:, -1, :]
        position = self.fc(last_out)
        return position, hidden


class RRLTradingEnvironment:
    """
    Trading environment that provides differential Sharpe reward.

    Key insight: Use differential Sharpe ratio as reward signal.
    This is more robust than raw returns.
    """

    def __init__(
        self,
        price_sequences: List[np.ndarray],
        start_idx: int = 60,
        constraints: AShareConstraints = None,
    ):
        self.price_sequences = price_sequences
        self.start_idx = start_idx
        self.constraints = constraints or AShareConstraints()

        self.current_sym = 0
        self.current_idx = start_idx
        self.position = 0.0
        self.portfolio_value = 1.0
        self.prev_action = 0.0

        # Sharpe tracking
        self.returns_history = []
        self.window_size = 20  # Rolling window for Sharpe

    def reset(self, sym_idx: int = None) -> Tuple[np.ndarray, np.ndarray, int]:
        if sym_idx is not None:
            self.current_sym = sym_idx

        self.current_idx = self.start_idx
        self.position = 0.0
        self.portfolio_value = 1.0
        self.prev_action = 0.0
        self.returns_history = []

        return self._get_sequence(), self._get_portfolio_state(), 0

    def step(self, action: float) -> Tuple[np.ndarray, np.ndarray, float, bool, float]:
        """
        Execute action and return feedback.

        Returns: (seq, port_state, reward, done, portfolio_return)
        """
        action = float(np.clip(action, 0.0, self.constraints.max_position_pct))

        price_seq = self.price_sequences[self.current_sym]
        current_price = price_seq[self.current_idx, 3]

        # Compute return
        if self.current_idx > self.start_idx:
            prev_price = price_seq[self.current_idx - 1, 3]
            actual_return = (current_price - prev_price) / prev_price if prev_price > 0 else 0.0
        else:
            actual_return = 0.0

        # Trading logic
        is_new = (action > 0.01 and self.position < 0.01)
        is_close = (action < 0.01 and self.position > 0.01)

        if is_new:
            cost = action * (1.0 + self.constraints.commission + self.constraints.slippage)
            self.portfolio_value *= (1.0 - cost)
            self.position = action
        elif is_close:
            rev = self.position * (1.0 - self.constraints.commission - self.constraints.stamp_tax - self.constraints.slippage)
            self.portfolio_value *= (1.0 + rev)
            self.position = 0.0
        elif action > 0.01 and self.position > 0.01:
            delta = abs(action - self.position)
            if delta > 0.001:
                cost = delta * (self.constraints.commission + self.constraints.slippage)
                self.portfolio_value *= (1.0 - cost)
            self.position = action

        self.prev_action = action
        self.current_idx += 1

        # Compute differential Sharpe reward
        # Portion of return attributable to position
        position_return = actual_return * self.position

        # Update returns history
        self.returns_history.append(position_return)
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # Compute differential Sharpe reward
        reward = self._compute_differential_sharpe()

        done = self.current_idx >= len(price_seq) - 1

        return self._get_sequence(), self._get_portfolio_state(), reward, done, actual_return

    def _compute_differential_sharpe(self) -> float:
        """
        Compute differential Sharpe ratio reward.

        Uses exponential moving average of returns and variances
        to compute a running Sharpe ratio.
        """
        if len(self.returns_history) < 5:
            return 0.0

        returns = np.array(self.returns_history)

        # Exponential moving average
        ema_alpha = 0.04  # Smoothing parameter
        if not hasattr(self, 'ema_return'):
            self.ema_return = returns.mean()
            self.ema_var = returns.var()

        self.ema_return = ema_alpha * returns.mean() + (1 - ema_alpha) * self.ema_return
        self.ema_var = ema_alpha * returns.var() + (1 - ema_alpha) * self.ema_var

        std = np.sqrt(self.ema_var + 1e-8)
        sharpe = self.ema_return / std if std > 1e-8 else 0.0

        # Scale to reasonable reward range
        return sharpe * 0.1

    def _get_sequence(self) -> np.ndarray:
        """Get price sequence for RNN input."""
        price_seq = self.price_sequences[self.current_sym]
        start = max(0, self.current_idx - 60)
        window = price_seq[start:self.current_idx + 1]

        if len(window) < 61:
            pad = np.zeros((61 - len(window), window.shape[1]), dtype=np.float32)
            window = np.vstack([pad, window])

        return window

    def _get_portfolio_state(self) -> np.ndarray:
        """Get portfolio state vector."""
        return np.array([
            self.position / self.constraints.max_position_pct,
            self.prev_action / self.constraints.max_position_pct,
            self.portfolio_value - 1.0,
            (1.0 - self.position) / (1.0 - self.constraints.cash_reserve),
        ], dtype=np.float32)


def prepare_data(symbols: List[str], start_date: str, end_date: str) -> List[np.ndarray]:
    """Load and prepare price data."""
    import pandas as pd

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    sequences = []

    for sym in tqdm(symbols, desc="Loading data"):
        try:
            arr = load_stock_history(sym, base_dir=DATA_DIR)
            if arr is None or len(arr) < 120:
                continue

            df = pd.DataFrame(arr)
            df["date"] = pd.to_datetime(df["date"]).dt.floor("D").astype("datetime64[ns]")
            df.sort_values("date", inplace=True)
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

            if len(df) < 120:
                continue

            price_df = df[PRICE_FEATURE_COLUMNS].copy()
            tech_df = compute_technical_indicators(df)
            combined = pd.concat([price_df, tech_df], axis=1)

            cols = list(PRICE_FEATURE_COLUMNS)
            for c in ["rsi_14", "macd_line", "macd_signal", "macd_hist",
                      "bb_position", "bb_width", "volume_ma5", "volume_ma20",
                      "atr_14", "stoch_k", "stoch_d", "obv_ma10", "roc_10", "momentum_10"]:
                if c in combined.columns:
                    cols.append(c)

            arr_out = combined[cols].values.astype(np.float32)
            arr_out = np.nan_to_num(arr_out, nan=0.0, posinf=0.0, neginf=0.0)

            for ci in range(arr_out.shape[1]):
                arr_out[:, ci] = np.clip(arr_out[:, ci], -1e6, 1e6)

            sequences.append(arr_out)
        except Exception:
            continue

    return sequences


def train_rrl_sharpe():
    """
    Train RRL network with direct Sharpe optimization.

    Key difference from standard RL:
    - Uses differential Sharpe ratio as reward
    - Simpler network (GRU-based)
    - No replay buffer (on-policy)
    """
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("RRL with Direct Sharpe Optimization")
    print("Using differential Sharpe as reward signal")
    print("=" * 60)

    # Load data
    all_symbols = list_symbols(base_dir=DATA_DIR)
    train_symbols = all_symbols[:100]
    val_symbols = all_symbols[100:130]

    train_sequences = prepare_data(train_symbols, "2020-01-01", "2024-01-01")
    val_sequences = prepare_data(val_symbols, "2024-01-01", "2025-12-31")

    print(f"\nTrain sequences: {len(train_sequences)}, Val sequences: {len(val_sequences)}")

    if len(train_sequences) < 10:
        print("Not enough data!")
        return None, None

    # Create RRL network
    price_dim = train_sequences[0].shape[1]
    input_dim = price_dim + 4  # price features + portfolio state

    model = RRLNetwork(input_dim=input_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    constraints = AShareConstraints()

    # Training
    num_epochs = 50
    batch_size = 8
    episode_length = 50  # Steps per episode

    best_val_sharpe = -float('inf')
    save_path = Path(DATA_DIR).parent / "models" / "rrl_sharpe.pt"

    print(f"\nTraining RRL-Sharpe...")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")

    for epoch in tqdm(range(num_epochs)):
        # Sample batch of symbols
        sampled_idxs = random.sample(range(len(train_sequences)), min(batch_size, len(train_sequences)))

        epoch_sharpes = []

        for sym_idx in sampled_idxs:
            env = RRLTradingEnvironment(train_sequences, start_idx=60, constraints=constraints)
            seq, port_state, hidden = env.reset(sym_idx=sym_idx)

            # Convert to tensors
            seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            port_t = torch.tensor(port_state, dtype=torch.float32).unsqueeze(0).to(device)

            # Combine features
            # Repeat port_state for each step in sequence
            port_tiled = port_t.repeat(1, seq_t.size(1), 1)
            combined = torch.cat([seq_t, port_tiled], dim=-1)

            hidden = None
            rewards = []
            actions = []

            for step in range(episode_length):
                # Forward pass
                position, hidden = model(combined[:, :step+1, :], hidden)

                # Convert position to action
                action = position.item() * constraints.max_position_pct
                action = float(np.clip(action + np.random.normal(0, 0.02), 0.0, constraints.max_position_pct))

                # Environment step
                _, _, reward, done, _ = env.step(action)

                rewards.append(reward)
                actions.append(action)

                if done:
                    break

            # Compute Sharpe ratio for tracking
            if len(rewards) > 5:
                rewards_arr = np.array(rewards)
                sharpe = np.mean(rewards_arr) / (np.std(rewards_arr) + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0.0

            epoch_sharpes.append(sharpe)

            # Policy gradient update using REINFORCE
            # Advantage = cumulative discounted reward - baseline
            if len(rewards) > 5:
                # Use actual rewards to compute advantage
                discounted_rewards = []
                cumulative = 0.0
                gamma = 0.99
                for r in reversed(rewards):
                    cumulative = r + gamma * cumulative
                    discounted_rewards.insert(0, cumulative)

                # Normalize advantages
                adv_tensor = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
                adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

                # Compute policy gradient loss
                # We re-run the forward pass to get log probabilities
                # Since our policy is deterministic, we use the position as the "score"
                # Gradient: -advantage * grad(log pi(a|s))
                hidden = None
                policy_loss = 0.0
                for step in range(len(rewards)):
                    position, hidden = model(combined[:, :step+1, :], hidden)
                    # Use position as action proxy, gradient flows through position
                    # Loss = -sum(adv * position)
                    policy_loss = policy_loss - (adv_tensor[step] * position.squeeze())

                optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            val_sharpes = []

            for sym_idx in range(min(10, len(val_sequences))):
                env = RRLTradingEnvironment(val_sequences, start_idx=60, constraints=constraints)
                seq, port_state, hidden = env.reset(sym_idx=sym_idx)

                seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                port_t = torch.tensor(port_state, dtype=torch.float32).unsqueeze(0).to(device)
                port_tiled = port_t.repeat(1, seq_t.size(1), 1)
                combined = torch.cat([seq_t, port_tiled], dim=-1)

                hidden = None
                rewards = []

                done = False
                step = 0
                while not done and step < 100:
                    position, hidden = model(combined[:, :step+1, :], hidden)
                    action = position.item() * constraints.max_position_pct
                    action = float(np.clip(action, 0.0, constraints.max_position_pct))

                    _, _, reward, done, _ = env.step(action)
                    rewards.append(reward)
                    step += 1

                if len(rewards) > 5:
                    rewards_arr = np.array(rewards)
                    sharpe = np.mean(rewards_arr) / (np.std(rewards_arr) + 1e-8) * np.sqrt(252)
                    val_sharpes.append(sharpe)

            avg_val_sharpe = np.mean(val_sharpes) if val_sharpes else 0.0
            avg_train_sharpe = np.mean(epoch_sharpes)

            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Sharpe: {avg_train_sharpe:.4f}, Val Sharpe: {avg_val_sharpe:.4f}")
            print(f"  Avg action: {np.mean(actions):.4f}, Action std: {np.std(actions):.4f}")

            if avg_val_sharpe > best_val_sharpe:
                best_val_sharpe = avg_val_sharpe
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best (sharpe={avg_val_sharpe:.4f})")

    print(f"\nTraining complete! Best val Sharpe: {best_val_sharpe:.4f}")
    return model, save_path


def backtest_rrl(checkpoint_path: str = None):
    """Backtest RRL agent."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if checkpoint_path and Path(checkpoint_path).exists():
        model = RRLNetwork(input_dim=29, hidden_dim=64).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("No checkpoint found")
        return

    # Load test data
    all_symbols = list_symbols(base_dir=DATA_DIR)
    test_symbols = all_symbols[200:240]

    test_sequences = prepare_data(test_symbols, "2024-01-01", "2025-12-31")
    print(f"\nTest sequences: {len(test_sequences)}")

    constraints = AShareConstraints()
    portfolio_values = []
    all_actions = []
    trade_count = 0

    for sym_idx, seq in enumerate(test_sequences):
        env = RRLTradingEnvironment([seq], start_idx=60, constraints=constraints)
        state_seq, port_state, hidden = env.reset()

        seq_t = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
        port_t = torch.tensor(port_state, dtype=torch.float32).unsqueeze(0).to(device)
        port_tiled = port_t.repeat(1, seq_t.size(1), 1)
        combined = torch.cat([seq_t, port_tiled], dim=-1)

        hidden = None
        done = False
        step = 0

        while not done and step < 200:
            position, hidden = model(combined[:, :step+1, :], hidden)
            action = position.item() * constraints.max_position_pct
            action = float(np.clip(action, 0.0, constraints.max_position_pct))

            if action > 0.01 and env.position < 0.01:
                trade_count += 1

            _, _, _, done, _ = env.step(action)
            step += 1

        portfolio_values.append(env.portfolio_value)
        all_actions.append(env.prev_action)

    returns = [v - 1.0 for v in portfolio_values]
    print("\n" + "=" * 50)
    print("RRL Backtest Results")
    print("=" * 50)
    print(f"Total Return:     {np.mean(returns)*100:.2f}%")
    print(f"Std of Return:    {np.std(returns)*100:.2f}%")
    print(f"Sharpe Ratio:    {np.mean(returns)/(np.std(returns)+1e-8)*np.sqrt(252):.4f}")
    print(f"Trade Count:     {trade_count}")
    print(f"Avg Position:    {np.mean(all_actions):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "backtest"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        train_rrl_sharpe()
    else:
        backtest_rrl(args.checkpoint)
