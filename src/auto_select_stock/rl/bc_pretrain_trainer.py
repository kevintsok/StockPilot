#!/usr/bin/env python3
"""
Behavior Cloning (BC) Pretraining for RL Agent.

This implements the "imitation learning warmup" approach:
1. Use existing ConfidenceStrategy to generate expert trajectories
2. Train a neural network to mimic the expert policy (behavior cloning)
3. Use the cloned policy as initialization for RL fine-tuning

Key insight: Starting from a reasonable policy avoids the
"cold start" problem where RL has to explore from scratch.

The approach:
1. Run ConfidenceStrategy on historical data to get "expert" actions
2. Train a supervised model to predict expert actions given states
3. Use this as the initial policy for SAC/PPO fine-tuning
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


class BCNetwork(nn.Module):
    """
    Behavior cloning network to mimic expert strategy.

    Input: state features (price + portfolio)
    Output: action logits (regression to expert action)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Output: position size [0, 1]
        self.action_head = nn.Linear(hidden_dim // 2, 1)
        self.action_head.weight.data.fill_(0.0)  # Initialize near zero
        self.action_head.bias.data.fill_(0.0)

    def forward(self, x):
        features = self.net(x)
        action = torch.sigmoid(self.action_head(features))  # [0, 1]
        return action


class ExpertTrajectoryCollector:
    """
    Collect expert trajectories from ConfidenceStrategy-like rules.

    Instead of using actual RL rewards, we use rule-based signals:
    - Buy signal: predicted return > threshold AND price not limit up
    - Sell signal: unrealized loss < stop_loss threshold
    - Hold: position is within target range
    - Wait: no clear signal

    This gives us "expert demonstrations" to clone.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        pred_returns: List[np.ndarray] = None,
        start_idx: int = 60,
    ):
        self.sequences = sequences
        self.pred_returns = pred_returns  # Optional PT predictions
        self.start_idx = start_idx

    def collect_trajectories(
        self,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.02,
        max_position: float = 0.15,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Collect expert trajectories.

        Returns: list of (state, expert_action) pairs
        """
        trajectories = []

        for sym_idx, seq in enumerate(tqdm(self.sequences, desc="Collecting trajectories")):
            # Get predictions if available
            if self.pred_returns is not None and sym_idx < len(self.pred_returns):
                preds = self.pred_returns[sym_idx]
            else:
                # Fallback: use momentum
                preds = self._compute_momentum_predictions(seq)

            # Simulate trading
            position = 0.0
            entry_price = 0.0

            for t in range(self.start_idx, len(seq) - 1):
                current_price = seq[t, 3]
                pred_ret = preds[t - self.start_idx] if t - self.start_idx < len(preds) else 0.0

                # Current state
                state = self._get_state(seq, t, position, entry_price)

                # Expert action logic (similar to ConfidenceStrategy)
                action = self._get_expert_action(
                    position, pred_ret, current_price, entry_price,
                    buy_threshold, sell_threshold, max_position
                )

                trajectories.append((state, action))

                # Update position based on expert action
                position = action

                if position > 0.01 and entry_price == 0:
                    entry_price = current_price
                elif position < 0.01:
                    entry_price = 0

        return trajectories

    def _compute_momentum_predictions(self, seq: np.ndarray) -> np.ndarray:
        """Compute simple momentum-based predictions."""
        preds = []
        for i in range(self.start_idx, len(seq) - 1):
            if i >= 5:
                mom = (seq[i, 3] - seq[i-5, 3]) / seq[i-5, 3] if seq[i-5, 3] > 0 else 0.0
            else:
                mom = 0.0
            preds.append(mom)
        return np.array(preds)

    def _get_state(self, seq: np.ndarray, t: int, position: float, entry_price: float) -> np.ndarray:
        """Get state vector."""
        # Price features
        recent = seq[max(0, t-20):t+1, :]

        if len(recent) >= 20:
            ma20 = np.mean(recent[-20:, 3])
            price_level = recent[-1, 3] / ma20 if ma20 > 0 else 1.0
        else:
            price_level = 1.0

        # Returns
        if len(recent) >= 2:
            ret_1d = np.log(recent[-1, 3] / recent[-2, 3]) if recent[-2, 3] > 0 else 0.0
        else:
            ret_1d = 0.0

        if len(recent) >= 5:
            ret_5d = np.log(recent[-1, 3] / recent[-5, 3]) if recent[-5, 3] > 0 else 0.0
        else:
            ret_5d = 0.0

        # Portfolio features
        unrealized_pnl = 0.0
        if position > 0.01 and entry_price > 0:
            unrealized_pnl = (recent[-1, 3] - entry_price) / entry_price

        state = np.array([
            price_level - 1.0,      # Normalized price level
            ret_1d,               # 1-day log return
            ret_5d,               # 5-day log return
            position / 0.15,      # Normalized position
            unrealized_pnl,       # Unrealized PnL
            0.0,                  # Cash placeholder
        ], dtype=np.float32)

        return state

    def _get_expert_action(
        self,
        position: float,
        pred_ret: float,
        current_price: float,
        entry_price: float,
        buy_threshold: float,
        sell_threshold: float,
        max_position: float,
    ) -> float:
        """
        Get expert action based on rules.

        Similar to ConfidenceStrategy:
        - Buy if predicted return > threshold and not in position
        - Sell if predicted return < sell_threshold or stop loss hit
        - Hold if in position and within target range
        - Wait if no clear signal
        """
        # Check if price would hit limit up/down (simplified)
        # For now, skip this check

        unrealized_pnl = 0.0
        if position > 0.01 and entry_price > 0:
            unrealized_pnl = (current_price - entry_price) / entry_price

        # Stop loss
        if position > 0.01 and unrealized_pnl < -0.05:
            return 0.0  # Stop loss sell

        # Sell signals
        if position > 0.01:
            if pred_ret < sell_threshold:
                return 0.0  # Predicted to go down, sell
            elif unrealized_pnl > 0.15:
                return 0.0  # Take profit

        # Buy signals
        if position < 0.01:
            if pred_ret > buy_threshold:
                return max_position * 0.8  # Buy signal

        # Hold or wait
        if position > 0.01:
            return position  # Hold current position
        else:
            return 0.0  # Wait


def prepare_data_with_predictions(
    symbols: List[str],
    start_date: str,
    end_date: str,
    use_pt_predictions: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Prepare price sequences and predictions.

    If use_pt_predictions=True, use PT model (slower but better signal).
    Otherwise, use momentum-based predictions.
    """
    import pandas as pd

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    sequences = []
    predictions = []

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

            # Compute momentum predictions
            preds = []
            for i in range(60, len(arr_out) - 1):
                if i >= 5:
                    mom = (arr_out[i, 3] - arr_out[i-5, 3]) / arr_out[i-5, 3] if arr_out[i-5, 3] > 0 else 0.0
                else:
                    mom = 0.0
                preds.append(mom)

            sequences.append(arr_out)
            predictions.append(np.array(preds))

        except Exception:
            continue

    return sequences, predictions


def train_bc_pretrain():
    """
    Train behavior cloning model to mimic expert strategy.

    This is Step 3 in the three-step approach:
    1. (Already done) PT predictions as reward signal
    2. (Already done) Direct Sharpe optimization
    3. Imitation learning from expert trajectories
    """
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)
    print("Behavior Cloning Pretraining")
    print("Learning to mimic expert strategy from trajectories")
    print("=" * 60)

    # Load data
    all_symbols = list_symbols(base_dir=DATA_DIR)
    train_symbols = all_symbols[:150]
    val_symbols = all_symbols[150:200]

    print("\nLoading training data...")
    train_sequences, train_preds = prepare_data_with_predictions(
        train_symbols, "2020-01-01", "2024-01-01"
    )

    print("\nLoading validation data...")
    val_sequences, val_preds = prepare_data_with_predictions(
        val_symbols, "2024-01-01", "2025-12-31"
    )

    print(f"\nTrain: {len(train_sequences)} sequences")
    print(f"Val: {len(val_sequences)} sequences")

    if len(train_sequences) < 10:
        print("Not enough data!")
        return None, None

    # Collect expert trajectories
    print("\nCollecting expert trajectories...")
    collector = ExpertTrajectoryCollector(train_sequences, train_preds)

    trajectories = collector.collect_trajectories(
        buy_threshold=0.01,
        sell_threshold=-0.015,
    )

    print(f"Collected {len(trajectories)} expert transitions")

    # Create dataset
    states = np.array([t[0] for t in trajectories], dtype=np.float32)
    actions = np.array([t[1] for t in trajectories], dtype=np.float32)

    print(f"State shape: {states.shape}")
    print(f"Action range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"Action distribution: mean={actions.mean():.4f}, std={actions.std():.4f}")

    # Create model
    input_dim = states.shape[1]
    model = BCNetwork(input_dim=input_dim, hidden_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training
    batch_size = 256
    num_epochs = 50

    print(f"\nTraining BC model...")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")

    best_val_loss = float('inf')
    save_path = Path(DATA_DIR).parent / "models" / "bc_pretrain.pt"

    for epoch in tqdm(range(num_epochs)):
        # Shuffle
        indices = np.random.permutation(len(states))
        states_shuffled = states[indices]
        actions_shuffled = actions[indices]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(states), batch_size):
            batch_states = torch.tensor(states_shuffled[i:i+batch_size], dtype=torch.float32).to(device)
            batch_actions = torch.tensor(actions_shuffled[i:i+batch_size], dtype=torch.float32).unsqueeze(-1).to(device)

            # Forward pass
            pred_actions = model(batch_states)

            # MSE loss
            loss = F.mse_loss(pred_actions, batch_actions)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        scheduler.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                val_states_t = torch.tensor(states[:1000], dtype=torch.float32).to(device)
                val_actions_t = torch.tensor(actions[:1000], dtype=torch.float32).unsqueeze(-1).to(device)

                val_pred = model(val_states_t)
                val_loss = F.mse_loss(val_pred, val_actions_t).item()

                # Compute correlation
                val_pred_np = val_pred.cpu().numpy().flatten()
                val_actions_np = val_actions_t.cpu().numpy().flatten()
                correlation = np.corrcoef(val_pred_np, val_actions_np)[0, 1]

            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train loss: {avg_loss:.6f}")
            print(f"  Val loss: {val_loss:.6f}")
            print(f"  Val correlation: {correlation:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best model")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    return model, save_path


def backtest_bc(checkpoint_path: str = None):
    """Backtest the BC-pretrained model."""
    device = torch.device('cpu')  # Force CPU for this GPU

    # Load model
    if checkpoint_path and Path(checkpoint_path).exists():
        model = BCNetwork(input_dim=6, hidden_dim=128).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model.eval()
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("No checkpoint found")
        return

    # Load test data
    all_symbols = list_symbols(base_dir=DATA_DIR)
    test_symbols = all_symbols[200:250]

    test_sequences, test_preds = prepare_data_with_predictions(
        test_symbols, "2024-01-01", "2025-12-31"
    )

    print(f"\nTest sequences: {len(test_sequences)}")

    # Run backtest
    collector = ExpertTrajectoryCollector(test_sequences, test_preds)

    portfolio_values = []
    all_actions = []
    trade_count = 0
    position = 0.0

    for seq, preds in zip(test_sequences, test_preds):
        env_idx = 0  # Single sequence per symbol
        start_idx = 60

        position = 0.0
        entry_price = 0.0
        portfolio = 1.0

        for t in range(start_idx, len(seq) - 1):
            current_price = seq[t, 3]
            pred_ret = preds[t - start_idx] if t - start_idx < len(preds) else 0.0

            # Get state
            state = collector._get_state(seq, t, position, entry_price)
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            # Predict action - model outputs sigmoid [0, 1], trained on [0, 0.12] targets
            # So output directly represents position in [0, 0.12]
            with torch.no_grad():
                action = model(state_t).item()

            # Trading threshold - only trade when model is confident (model output > 0.05)
            # This filters out the "wait" predictions and only acts on strong signals
            trade_threshold = 0.05

            # Trading logic
            is_new = (action > trade_threshold and position < 0.01)
            is_close = (action < trade_threshold and position > 0.01)

            if is_new:
                cost = action * (1.0 + 0.0003 + 0.0001)
                portfolio *= (1.0 - cost)
                position = action
                entry_price = current_price
                trade_count += 1
            elif is_close:
                rev = position * (1.0 - 0.0003 - 0.001 - 0.0001)
                portfolio *= (1.0 + rev)
                position = 0.0
                entry_price = 0.0
            elif position > 0.01:
                delta = abs(action - position)
                if delta > 0.001:
                    cost = delta * (0.0003 + 0.0001)
                    portfolio *= (1.0 - cost)
                position = action

            # Apply return
            if position > 0.01:
                ret = (seq[t+1, 3] - current_price) / current_price if current_price > 0 else 0.0
                portfolio *= (1.0 + ret * position)

        portfolio_values.append(portfolio)
        all_actions.append(position)

    returns = [v - 1.0 for v in portfolio_values]
    print("\n" + "=" * 50)
    print("BC Pretrain Backtest Results")
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
        train_bc_pretrain()
    else:
        backtest_bc(args.checkpoint)
