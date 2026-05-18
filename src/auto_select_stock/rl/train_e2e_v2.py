#!/usr/bin/env python3
"""
E2E Transformer v2: Fixed Labeling

Key fixes vs train_e2e.py:
1. Use momentum-based predictions instead of future returns (NO look-ahead bias)
2. Lower thresholds: buy=1%, sell=-1.5% (like BC, not 2%/-2%)
3. Class-weighted CrossEntropyLoss for imbalance
4. Optional BC pretrain initialization

Usage:
    python -m auto_select_stock.rl.train_e2e_v2 --device cuda --epochs 20
"""

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..config import DATA_DIR
from ..data.storage import list_symbols
from .e2e_transformer import E2ETransformer, NUM_ACTIONS


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# BC-style action labeling (THE KEY FIX)
# ============================================================================

def _compute_action_label(
    pred_ret: float,
    unrealized_pnl: float,
    has_position: bool,
    stop_loss: float = 0.05,
    take_profit: float = 0.15,
    buy_threshold: float = 0.01,
    sell_threshold: float = -0.015,
) -> int:
    """
    BC-style action labeling - NO look-ahead bias.

    Uses model predictions (momentum), NOT future returns.

    Returns:
        0=buy, 1=sell, 2=hold, 3=wait
    """
    if has_position:
        if unrealized_pnl < -stop_loss:
            return 1  # sell - stop loss
        if unrealized_pnl > take_profit:
            return 1  # sell - take profit
        if pred_ret < sell_threshold:
            return 1  # sell - negative signal
        return 2  # hold
    else:
        if pred_ret > buy_threshold:
            return 0  # buy
        return 3  # wait


def _compute_momentum_prediction(arr: np.ndarray, t: int, lookback: int = 5) -> float:
    """Compute momentum-based prediction (like BC's ExpertTrajectoryCollector)."""
    if t >= lookback:
        return (arr[t, 3] - arr[t - lookback, 3]) / arr[t - lookback, 3]
    return 0.0


# ============================================================================
# Dataset with FIXED labeling
# ============================================================================

class TradingDatasetV2(Dataset):
    """
    Dataset for E2E v2 with BC-style labeling.

    Key difference vs V1:
    - Uses momentum predictions (not future returns)
    - Lower thresholds for more balanced classes
    - Tracks position state for hold/sell decisions
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        seq_len: int = 60,
        close_index: int = 3,
        horizon: int = 5,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.015,
        stop_loss: float = 0.05,
        take_profit: float = 0.15,
        lookback: int = 5,
    ):
        self.seq_len = seq_len
        self.close_index = close_index
        self.horizon = horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.lookback = lookback

        self.samples: List[Tuple[np.ndarray, int, float]] = []
        self._build_samples(sequences)

        # Compute class distribution for weighting
        self._compute_class_weights()

    def _build_samples(self, sequences: List[np.ndarray]):
        for arr in sequences:
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) <= self.seq_len + self.horizon + 1:
                continue

            has_position = False
            entry_price = 0.0

            for i in range(self.seq_len, len(arr) - self.horizon):
                window = arr[i - self.seq_len:i + 1]
                price_seq = window[:-1]

                # Momentum-based prediction (like BC) - NO look-ahead bias
                pred_ret = _compute_momentum_prediction(arr, i, self.lookback)

                # Current state
                current_price = arr[i, self.close_index]

                # Unrealized PnL
                unrealized_pnl = 0.0
                if has_position and entry_price > 0:
                    unrealized_pnl = (current_price - entry_price) / entry_price

                # Compute action using BC-style labeling
                action = _compute_action_label(
                    pred_ret=pred_ret,
                    unrealized_pnl=unrealized_pnl,
                    has_position=has_position,
                    stop_loss=self.stop_loss,
                    take_profit=self.take_profit,
                    buy_threshold=self.buy_threshold,
                    sell_threshold=self.sell_threshold,
                )

                # Position size (continuous, like BC)
                if action == 0:  # buy
                    position_size = 0.12  # ~max_position * 0.8
                    has_position = True
                    entry_price = current_price
                elif action == 1:  # sell
                    position_size = 0.0
                    has_position = False
                    entry_price = 0.0
                elif action == 2:  # hold
                    position_size = 0.12
                else:  # wait
                    position_size = 0.0

                self.samples.append((price_seq, action, position_size))

    def _compute_class_weights(self):
        """Compute class weights for imbalanced dataset."""
        counts = [0] * NUM_ACTIONS
        for _, action, _ in self.samples:
            counts[action] += 1
        total = len(self.samples)
        # Inverse frequency weighting
        self.class_weights = [total / (NUM_ACTIONS * c) if c > 0 else 0.0 for c in counts]
        print(f"  Class distribution: buy={counts[0]} ({counts[0]/total*100:.1f}%), "
              f"sell={counts[1]} ({counts[1]/total*100:.1f}%), "
              f"hold={counts[2]} ({counts[2]/total*100:.1f}%), "
              f"wait={counts[3]} ({counts[3]/total*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        price_seq, action, position_size = self.samples[idx]
        return (
            torch.tensor(price_seq, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(position_size, dtype=torch.float32),
        )


def prepare_trading_data(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    base_dir: Path = DATA_DIR,
    seq_len: int = 60,
) -> Tuple[List[np.ndarray], int, int]:
    """Load price sequences for trading dataset."""
    from ..data.storage import load_stock_history
    from ..predict.data import compute_technical_indicators, PRICE_FEATURE_COLUMNS

    close_index = PRICE_FEATURE_COLUMNS.index("close")
    sequences = []

    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None

    for sym in tqdm(symbols, desc="Loading data"):
        try:
            arr = load_stock_history(sym, base_dir=base_dir)
            if arr is None or len(arr) < seq_len + 50:
                continue

            df = pd.DataFrame(arr)
            df["date"] = pd.to_datetime(df["date"]).dt.floor("D").astype("datetime64[ns]")
            df.sort_values("date", inplace=True)

            if start_dt is not None:
                df = df[df["date"] >= start_dt]
            if end_dt is not None:
                df = df[df["date"] <= end_dt]

            if len(df) < seq_len + 50:
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

    feature_dim = sequences[0].shape[1] if sequences else len(PRICE_FEATURE_COLUMNS)
    return sequences, close_index, feature_dim


# ============================================================================
# Training
# ============================================================================

def train_e2e_v2(
    train_sequences: List[np.ndarray],
    val_sequences: List[np.ndarray],
    close_index: int,
    feature_dim: int,
    device: torch.device,
    seq_len: int = 60,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    max_position_pct: float = 0.15,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    save_path: Optional[Path] = None,
    bc_pretrain_path: Optional[str] = None,
) -> E2ETransformer:
    """Train E2E v2 with BC-style labeling."""
    _set_seed(42)

    print(f"\n{'='*60}")
    print("E2E v2 Training (BC-style labeling)")
    print(f"seq_len={seq_len}, d_model={d_model}, layers={num_layers}")
    print(f"{'='*60}")

    # Create datasets
    train_ds = TradingDatasetV2(train_sequences, seq_len=seq_len, close_index=close_index)
    val_ds = TradingDatasetV2(val_sequences, seq_len=seq_len, close_index=close_index)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    model = E2ETransformer(
        price_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_position_pct=max_position_pct,
        use_financial=False,
        use_portfolio=False,
    ).to(device)

    # Optional: Load BC pretrain weights for initialization
    if bc_pretrain_path:
        print(f"Loading BC pretrain weights from {bc_pretrain_path}...")
        try:
            bc_state = torch.load(bc_pretrain_path, map_location=device)
            # Copy FC layers to Transformer projection + LN
            model.price_proj.load_state_dict({k.replace('net.0.', 'net.0.'): v
                for k, v in bc_state.items() if 'net.0.' in k})
            model.price_ln.load_state_dict({k.replace('net.1.', 'price_ln.'): v
                for k, v in bc_state.items() if 'net.1.' in k})
            print("  BC weights loaded successfully")
        except Exception as e:
            print(f"  Warning: Could not load BC weights: {e}")

    # Zero out regression heads
    for h in model.reg_heads:
        h.weight.data.fill_(0.0)
        h.bias.data.fill_(0.0)

    # Class-weighted CrossEntropyLoss (KEY FIX for imbalance)
    cls_weights = torch.tensor(train_ds.class_weights, dtype=torch.float32, device=device)

    def _compute_loss(action_logits, actions, position_size, pos_sizes):
        # Weighted classification loss
        cls_loss = F.cross_entropy(action_logits, actions, weight=cls_weights)
        # Position size loss (only for non-wait)
        mask = (actions != 3).float()
        if mask.sum() > 0:
            pos_loss = (F.mse_loss(position_size.squeeze(-1), pos_sizes, reduction='none') * mask).sum() / (mask.sum() + 1e-8)
        else:
            pos_loss = torch.tensor(0.0, device=device)
        return cls_loss + 0.1 * pos_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_cls = 0.0

        for price_seq, actions, pos_sizes in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            price_seq = price_seq.to(device)
            actions = actions.to(device)
            pos_sizes = pos_sizes.to(device)

            optimizer.zero_grad()
            action_logits, position_size, _ = model(price_seq)
            loss = _compute_loss(action_logits, actions, position_size, pos_sizes)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_cls += F.cross_entropy(action_logits, actions).item()

        train_loss /= len(train_loader)
        train_cls /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        action_counts = [0] * NUM_ACTIONS

        with torch.no_grad():
            for price_seq, actions, pos_sizes in val_loader:
                price_seq = price_seq.to(device)
                actions = actions.to(device)

                action_logits, _, _ = model(price_seq)

                if torch.isnan(action_logits).any():
                    continue

                cls_loss = F.cross_entropy(action_logits, actions, weight=cls_weights)
                val_loss += cls_loss.item()

                preds = action_logits.argmax(dim=-1)
                val_correct += (preds == actions).sum().item()
                val_total += actions.size(0)

                for a in preds.cpu().numpy():
                    action_counts[a] += 1

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        scheduler.step()

        pred_dist = f"pred_dist: buy={action_counts[0]}, sell={action_counts[1]}, hold={action_counts[2]}, wait={action_counts[3]}"
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_acc={val_acc:.4f} | {pred_dist}")

        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train E2E Transformer v2 (BC-style labeling)")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date")
    parser.add_argument("--symbols", type=int, default=200, help="Number of symbols")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--save-path", type=str, default="models/e2e_v2.pt", help="Save path")
    parser.add_argument("--bc-pretrain", type=str, default=None, help="BC pretrain checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    device = _device()
    print(f"Device: {device}")

    # Get symbols
    symbols = list_symbols(base_dir=DATA_DIR)[:args.symbols]

    # Prepare data
    all_sequences, close_index, feature_dim = prepare_trading_data(
        symbols, args.start, args.end, seq_len=args.seq_len
    )

    print(f"Loaded {len(all_sequences)} sequences, feature_dim={feature_dim}")

    # Split train/val
    n_train = int(len(all_sequences) * 0.8)
    train_sequences = all_sequences[:n_train]
    val_sequences = all_sequences[n_train:]

    print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}")

    # Train
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = train_e2e_v2(
        train_sequences,
        val_sequences,
        close_index,
        feature_dim,
        device,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=save_path,
        bc_pretrain_path=args.bc_pretrain,
    )

    print(f"\nTraining complete! Model saved to {save_path}")


if __name__ == "__main__":
    main()
