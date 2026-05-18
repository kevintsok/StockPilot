#!/usr/bin/env python3
"""
E2E Transformer v3: Continuous Output (like BC)

Key insight: BC works because it uses CONTINUOUS output [0, 0.12], not 4 discrete classes.
Discrete classification with heavy class imbalance is fundamentally hard.

This version:
- Outputs continuous position size [0, 0.12] like BC
- Uses Transformer encoder for better sequence modeling
- Uses MSE loss (like BC) instead of CrossEntropy
- Much simpler training objective

Usage:
    python -m auto_select_stock.rl.train_e2e_v3 --device cuda --epochs 20
"""

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..config import DATA_DIR
from ..core.torch_model import PositionalEncoding
from ..data.storage import list_symbols


class ContinuousE2E(nn.Module):
    """
    E2E with CONTINUOUS output (like BC).

    Input: price sequence (B, seq_len, feature_dim)
    Output: position size (B, 1) in [0, max_position_pct]
    """

    def __init__(
        self,
        feature_dim: int = 25,
        seq_len: int = 60,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_position_pct: float = 0.15,
    ):
        super().__init__()
        self.max_position_pct = max_position_pct

        # Price encoder (Transformer)
        self.price_proj = nn.Linear(feature_dim, d_model)
        self.price_ln = nn.LayerNorm(d_model)
        self.price_pe = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.price_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head (predicts position size directly)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._cached_mask: Optional[Tensor] = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        if self._cached_mask is None or self._cached_mask.size(0) != seq_len or self._cached_mask.device != device:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            self._cached_mask = mask
        return self._cached_mask

    def forward(self, price_seq: Tensor) -> Tensor:
        """
        Args:
            price_seq: (batch, seq_len, feature_dim)
        Returns:
            position_size: (batch, 1) in [0, max_position_pct]
        """
        x = self.price_proj(price_seq)
        x = self.price_ln(x)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.price_pe(x)
        mask = self._causal_mask(x.size(0), x.device)
        x = self.price_encoder(x, mask=mask)
        x = x[-1]  # Take last timestep
        out = self.head(x)
        return torch.sigmoid(out) * self.max_position_pct


class TradingDatasetV3(Dataset):
    """
    Dataset for continuous E2E.

    Label is position size [0, 0.12] like BC.
    No discrete action classification - just regression to expert position.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        seq_len: int = 60,
        close_index: int = 3,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.015,
        stop_loss: float = 0.05,
        take_profit: float = 0.15,
        lookback: int = 5,
    ):
        self.seq_len = seq_len
        self.close_index = close_index
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.lookback = lookback

        self.samples: List[Tuple[np.ndarray, float]] = []
        self._build_samples(sequences)

        # Stats
        actions = [a for _, a in self.samples]
        print(f"  Label distribution: mean={np.mean(actions):.4f}, std={np.std(actions):.4f}")
        print(f"  Zero positions: {sum(1 for a in actions if a < 0.01) / len(actions) * 100:.1f}%")
        print(f"  Non-zero positions: {sum(1 for a in actions if a >= 0.01) / len(actions) * 100:.1f}%")

    def _build_samples(self, sequences: List[np.ndarray]):
        for arr in sequences:
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) <= self.seq_len + 10:
                continue

            has_position = False
            entry_price = 0.0

            for i in range(self.seq_len, len(arr) - 1):
                window = arr[i - self.seq_len:i + 1]
                price_seq = window[:-1]

                # Momentum prediction (like BC)
                if i >= self.lookback:
                    pred_ret = (arr[i, self.close_index] - arr[i - self.lookback, self.close_index]) / arr[i - self.lookback, self.close_index]
                else:
                    pred_ret = 0.0

                current_price = arr[i, self.close_index]
                unrealized_pnl = 0.0
                if has_position and entry_price > 0:
                    unrealized_pnl = (current_price - entry_price) / entry_price

                # Expert action (same logic as BC)
                if has_position:
                    if unrealized_pnl < -self.stop_loss:
                        target = 0.0
                        has_position = False
                        entry_price = 0.0
                    elif unrealized_pnl > self.take_profit:
                        target = 0.0
                        has_position = False
                        entry_price = 0.0
                    elif pred_ret < self.sell_threshold:
                        target = 0.0
                        has_position = False
                        entry_price = 0.0
                    else:
                        target = 0.12
                else:
                    if pred_ret > self.buy_threshold:
                        target = 0.12
                        has_position = True
                        entry_price = current_price
                    else:
                        target = 0.0

                self.samples.append((price_seq, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        price_seq, target = self.samples[idx]
        return (
            torch.tensor(price_seq, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


def prepare_trading_data(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    base_dir: Path = DATA_DIR,
    seq_len: int = 60,
) -> Tuple[List[np.ndarray], int, int]:
    """Load price sequences."""
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


def train_e2e_v3(
    train_sequences: List[np.ndarray],
    val_sequences: List[np.ndarray],
    close_index: int,
    feature_dim: int,
    device: torch.device,
    seq_len: int = 60,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    save_path: Optional[Path] = None,
) -> ContinuousE2E:
    """Train continuous E2E model."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"\n{'='*60}")
    print("E2E v3 Training (Continuous Output)")
    print(f"seq_len={seq_len}, d_model={d_model}, layers={num_layers}")
    print(f"{'='*60}")

    train_ds = TradingDatasetV3(train_sequences, seq_len=seq_len, close_index=close_index)
    val_ds = TradingDatasetV3(val_sequences, seq_len=seq_len, close_index=close_index)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ContinuousE2E(
        feature_dim=feature_dim,
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for price_seq, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            price_seq = price_seq.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(price_seq).squeeze(-1)

            # MSE loss like BC
            loss = F.mse_loss(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []

        with torch.no_grad():
            for price_seq, targets in val_loader:
                price_seq = price_seq.to(device)
                targets = targets.to(device)

                preds = model(price_seq).squeeze(-1)
                loss = F.mse_loss(preds, targets)
                val_loss += loss.item()
                val_preds.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step()

        # Prediction distribution
        val_preds = np.array(val_preds)
        pred_positive = (val_preds > 0.05).sum() / len(val_preds) * 100

        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
              f"pred_positive={pred_positive:.1f}%")

        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train E2E Transformer v3 (continuous)")
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    parser.add_argument("--symbols", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="models/e2e_v3.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    symbols = list_symbols(base_dir=DATA_DIR)[:args.symbols]

    all_sequences, close_index, feature_dim = prepare_trading_data(
        symbols, args.start, args.end, seq_len=args.seq_len
    )

    print(f"Loaded {len(all_sequences)} sequences, feature_dim={feature_dim}")

    n_train = int(len(all_sequences) * 0.8)
    train_sequences = all_sequences[:n_train]
    val_sequences = all_sequences[n_train:]

    print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = train_e2e_v3(
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
    )

    print(f"\nTraining complete! Model saved to {save_path}")


if __name__ == "__main__":
    main()
