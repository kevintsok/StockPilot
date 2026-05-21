#!/usr/bin/env python3
"""
Backtest for E2E v3 with BC Warm-Start.
"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from auto_select_stock.config import DATA_DIR
from auto_select_stock.data.storage import list_symbols, load_stock_history
from auto_select_stock.predict.data import compute_technical_indicators, PRICE_FEATURE_COLUMNS
from .train_e2e_v3_bc_warmup import ContinuousE2EWithBCWarmup


def load_stock_data(symbols, start_date="2024-01-01"):
    """Load stock sequences."""
    import pandas as pd

    sequences = []
    for sym in tqdm(symbols, desc="Loading data"):
        try:
            arr = load_stock_history(sym, base_dir=DATA_DIR)
            if arr is None or len(arr) < 120:
                continue

            df = pd.DataFrame(arr)
            df["date"] = pd.to_datetime(df["date"]).dt.floor("D").astype("datetime64[ns]")
            df.sort_values("date", inplace=True)
            df = df[df["date"] >= pd.to_datetime(start_date)]

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

            sequences.append((sym, arr_out))
        except Exception:
            continue

    return sequences


def backtest_e2e_v3_bc_warmup(checkpoint_path, bc_pretrained_path, test_symbols_start=200, test_symbols_end=250):
    """Run backtest on test symbols."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Load model
    model = ContinuousE2EWithBCWarmup(
        bc_pretrained_path=bc_pretrained_path,
        feature_dim=25, seq_len=60, d_model=128, nhead=4, num_layers=4
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # Load test data
    all_symbols = list_symbols(base_dir=DATA_DIR)
    test_symbols = all_symbols[test_symbols_start:test_symbols_end]

    sequences = load_stock_data(test_symbols, start_date="2024-01-01")
    print(f"\nTest sequences: {len(sequences)}")

    if not sequences:
        print("No data loaded!")
        return

    trade_threshold = 0.05
    commission = 0.0003
    stamp_tax = 0.001
    slippage = 0.0001
    max_position_pct = 0.15

    portfolio_values = []
    all_actions = []
    trade_count = 0

    for sym, arr in tqdm(sequences, desc="Backtesting"):
        if len(arr) < 120:
            continue

        portfolio = 1.0
        position = 0.0
        entry_price = 0.0

        for t in range(60, len(arr) - 1):
            window = arr[t - 60:t]
            price_seq = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(price_seq).item()

            current_price = arr[t, 3]
            next_price = arr[t + 1, 3]

            is_new = (action > trade_threshold and position < 0.01)
            is_close = (action < trade_threshold and position > 0.01)

            if is_new:
                cost = action * (1 + commission + slippage)
                portfolio *= (1 - cost)
                position = action
                entry_price = current_price
                trade_count += 1
            elif is_close:
                rev = position * (1 - commission - stamp_tax - slippage)
                portfolio *= (1 + rev)
                position = 0.0
                entry_price = 0.0
            elif position > 0.01:
                delta = abs(action - position)
                if delta > 0.001:
                    cost = delta * (commission + slippage)
                    portfolio *= (1 - cost)
                position = action

            if position > 0.01:
                ret = (next_price - current_price) / current_price if current_price > 0 else 0
                portfolio *= (1 + ret * position)

        portfolio_values.append(portfolio)
        all_actions.append(position)

    returns = [v - 1.0 for v in portfolio_values]
    print("\n" + "=" * 50)
    print("E2E v3 BC Warm-Start Backtest Results")
    print("=" * 50)
    print(f"Total Return:     {np.mean(returns)*100:.2f}%")
    print(f"Std of Return:   {np.std(returns)*100:.2f}%")
    print(f"Sharpe Ratio:   {np.mean(returns)/(np.std(returns)+1e-8)*np.sqrt(252):.4f}")
    print(f"Trade Count:    {trade_count}")
    print(f"Avg Position:   {np.mean(all_actions):.4f}")
    print(f"Positive Positions: {sum(1 for p in all_actions if p > 0.01) / len(all_actions) * 100:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/e2e_v3_bc_warmup.pt")
    parser.add_argument("--bc-pretrained", type=str, default="models/bc_pretrain.pt")
    args = parser.parse_args()

    backtest_e2e_v3_bc_warmup(args.checkpoint, args.bc_pretrained)
