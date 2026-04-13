"""
Debug: run the actual backtest signal collection for a few days and inspect.
"""
import sys
sys.path.insert(0, 'src')
import math
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from auto_select_stock.predict.inference import PricePredictor
from auto_select_stock.predict.backtest import _collect_signals_batched, BacktestConfig, filter_a_share_symbols
from auto_select_stock.storage import list_symbols

# Load model
p = PricePredictor('models/price_transformer_multihorizon_full.pt')
print(f"Model: target_mode={p.cfg.target_mode}, horizons={p.horizons}")

# Setup backtest
symbols = list_symbols(base_dir=Path('data'))
a_shares = filter_a_share_symbols(symbols)
print(f"Total A-share symbols: {len(a_shares)}")

# Test with a SMALL subset for quick debugging
test_symbols = a_shares[:50]  # First 50 symbols
print(f"Testing with {len(test_symbols)} symbols")

cfg = BacktestConfig(
    checkpoint=Path('models/price_transformer_multihorizon_full.pt'),
    start_date='2023-01-03',
    end_date='2023-01-10',  # Just first week
    symbols=test_symbols,
    cost_bps=0,
    slippage_bps=0,
    base_dir=Path('data'),
)

# Collect signals
daily_signals = _collect_signals_batched(
    test_symbols, p, cfg,
    start_date=pd.Timestamp('2023-01-03'),
    end_date=pd.Timestamp('2023-01-10'),
    show_progress=False,
    horizon='1d',
)

print(f"\nDays with signals: {len(daily_signals)}")
dates = sorted(daily_signals.keys())
print(f"Date range: {dates[0]} to {dates[-1]}")

for dt in dates[:3]:
    signals = daily_signals[dt]
    print(f"\n=== {dt.date()} ===")
    print(f"  Number of signals: {len(signals)}")

    # Show distribution of predicted and realized returns
    pred_rets = [s[1] for s in signals]  # predicted_ret
    real_rets = [s[2] for s in signals]  # realized_ret

    pred_rets_f = [r for r in pred_rets if not np.isnan(r)]
    real_rets_f = [r for r in real_rets if not np.isnan(r)]

    print(f"  Predicted ret: min={min(pred_rets_f):.4%}, max={max(pred_rets_f):.4%}, mean={np.mean(pred_rets_f):.4%}")
    print(f"  Realized ret: min={min(real_rets_f):.4%}, max={max(real_rets_f):.4%}, mean={np.mean(real_rets_f):.4%}")

    # Top 5 by predicted return
    sorted_signals = sorted(signals, key=lambda s: s[1], reverse=True)
    top5 = sorted_signals[:5]
    print(f"  Top 5 stocks by prediction:")
    for s in top5:
        print(f"    {s[0]}: pred={s[1]:.4%}, real={s[2]:.4%}")

    # What would TopK strategy return?
    top_sigs = [s for s in sorted_signals if s[1] > 0][:5]
    if top_sigs:
        total_pred = sum(s[1] for s in top_sigs)
        weights = {s[0]: s[1]/total_pred for s in top_sigs}
        gross = sum(w * s[2] for s, w in zip(top_sigs, weights.values()))
        print(f"  TopK portfolio weights: {[(s[0], f'{w:.3f}') for s, w in weights.items()]}")
        print(f"  TopK gross return: {gross:.4%}")
        print(f"  (Each stock contributes: {[f'{weights[s[0]]*s[2]:.4%}' for s in top_sigs]})")

    # Also check what the top 5 REALIZED returns are
    sorted_real = sorted(signals, key=lambda s: s[2], reverse=True)
    top5_real = sorted_real[:5]
    print(f"  Top 5 stocks by REALIZED return:")
    for s in top5_real:
        print(f"    {s[0]}: pred={s[1]:.4%}, real={s[2]:.4%}")
