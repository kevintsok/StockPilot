import sys
sys.path.insert(0, 'src')
import pandas as pd
from auto_select_stock.predict.backtest import _collect_signals_batched, BacktestConfig, filter_a_share_symbols
from auto_select_stock.data.storage import list_symbols
from auto_select_stock.predict.inference import PricePredictor
from pathlib import Path

cfg = BacktestConfig(
    checkpoint='models/price_transformer_full.pt',
    start_date='2023-01-03',
    end_date='2023-01-10',  # Just 1 week for debug
)
predictor = PricePredictor(cfg.checkpoint)
symbols = filter_a_share_symbols(list_symbols(base_dir=cfg.base_dir))[:50]  # Only first 50 for speed

print(f"Testing with {len(symbols)} symbols")
print(f"Model horizons: {predictor.horizons}")
print(f"Model target_mode: {getattr(predictor.cfg, 'target_mode', 'close')}")
print(f"close_idx: {predictor.close_idx}")

daily_signals = _collect_signals_batched(
    symbols, predictor, cfg,
    pd.Timestamp('2023-01-03'), pd.Timestamp('2023-01-10'),
    show_progress=False
)

print(f"\nDates with signals: {sorted(daily_signals.keys())}")
for dt, sigs in sorted(daily_signals.items())[:5]:
    print(f"\nDate {dt.date()}: {len(sigs)} signals")
    # Show first 3 signals
    for s in sigs[:3]:
        print(f"  {s.symbol}: pred={s.predicted_ret:.4f} realized={s.realized_ret:.4f} "
              f"entry={s.entry_price:.2f} next_open={s.next_open:.2f} next_close={s.next_close:.2f} "
              f"auc={s.auc_limit}")
    break