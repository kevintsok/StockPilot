import sys
sys.path.insert(0, 'src')
from pathlib import Path
import pandas as pd
from auto_select_stock.predict.backtest import _collect_signals_batched, BacktestConfig
from auto_select_stock.predict.inference import PricePredictor

p = PricePredictor('models/price_transformer_multihorizon_full.pt')
cfg = BacktestConfig(
    checkpoint=Path('models/price_transformer_multihorizon_full.pt'),
    start_date='2024-06-01',
    end_date='2024-06-05',
    symbols=['600000'],
    cost_bps=0, slippage_bps=0, base_dir=Path('data'),
)
daily_signals = _collect_signals_batched(
    ['600000'], p, cfg,
    start_date=pd.Timestamp('2024-06-01'),
    end_date=pd.Timestamp('2024-06-05'),
    show_progress=False,
    horizon='1d',
)
print('Days with signals:', len(daily_signals))
for dt, signals in sorted(daily_signals.items()):
    print(f'{dt.date()}: {len(signals)} signals')
    for s in signals[:3]:
        print(f'  {s[0]}: pred={s[1]:.4%}, real={s[2]:.4%}, entry_price={s[5]:.2f}, auc={s[6]}')
