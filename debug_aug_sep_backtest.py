import sys
sys.path.insert(0, 'src')
from auto_select_stock.predict.backtest import BacktestConfig
from auto_select_stock.predict.strategies.runner import run_all_strategies_shared
from pathlib import Path

cfg = BacktestConfig(
    checkpoint='models/price_transformer_full.pt',
    start_date='2024-08-01',
    end_date='2024-09-30',
    cost_bps=15,
    slippage_bps=10,
)
results = run_all_strategies_shared(
    Path('src/auto_select_stock/predict/strategies/configs'),
    cfg,
    show_progress=False
)
r = results[0]
print('Strategy:', r.strategy_name, flush=True)
extreme_count = 0
for i, (date_idx) in enumerate(r.capital.index):
    ret = r.daily_returns.iloc[i] * 100
    date_str = str(date_idx.date())
    marker = ''
    if abs(ret) > 30:
        extreme_count += 1
        marker = '  *** EXTREME ***'
    if '2024-08' in date_str or '2024-09' in date_str[:7]:
        print(f'  {date_str}: cap={r.capital.iloc[i]:.0f} ret={ret:+.2f}%{marker}', flush=True)
print(f'Total extreme returns (>30%): {extreme_count}', flush=True)
print(f'Final capital: {r.capital.iloc[-1]:.0f}  Total return: {r.metrics["total_return_gross"]*100:+.1f}%', flush=True)
split_trades = [t for t in r.trades if t.reason == 'split_filter']
print(f'Split filter trades: {len(split_trades)}', flush=True)
