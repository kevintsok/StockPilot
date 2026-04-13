import sys
sys.path.insert(0, 'src')
from auto_select_stock.predict.backtest import BacktestConfig
from auto_select_stock.predict.strategies.runner import run_all_strategies_shared
from pathlib import Path

# Only test Aug-Sep 2024 (fast)
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
# Print daily capital for first strategy
r = results[0]
print('Strategy:', r.strategy_name)
for i, (date_idx) in enumerate(r.capital.index):
    print(f'{str(date_idx.date())}: cap={r.capital.iloc[i]:.0f} ret={r.daily_returns.iloc[i]*100:+.2f}% hold={r.holdings_value.iloc[i]:.0f} cash={r.cash_series.iloc[i]:.0f}')

# Check for extreme returns
print('\n--- Extreme returns check (>30% or <-30%) ---')
found = False
for i, (date_idx) in enumerate(r.capital.index):
    ret = r.daily_returns.iloc[i]
    if abs(ret) > 0.30:
        found = True
        print(f'  EXTREME: {str(date_idx.date())}: ret={ret*100:+.2f}%')
if not found:
    print('  None found')
