import sys
sys.path.insert(0, 'src')
from auto_select_stock.predict.strategies.runner import _INITIAL_CAPITAL, _LOT_SIZE
from auto_select_stock.predict.backtest import BacktestConfig
from auto_select_stock.predict.strategies.runner import run_all_strategies_shared
from pathlib import Path

cfg = BacktestConfig(
    checkpoint='models/price_transformer_full.pt',
    start_date='2023-01-01',
    end_date='2023-01-31',  # Just 1 month for debug
    cost_bps=15,
    slippage_bps=10,
)
results = run_all_strategies_shared(
    Path('src/auto_select_stock/predict/strategies/configs'),
    cfg,
    show_progress=False
)

# Debug: print first 10 capital values for first strategy
r = results[0]
print(f"Strategy: {r.strategy_name}")
print(f"Capital (first 10): {list(r.cumulative[:10]) if r.cumulative is not None else 'None'}")
print(f"Capital (last 5): {list(r.cumulative[-5:]) if r.cumulative is not None else 'None'}")
print(f"Daily returns (first 5): {list(r.daily_returns[:5]) if r.daily_returns is not None else 'None'}")
print(f"Final capital: {r.metrics['final_capital']}")
print(f"Total return: {r.metrics['total_return_gross']}")
print(f"Initial capital: {_INITIAL_CAPITAL}")
print(f"Num trades: {len(r.trades)}")
print(f"Trades sample: {r.trades[:5]}")
