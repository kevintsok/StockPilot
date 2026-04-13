import sys
sys.path.insert(0, 'src')
from auto_select_stock.predict.strategies.runner import run_all_strategies_shared, _INITIAL_CAPITAL
from auto_select_stock.predict.backtest import BacktestConfig
from pathlib import Path
import json

cfg = BacktestConfig(
    checkpoint='models/price_transformer_full.pt',
    start_date='2023-01-01',
    end_date='2024-12-31',
    cost_bps=15,
    slippage_bps=10,
)
results = run_all_strategies_shared(
    Path('src/auto_select_stock/predict/strategies/configs'),
    cfg,
    show_progress=True
)

print("\n=== Backtest Results ===")
sorted_results = sorted(results, key=lambda r: r.metrics.get('total_return_gross', 0), reverse=True)
for r in sorted_results:
    total_ret = r.metrics['total_return_gross'] * 100
    sharpe = r.metrics['sharpe_gross']
    mdd = r.metrics['max_drawdown_gross'] * 100
    winrate = r.metrics.get('win_rate', float('nan'))
    ann_ret = r.metrics['annual_return_gross'] * 100
    print(f"  {r.strategy_name:30s}: ret={total_ret:+8.1f}% ann={ann_ret:+7.1f}% sharpe={sharpe:6.2f} mdd={mdd:7.1f}% trades={len(r.trades)}")

# Save results with full trade details and daily time series
out = []
for r in results:
    # Build daily time series
    daily_records = []
    if r.capital is not None:
        for i, (date_idx) in enumerate(r.capital.index):
            daily_records.append({
                "date": str(date_idx.date()) if hasattr(date_idx, 'date') else str(date_idx),
                "capital": float(r.capital.iloc[i]),
                "holdings_value": float(r.holdings_value.iloc[i]) if r.holdings_value is not None else 0.0,
                "cash": float(r.cash_series.iloc[i]) if r.cash_series is not None else 0.0,
                "daily_return": float(r.daily_returns.iloc[i]) if r.daily_returns is not None else 0.0,
                "turnover": float(r.turnover.iloc[i]) if r.turnover is not None else 0.0,
            })

    # Build trade log
    trade_records = []
    for t in r.trades:
        trade_records.append({
            "date": t.date,
            "symbol": t.symbol,
            "action": t.action,
            "price": float(t.price),
            "shares": int(t.shares),
            "amount": float(t.amount),
            "reason": t.reason,
        })

    out.append({
        "strategy": r.strategy_name,
        "tag": r.tag,
        "start": "2023-01-01",
        "end": "2024-12-31",
        "initial_capital": _INITIAL_CAPITAL,
        "total_return": r.metrics['total_return_gross'],
        "annual_return": r.metrics['annual_return_gross'],
        "sharpe": r.metrics['sharpe_gross'],
        "max_drawdown": r.metrics['max_drawdown_gross'],
        "win_rate": r.metrics.get('win_rate', float('nan')),
        "final_capital": r.metrics['final_capital'],
        "num_trades": len(r.trades),
        "timeseries": daily_records,
        "trades": trade_records,
    })

with open('models/backtest_results_full.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved to models/backtest_results_full.json")

# Also save a summary CSV for quick viewing
import csv
csv_path = 'models/backtest_results_summary.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Strategy', 'Total Return (%)', 'Annual Return (%)', 'Sharpe', 'Max Drawdown (%)', 'Final Capital', 'Num Trades'])
    for r in sorted(results, key=lambda x: x.metrics['total_return_gross'], reverse=True):
        writer.writerow([
            r.strategy_name,
            f"{r.metrics['total_return_gross']*100:.2f}",
            f"{r.metrics['annual_return_gross']*100:.2f}",
            f"{r.metrics['sharpe_gross']:.4f}",
            f"{r.metrics['max_drawdown_gross']*100:.2f}",
            f"{r.metrics['final_capital']:.0f}",
            len(r.trades),
        ])
print(f"Saved to {csv_path}")
