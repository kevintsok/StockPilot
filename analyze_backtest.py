import json
import numpy as np

with open('models/price_transformer_2025-train20250331-val20260327_strategies_comparison.json', 'r') as f:
    data = json.load(f)

conf_strategy = None
for r in data['results']:
    if r['strategy_name'] == 'Conf-MC5bp-5d':
        conf_strategy = r
        break

print('=== Conf-MC5bp-5d Metrics ===')
for k, v in conf_strategy['metrics'].items():
    print(f'  {k}: {v}')

ts = conf_strategy['timeseries']
print(f'\n=== Timeseries ({len(ts)} entries) ===')
print(f"Date range: {ts[0]['date']} to {ts[-1]['date']}")

dates = [t['date'] for t in ts]
gross_rets = [t['gross_ret'] for t in ts]
capital = [t['capital'] for t in ts]

rets_array = np.array(gross_rets)
capital_array = np.array(capital)

print(f'\n=== Daily Return Statistics ===')
print(f'Mean daily return: {np.mean(rets_array):.4f} ({np.mean(rets_array)*100:.2f}%)')
print(f'Std daily return: {np.std(rets_array):.4f} ({np.std(rets_array)*100:.2f}%)')
print(f'Min: {np.min(rets_array):.4f}, Max: {np.max(rets_array):.4f}')

worst_idx = np.argmin(rets_array)
best_idx = np.argmax(rets_array)
print(f'\n=== Worst Day ===')
print(f'Date: {dates[worst_idx]}, Return: {rets_array[worst_idx]:.4f} ({rets_array[worst_idx]*100:.2f}%)')
print(f'\n=== Best Day ===')
print(f'Date: {dates[best_idx]}, Return: {rets_array[best_idx]:.4f} ({rets_array[best_idx]*100:.2f}%)')

positive_days = np.sum(rets_array > 0)
negative_days = np.sum(rets_array < 0)
print(f'\n=== Win Rate ===')
print(f'Positive: {positive_days} ({positive_days/len(rets_array)*100:.1f}%), Negative: {negative_days} ({negative_days/len(rets_array)*100:.1f}%)')

# Longest losing streak
best_streak_len = 0
best_streak_start = 0
best_streak_end = 0
current_streak = 0
current_streak_start = 0
for i, r in enumerate(rets_array):
    if r < 0:
        if current_streak == 0:
            current_streak_start = i
        current_streak += 1
    else:
        if current_streak > best_streak_len:
            best_streak_len = current_streak
            best_streak_start = current_streak_start
            best_streak_end = i - 1
        current_streak = 0
if current_streak > best_streak_len:
    best_streak_len = current_streak
    best_streak_start = current_streak_start
    best_streak_end = len(rets_array) - 1

print(f'\n=== Longest Losing Streak ===')
print(f'{best_streak_len} days: {dates[best_streak_start]} to {dates[best_streak_end]}')

# Drawdown
peak = np.maximum.accumulate(capital_array)
drawdown = capital_array / peak - 1.0
max_dd_idx = np.argmin(drawdown)
peak_before = np.argmax(capital_array[:max_dd_idx+1])
print(f'\n=== Max Drawdown ===')
print(f'Max DD: {drawdown[max_dd_idx]:.4f} ({drawdown[max_dd_idx]*100:.2f}%)')
print(f'Date: {dates[max_dd_idx]}, Peak: {dates[peak_before]} (capital: {capital_array[peak_before]:.0f})')

# Top 10 worst
print(f'\n=== Top 10 Worst Days ===')
worst_indices = np.argsort(rets_array)[:10]
for idx in worst_indices:
    print(f'  {dates[idx]}: {rets_array[idx]:.4f} ({rets_array[idx]*100:.2f}%)')

# Top 10 best
print(f'\n=== Top 10 Best Days ===')
best_indices = np.argsort(rets_array)[-10:][::-1]
for idx in best_indices:
    print(f'  {dates[idx]}: {rets_array[idx]:.4f} ({rets_array[idx]*100:.2f}%)')

# Monthly
print(f'\n=== Monthly Performance ===')
monthly_returns = {}
for d, r in zip(dates, gross_rets):
    month = d[:7]
    if month not in monthly_returns:
        monthly_returns[month] = []
    monthly_returns[month].append(r)

for month in sorted(monthly_returns.keys()):
    rets = monthly_returns[month]
    total = np.prod(np.array(rets) + 1) - 1
    n_pos = np.sum(np.array(rets) > 0)
    print(f'  {month}: total={total*100:.2f}%, pos={n_pos}/{len(rets)}')

# All strategies comparison
print(f'\n=== All Strategies Comparison ===')
for r in data['results']:
    m = r['metrics']
    print(f"  {r['strategy_name']}: Sharpe={m.get('sharpe_gross', 'N/A'):.3f}, MaxDD={m.get('max_drawdown_gross', 'N/A'):.3f}, AnnRet={m.get('annual_return_gross', 'N/A'):.3f}, Vol={m.get('annual_vol_gross', 'N/A'):.3f}")
