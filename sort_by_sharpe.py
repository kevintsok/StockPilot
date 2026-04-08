import json
import sys
top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

with open('models/price_transformer_2025-train20250331-val20260327_strategies_comparison.json') as f:
    data = json.load(f)
results = data['results']
results.sort(key=lambda r: r['metrics']['sharpe_gross'], reverse=True)

print(f"{'排名':<4} {'策略':<35} {'总收益':>10} {'夏普':>8} {'最大回撤':>10} {'年化收益':>10}")
print('-' * 85)
for rank, r in enumerate(results[:top_n], 1):
    m = r['metrics']
    tag = r.get('tag','')
    name = r['strategy_name']
    label = f"{name} [{tag}]" if tag else name
    sharpe = m['sharpe_gross']
    ann = m.get('annual_return_gross', 0)
    print(f"{rank:<4} {label:<35} {m['total_return_gross']*100:>+9.1f}% {sharpe:>8.3f} {m['max_drawdown_gross']*100:>+9.1f}% {ann*100:>+9.1f}%")

# Save top 10 names to file for next step
top10_names = [r['strategy_name'] for r in results[:top_n]]
with open('models/top10_strategies.json', 'w') as f:
    json.dump(top10_names, f)

best = results[0]
print(f"\nBest strategy: {best['strategy_name']} [{best.get('tag','')}]")
print(f"  Sharpe: {best['metrics']['sharpe_gross']:.3f}")
print(f"  Total Return: {best['metrics']['total_return_gross']*100:+.1f}%")
print(f"  Max DD: {best['metrics']['max_drawdown_gross']*100:+.1f}%")
