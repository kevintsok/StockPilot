import json

d = json.load(open('models/backtest_diverse_30.json'))
initial = d['initial_capital']
print(f"initial_capital: {initial}")

r = d['results'][0]  # RiskParity-VL10-1d
ts = r['timeseries']
first_cap = ts[0].get('capital') or initial
last_cap = ts[-1].get('capital')
print(f"first_capital: {first_cap}")
print(f"last_capital: {last_cap}")
print(f"last_capital / first_cap - 1 = {last_cap / first_cap - 1:.4f}")

m = r['metrics']
print(f"total_return_gross from metrics: {m.get('total_return_gross')}")
print()
print("What the right axis currently shows:")
print(f"  Index close[0] = 3078, close[-1] = 3347")
print(f"  Index return = 3347/3078 - 1 = {3347/3078 - 1:.4f} = {(3347/3078 - 1)*100:.1f}%")
print()
print("What strategy capital looks like on left axis:")
print(f"  first_capital = {first_cap}")
print(f"  last_capital = {last_cap}")
print(f"  last_capital / initial - 1 = {last_cap / initial - 1:.4f} = {(last_cap / initial - 1)*100:.1f}%")
