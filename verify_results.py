import json

with open('/mnt/d/Projects/auto-select-stock/models/price_transformer_multihorizon_full_strategies_comparison.json') as f:
    d = json.load(f)

print('Period:', d['start'], '->', d['end'])
print('Initial capital:', d['initial_capital'])

r = d['results'][0]
m = r['metrics']
ts = r['timeseries']
print('Strategy:', r['strategy_name'])
print('Initial:', m['initial_capital'], 'Final:', m['final_capital'])
print('Return:', m['total_return_gross'] * 100)
print('Sharpe:', m['sharpe_gross'])
print('Max DD:', m['max_drawdown_gross'] * 100)
print('Days:', m['num_days'])
print('First 5 days:')
for t in ts[:5]:
    c = t['capital']
    g = t['gross_ret'] * 100
    h = t['holdings_value']
    cash = t['cash']
    print('  %s capital=%.0f ret=%.2f%% holdings=%.0f cash=%.0f' % (t['date'], c, g, h, cash))
print('Last 3 days:')
for t in ts[-3:]:
    c = t['capital']
    g = t['gross_ret'] * 100
    h = t['holdings_value']
    cash = t['cash']
    print('  %s capital=%.0f ret=%.2f%% holdings=%.0f cash=%.0f' % (t['date'], c, g, h, cash))

trades = r['trades']
buy_trades = [t for t in trades if t['action'] == 'buy' and t['shares'] > 0]
sell_trades = [t for t in trades if t['action'] == 'sell']
print('Total trades:', len(trades), 'Buy:', len(buy_trades), 'Sell:', len(sell_trades))
print('First 5 successful buy trades:')
for t in buy_trades[:5]:
    print('  %s BUY %s %d@%.2f=%.2f' % (t['date'], t['symbol'], t['shares'], t['price'], t['amount']))

print()
print('All strategies:')
for r in d['results']:
    m = r['metrics']
    print('  %-30s final=%.0f return=%+.2f%% sharpe=%.2f maxdd=%.2f%%' % (
        r['strategy_name'], m['final_capital'], m['total_return_gross']*100,
        m['sharpe_gross'], m['max_drawdown_gross']*100))
