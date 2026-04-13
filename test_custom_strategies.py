import warnings
warnings.filterwarnings('ignore')
import traceback
import sys

from auto_select_stock.predict.strategies.registry import make_strategy, StrategyRegistry
from pathlib import Path

r = StrategyRegistry(Path('src/auto_select_stock/predict/strategies/configs'))
custom = [c for c in r.list_strategies() if 'custom' in c['type']]
print(f'Custom strategies: {len(custom)}')

for c in custom:
    try:
        cfg = r.get(c['name'])
        s = make_strategy(cfg)
        print(f'  {cfg["name"]}: {type(s).__name__}')
    except Exception as e:
        print(f'  ERROR {c["name"]}: {e}')
        traceback.print_exc()

print('All custom strategies instantiated OK')
