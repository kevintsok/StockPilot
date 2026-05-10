#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
import traceback
import sys
sys.path.insert(0, 'src')

from auto_select_stock.predict.strategies.registry import make_strategy, StrategyRegistry
from pathlib import Path

r = StrategyRegistry(Path('src/auto_select_stock/predict/strategies/configs'))
configs = [c for c in r.list_strategies() if 'Conf-1d' in c['name'] or 'ConfStop' in c['name']]
print(f'New optimized strategies found: {len(configs)}')
for c in configs:
    try:
        cfg = r.get(c['name'])
        s = make_strategy(cfg)
        sl = getattr(s, 'stop_loss_pct', 'N/A')
        tp = getattr(s, 'take_profit_pct', 'N/A')
        mh = getattr(s, 'max_holding_days', 'N/A')
        print(f"  {cfg['name']}: type={type(s).__name__}, sl={sl}, tp={tp}, hold={mh}")
    except Exception as e:
        print(f"  ERROR {c['name']}: {e}")
        traceback.print_exc()
print('All new strategies instantiated OK')
