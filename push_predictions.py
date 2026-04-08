#!/usr/bin/env python3
"""
Direct prediction push using the best strategy (Conf-MC5bp-5d).
Bypasses data freshness check.
"""
import sys
sys.path.insert(0, 'src')

import os
from auto_select_stock.notify.push_providers import PushPlusProvider
from auto_select_stock.notify.daily_report import generate_report, get_latest_price_date

CHECKPOINT = "/mnt/d/Projects/auto-select-stock/models/price_transformer_2025-train20250331-val20260327.pt"
TOKEN = os.getenv("PUSHPLUS_TOKEN", "183cae5e7d8148f0b85754a2912fc81c")
STRATEGY = "Conf-MC5bp-5d"
TOP_K = 10
HORIZON = "5d"

print(f"Latest price date: {get_latest_price_date()}")
print(f"Strategy: {STRATEGY}, Horizon: {HORIZON}, Top-K: {TOP_K}")

html, results = generate_report(
    checkpoint=CHECKPOINT,
    strategy=STRATEGY,
    top_k=TOP_K,
    horizon=HORIZON,
)

print(f"\nTop {len(results)} stocks by {STRATEGY} ({HORIZON}):")
for sym, pred_rets, weight in results:
    if isinstance(pred_rets, dict):
        ret_str = ", ".join([f"{h}:{v*100:+.2f}%" for h, v in sorted(pred_rets.items())])
        print(f"  {sym}: [{ret_str}] weight={weight:.4f}")
    else:
        print(f"  {sym}: {pred_rets*100:+.2f}% weight={weight:.4f}")

# Send via PushPlus
latest_date = get_latest_price_date()
provider = PushPlusProvider(token=TOKEN)
provider.send(title=f"StockPilot {STRATEGY} Top-{len(results)} {latest_date}", content=html)
print(f"\nPushPlus: sent {len(results)} stocks")
