"""
CLI entry point for the daily notification service.

Usage:
    PUSHPLUS_TOKEN=xxx PYTHONPATH=./src python -m auto_select_stock.notify.runner
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .config import PUSHPLUS_TOKEN, NOTIFY_CHECKPOINT, NOTIFY_STRATEGY, NOTIFY_TOP_K
from .push_providers import PushPlusProvider
from .daily_report import generate_report, get_latest_price_date


def main() -> None:
    if not PUSHPLUS_TOKEN:
        print("ERROR: PUSHPLUS_TOKEN environment variable is not set", file=sys.stderr)
        sys.exit(1)

    latest_date = get_latest_price_date()
    print(f"Latest price date in DB: {latest_date}")

    html, results = generate_report(
        checkpoint=NOTIFY_CHECKPOINT,
        strategy=NOTIFY_STRATEGY,
        top_k=NOTIFY_TOP_K,
    )

    title = f"StockPilot 每日推荐 {latest_date}"
    provider = PushPlusProvider(token=PUSHPLUS_TOKEN)
    provider.send(title=title, content=html)

    print(f"OK: pushed {len(results)} stocks to PushPlus")
    for sym, pred_ret, weight in results:
        print(f"  {sym}: {pred_ret*100:+.2f}% (weight={weight:.4f})")


if __name__ == "__main__":
    main()
