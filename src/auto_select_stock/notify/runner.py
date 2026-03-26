"""
CLI entry point for the daily notification service.

Usage:
    # Daily stock recommendations
    PUSHPLUS_TOKEN=xxx PYTHONPATH=./src python -m auto_select_stock.notify.runner

    # Holdings diagnosis
    PUSHPLUS_TOKEN=xxx PYTHONPATH=./src python -m auto_select_stock.notify.runner --holdings

    # With custom horizon
    PUSHPLUS_TOKEN=xxx HORIZON=3d python -m auto_select_stock.notify.runner
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .config import PUSHPLUS_TOKEN, NOTIFY_CHECKPOINT, NOTIFY_STRATEGY, NOTIFY_TOP_K, NOTIFY_HORIZON
from .push_providers import PushPlusProvider
from .daily_report import generate_report, get_latest_price_date
from .holdings import generate_holdings_report


def main() -> None:
    parser = argparse.ArgumentParser(description="StockPilot daily notification service")
    parser.add_argument(
        "--holdings",
        action="store_true",
        help="发送持仓诊断报告（替代每日推荐）",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default=None,
        help=f"Prediction horizon for ranking (default: {NOTIFY_HORIZON})",
    )
    parser.add_argument(
        "--holdings-path",
        type=str,
        default=None,
        help="Path to holdings JSON file (default: data/holdings.json)",
    )
    args = parser.parse_args()

    horizon = args.horizon or NOTIFY_HORIZON

    if not PUSHPLUS_TOKEN:
        print("ERROR: PUSHPLUS_TOKEN environment variable is not set", file=sys.stderr)
        sys.exit(1)

    latest_date = get_latest_price_date()
    print(f"Latest price date in DB: {latest_date}")

    if args.holdings:
        # Holdings diagnosis report
        holdings_path = Path(args.holdings_path) if args.holdings_path else Path("data/holdings.json")
        if not holdings_path.exists():
            print(f"ERROR: Holdings file not found: {holdings_path}", file=sys.stderr)
            sys.exit(1)

        html, analyses = generate_holdings_report(
            checkpoint=NOTIFY_CHECKPOINT,
            holdings_path=holdings_path,
        )

        title = f"StockPilot 持仓诊断 {latest_date}"
        provider = PushPlusProvider(token=PUSHPLUS_TOKEN)
        provider.send(title=title, content=html)

        print(f"OK: pushed {len(analyses)} positions to PushPlus")
        for a in analyses:
            print(
                f"  {a.symbol}: 当前价={a.current_price:.2f} 浮盈亏={a.unrealized_pct*100:+.2f}% "
                f"建议={a.recommendation} 5d预测={a.pred_rets.get('5d', a.pred_rets.get('1d', 0))*100:+.2f}%"
            )
        return

    # Default: daily stock recommendations
    html, results = generate_report(
        checkpoint=NOTIFY_CHECKPOINT,
        strategy=NOTIFY_STRATEGY,
        top_k=NOTIFY_TOP_K,
        horizon=horizon,
    )

    title = f"StockPilot 每日推荐 {latest_date}"
    provider = PushPlusProvider(token=PUSHPLUS_TOKEN)
    provider.send(title=title, content=html)

    print(f"OK: pushed {len(results)} stocks to PushPlus (horizon={horizon})")
    for sym, pred_rets, weight in results:
        if isinstance(pred_rets, dict):
            # Multi-horizon display
            ret_str = ", ".join([f"{h}:{v*100:+.2f}%" for h, v in sorted(pred_rets.items())])
            print(f"  {sym}: [{ret_str}] (weight={weight:.4f})")
        else:
            print(f"  {sym}: {pred_rets*100:+.2f}% (weight={weight:.4f})")


if __name__ == "__main__":
    main()
