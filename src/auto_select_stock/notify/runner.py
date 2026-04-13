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
import datetime
import sys
import os
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .config import PUSHPLUS_TOKEN, NOTIFY_CHECKPOINT, NOTIFY_STRATEGY, NOTIFY_TOP_K, NOTIFY_HORIZON
from .push_providers import PushPlusProvider
from .daily_report import generate_report, get_latest_price_date
from .holdings import generate_holdings_report


def _is_trading_day(db_date_str: str) -> bool:
    """Check if the DB's latest date is today or yesterday (accounting for weekends)."""
    if not db_date_str:
        return False
    db_date = datetime.datetime.strptime(db_date_str, "%Y-%m-%d").date()
    today = datetime.date.today()
    # Allow 1 day gap for weekends (Fri->Mon gap is 3 days, but we only allow 1)
    diff = (today - db_date).days
    return diff <= 1


def _ensure_fresh_data() -> None:
    """Ensure the price database is up-to-date before pushing."""
    latest_date = get_latest_price_date()
    if _is_trading_day(latest_date):
        print(f"Database is fresh (latest: {latest_date}), skipping update.")
        return

    print(f"Database is stale (latest: {latest_date}, today: {datetime.date.today()}).")
    print("Running update-daily to fetch latest data...")

    # Run update-daily via CLI
    result = subprocess.run(
        [
            sys.executable, "-m", "auto_select_stock.cli",
            "update-daily",
        ],
        env={**os.environ, "PYTHONPATH": "./src"},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: update-daily failed: {result.stderr}", file=sys.stderr)
    else:
        new_date = get_latest_price_date()
        print(f"Update complete. Latest date now: {new_date}")


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
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="运行虚拟盘模拟并推送 Top-3 策略",
    )
    args = parser.parse_args()

    horizon = args.horizon or NOTIFY_HORIZON

    if not PUSHPLUS_TOKEN:
        print("ERROR: PUSHPLUS_TOKEN environment variable is not set", file=sys.stderr)
        sys.exit(1)

    # Check and update data if stale
    _ensure_fresh_data()
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

    if args.portfolio:
        from .virtual_portfolio import VirtualPortfolio
        from .portfolio_report import generate_portfolio_report

        state_path = Path("data/virtual_portfolio_state.json")
        portfolio = VirtualPortfolio.from_json(state_path) if state_path.exists() else VirtualPortfolio()
        portfolio.update(latest_date, NOTIFY_CHECKPOINT)
        portfolio.to_json(state_path)

        top3 = portfolio.get_top_n(3)
        print(f"\n=== Virtual Portfolio ({latest_date}) ===")
        print(portfolio.summary())

        html = generate_portfolio_report(portfolio, top_n=3, checkpoint=NOTIFY_CHECKPOINT)
        provider = PushPlusProvider(token=PUSHPLUS_TOKEN)
        provider.send(title=f"StockPilot 虚拟盘 Top3 {latest_date}", content=html)
        print(f"OK: pushed top-3 virtual portfolio to PushPlus")
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
