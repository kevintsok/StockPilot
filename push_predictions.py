#!/usr/bin/env python3
"""
Direct prediction push using the best strategy from backtest.
Reads strategy name from models/top10_strategies.json (written by sort_by_sharpe.py).
"""
import sys
sys.path.insert(0, 'src')
import datetime
import json
import os
import subprocess
from auto_select_stock.notify.push_providers import PushPlusProvider
from auto_select_stock.notify.daily_report import generate_report, get_latest_price_date


def _is_trading_day(db_date_str: str) -> bool:
    """Check if the DB's latest date is today or yesterday (accounting for weekends)."""
    if not db_date_str:
        return False
    db_date = datetime.datetime.strptime(db_date_str, "%Y-%m-%d").date()
    today = datetime.date.today()
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

    result = subprocess.run(
        [sys.executable, "-m", "auto_select_stock.cli", "update-daily"],
        env={**os.environ, "PYTHONPATH": "./src"},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: update-daily failed: {result.stderr}", file=sys.stderr)
    else:
        new_date = get_latest_price_date()
        print(f"Update complete. Latest date now: {new_date}")

CHECKPOINT = "/mnt/d/Projects/auto-select-stock/models/price_transformer_2025-train20250331-val20260327.pt"
TOKEN = os.getenv("PUSHPLUS_TOKEN", "183cae5e7d8148f0b85754a2912fc81c")

# Read best strategy from top10 file (written by sort_by_sharpe.py)
TOP10_FILE = "models/top10_strategies.json"
try:
    with open(TOP10_FILE) as f:
        top10 = json.load(f)
    STRATEGY = top10[0]  # Best strategy (highest Sharpe)
    print(f"Using best strategy from top10: {STRATEGY}")
except Exception as e:
    print(f"Warning: could not read {TOP10_FILE}: {e}")
    print("Falling back to hardcoded strategy")
    STRATEGY = "Conf-5d-K15-SL5pct-TP15pct"

TOP_K = 10
HORIZON = "5d"

_ensure_fresh_data()
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
