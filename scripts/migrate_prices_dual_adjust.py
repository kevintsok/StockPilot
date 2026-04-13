#!/usr/bin/env python3
"""
Migration script: re-fetch all existing price data with both qfq and hfq adjustments.

- qfq (前复权) → stored in 'price' table  (for current prices / P&L / holdings)
- hfq (后复权) → stored in 'price_hfq' table (for model training continuity)

Fund flow (--fund-flow) and chip (--chip) are fetched separately due to API rate limits.

Usage:
    python scripts/migrate_prices_dual_adjust.py [--limit N] [--start 1990-01-01]
    python scripts/migrate_prices_dual_adjust.py --fund-flow   # include fund_flow
    python scripts/migrate_prices_dual_adjust.py --chip        # include chip
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auto_select_stock.data.fetcher import (
    fetch_history,
    fetch_fund_flow,
    fetch_chip,
    fetch_and_store_fund_flow,
    list_all_symbols,
)
from auto_select_stock.data.storage import save_stock_history
from auto_select_stock.data.storage import save_fund_flow, save_chip
from auto_select_stock.core.types import to_structured_array
from auto_select_stock.config import DATA_DIR, REQUEST_SLEEP_SECONDS


def migrate_all(limit: int | None = None, start_date: str = "1990-01-01",
                base_dir: Path = DATA_DIR, include_fund_flow: bool = False,
                include_chip: bool = False, fund_flow_sleep: float = 3.0):
    symbols = list_all_symbols(base_dir=base_dir)
    if limit:
        symbols = symbols[:limit]
    print(f"Migrating {len(symbols)} symbols (qfq + hfq prices)...")
    print("  'price'     ← qfq (前复权) for viewing / P&L")
    print("  'price_hfq' ← hfq (后复权) for model training")
    if include_fund_flow:
        print(f"  'fund_flow' ← 主力/超大/大/中/小单净流入 (sleep={fund_flow_sleep}s between requests)")
    if include_chip:
        print("  'chip'      ← 获利比例/平均成本/90&70成本区间")
    print()

    success = 0
    failed = 0
    for i, sym in enumerate(symbols, 1):
        try:
            print(f"[{i}/{len(symbols)}] {sym}: fetching qfq + hfq...", end=" ", flush=True)
            # Force re-fetch prices from akshare (bypass skip logic)
            df_qfq = fetch_history(sym, start_date=start_date, adjust="qfq")
            df_hfq = fetch_history(sym, start_date=start_date, adjust="hfq")
            arr_qfq = to_structured_array(df_qfq)
            arr_hfq = to_structured_array(df_hfq)
            save_stock_history(sym, arr_qfq, base_dir=base_dir, table="price")
            save_stock_history(sym, arr_hfq, base_dir=base_dir, table="price_hfq")
            print(f"qfq={df_qfq.iloc[-1]['close']:.2f} hfq={df_hfq.iloc[-1]['close']:.2f}", flush=True)

            # Fund flow — rate-limited by eastmoney, needs sleep between requests
            if include_fund_flow:
                try:
                    df_ff = fetch_fund_flow(sym)
                    save_fund_flow(sym, df_ff, base_dir=base_dir)
                    print(f"  fund_flow: {len(df_ff)} rows", flush=True)
                except Exception as ex:
                    # Retry once with longer backoff on rate limit
                    err_str = str(ex)
                    if "RemoteDisconnected" in err_str or "Connection aborted" in err_str:
                        print(f"  fund_flow rate-limited, retry after {fund_flow_sleep}s...", flush=True)
                        time.sleep(fund_flow_sleep)
                        try:
                            df_ff = fetch_fund_flow(sym)
                            save_fund_flow(sym, df_ff, base_dir=base_dir)
                            print(f"  fund_flow: {len(df_ff)} rows (retry OK)", flush=True)
                        except Exception as ex2:
                            print(f"  fund_flow FAILED: {ex2}", flush=True)
                    else:
                        print(f"  fund_flow FAILED: {ex}", flush=True)
                # Sleep between fund_flow requests to avoid rate limit
                time.sleep(fund_flow_sleep)

            # Chip distribution — API often returns 404 (endpoint retired)
            if include_chip:
                try:
                    df_chip = fetch_chip(sym)
                    save_chip(sym, df_chip, base_dir=base_dir)
                    print(f"  chip: {len(df_chip)} rows", flush=True)
                except Exception as ex:
                    print(f"  chip SKIPPED: {ex}", flush=True)

            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

        if i % 50 == 0 or i == len(symbols):
            print(f"Progress: {i}/{len(symbols)} ({success} ok, {failed} failed)")

    print(f"\nDone: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate price tables to dual-adjustment (qfq + hfq)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols to process")
    parser.add_argument("--start", type=str, default="1990-01-01", help="Start date for fetching")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--fund-flow", action="store_true",
                        help="Also fetch fund_flow (主力资金流). NOTE: API is rate-limited; "
                             "use --fund-flow-sleep to avoid blocks.")
    parser.add_argument("--fund-flow-sleep", type=float, default=2.0,
                        help="Sleep seconds between fund_flow requests (default: 2.0)")
    parser.add_argument("--chip", action="store_true",
                        help="Also fetch chip distribution data (筹码分布, often 404)")
    args = parser.parse_args()

    base_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    migrate_all(
        limit=args.limit,
        start_date=args.start,
        base_dir=base_dir,
        include_fund_flow=args.fund_flow,
        include_chip=args.chip,
        fund_flow_sleep=args.fund_flow_sleep,
    )
