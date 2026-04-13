#!/usr/bin/env python3
"""Resume migration: ONLY fund_flow (price_hfq is mostly complete from backup)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sqlite3
from auto_select_stock.data.fetcher import fetch_fund_flow, list_all_symbols
from auto_select_stock.data.storage import save_fund_flow
from auto_select_stock.config import DATA_DIR

DB_PATH = DATA_DIR / "stock.db"

def get_ff_done_symbols():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT symbol FROM fund_flow")
    done = {row[0] for row in cur.fetchall()}
    conn.close()
    return done

def main():
    all_symbols = list_all_symbols(base_dir=DATA_DIR)
    done_ff = get_ff_done_symbols()
    todo_ff = [s for s in all_symbols if s not in done_ff]

    print(f"Total symbols: {len(all_symbols)}")
    print(f"fund_flow done: {len(done_ff)} ({len(todo_ff)} remaining)")
    print(f"Estimated time: {len(todo_ff) * 3 / 3600:.1f} hours (at 3s/symbol)")
    print()

    if not todo_ff:
        print("Fund_flow already complete!")
        return

    success = 0
    failed = 0
    for i, sym in enumerate(todo_ff, 1):
        try:
            print(f"[{i}/{len(todo_ff)}] {sym}: fetching fund_flow...", end=" ", flush=True)
            df_ff = fetch_fund_flow(sym)
            save_fund_flow(sym, df_ff, base_dir=DATA_DIR)
            print(f"done ({len(df_ff)} rows)", flush=True)
            success += 1
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            failed += 1

        time.sleep(3.0)

        if i % 50 == 0 or i == len(todo_ff):
            print(f"Progress: {i}/{len(todo_ff)} ({success} ok, {failed} failed)")

    print(f"\nFund_flow fetch done: {success} ok, {failed} failed")

    # Verify
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT symbol) FROM fund_flow")
    print(f"Total fund_flow symbols in DB: {cur.fetchone()[0]}")
    conn.close()

if __name__ == "__main__":
    main()
