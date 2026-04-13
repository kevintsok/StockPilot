#!/usr/bin/env python3
"""
Resume migration: only fetch symbols NOT yet in price_hfq, plus fund_flow for all.
Replaces the existing price_hfq entries via ON CONFLICT upsert.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sqlite3
from auto_select_stock.data.fetcher import (
    fetch_history,
    fetch_fund_flow,
    fetch_and_store_fund_flow,
    list_all_symbols,
)
from auto_select_stock.data.storage import save_stock_history, save_fund_flow
from auto_select_stock.core.types import to_structured_array
from auto_select_stock.config import DATA_DIR

DB_PATH = DATA_DIR / "stock.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")  # Wait up to 30s for locks
    return conn

def get_done_symbols():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT symbol FROM price_hfq")
    done = {row[0] for row in cur.fetchall()}
    conn.close()
    return done

def get_ff_done_symbols():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT symbol FROM fund_flow")
    done = {row[0] for row in cur.fetchall()}
    conn.close()
    return done

def main():
    all_symbols = list_all_symbols(base_dir=DATA_DIR)
    done_price = get_done_symbols()
    done_ff = get_ff_done_symbols()

    todo_price = [s for s in all_symbols if s not in done_price]
    todo_ff = [s for s in all_symbols if s not in done_ff]

    print(f"Total symbols : {len(all_symbols)}")
    print(f"price_hfq done: {len(done_price)} ({len(todo_price)} remaining)")
    print(f"fund_flow done : {len(done_ff)} ({len(todo_ff)} remaining)")
    print()

    # Step 1: Fetch missing price_hfq entries
    if todo_price:
        print(f"=== Step 1: Fetching price_hfq for {len(todo_price)} symbols ===")
        success = 0
        failed = 0
        skipped_etf = 0
        for i, sym in enumerate(todo_price, 1):
            # Skip ETFs/funds that akshare doesn't support (689xxx, 920xxx ranges)
            if sym.startswith("689") or sym.startswith("920"):
                skipped_etf += 1
                continue
            try:
                print(f"[{i}/{len(todo_price)}] {sym}: fetching hfq...", end=" ", flush=True)
                df_hfq = fetch_history(sym, start_date="1990-01-01", adjust="hfq")
                arr_hfq = to_structured_array(df_hfq)
                save_stock_history(sym, arr_hfq, base_dir=DATA_DIR, table="price_hfq")
                print(f"done ({len(df_hfq)} rows)", flush=True)
                success += 1
            except Exception as e:
                print(f"FAILED: {e}", flush=True)
                failed += 1

            if i % 100 == 0 or i == len(todo_price):
                print(f"Progress: {i}/{len(todo_price)} ({success} ok, {failed} failed, {skipped_etf} etf_skipped)")
        print(f"Price fetch done: {success} ok, {failed} failed, {skipped_etf} etf_skipped\n")
    else:
        print("=== Step 1: price_hfq already complete ===\n")

    # Step 2: Fetch fund_flow for missing symbols
    if todo_ff:
        print(f"=== Step 2: Fetching fund_flow for {len(todo_ff)} symbols ===")
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

            # Exponential backoff on failures to avoid rate limit
            if failed > 5:
                wait_time = min(30.0, 3.0 * (2 ** (failed - 5)))
                time.sleep(wait_time)
            else:
                time.sleep(3.0)

            if i % 50 == 0 or i == len(todo_ff):
                print(f"Progress: {i}/{len(todo_ff)} ({success} ok, {failed} failed)")
        print(f"Fund_flow fetch done: {success} ok, {failed} failed\n")
    else:
        print("=== Step 2: fund_flow already complete ===\n")

    # Verify final state
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT symbol) FROM price_hfq")
    total_symbols = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM price_hfq")
    total_rows = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT symbol) FROM fund_flow")
    ff_symbols = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM fund_flow")
    ff_rows = cur.fetchone()[0]
    conn.close()

    print("=== Final DB State ===")
    print(f"price_hfq : {total_rows} rows, {total_symbols} symbols")
    print(f"fund_flow : {ff_rows} rows, {ff_symbols} symbols")
    print("\nMigration complete! Ready for training.")

if __name__ == "__main__":
    main()
