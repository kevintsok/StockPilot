#!/usr/bin/env python
"""
Standalone sector data fetcher - runs without requiring torch or the full package __init__.
Usage: python scripts/fetch_sectors.py --start 19970101 --category both
"""
import argparse
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import akshare as ak
import pandas as pd
import requests


# ─── Inline minimal config ────────────────────────────────────────────────────────
DEFAULT_START_DATE = "20180101"
REQUEST_SLEEP_SECONDS = float(os.getenv("AUTO_SELECT_STOCK_REQUEST_SLEEP_SECONDS", "0.5"))
MAX_SECTOR_SECONDS = float(os.getenv("AUTO_SELECT_STOCK_FETCH_TIMEOUT", "120"))

# Data directory (same default as auto_select_stock.config)
DATA_DIR = Path(os.getenv("AUTO_SELECT_STOCK_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
DB_PATH = DATA_DIR / "stock.db"


# ─── Inline storage (sqlite operations for sector tables) ─────────────────────────

def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _connect(read_only=False):
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=DELETE;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    _init_db(conn)
    return conn


def _init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sector (
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            update_date TEXT,
            PRIMARY KEY(code, category)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sector_category ON sector(category)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sector_daily (
            sector_code TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL, amount REAL, turnover_rate REAL, pct_change REAL,
            PRIMARY KEY(sector_code, date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sector_daily_code_date ON sector_daily(sector_code,date)")
    conn.commit()


def save_sector(code: str, name: str, category: str, base_dir=None) -> Path:
    conn = _connect(base_dir)
    with conn:
        conn.execute(
            "INSERT INTO sector(code, name, category, update_date) VALUES(?, ?, ?, date('now')) "
            "ON CONFLICT(code, category) DO UPDATE SET name=excluded.name, update_date=date('now')",
            (code, name, category),
        )
    return DB_PATH


SECTOR_DAILY_SQL = """
    INSERT INTO sector_daily(sector_code,date,open,high,low,close,volume,amount,turnover_rate,pct_change)
    VALUES(:sector_code,:date,:open,:high,:low,:close,:volume,:amount,:turnover_rate,:pct_change)
    ON CONFLICT(sector_code,date) DO UPDATE SET
        open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
        volume=excluded.volume, amount=excluded.amount,
        turnover_rate=excluded.turnover_rate, pct_change=excluded.pct_change
"""


def save_sector_daily(sector_code: str, df: pd.DataFrame, base_dir=None) -> Path:
    if df.empty:
        return DB_PATH
    conn = _connect(base_dir)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    rows = [{**r, "sector_code": sector_code} for r in df.to_dict("records")]
    with conn:
        conn.executemany(SECTOR_DAILY_SQL, rows)
    return DB_PATH


def load_sector_daily(sector_code: str, base_dir=None) -> pd.DataFrame:
    conn = _connect(read_only=True)
    df = pd.read_sql_query(
        "SELECT date,open,high,low,close,volume,amount,turnover_rate,pct_change FROM sector_daily "
        "WHERE sector_code=? ORDER BY date ASC",
        conn, params=(sector_code,),
    )
    if df.empty:
        raise FileNotFoundError(f"No sector daily data for {sector_code}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def list_all_sectors(base_dir=None) -> list:
    conn = _connect(read_only=True)
    cur = conn.execute("SELECT code, name, category FROM sector ORDER BY category, name")
    return [(r[0], r[1], r[2]) for r in cur.fetchall()]


def sector_date_range(sector_code: str, base_dir=None) -> Optional[Tuple]:
    conn = _connect(read_only=True)
    cur = conn.execute("SELECT MIN(date), MAX(date) FROM sector_daily WHERE sector_code=?", (sector_code,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return pd.to_datetime(row[0]), pd.to_datetime(row[1])


# ─── Sector fetching logic ─────────────────────────────────────────────────────────

def _run_with_timeout(func, timeout: float, *args, **kwargs):
    result = {}
    exc: list[Exception] = []

    def worker():
        try:
            result["value"] = func(*args, **kwargs)
        except Exception as e:
            exc.append(e)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError("fetch timeout")
    if exc:
        raise exc[0]
    return result.get("value")


def fetch_industry_list() -> List[Tuple[str, str]]:
    """Fetch industry sector list from THS (同花顺)."""
    for attempt in range(1, 10):
        try:
            df = ak.stock_board_industry_name_ths()
            return [(str(row.code), str(row.name)) for row in df.itertuples(index=False)]
        except (requests.RequestException, ConnectionError) as exc:
            wait = min(30 * (2 ** attempt), 300)
            print(f"[{attempt}/9] THS industry list failed: {exc}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError("Failed to fetch industry list from THS")


def fetch_concept_list() -> List[Tuple[str, str]]:
    """Fetch concept sector list from THS (同花顺)."""
    for attempt in range(1, 10):
        try:
            df = ak.stock_board_concept_name_ths()
            return [(str(row.code), str(row.name)) for row in df.itertuples(index=False)]
        except (requests.RequestException, ConnectionError) as exc:
            wait = min(30 * (2 ** attempt), 300)
            print(f"[{attempt}/9] THS concept list failed: {exc}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError("Failed to fetch concept list from THS")


def fetch_sector_daily(
    sector_name: str,
    category: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
    retries: int = 5,
    backoff: float = 1.5,
) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    start = start_date.replace("-", "") if start_date else None
    end = end_date.replace("-", "") if end_date else None

    for attempt in range(1, retries + 1):
        try:
            if category == "industry":
                raw = _run_with_timeout(
                    ak.stock_board_industry_index_ths,
                    MAX_SECTOR_SECONDS,
                    symbol=sector_name,
                    start_date=start,
                    end_date=end,
                )
            else:
                raw = _run_with_timeout(
                    ak.stock_board_concept_index_ths,
                    MAX_SECTOR_SECONDS,
                    symbol=sector_name,
                    start_date=start,
                    end_date=end,
                )
            break
        except TimeoutError as exc:
            last_err = exc
            print(f"timeout fetch sector {sector_name} ({category}) attempt {attempt}/{retries}", flush=True)
        except (requests.RequestException, ConnectionError) as exc:
            last_err = exc
            wait = min(backoff ** attempt, 120)
            print(f"Connection error fetching {sector_name}: {exc}. Retry in {wait}s...", flush=True)
            time.sleep(wait)
        except Exception as exc:
            last_err = exc
            wait = min(backoff ** attempt, 60)
            print(f"Error fetching {sector_name}: {exc}. Retry in {wait}s...", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"Failed to fetch sector {sector_name} ({category}): {last_err}") from last_err

    if raw is None or raw.empty:
        raise RuntimeError(f"Empty sector data for {sector_name} ({category})")

    df = raw.copy()
    # THS column mapping: 日期->date, 开盘价->open, 最高价->high, 最低价->low, 收盘价->close, 成交量->volume, 成交额->amount
    rename_map = {
        "日期": "date",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "收盘价": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for col in ["date", "open", "high", "low", "close", "volume", "amount"]:
        if col not in df.columns:
            df[col] = 0.0
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.replace("%", ""),
                errors="coerce",
            ).fillna(0.0)
    # Compute pct_change and turnover_rate from OHLCV if not present
    if "pct_change" not in df.columns:
        df["pct_change"] = df["close"].pct_change().fillna(0) * 100
    if "turnover_rate" not in df.columns:
        df["turnover_rate"] = 0.0
    return df[["date", "open", "high", "low", "close", "volume", "amount", "turnover_rate", "pct_change"]]


def fetch_and_save_sector(sector_name: str, category: str, start_date: str, base_dir=None) -> bool:
    range_info = sector_date_range(sector_name, base_dir=base_dir)
    last_trade = pd.Timestamp.today().normalize()
    if range_info is not None:
        max_exist = range_info[1].date()
        if max_exist >= last_trade.date():
            print(f"skip sector {sector_name} ({category}): already up to {max_exist}", flush=True)
            return True
    try:
        df = fetch_sector_daily(sector_name, category, start_date=start_date)
        if df.empty:
            return False
        save_sector(sector_name, sector_name, category, base_dir=base_dir)
        save_sector_daily(sector_name, df, base_dir=base_dir)
        print(f"sector {sector_name} ({category}): saved {len(df)} rows", flush=True)
        return True
    except Exception as exc:
        print(f"sector {sector_name} ({category}): fetch failed — {exc}", flush=True)
        return False


def fetch_all_sectors(
    start_date: str = DEFAULT_START_DATE,
    category: Optional[str] = None,
    base_dir=None,
    limit: Optional[int] = None,
) -> List[str]:
    processed = []
    categories = ["industry", "concept"] if category == "both" else ([category] if category else ["industry", "concept"])

    for cat in categories:
        sectors = None
        for attempt in range(1, 20):
            try:
                if cat == "industry":
                    sectors = fetch_industry_list()
                else:
                    sectors = fetch_concept_list()
                print(f"Fetched {len(sectors)} {cat} sectors", flush=True)
                break
            except Exception as exc:
                wait = min(60 * (2 ** min(attempt, 5)), 600)
                print(f"[{attempt}/20] Failed to fetch {cat} list: {exc}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
        if sectors is None:
            print(f"Giving up on {cat} sector list after 20 retries.", flush=True)
            continue

        if limit:
            sectors = sectors[:limit]

        for idx, (code, name) in enumerate(sectors, 1):
            for attempt in range(1, 5):
                try:
                    fetch_and_save_sector(name, cat, start_date, base_dir=base_dir)
                    processed.append(name)
                    break
                except Exception as exc:
                    wait = min(10 * (2 ** attempt), 60)
                    if attempt < 4:
                        print(f"  Retry {attempt} for {name}: {exc}. Wait {wait}s...", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"  Failed to fetch sector {name} after 3 retries: {exc}", flush=True)
            if REQUEST_SLEEP_SECONDS > 0:
                time.sleep(REQUEST_SLEEP_SECONDS)
            if idx % 20 == 0 or idx == len(sectors):
                print(f"Progress: {idx}/{len(sectors)} {cat} sectors processed", flush=True)

    return processed


def update_sectors(base_dir=None) -> List[str]:
    updated = []
    all_sectors = list_all_sectors(base_dir=base_dir)
    for code, name, category in all_sectors:
        try:
            if fetch_and_save_sector(name, category, start_date="20180101", base_dir=base_dir):
                updated.append(name)
        except Exception as exc:
            print(f"Warning: failed to update sector {name}: {exc}", flush=True)
        if REQUEST_SLEEP_SECONDS > 0:
            time.sleep(REQUEST_SLEEP_SECONDS)
    return updated


# ─── CLI ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch sector (板块) daily data from akshare")
    parser.add_argument("--start", default="20180101", help="Start date (YYYYMMDD), default: 20180101")
    parser.add_argument("--category", choices=["industry", "concept", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Max sectors per category (for testing)")
    parser.add_argument("--update", action="store_true", help="Incrementally update existing sectors")
    args = parser.parse_args()

    if args.update:
        while True:
            updated = update_sectors(base_dir=DATA_DIR)
            if updated:
                print(f"Updated {len(updated)} sectors.", flush=True)
            else:
                print("No sectors updated (all up to date or all failed). Waiting 5min before retry...", flush=True)
                time.sleep(300)
    else:
        while True:
            processed = fetch_all_sectors(
                start_date=args.start,
                category=args.category,
                base_dir=DATA_DIR,
                limit=args.limit,
            )
            if processed:
                print(f"Processed {len(processed)} sectors.", flush=True)
                break
            else:
                print("No sectors processed. Retrying in 2min...", flush=True)
                time.sleep(120)


if __name__ == "__main__":
    main()
