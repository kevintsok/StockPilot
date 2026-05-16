"""Fetch industry (行业) and concept (概念) sector daily data from akshare."""

import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import akshare as ak
import pandas as pd
import requests

# Inline config defaults to avoid importing from auto_select_stock package
DEFAULT_START_DATE = "20180101"
REQUEST_SLEEP_SECONDS = float(os.getenv("AUTO_SELECT_STOCK_REQUEST_SLEEP_SECONDS", "0.5"))

# Direct imports of storage functions to avoid circular import through __init__.py
import auto_select_stock.data.storage as storage_module

save_sector = storage_module.save_sector
save_sector_daily = storage_module.save_sector_daily
load_sector_daily = storage_module.load_sector_daily
list_all_sectors = storage_module.list_all_sectors
list_sectors = storage_module.list_sectors
sector_date_range = storage_module.sector_date_range


MAX_SECTOR_SECONDS = float(os.getenv("AUTO_SELECT_STOCK_FETCH_TIMEOUT", "120"))


def _run_with_timeout(func, timeout: float, *args, **kwargs):
    """Run callable with a wall-clock timeout using a worker thread."""
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
    """
    Fetch all industry sector names and codes from akshare.
    Returns list of (code, name) tuples.
    """
    df = ak.stock_board_industry_name_em()
    # code column may be "板块代码" or "代码", name may be "板块名称" or "名称"
    code_col = "板块代码" if "板块代码" in df.columns else "代码"
    name_col = "板块名称" if "板块名称" in df.columns else "名称"
    return [(str(row[code_col]), str(row[name_col])) for row in df.itertuples(index=False)]


def fetch_concept_list() -> List[Tuple[str, str]]:
    """
    Fetch all concept sector names and codes from akshare.
    Returns list of (code, name) tuples.
    """
    df = ak.stock_board_concept_name_em()
    # code column may be "板块代码" or "代码", name may be "板块名称" or "名称"
    code_col = "板块代码" if "板块代码" in df.columns else "代码"
    name_col = "板块名称" if "板块名称" in df.columns else "名称"
    return [(str(row[code_col]), str(row[name_col])) for row in df.itertuples(index=False)]


def fetch_sector_daily(
    sector_name: str,
    category: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a sector (industry or concept).

    category: "industry" or "concept"
    """
    last_err: Optional[Exception] = None
    start = start_date.replace("-", "") if start_date else None
    end = end_date.replace("-", "") if end_date else None

    for attempt in range(1, retries + 1):
        try:
            if category == "industry":
                raw = _run_with_timeout(
                    ak.stock_board_industry_hist_em,
                    MAX_SECTOR_SECONDS,
                    symbol=sector_name,
                    start_date=start,
                    end_date=end,
                )
            else:
                raw = _run_with_timeout(
                    ak.stock_board_concept_hist_em,
                    MAX_SECTOR_SECONDS,
                    symbol=sector_name,
                    start_date=start,
                    end_date=end,
                )
            break
        except TimeoutError as exc:
            last_err = exc
            print(f"timeout fetch sector {sector_name} ({category}) attempt {attempt}/{retries}", flush=True)
        except requests.RequestException as exc:
            last_err = exc
        except Exception as exc:
            last_err = exc

        time.sleep(backoff ** attempt)
    else:
        raise RuntimeError(f"Failed to fetch sector {sector_name} ({category}): {last_err}") from last_err

    if raw is None or raw.empty:
        raise RuntimeError(f"Empty sector data for {sector_name} ({category})")

    df = raw.copy()
    # Normalize column names
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_change",
        "换手率": "turnover_rate",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure required columns exist
    for col in ["date", "open", "high", "low", "close", "volume", "amount", "pct_change", "turnover_rate"]:
        if col not in df.columns:
            df[col] = 0.0

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    # Numeric cleanup
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.replace("%", ""),
                errors="coerce",
            ).fillna(0.0)

    # Use sector_name as the code for consistency with akshare naming
    return df[["date", "open", "high", "low", "close", "volume", "amount", "turnover_rate", "pct_change"]]


def fetch_and_save_sector(
    sector_name: str,
    category: str,
    start_date: str,
    base_dir: Optional[Path] = None,
) -> bool:
    """
    Fetch and save sector daily data, skipping if already up to date.
    Returns True if successful.
    """
    # Get existing date range
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
    base_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Fetch all industry and/or concept sector histories.

    Args:
        start_date: start date for fetching history
        category: "industry", "concept", or None for both
        base_dir: data directory
        limit: max sectors per category (for testing)

    Returns:
        List of sector names that were processed
    """
    processed = []
    categories = [category] if category else ["industry", "concept"]

    for cat in categories:
        if cat == "industry":
            try:
                sectors = fetch_industry_list()
                print(f"Fetched {len(sectors)} industry sectors", flush=True)
            except Exception as exc:
                print(f"Failed to fetch industry list: {exc}", flush=True)
                continue
        elif cat == "concept":
            try:
                sectors = fetch_concept_list()
                print(f"Fetched {len(sectors)} concept sectors", flush=True)
            except Exception as exc:
                print(f"Failed to fetch concept list: {exc}", flush=True)
                continue
        else:
            continue

        if limit:
            sectors = sectors[:limit]

        for idx, (code, name) in enumerate(sectors, 1):
            try:
                # Use name as the sector identifier (akshare uses name for fetching history)
                fetch_and_save_sector(name, cat, start_date, base_dir=base_dir)
                processed.append(name)
            except Exception as exc:
                print(f"Warning: failed to fetch sector {name}: {exc}", flush=True)

            if REQUEST_SLEEP_SECONDS > 0:
                time.sleep(REQUEST_SLEEP_SECONDS)

            if idx % 20 == 0 or idx == len(sectors):
                print(f"Progress: {idx}/{len(sectors)} {cat} sectors processed", flush=True)

    return processed


def update_sectors(base_dir: Optional[Path] = None) -> List[str]:
    """
    Incrementally update all sectors with latest data.
    Returns list of updated sector names.
    """
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
