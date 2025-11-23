from typing import List, Optional
import time
import os
import signal
import threading
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

from .config import DATA_DIR, DEFAULT_START_DATE, REQUEST_SLEEP_SECONDS
from .stock_types import to_structured_array
from .storage import load_stock_history, save_stock_history, price_date_range, list_symbols

_TRADE_DATE_CACHE: dict[str, pd.Timestamp] = {}
MAX_SYMBOL_SECONDS = float(os.getenv("AUTO_SELECT_STOCK_FETCH_TIMEOUT", "120"))
MAX_SYMBOL_ATTEMPTS = int(os.getenv("AUTO_SELECT_STOCK_FETCH_ATTEMPTS", "3"))


def _is_trading_day_sse(day: pd.Timestamp) -> Optional[bool]:
    """
    Fast check via上交所成交数据接口。返回 True/False，接口异常时返回 None。
    """
    try:
        df = ak.stock_sse_deal_daily(date=day.strftime("%Y%m%d"))
        return df is not None and not df.empty
    except Exception:
        return None


def _last_trading_date(reference: pd.Timestamp) -> pd.Timestamp:
    """
    Resolve the latest trading date on or before the given reference date.
    Prefer a fast SSE daily check; fall back to Sina trade calendar or weekday heuristics.
    """
    ref_date = reference.normalize()
    key = ref_date.strftime("%Y-%m-%d")
    cached = _TRADE_DATE_CACHE.get(key)
    if cached is not None:
        return cached

    # Quick path: use SSE daily endpoint for up to 10 days backward to handle holidays/weekends.
    for offset in range(0, 10):
        candidate = ref_date - pd.Timedelta(days=offset)
        is_trade = _is_trading_day_sse(candidate)
        if is_trade is True:
            _TRADE_DATE_CACHE[key] = candidate
            return candidate
        if is_trade is None:
            break  # fall back if API unavailable

    try:
        df = ak.tool_trade_date_hist_sina()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[df["trade_date"] <= ref_date]
        if not df.empty:
            last_date = df["trade_date"].max().normalize()
            _TRADE_DATE_CACHE[key] = last_date
            return last_date
    except Exception:
        pass  # fall back to weekday heuristic

    # Heuristic: weekend adjustment; holidays will fetch again next run.
    weekday = ref_date.weekday()
    if weekday == 5:  # Saturday
        ref_date = ref_date - pd.Timedelta(days=1)
    elif weekday == 6:  # Sunday
        ref_date = ref_date - pd.Timedelta(days=2)
    _TRADE_DATE_CACHE[key] = ref_date
    return ref_date


def _fallback_symbols(base_dir: Path) -> List[str]:
    try:
        return list_symbols(base_dir=base_dir)
    except Exception:
        return []


def list_all_symbols(base_dir: Optional[Path] = None, retries: int = 3, backoff: float = 1.5) -> List[str]:
    """
    Fetch all A-share stock codes with retries and fallbacks.
    """
    base_dir = base_dir or DATA_DIR
    last_err: Optional[Exception] = None
    # Prefer code_name endpoint (fast, stable)
    for attempt in range(1, retries + 1):
        try:
            df = ak.stock_info_a_code_name()
            symbols = df["code"].astype(str).tolist()
            print(f"Fetched symbol list from code_name: {len(symbols)}")
            return symbols
        except (requests.RequestException, Exception) as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(backoff ** attempt)

    # Fallback to snapshot endpoint
    for attempt in range(1, retries + 1):
        try:
            df = ak.stock_zh_a_spot_em()
            symbols = df["代码"].astype(str).tolist()
            print(f"Fetched symbol list from spot_em: {len(symbols)}")
            return symbols
        except (requests.RequestException, Exception) as exc:  # keep broad to cover akshare internal errors
            last_err = exc
            time.sleep(backoff ** attempt)

    # Last resort: use any existing downloaded symbols so update-daily still works.
    cached = _fallback_symbols(base_dir)
    if cached:
        print(f"Warning: failed to fetch symbol list, using cached symbols: {len(cached)}")
        return cached

    raise RuntimeError(f"Unable to fetch symbol list from network: {last_err}")


def _symbol_with_prefix(symbol: str) -> str:
    return f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"


HIST_RENAME_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_change",
    "涨跌额": "change_amount",
    "振幅": "amplitude",
    "换手率": "turnover_rate",
    "量比": "volume_ratio",
}


def _finalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and are numeric, adding derived fields when missing."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    # Core numeric columns we want to persist.
    numeric_fields = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover_rate",
        "volume_ratio",
        "pct_change",
        "amplitude",
        "change_amount",
    ]

    # Derive missing columns.
    if "volume_ratio" not in df:
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=5, min_periods=1).mean()
    if "pct_change" not in df:
        df["pct_change"] = df["close"].pct_change().fillna(0.0) * 100
    if "change_amount" not in df:
        df["change_amount"] = df["close"].diff().fillna(0.0)
    if "amplitude" not in df:
        prev_close = df["close"].shift(1)
        amplitude = (df["high"] - df["low"]) / prev_close.replace(0, pd.NA)
        df["amplitude"] = amplitude.fillna(0.0) * 100

    for col in numeric_fields:
        if col not in df:
            df[col] = 0.0
        cleaned = (
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(cleaned, errors="coerce").fillna(0.0)

    ordered_cols = ["date"] + numeric_fields
    return df[ordered_cols]


def _run_with_timeout(func, timeout: float, *args, **kwargs):
    """
    Run callable with a wall-clock timeout using a worker thread to avoid process-wide signal issues.
    """
    result = {}
    exc: list[Exception] = []

    def worker():
        try:
            result["value"] = func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            exc.append(e)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError("fetch timeout")
    if exc:
        raise exc[0]
    return result.get("value")


def _fetch_history_fallback(symbol: str, start_date: str) -> pd.DataFrame:
    prefixed = _symbol_with_prefix(symbol)
    df = ak.stock_zh_a_daily(symbol=prefixed)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.to_datetime(start_date)]
    df = df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
            "turnover": "turnover_rate",
        }
    )
    return _finalize_history_df(df)


def fetch_history(
    symbol: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
    retries: int = 3,
    backoff: float = 1.5,
    adjust: str = "hfq",
) -> pd.DataFrame:
    """
    Download daily historical data for a single symbol using ak.stock_zh_a_daily
    and add derived columns.
    """
    last_err: Optional[Exception] = None
    start = start_date.replace("-", "") if start_date else None
    end = end_date.replace("-", "") if end_date else None
    raw: Optional[pd.DataFrame] = None
    prefixed = _symbol_with_prefix(symbol)

    for attempt in range(1, retries + 1):
        try:
            raw = _run_with_timeout(
                ak.stock_zh_a_daily,
                MAX_SYMBOL_SECONDS,
                symbol=prefixed,
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            break
        except TimeoutError as exc:
            last_err = exc
            print(f"timeout fetch {symbol} attempt {attempt}/{retries}", flush=True)
        except (requests.RequestException, Exception) as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(backoff ** attempt)

    if raw is None or raw.empty:
        raise RuntimeError(f"Failed to fetch history for {symbol}: {last_err}") from last_err

    # Normalize fields to expected schema.
    df = raw.copy()
    if "turnover" in df:
        df["turnover_rate"] = pd.to_numeric(df["turnover"], errors="coerce") * 100
        df = df.drop(columns=["turnover"])
    if "outstanding_share" in df:
        df = df.drop(columns=["outstanding_share"])
    if "date" not in df:
        raise RuntimeError(f"Failed to parse date column for {symbol}")
    return _finalize_history_df(df)


def fetch_and_store(symbol: str, start_date: str = DEFAULT_START_DATE, base_dir=None) -> None:
    range_info = price_date_range(symbol, base_dir=base_dir)
    desired_start = pd.to_datetime(start_date).date()
    today = pd.Timestamp.today().normalize().date()
    last_trade = _last_trading_date(pd.Timestamp.today()).date()

    if range_info is None:
        df = fetch_history(symbol, start_date=start_date)
        arr = to_structured_array(df)
        save_stock_history(symbol, arr, base_dir=base_dir)
        return

    min_exist, max_exist = range_info
    min_exist = min_exist.date()
    max_exist = max_exist.date()

    # If we already have data up to the last trading day (or later), skip network calls.
    if max_exist >= last_trade:
        print(f"skip {symbol}: already up to {max_exist} (last trading {last_trade})", flush=True)
        return

    frames = []
    gap_days = (min_exist - desired_start).days if min_exist > desired_start else 0
    if desired_start < min_exist and gap_days > 7:
        for attempt in range(1, MAX_SYMBOL_ATTEMPTS + 1):
            try:
                older_df = _run_with_timeout(fetch_history, MAX_SYMBOL_SECONDS, symbol, start_date=start_date)
                older_df["date"] = pd.to_datetime(older_df["date"])
                frames.append(older_df[older_df["date"].dt.date < min_exist])
                break
            except TimeoutError as exc:
                last_err = exc
                print(f"timeout history {symbol} backfill attempt {attempt}/{MAX_SYMBOL_ATTEMPTS}", flush=True)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            time.sleep(1.0 * attempt)

    next_day = max_exist + pd.Timedelta(days=1)
    if next_day > last_trade:
        print(f"skip {symbol}: max_date={max_exist}, no newer trading days (last trading {last_trade})", flush=True)
        return
    fresh_ok = False
    for attempt in range(1, MAX_SYMBOL_ATTEMPTS + 1):
        try:
            newer_df = _run_with_timeout(fetch_history, MAX_SYMBOL_SECONDS, symbol, start_date=next_day.strftime("%Y-%m-%d"))
            newer_df["date"] = pd.to_datetime(newer_df["date"])
            frames.append(newer_df[newer_df["date"].dt.date > max_exist])
            fresh_ok = True
            break
        except TimeoutError as exc:
            last_err = exc
            print(f"timeout history {symbol} new data attempt {attempt}/{MAX_SYMBOL_ATTEMPTS}", flush=True)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1.0 * attempt)

    if not fresh_ok and not frames:
        print(f"skip {symbol}: history fetch failed after retries ({last_err})", flush=True)
        return

    if frames:
        df_exist = pd.DataFrame(load_stock_history(symbol, base_dir=base_dir))
        df_exist["date"] = pd.to_datetime(df_exist["date"])
        frames.append(df_exist)
        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values("date", inplace=True)
        combined = combined.drop_duplicates(subset="date", keep="last")
        arr = to_structured_array(combined)
        save_stock_history(symbol, arr, base_dir=base_dir)


def fetch_all(start_date: str = DEFAULT_START_DATE, limit: Optional[int] = None, base_dir=None) -> List[str]:
    base_dir = base_dir or DATA_DIR
    universe = list_all_symbols(base_dir=base_dir)
    # Build a plan that skips symbols already up-to-date and narrows fetch start to only missing ranges.
    desired_start = pd.to_datetime(start_date).date()
    today = pd.Timestamp.today().normalize().date()
    last_trade = _last_trading_date(pd.Timestamp.today()).date()
    plan: List[tuple[str, str]] = []
    for code in universe:
        range_info = price_date_range(code, base_dir=base_dir)
        if range_info is None:
            plan.append((code, start_date))
            continue
        min_dt, max_dt = range_info
        min_dt = min_dt.date()
        max_dt = max_dt.date()
        # Treat slight gaps (weekends/holidays) as covered to avoid repeated backfill attempts.
        gap_days = (min_dt - desired_start).days if min_dt > desired_start else 0
        covered_from_start = min_dt <= desired_start or gap_days <= 7

        # If we already have data up to (or beyond) last trading day and historical coverage is sufficient, skip.
        if max_dt >= last_trade and covered_from_start:
            continue
        # If historical data starts after desired_start, fetch from desired_start to backfill.
        if min_dt > desired_start and gap_days > 7:
            plan.append((code, start_date))
            continue
        # Otherwise only fetch from the day after the latest existing record.
        start_missing = (pd.to_datetime(max_dt) + pd.Timedelta(days=1)).date()
        if start_missing > last_trade:
            continue  # nothing new to fetch
        plan.append((code, start_missing.strftime("%Y-%m-%d")))

    if limit:
        plan = plan[:limit]

    print(f"Start fetching {len(plan)} symbols (planned from universe={len(universe)}, start={start_date})", flush=True)
    for idx, (code, start_at) in enumerate(plan, 1):
        try:
            print(f"[{idx}/{len(plan)}] fetching {code} (start {start_at})", flush=True)
            fetch_and_store(code, start_date=start_at, base_dir=base_dir)
            if idx % 100 == 0 or idx == len(plan):
                print(f"Progress: {idx}/{len(plan)} fetched", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to fetch {code}: {exc}", flush=True)
        if REQUEST_SLEEP_SECONDS > 0:
            time.sleep(REQUEST_SLEEP_SECONDS)
    return [code for code, _ in plan]


def append_latest(symbol: str, base_dir=None) -> None:
    """
    Fetch latest data and append new days to existing file.
    """
    try:
        existing = load_stock_history(symbol, base_dir=base_dir)
        last_date = pd.to_datetime(existing["date"][-1]).date()
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    except FileNotFoundError:
        fetch_and_store(symbol, start_date=DEFAULT_START_DATE, base_dir=base_dir)
        return
    df = fetch_history(symbol, start_date=start_date)
    if df.empty:
        return
    new_arr = to_structured_array(df)
    combined = pd.concat([pd.DataFrame(existing), pd.DataFrame(new_arr)], ignore_index=True)
    combined_arr = to_structured_array(combined)
    save_stock_history(symbol, combined_arr, base_dir=base_dir)
