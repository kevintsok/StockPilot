from typing import List, Optional
import time
import os
import threading
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

from ..config import DATA_DIR, DEFAULT_START_DATE, REQUEST_SLEEP_SECONDS
from ..core.types import to_structured_array
from .storage import load_stock_history, save_stock_history, price_date_range, list_symbols
from .storage import save_fund_flow, save_chip, fund_flow_date_range, chip_date_range

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
        if "volume" not in df:
            df["volume"] = 0.0
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
    adjust: str = "qfq",
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


def _fetch_and_store_single(symbol: str, start_date: str, base_dir=None, adjust: str = "qfq", table: str = "price") -> None:
    """Fetch historical data with given adjust and save to the specified table."""
    try:
        df = fetch_history(symbol, start_date=start_date, adjust=adjust)
        arr = to_structured_array(df)
        save_stock_history(symbol, arr, base_dir=base_dir, table=table)
    except Exception as exc:
        print(f"  {symbol} ({table}): fetch failed — {exc}", flush=True)


def fetch_and_store(symbol: str, start_date: str = DEFAULT_START_DATE, base_dir=None) -> None:
    """Fetch and store both qfq (前复权) and hfq (后复权) price data.

    qfq data goes to 'price' table (for current prices / P&L).
    hfq data goes to 'price_hfq' table (for model training continuity).
    """
    range_info = price_date_range(symbol, base_dir=base_dir, table="price")
    desired_start = pd.to_datetime(start_date).date()
    today = pd.Timestamp.today().normalize().date()
    last_trade = _last_trading_date(pd.Timestamp.today()).date()

    if range_info is None:
        # Full fetch: fetch both adjustments
        _fetch_and_store_single(symbol, start_date, base_dir, adjust="qfq", table="price")
        _fetch_and_store_single(symbol, start_date, base_dir, adjust="hfq", table="price_hfq")
        return

    min_exist, max_exist = range_info
    min_exist = min_exist.date()
    max_exist = max_exist.date()

    # If we already have data up to the last trading day (or later), skip network calls.
    if max_exist >= last_trade:
        print(f"skip {symbol}: already up to {max_exist} (last trading {last_trade})", flush=True)
        return

    frames_qfq: list[pd.DataFrame] = []
    frames_hfq: list[pd.DataFrame] = []

    gap_days = (min_exist - desired_start).days if min_exist > desired_start else 0
    if desired_start < min_exist and gap_days > 7:
        for attempt in range(1, MAX_SYMBOL_ATTEMPTS + 1):
            try:
                older_qfq = _run_with_timeout(fetch_history, MAX_SYMBOL_SECONDS, symbol, start_date=start_date, adjust="qfq")
                older_hfq = _run_with_timeout(fetch_history, MAX_SYMBOL_SECONDS, symbol, start_date=start_date, adjust="hfq")
                older_qfq["date"] = pd.to_datetime(older_qfq["date"])
                older_hfq["date"] = pd.to_datetime(older_hfq["date"])
                frames_qfq.append(older_qfq[older_qfq["date"].dt.date < min_exist])
                frames_hfq.append(older_hfq[older_hfq["date"].dt.date < min_exist])
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
            newer_qfq = _run_with_timeout(fetch_history, MAX_SYMBOL_SECONDS, symbol, start_date=next_day.strftime("%Y-%m-%d"), adjust="qfq")
            newer_hfq = _run_with_timeout(fetch_history, MAX_SYMBOL_SECONDS, symbol, start_date=next_day.strftime("%Y-%m-%d"), adjust="hfq")
            newer_qfq["date"] = pd.to_datetime(newer_qfq["date"])
            newer_hfq["date"] = pd.to_datetime(newer_hfq["date"])
            frames_qfq.append(newer_qfq[newer_qfq["date"].dt.date > max_exist])
            frames_hfq.append(newer_hfq[newer_hfq["date"].dt.date > max_exist])
            fresh_ok = True
            break
        except TimeoutError as exc:
            last_err = exc
            print(f"timeout history {symbol} new data attempt {attempt}/{MAX_SYMBOL_ATTEMPTS}", flush=True)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1.0 * attempt)

    if not fresh_ok and not frames_qfq and not frames_hfq:
        print(f"skip {symbol}: history fetch failed after retries ({last_err})", flush=True)
        return

    if frames_qfq or frames_hfq:
        # Load existing qfq data and merge
        try:
            df_exist_qfq = pd.DataFrame(load_stock_history(symbol, base_dir=base_dir, table="price"))
            df_exist_qfq["date"] = pd.to_datetime(df_exist_qfq["date"])
            frames_qfq.append(df_exist_qfq)
        except FileNotFoundError:
            pass
        try:
            df_exist_hfq = pd.DataFrame(load_stock_history(symbol, base_dir=base_dir, table="price_hfq"))
            df_exist_hfq["date"] = pd.to_datetime(df_exist_hfq["date"])
            frames_hfq.append(df_exist_hfq)
        except FileNotFoundError:
            pass

        if frames_qfq:
            combined_qfq = pd.concat(frames_qfq, ignore_index=True)
            combined_qfq.sort_values("date", inplace=True)
            combined_qfq = combined_qfq.drop_duplicates(subset="date", keep="last")
            arr_qfq = to_structured_array(combined_qfq)
            save_stock_history(symbol, arr_qfq, base_dir=base_dir, table="price")

        if frames_hfq:
            combined_hfq = pd.concat(frames_hfq, ignore_index=True)
            combined_hfq.sort_values("date", inplace=True)
            combined_hfq = combined_hfq.drop_duplicates(subset="date", keep="last")
            arr_hfq = to_structured_array(combined_hfq)
            save_stock_history(symbol, arr_hfq, base_dir=base_dir, table="price_hfq")


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
    Fetch latest qfq and hfq data and append new days to existing price tables.
    """
    # Use qfq table as reference for existing data range
    try:
        existing_qfq = load_stock_history(symbol, base_dir=base_dir, table="price")
        last_date = pd.to_datetime(existing_qfq["date"][-1]).date()
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    except FileNotFoundError:
        fetch_and_store(symbol, start_date=DEFAULT_START_DATE, base_dir=base_dir)
        return

    next_day = pd.to_datetime(start_date).date()
    last_trade = _last_trading_date(pd.Timestamp.today()).date()
    if next_day > last_trade:
        print(f"skip {symbol}: max_date={last_date}, no newer trading days (last trading {last_trade})", flush=True)
        return

    try:
        df_qfq = fetch_history(symbol, start_date=start_date, adjust="qfq")
        df_hfq = fetch_history(symbol, start_date=start_date, adjust="hfq")
        if df_qfq.empty or df_hfq.empty:
            return
        new_arr_qfq = to_structured_array(df_qfq)
        new_arr_hfq = to_structured_array(df_hfq)

        # Merge with existing qfq
        combined_qfq = pd.concat([pd.DataFrame(existing_qfq), pd.DataFrame(new_arr_qfq)], ignore_index=True)
        combined_qfq.sort_values("date", inplace=True)
        combined_qfq = combined_qfq.drop_duplicates(subset="date", keep="last")
        arr_qfq = to_structured_array(combined_qfq)
        save_stock_history(symbol, arr_qfq, base_dir=base_dir, table="price")

        # Merge with existing hfq
        try:
            existing_hfq = load_stock_history(symbol, base_dir=base_dir, table="price_hfq")
            combined_hfq = pd.concat([pd.DataFrame(existing_hfq), pd.DataFrame(new_arr_hfq)], ignore_index=True)
        except FileNotFoundError:
            combined_hfq = new_arr_hfq
        combined_hfq.sort_values("date", inplace=True)
        combined_hfq = combined_hfq.drop_duplicates(subset="date", keep="last")
        arr_hfq = to_structured_array(combined_hfq)
        save_stock_history(symbol, arr_hfq, base_dir=base_dir, table="price_hfq")
    except Exception as exc:
        print(f"append_latest {symbol}: fetch failed — {exc}", flush=True)
        fetch_and_store(symbol, start_date=DEFAULT_START_DATE, base_dir=base_dir)


# ─── Fund flow (主力资金流) ─────────────────────────────────────────────────────

FUND_FLOW_RENAME = {
    "日期": "date",
    "收盘价": "close",
    "涨跌幅": "pct_change",
    "主力净流入-净额": "main_net_inflow",
    "主力净流入-净占比": "main_net_pct",
    "超大单净流入-净额": "super_net_inflow",
    "超大单净流入-净占比": "super_net_pct",
    "大单净流入-净额": "big_net_inflow",
    "大单净流入-净占比": "big_net_pct",
    "中单净流入-净额": "mid_net_inflow",
    "中单净流入-净占比": "mid_net_pct",
    "小单净流入-净额": "small_net_inflow",
    "小单净流入-净占比": "small_net_pct",
}


def _market_for_symbol(symbol: str) -> str:
    """Return akshare market code: 'sh' for Shanghai (6xxxxx), 'sz' for Shenzhen (0/3xxxxx)."""
    return "sh" if symbol.startswith("6") else "sz"


def fetch_fund_flow(symbol: str, retries: int = 5, backoff: float = 2.0) -> pd.DataFrame:
    """
    Fetch main-fund-flow data (主力/超大/大/中/小单净流入及占比).
    Returns ~100 trading days of data.
    """
    last_err: Optional[Exception] = None
    market = _market_for_symbol(symbol)
    for attempt in range(1, retries + 1):
        try:
            raw = _run_with_timeout(
                ak.stock_individual_fund_flow,
                MAX_SYMBOL_SECONDS,
                stock=symbol,
                market=market,
            )
            break
        except TimeoutError as exc:
            last_err = exc
            print(f"timeout fund_flow {symbol} attempt {attempt}/{retries}", flush=True)
        except Exception as exc:
            last_err = exc
        time.sleep(backoff ** attempt)
    else:
        raise RuntimeError(f"Failed to fetch fund_flow for {symbol}: {last_err}") from last_err

    if raw is None or raw.empty:
        raise RuntimeError(f"Empty fund_flow result for {symbol}")

    df = raw.copy()
    # Rename columns
    rename = {k: v for k, v in FUND_FLOW_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename)
    # Ensure required columns exist
    for col in ["date", "close", "pct_change", "main_net_inflow", "main_net_pct",
                "super_net_inflow", "super_net_pct", "big_net_inflow", "big_net_pct",
                "mid_net_inflow", "mid_net_pct", "small_net_inflow", "small_net_pct"]:
        if col not in df.columns:
            df[col] = 0.0
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    # Numeric cleanup
    for col in df.columns:
        if col == "date":
            continue
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce").fillna(0.0)
    return df[["date", "close", "pct_change", "main_net_inflow", "main_net_pct",
               "super_net_inflow", "super_net_pct", "big_net_inflow", "big_net_pct",
               "mid_net_inflow", "mid_net_pct", "small_net_inflow", "small_net_pct"]]


def fetch_and_store_fund_flow(symbol: str, base_dir=None, rate_limit_sleep: float = 3.0) -> None:
    """
    Fetch and store fund flow data, skipping if already up to date.
    rate_limit_sleep: seconds to wait between requests to avoid eastmoney rate limiting.
    """
    last_trade = _last_trading_date(pd.Timestamp.today())
    range_info = fund_flow_date_range(symbol, base_dir=base_dir)
    if range_info is not None:
        max_exist = range_info[1].date()
        if max_exist >= last_trade.date():
            print(f"skip fund_flow {symbol}: already up to {max_exist}", flush=True)
            return

    try:
        df = fetch_fund_flow(symbol)
        save_fund_flow(symbol, df, base_dir=base_dir)
        print(f"fund_flow {symbol}: saved {len(df)} rows", flush=True)
    except Exception as exc:
        # Rate limit (RemoteDisconnected) — back off and retry once
        err_str = str(exc)
        if "RemoteDisconnected" in err_str or "Connection aborted" in err_str:
            print(f"fund_flow {symbol}: rate-limited, backing off {rate_limit_sleep}s...", flush=True)
            time.sleep(rate_limit_sleep)
            try:
                df = fetch_fund_flow(symbol)
                save_fund_flow(symbol, df, base_dir=base_dir)
                print(f"fund_flow {symbol}: saved {len(df)} rows (retry OK)", flush=True)
                return
            except Exception:
                pass  # fall through to log
        print(f"fund_flow {symbol}: fetch failed — {exc}", flush=True)


# ─── Chip distribution (筹码分布) ───────────────────────────────────────────────

CHIP_RENAME = {
    "日期": "date",
    "获利比例": "profit_ratio",
    "平均成本": "avg_cost",
    "90成本-低": "c90_low",
    "90成本-高": "c90_high",
    "90集中度": "c90集中度",
    "70成本-低": "c70_low",
    "70成本-高": "c70_high",
    "70集中度": "c70集中度",
}


def fetch_chip(symbol: str, retries: int = 5, backoff: float = 2.0) -> pd.DataFrame:
    """
    Fetch chip distribution data (筹码分布: 获利比例/平均成本/90&70成本区间和集中度).
    Returns ~90 days of data.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            raw = _run_with_timeout(ak.stock_cyq_em, MAX_SYMBOL_SECONDS, symbol=symbol)
            break
        except TimeoutError as exc:
            last_err = exc
            print(f"timeout chip {symbol} attempt {attempt}/{retries}", flush=True)
        except Exception as exc:
            last_err = exc
        time.sleep(backoff ** attempt)
    else:
        raise RuntimeError(f"Failed to fetch chip for {symbol}: {last_err}") from last_err

    if raw is None or raw.empty:
        raise RuntimeError(f"Empty chip result for {symbol}")

    df = raw.copy()
    rename = {k: v for k, v in CHIP_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename)
    for col in ["date", "profit_ratio", "avg_cost", "c90_low", "c90_high", "c90集中度",
                "c70_low", "c70_high", "c70集中度"]:
        if col not in df.columns:
            df[col] = 0.0
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    for col in df.columns:
        if col == "date":
            continue
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)
    return df[["date", "profit_ratio", "avg_cost", "c90_low", "c90_high", "c90集中度",
               "c70_low", "c70_high", "c70集中度"]]


def fetch_and_store_chip(symbol: str, base_dir=None) -> None:
    """Fetch and store chip distribution data, skipping if already up to date."""
    last_trade = _last_trading_date(pd.Timestamp.today())
    range_info = chip_date_range(symbol, base_dir=base_dir)
    if range_info is not None:
        max_exist = range_info[1].date()
        if max_exist >= last_trade.date():
            print(f"skip chip {symbol}: already up to {max_exist}", flush=True)
            return

    try:
        df = fetch_chip(symbol)
        save_chip(symbol, df, base_dir=base_dir)
        print(f"chip {symbol}: saved {len(df)} rows", flush=True)
    except Exception as exc:
        print(f"chip {symbol}: fetch failed — {exc}", flush=True)


def fetch_all_enriched(start_date: str = DEFAULT_START_DATE, limit: Optional[int] = None, base_dir=None) -> List[str]:
    """
    Full fetch: prices (qfq+hfq) + fund_flow + chip for all symbols.
    Replaces fetch_all to be called from the migration script.
    """
    from .storage import price_date_range
    base_dir = base_dir or DATA_DIR
    universe = list_all_symbols(base_dir=base_dir)
    if limit:
        universe = universe[:limit]

    desired_start = pd.to_datetime(start_date).date()
    last_trade = _last_trading_date(pd.Timestamp.today()).date()
    today = pd.Timestamp.today().normalize().date()

    # Build fetch plan for prices
    price_plan: List[tuple[str, str]] = []
    for code in universe:
        range_info = price_date_range(code, base_dir=base_dir)
        if range_info is None:
            price_plan.append((code, start_date))
            continue
        min_dt, max_dt = range_info
        min_dt, max_dt = min_dt.date(), max_dt.date()
        gap_days = (min_dt - desired_start).days if min_dt > desired_start else 0
        covered_from_start = min_dt <= desired_start or gap_days <= 7
        if max_dt >= last_trade and covered_from_start:
            continue
        if min_dt > desired_start and gap_days > 7:
            price_plan.append((code, start_date))
            continue
        start_missing = (pd.to_datetime(max_dt) + pd.Timedelta(days=1)).date()
        if start_missing > last_trade:
            continue
        price_plan.append((code, start_missing.strftime("%Y-%m-%d")))

    if limit:
        price_plan = price_plan[:limit]

    print(f"Fetch plan: {len(price_plan)} symbols for prices, {len(universe)} for fund_flow/chip", flush=True)

    for idx, (code, start_at) in enumerate(price_plan, 1):
        try:
            print(f"[{idx}/{len(price_plan)}] {code} (prices from {start_at})", flush=True)
            fetch_and_store(code, start_date=start_at, base_dir=base_dir)
        except Exception as exc:
            print(f"Warning: failed to fetch prices for {code}: {exc}", flush=True)
        if REQUEST_SLEEP_SECONDS > 0:
            time.sleep(REQUEST_SLEEP_SECONDS)

    # Fund flow and chip are fetched for all universe (they have limited history anyway)
    fund_flow_universe = universe[:limit] if limit else universe
    for idx, code in enumerate(fund_flow_universe, 1):
        try:
            fetch_and_store_fund_flow(code, base_dir=base_dir)
            fetch_and_store_chip(code, base_dir=base_dir)
        except Exception as exc:
            print(f"Warning: failed fund_flow/chip for {code}: {exc}", flush=True)
        if REQUEST_SLEEP_SECONDS > 0:
            time.sleep(REQUEST_SLEEP_SECONDS)
        if idx % 100 == 0 or idx == len(fund_flow_universe):
            print(f"Fund flow / chip progress: {idx}/{len(fund_flow_universe)}", flush=True)

    return [code for code, _ in price_plan]
