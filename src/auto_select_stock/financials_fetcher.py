from datetime import date
from pathlib import Path
import time
import os
import threading
from typing import Iterable, List, Optional

import akshare as ak
import pandas as pd

from .config import DATA_DIR
from .storage import load_financial, save_financial
from .storage import financial_date_range

MAX_FIN_SECONDS = float(os.getenv("AUTO_SELECT_FIN_TIMEOUT", "10"))
MAX_FIN_ATTEMPTS = int(os.getenv("AUTO_SELECT_FIN_ATTEMPTS", "10"))


def _latest_quarter_end(ref: date) -> date:
    """Return the most recent fiscal quarter end on or before ref."""
    year = ref.year
    quarter_ends = [date(year, 3, 31), date(year, 6, 30), date(year, 9, 30), date(year, 12, 31)]
    for qe in reversed(quarter_ends):
        if ref >= qe:
            return qe
    # If ref is before Jan 1 (shouldn't happen), fall back to previous year's Q4.
    return date(year - 1, 12, 31)


def _clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace("False", pd.NA)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _pick_first(df: pd.DataFrame, target: str, candidates: list[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            df[target] = df[name]
            return name
    return None


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


def _format_em_symbol(symbol: str) -> str:
    """Translate a 6-digit ticker into the Eastmoney format with exchange suffix."""
    normalized = symbol.upper()
    if normalized.endswith((".SH", ".SZ", ".BJ")):
        return normalized
    code = normalized[-6:]
    if code.startswith(("6", "9")):
        suffix = "SH"
    elif code.startswith(("4", "8")):
        suffix = "BJ"
    else:
        suffix = "SZ"
    return f"{code}.{suffix}"


def _tidy_financial_abstract(wide: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Convert Akshare financial abstract wide table into a tidy, date-indexed table
    with all indicators kept as columns.
    """
    if wide is None or wide.empty:
        raise RuntimeError(f"No financial abstract data for {symbol}")

    wide = wide.copy()
    date_cols = [c for c in wide.columns if c not in {"选项", "指标"}]
    parsed = {c: pd.to_datetime(c, errors="coerce") for c in date_cols}
    valid_cols = [c for c, dt in parsed.items() if pd.notna(dt)]
    if not valid_cols:
        raise RuntimeError(f"No valid date columns in abstract for {symbol}")

    # Keep indicator rows, clean numeric values, then transpose
    indicators = wide.set_index("指标")[valid_cols]
    indicators = indicators.apply(_clean_numeric)
    tidy = indicators.T
    tidy["date"] = [parsed[c].date() for c in valid_cols]
    tidy.sort_values("date", inplace=True)
    tidy.reset_index(drop=True, inplace=True)

    # Add english aliases to keep downstream consumers working
    _pick_first(tidy, "roe", ["净资产收益率(ROE)", "净资产收益率", "净资产收益率(%)", "净资产收益率-加权", "净资产收益率-摊薄"])
    _pick_first(tidy, "net_profit_margin", ["销售净利率", "净利润率", "净利润率(%)", "净利率"])
    _pick_first(tidy, "gross_margin", ["毛利率", "毛利率(%)", "销售毛利率"])
    _pick_first(tidy, "debt_to_asset", ["资产负债率", "资产负债率(%)"])
    _pick_first(tidy, "eps", ["基本每股收益", "基本每股收益(元)", "每股收益", "每股收益(元)"])
    ocf_source = _pick_first(tidy, "operating_cashflow_per_share", ["每股经营性现金流", "每股经营性现金流(元)", "每股经营活动产生的现金流量净额"])
    if ocf_source:
        tidy["operating_cashflow_growth"] = tidy["operating_cashflow_per_share"].pct_change().fillna(0.0) * 100.0

    cols = ["date"] + [c for c in tidy.columns if c != "date"]
    return tidy[cols]


def _tidy_indicator(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Fallback tidy for indicator endpoint when abstract fails."""
    rename_map = {"报表期": "date", "报告期": "date", "报告日期": "date", "公告日期": "date"}
    df = df.rename(columns=rename_map)
    if "date" not in df.columns:
        raise RuntimeError(f"Indicator data missing date column for {symbol}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    if df.empty:
        raise RuntimeError(f"No dated indicator rows for {symbol}")
    for col in df.columns:
        if col == "date":
            continue
        df[col] = _clean_numeric(df[col])
    _pick_first(df, "roe", ["净资产收益率", "净资产收益率(%)", "净资产收益率-加权", "净资产收益率-摊薄"])
    _pick_first(df, "net_profit_margin", ["销售净利率", "净利润率", "净利润率(%)", "净利率"])
    _pick_first(df, "gross_margin", ["毛利率", "毛利率(%)", "销售毛利率"])
    _pick_first(df, "debt_to_asset", ["资产负债率", "资产负债率(%)"])
    _pick_first(df, "eps", ["基本每股收益", "基本每股收益(元)", "每股收益", "每股收益(元)"])
    ocf_source = _pick_first(df, "operating_cashflow_per_share", ["每股经营性现金流", "每股经营性现金流(元)", "每股经营活动产生的现金流量净额"])
    if ocf_source:
        df["operating_cashflow_growth"] = df["operating_cashflow_per_share"].pct_change().fillna(0.0) * 100.0
    cols = ["date"] + [c for c in df.columns if c != "date"]
    return df[cols]


def _tidy_indicator_em(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Tidy Eastmoney indicator payload, keeping all returned indicator columns and
    adding the downstream-friendly aliases used elsewhere in the pipeline.
    """
    if df is None or df.empty:
        raise RuntimeError(f"No Eastmoney indicator data for {symbol}")
    df = df.copy()
    df = df.rename(columns={"REPORT_DATE": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    mappings = {
        "roe": "ROEJQ",
        "net_profit_margin": "XSJLL",
        "gross_margin": "XSMLL",
        "debt_to_asset": "ZCFZL",
        "eps": "EPSJB",
        "operating_cashflow_per_share": "MGJYXJJE",
    }
    for target, source in mappings.items():
        df[target] = _clean_numeric(df[source]) if source in df.columns else 0.0
    if "operating_cashflow_per_share" in df:
        df["operating_cashflow_growth"] = (
            df["operating_cashflow_per_share"].pct_change().fillna(0.0) * 100.0
        )
    cols = ["date"] + [c for c in df.columns if c != "date"]
    return df[cols]


def fetch_financials(symbol: str, start_year: str = "2018", retries: int = 3, backoff: float = 1.5) -> pd.DataFrame:
    """
    Fetch historical financial indicators using Akshare stock_financial_abstract,
    keeping all returned indicators as columns. English aliases are added for
    downstream compatibility.
    """
    code = symbol[-6:]
    last_err: Optional[Exception] = None

    # Prefer Eastmoney main indicator endpoint to avoid Sina anti-scraping blocks.
    try:
        df_em = _run_with_timeout(
            ak.stock_financial_analysis_indicator_em,
            MAX_FIN_SECONDS,
            symbol=_format_em_symbol(code),
            indicator="按单季度",
        )
        if df_em is not None and not df_em.empty:
            return _tidy_indicator_em(df_em, symbol=code)
    except TimeoutError as exc:
        last_err = exc
        print(f"timeout financial em {symbol}", flush=True)
    except Exception as exc:  # noqa: BLE001
        last_err = exc

    df: Optional[pd.DataFrame] = None
    for attempt in range(1, retries + 1):
        try:
            df = _run_with_timeout(ak.stock_financial_abstract, MAX_FIN_SECONDS, symbol=code)
            break
        except TimeoutError as exc:
            last_err = exc
            print(f"timeout financial abstract {symbol} attempt {attempt}/{retries}", flush=True)
            if attempt < retries:
                time.sleep(backoff ** attempt)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(backoff ** attempt)

    if df is not None and not df.empty:
        return _tidy_financial_abstract(df, symbol=code)

    # Fallback to indicator endpoint to avoid full failure in batch runs.
    try:
        df_ind = _run_with_timeout(
            ak.stock_financial_analysis_indicator, MAX_FIN_SECONDS, symbol=code, start_year=start_year
        )
        if df_ind is not None and not df_ind.empty:
            return _tidy_indicator(df_ind, symbol=code)
    except TimeoutError as exc:
        last_err = last_err or exc
        print(f"timeout financial indicator {symbol}", flush=True)
    except Exception as exc:  # noqa: BLE001
        last_err = last_err or exc

    raise RuntimeError(f"No financial data for {symbol}: {last_err or 'all providers returned empty data'}")


def save_financials(df: pd.DataFrame, symbol: str, base_dir: Path = DATA_DIR) -> Path:
    return save_financial(symbol, df, base_dir=base_dir)


def fetch_and_store_financials(symbol: str, base_dir: Path = DATA_DIR) -> Path:
    existing = None
    try:
        existing = load_financial(symbol, base_dir=base_dir)
    except Exception:
        existing = None

    latest_existing: Optional[date] = None
    if existing is not None and not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        latest_existing = existing["date"].max().date()

    today = pd.Timestamp.today().normalize().date()
    # Financials update quarterly; if we already have data up to the latest completed quarter, skip.
    latest_qe = _latest_quarter_end(today)
    if latest_existing and latest_existing >= latest_qe:
        print(f"skip {symbol}: financials up to latest quarter ({latest_existing})", flush=True)
        return save_financials(existing, symbol, base_dir=base_dir)

    last_err: Optional[Exception] = None
    fresh = None
    for attempt in range(1, MAX_FIN_ATTEMPTS + 1):
        try:
            fresh = _run_with_timeout(fetch_financials, MAX_FIN_SECONDS, symbol)
            fresh["date"] = pd.to_datetime(fresh["date"])
            break
        except TimeoutError as exc:
            last_err = exc
            print(f"timeout financial fetch {symbol} attempt {attempt}/{MAX_FIN_ATTEMPTS}", flush=True)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1.0 * attempt)

    if fresh is None:
        raise RuntimeError(f"financial fetch failed for {symbol} after retries: {last_err}")

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined.sort_values("date", inplace=True)
        combined = combined.drop_duplicates(subset="date", keep="last")
        return save_financials(combined, symbol, base_dir=base_dir)

    return save_financials(fresh, symbol, base_dir=base_dir)


def fetch_financials_for_symbols(symbols: Iterable[str], base_dir: Path = DATA_DIR, limit: Optional[int] = None) -> List[str]:
    symbols = list(symbols)
    today = pd.Timestamp.today().normalize().date()
    latest_qe = _latest_quarter_end(today)

    # Build a plan that skips symbols already up-to-date and honors limit after filtering.
    planned: list[str] = []
    for sym in symbols:
        try:
            rng = financial_date_range(sym, base_dir=base_dir)
        except Exception:
            rng = None
        if rng:
            _, max_dt = rng
            if max_dt.date() >= latest_qe:
                continue
        planned.append(sym)
    if limit is not None:
        planned = planned[:limit]

    written: List[str] = []
    for idx, sym in enumerate(planned, 1):
        try:
            print(f"[{idx}/{len(planned)}] fetching financials {sym}", flush=True)
            path = fetch_and_store_financials(sym, base_dir=base_dir)
            written.append(sym)
            if idx % 100 == 0 or idx == len(planned):
                print(f"Financial progress: {idx}/{len(planned)} written (db: {path})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to fetch financials for {sym}: {exc}", flush=True)
    return written
