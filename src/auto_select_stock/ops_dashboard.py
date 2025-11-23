"""
Simple local control panel server to run dataset fetching, financial downloads, training, and dashboard rendering
from a single web page. Start with:
    python -m auto_select_stock.ops_dashboard
Then open http://127.0.0.1:8000
"""

import json
import math
from datetime import datetime
import shutil
import signal
import sqlite3
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from subprocess import Popen
from typing import Dict, List, Optional
import sys
from urllib.parse import parse_qs, urlparse

import akshare as ak
import pandas as pd
import requests

from .config import DATA_DIR, PROJECT_ROOT, REPORT_DIR, PREPROCESSED_DIR
from .dashboard import build_rows
from .data_fetcher import list_all_symbols
from .storage import DB_PATH, list_symbols, load_financial, load_stock_history


LOG_DIR = PROJECT_ROOT / "logs"
SRC_DIR = PROJECT_ROOT / "src"
RUNNING: Dict[str, int] = {}
RUNNING_PROCS: Dict[str, Popen] = {}
LOG_HANDLES: Dict[str, object] = {}
LOG_PATHS: Dict[str, str] = {}
STATS_CACHE: Dict[str, object] = {"ts": 0.0, "data": None}
NAME_CACHE: Dict[str, object] = {"ts": 0.0, "map": {}}
DASH_CACHE: Dict[str, object] = {"ts": 0.0, "data": None}
REALTIME_CACHE: Dict[str, object] = {"ts": 0.0, "rows": None, "cols": None, "err": None}
REALTIME_CACHE_FILE = DATA_DIR / "realtime_cache.json"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json",
}
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json",
}


def ensure_logs() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _build_cmd(action: str, payload: Dict) -> List[str]:
    """
    Map frontend actions to CLI commands. Only allow known actions.
    """
    base = [sys.executable, "-m", "auto_select_stock.cli"]
    symbols = payload.get("symbols") or ""
    symlist = [s for s in symbols.replace(",", " ").split() if s]

    if action == "fetch_all":
        cmd = base + ["fetch-all", "--start", payload.get("start", "2018-01-01")]
        if payload.get("limit"):
            cmd += ["--limit", str(payload["limit"])]
        return cmd

    if action == "fetch_financials":
        cmd = base + ["fetch-financials"]
        if symlist:
            cmd += symlist
        if payload.get("limit"):
            cmd += ["--limit", str(payload["limit"])]
        return cmd

    if action == "render_dashboard":
        cmd = base + ["render-dashboard"]
        if symlist:
            cmd += symlist
        if payload.get("output"):
            cmd += ["--output", payload["output"]]
        if payload.get("lookback_short"):
            cmd += ["--lookback-short", str(payload["lookback_short"])]
        if payload.get("lookback_long"):
            cmd += ["--lookback-long", str(payload["lookback_long"])]
        return cmd

    if action == "train_transformer":
        cmd = base + ["train-transformer"]
        if symlist:
            cmd += symlist
        if payload.get("epochs"):
            cmd += ["--epochs", str(payload["epochs"])]
        if payload.get("seq_len"):
            cmd += ["--seq-len", str(payload["seq_len"])]
        if payload.get("batch_size"):
            cmd += ["--batch-size", str(payload["batch_size"])]
        if payload.get("lr"):
            cmd += ["--lr", str(payload["lr"])]
        if payload.get("device"):
            cmd += ["--device", payload["device"]]
        return cmd

    if action == "predict_transformer":
        symbol = payload.get("symbol")
        if not symbol:
            raise ValueError("symbol required for prediction")
        cmd = base + ["predict-transformer", symbol]
        if payload.get("seq_len"):
            cmd += ["--seq-len", str(payload["seq_len"])]
        if payload.get("checkpoint"):
            cmd += ["--checkpoint", payload["checkpoint"]]
        if payload.get("device"):
            cmd += ["--device", payload["device"]]
        return cmd

    if action == "backtest_transformer":
        cmd = base + ["backtest-transformer"]
        if symlist:
            cmd += symlist
        if payload.get("start"):
            cmd += ["--start", payload["start"]]
        if payload.get("end"):
            cmd += ["--end", payload["end"]]
        if payload.get("top_pct"):
            cmd += ["--top-pct", str(payload["top_pct"])]
        if payload.get("allow_short"):
            cmd += ["--allow-short"]
        if payload.get("checkpoint"):
            cmd += ["--checkpoint", payload["checkpoint"]]
        if payload.get("cost_bps") not in (None, ""):
            cmd += ["--cost-bps", str(payload["cost_bps"])]
        if payload.get("slippage_bps") not in (None, ""):
            cmd += ["--slippage-bps", str(payload["slippage_bps"])]
        return cmd

    raise ValueError(f"unsupported action: {action}")


def _start_process(cmd: List[str], action: str) -> str:
    _purge_dead_processes()
    existing = RUNNING_PROCS.get(action)
    if existing and existing.poll() is None:
        raise RuntimeError(f"{action} is already running (pid {existing.pid})")
    ensure_logs()
    log_path = LOG_DIR / f"ops_{action}_{_timestamp()}.log"
    fout = open(log_path, "w", encoding="utf-8")
    LOG_PATHS[action] = str(log_path)

    proc = Popen(cmd, cwd=SRC_DIR, stdout=fout, stderr=fout, start_new_session=True)
    RUNNING[action] = proc.pid
    RUNNING_PROCS[action] = proc
    LOG_HANDLES[action] = fout
    # Clear cached dashboard data so the next refresh will recompute after new data lands.
    DASH_CACHE["data"] = None
    DASH_CACHE["ts"] = 0.0

    # Cleanup watcher so RUNNING map reflects actual process state and log handles close.
    def watcher():
        proc.wait()
        RUNNING.pop(action, None)
        RUNNING_PROCS.pop(action, None)
        try:
            fout.close()
        except Exception:
            pass
        LOG_HANDLES.pop(action, None)

    threading.Thread(target=watcher, daemon=True).start()
    return str(log_path)


def _purge_dead_processes() -> None:
    """Cleanup tracking maps for finished processes to avoid stale state."""
    for action, proc in list(RUNNING_PROCS.items()):
        if proc.poll() is None:
            continue
        RUNNING_PROCS.pop(action, None)
        RUNNING.pop(action, None)
        try:
            LOG_HANDLES.pop(action, None).close()  # type: ignore[call-arg]
        except Exception:
            pass


def _read_log_tail(path: Path, max_bytes: int = 64000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
            return data
    except FileNotFoundError:
        return "Log not found yet..."


def _stop_action(action: str) -> str:
    pid = RUNNING.get(action)
    proc = RUNNING_PROCS.get(action)
    if pid is None or proc is None:
        return "No running process for this action."
    try:
        import os
        # Kill the whole process group to ensure child processes are stopped.
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        RUNNING.pop(action, None)
        RUNNING_PROCS.pop(action, None)
        try:
            LOG_HANDLES.pop(action, None).close()  # type: ignore[call-arg]
        except Exception:
            pass
        return f"Sent SIGTERM to PID {pid}"
    except Exception as exc:  # noqa: BLE001
        return f"Failed to stop: {exc}"


def _collect_stats(ttl_seconds: int = 300) -> Dict[str, object]:
    """
    Compute database coverage stats and cache them for a short period to avoid
    repeatedly hitting the symbol list endpoint.
    """
    now = time.time()
    cached = STATS_CACHE.get("data")
    if cached and now - float(STATS_CACHE.get("ts", 0.0)) < ttl_seconds:
        return cached  # type: ignore[return-value]

    try:
        price_symbols = set(list_symbols("price"))
    except Exception:
        price_symbols = set()
    try:
        fin_symbols = set(list_symbols("financial"))
    except Exception:
        fin_symbols = set()

    total_universe = None
    source = "local"
    try:
        total_universe = len(list_all_symbols())
        source = "remote"
    except Exception:
        total_universe = len(price_symbols | fin_symbols)
        source = "local"

    price_total = total_universe or len(price_symbols)
    fin_total = total_universe or len(fin_symbols)
    stats = {
        "price_downloaded": len(price_symbols),
        "price_total": price_total,
        "financial_downloaded": len(fin_symbols),
        "financial_total": fin_total,
        "total_universe": total_universe,
        "db_symbols": len(price_symbols | fin_symbols),
        "source": source,
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "price_latest_date": _latest_date("price"),
        "financial_latest_date": _latest_date("financial"),
    }
    STATS_CACHE["ts"] = now
    STATS_CACHE["data"] = stats
    return stats


def _latest_date(table: str) -> Optional[str]:
    """Fetch latest date string for a table; returns None if unavailable."""
    db_path = DB_PATH
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.execute(f"SELECT MAX(date) FROM {table}")
        row = cur.fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return None


def _name_map(ttl_seconds: int = 3600) -> Dict[str, str]:
    now = time.time()
    cached = NAME_CACHE.get("map") or {}
    if cached and now - float(NAME_CACHE.get("ts", 0.0)) < ttl_seconds:
        return cached  # type: ignore[return-value]
    if NAME_CACHE.get("refreshing"):
        return cached  # type: ignore[return-value]

    result = {"map": cached}

    def _fetch():
        try:
            df = ak.stock_info_a_code_name()
            mapping = {str(code): str(name) for code, name in zip(df["code"], df["name"])}
            NAME_CACHE["map"] = mapping
            NAME_CACHE["ts"] = time.time()
            result["map"] = mapping
        except Exception:
            # Swallow network errors; we will retry on next call.
            result["map"] = cached
        finally:
            NAME_CACHE["refreshing"] = False

    NAME_CACHE["refreshing"] = True
    thread = threading.Thread(target=_fetch, daemon=True)
    thread.start()
    thread.join(3.0)  # best-effort fetch, avoid blocking the main request thread
    return result["map"]  # type: ignore[return-value]


def _classify_market(symbol: str) -> str:
    sym = str(symbol)
    if sym.startswith(("688", "689")):
        return "科创板"
    if sym.startswith(("300", "301")):
        return "创业板"
    if sym.startswith("6"):
        return "沪市"
    if sym.startswith(("000", "001", "002", "003")):
        return "深市"
    if sym.startswith(
        (
            "830",
            "831",
            "832",
            "833",
            "834",
            "835",
            "836",
            "837",
            "838",
            "839",
            "870",
            "871",
            "872",
            "873",
            "874",
            "875",
            "876",
            "877",
            "878",
            "879",
        )
    ):
        return "北交所"
    return "其他"


def _is_market_open(now: Optional[datetime] = None) -> bool:
    """Server-side trading hour check (Mon-Fri, 9:30-11:30, 13:00-15:00)."""
    now = now or datetime.now()
    weekday = now.weekday()
    if weekday >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    morning = 9 * 60 + 30 <= minutes <= 11 * 60 + 30
    afternoon = 13 * 60 <= minutes <= 15 * 60
    return morning or afternoon


def _read_realtime_cache() -> Optional[Dict[str, object]]:
    try:
        if not REALTIME_CACHE_FILE.exists():
            return None
        with REALTIME_CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_realtime_cache(payload: Dict[str, object]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with REALTIME_CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        return


def _fetch_realtime(ttl_seconds: int = 30) -> Dict[str, object]:
    """
    Fetch full-market realtime snapshot via stock_zh_a_spot_em once, cache briefly,
    and fall back to last good cache or empty columns if necessary.
    """
    now = time.time()
    if (
        REALTIME_CACHE.get("rows") is not None
        and now - float(REALTIME_CACHE.get("ts", 0.0)) < ttl_seconds
    ):
        return {
            "rows": REALTIME_CACHE["rows"],
            "cols": REALTIME_CACHE["cols"],
            "ts": REALTIME_CACHE["ts"],
            "source": REALTIME_CACHE.get("source", "cache"),
            "error": REALTIME_CACHE.get("err"),
        }

    col_map = {
        "代码": ("symbol", "代码", {"fixed": True}),
        "名称": ("name", "名称", {"fixed": True}),
        "最新价": ("price", "最新价", {}),
        "涨跌幅": ("pct_change", "涨跌幅%", {}),
        "涨跌额": ("change_amount", "涨跌额", {}),
        "成交量": ("volume", "成交量", {}),
        "成交额": ("amount", "成交额", {}),
        "振幅": ("amplitude", "振幅%", {}),
        "最高": ("high", "最高", {}),
        "最低": ("low", "最低", {}),
        "今开": ("open", "今开", {}),
        "昨收": ("prev_close", "昨收", {}),
        "量比": ("volume_ratio", "量比", {}),
        "换手率": ("turnover_rate", "换手率%", {}),
        "市盈率-动态": ("pe_dynamic", "市盈率-动态", {}),
        "市净率": ("pb", "市净率", {}),
        "市盈率(动态)": ("pe_dynamic", "市盈率(动态)", {}),
        "上市日期": ("list_date", "上市日期", {}),
        "总市值": ("market_cap", "总市值", {}),
        "流通市值": ("float_cap", "流通市值", {}),
        "涨速": ("rise_speed", "涨速", {}),
        "5分钟涨跌": ("pct_5m", "5分钟涨跌%", {}),
        "60日涨跌幅": ("pct_60d", "60日涨跌幅%", {}),
        "年初至今涨跌幅": ("pct_ytd", "年初至今涨跌幅%", {}),
        "序号": ("rank", "序号", {}),
    }

    def _build_columns(cols_order: List[str]) -> List[Dict[str, object]]:
        columns: List[Dict[str, object]] = []
        for c in cols_order:
            key, label, meta = col_map[c]
            default_visible = key in {"symbol", "name", "price", "pct_change", "pe_dynamic"}
            columns.append({"key": key, "label": label, "defaultVisible": default_visible, **meta})
        columns.append({"key": "market", "label": "板块", "defaultVisible": False})
        return columns

    def _db_fallback(err_msg: str) -> Optional[Dict[str, object]]:
        try:
            fallback_rows = build_rows([], [20, 60], base_dir=DATA_DIR)
            names = _name_map()
            rows: List[Dict[str, object]] = []
            for r in fallback_rows:
                rows.append(
                    {
                        "symbol": r.symbol,
                        "name": names.get(r.symbol, ""),
                        "price": r.price,
                        "pct_change": r.pct_change,
                        "turnover_rate": r.turnover_rate,
                        "volume_ratio": r.volume_ratio if hasattr(r, "volume_ratio") else None,
                        "amplitude": r.amplitude if hasattr(r, "amplitude") else None,
                        "change_amount": r.change_amount if hasattr(r, "change_amount") else None,
                        "pe_dynamic": r.pe,
                        "market": _classify_market(r.symbol),
                    }
                )
            if not rows:
                return None
            columns = _build_columns(list(col_map.keys()))
            return _build_payload(rows, columns, now, "db_fallback", err_msg)
        except Exception:
            return None

    def _build_payload(
        rows: List[Dict[str, object]],
        cols: List[Dict[str, object]],
        ts_val: float,
        source: str,
        err_msg: Optional[str] = None,
    ) -> Dict[str, object]:
        REALTIME_CACHE["rows"] = rows
        REALTIME_CACHE["cols"] = cols
        REALTIME_CACHE["ts"] = ts_val
        REALTIME_CACHE["err"] = err_msg
        REALTIME_CACHE["source"] = source
        payload = {"rows": rows, "cols": cols, "ts": ts_val, "source": source, "error": err_msg}
        if rows or cols:
            _write_realtime_cache(payload)
        return payload

    # If休市，直接尝试读取本地缓存文件，避免无谓网络请求
    if not _is_market_open():
        cached = _read_realtime_cache()
        if cached:
            rows = cached.get("rows") or []
            cols = cached.get("cols") or _build_columns(list(col_map.keys()))
            ts_val = float(cached.get("ts", now))
            return _build_payload(rows, cols, ts_val, cached.get("source", "file_cache"), cached.get("error"))
        fallback = _db_fallback("market closed; no cache")
        if fallback:
            return fallback
        fallback_cols = _build_columns(list(col_map.keys()))
        return _build_payload([], fallback_cols, now, "closed_empty", "market closed; no data")

    def _fetch_spot_em_http() -> pd.DataFrame:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": "1",
            "pz": "5000",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "wbp2u": "|0|0|0|web",
            "fid": "f3",
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81",
            "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f22,f23,f24,f25,f26,f11",
        }
        resp = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("diff", [])
        records = []
        for row in data:
            records.append(
                {
                    "代码": row.get("f12"),
                    "名称": row.get("f14"),
                    "最新价": row.get("f2"),
                    "涨跌幅": row.get("f3"),
                    "涨跌额": row.get("f4"),
                    "成交量": row.get("f5"),
                    "成交额": row.get("f6"),
                    "振幅": row.get("f7"),
                    "换手率": row.get("f8"),
                    "市盈率-动态": row.get("f9"),
                    "量比": row.get("f10"),
                    "最高": row.get("f15"),
                    "最低": row.get("f16"),
                    "今开": row.get("f17"),
                    "昨收": row.get("f18"),
                    "总市值": row.get("f20"),
                    "流通市值": row.get("f21"),
                    "涨速": row.get("f22"),
                    "市净率": row.get("f23"),
                    "60日涨跌幅": row.get("f24"),
                    "年初至今涨跌幅": row.get("f25"),
                    "上市日期": row.get("f26"),
                    "5分钟涨跌": row.get("f11"),
                }
            )
        return pd.DataFrame(records)

    try:
        df = ak.stock_zh_a_spot()
        source = "spot_full"
    except Exception:
        try:
            df = _fetch_spot_em_http()
            source = "spot_em_http"
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)
            fallback = _db_fallback(err_msg)
            if fallback:
                return fallback
            if REALTIME_CACHE.get("rows") is not None:
                return _build_payload(
                    REALTIME_CACHE["rows"],
                    REALTIME_CACHE["cols"],
                    float(REALTIME_CACHE.get("ts", now)),
                    REALTIME_CACHE.get("source", "cache"),
                    err_msg,
                )
            fallback_cols = _build_columns(list(col_map.keys()))
            return _build_payload([], fallback_cols, now, "http_error", err_msg)

    if df is None or df.empty:
        err_msg = "empty realtime data"
        if source == "spot_full":
            # Try HTTP fallback if SDK returned empty silently
            try:
                df = _fetch_spot_em_http()
                source = "spot_em_http"
            except Exception as exc:  # noqa: BLE001
                err_msg = str(exc)
                df = None
        if df is None or df.empty:
            # Still empty: use cache/db fallback
            fallback = _db_fallback(err_msg)
            if fallback:
                return fallback
            if REALTIME_CACHE.get("rows"):
                return _build_payload(
                    REALTIME_CACHE["rows"],
                    REALTIME_CACHE["cols"],
                    float(REALTIME_CACHE.get("ts", now)),
                    REALTIME_CACHE.get("source", "cache"),
                    err_msg,
                )
            fallback_cols = _build_columns(list(col_map.keys()))
            return _build_payload([], fallback_cols, now, "empty_data", err_msg)

    cols_order = [c for c in df.columns if c in col_map]
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        item: Dict[str, object] = {}
        for c in cols_order:
            key, _label, _meta = col_map[c]
            val = r[c]
            if key == "list_date" and pd.notna(val):
                val = str(val)
            item[key] = val if not pd.isna(val) else None
        item["market"] = _classify_market(item.get("symbol", ""))
        rows.append(item)

    columns = _build_columns(cols_order)
    return _build_payload(rows, columns, now, source, None)


def _dashboard_rows(ttl_seconds: int = 5) -> Dict[str, object]:
    now = time.time()
    cached_rows = DASH_CACHE.get("data")
    cached_cols = DASH_CACHE.get("cols")
    if cached_rows and cached_cols and now - float(DASH_CACHE.get("ts", 0.0)) < ttl_seconds:
        return {"rows": cached_rows, "columns": cached_cols, "ts": DASH_CACHE["ts"]}

    data = _fetch_realtime(ttl_seconds=ttl_seconds)
    DASH_CACHE["data"] = data["rows"]
    DASH_CACHE["cols"] = data["cols"]
    DASH_CACHE["ts"] = data["ts"]
    return {"rows": data["rows"], "columns": data["cols"], "ts": data["ts"]}


def _to_float(val: object) -> Optional[float]:
    try:
        if val is None:
            return None
        num = float(val)
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    except (TypeError, ValueError):
        return None


def _load_stock_detail(symbol: str) -> Dict[str, object]:
    """
    Fetch historical daily bars and financial indicators for a symbol.
    Returns Python-native types for JSON serialization.
    """
    name = _name_map().get(symbol, "")
    payload: Dict[str, object] = {"symbol": symbol, "name": name}

    prices: List[Dict[str, object]] = []
    try:
        price_arr = load_stock_history(symbol)
        price_df = pd.DataFrame(price_arr)
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.date.astype(str)
        for row in price_df.to_dict("records"):
            clean = {k: (_to_float(v) if k != "date" else v) for k, v in row.items()}
            prices.append(clean)
    except FileNotFoundError:
        prices = []
    payload["prices"] = prices

    financial_rows: List[Dict[str, object]] = []
    fin_cols: List[str] = []
    try:
        fin_df = load_financial(symbol)
        fin_df["date"] = pd.to_datetime(fin_df["date"]).dt.date.astype(str)
        numeric_cols = [c for c in fin_df.columns if c not in {"symbol", "date"}]
        for col in numeric_cols:
            fin_df[col] = pd.to_numeric(fin_df[col], errors="coerce")
        fin_cols = [c for c in fin_df.columns if c not in {"symbol", "date"}]
        for row in fin_df.to_dict("records"):
            clean: Dict[str, object] = {}
            for key, val in row.items():
                if key == "symbol":
                    continue
                if key == "date":
                    clean[key] = val
                else:
                    clean[key] = _to_float(val)
            financial_rows.append(clean)
    except FileNotFoundError:
        financial_rows = []

    payload["financial"] = financial_rows
    payload["financial_columns"] = fin_cols
    payload["price_count"] = len(prices)
    payload["financial_count"] = len(financial_rows)
    return payload


def _wipe_data() -> str:
    """
    Delete everything under the configured DATA_DIR with safety checks.
    """
    base = DATA_DIR.resolve()
    if not base.exists():
        return f"{base} 不存在，无需清理。"
    removed = 0
    for child in base.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
        except Exception as exc:  # noqa: BLE001
            return f"删除失败: {exc}"
    base.mkdir(parents=True, exist_ok=True)
    return f"已删除 {removed} 个项，目录已重建: {base}"


def _wipe_price_data() -> str:
    """Delete all daily price records only."""
    if not DB_PATH.exists():
        return f"{DB_PATH} 不存在，无需清理。"
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute("SELECT COUNT(*) FROM price")
        count = cur.fetchone()[0] or 0
        conn.execute("DELETE FROM price")
        conn.commit()
        STATS_CACHE["ts"] = 0.0
        STATS_CACHE["data"] = None
        DASH_CACHE["ts"] = 0.0
        DASH_CACHE["data"] = None
        return f"已删除日线数据 {count} 条记录。"
    except Exception as exc:  # noqa: BLE001
        return f"删除失败: {exc}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _wipe_financial_data() -> str:
    """Delete all financial records only."""
    if not DB_PATH.exists():
        return f"{DB_PATH} 不存在，无需清理。"
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute("SELECT COUNT(*) FROM financial")
        count = cur.fetchone()[0] or 0
        conn.execute("DELETE FROM financial")
        conn.commit()
        STATS_CACHE["ts"] = 0.0
        STATS_CACHE["data"] = None
        DASH_CACHE["ts"] = 0.0
        DASH_CACHE["data"] = None
        return f"已删除财报数据 {count} 条记录。"
    except Exception as exc:  # noqa: BLE001
        return f"删除失败: {exc}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _wipe_feature_cache() -> str:
    if not PREPROCESSED_DIR.exists():
        return "预处理缓存目录不存在，无需清理。"
    removed = 0
    for path in PREPROCESSED_DIR.glob("*.npz"):
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return f"已删除特征缓存 {removed} 个 (.npz)"


def _wipe_dataset_cache() -> str:
    cache_dir = PREPROCESSED_DIR / "datasets"
    if not cache_dir.exists():
        return "数据集缓存目录不存在，无需清理。"
    try:
        shutil.rmtree(cache_dir)
        return "已删除数据集缓存目录"
    except Exception as exc:  # noqa: BLE001
        return f"删除数据集缓存失败: {exc}"


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Auto Stock 控制台</title>
  <style>
    body { font-family: "Inter", "Arial", sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:24px; }
    h1 { margin-top:0; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px; }
    .card { background:#111827; border:1px solid #1f2937; padding:18px; border-radius:12px; box-shadow:0 10px 40px rgba(0,0,0,0.35); }
    .card-title { font-size:18px; font-weight:700; margin:0 0 10px; color:#e2e8f0; }
    label { display:block; margin-top:8px; font-size:13px; color:#94a3b8; }
    input { width:100%; padding:8px; border-radius:8px; border:1px solid #1f2937; background:#0b1221; color:#e2e8f0; }
    .btn { margin-top:10px; width:100%; padding:10px; border:none; border-radius:10px; font-weight:600; cursor:pointer; }
    .btn[disabled] { opacity:0.7; cursor:not-allowed; }
    .btn-primary { background:#2563eb; color:#e2e8f0; }
    .btn-primary:hover:not([disabled]) { background:#1d4ed8; }
    .btn-primary[disabled] { background:#475569; color:#cbd5e1; }
    .btn-secondary { background:#64748b; color:#e2e8f0; }
    .btn-secondary:hover:not([disabled]) { background:#475569; }
    .btn-secondary[disabled] { background:#475569; color:#cbd5e1; }
    .btn-secondary.active { background:#2563eb; border:1px solid #2563eb; color:#e2e8f0; }
    .btn-secondary.active:hover { background:#1d4ed8; }
    .log { margin-top:12px; font-size:13px; color:#93c5fd; word-break:break-all; white-space:pre-wrap; background:linear-gradient(135deg,#0b1221,#0f172a); padding:8px; border-radius:10px; min-height:80px; max-height:200px; overflow-y:auto; border:1px solid #1f2937; box-shadow:inset 0 1px 0 rgba(255,255,255,0.04); }
    .desc { color:#cbd5e1; font-size:14px; }
    .tabs { display:flex; gap:12px; margin-bottom:16px; }
    .tab { display:none; }
    .tab.active { display:block; }
    .btn-tab { padding:10px 14px; border-radius:10px; border:1px solid #1f2937; background:#0b1221; color:#e2e8f0; cursor:pointer; }
    .btn-tab.active { background:#2563eb; border-color:#2563eb; }
    .row-pair { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px; align-items:start; }
    .full-row { grid-column:1 / -1; }
    .stat-row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1f2937; }
    .stat-row:last-child { border-bottom:none; }
    .stat-value { font-weight:700; color:#f8fafc; }
    .danger { color:#f87171; }
    .inline-action { padding:8px 0; color:#cbd5e1; }
    .meta.ok { color:#86efac; }
    .meta.err { color:#fca5a5; }
  </style>
</head>
<body>
  <h1>Auto Stock 一键控制台</h1>
  <div class="tabs">
    <button id="tabbtn-data" class="btn-tab active" onclick="showTab('tab-data')">数据获取</button>
    <button id="tabbtn-dashboard" class="btn-tab" onclick="showTab('tab-dashboard')">看板</button>
    <button id="tabbtn-train" class="btn-tab" onclick="showTab('tab-train')">训练与推理</button>
    <button id="tabbtn-eval" class="btn-tab" onclick="showTab('tab-eval')">评估与回测</button>
  </div>
  <div id="tab-data" class="tab active">
    <div class="row-pair full-row" style="margin-bottom:16px;">
      <div class="card">
        <div class="card-title">数据抓取统计</div>
        <div class="stat-row">
          <span>日线覆盖</span>
          <span id="stat-price" class="stat-value">--</span>
        </div>
        <div class="stat-row">
          <span>财务覆盖</span>
          <span id="stat-fin" class="stat-value">--</span>
        </div>
        <div class="stat-row">
          <span>最新日线日期</span>
          <span id="stat-price-date" class="stat-value">--</span>
        </div>
        <div class="stat-row">
          <span>最新财务日期</span>
          <span id="stat-fin-date" class="stat-value">--</span>
        </div>
        <div class="meta" id="stat-meta" style="margin-top:10px;">自动刷新中...</div>
      </div>
      <div class="inline-action">
        <div class="desc danger">危险操作：删除全部数据</div>
        <div class="meta">清空 data 目录下的数据库与文件，不可恢复。</div>
        <button class="btn btn-secondary danger" style="margin-top:10px; width:auto;" onclick="wipeData()">删除所有数据</button>
        <div style="margin-top:8px;">
          <button class="btn btn-secondary danger" style="width:auto;" onclick="wipePrice()">仅删除日线数据</button>
          <button class="btn btn-secondary danger" style="width:auto; margin-left:8px;" onclick="wipeFinancial()">仅删除财报数据</button>
        </div>
        <div class="meta" id="stat-wipe-msg" style="margin-top:10px;"></div>
        <div class="meta" id="stat-wipe-price" style="margin-top:6px;"></div>
        <div class="meta" id="stat-wipe-fin" style="margin-top:6px;"></div>
      </div>
    </div>
    <div class="grid">
      <div class="card">
        <div class="card-title">抓取全量历史日线数据</div>
        <label>起始日期 (YYYY-MM-DD)</label><input id="fetch_start" value="2000-01-01" />
        <label>只抓取前 N 只（可空）</label><input id="fetch_limit" placeholder="留空=全部" />
        <button type="button" id="btn_start_fetch_all" class="btn btn-primary" onclick="runAction('fetch_all')">开始抓取</button>
        <button type="button" id="btn_stop_fetch_all" class="btn btn-secondary" onclick="stopAction('fetch_all')">暂停</button>
        <div class="log" id="log_fetch_all"></div>
      </div>
      <div class="card">
        <div class="card-title">抓取财报指标到 data/financials</div>
        <label>股票代码列表(空格/逗号分隔，留空=全部)</label><input id="fin_symbols" />
        <label>只抓取前 N 只（可空）</label><input id="fin_limit" placeholder="留空=全部" />
        <button type="button" id="btn_start_fin" class="btn btn-primary" onclick="runAction('fetch_financials')">抓取财报</button>
        <button type="button" id="btn_stop_fin" class="btn btn-secondary" onclick="stopAction('fetch_financials')">暂停</button>
        <div class="log" id="log_fetch_financials"></div>
      </div>
    </div>
  </div>
  <div id="tab-dashboard" class="tab">
    <div class="grid">
      <div class="card">
        <div class="card-title">实时行情看板（新股/全市场快照）</div>
        <button class="btn btn-primary" onclick="refreshDashboard()">刷新看板</button>
        <iframe id="dashFrame" src="/dashboard" style="width:100%; height:720px; border:1px solid #1f2937; border-radius:10px; margin-top:12px; background:#0b1221;"></iframe>
        <div class="log" id="log_render_dashboard"></div>
      </div>
    </div>
  </div>
  <div id="tab-train" class="tab">
    <div class="grid">
      <div class="card">
        <div class="desc">训练 Transformer 价格预测模型</div>
        <label>股票代码列表(空=全部)</label><input id="train_symbols" />
        <label>Epochs</label><input id="train_epochs" placeholder="20" />
        <label>Seq Len</label><input id="train_seq" placeholder="60" />
        <label>Batch Size</label><input id="train_batch" placeholder="64" />
        <label>LR</label><input id="train_lr" placeholder="0.001" />
        <label>设备(cuda/cpu，空=自动)</label><input id="train_device" placeholder="" />
        <button class="btn btn-primary" onclick="runAction('train_transformer')">开始训练</button>
        <button class="btn btn-secondary" onclick="stopAction('train_transformer')">暂停</button>
        <div class="log" id="log_train_transformer"></div>
        <hr style="border:1px solid #1f2937; margin:12px 0;" />
        <div class="desc">推理预测次日收盘价</div>
        <label>股票代码</label><input id="pred_symbol" />
        <label>Seq Len (可空)</label><input id="pred_seq" placeholder="60" />
        <label>Checkpoint 路径 (可空)</label><input id="pred_ckpt" placeholder="models/price_transformer.pt" />
        <label>设备(cuda/cpu，空=自动)</label><input id="pred_device" />
        <button class="btn btn-primary" onclick="runAction('predict_transformer')">开始推理</button>
        <button class="btn btn-secondary" onclick="stopAction('predict_transformer')">暂停</button>
        <div class="log" id="log_predict_transformer"></div>
        <hr style="border:1px solid #1f2937; margin:12px 0;" />
        <div class="desc">缓存管理</div>
        <div class="meta">删除预处理特征缓存（*.npz）或数据集缓存（datasets/*.pt）</div>
        <div style="margin-top:8px;">
          <button class="btn btn-secondary danger" style="width:auto;" onclick="wipeFeatureCache()">删除特征缓存</button>
          <button class="btn btn-secondary danger" style="width:auto; margin-left:8px;" onclick="wipeDatasetCache()">删除数据集缓存</button>
        </div>
        <div class="meta" id="stat-wipe-feature" style="margin-top:8px;"></div>
        <div class="meta" id="stat-wipe-dataset" style="margin-top:6px;"></div>
      </div>
    </div>
  </div>
  <div id="tab-eval" class="tab">
    <div class="grid">
      <div class="card">
        <div class="card-title">简单多空回测</div>
        <div class="desc">使用训练好的 Transformer 预测次日收益，构建多/空组合并计算收益、波动、Sharpe、回撤、换手等指标。</div>
        <label>股票代码列表(空=全部)</label><input id="bt_symbols" />
        <label>开始日期</label><input id="bt_start" placeholder="YYYY-MM-DD" />
        <label>结束日期</label><input id="bt_end" placeholder="YYYY-MM-DD" />
        <label>Top 百分比 (0-1)</label><input id="bt_top_pct" placeholder="0.1" />
        <label>Checkpoint 路径</label><input id="bt_ckpt" placeholder="models/price_transformer.pt" />
        <label>成本 (bp)</label><input id="bt_cost" placeholder="0" />
        <label>滑点 (bp)</label><input id="bt_slippage" placeholder="0" />
        <label style="display:flex; align-items:center; gap:8px; margin-top:12px;">
          <input type="checkbox" id="bt_allow_short" style="width:auto;" /> 允许做空尾部组合
        </label>
        <button class="btn btn-primary" onclick="runAction('backtest_transformer')">开始回测</button>
        <button class="btn btn-secondary" onclick="stopAction('backtest_transformer')">暂停</button>
        <div class="log" id="log_backtest_transformer"></div>
      </div>
    </div>
  </div>
  <script>
    const timers = {};
    const DEFAULT_TIMEOUT = 15000; // 15s 网络超时，避免请求挂起

    async function fetchWithTimeout(url, options = {}, timeout = DEFAULT_TIMEOUT) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeout);
      try {
        return await fetch(url, {...options, signal: controller.signal});
      } finally {
        clearTimeout(timer);
      }
    }
    function showTab(tabId) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.btn-tab').forEach(b => b.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      document.getElementById('tabbtn-' + tabId.split('-')[1]).classList.add('active');
    }
    function stopTimer(action) {
      if (timers[action]) { clearInterval(timers[action]); delete timers[action]; }
    }
    function startLogStream(action, logPath) {
      stopTimer(action);
      const logEl = document.getElementById('log_' + action);
      const fetchLog = async () => {
        try {
          const resp = await fetchWithTimeout('/log?path=' + encodeURIComponent(logPath), {}, 30000);
          const txt = await resp.text();
          logEl.textContent = txt;
          logEl.scrollTop = logEl.scrollHeight;
        } catch (e) {
          if (e.name === 'AbortError') {
            // Swallow transient aborts to avoid flashing errors in UI; next tick will retry.
            return;
          }
          logEl.textContent = 'Log error: ' + e;
        }
      };
      fetchLog();
      timers[action] = setInterval(fetchLog, 2000);
    }
    async function runAction(action) {
      const payload = {};
      if (action === 'fetch_all') {
        payload.start = document.getElementById('fetch_start').value;
        payload.limit = document.getElementById('fetch_limit').value;
      } else if (action === 'fetch_financials') {
        payload.symbols = document.getElementById('fin_symbols').value;
        payload.limit = document.getElementById('fin_limit').value;
      } else if (action === 'train_transformer') {
        payload.symbols = document.getElementById('train_symbols').value;
        payload.epochs = document.getElementById('train_epochs').value;
        payload.seq_len = document.getElementById('train_seq').value;
        payload.batch_size = document.getElementById('train_batch').value;
        payload.lr = document.getElementById('train_lr').value;
        payload.device = document.getElementById('train_device').value;
      } else if (action === 'predict_transformer') {
        payload.symbol = document.getElementById('pred_symbol').value;
        payload.seq_len = document.getElementById('pred_seq').value;
        payload.checkpoint = document.getElementById('pred_ckpt').value;
        payload.device = document.getElementById('pred_device').value;
      } else if (action === 'backtest_transformer') {
        payload.symbols = document.getElementById('bt_symbols').value;
        payload.start = document.getElementById('bt_start').value;
        payload.end = document.getElementById('bt_end').value;
        payload.top_pct = document.getElementById('bt_top_pct').value;
        payload.checkpoint = document.getElementById('bt_ckpt').value;
        payload.cost_bps = document.getElementById('bt_cost').value;
        payload.slippage_bps = document.getElementById('bt_slippage').value;
        payload.allow_short = document.getElementById('bt_allow_short').checked;
      } else if (action === 'render_dashboard') {
        payload.symbols = document.getElementById('dash_symbols').value;
        payload.lookback_short = document.getElementById('dash_short').value;
        payload.lookback_long = document.getElementById('dash_long').value;
        payload.output = document.getElementById('dash_output').value;
      }

      const logEl = document.getElementById('log_' + action);
      // Optimistically switch button states; will revert on failure. Ignore UI errors.
      try { toggleButtons(action, true); } catch (e) { console.warn('toggleButtons failed', e); }
      logEl.textContent = '启动中...';
      try {
        const resp = await fetchWithTimeout('/run', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({action, payload})
        });
        if (!resp.ok) {
          const text = await resp.text();
          logEl.textContent = text || '请求失败';
          stopTimer(action);
          try { toggleButtons(action, false); } catch (e) { console.warn('toggleButtons failed', e); }
          return;
        }
        const info = await resp.json();
        logEl.textContent = `已启动 ${action}. 日志: ${info.log}`;
        startLogStream(action, info.log);
      } catch (e) {
        logEl.textContent = '请求失败: ' + e;
        try { toggleButtons(action, false); } catch (err) { console.warn('toggleButtons failed', err); }
      }
    }
    function refreshDashboard() {
      const frame = document.getElementById('dashFrame');
      frame.src = '/dashboard?ts=' + Date.now();
      document.getElementById('log_render_dashboard').textContent = '已刷新看板';
    }
    async function refreshStats() {
      const meta = document.getElementById('stat-meta');
      meta.classList.remove('ok', 'err');
      meta.textContent = '统计计算中...';
      try {
        const resp = await fetchWithTimeout('/stats');
        if (!resp.ok) {
          meta.textContent = await resp.text();
          meta.classList.add('err');
          return;
        }
        const stats = await resp.json();
        const priceTotal = stats.price_total || stats.total_universe || 0;
        const finTotal = stats.financial_total || stats.total_universe || 0;
        document.getElementById('stat-price').textContent = `${stats.price_downloaded}/${priceTotal}`;
        document.getElementById('stat-fin').textContent = `${stats.financial_downloaded}/${finTotal}`;
        document.getElementById('stat-price-date').textContent = stats.price_latest_date || '--';
        document.getElementById('stat-fin-date').textContent = stats.financial_latest_date || '--';
        const srcLabel = stats.source === 'remote' ? '符号总数来自网络接口' : '符号总数来自本地数据';
        const totalUniverse = stats.total_universe || stats.db_symbols || Math.max(priceTotal, finTotal);
        meta.textContent = `统计成功 · 符号总数: ${totalUniverse} (${srcLabel}) · 计算于 ${stats.cached_at}`;
        meta.classList.add('ok');
      } catch (e) {
        if (e.name === 'AbortError') {
          // 网络超时中止不展示为错误，下一轮刷新再试
          return;
        }
        meta.textContent = '统计失败: ' + e;
        meta.classList.add('err');
      }
    }
    setInterval(refreshStats, 8000);
    async function stopAction(action) {
      stopTimer(action);
      const logEl = document.getElementById('log_' + action);
      try {
        await fetchWithTimeout('/stop?action=' + encodeURIComponent(action), {method:'POST'});
        logEl.textContent += \"\\n已请求暂停（请查看对应日志确认停止）\";
        logEl.scrollTop = logEl.scrollHeight;
      } catch (e) {
        logEl.textContent += \"\\n暂停失败: \" + e;
      }
      try { toggleButtons(action, false); } catch (err) { console.warn('toggleButtons failed', err); }
      refreshStats();
    }
    async function wipeData() {
      const logEl = document.getElementById('stat-wipe-msg');
      if (!confirm('确定要删除全部数据吗？此操作不可恢复。')) return;
      if (!confirm('再次确认：删除 data 目录下的全部数据？')) return;
      logEl.textContent = '删除中...';
      try {
        const resp = await fetch('/wipe-data', {method:'POST'});
        const txt = await resp.text();
        logEl.textContent = txt;
        refreshStats();
      } catch (e) {
        logEl.textContent = '删除失败: ' + e;
      }
    }
    async function wipePrice() {
      const logEl = document.getElementById('stat-wipe-price');
      if (!confirm('确定要删除所有日线数据吗？')) return;
      logEl.textContent = '删除中...';
      try {
        const resp = await fetch('/wipe-price', {method:'POST'});
        const txt = await resp.text();
        logEl.textContent = txt;
        refreshStats();
      } catch (e) {
        logEl.textContent = '删除失败: ' + e;
      }
    }
    async function wipeFinancial() {
      const logEl = document.getElementById('stat-wipe-fin');
      if (!confirm('确定要删除所有财报数据吗？')) return;
      logEl.textContent = '删除中...';
      try {
        const resp = await fetch('/wipe-financial', {method:'POST'});
        const txt = await resp.text();
        logEl.textContent = txt;
        refreshStats();
      } catch (e) {
        logEl.textContent = '删除失败: ' + e;
      }
    }
    async function wipeFeatureCache() {
      const logEl = document.getElementById('stat-wipe-feature');
      logEl.textContent = '删除中...';
      try {
        const resp = await fetch('/wipe-feature-cache', {method:'POST'});
        const txt = await resp.text();
        logEl.textContent = txt;
      } catch (e) {
        logEl.textContent = '删除失败: ' + e;
      }
    }
    async function wipeDatasetCache() {
      const logEl = document.getElementById('stat-wipe-dataset');
      logEl.textContent = '删除中...';
      try {
        const resp = await fetch('/wipe-dataset-cache', {method:'POST'});
        const txt = await resp.text();
        logEl.textContent = txt;
      } catch (e) {
        logEl.textContent = '删除失败: ' + e;
      }
    }
    refreshStats();
  </script>
</body>
</html>
"""

DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Auto Stock 实时看板</title>
  <style>
    body { margin:0; padding:16px; font-family:"Inter","Arial",sans-serif; background:#0b1221; color:#e2e8f0; }
    h1 { margin:0 0 12px; }
    .card { background:#111827; border:1px solid #1f2937; border-radius:12px; padding:16px; box-shadow:0 10px 40px rgba(0,0,0,0.35); }
    .meta { color:#94a3b8; font-size:13px; }
    .table-wrap { max-height:calc(100vh - 260px); overflow:auto; border-radius:10px; border:1px solid #1f2937; }
    table { width:100%; border-collapse:collapse; margin-top:12px; }
    th, td { padding:10px; border-bottom:1px solid #1f2937; text-align:right; }
    th { cursor:pointer; color:#93c5fd; position:sticky; top:0; background:#0b1221; }
    th:first-child, td:first-child { text-align:left; }
    tr { transition:background-color 0.12s ease, color 0.12s ease; }
    tr:hover { background:rgba(59,130,246,0.12); }
    tr.active-row { background:rgba(34,197,94,0.12); }
    .controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:8px 0; }
    .pill { padding:6px 10px; border-radius:10px; border:1px solid #1f2937; background:#0f172a; color:#e2e8f0; font-size:13px; }
    .pill input { margin-right:6px; }
    .btn { padding:8px 12px; border-radius:10px; border:1px solid #2563eb; background:#2563eb; color:#e2e8f0; cursor:pointer; font-weight:600; }
    .status { margin-left:8px; }
    .pos { color:#ef4444; font-weight:600; }
    .neg { color:#22c55e; font-weight:600; }
    .zero { color:#e2e8f0; }
    .detail-title { font-size:18px; margin:0 0 6px; }
    .chart-head { display:flex; justify-content:space-between; align-items:center; }
    .chart-block { margin-top:12px; padding:10px; background:#0b1221; border:1px solid #1f2937; border-radius:10px; }
    .mini-table { margin-top:8px; border:1px solid #1f2937; border-radius:10px; overflow:auto; }
    .mini-table table { width:100%; border-collapse:collapse; min-width:640px; }
    .mini-table th, .mini-table td { padding:8px; border-bottom:1px solid #1f2937; font-size:12px; text-align:right; }
    .mini-table th { text-align:left; color:#93c5fd; background:#0f172a; }
    .badge { display:inline-block; padding:4px 8px; border-radius:999px; background:#0f172a; border:1px solid #1f2937; font-size:12px; color:#cbd5e1; margin-right:6px; }
    .modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.65); display:none; align-items:center; justify-content:center; z-index:50; padding:16px; }
    .modal { width:min(1080px, 100%); max-height:90vh; overflow:auto; background:linear-gradient(135deg,#0a1020,#0f172a); border:1px solid #2b3548; border-radius:14px; box-shadow:0 20px 60px rgba(0,0,0,0.6); padding:16px; }
    .modal-header { display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .modal-close { background:#1f2937; color:#e2e8f0; border:1px solid #334155; width:32px; height:32px; border-radius:10px; cursor:pointer; font-size:18px; line-height:1; }
    .modal-close:hover { background:#2563eb; border-color:#2563eb; }
  </style>
</head>
<body>
  <div class="card">
    <h1>实时看板</h1>
    <div class="controls">
      <button class="btn" onclick="loadData()">刷新数据</button>
      <span class="meta status" id="status">加载中...</span>
    </div>
    <div class="controls" id="market-filters"></div>
    <div class="controls" id="filters"></div>
    <div class="table-wrap">
      <table>
        <thead><tr id="header-row"></tr></thead>
        <tbody id="body-rows"></tbody>
      </table>
    </div>
  </div>
  <div id="detail-modal" class="modal-overlay" onclick="closeDetailModal()">
    <div class="modal" onclick="event.stopPropagation();">
      <div class="modal-header">
        <div>
          <h2 class="detail-title" id="detail-title">点击列表打开详情</h2>
          <div class="meta" id="detail-meta">展示该股票的历史日线与财务指标。</div>
        </div>
        <button class="modal-close" onclick="closeDetailModal()">×</button>
      </div>
      <div class="chart-block">
        <div class="chart-head">
          <span>历史日线</span>
          <span class="meta" id="price-meta"></span>
        </div>
        <canvas id="price-chart" height="180"></canvas>
        <div class="mini-table" id="price-last"></div>
      </div>
      <div class="chart-block">
        <div class="chart-head">
          <span>财务趋势</span>
          <span class="meta" id="fin-meta"></span>
        </div>
        <canvas id="financial-chart" height="180"></canvas>
        <div class="mini-table" id="financial-table"></div>
      </div>
    </div>
  </div>
  <script>
    const AUTO_REFRESH_MS = 10000;
    const MARKET_CHECK_MS = 60000;
    let columns = [];
    const signedColumns = ["pct_change","pct_5m","pct_60d","pct_ytd","change_amount","rise_speed"];
    const marketOptions = [
      {key:"沪市", label:"沪市"},
      {key:"深市", label:"深市"},
      {key:"创业板", label:"创业板"},
      {key:"科创板", label:"科创板"},
      {key:"北交所", label:"北交所"},
      {key:"其他", label:"其他"},
    ];
    let data = [];
    let sortKey = "symbol";
    let sortAsc = true;
    let visibleKeys = [];
    let activeMarkets = new Set(marketOptions.map(o => o.key));
    let columnsReady = false;
    let selectedSymbol = null;
    let refreshTimer = null;

    function format(val) {
      if (val === null || val === undefined || Number.isNaN(val)) return "";
      if (typeof val === "number") return val.toFixed(2);
      return val;
    }
    function applySignColor(td, key, value) {
      if (!signedColumns.includes(key) || typeof value !== "number" || Number.isNaN(value)) return;
      if (value > 0) td.classList.add("pos");
      else if (value < 0) td.classList.add("neg");
      else td.classList.add("zero");
    }
    function setDetailHeader(symbol, name) {
      const title = document.getElementById("detail-title");
      if (!symbol) {
        title.textContent = "点击列表打开详情";
        return;
      }
      title.textContent = name ? `${symbol} · ${name}` : symbol;
    }
    function openDetailModal(symbol, name) {
      selectedSymbol = symbol;
      const overlay = document.getElementById("detail-modal");
      overlay.style.display = "flex";
      setDetailHeader(symbol, name || "");
      document.getElementById("detail-meta").textContent = "加载历史数据中...";
      clearCanvas("price-chart", "加载中...");
      clearCanvas("financial-chart", "加载中...");
      renderMiniTable("price-last", [], []);
      renderMiniTable("financial-table", [], []);
    }
    function closeDetailModal() {
      const overlay = document.getElementById("detail-modal");
      overlay.style.display = "none";
    }
    function clearCanvas(canvasId, message) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      const width = canvas.clientWidth || 300;
      const height = canvas.clientHeight || 180;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      if (message) {
        ctx.fillStyle = "#64748b";
        ctx.font = "12px Arial";
        ctx.fillText(message, 10, height / 2);
      }
    }
    function drawLineChart(canvasId, points, key = "close") {
      const canvas = document.getElementById(canvasId);
      if (!points || !points.length) {
        clearCanvas(canvasId, "暂无数据");
        return;
      }
      const values = points.map(p => p[key]).filter(v => v !== null && v !== undefined && !Number.isNaN(v));
      if (!values.length) { clearCanvas(canvasId, "暂无数据"); return; }
      const width = canvas.clientWidth || 320;
      const height = canvas.clientHeight || 180;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const minVal = Math.min(...values);
      const maxVal = Math.max(...values);
      const span = maxVal - minVal || 1;
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      points.forEach((p, idx) => {
        const val = p[key];
        if (val === null || val === undefined || Number.isNaN(val)) return;
        const x = (idx / Math.max(points.length - 1, 1)) * (width - 8) + 4;
        const y = height - ((val - minVal) / span) * (height - 12) - 6;
        if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
      });
      if (!started) { clearCanvas(canvasId, "暂无数据"); return; }
      ctx.stroke();
      ctx.fillStyle = "rgba(37, 99, 235, 0.14)";
      ctx.lineTo(width - 4, height - 4);
      ctx.lineTo(4, height - 4);
      ctx.closePath();
      ctx.fill();
    }
    function drawBarChart(canvasId, points, key, color) {
      const canvas = document.getElementById(canvasId);
      if (!points || !points.length) { clearCanvas(canvasId, "暂无数据"); return; }
      const filtered = points.filter(p => Number.isFinite(p[key]));
      if (!filtered.length) { clearCanvas(canvasId, "暂无数据"); return; }
      const values = filtered.map(p => p[key]);
      let width = canvas.clientWidth || 320;
      const height = canvas.clientHeight || 180;
      if (!width || width < 40) { width = 480; }
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const maxVal = Math.max(...values);
      const minVal = Math.min(...values);
      const span = (maxVal - minVal) || 1;
      const barWidth = Math.max(4, (width - 12) / filtered.length);
      filtered.forEach((p, idx) => {
        const val = p[key];
        const x = 6 + idx * barWidth;
        const ratioY = (val - minVal) / span;
        const y = height - ratioY * (height - 16) - 6;
        const barHeight = height - y - 6;
        ctx.fillStyle = color || "#22c55e";
        ctx.fillRect(x, y, barWidth * 0.8, Math.max(2, barHeight));
      });
    }
    function renderMiniTable(containerId, cols, rows, formatter = format) {
      const container = document.getElementById(containerId);
      if (!rows || !rows.length || !cols || !cols.length) {
        container.innerHTML = '<div class="meta" style="padding:8px;">暂无数据</div>';
        return;
      }
      let html = "<table><thead><tr><th>日期</th>";
      cols.forEach(col => { html += `<th>${col}</th>`; });
      html += "</tr></thead><tbody>";
      rows.forEach(r => {
        html += `<tr><td>${r.date || ""}</td>`;
        cols.forEach(col => { html += `<td>${formatter(r[col])}</td>`; });
        html += "</tr>";
      });
      html += "</tbody></table>";
      container.innerHTML = html;
    }
    function renderPriceSection(prices) {
      const meta = document.getElementById("price-meta");
      if (!prices || !prices.length) {
        meta.textContent = "无日线数据";
        clearCanvas("price-chart", "暂无日线数据");
        renderMiniTable("price-last", [], []);
        return;
      }
      drawLineChart("price-chart", prices, "close");
      meta.textContent = `覆盖 ${prices[0].date} - ${prices[prices.length - 1].date} · ${prices.length} 天`;
      const recent = prices.slice(-6).reverse();
      renderMiniTable("price-last", ["close", "pct_change", "volume"], recent);
    }
    function renderFinancialSection(financial, columnsList) {
      const meta = document.getElementById("fin-meta");
      if (!financial || !financial.length) {
        meta.textContent = "无财务数据";
        clearCanvas("financial-chart", "暂无财务数据");
        renderMiniTable("financial-table", [], []);
        return;
      }
      const colsAll = (columnsList && columnsList.length ? columnsList : Object.keys(financial[0]).filter(k => k !== "date"));
      // Normalize to numeric where possible for charting
      const normalized = financial.map(row => {
        const next = {...row};
        colsAll.forEach(col => {
          const parsed = parseFloat(row[col]);
          if (!Number.isNaN(parsed) && Number.isFinite(parsed)) {
            next[col] = parsed;
          }
        });
        return next;
      });
      const numericCols = colsAll.filter(col => normalized.some(r => Number.isFinite(r[col])));
      const chartKey = numericCols[0] || colsAll[0];
      const recent = normalized.slice(-Math.min(12, normalized.length));
      meta.textContent = `覆盖 ${financial[0].date} - ${financial[financial.length - 1].date} · 采样 ${financial.length} · 指标 ${colsAll.length} · 图表: ${chartKey || "无"}`;
      if (chartKey) {
        const chartPoints = recent.map(r => {
          const val = Number.isFinite(r[chartKey]) ? r[chartKey] : 0;
          return {...r, [chartKey]: val};
        });
        drawBarChart("financial-chart", chartPoints, chartKey, "#22c55e");
      } else {
        clearCanvas("financial-chart", "无可用数字指标");
      }
      renderMiniTable("financial-table", colsAll, normalized.slice().reverse());
    }
    async function loadDetail(symbol, name) {
      const meta = document.getElementById("detail-meta");
      try {
        const resp = await fetch(`/stock-detail?symbol=${encodeURIComponent(symbol)}`);
        if (!resp.ok) {
          meta.textContent = "加载失败: " + await resp.text();
          renderPriceSection([]);
          renderFinancialSection([], []);
          return;
        }
        const text = await resp.text();
        let payload;
        try {
          payload = JSON.parse(text);
        } catch (e) {
          meta.textContent = "解析失败，可能包含非法数值";
          renderPriceSection([]);
          renderFinancialSection([], []);
          console.error("detail payload parse error", e, text);
          return;
        }
        meta.textContent = `最新样本：日线 ${payload.price_count || 0} 条 · 财务 ${payload.financial_count || 0} 条`;
        renderPriceSection(payload.prices || []);
        renderFinancialSection(payload.financial || [], payload.financial_columns || []);
      } catch (e) {
        meta.textContent = "加载失败: " + e;
        renderPriceSection([]);
        renderFinancialSection([], []);
      }
    }
    function renderFilters() {
      const container = document.getElementById("filters");
      if (!columns.length) { container.innerHTML = ""; return; }
      container.innerHTML = "";
      columns.filter(c => !c.fixed).forEach(col => {
        const label = document.createElement("label");
        label.className = "pill";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = visibleKeys.includes(col.key);
        cb.onchange = () => {
          if (cb.checked) visibleKeys.push(col.key);
          else visibleKeys = visibleKeys.filter(k => k !== col.key);
          renderHeaders();
          renderRows();
        };
        label.appendChild(cb);
        label.append(col.label);
        container.appendChild(label);
      });
    }
    function setColumns(newCols) {
      columns = newCols || [];
      visibleKeys = columns
        .filter(c => !c.fixed && c.defaultVisible !== false)
        .map(c => c.key);
      columnsReady = true;
      renderFilters();
      renderHeaders();
    }
    function renderMarketFilters() {
      const container = document.getElementById("market-filters");
      container.innerHTML = "";
      marketOptions.forEach(opt => {
        const label = document.createElement("label");
        label.className = "pill";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = activeMarkets.has(opt.key);
        cb.onchange = () => {
          if (cb.checked) activeMarkets.add(opt.key);
          else activeMarkets.delete(opt.key);
          renderRows();
          updateStatusCount();
        };
        label.appendChild(cb);
        label.append(opt.label);
        container.appendChild(label);
      });
    }
    function filteredData() {
      return data.filter(row => activeMarkets.has(row.market || "其他"));
    }
    function renderHeaders() {
      const header = document.getElementById("header-row");
      if (!columns.length) { header.innerHTML = ""; return; }
      header.innerHTML = "";
      const active = columns.filter(c => c.fixed || visibleKeys.includes(c.key));
      active.forEach(col => {
        const th = document.createElement("th");
        th.textContent = col.label;
        th.onclick = () => sortBy(col.key);
        header.appendChild(th);
      });
    }
    function renderRows() {
      const body = document.getElementById("body-rows");
      if (!columns.length) { body.innerHTML = ""; return; }
      body.innerHTML = "";
      const active = columns.filter(c => c.fixed || visibleKeys.includes(c.key));
      filteredData().forEach(row => {
        const tr = document.createElement("tr");
        if (selectedSymbol && row.symbol === selectedSymbol) {
          tr.classList.add("active-row");
        }
        active.forEach(col => {
          const td = document.createElement("td");
          const value = row[col.key];
          td.textContent = format(value);
          applySignColor(td, col.key, value);
          tr.appendChild(td);
        });
        tr.onclick = () => {
          selectedSymbol = row.symbol;
          renderRows();
          openDetailModal(row.symbol, row.name);
          loadDetail(row.symbol, row.name);
        };
        body.appendChild(tr);
      });
    }
    function sortBy(key) {
      if (sortKey === key) sortAsc = !sortAsc; else { sortKey = key; sortAsc = true; }
      data.sort((a,b) => {
        const va = a[sortKey]; const vb = b[sortKey];
        if (va === undefined || va === null) return 1;
        if (vb === undefined || vb === null) return -1;
        if (typeof va === "number" && typeof vb === "number") return sortAsc ? va - vb : vb - va;
        return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
      });
      renderRows();
    }
    function updateStatusCount(updatedAt, errorMessage) {
      const status = document.getElementById("status");
      if (errorMessage) {
        status.textContent = "加载失败（使用缓存/空数据）: " + errorMessage;
        status.style.color = "#fca5a5";
        return;
      }
      const filteredLen = filteredData().length;
      status.textContent = `已加载 ${data.length} 只股票 · 筛选后 ${filteredLen} · 更新时间 ${updatedAt || ""}`;
      status.style.color = "#86efac";
    }
    async function loadData() {
      const status = document.getElementById("status");
      status.textContent = "加载中...";
      try {
        const resp = await fetch("/dashboard-data");
        if (!resp.ok) {
          status.textContent = "加载失败: " + await resp.text();
          status.style.color = "#fca5a5";
          return;
        }
        const payload = await resp.json();
        if (payload.columns && payload.columns.length) {
          setColumns(payload.columns);
        }
        data = payload.rows || [];
        updateStatusCount(payload.updated_at, payload.error);
        sortBy(sortKey);
      } catch (e) {
        status.textContent = "加载失败: " + e;
        status.style.color = "#fca5a5";
      }
    }
    function isMarketOpen() {
      const now = new Date();
      const day = now.getDay();
      if (day === 0 || day === 6) return false;
      const minutes = now.getHours() * 60 + now.getMinutes();
      const morning = minutes >= 9 * 60 + 30 && minutes <= 11 * 60 + 30;
      const afternoon = minutes >= 13 * 60 && minutes <= 15 * 60;
      return morning || afternoon;
    }
    function ensureAutoRefresh() {
      const status = document.getElementById("status");
      if (!isMarketOpen()) {
        if (refreshTimer) {
          clearInterval(refreshTimer);
          refreshTimer = null;
        }
        status.textContent = "休市中，自动刷新已暂停";
        status.style.color = "#cbd5e1";
        return;
      }
      if (!refreshTimer) {
        refreshTimer = setInterval(loadData, AUTO_REFRESH_MS);
      }
    }
    renderMarketFilters();
    renderFilters();
    renderHeaders();
    loadData();
    ensureAutoRefresh();
    setInterval(ensureAutoRefresh, MARKET_CHECK_MS);

    async function loadState() {
      try {
        const resp = await fetch('/state');
        if (!resp.ok) return;
        const state = await resp.json();
        ['fetch_all','fetch_financials'].forEach(action => {
          if (state.running && state.running[action]) {
            toggleButtons(action, true);
          }
          if (state.logs && state.logs[action]) {
            startLogStream(action, state.logs[action]);
            const logEl = document.getElementById('log_' + action);
            logEl.textContent = '恢复日志...';
          }
        });
      } catch (e) {
        console.warn('loadState failed', e);
      }
    }
    loadState();

    function toggleButtons(action, running) {
      if (action === 'fetch_all') {
        const startBtn = document.getElementById('btn_start_fetch_all');
        const stopBtn = document.getElementById('btn_stop_fetch_all');
        if (!startBtn || !stopBtn) return;
        startBtn.disabled = running;
        stopBtn.disabled = false; // always clickable per request
        stopBtn.classList.toggle('active', running);
      } else if (action === 'fetch_financials') {
        const startBtn = document.getElementById('btn_start_fin');
        const stopBtn = document.getElementById('btn_stop_fin');
        if (!startBtn || !stopBtn) return;
        startBtn.disabled = running;
        stopBtn.disabled = false; // always clickable
        stopBtn.classList.toggle('active', running);
      } else if (action === 'backtest_transformer') {
        const logEl = document.getElementById('log_backtest_transformer');
        if (logEl && running) {
          logEl.textContent = '回测任务已启动...';
        }
      }
    }
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # keep quiet in console

    def _write_ok(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            return

    def do_GET(self):
        _purge_dead_processes()
        if self.path == "/" or self.path.startswith("/index"):
            self._write_ok(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path.startswith("/state"):
            payload = {"running": RUNNING, "logs": LOG_PATHS, "ts": time.time()}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._write_ok(body, "application/json; charset=utf-8")
            return
        if self.path.startswith("/log"):
            qs = parse_qs(urlparse(self.path).query)
            log_param = qs.get("path", [None])[0]
            if not log_param:
                self.send_error(HTTPStatus.BAD_REQUEST, "missing path")
                return
            log_path = Path(log_param)
            try:
                log_path = log_path.resolve()
            except Exception:
                self.send_error(HTTPStatus.BAD_REQUEST, "bad path")
                return
            if LOG_DIR not in log_path.parents and LOG_DIR != log_path.parent:
                self.send_error(HTTPStatus.BAD_REQUEST, "log path not allowed")
                return
            content = _read_log_tail(log_path)
            body = content.encode("utf-8", errors="replace")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if self.path.startswith("/stats"):
            stats = _collect_stats()
            body = json.dumps(stats, ensure_ascii=False).encode("utf-8")
            self._write_ok(body, "application/json; charset=utf-8")
            return
        if self.path.startswith("/dashboard-data"):
            try:
                snapshot = _dashboard_rows()
                rows = snapshot["rows"]
                cols = snapshot["columns"]
                ts = float(snapshot.get("ts", time.time()))
                payload = {
                    "rows": rows,
                    "columns": cols,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                    "count": len(rows),
                    "source": snapshot.get("source", "realtime"),
                    "error": snapshot.get("error"),
                }
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._write_ok(body, "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                err = f"failed to load dashboard data: {exc}"
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, err)
            return
        if self.path.startswith("/dashboard"):
            self._write_ok(DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path.startswith("/stock-detail"):
            qs = parse_qs(urlparse(self.path).query)
            symbol = (qs.get("symbol") or [""])[0].strip()
            if not symbol:
                self.send_error(HTTPStatus.BAD_REQUEST, "missing symbol")
                return
            try:
                detail = _load_stock_detail(symbol)
                body = json.dumps(detail, ensure_ascii=False).encode("utf-8")
                self._write_ok(body, "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                err = f"failed to load detail for {symbol}: {exc}"
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, err)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        _purge_dead_processes()
        if self.path == "/run":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
                action = data.get("action")
                payload = data.get("payload") or {}
                cmd = _build_cmd(action, payload)
                log_path = _start_process(cmd, action)
                resp_payload = {"status": "started", "cmd": cmd, "log": log_path}
                body = json.dumps(resp_payload, ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:  # noqa: BLE001
                body = f"Error: {exc}".encode("utf-8")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return
        if self.path.startswith("/stop"):
            qs = parse_qs(urlparse(self.path).query)
            action = qs.get("action", [None])[0]
            if not action:
                self.send_error(HTTPStatus.BAD_REQUEST, "missing action")
                return
            msg = _stop_action(action)
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if self.path == "/wipe-data":
            msg = _wipe_data()
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if self.path == "/wipe-price":
            msg = _wipe_price_data()
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if self.path == "/wipe-financial":
            msg = _wipe_financial_data()
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if self.path == "/wipe-feature-cache":
            msg = _wipe_feature_cache()
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        if self.path == "/wipe-dataset-cache":
            msg = _wipe_dataset_cache()
            body = msg.encode("utf-8")
            self._write_ok(body, "text/plain; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")


def run(server_address=("127.0.0.1", 8000)):
    ensure_logs()
    httpd = ThreadingHTTPServer(server_address, Handler)
    print(f"Serving control panel on http://{server_address[0]}:{server_address[1]}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
