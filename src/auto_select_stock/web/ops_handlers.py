"""
HTTP request handlers for ops_dashboard.

Contains all do_GET and do_POST methods. Loaded by ops_dashboard server.
"""

import json
import math
import os
import shutil
import signal
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from subprocess import Popen
from typing import Dict, List, Optional

import akshare as ak
import pandas as pd
import requests

from ..config import DATA_DIR, PREPROCESSED_DIR, PROJECT_ROOT
from .dashboard import build_rows
from ..data.fetcher import list_all_symbols
from .screener import ScreenCriteria, parse_nl_query, screen_stocks
from ..data.storage import DB_PATH, list_symbols, load_financial, load_stock_history

# Token for authentication (from env var)
OPS_DASHBOARD_TOKEN = os.getenv("OPS_DASHBOARD_TOKEN", "")

# Template directory
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

LOG_DIR = PROJECT_ROOT / "logs"
SRC_DIR = PROJECT_ROOT / "src"
RUNNING: Dict[str, int] = {}
RUNNING_PROCS: Dict[str, Popen] = {}
LOG_HANDLES: Dict[str, object] = {}
LOG_PATHS: Dict[str, str] = {}
STATS_CACHE: Dict[str, object] = {"ts": 0.0, "data": None}
NAME_CACHE: Dict[str, object] = {"ts": 0.0, "map": {}}
DASH_CACHE: Dict[str, object] = {"ts": 0.0, "data": None, "cols": None}
REALTIME_CACHE: Dict[str, object] = {"ts": 0.0, "rows": None, "cols": None, "err": None}
REALTIME_CACHE_FILE = DATA_DIR / "realtime_cache.json"
SCREENER_CACHE: Dict[str, object] = {"ts": 0.0, "query": "", "criteria": None, "rows": None}


def _load_template(name: str) -> str:
    """Load HTML template from templates directory."""
    path = TEMPLATE_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    # Fallback: return simple error page
    return "<html><body><h1>Template not found</h1></body></html>"


def _criteria_to_str(criteria: ScreenCriteria) -> str:
    """Convert ScreenCriteria to human-readable string."""
    parts = [f"近{criteria.lookback_days}天"]
    if criteria.min_pct_change is not None:
        parts.append(f"涨幅>{criteria.min_pct_change}%")
    if criteria.max_pct_change is not None:
        parts.append(f"涨幅<{criteria.max_pct_change}%")
    if criteria.min_roe is not None:
        parts.append(f"ROE>{criteria.min_roe}%")
    if criteria.min_eps is not None:
        parts.append(f"EPS>{criteria.min_eps}")
    if criteria.min_turnover_rate is not None:
        parts.append(f"换手率>{criteria.min_turnover_rate}%")
    if criteria.max_turnover_rate is not None:
        parts.append(f"换手率<{criteria.max_turnover_rate}%")
    return " | ".join(parts) or "全部股票"


def _parse_criteria(query: str) -> ScreenCriteria:
    """Parse natural language query to ScreenCriteria using LLM or regex fallback."""
    try:
        from .llm.nl_parser import parse_nl_query_with_llm
        from .llm.openai_client import OpenAIClient

        llm_client = OpenAIClient(provider="minimax")
        return parse_nl_query_with_llm(query, llm_client)
    except Exception:
        return parse_nl_query(query)


def _screener_rows(query: str) -> Dict[str, object]:
    """Run screener query with 5-minute caching."""
    now = time.time()
    cached = SCREENER_CACHE
    if cached.get("query") == query and cached.get("rows") is not None:
        if now - float(cached.get("ts", 0.0)) < 300:
            return {"rows": cached["rows"], "criteria": cached["criteria"]}
    criteria = _parse_criteria(query)
    rows = screen_stocks(criteria, base_dir=DATA_DIR)
    SCREENER_CACHE.update({"ts": now, "query": query, "criteria": criteria, "rows": rows})
    return {"rows": rows, "criteria": criteria}


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
    import sys
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
            "830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
            "870", "871", "872", "873", "874", "875", "876", "877", "878", "879",
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
