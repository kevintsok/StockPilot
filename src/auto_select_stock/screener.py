"""
Natural language stock screener.

Parses Chinese natural language queries, applies criteria against the SQLite database,
and renders a sortable HTML results table.
"""

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from jinja2 import Environment, select_autoescape

from .config import DATA_DIR, REPORT_DIR
from .storage import _connect, list_symbols

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScreenCriteria:
    """Parsed screening criteria from a natural language query."""

    lookback_days: int = 60
    min_pct_change: Optional[float] = None  # e.g. 10.0 means > 10%
    max_pct_change: Optional[float] = None  # e.g. 10.0 means < 10%
    min_roe: Optional[float] = None
    min_eps: Optional[float] = None
    min_turnover_rate: Optional[float] = None
    max_turnover_rate: Optional[float] = None


@dataclass
class ScreenerRow:
    """A single stock matching the screening criteria."""

    symbol: str
    name: str
    lookback_pct_change: float
    last_close: float
    last_date: str
    roe: Optional[float]
    eps: Optional[float]
    turnover_rate: Optional[float]


# ---------------------------------------------------------------------------
# Natural language parser
# ---------------------------------------------------------------------------

# Unit multipliers
_UNIT_MAP = {"亿": 1e8, "万": 1e4}


def _parse_number(text: str) -> Optional[float]:
    """Extract a numeric value, stripping commas / units."""
    text = text.strip()
    for unit, multiplier in _UNIT_MAP.items():
        if unit in text:
            try:
                return float(text.replace(unit, "").replace(",", "").strip()) * multiplier
            except ValueError:
                return None
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def parse_nl_query(text: str) -> ScreenCriteria:
    """
    Parse a Chinese natural language query into a ScreenCriteria object.

    Supported patterns:
      - 时间窗口:  "3个月内"  "60日"  "1年内"
      - 涨幅上限:  "涨幅不超过+10%"  "涨幅小于10%"  "涨幅低于10%"
      - 涨幅下限:  "涨幅超过20%"  "涨幅大于20%"  "涨幅高于20%"
      - ROE 下限:  "ROE高于10%"  "ROE大于10%"
      - EPS 下限:  "EPS高于0.5元"  "EPS大于0.5"
      - 换手率上限:"换手率低于5%"  "换手率小于5%"
      - 换手率下限:"换手率高于2%"  "换手率大于2%"

    Default lookback is 60 days.
    """
    criteria = ScreenCriteria()

    text = text.replace(" ", "").replace("，", ",").replace("。", ".")

    # --- Time window ---
    m = re.search(r"(\d+)\s*个月\s*内", text)
    if m:
        criteria.lookback_days = int(m.group(1)) * 30
    else:
        m = re.search(r"(\d+)\s*日", text)
        if m:
            criteria.lookback_days = int(m.group(1))
        else:
            m = re.search(r"(\d+)\s*年\s*内", text)
            if m:
                criteria.lookback_days = int(m.group(1)) * 365

    # --- Price change: min (greater than) ---
    for pattern in [
        r"涨幅\s*(?:超过|大于|高于)\s*([-+]?[\d.]+)%?",
        r"涨幅\s*(?:大于|超过|高于)\s*\+?([\d.]+)%?",
    ]:
        m = re.search(pattern, text)
        if m:
            criteria.min_pct_change = float(m.group(1))
            break

    # --- Price change: max (less than) ---
    for pattern in [
        r"涨幅\s*(?:不超过|小于|低于|不多于)\s*([-+]?[\d.]+)%?",
        r"涨幅\s*(?:小于|低于|不多于)\s*\+?([\d.]+)%?",
    ]:
        m = re.search(pattern, text)
        if m:
            criteria.max_pct_change = float(m.group(1))
            break

    # --- ROE min ---
    m = re.search(r"ROE\s*(?:高于|大于|超过)\s*([\d.]+)%?", text)
    if m:
        criteria.min_roe = float(m.group(1))

    # --- EPS min ---
    m = re.search(r"EPS\s*(?:高于|大于|超过)\s*([\d.]+)", text)
    if m:
        criteria.min_eps = float(m.group(1))

    # --- Turnover rate max ---
    m = re.search(r"换手率\s*(?:低于|小于|不超过)\s*([\d.]+)%?", text)
    if m:
        criteria.max_turnover_rate = float(m.group(1))

    # --- Turnover rate min ---
    m = re.search(r"换手率\s*(?:高于|大于|超过)\s*([\d.]+)%?", text)
    if m:
        criteria.min_turnover_rate = float(m.group(1))

    return criteria


# ---------------------------------------------------------------------------
# Stock name cache (akshare, called once per session)
# ---------------------------------------------------------------------------

_NAME_CACHE: Dict[str, str] = {}
_NAME_CACHE_TIME: float = 0
_NAME_CACHE_TTL: float = 300  # 5 minutes


def _fetch_stock_names_from_akshare() -> Dict[str, str]:
    """Batch-fetch stock names via akshare (session-cached)."""
    global _NAME_CACHE, _NAME_CACHE_TIME
    if _NAME_CACHE and (time.time() - _NAME_CACHE_TIME) < _NAME_CACHE_TTL:
        return _NAME_CACHE
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        if "代码" in df.columns and "名称" in df.columns:
            _NAME_CACHE = dict(zip(df["代码"].astype(str), df["名称"].astype(str)))
        elif "symbol" in df.columns and "name" in df.columns:
            _NAME_CACHE = dict(zip(df["symbol"].astype(str), df["name"].astype(str)))
        _NAME_CACHE_TIME = time.time()
    except Exception:
        _NAME_CACHE = {}
    return _NAME_CACHE


# ---------------------------------------------------------------------------
# Core screening logic
# ---------------------------------------------------------------------------

_A_SHARE_PREFIXES = ("0", "3", "6")
_EXCLUDE_PREFIXES = ("688",)


def _filter_a_share_symbols(symbols: List[str]) -> List[str]:
    return [
        s
        for s in symbols
        if not s.startswith(_EXCLUDE_PREFIXES) and s.startswith(_A_SHARE_PREFIXES)
    ]


def screen_stocks(criteria: ScreenCriteria, base_dir: Optional[Path] = None) -> List[ScreenerRow]:
    """
    Apply screening criteria against the SQLite database.

    Returns a list of ScreenerRow sorted by lookback_pct_change descending.
    """
    conn = _connect(base_dir, read_only=True)
    names = _fetch_stock_names_from_akshare()

    symbols = _filter_a_share_symbols(list_symbols(base_dir=base_dir))
    if not symbols:
        return []

    # Simple approach: load all price data into pandas, compute lookback pct change in memory.
    # This avoids complex SQL with many OR conditions.
    from datetime import datetime, timedelta

    df_price_all = pd.read_sql_query(
        "SELECT symbol, date, close, turnover_rate FROM price ORDER BY symbol, date",
        conn,
    )
    if df_price_all.empty:
        return []

    df_price_all["date"] = pd.to_datetime(df_price_all["date"])

    # Latest row per symbol
    df_latest = df_price_all.sort_values("date").groupby("symbol").last().reset_index()
    df_latest.columns = ["symbol", "last_date", "last_close", "turnover_rate"]

    # Lookback: for each symbol, find the row closest to (latest_date - lookback_days)
    lookback_dates = {}
    for sym, grp in df_price_all.groupby("symbol"):
        latest_d = grp["date"].max()
        target_d = latest_d - timedelta(days=criteria.lookback_days)
        # Find the row with date <= target_d (last known price before/at target)
        candidates = grp[grp["date"] <= target_d]
        if not candidates.empty:
            closest = candidates.loc[candidates["date"].idxmax()]
            lookback_dates[sym] = (float(closest["close"]), str(closest["date"].date()))
        else:
            lookback_dates[sym] = (None, None)

    df_lb = pd.DataFrame(
        [(s, v[0], v[1]) for s, v in lookback_dates.items()],
        columns=["symbol", "close_lb", "date_lb"],
    )

    df_price = df_latest.merge(df_lb, on="symbol", how="left")

    # Financial data: load all and get latest per symbol
    df_fin_all = pd.read_sql_query(
        "SELECT symbol, date, roe, eps FROM financial ORDER BY symbol, date",
        conn,
    )
    if not df_fin_all.empty:
        df_fin_all["date"] = pd.to_datetime(df_fin_all["date"])
        df_fin = df_fin_all.sort_values("date").groupby("symbol").last().reset_index()
        df_fin = df_fin[["symbol", "roe", "eps"]]
    else:
        df_fin = pd.DataFrame(columns=["symbol", "roe", "eps"])

    conn.close()

    # Merge
    df = df_price.merge(df_fin, on="symbol", how="left")

    # Compute lookback pct change - ensure numeric types
    df["close_lb"] = pd.to_numeric(df["close_lb"], errors="coerce")
    df["last_close"] = pd.to_numeric(df["last_close"], errors="coerce")
    df["lookback_pct_change"] = (df["last_close"] - df["close_lb"]) / df["close_lb"] * 100
    df["lookback_pct_change"] = df["lookback_pct_change"].round(2)
    df["last_close"] = df["last_close"].round(2)
    df["name"] = df["symbol"].map(lambda s: names.get(s, ""))

    # Apply filters
    mask = pd.Series(True, index=df.index)

    if criteria.min_pct_change is not None:
        mask &= df["lookback_pct_change"] > criteria.min_pct_change

    if criteria.max_pct_change is not None:
        mask &= df["lookback_pct_change"] < criteria.max_pct_change

    if criteria.min_roe is not None:
        mask &= df["roe"].fillna(-999) > criteria.min_roe

    if criteria.min_eps is not None:
        mask &= df["eps"].fillna(-999) > criteria.min_eps

    if criteria.min_turnover_rate is not None:
        mask &= df["turnover_rate"].fillna(-999) > criteria.min_turnover_rate

    if criteria.max_turnover_rate is not None:
        mask &= df["turnover_rate"].fillna(999) < criteria.max_turnover_rate

    df = df[mask].sort_values("lookback_pct_change", ascending=False)

    rows = []
    for _, r in df.iterrows():
        rows.append(
            ScreenerRow(
                symbol=str(r["symbol"]),
                name=r["name"],
                lookback_pct_change=float(r["lookback_pct_change"]),
                last_close=float(r["last_close"]),
                last_date=str(r["last_date"]),
                roe=float(r["roe"]) if pd.notna(r.get("roe")) else None,
                eps=float(r["eps"]) if pd.notna(r.get("eps")) else None,
                turnover_rate=float(r["turnover_rate"]) if pd.notna(r.get("turnover_rate")) else None,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

SORTABLE_TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>选股结果 - {{ criteria_str }}</title>
  <style>
    :root { --bg:#0e1116; --card:#161b22; --text:#e6edf3; --accent:#58a6ff; --dim:#8b949e; }
    body { background: var(--bg); color: var(--text); font-family: "Segoe UI", Arial, sans-serif; margin:0; padding:24px; }
    h1 { margin:0 0 8px; font-size:20px; }
    p.meta { color: var(--dim); margin:0 0 20px; font-size:13px; }
    table { width:100%; border-collapse:collapse; font-size:14px; }
    th { text-align:left; padding:10px 12px; background:#1c2128; cursor:pointer; user-select:none; white-space:nowrap; }
    th:hover { background:#2d333b; }
    th .arrow { color: var(--accent); margin-left:4px; }
    td { padding:8px 12px; border-bottom:1px solid #21262d; white-space:nowrap; }
    tr:hover td { background:#161b22; }
    .num { text-align:right; font-variant-numeric:tabular-nums; }
    .tag { display:inline-block; padding:2px 8px; border-radius:4px; font-size:12px; }
    .tag-up { background:#1f3830; color:#3fb950; }
    .tag-down { background:#3d1f1f; color:#f85149; }
    .tag-none { background:#21262d; color: var(--dim); }
    th.sorted { color: var(--accent); }
  </style>
</head>
<body>
  <h1>选股结果</h1>
  <p class="meta">条件: {{ criteria_str }} &nbsp;|&nbsp; 符合条件: {{ rows|length }} 只</p>
  <table id="tbl">
    <thead>
      <tr>
        <th data-key="symbol" onclick="sort('symbol')">股票代码 <span class="arrow"></span></th>
        <th data-key="name" onclick="sort('name')">名字 <span class="arrow"></span></th>
        <th data-key="lookback_pct_change" onclick="sort('lookback_pct_change')" class="sorted">N天内涨幅(%) <span class="arrow">▼</span></th>
        <th data-key="last_close" onclick="sort('last_close')">最新收盘价 <span class="arrow"></span></th>
        <th data-key="roe" onclick="sort('roe')">ROE(%) <span class="arrow"></span></th>
        <th data-key="eps" onclick="sort('eps')">EPS(元) <span class="arrow"></span></th>
        <th data-key="turnover_rate" onclick="sort('turnover_rate')">换手率(%) <span class="arrow"></span></th>
        <th data-key="last_date" onclick="sort('last_date')">最新日期 <span class="arrow"></span></th>
      </tr>
    </thead>
    <tbody>
    {% for row in rows %}
      <tr>
        <td>{{ row.symbol }}</td>
        <td>{{ row.name }}</td>
        <td class="num">{{ "%.2f"|format(row.lookback_pct_change) }}</td>
        <td class="num">{{ "%.2f"|format(row.last_close) }}</td>
        <td class="num">{{ "%.2f"|format(row.roe) if row.roe is not none else '-' }}</td>
        <td class="num">{{ "%.3f"|format(row.eps) if row.eps is not none else '-' }}</td>
        <td class="num">{{ "%.2f"|format(row.turnover_rate) if row.turnover_rate is not none else '-' }}</td>
        <td>{{ row.last_date }}</td>
      </tr>
    {% else %}
      <tr><td colspan="8" style="text-align:center;color:var(--dim);padding:32px;">暂无符合条件的数据</td></tr>
    {% endfor %}
    </tbody>
  </table>
  <script>
    var dir = -1;
    function sort(key) {
      dir *= -1;
      var tbl = document.getElementById('tbl');
      var rows = Array.from(tbl.tBodies[0].rows);
      var col = document.querySelector('th[data-key="' + key + '"]');
      document.querySelectorAll('th').forEach(function(h){ h.classList.remove('sorted'); h.querySelector('.arrow').textContent = ''; });
      col.classList.add('sorted');
      col.querySelector('.arrow').textContent = dir > 0 ? '▲' : '▼';
      rows.sort(function(a, b) {
        var A = a.children[Array.from(tbl.tHead.rows[0].cells).indexOf(col)].textContent.trim();
        var B = b.children[Array.from(tbl.tHead.rows[0].cells).indexOf(col)].textContent.trim();
        var NA = A.replace(/[^\\d.\\-]/g, '');
        var NB = B.replace(/[^\\d.\\-]/g, '');
        if (NA === '' && NB === '') return A > B ? dir : -dir;
        if (NA === '') return 1;
        if (NB === '') return -1;
        return (parseFloat(NA) - parseFloat(NB)) * dir;
      });
      var tbody = tbl.tBodies[0];
      rows.forEach(function(r){ tbody.appendChild(r); });
    }
  </script>
</body>
</html>
"""


def render_screener_html(
    rows: List[ScreenerRow],
    criteria: ScreenCriteria,
    output_path: Path,
) -> Path:
    """Render a sortable HTML table to output_path."""
    criteria_str = _criteria_to_str(criteria)
    env = Environment(loader=None, autoescape=select_autoescape(["html"]))
    template = env.from_string(SORTABLE_TEMPLATE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = template.render(rows=rows, criteria_str=criteria_str)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def _criteria_to_str(criteria: ScreenCriteria) -> str:
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
