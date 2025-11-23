import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import DATA_DIR, REPORT_DIR


@dataclass
class StockRow:
    symbol: str
    market: str
    last_date: str
    price: float
    turnover_rate: float
    pct_change: float
    volume_ratio: Optional[float]
    amplitude: Optional[float]
    change_amount: Optional[float]
    chg_20d: Optional[float]
    chg_60d: Optional[float]
    pe: Optional[float]
    roe: Optional[float]
    net_profit_margin: Optional[float]
    gross_margin: Optional[float]
    ocf_growth: Optional[float]
    debt_to_asset: Optional[float]


def _classify_market(symbol: str) -> str:
    """Lightweight market classifier based on代码前缀."""
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


def _price_snapshot(symbols: Iterable[str], lookbacks: Iterable[int], base_dir: Path = DATA_DIR) -> Dict[str, Dict[str, object]]:
    """
    Fetch the latest price row and the lookback rows (e.g. 20d/60d ago) in a single
    SQL query to avoid loading thousands of files/arrays one by one.
    """
    db_path = (base_dir or DATA_DIR) / "stock.db"
    if not db_path.exists():
        return {}

    symbol_list = list(symbols)
    lookback_list = sorted({int(lb) for lb in lookbacks if int(lb) > 0})

    # SQLite has a 999-parameter limit; avoid building a massive IN clause if the user
    # asked for "all symbols".
    params: List[object] = []
    symbol_clause = ""
    if symbol_list and len(symbol_list) <= 900:
        symbol_clause = f"WHERE symbol IN ({','.join(['?'] * len(symbol_list))})"
        params.extend(symbol_list)

    lookback_columns = []
    for lb in lookback_list:
        offset = max(lb - 1, 0)
        lookback_columns.append(
            f", (SELECT close FROM price p{lb} "
            f"WHERE p{lb}.symbol = latest.symbol AND p{lb}.date <= latest.date "
            f"ORDER BY date DESC LIMIT 1 OFFSET {offset}) AS close_{lb}"
        )
    lookback_sql = "\n".join(lookback_columns)

    sql = f"""
    WITH latest AS (
      SELECT p.symbol, p.date, p.close, p.turnover_rate, p.volume_ratio, p.amplitude, p.change_amount, p.pct_change
      FROM price p
      JOIN (
        SELECT symbol, MAX(date) AS max_date
        FROM price
        {symbol_clause}
        GROUP BY symbol
      ) m ON p.symbol = m.symbol AND p.date = m.max_date
    )
    SELECT symbol, date, close, turnover_rate, volume_ratio, amplitude, change_amount, pct_change
    {lookback_sql}
    FROM latest
    """

    snapshot: Dict[str, Dict[str, object]] = {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    for row in rows:
        sym = str(row["symbol"])
        entry = snapshot.setdefault(sym, {"last": None, "back": {}})
        entry["last"] = {
            "date": row["date"],
            "close": row["close"],
            "turnover_rate": row["turnover_rate"],
            "volume_ratio": row["volume_ratio"],
            "amplitude": row["amplitude"],
            "change_amount": row["change_amount"],
            "pct_change": row["pct_change"],
        }
        for lb in lookback_list:
            key = f"close_{lb}"
            entry["back"][lb] = row[key] if key in row.keys() else None
    return snapshot


def _latest_financial_snapshot(symbols: Iterable[str], base_dir: Path = DATA_DIR) -> Dict[str, Dict[str, object]]:
    """
    Grab the latest financial row for each symbol in one query. Only the columns needed
    by StockRow are selected to keep the payload small.
    """
    db_path = (base_dir or DATA_DIR) / "stock.db"
    if not db_path.exists():
        return {}

    symbol_list = list(symbols)
    params: List[object] = []
    symbol_clause = ""
    if symbol_list and len(symbol_list) <= 900:
        symbol_clause = f"WHERE symbol IN ({','.join(['?'] * len(symbol_list))})"
        params.extend(symbol_list)

    fields = ["roe", "net_profit_margin", "gross_margin", "operating_cashflow_growth", "debt_to_asset", "eps"]
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(financial)")}
        selected_cols = [c for c in fields if c in existing_cols]
        if not selected_cols:
            return {}
        col_sql = ",".join(selected_cols)
        sql = f"""
        WITH ranked AS (
          SELECT symbol, date, {col_sql},
                 ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
          FROM financial
          {symbol_clause}
        )
        SELECT symbol, date, {col_sql}
        FROM ranked
        WHERE rn = 1
        """
        rows = conn.execute(sql, params).fetchall()

    fin_map: Dict[str, Dict[str, object]] = {}
    for row in rows:
        fin_map[str(row["symbol"])] = {key: row[key] for key in row.keys() if key != "symbol"}
    return fin_map


def build_rows(symbols: Iterable[str], lookbacks: Iterable[int], base_dir: Path = DATA_DIR) -> List[StockRow]:
    symbol_list = list(symbols)
    price_snapshot = _price_snapshot(symbol_list, lookbacks, base_dir=base_dir)
    # When symbols is empty (caller wants "all"), use the ones present in price_snapshot.
    if not symbol_list:
        symbol_list = list(price_snapshot.keys())
    financials = _latest_financial_snapshot(symbol_list, base_dir=base_dir)

    lookback_list = sorted({int(lb) for lb in lookbacks if int(lb) > 0})
    rows: List[StockRow] = []
    for sym, info in price_snapshot.items():
        last = info.get("last") or {}
        if last.get("close") is None:
            continue
        try:
            price = float(last["close"])
        except (TypeError, ValueError):
            continue

        market = _classify_market(sym)
        try:
            turnover_rate = float(last.get("turnover_rate")) if last.get("turnover_rate") is not None else 0.0
        except (TypeError, ValueError):
            turnover_rate = 0.0
        try:
            volume_ratio = float(last.get("volume_ratio")) if last.get("volume_ratio") is not None else None
        except (TypeError, ValueError):
            volume_ratio = None
        try:
            pct_change = float(last.get("pct_change")) if last.get("pct_change") is not None else 0.0
        except (TypeError, ValueError):
            pct_change = 0.0
        try:
            amplitude = float(last.get("amplitude")) if last.get("amplitude") is not None else None
        except (TypeError, ValueError):
            amplitude = None
        try:
            change_amount = float(last.get("change_amount")) if last.get("change_amount") is not None else None
        except (TypeError, ValueError):
            change_amount = None

        chg_values: Dict[str, Optional[float]] = {}
        back_prices: Dict[int, object] = info.get("back", {})
        for lb in lookback_list:
            base_price = back_prices.get(lb)
            key = f"chg_{lb}d"
            if base_price in (None, 0):
                chg_values[key] = None
                continue
            try:
                chg_values[key] = (price / float(base_price) - 1.0) * 100.0
            except Exception:
                chg_values[key] = None

        fin = financials.get(sym, {})
        eps = fin.get("eps") or 0.0
        pe = price / eps if eps else None

        last_date = str(last.get("date") or "")
        rows.append(
            StockRow(
                symbol=sym,
                market=market,
                last_date=last_date,
                price=price,
                turnover_rate=turnover_rate,
                pct_change=pct_change,
                volume_ratio=volume_ratio,
                amplitude=amplitude,
                change_amount=change_amount,
                chg_20d=chg_values.get("chg_20d"),
                chg_60d=chg_values.get("chg_60d"),
                pe=pe,
                roe=fin.get("roe"),
                net_profit_margin=fin.get("net_profit_margin"),
                gross_margin=fin.get("gross_margin"),
                ocf_growth=fin.get("operating_cashflow_growth"),
                debt_to_asset=fin.get("debt_to_asset"),
            )
        )
    return rows


def render_dashboard(rows: List[StockRow], output: Optional[Path] = None) -> Path:
    output_path = output or REPORT_DIR / "dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [row.__dict__ for row in rows]
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>A股价格与财报看板</title>
  <style>
    body {{ font-family: "Arial", sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:24px; }}
    h1 {{ margin:0 0 12px; }}
    .card {{ background:#111827; border:1px solid #1f2937; border-radius:12px; padding:16px; box-shadow:0 10px 40px rgba(0,0,0,0.45); }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
    th, td {{ padding:10px; border-bottom:1px solid #1f2937; text-align:right; }}
    th {{ cursor:pointer; color:#93c5fd; position:sticky; top:0; background:#0b1221; }}
    th:first-child, td:first-child {{ text-align:left; }}
    tr:hover {{ background:rgba(59,130,246,0.08); }}
    .meta {{ color:#94a3b8; font-size:13px; }}
    .badge {{ padding:4px 8px; border-radius:8px; background:#1d4ed8; color:#e0f2fe; font-size:12px; }}
    .pos {{ color:#ef4444; font-weight:600; }}
    .neg {{ color:#22c55e; font-weight:600; }}
    .zero {{ color:#e2e8f0; }}
  </style>
</head>
<body>
  <h1>A股价格与财报看板 <span class="meta">点击表头排序</span></h1>
  <div class="card">
    <table id="stock-table">
      <thead>
        <tr id="header-row"></tr>
      </thead>
      <tbody id="body-rows"></tbody>
    </table>
  </div>
  <script>
    const columns = [
      {{key:"symbol", label:"代码"}},
      {{key:"market", label:"板块"}},
      {{key:"last_date", label:"最新日期"}},
      {{key:"price", label:"收盘价"}},
      {{key:"volume_ratio", label:"量比"}},
      {{key:"amplitude", label:"振幅%"}},
      {{key:"change_amount", label:"涨跌额"}},
      {{key:"turnover_rate", label:"换手率%"}},
      {{key:"pct_change", label:"当日涨跌幅%"}},
      {{key:"chg_20d", label:"20日涨跌幅%"}},
      {{key:"chg_60d", label:"60日涨跌幅%"}},
      {{key:"pe", label:"市盈率(估)"}},
      {{key:"roe", label:"ROE%"}},
      {{key:"net_profit_margin", label:"净利率%"}},
      {{key:"gross_margin", label:"毛利率%"}},
      {{key:"ocf_growth", label:"经营现金流增速%"}},
      {{key:"debt_to_asset", label:"资产负债率%"}},
    ];
    const signedColumns = ["pct_change","chg_20d","chg_60d","roe","net_profit_margin","gross_margin","ocf_growth","debt_to_asset"];
    let data = {json.dumps(data, ensure_ascii=False)};
    let sortKey = "symbol";
    let sortAsc = true;

    function format(val) {{
      if (val === null || val === undefined || Number.isNaN(val)) return "";
      if (typeof val === "number") return val.toFixed(2);
      return val;
    }}

    function applySignColor(td, key, value) {{
      if (!signedColumns.includes(key) || typeof value !== "number" || Number.isNaN(value)) return;
      if (value > 0) {{
        td.classList.add("pos");
      }} else if (value < 0) {{
        td.classList.add("neg");
      }} else {{
        td.classList.add("zero");
      }}
    }}

    function renderHeaders() {{
      const header = document.getElementById("header-row");
      header.innerHTML = "";
      columns.forEach(col => {{
        const th = document.createElement("th");
        th.textContent = col.label;
        th.onclick = () => sortBy(col.key);
        header.appendChild(th);
      }});
    }}

    function renderRows() {{
      const body = document.getElementById("body-rows");
      body.innerHTML = "";
      data.forEach(row => {{
        const tr = document.createElement("tr");
        columns.forEach(col => {{
          const td = document.createElement("td");
          const value = row[col.key];
          td.textContent = format(value);
          applySignColor(td, col.key, value);
          tr.appendChild(td);
        }});
        body.appendChild(tr);
      }});
    }}

    function sortBy(key) {{
      if (sortKey === key) {{
        sortAsc = !sortAsc;
      }} else {{
        sortKey = key; sortAsc = true;
      }}
      data.sort((a,b) => {{
        const va = a[sortKey]; const vb = b[sortKey];
        if (va === undefined || va === null) return 1;
        if (vb === undefined || vb === null) return -1;
        if (typeof va === "number" && typeof vb === "number") {{
          return sortAsc ? va - vb : vb - va;
        }}
        return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
      }});
      renderRows();
    }}

    renderHeaders();
    sortBy("symbol");
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
