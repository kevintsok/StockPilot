import sqlite3
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from ..config import DATA_DIR

DB_PATH = DATA_DIR / "stock.db"


def ensure_data_dir(path: Optional[Path] = None) -> Path:
    target = path or DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _connect(base_dir: Optional[Path] = None, read_only: bool = False) -> sqlite3.Connection:
    db_path = (base_dir or DATA_DIR) / "stock.db"
    if read_only:
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found at {db_path}")
        uri = f"file:{db_path}?mode=ro&immutable=1"
        return sqlite3.connect(uri, uri=True)
    ensure_data_dir(base_dir)
    conn = sqlite3.connect(db_path, timeout=30)
    # journal_mode=DELETE: SQLite uses a rollback journal with exclusive locking.
    # WAL (Write-Ahead Logging) is normally faster but on WSL/Windows shares it
    # requires file sharing to be enabled and can cause "database is locked" errors
    # when transitioning between environments. DELETE avoids this at the cost of
    # slightly higher write contention in concurrent scenarios.
    conn.execute("PRAGMA journal_mode=DELETE;")
    conn.execute("PRAGMA busy_timeout = 30000;")  # Wait up to 30s for locks
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL, amount REAL, turnover_rate REAL, volume_ratio REAL, pct_change REAL,
            amplitude REAL, change_amount REAL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_symbol_date ON price(symbol,date)")
    # hfq (后复权) prices for training continuity — same schema as price
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_hfq (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL, amount REAL, turnover_rate REAL, volume_ratio REAL, pct_change REAL,
            amplitude REAL, change_amount REAL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_hfq_symbol_date ON price_hfq(symbol,date)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS financial (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            roe REAL, net_profit_margin REAL, gross_margin REAL,
            operating_cashflow_growth REAL, debt_to_asset REAL,
            eps REAL, operating_cashflow_per_share REAL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fin_symbol_date ON financial(symbol,date)")
    # 主力资金流 table (from stock_individual_fund_flow)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fund_flow (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            pct_change REAL,
            main_net_inflow REAL,
            main_net_pct REAL,
            super_net_inflow REAL,
            super_net_pct REAL,
            big_net_inflow REAL,
            big_net_pct REAL,
            mid_net_inflow REAL,
            mid_net_pct REAL,
            small_net_inflow REAL,
            small_net_pct REAL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fund_flow_symbol_date ON fund_flow(symbol,date)")
    # Chip distribution table (from stock_cyq_em)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chip (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            profit_ratio REAL,
            avg_cost REAL,
            c90_low REAL, c90_high REAL, c90集中度 REAL,
            c70_low REAL, c70_high REAL, c70集中度 REAL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chip_symbol_date ON chip(symbol,date)")
    # Backfill new columns if table already existed
    for column_def in ["amplitude REAL", "change_amount REAL"]:
        try:
            conn.execute(f"ALTER TABLE price ADD COLUMN {column_def}")
        except sqlite3.OperationalError as exc:
            # Ignore if column already exists
            if "duplicate column name" not in str(exc).lower():
                raise
    conn.commit()


def _ensure_financial_columns(conn: sqlite3.Connection, columns: list[str]) -> None:
    """Add new financial columns if missing (stored as REAL)."""
    cur = conn.execute("PRAGMA table_info(financial)")
    existing = {row[1] for row in cur.fetchall()}
    for col in columns:
        if col in {"symbol", "date"} or col in existing:
            continue
        col_type = "TEXT" if col == "publish_date" or col.lower().endswith("date") else "REAL"
        try:
            escaped = col.replace('"', '""')
            conn.execute(f'ALTER TABLE financial ADD COLUMN "{escaped}" {col_type}')
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
    conn.commit()


def save_stock_history(symbol: str, structured_array: np.ndarray, base_dir: Optional[Path] = None, table: str = "price") -> Path:
    """Save price history to the specified table (default: 'price' = qfq 前复权)."""
    conn = _connect(base_dir)
    df = pd.DataFrame(structured_array)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    rows = df.to_dict("records")
    with conn:
        conn.executemany(
            f"""
            INSERT INTO {table}(symbol,date,open,high,low,close,volume,amount,turnover_rate,volume_ratio,pct_change,amplitude,change_amount)
            VALUES(:symbol,:date,:open,:high,:low,:close,:volume,:amount,:turnover_rate,:volume_ratio,:pct_change,:amplitude,:change_amount)
            ON CONFLICT(symbol,date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                amount=excluded.amount,
                turnover_rate=excluded.turnover_rate,
                volume_ratio=excluded.volume_ratio,
                pct_change=excluded.pct_change,
                amplitude=excluded.amplitude,
                change_amount=excluded.change_amount
            """,
            [{**r, "symbol": symbol} for r in rows],
        )
    return (base_dir or DATA_DIR) / "stock.db"


def load_stock_history(symbol: str, base_dir: Optional[Path] = None, table: str = "price") -> np.ndarray:
    """Load price history from the specified table (default: 'price' = qfq 前复权)."""
    conn = _connect(base_dir, read_only=True)
    df = pd.read_sql_query(
        f"SELECT date, open, high, low, close, volume, amount, turnover_rate, volume_ratio, pct_change, amplitude, change_amount "
        f"FROM {table} WHERE symbol=? ORDER BY date ASC",
        conn,
        params=(symbol,),
    )
    if df.empty:
        raise FileNotFoundError(f"No price data for {symbol} in table {table}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    dtype = [
        ("date", "datetime64[D]"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("volume", "f8"),
        ("amount", "f8"),
        ("turnover_rate", "f8"),
        ("volume_ratio", "f8"),
        ("pct_change", "f8"),
        ("amplitude", "f8"),
        ("change_amount", "f8"),
    ]
    arr = np.zeros(len(df), dtype=dtype)
    for name in df.columns:
        arr[name] = df[name].to_numpy()
    return arr


# Convenience aliases for clarity
def save_stock_history_qfq(symbol: str, arr: np.ndarray, base_dir: Optional[Path] = None) -> Path:
    """Save qfq (前复权) price data — default table for viewing / P&L."""
    return save_stock_history(symbol, arr, base_dir=base_dir, table="price")


def save_stock_history_hfq(symbol: str, arr: np.ndarray, base_dir: Optional[Path] = None) -> Path:
    """Save hfq (后复权) price data — for model training continuity."""
    return save_stock_history(symbol, arr, base_dir=base_dir, table="price_hfq")


def load_stock_history_qfq(symbol: str, base_dir: Optional[Path] = None) -> np.ndarray:
    """Load qfq (前复权) price data — default for current prices / P&L."""
    return load_stock_history(symbol, base_dir=base_dir, table="price")


def load_stock_history_hfq(symbol: str, base_dir: Optional[Path] = None) -> np.ndarray:
    """Load hfq (后复权) price data — for model training."""
    return load_stock_history(symbol, base_dir=base_dir, table="price_hfq")


def price_date_range(symbol: str, base_dir: Optional[Path] = None, table: str = "price") -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return min/max dates from the specified price table (default: 'price' = qfq)."""
    try:
        conn = _connect(base_dir, read_only=True)
    except FileNotFoundError:
        return None
    cur = conn.execute(f"SELECT MIN(date), MAX(date) FROM {table} WHERE symbol=?", (symbol,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return pd.to_datetime(row[0]), pd.to_datetime(row[1])


def financial_date_range(symbol: str, base_dir: Optional[Path] = None) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return min/max financial dates for symbol if present in DB."""
    try:
        conn = _connect(base_dir, read_only=True)
    except FileNotFoundError:
        return None
    cur = conn.execute("SELECT MIN(date), MAX(date) FROM financial WHERE symbol=?", (symbol,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return pd.to_datetime(row[0]), pd.to_datetime(row[1])


def list_symbols(table: str = "price", base_dir: Optional[Path] = None) -> list[str]:
    try:
        conn = _connect(base_dir, read_only=True)
    except FileNotFoundError:
        return []
    cur = conn.execute(f"SELECT DISTINCT symbol FROM {table}")
    return [r[0] for r in cur.fetchall()]


def save_financial(symbol: str, df: pd.DataFrame, base_dir: Optional[Path] = None) -> Path:
    conn = _connect(base_dir)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    if "publish_date" in df.columns:
        df["publish_date"] = pd.to_datetime(df["publish_date"]).dt.date.astype(str)
    value_cols = [c for c in df.columns if c != "date"]
    _ensure_financial_columns(conn, value_cols)
    escaped_cols = [col.replace('"', '""') for col in value_cols]
    insert_cols = ['"symbol"', '"date"'] + [f'"{c}"' for c in escaped_cols]
    placeholders = ",".join(["?"] * len(insert_cols))
    update_parts = [f'{c}=excluded.{c}' for c in insert_cols[2:]]
    update_clause = ", ".join(update_parts) if update_parts else "symbol=excluded.symbol"
    sql = (
        f"INSERT INTO financial({','.join(insert_cols)}) VALUES({placeholders}) "
        f"ON CONFLICT(symbol,date) DO UPDATE SET {update_clause}"
    )
    rows = []
    for r in df.to_dict("records"):
        row = [symbol, r.get("date", "")]
        for col in value_cols:
            row.append(r.get(col))
        rows.append(row)
    with conn:
        conn.executemany(sql, rows)
    return (base_dir or DATA_DIR) / "stock.db"


def load_financial(symbol: str, base_dir: Optional[Path] = None) -> pd.DataFrame:
    conn = _connect(base_dir, read_only=True)
    cur = conn.execute("PRAGMA table_info(financial)")
    columns = [row[1] for row in cur.fetchall()]
    select_cols = ['"' + c.replace('"', '""') + '"' for c in columns]
    sql = f"SELECT {','.join(select_cols)} FROM financial WHERE symbol=? ORDER BY date ASC"
    df = pd.read_sql_query(sql, conn, params=(symbol,))
    if df.empty:
        raise FileNotFoundError(f"No financial data for {symbol}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ─── fund_flow ────────────────────────────────────────────────────────────────────

FUND_FLOW_COLS = [
    "date", "close", "pct_change",
    "main_net_inflow", "main_net_pct",
    "super_net_inflow", "super_net_pct",
    "big_net_inflow", "big_net_pct",
    "mid_net_inflow", "mid_net_pct",
    "small_net_inflow", "small_net_pct",
]

FUND_FLOW_SQL = """
    INSERT INTO fund_flow(symbol,date,close,pct_change,main_net_inflow,main_net_pct,
        super_net_inflow,super_net_pct,big_net_inflow,big_net_pct,
        mid_net_inflow,mid_net_pct,small_net_inflow,small_net_pct)
    VALUES(:symbol,:date,:close,:pct_change,:main_net_inflow,:main_net_pct,
        :super_net_inflow,:super_net_pct,:big_net_inflow,:big_net_pct,
        :mid_net_inflow,:mid_net_pct,:small_net_inflow,:small_net_pct)
    ON CONFLICT(symbol,date) DO UPDATE SET
        close=excluded.close, pct_change=excluded.pct_change,
        main_net_inflow=excluded.main_net_inflow, main_net_pct=excluded.main_net_pct,
        super_net_inflow=excluded.super_net_inflow, super_net_pct=excluded.super_net_pct,
        big_net_inflow=excluded.big_net_inflow, big_net_pct=excluded.big_net_pct,
        mid_net_inflow=excluded.mid_net_inflow, mid_net_pct=excluded.mid_net_pct,
        small_net_inflow=excluded.small_net_inflow, small_net_pct=excluded.small_net_pct
"""


def save_fund_flow(symbol: str, df: pd.DataFrame, base_dir: Optional[Path] = None) -> Path:
    """Save fund flow data (主力/超大/大/中/小单净流入)."""
    if df.empty:
        return (base_dir or DATA_DIR) / "stock.db"
    conn = _connect(base_dir)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    rows = [{**r, "symbol": symbol} for r in df.to_dict("records")]
    with conn:
        conn.executemany(FUND_FLOW_SQL, rows)
    return (base_dir or DATA_DIR) / "stock.db"


def load_fund_flow(symbol: str, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load fund flow data for a symbol."""
    conn = _connect(base_dir, read_only=True)
    cols = ", ".join(FUND_FLOW_COLS[1:])  # skip date, we select all
    df = pd.read_sql_query(
        f"SELECT date,{cols} FROM fund_flow WHERE symbol=? ORDER BY date ASC",
        conn, params=(symbol,),
    )
    if df.empty:
        raise FileNotFoundError(f"No fund flow data for {symbol}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def fund_flow_date_range(symbol: str, base_dir: Optional[Path] = None) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        conn = _connect(base_dir, read_only=True)
    except FileNotFoundError:
        return None
    cur = conn.execute("SELECT MIN(date), MAX(date) FROM fund_flow WHERE symbol=?", (symbol,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return pd.to_datetime(row[0]), pd.to_datetime(row[1])


# ─── chip ───────────────────────────────────────────────────────────────────────

CHIP_COLS = ["date", "profit_ratio", "avg_cost", "c90_low", "c90_high", "c90集中度",
             "c70_low", "c70_high", "c70集中度"]

CHIP_SQL = """
    INSERT INTO chip(symbol,date,profit_ratio,avg_cost,c90_low,c90_high,c90集中度,
        c70_low,c70_high,c70集中度)
    VALUES(:symbol,:date,:profit_ratio,:avg_cost,:c90_low,:c90_high,:c90集中度,
        :c70_low,:c70_high,:c70集中度)
    ON CONFLICT(symbol,date) DO UPDATE SET
        profit_ratio=excluded.profit_ratio, avg_cost=excluded.avg_cost,
        c90_low=excluded.c90_low, c90_high=excluded.c90_high, c90集中度=excluded.c90集中度,
        c70_low=excluded.c70_low, c70_high=excluded.c70_high, c70集中度=excluded.c70集中度
"""


def save_chip(symbol: str, df: pd.DataFrame, base_dir: Optional[Path] = None) -> Path:
    """Save chip distribution data (筹码分布)."""
    if df.empty:
        return (base_dir or DATA_DIR) / "stock.db"
    conn = _connect(base_dir)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    rows = [{**r, "symbol": symbol} for r in df.to_dict("records")]
    with conn:
        conn.executemany(CHIP_SQL, rows)
    return (base_dir or DATA_DIR) / "stock.db"


def load_chip(symbol: str, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load chip distribution data for a symbol."""
    conn = _connect(base_dir, read_only=True)
    cols = ", ".join(CHIP_COLS[1:])
    df = pd.read_sql_query(
        f"SELECT date,{cols} FROM chip WHERE symbol=? ORDER BY date ASC",
        conn, params=(symbol,),
    )
    if df.empty:
        raise FileNotFoundError(f"No chip data for {symbol}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def chip_date_range(symbol: str, base_dir: Optional[Path] = None) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        conn = _connect(base_dir, read_only=True)
    except FileNotFoundError:
        return None
    cur = conn.execute("SELECT MIN(date), MAX(date) FROM chip WHERE symbol=?", (symbol,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return pd.to_datetime(row[0]), pd.to_datetime(row[1])
