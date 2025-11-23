import sqlite3
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .config import DATA_DIR

DB_PATH = DATA_DIR / "stock.db"


def ensure_data_dir(path: Optional[Path] = None) -> Path:
    target = path or DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _connect(base_dir: Optional[Path] = None, read_only: bool = False) -> sqlite3.Connection:
    db_path = (base_dir or DATA_DIR) / "stock.db"
    if read_only:
        uri = f"file:{db_path}?mode=ro&immutable=1"
        return sqlite3.connect(uri, uri=True)
    ensure_data_dir(base_dir)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
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
        try:
            escaped = col.replace('"', '""')
            conn.execute(f'ALTER TABLE financial ADD COLUMN "{escaped}" REAL')
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
    conn.commit()


def save_stock_history(symbol: str, structured_array: np.ndarray, base_dir: Optional[Path] = None) -> Path:
    conn = _connect(base_dir)
    df = pd.DataFrame(structured_array)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    rows = df.to_dict("records")
    with conn:
        conn.executemany(
            """
            INSERT INTO price(symbol,date,open,high,low,close,volume,amount,turnover_rate,volume_ratio,pct_change,amplitude,change_amount)
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


def load_stock_history(symbol: str, base_dir: Optional[Path] = None) -> np.ndarray:
    conn = _connect(base_dir, read_only=True)
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume, amount, turnover_rate, volume_ratio, pct_change, amplitude, change_amount "
        "FROM price WHERE symbol=? ORDER BY date ASC",
        conn,
        params=(symbol,),
    )
    if df.empty:
        raise FileNotFoundError(f"No price data for {symbol}")
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


def price_date_range(symbol: str, base_dir: Optional[Path] = None) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    conn = _connect(base_dir, read_only=True)
    cur = conn.execute("SELECT MIN(date), MAX(date) FROM price WHERE symbol=?", (symbol,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return pd.to_datetime(row[0]), pd.to_datetime(row[1])


def financial_date_range(symbol: str, base_dir: Optional[Path] = None) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return min/max financial dates for symbol if present in DB."""
    conn = _connect(base_dir, read_only=True)
    cur = conn.execute("SELECT MIN(date), MAX(date) FROM financial WHERE symbol=?", (symbol,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return pd.to_datetime(row[0]), pd.to_datetime(row[1])


def list_symbols(table: str = "price", base_dir: Optional[Path] = None) -> list[str]:
    conn = _connect(base_dir, read_only=True)
    cur = conn.execute(f"SELECT DISTINCT symbol FROM {table}")
    return [r[0] for r in cur.fetchall()]


def save_financial(symbol: str, df: pd.DataFrame, base_dir: Optional[Path] = None) -> Path:
    conn = _connect(base_dir)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
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
