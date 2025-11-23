from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


NUMERIC_FIELDS = [
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


@dataclass
class StockMeta:
    symbol: str
    name: Optional[str] = None
    industry: Optional[str] = None


@dataclass
class StockDailyRow:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    turnover_rate: float
    volume_ratio: float
    pct_change: float
    amplitude: float
    change_amount: float


@dataclass
class StockSnapshot:
    meta: StockMeta
    recent_rows: List[StockDailyRow]
    factors: Dict[str, float]


@dataclass
class StockScore:
    symbol: str
    score: float
    rationale: str
    meta: StockMeta
    factors: Dict[str, float]


def to_structured_array(df) -> np.ndarray:
    """
    Convert a pandas DataFrame into a numpy structured array with consistent dtypes.
    """
    dtype = [("date", "datetime64[D]")] + [(name, "f8") for name in NUMERIC_FIELDS]
    df = df.copy()
    dates = pd.to_datetime(df["date"]).dt.normalize().values.astype("datetime64[D]")
    df["date"] = dates

    # Ensure all numeric fields exist.
    for name in NUMERIC_FIELDS:
        if name not in df.columns:
            df[name] = 0.0
    arr = np.zeros(len(df), dtype=dtype)
    arr["date"] = df["date"].values.astype("datetime64[D]")
    for col in NUMERIC_FIELDS:
        arr[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float).to_numpy()
    return arr
