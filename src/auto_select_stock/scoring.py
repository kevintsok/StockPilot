from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .storage import load_stock_history
from .stock_types import StockDailyRow, StockMeta, StockScore, StockSnapshot


def select_llm_client(provider: str, model: Optional[str] = None):
    provider = provider.lower()
    if provider == "openai":
        from .llm.openai_client import OpenAIClient

        return OpenAIClient(model=model)
    if provider == "dummy":
        from .llm.dummy import DummyLLM

        return DummyLLM()
    raise ValueError(f"Unsupported LLM provider: {provider}")


def compute_factors(df: pd.DataFrame) -> dict:
    factors = {}
    factors["vol_ratio_5"] = df["volume"].iloc[-1] / df["volume"].tail(5).mean()
    factors["pct_change_20"] = df["pct_change"].tail(20).sum()
    factors["close_ma_20"] = df["close"].tail(20).mean()
    factors["momentum_60"] = df["close"].iloc[-1] / df["close"].iloc[-60] - 1 if len(df) >= 60 else 0.0
    # 占位估值因子（可替换为真实财务数据，如 PE/ROE/PB）
    factors["pe_ttm"] = float("nan")
    return factors


def to_snapshot(symbol: str, arr: np.ndarray) -> StockSnapshot:
    df = pd.DataFrame(arr)
    factors = compute_factors(df)
    recent_rows = [
        StockDailyRow(
            date=pd.to_datetime(row["date"]).to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            amount=float(row["amount"]),
            turnover_rate=float(row["turnover_rate"]),
            volume_ratio=float(row["volume_ratio"]),
            pct_change=float(row["pct_change"]),
        )
        for row in arr[-60:]
    ]
    snapshot = StockSnapshot(meta=StockMeta(symbol=symbol), recent_rows=recent_rows, factors=factors)
    return snapshot


def score_symbols(symbols: Iterable[str], provider: str, model: Optional[str] = None) -> List[StockScore]:
    client = select_llm_client(provider, model=model)
    scores: List[StockScore] = []
    for symbol in symbols:
        try:
            arr = load_stock_history(symbol)
        except FileNotFoundError:
            continue
        if len(arr) == 0:
            continue
        snapshot = to_snapshot(symbol, arr)
        res = client.score(snapshot)
        scores.append(res)
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores
