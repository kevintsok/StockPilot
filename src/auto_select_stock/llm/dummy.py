from ..stock_types import StockScore, StockSnapshot
from .base import LLMClient


class DummyLLM(LLMClient):
    """
    A deterministic scorer for quick tests without calling external APIs.
    """

    def score(self, snapshot: StockSnapshot) -> StockScore:
        vol_factor = snapshot.factors.get("vol_ratio_5", 1.0)
        momentum = snapshot.factors.get("pct_change_20", 0.0)
        value_hint = snapshot.factors.get("pe_ttm", 15.0)

        score = 5.0
        score += max(0, 2.0 - value_hint / 15.0)  # lower PE implies higher score
        score += -momentum * 0.1  # prefer down/sideways for value pick
        score += max(0, 2.0 - abs(vol_factor - 1.0))
        score = max(0.0, min(10.0, score))

        return StockScore(
            symbol=snapshot.meta.symbol,
            score=score,
            rationale="Dummy scorer based on PE, momentum, and volume ratio.",
            meta=snapshot.meta,
            factors=snapshot.factors,
        )
