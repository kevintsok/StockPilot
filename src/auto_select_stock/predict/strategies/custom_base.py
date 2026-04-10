"""Extended base class for custom strategies with access to technical context and position info."""

from typing import Any, Dict, List, Optional

from .base import BaseStrategy, Signal


class ExtendedBaseStrategy(BaseStrategy):
    """Base class for custom strategies that use technical indicators and position context.

    Strategies can access:
    - signal.context: Dict[str, float] with pre-computed technical indicators
    - get_position_info(symbol, cache): current position entry price, shares, holding days
    """

    def get_position_info(self, symbol: str, cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return position info for symbol, or None if not held.

        Returns: {entry_price, shares, holding_days, high_price, unrealized_pnl_pct}
        """
        portfolio = cache.get("_portfolio", {})
        return portfolio.get(symbol)

    def is_oversold(self, signal: Signal) -> bool:
        """RSI < 35 or Bollinger position < 0.25 (oversold bounce potential)."""
        if signal.context is None:
            return False
        rsi = signal.context.get("rsi_14", 50.0)
        bb = signal.context.get("bb_position", 0.5)
        return rsi < 35 or bb < 0.25

    def is_overbought(self, signal: Signal) -> bool:
        """RSI > 65 or Bollinger position > 0.75 (overbought pullback risk)."""
        if signal.context is None:
            return False
        rsi = signal.context.get("rsi_14", 50.0)
        bb = signal.context.get("bb_position", 0.5)
        return rsi > 65 or bb > 0.75

    def get_volatility_regime(self, signal: Signal) -> str:
        """Return 'low', 'medium', or 'high' based on 20d realized volatility.

        Uses fixed thresholds calibrated for Chinese A-shares (daily vol ~1-2%).
        """
        if signal.context is None:
            return "medium"
        vol = signal.context.get("vol_20d", 0.0)
        if vol < 0.015:
            return "low"
        elif vol > 0.035:
            return "high"
        return "medium"

    def is_new_high(self, symbol: str, cache: Dict[str, Any], lookback: int = 20) -> bool:
        """Return True if symbol is at a N-day high (tracked in cache)."""
        high_tracker = cache.get("_high_tracker", {})
        return high_tracker.get(symbol, False)

    def get_context_value(self, signal: Signal, key: str, default: float = 0.0) -> float:
        """Safely get a context value, returning default if not available."""
        if signal.context is None:
            return default
        return signal.context.get(key, default)

    def rank_stocks(
        self,
        signals: List[Signal],
        horizon: str,
        ascending: bool = False,
    ) -> List[Signal]:
        """Return signals sorted by predicted return for given horizon."""
        return sorted(signals, key=lambda s: s.get_horizon_ret(horizon), reverse=not ascending)

    def top_k_signals(
        self,
        signals: List[Signal],
        k: int,
        horizon: str = "5d",
    ) -> List[Signal]:
        """Return top-K signals by predicted return for given horizon."""
        ranked = self.rank_stocks(signals, horizon)
        return ranked[:k]
