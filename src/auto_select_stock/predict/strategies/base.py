from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Signal:
    """A single stock's prediction for a given trading date."""
    symbol: str
    predicted_ret: float  # model predicted return (1d horizon, backward compat)
    realized_ret: float  # actual realized return (entry at T close, exit at T+1 close)
    industry: Optional[str] = None  # for sector-neutral strategies
    predicted_rets: Optional[Dict[str, float]] = None  # e.g. {"3d": 0.05, "5d": 0.08, ...}
    entry_price: float = 0.0   # T's close price (execution price for buying)
    auc_limit: int = 0         # 0=none, 1=limit_up(cannot buy), -1=limit_down(cannot sell)

    def get_horizon_ret(self, horizon: str = "1d") -> float:
        """Get predicted return for a specific horizon."""
        if horizon == "1d":
            return self.predicted_ret
        if self.predicted_rets:
            return self.predicted_rets.get(horizon, self.predicted_ret)
        return self.predicted_ret


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses implement ``select_positions()`` which given a list of Signals
    for one trading day and the previous day's portfolio weights returns a new
    dictionary of ``{symbol: weight}``.
    """

    # Class-level defaults (overridden by instance-level attrs set via __init__)
    name: str = "BaseStrategy"
    tag: str = ""

    def __init__(self, horizon: str = "1d", name: str = None, tag: str = None):
        self._horizon = horizon
        if name is not None:
            self.name = name
        if tag is not None:
            self.tag = tag

    @property
    def horizon(self) -> str:
        """The prediction horizon this strategy uses for ranking/selection (default: 1d)."""
        return self._horizon

    def _get_predicted_ret(self, signal: Signal) -> float:
        """Get the predicted return for this strategy's horizon."""
        return signal.get_horizon_ret(self.horizon)

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        """Select positions for a single trading day.

        Args:
            signals: All stock signals for today (predicted/realized returns).
            prev_weights: Portfolio weights from the previous day.
            cache: Mutable dict that strategies can use to store per-day state
                   across calls (e.g., trailing stop price levels).

        Returns:
            Dictionary mapping symbol -> weight (long positive, short negative).
            Weights are typically normalized so they sum to |1| or less.
        """
        # Default implementation: equal-weight top-N positive predicted returns
        raise NotImplementedError

    def on_day_end(
        self,
        date: str,
        weights: Dict[str, float],
        realized_rets: Dict[str, float],
        cache: Dict[str, Any],
    ) -> None:
        """Optional hook called after a trading day ends.

        Strategies can use this to update internal cache state (e.g., record
        realized returns for trailing-stop tracking).
        """
