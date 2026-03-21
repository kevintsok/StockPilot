from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Signal:
    """A single stock's prediction for a given trading date."""
    symbol: str
    predicted_ret: float  # model predicted return
    realized_ret: float  # actual realized return (for reporting, not used in decisions)
    industry: Optional[str] = None  # for sector-neutral strategies


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses implement ``select_positions()`` which given a list of Signals
    for one trading day and the previous day's portfolio weights returns a new
    dictionary of ``{symbol: weight}``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

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
