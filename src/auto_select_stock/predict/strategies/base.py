from abc import ABC, ABCMeta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

# ----------------------------------------------------------------------
# Auto-discovery registry for strategy types
# ----------------------------------------------------------------------
_STRATEGY_TYPE_REGISTRY: Dict[str, Type["BaseStrategy"]] = {}


def _register_strategy_subclass(cls: Type["BaseStrategy"]) -> None:
    """Register a BaseStrategy subclass by its type string.

    The type string is derived from the class name by stripping the
    'Strategy' suffix, splitting on lowercase-to-uppercase transitions
    (keeping consecutive capitals together), lowercasing, and joining with
    underscores. E.g. 'TopKStrategy' -> 'topk', 'LongShortStrategy' ->
    'long_short'.
    """
    name = cls.__name__.replace("Strategy", "")
    # Split on transition from lowercase to uppercase, keep consecutive capitals together
    parts = []
    current = []
    for i, c in enumerate(name):
        if i > 0 and c.isupper() and name[i - 1].islower():
            parts.append("".join(current))
            current = [c]
        else:
            current.append(c)
    if current:
        parts.append("".join(current))
    type_str = "_".join(p.lower() for p in parts if p)
    # Special case: TopK -> topk (acronym without underscore)
    type_str = type_str.replace("top_k", "topk")
    _STRATEGY_TYPE_REGISTRY[type_str] = cls


class _AutoDiscoveryMeta(ABCMeta):
    """Metaclass that auto-registers any BaseStrategy subclass."""

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs: Any) -> type:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if name != "BaseStrategy" and issubclass(cls, BaseStrategy):
            _register_strategy_subclass(cls)
        return cls


# ----------------------------------------------------------------------
# Signal dataclass
# ----------------------------------------------------------------------
@dataclass
class Signal:
    """A single stock's prediction for a given trading date.

    Supports tuple-style indexing for backward compatibility with code that
    unpacks signals as plain tuples (e.g., ``sym, pred, real, *_ = signal``).
    The canonical field access is via attributes.
    """
    symbol: str
    predicted_ret: float  # model predicted return (1d horizon, backward compat)
    realized_ret: float  # actual realized return (entry at T close, exit at T+1 close)
    industry: Optional[str] = None  # for sector-neutral strategies
    predicted_rets: Optional[Dict[str, float]] = None  # e.g. {"3d": 0.05, "5d": 0.08, ...}
    entry_price: float = 0.0   # T's close price (previous day's close; for auction check)
    auc_limit: int = 0         # 0=none, 1=limit_up(cannot buy), -1=limit_down(cannot sell)
    next_open: float = 0.0     # T+1's open price (buy execution price)
    next_close: float = 0.0    # T+1's close price (sell execution price)
    split_move: float = 0.0    # raw close-to-close move that was filtered (e.g. 0.20 = +20% split); 0 if normal
    context: Optional[Dict[str, float]] = None  # pre-computed technical indicators (RSI, vol, etc.)

    # Tuple index map for backward-compatible __getitem__
    _FIELDS = ("symbol", "predicted_ret", "realized_ret", "industry",
               "predicted_rets", "entry_price", "auc_limit", "next_open", "next_close",
               "split_move", "context")

    def __getitem__(self, index: int):
        """Allow tuple-style indexing (e.g. sig[1]) while preserving attribute access."""
        return getattr(self, self._FIELDS[index])

    def __iter__(self):
        """Allow tuple-style unpacking (e.g. sym, pred, *_ = sig)."""
        return iter(getattr(self, f) for f in self._FIELDS)

    def get_horizon_ret(self, horizon: str = "1d") -> float:
        """Get predicted return for a specific horizon."""
        if horizon == "1d":
            return self.predicted_ret
        if self.predicted_rets:
            return self.predicted_rets.get(horizon, self.predicted_ret)
        return self.predicted_ret


# ----------------------------------------------------------------------
# Strategy params validation
# ----------------------------------------------------------------------
_PARAMS_SCHEMA_PROPERTIES = {
    "top_k": {"type": "integer", "minimum": 1},
    "top_pct": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "lookback": {"type": "integer", "minimum": 1},
    "vol_lookback": {"type": "integer", "minimum": 1},
    "stop_loss": {"type": "number", "minimum": 0.0},
    "threshold": {"type": "number"},
    "upper_thresh": {"type": "number"},
    "lower_thresh": {"type": "number"},
    "min_confidence": {"type": "number", "minimum": 0.0},
    "allow_short": {"type": "boolean"},
    "horizon": {"type": "string"},
    # Custom strategy params
    "low_vol_k": {"type": "integer", "minimum": 1},
    "high_vol_k": {"type": "integer", "minimum": 1},
    "vol_thresh": {"type": "number", "minimum": 0.0},
    "rsi_lower": {"type": "number", "minimum": 0.0, "maximum": 100.0},
    "rsi_upper": {"type": "number", "minimum": 0.0, "maximum": 100.0},
    "min_pred_ret": {"type": "number", "minimum": 0.0},
    "bb_entry": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "bb_exit": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "atr_multiplier": {"type": "number", "minimum": 0.0},
    "regime_lookback": {"type": "integer", "minimum": 1},
    "consensus_threshold": {"type": "number", "minimum": 0.0},
    "reduce_thresh": {"type": "number", "minimum": 0.0},
    "profit_take": {"type": "number", "minimum": 0.0},
    "reduce_pct": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "vol_ratio_thresh": {"type": "number", "minimum": 0.0},
    "trailing_stop_pct": {"type": "number", "minimum": 0.0},
    "ma_filter": {"type": "number", "minimum": 0.0},
    "min_divergence": {"type": "number", "minimum": 0.0},
    # Stop-loss / take-profit params
    "stop_loss_pct": {"type": "number", "minimum": 0.0},
    "take_profit_pct": {"type": "number", "minimum": 0.0},
    "max_holding_days": {"type": "integer", "minimum": 1},
}


def _validate_params(params: Dict[str, Any]) -> None:
    """Validate strategy params when loading from JSON.

    Raises ValueError if any param violates its schema constraints.
    """
    for key, value in params.items():
        schema = _PARAMS_SCHEMA_PROPERTIES.get(key)
        if schema is None:
            continue  # unknown param, skip
        ptype = schema.get("type")
        if ptype in ("integer", "number"):
            if not isinstance(value, (int, float)):
                raise ValueError(f"Parameter '{key}' must be a number, got {type(value).__name__}")
            if "minimum" in schema and value < schema["minimum"]:
                raise ValueError(f"Parameter '{key}' must be >= {schema['minimum']}, got {value}")
            if "maximum" in schema and value > schema["maximum"]:
                raise ValueError(f"Parameter '{key}' must be <= {schema['maximum']}, got {value}")
        elif ptype == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"Parameter '{key}' must be a boolean, got {type(value).__name__}")


# ----------------------------------------------------------------------
# BaseStrategy (uses metaclass for auto-discovery)
# ----------------------------------------------------------------------
class BaseStrategy(ABC, metaclass=_AutoDiscoveryMeta):
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
