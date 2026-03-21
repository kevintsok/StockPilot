"""
Strategy implementations and registry for JSON-driven backtesting.

Each strategy is a ``BaseStrategy`` subclass that receives daily signals
and returns portfolio weights.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import BaseStrategy, Signal

__all__ = [
    "Signal",
    "BaseStrategy",
    "TopKStrategy",
    "ThresholdStrategy",
    "LongShortStrategy",
    "MomentumFilterStrategy",
    "RiskParityStrategy",
    "MeanReversionStrategy",
    "ConfidenceStrategy",
    "SectorNeutralStrategy",
    "TrailingStopStrategy",
    "DualThreshStrategy",
]


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------


def _normalize_weights(weights: Dict[str, float], total: float) -> Dict[str, float]:
    """Divide all weights by *total* so they sum to |1| (or keep as-is if total==0)."""
    if abs(total) < 1e-10:
        return {sym: 0.0 for sym in weights}
    return {sym: w / total for sym, w in weights.items()}


def _top_k(sigs: List[Signal], k: int, ascending: bool = False) -> List[Signal]:
    """Return the top (or bottom) K signals by predicted return."""
    sorted_sigs = sorted(sigs, key=lambda s: s.predicted_ret, reverse=not ascending)
    return sorted_sigs[:k]


def _realized_volatility(realized_rets: List[float], lookback: int = 20) -> float:
    """Sample std-dev of realized returns over *lookback* days."""
    if len(realized_rets) < 2:
        return 1.0
    tail = realized_rets[-lookback:]
    mean = sum(tail) / len(tail)
    variance = sum((r - mean) ** 2 for r in tail) / max(len(tail) - 1, 1)
    return math.sqrt(variance) if variance > 0 else 1e-6


# ----------------------------------------------------------------------
# 1. TopK-Proportional (topk)
# ----------------------------------------------------------------------


class TopKStrategy(BaseStrategy):
    """Buy the top-K stocks proportional to their predicted returns, long-only."""

    name = "TopK-Proportional"

    def __init__(self, top_k: int = 5, allow_short: bool = False):
        self.top_k = top_k
        self.allow_short = allow_short

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        sorted_sigs = sorted(signals, key=lambda s: s.predicted_ret, reverse=True)
        if self.allow_short:
            n = len(sorted_sigs)
            top_n = max(1, n // 10)
            long_sigs = sorted_sigs[:top_n]
            short_sigs = sorted_sigs[-top_n:]
            all_sigs = long_sigs + short_sigs
            total_pred = sum(abs(s.predicted_ret) for s in all_sigs)
            weights = {}
            for s in long_sigs:
                weights[s.symbol] = s.predicted_ret / total_pred
            for s in short_sigs:
                weights[s.symbol] = s.predicted_ret / total_pred
            return weights
        else:
            top_sigs = [s for s in sorted_sigs if s.predicted_ret > 0][: self.top_k]
            total_pred = sum(s.predicted_ret for s in top_sigs)
            if total_pred <= 0:
                return {}
            return {s.symbol: s.predicted_ret / total_pred for s in top_sigs}


# ----------------------------------------------------------------------
# 2. Threshold Strategy (threshold)
# ----------------------------------------------------------------------


class ThresholdStrategy(BaseStrategy):
    """Buy all stocks with predicted return above a threshold, equal weight."""

    name = "TopK-Threshold"

    def __init__(self, top_k: int = 5, threshold: float = 0.01, allow_short: bool = False):
        self.top_k = top_k
        self.threshold = threshold
        self.allow_short = allow_short

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if self.allow_short:
            long_sigs = [s for s in signals if s.predicted_ret > self.threshold]
            short_sigs = [s for s in signals if s.predicted_ret < -self.threshold]
            n_long = len(long_sigs) or 1
            n_short = len(short_sigs) or 1
            long_w = 0.5 / n_long
            short_w = -0.5 / n_short
            weights = {s.symbol: long_w for s in long_sigs}
            weights.update({s.symbol: short_w for s in short_sigs})
            return weights
        else:
            candidates = [s for s in signals if s.predicted_ret > self.threshold]
            top_sigs = sorted(candidates, key=lambda s: s.predicted_ret, reverse=True)[: self.top_k]
            if not top_sigs:
                return {}
            w = 1.0 / len(top_sigs)
            return {s.symbol: w for s in top_sigs}


# ----------------------------------------------------------------------
# 3. Long/Short Equal Weight (long_short)
# ----------------------------------------------------------------------


class LongShortStrategy(BaseStrategy):
    """Long top-pct and short bottom-pct, equal weight within each side."""

    name = "LongShort-Equal"

    def __init__(self, top_pct: float = 0.1, allow_short: bool = True):
        self.top_pct = top_pct
        self.allow_short = allow_short

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}
        sorted_sigs = sorted(signals, key=lambda s: s.predicted_ret, reverse=True)
        n = len(sorted_sigs)
        top_n = max(1, int(n * self.top_pct))
        long_sigs = sorted_sigs[:top_n]
        short_sigs = sorted_sigs[-top_n:] if self.allow_short else []

        weights: Dict[str, float] = {}
        long_alloc = 0.5
        short_alloc = 0.5 if self.allow_short else 0.0

        if long_sigs:
            w = long_alloc / len(long_sigs)
            for s in long_sigs:
                weights[s.symbol] = w
        if short_sigs:
            w = -short_alloc / len(short_sigs)
            for s in short_sigs:
                weights[s.symbol] = w
        return weights


# ----------------------------------------------------------------------
# 4. Momentum Filter (momentum_filter)
# ----------------------------------------------------------------------


class MomentumFilterStrategy(BaseStrategy):
    """Only buy top-K when today's pred_ret > short-term MA of recent pred_rets."""

    name = "Momentum-Filter"

    def __init__(
        self,
        top_k: int = 5,
        lookback: int = 10,
        threshold: float = 0.0,
        allow_short: bool = False,
    ):
        self.top_k = top_k
        self.lookback = lookback
        self.threshold = threshold
        self.allow_short = allow_short

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        momentum_cache = cache.setdefault("momentum", {})  # symbol -> list of pred_rets

        for s in signals:
            hist = momentum_cache.setdefault(s.symbol, [])
            hist.append(s.predicted_ret)
            # Keep only last lookback entries
            if len(hist) > self.lookback:
                momentum_cache[s.symbol] = hist[-self.lookback :]

        filtered: List[Signal] = []
        for s in signals:
            hist = momentum_cache.get(s.symbol, [])
            if len(hist) < 2:
                # Not enough history; use raw signal
                if s.predicted_ret > self.threshold:
                    filtered.append(s)
                continue
            ma = sum(hist[-self.lookback :]) / len(hist[-self.lookback :])
            if s.predicted_ret > ma:
                filtered.append(s)

        if not filtered:
            return {}

        top_sigs = sorted(filtered, key=lambda s: s.predicted_ret, reverse=True)[: self.top_k]
        total_pred = sum(s.predicted_ret for s in top_sigs)
        if total_pred <= 0:
            return {}
        return {s.symbol: s.predicted_ret / total_pred for s in top_sigs}

    def on_day_end(
        self,
        date: str,
        weights: Dict[str, float],
        realized_rets: Dict[str, float],
        cache: Dict[str, Any],
    ) -> None:
        pass  # Momentum cache is already updated in select_positions


# ----------------------------------------------------------------------
# 5. Risk Parity (risk_parity)
# ----------------------------------------------------------------------


class RiskParityStrategy(BaseStrategy):
    """Among top-K, weight inversely proportional to realized volatility."""

    name = "Risk-Parity"

    def __init__(
        self,
        top_k: int = 5,
        allow_short: bool = False,
        vol_lookback: int = 20,
    ):
        self.top_k = top_k
        self.allow_short = allow_short
        self.vol_lookback = vol_lookback

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        vol_cache = cache.setdefault("volatility", {})  # symbol -> list of realized_rets

        # Update realized returns history first
        for s in signals:
            hist = vol_cache.setdefault(s.symbol, [])
            hist.append(s.realized_ret)
            if len(hist) > self.vol_lookback:
                vol_cache[s.symbol] = hist[-self.vol_lookback :]

        # Take top-K by predicted return
        sorted_sigs = sorted(signals, key=lambda s: s.predicted_ret, reverse=True)
        if self.allow_short:
            top_sigs = sorted_sigs[: self.top_k]
            short_sigs = sorted_sigs[-self.top_k :]
            all_sigs = top_sigs + short_sigs
        else:
            top_sigs = [s for s in sorted_sigs if s.predicted_ret > 0][: self.top_k]
            all_sigs = top_sigs

        if not all_sigs:
            return {}

        # Inverse-vol weights
        inv_vols = []
        for s in all_sigs:
            vol = _realized_volatility(vol_cache.get(s.symbol, []), self.vol_lookback)
            inv_vols.append(1.0 / vol)

        total_inv = sum(inv_vols)
        if total_inv <= 0:
            return {}

        long_side = 1.0 if not self.allow_short else 0.5
        short_side = 0.0 if not self.allow_short else -0.5

        weights = {}
        for s, inv_vol in zip(all_sigs, inv_vols):
            w = inv_vol / total_inv
            if s in top_sigs:
                weights[s.symbol] = long_side * w
            else:
                weights[s.symbol] = short_side * w
        return weights


# ----------------------------------------------------------------------
# 6. Mean Reversion (mean_reversion)
# ----------------------------------------------------------------------


class MeanReversionStrategy(BaseStrategy):
    """Long bottom-K (losers), short top-K (winners) - classic reversal strategy."""

    name = "BottomK-Reversal"

    def __init__(self, top_k: int = 5, allow_short: bool = True):
        self.top_k = top_k
        self.allow_short = allow_short

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}
        sorted_sigs = sorted(signals, key=lambda s: s.predicted_ret, reverse=True)
        losers = sorted_sigs[: self.top_k]  # bottom-K by pred (worst) = long
        winners = sorted_sigs[-self.top_k :]  # top-K by pred (best) = short
        if not self.allow_short:
            # Long-only: just long the bottom-K
            n = len(losers) or 1
            w = 1.0 / n
            return {s.symbol: w for s in losers}

        weights: Dict[str, float] = {}
        n_long = len(losers) or 1
        n_short = len(winners) or 1
        for s in losers:
            weights[s.symbol] = 0.5 / n_long
        for s in winners:
            weights[s.symbol] = -0.5 / n_short
        return weights


# ----------------------------------------------------------------------
# 7. Confidence Sized (confidence)
# ----------------------------------------------------------------------


class ConfidenceStrategy(BaseStrategy):
    """Weight by absolute predicted return (confidence), long-only."""

    name = "Confidence-Sized"

    def __init__(
        self,
        top_k: int = 5,
        allow_short: bool = False,
        min_confidence: float = 0.005,
    ):
        self.top_k = top_k
        self.allow_short = allow_short
        self.min_confidence = min_confidence

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if self.allow_short:
            long_sigs = [s for s in signals if s.predicted_ret > self.min_confidence]
            short_sigs = [s for s in signals if s.predicted_ret < -self.min_confidence]
            top_long = sorted(long_sigs, key=lambda s: s.predicted_ret, reverse=True)[: self.top_k]
            top_short = sorted(short_sigs, key=lambda s: s.predicted_ret)[: self.top_k]
            all_sigs = top_long + top_short
        else:
            candidates = [s for s in signals if s.predicted_ret > self.min_confidence]
            all_sigs = sorted(candidates, key=lambda s: s.predicted_ret, reverse=True)[: self.top_k]

        if not all_sigs:
            return {}

        total_conf = sum(abs(s.predicted_ret) for s in all_sigs)
        if total_conf <= 0:
            return {}

        weights = {}
        long_alloc = 1.0 if not self.allow_short else 0.5
        short_alloc = 0.0 if not self.allow_short else -0.5

        for s in all_sigs:
            conf = abs(s.predicted_ret) / total_conf
            if s.predicted_ret > 0:
                weights[s.symbol] = long_alloc * conf
            else:
                weights[s.symbol] = short_alloc * conf
        return weights


# ----------------------------------------------------------------------
# 8. Sector Neutral (sector_neutral)
# ----------------------------------------------------------------------


class SectorNeutralStrategy(BaseStrategy):
    """Long/short equal-weight with net-zero sector exposure."""

    name = "Sector-Neutral"

    def __init__(self, top_pct: float = 0.1, allow_short: bool = True):
        self.top_pct = top_pct
        self.allow_short = allow_short

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        sorted_sigs = sorted(signals, key=lambda s: s.predicted_ret, reverse=True)
        n = len(sorted_sigs)
        top_n = max(1, int(n * self.top_pct))
        long_sigs = sorted_sigs[:top_n]
        short_sigs = sorted_sigs[-top_n:] if self.allow_short else []

        # Group by industry
        sector_holdings: Dict[str, List[tuple]] = {}
        for s in long_sigs:
            ind = s.industry or "UNKNOWN"
            sector_holdings.setdefault(ind, []).append((s, 1.0 / len(long_sigs)))
        for s in short_sigs:
            ind = s.industry or "UNKNOWN"
            sector_holdings.setdefault(ind, []).append((s, -1.0 / len(short_sigs)))

        # Compute equal sector weights
        sector_weights: Dict[str, float] = {}
        for ind, holdings in sector_holdings.items():
            # Net sector weight: long - short
            sector_weights[ind] = sum(w for _, w in holdings)

        # Build final weights with sector neutralization
        weights: Dict[str, float] = {}
        for ind, holdings in sector_holdings.items():
            sec_w = sector_weights.get(ind, 0.0)
            for s, w in holdings:
                weights[s.symbol] = w - sec_w * w  # subtract sector exposure proportionally

        # Normalize to sum to |1|
        total_abs = sum(abs(w) for w in weights.values())
        if total_abs > 0:
            weights = {sym: w / total_abs for sym, w in weights.items()}
        return weights


# ----------------------------------------------------------------------
# 9. Trailing Stop (trailing_stop)
# ----------------------------------------------------------------------


class TrailingStopStrategy(BaseStrategy):
    """TopK strategy with a trailing stop-loss: exit if daily loss > stop_loss."""

    name = "TopK-StopLoss"

    def __init__(
        self,
        top_k: int = 5,
        stop_loss: float = 0.05,
        allow_short: bool = False,
    ):
        self.top_k = top_k
        self.stop_loss = stop_loss
        self.allow_short = allow_short

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        stop_cache = cache.setdefault("trailing_stop", {})
        entry_prices = stop_cache.setdefault("entry_prices", {})  # symbol -> entry close price
        held = stop_cache.setdefault("held", set())  # symbols currently held

        # Determine new top-K
        sorted_sigs = sorted(signals, key=lambda s: s.predicted_ret, reverse=True)
        top_sigs = [s for s in sorted_sigs if s.predicted_ret > 0][: self.top_k]
        new_top = {s.symbol for s in top_sigs}

        # Check stop-loss on previously held positions
        # realized_rets gives us today's return for each symbol
        realized_rets = {s.symbol: s.realized_ret for s in signals}
        keep = {}
        for sym in list(held):
            ret_today = realized_rets.get(sym, 0.0)
            entry = entry_prices.get(sym)
            if entry is not None and entry > 0:
                # Compute cumulative return from entry
                current_price = entry * (1 + ret_today)
                pct_from_entry = (current_price / entry) - 1.0
                if pct_from_entry < -self.stop_loss:
                    # Stopped out
                    held.discard(sym)
                    entry_prices.pop(sym, None)
                    continue
            if sym in new_top:
                keep[sym] = entry_prices.get(sym)
                held.add(sym)

        # Add new positions at today's close
        for s in top_sigs:
            if s.symbol not in held:
                entry_prices[s.symbol] = 1.0  # normalized entry; realized ret is relative
                held.add(s.symbol)

        # Compute weights from held symbols
        if not held:
            return {}

        held_sigs = [s for s in signals if s.symbol in held]
        total_pred = sum(s.predicted_ret for s in held_sigs) or 1e-10
        return {s.symbol: s.predicted_ret / total_pred for s in held_sigs}

    def on_day_end(
        self,
        date: str,
        weights: Dict[str, float],
        realized_rets: Dict[str, float],
        cache: Dict[str, Any],
    ) -> None:
        stop_cache = cache.setdefault("trailing_stop", {})
        held = stop_cache.setdefault("held", set())
        entry_prices = stop_cache.setdefault("entry_prices", {})
        # Update entry prices using realized returns
        for sym in list(held):
            ret = realized_rets.get(sym, 0.0)
            entry = entry_prices.get(sym, 1.0)
            entry_prices[sym] = entry * (1 + ret)
        stop_cache["held"] = held
        stop_cache["entry_prices"] = entry_prices


# ----------------------------------------------------------------------
# 10. Dual Threshold (dual_thresh)
# ----------------------------------------------------------------------


class DualThreshStrategy(BaseStrategy):
    """Long when pred > upper_thresh, short when pred < lower_thresh, top-K each side."""

    name = "Dual-Threshold"

    def __init__(
        self,
        upper_thresh: float = 0.02,
        lower_thresh: float = -0.01,
        top_k: int = 5,
    ):
        self.upper_thresh = upper_thresh
        self.lower_thresh = lower_thresh
        self.top_k = top_k

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        long_candidates = sorted(
            [s for s in signals if s.predicted_ret > self.upper_thresh],
            key=lambda s: s.predicted_ret,
            reverse=True,
        )[: self.top_k]
        short_candidates = sorted(
            [s for s in signals if s.predicted_ret < self.lower_thresh],
            key=lambda s: s.predicted_ret,
        )[: self.top_k]

        n_long = len(long_candidates) or 1
        n_short = len(short_candidates) or 1
        long_w = 0.5 / n_long
        short_w = -0.5 / n_short

        weights: Dict[str, float] = {}
        for s in long_candidates:
            weights[s.symbol] = long_w
        for s in short_candidates:
            weights[s.symbol] = short_w
        return weights
