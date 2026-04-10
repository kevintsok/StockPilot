"""10 custom Python strategies using technical context and position awareness.

Each strategy inherits from ExtendedBaseStrategy and can access:
- signal.context: RSI, Bollinger position, volatility, volume ratio, etc.
- get_position_info(symbol, cache): current holding details
"""

import math
from typing import Any, Dict, List, Optional

from .custom_base import ExtendedBaseStrategy
from .base import Signal


# ----------------------------------------------------------------------
# 1. AdaptiveKStrategy — volatility-regime-adaptive K selection
# ----------------------------------------------------------------------
class AdaptiveKStrategy(ExtendedBaseStrategy):
    """Dynamically adjusts K based on market volatility regime.

    High vol (uncertainty) → K=2 (concentrated, high-conviction only)
    Low vol (trending) → K=5 (wider diversification)
    """

    def __init__(self, horizon: str = "5d", name: str = None, tag: str = None,
                 low_vol_k: int = 5, high_vol_k: int = 2, vol_thresh: float = 0.02):
        super().__init__(horizon=horizon, name=name, tag=tag)
        self.low_vol_k = low_vol_k
        self.high_vol_k = high_vol_k
        self.vol_thresh = vol_thresh

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        # Determine K from average market volatility
        vols = [s.context.get("vol_20d", 0.02) for s in signals if s.context]
        avg_vol = sum(vols) / len(vols) if vols else 0.02
        k = self.high_vol_k if avg_vol > self.vol_thresh else self.low_vol_k

        # Filter by auc_limit
        valid = [s for s in signals if s.auc_limit != 1]
        ranked = self.rank_stocks(valid, self.horizon)
        top = ranked[:k]

        weights = {s.symbol: 1.0 / len(top) for s in top}
        return weights


# ----------------------------------------------------------------------
# 2. RSIReversalStrategy — mean-reversion on oversold/overbought
# ----------------------------------------------------------------------
class RSIReversalStrategy(ExtendedBaseStrategy):
    """Buy when RSI < 35 (oversold), sell when RSI > 65 (overbought).

    Only enters when BOTH: RSI condition met AND 5d prediction confirms direction.
    """

    def __init__(self, horizon: str = "5d", name: str = None, tag: str = None,
                 top_k: int = 3, rsi_lower: float = 35.0, rsi_upper: float = 65.0,
                 min_pred_ret: float = 0.005):
        super().__init__(horizon=horizon, name=name, tag=tag)
        self.top_k = top_k
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.min_pred_ret = min_pred_ret

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        candidates = []
        for s in signals:
            if s.auc_limit == 1:
                continue
            if s.context is None:
                continue
            rsi = s.context.get("rsi_14", 50.0)
            pred_5d = s.get_horizon_ret("5d")

            if rsi < self.rsi_lower and pred_5d > self.min_pred_ret:
                # Oversold + predicted to bounce
                candidates.append((s, pred_5d))
            elif rsi > self.rsi_upper and pred_5d < -self.min_pred_ret:
                pass  # Overbought + predicted to drop — skip (no short in long-only)

            # Also consider: keep existing positions that are still in oversold zone
            pos = self.get_position_info(s.symbol, cache)
            if pos is not None and rsi > self.rsi_upper:
                # Position was oversold but now overbought — close it
                continue
            if pos is not None and rsi < self.rsi_upper:
                candidates.append((s, pred_5d))

        if not candidates:
            return {}

        # Sort by predicted return and take top K
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:self.top_k]
        return {s.symbol: 1.0 / len(top) for s, _ in top}


# ----------------------------------------------------------------------
# 3. BollingerBreakoutStrategy — trend following on band expansion
# ----------------------------------------------------------------------
class BollingerBreakoutStrategy(ExtendedBaseStrategy):
    """Buy when price breaks above upper Bollinger band, sell when it falls below.

    Entry: bb_position > 0.8 AND 5d prediction confirms uptrend
    Exit:  bb_position < 0.2 OR 5d prediction turns negative
    """

    def __init__(self, horizon: str = "5d", name: str = None, tag: str = None,
                 top_k: int = 3, bb_entry: float = 0.8, bb_exit: float = 0.2):
        super().__init__(horizon=horizon, name=name, tag=tag)
        self.top_k = top_k
        self.bb_entry = bb_entry
        self.bb_exit = bb_exit

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        candidates = []
        for s in signals:
            if s.auc_limit == 1:
                continue
            if s.context is None:
                continue

            bb = s.context.get("bb_position", 0.5)
            pred = s.get_horizon_ret(self.horizon)

            # Check if breakout is confirmed by prediction
            if bb > self.bb_entry and pred > 0.005:
                candidates.append((s, pred))

        if not candidates:
            return {}

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:self.top_k]
        return {s.symbol: 1.0 / len(top) for s, _ in top}


# ----------------------------------------------------------------------
# 4. VolatilityAdaptiveStopStrategy — ATR-based dynamic stop-loss
# ----------------------------------------------------------------------
class VolatilityAdaptiveStopStrategy(ExtendedBaseStrategy):
    """Trailing stop based on 2×ATR. Stop distance expands in high-vol, tightens in low-vol.

    Replaces fixed stop-loss with adaptive ATR-based exit.
    """

    def __init__(self, horizon: str = "5d", name: str = None, tag: str = None,
                 top_k: int = 3, atr_multiplier: float = 2.0, stop_pct: float = 0.03):
        super().__init__(horizon=horizon, name=name, tag=tag)
        self.top_k = top_k
        self.atr_multiplier = atr_multiplier
        self.stop_pct = stop_pct  # fallback stop if ATR unavailable

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        # Update trailing stops in cache
        for sym, pos_info in cache.get("_portfolio", {}).items():
            atr = 0.0
            for s in signals:
                if s.symbol == sym and s.context:
                    atr = s.context.get("atr_14", 0.0)
                    break
            if atr > 0:
                stop_distance = atr * self.atr_multiplier
            else:
                stop_distance = self.stop_pct

            # Store adaptive stop in cache
            cache[f"_stop_{sym}"] = stop_distance

        # Select new positions
        valid = [s for s in signals if s.auc_limit != 1]
        ranked = self.rank_stocks(valid, self.horizon)
        top = ranked[:self.top_k]

        weights = {s.symbol: 1.0 / len(top) for s in top}
        return weights

    def on_day_end(
        self,
        date: str,
        weights: Dict[str, float],
        realized_rets: Dict[str, float],
        cache: Dict[str, Any],
    ) -> None:
        # Update peak prices for trailing stop tracking
        for sym, ret in realized_rets.items():
            if ret == 0.0:
                continue
            key = f"_peak_{sym}"
            prev_peak = cache.get(key, 0.0)
            # Peak is updated using entry_price from portfolio
            pos = cache.get("_portfolio", {}).get(sym)
            if pos:
                current = pos.get("entry_price", 0.0)
                if current > prev_peak:
                    cache[key] = current


# ----------------------------------------------------------------------
# 5. MomentumRegimeStrategy — regime-switching momentum/reversal
# ----------------------------------------------------------------------
class MomentumRegimeStrategy(ExtendedBaseStrategy):
    """Switches between momentum (trend-following) and reversal based on regime.

    Regime detected by 5d return direction:
    - Market up 3 consecutive days → momentum regime → buy top-K
    - Market down 3 consecutive days → reversal regime → buy bottom-K (reversal)
    """

    def __init__(self, horizon: str = "5d", name: str = None, tag: str = None,
                 k: int = 3, regime_lookback: int = 5):
        super().__init__(horizon=horizon, name=name, tag=tag)
        self.k = k
        self.regime_lookback = regime_lookback

    def _detect_regime(self, cache: Dict[str, Any]) -> str:
        """Detect market regime from recent market returns stored in cache."""
        rets = cache.get("_market_rets", [])
        if len(rets) < 3:
            return "neutral"
        recent = rets[-self.regime_lookback:]
        avg_ret = sum(recent) / len(recent)
        if avg_ret > 0.003:
            return "momentum"
        elif avg_ret < -0.003:
            return "reversal"
        return "neutral"

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        regime = self._detect_regime(cache)
        valid = [s for s in signals if s.auc_limit != 1]

        if regime == "momentum":
            # Buy top-K by 5d prediction
            ranked = self.rank_stocks(valid, "5d", ascending=False)
            top = ranked[:self.k]
        elif regime == "reversal":
            # Buy BOTTOM-K (reversal strategy)
            ranked = self.rank_stocks(valid, "5d", ascending=True)
            top = ranked[:self.k]
        else:
            # Neutral: moderate conviction, K=3
            ranked = self.rank_stocks(valid, "5d", ascending=False)
            top = ranked[:self.k]

        return {s.symbol: 1.0 / len(top) for s in top}

    def on_day_end(
        self,
        date: str,
        weights: Dict[str, float],
        realized_rets: Dict[str, float],
        cache: Dict[str, Any],
    ) -> None:
        # Track market average return for regime detection
        if realized_rets:
            avg_ret = sum(realized_rets.values()) / len(realized_rets)
            market_rets = cache.get("_market_rets", [])
            market_rets.append(avg_ret)
            # Keep only last 20 days
            if len(market_rets) > 20:
                market_rets = market_rets[-20:]
            cache["_market_rets"] = market_rets


# ----------------------------------------------------------------------
# 6. MultiHorizonEnsembleStrategy — multi-horizon consensus weighting
# ----------------------------------------------------------------------
class MultiHorizonEnsembleStrategy(ExtendedBaseStrategy):
    """Only enter when 1d, 5d, and 14d predictions agree on direction.

    When all three horizons predict the same direction, weight ×1.5.
    """

    def __init__(self, horizon: str = "5d", name: str = None, tag: str = None,
                 top_k: int = 3, consensus_threshold: float = 0.005):
        super().__init__(horizon=horizon, name=name, tag=tag)
        self.top_k = top_k
        self.consensus_threshold = consensus_threshold

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        scored = []
        for s in signals:
            if s.auc_limit == 1:
                continue
            r1 = s.get_horizon_ret("1d")
            r5 = s.get_horizon_ret("5d")
            r14 = s.get_horizon_ret("14d")

            # Check consensus
            dirs = [r1 > self.consensus_threshold, r5 > self.consensus_threshold,
                    r14 > self.consensus_threshold]
            if all(dirs) or not any(dirs):
                # All agree (all up or all down)
                avg_ret = (r1 + r5 + r14) / 3.0
                # Bonus if strong consensus
                if all(dirs):
                    avg_ret *= 1.5
                scored.append((s, avg_ret))

        if not scored:
            return {}

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:self.top_k]
        total_w = sum(w for _, w in top)
        weights = {s.symbol: (1.0 / total_w) * (1.0 / len(top)) for s, _ in top}
        return weights


# ----------------------------------------------------------------------
# 7. DrawdownCutStrategy — strict loss management
# ----------------------------------------------------------------------
class DrawdownCutStrategy(ExtendedBaseStrategy):
    """Aggressive drawdown protection with tiered exits.

    - Loss > 8% → full stop-loss
    - Loss > 5% → reduce to half position
    - Gain > 10% → partial profit-taking (reduce by 30%)
    """

    def __init__(self, horizon: str = "5d", name: str = None, tag: str = None,
                 top_k: int = 3, stop_loss: float = 0.08, reduce_thresh: float = 0.05,
                 profit_take: float = 0.10, reduce_pct: float = 0.30):
        super().__init__(horizon=horizon, name=name, tag=tag)
        self.top_k = top_k
        self.stop_loss = stop_loss
        self.reduce_thresh = reduce_thresh
        self.profit_take = profit_take
        self.reduce_pct = reduce_pct

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        valid = [s for s in signals if s.auc_limit != 1]
        ranked = self.rank_stocks(valid, self.horizon)
        new_top = ranked[:self.top_k]

        # Determine new target weights considering existing positions
        weights: Dict[str, float] = {}

        for s in new_top:
            pos = self.get_position_info(s.symbol, cache)
            pred_ret = s.get_horizon_ret(self.horizon)

            if pos is None:
                # New position
                weights[s.symbol] = 1.0 / len(new_top)
            else:
                # Existing position — check drawdown
                entry = pos["entry_price"]
                current = s.entry_price
                unrealized = (current - entry) / entry if entry > 0 else 0.0

                if unrealized < -self.stop_loss:
                    # Stop-loss triggered: exit completely
                    continue
                elif unrealized < -self.reduce_thresh:
                    # Reduce to half
                    weights[s.symbol] = (1.0 / len(new_top)) * 0.5
                elif unrealized > self.profit_take:
                    # Partial profit-taking
                    weights[s.symbol] = (1.0 / len(new_top)) * (1.0 - self.reduce_pct)
                else:
                    # Hold with full weight
                    weights[s.symbol] = 1.0 / len(new_top)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights


# ----------------------------------------------------------------------
# 8. VolumeConfirmationStrategy — volume-filtered entry
# ----------------------------------------------------------------------
class VolumeConfirmationStrategy(ExtendedBaseStrategy):
    """Only buy when BOTH: prediction is positive AND volume is significantly higher than average.

    Rationale: Volume surge confirms the price move is backed by real money flow.
    """

    def __init__(self, horizon: str = "5d", name: str = None, tag: str = None,
                 top_k: int = 3, vol_ratio_thresh: float = 1.5,
                 min_pred_ret: float = 0.005):
        super().__init__(horizon=horizon, name=name, tag=tag)
        self.top_k = top_k
        self.vol_ratio_thresh = vol_ratio_thresh
        self.min_pred_ret = min_pred_ret

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        candidates = []
        for s in signals:
            if s.auc_limit == 1:
                continue
            if s.context is None:
                continue

            vol_ratio = s.context.get("volume_ratio", 1.0)
            pred = s.get_horizon_ret(self.horizon)

            # Volume confirmation: surge AND positive prediction
            if vol_ratio > self.vol_ratio_thresh and pred > self.min_pred_ret:
                # Volume-adjusted score: higher vol_ratio + higher pred = better
                score = pred * math.sqrt(vol_ratio)
                candidates.append((s, score))

        if not candidates:
            # Fallback: if nothing passes filter, return empty
            return {}

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:self.top_k]
        return {s.symbol: 1.0 / len(top) for s, _ in top}


# ----------------------------------------------------------------------
# 9. TrendFollowingStopStrategy — trailing profit-taking
# ----------------------------------------------------------------------
class TrendFollowingStopStrategy(ExtendedBaseStrategy):
    """Trend-following with trailing stop: hold while price keeps making higher highs.

    Stop: exit if price falls more than 6% from running peak (trailing stop).
    Entry: buy top-K with positive 5d prediction + above MA20 (trend confirmation).
    """

    def __init__(self, horizon: str = "5d", name: str = None, tag: str = None,
                 top_k: int = 3, trailing_stop_pct: float = 0.06,
                 ma_filter: float = 1.0):
        super().__init__(horizon=horizon, name=name, tag=tag)
        self.top_k = top_k
        self.trailing_stop_pct = trailing_stop_pct
        self.ma_filter = ma_filter  # price must be > ma_filter×MA20 to enter

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        candidates = []
        for s in signals:
            if s.auc_limit == 1:
                continue
            if s.context is None:
                continue

            pred = s.get_horizon_ret(self.horizon)
            price_vs_ma = s.context.get("price_vs_ma20", 1.0)

            # Trend confirmation: above MA20 + positive prediction
            if price_vs_ma > self.ma_filter and pred > 0.005:
                # Check trailing stop
                pos = self.get_position_info(s.symbol, cache)
                if pos is not None:
                    peak_price = cache.get(f"_peak_{s.symbol}", pos["entry_price"])
                    current_price = s.entry_price
                    drawdown = (peak_price - current_price) / peak_price if peak_price > 0 else 0.0
                    if drawdown > self.trailing_stop_pct:
                        # Stop triggered — do not re-enter
                        continue
                candidates.append((s, pred))

        if not candidates:
            return {}

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:self.top_k]
        return {s.symbol: 1.0 / len(top) for s, _ in top}

    def on_day_end(
        self,
        date: str,
        weights: Dict[str, float],
        realized_rets: Dict[str, float],
        cache: Dict[str, Any],
    ) -> None:
        # Update peak prices from portfolio
        portfolio = cache.get("_portfolio", {})
        for sym, pos_info in portfolio.items():
            entry_price = pos_info.get("entry_price", 0.0)
            current_peak = cache.get(f"_peak_{sym}", entry_price)
            # Peak is tracked at portfolio level; on_day_end can refresh
            if entry_price > current_peak:
                cache[f"_peak_{sym}"] = entry_price


# ----------------------------------------------------------------------
# 10. RSIDivergenceStrategy — divergence-based reversal signals
# ----------------------------------------------------------------------
class RSIDivergenceStrategy(ExtendedBaseStrategy):
    """Detect RSI divergence: price makes new low but RSI doesn't (bullish), or vice versa.

    This requires tracking historical lows. Uses cache to store RSI/price history.
    """

    def __init__(self, horizon: str = "5d", name: str = None, tag: str = None,
                 top_k: int = 3, min_divergence: float = 10.0):
        super().__init__(horizon=horizon, name=name, tag=tag)
        self.top_k = top_k
        self.min_divergence = min_divergence  # minimum RSI divergence to trigger

    def select_positions(
        self,
        signals: List[Signal],
        prev_weights: Dict[str, float],
        cache: Dict[str, Any],
    ) -> Dict[str, float]:
        if not signals:
            return {}

        candidates = []
        for s in signals:
            if s.auc_limit == 1:
                continue
            if s.context is None:
                continue

            rsi = s.context.get("rsi_14", 50.0)
            pred = s.get_horizon_ret(self.horizon)

            # Track RSI history in cache for divergence detection
            rsi_hist = cache.get("_rsi_history", {})
            price_hist = cache.get("_price_history", {})

            sym_rsi = rsi_hist.get(s.symbol, [])
            sym_price = price_hist.get(s.symbol, [])

            sym_rsi.append(rsi)
            sym_price.append(s.entry_price)
            if len(sym_rsi) > 20:
                sym_rsi = sym_rsi[-20:]
            if len(sym_price) > 20:
                sym_price = sym_price[-20:]

            rsi_hist[s.symbol] = sym_rsi
            price_hist[s.symbol] = sym_price
            cache["_rsi_history"] = rsi_hist
            cache["_price_history"] = price_hist

            # Detect divergence
            divergence = 0.0
            if len(sym_rsi) >= 5 and len(sym_price) >= 5:
                recent_price_low = min(sym_price[-5:])
                all_price_low = min(sym_price)
                recent_rsi_low = min(sym_rsi[-5:])
                all_rsi_low = min(sym_rsi)

                # Bullish divergence: price made new low but RSI didn't
                if sym_price[-1] <= recent_price_low and all_price_low == sym_price[-1]:
                    divergence = all_rsi_low - recent_rsi_low
                # Bearish divergence: price made new high but RSI didn't
                elif sym_price[-1] >= max(sym_price[-5:]) and sym_price[-1] == max(sym_price):
                    divergence = -(recent_rsi_low - all_rsi_low)

            # Use divergence as a signal boost
            if abs(divergence) > self.min_divergence and pred > 0.005:
                score = pred + divergence / 100.0
                candidates.append((s, score))
            elif pred > 0.005 and rsi < 40:
                # Plain oversold with positive prediction
                candidates.append((s, pred))

        if not candidates:
            return {}

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:self.top_k]
        return {s.symbol: 1.0 / len(top) for s, _ in top}
