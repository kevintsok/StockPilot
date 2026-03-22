"""
Unit tests for the notify.pipeline module and ConfidenceStrategy.
"""

import pytest
from auto_select_stock.predict.strategies import ConfidenceStrategy
from auto_select_stock.predict.strategies.base import Signal
from auto_select_stock.notify.pipeline import apply_strategy


# ---------------------------------------------------------------------------
# ConfidenceStrategy unit tests
# ---------------------------------------------------------------------------

class TestConfidenceStrategy:
    """Tests for ConfidenceStrategy.select_positions()."""

    def _make_signal(self, symbol: str, predicted_ret: float) -> Signal:
        return Signal(symbol=symbol, predicted_ret=predicted_ret, realized_ret=0.0, industry=None)

    def test_empty_signals_returns_empty_weights(self):
        strategy = ConfidenceStrategy(top_k=5)
        weights = strategy.select_positions([], {}, {})
        assert weights == {}

    def test_top_k_limits_results(self):
        strategy = ConfidenceStrategy(top_k=2)
        signals = [
            self._make_signal("A", 0.05),
            self._make_signal("B", 0.04),
            self._make_signal("C", 0.03),
            self._make_signal("D", 0.02),
        ]
        weights = strategy.select_positions(signals, {}, {})
        assert len(weights) == 2
        assert set(weights.keys()) == {"A", "B"}

    def test_weights_sum_to_one(self):
        strategy = ConfidenceStrategy(top_k=3)
        signals = [
            self._make_signal("A", 0.06),
            self._make_signal("B", 0.03),
            self._make_signal("C", 0.01),
        ]
        weights = strategy.select_positions(signals, {}, {})
        # long_alloc = 1.0 (no short), so weights sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_higher_confidence_gets_higher_weight(self):
        strategy = ConfidenceStrategy(top_k=2)
        signals = [
            self._make_signal("A", 0.09),  # 3x the confidence of B
            self._make_signal("B", 0.03),
        ]
        weights = strategy.select_positions(signals, {}, {})
        assert weights["A"] > weights["B"]
        # A has 3x the confidence, so weight should be ~3x B's weight
        ratio = weights["A"] / weights["B"]
        assert abs(ratio - 3.0) < 1e-9

    def test_negative_returns_are_excluded_long_only(self):
        strategy = ConfidenceStrategy(top_k=5)
        signals = [
            self._make_signal("A", 0.05),
            self._make_signal("B", -0.02),  # negative, should be excluded
            self._make_signal("C", 0.03),
        ]
        weights = strategy.select_positions(signals, {}, {})
        assert "B" not in weights
        assert set(weights.keys()) == {"A", "C"}

    def test_min_confidence_filters_low_signal(self):
        strategy = ConfidenceStrategy(top_k=5, min_confidence=0.02)
        signals = [
            self._make_signal("A", 0.05),
            self._make_signal("B", 0.01),  # below min_confidence
            self._make_signal("C", 0.03),
        ]
        weights = strategy.select_positions(signals, {}, {})
        assert "B" not in weights
        assert set(weights.keys()) == {"A", "C"}

    def test_all_below_min_confidence_returns_empty(self):
        strategy = ConfidenceStrategy(top_k=5, min_confidence=0.02)
        signals = [
            self._make_signal("A", 0.01),
            self._make_signal("B", 0.005),
        ]
        weights = strategy.select_positions(signals, {}, {})
        assert weights == {}

    def test_short_signals_allowed_when_enabled(self):
        """With allow_short=True, both long and short positions are opened."""
        strategy = ConfidenceStrategy(top_k=10, allow_short=True)
        signals = [
            self._make_signal("A", 0.05),
            self._make_signal("B", -0.04),
            self._make_signal("C", 0.03),
            self._make_signal("D", -0.02),
        ]
        weights = strategy.select_positions(signals, {}, {})
        # Longs and shorts must both be present
        assert any(w > 0 for w in weights.values()), "No long positions"
        assert any(w < 0 for w in weights.values()), "No short positions"
        # A and C should be longs (top 2 positive), B and D shorts (top 2 negative)
        assert "A" in weights and "C" in weights
        assert "B" in weights and "D" in weights
        # With enough top_k (10) to consume all signals:
        # long_alloc=0.5, short_alloc=-0.5; weights normalized by total_conf
        # long_conf = 0.05+0.03=0.08, short_conf=0.04+0.02=0.06, total=0.14
        # long_w = 0.5 * 0.08/0.14 ≈ 0.2857; short_w = -0.5 * 0.06/0.14 ≈ -0.2143
        long_w = sum(w for w in weights.values() if w > 0)
        short_w = sum(w for w in weights.values() if w < 0)
        assert long_w == pytest.approx(0.5 * 0.08 / 0.14, abs=1e-9)
        assert short_w == pytest.approx(-0.5 * 0.06 / 0.14, abs=1e-9)

    def test_sorted_by_predicted_ret_descending(self):
        """Top-K should select highest predicted returns first."""
        strategy = ConfidenceStrategy(top_k=2)
        signals = [
            self._make_signal("C", 0.02),
            self._make_signal("A", 0.05),
            self._make_signal("B", 0.03),
        ]
        weights = strategy.select_positions(signals, {}, {})
        # Should pick A (0.05) and B (0.03), not C
        assert set(weights.keys()) == {"A", "B"}

    def test_prev_weights_and_cache_do_not_affect_result(self):
        """select_positions should be stateless — prev_weights and cache ignored."""
        strategy = ConfidenceStrategy(top_k=2)
        signals = [self._make_signal("A", 0.05), self._make_signal("B", 0.03)]
        prev_weights = {"X": 0.5}  # should be ignored
        cache = {"key": "value"}   # should be ignored
        weights = strategy.select_positions(signals, prev_weights, cache)
        assert set(weights.keys()) == {"A", "B"}


# ---------------------------------------------------------------------------
# apply_strategy integration tests
# ---------------------------------------------------------------------------

class TestApplyStrategy:
    """Tests for apply_strategy() using real ConfidenceStrategy."""

    def test_apply_strategy_returns_correct_length(self):
        signals = [(f"00{i:04d}", 0.01 + i * 0.005, 0.0, None) for i in range(20)]
        results = apply_strategy(signals, strategy_name="confidence", top_k=5)
        assert len(results) == 5

    def test_apply_strategy_sorted_by_weight_descending(self):
        signals = [
            ("A", 0.10, 0.0, None),
            ("B", 0.05, 0.0, None),
            ("C", 0.01, 0.0, None),
        ]
        results = apply_strategy(signals, strategy_name="confidence", top_k=3)
        weights = [w for _, _, w in results]
        assert weights == sorted(weights, reverse=True)

    def test_apply_strategy_includes_predicted_return(self):
        signals = [
            ("A", 0.10, 0.0, None),
            ("B", 0.05, 0.0, None),
        ]
        results = apply_strategy(signals, strategy_name="confidence", top_k=2)
        result_dict = {sym: (pred, w) for sym, pred, w in results}
        assert result_dict["A"][0] == pytest.approx(0.10)
        assert result_dict["B"][0] == pytest.approx(0.05)

    def test_apply_strategy_unknown_strategy_raises(self):
        signals = [("A", 0.05, 0.0, None)]
        with pytest.raises(ValueError, match="Unknown strategy"):
            apply_strategy(signals, strategy_name="unknown_strategy", top_k=5)

    def test_apply_strategy_empty_signals_returns_empty(self):
        results = apply_strategy([], strategy_name="confidence", top_k=5)
        assert results == []

    def test_apply_strategy_respects_top_k(self):
        signals = [(f"00{i:04d}", 0.01 + i * 0.001, 0.0, None) for i in range(100)]
        results = apply_strategy(signals, strategy_name="confidence", top_k=10)
        assert len(results) == 10
        # All symbols should be unique
        assert len({sym for sym, _, _ in results}) == 10
