"""
Unit tests for the inference module (predict/inference.py).

Tests PricePredictor class and predict_next_close function
with mock model checkpoints.
"""

import math
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from auto_select_stock.predict.inference import PricePredictor, load_model, predict_next_close


class TestPricePredictor:
    """Tests for the PricePredictor class."""

    def test_predictor_init_with_mock(self, mock_predictor):
        """PricePredictor initializes correctly with mock."""
        assert mock_predictor.cfg is not None
        assert mock_predictor.scaler is not None
        assert len(mock_predictor.horizons) > 0
        assert mock_predictor.close_idx == 3  # close column index

    def test_predict_returns_dict_by_default(self, mock_predictor):
        """predict() returns dict when horizon is None."""
        result = mock_predictor.predict("000001")
        assert isinstance(result, dict)
        assert "1d" in result
        assert "3d" in result
        assert "5d" in result

    def test_predict_returns_float_for_specific_horizon(self, mock_predictor):
        """predict() returns float when horizon is specified."""
        result = mock_predictor.predict("000001", horizon="1d")
        assert isinstance(result, float)

    def test_predict_with_horizon_int(self, mock_predictor):
        """predict() handles integer horizon."""
        result = mock_predictor.predict("000001", horizon=5)
        assert isinstance(result, float)

    def test_predict_all_horizons_present(self, mock_predictor):
        """All configured horizons are returned in dict mode."""
        result = mock_predictor.predict("000001")
        expected_horizons = ["1d", "3d", "5d", "7d", "14d", "20d"]
        for h in expected_horizons:
            assert h in result, f"Missing horizon {h}"

    def test_close_idx_matches_feature_columns(self, mock_predictor):
        """close_idx correctly points to 'close' in feature_columns."""
        assert mock_predictor.feature_columns[mock_predictor.close_idx] == "close"


class TestLoadModel:
    """Tests for load_model() with checkpoint handling."""

    def test_load_model_nonexistent_file(self):
        """load_model raises FileNotFoundError for missing checkpoint."""
        with pytest.raises(FileNotFoundError):
            load_model(Path("/nonexistent/path/model.pt"))


class TestPredictNextClose:
    """Tests for predict_next_close() convenience function."""

    def test_predict_next_close_creates_predictor(self, tmp_path, monkeypatch):
        """predict_next_close creates and uses PricePredictor."""
        mock_predictor = MagicMock()
        # When horizon is specified, predict returns a float
        mock_predictor.predict.return_value = 0.05

        def mock_predictor_init(path, device=None):
            return mock_predictor

        monkeypatch.setattr(
            "auto_select_stock.predict.inference.PricePredictor",
            mock_predictor_init
        )

        result = predict_next_close(
            "000001",
            checkpoint_path=tmp_path / "model.pt",
            horizon="1d",
        )
        assert result == 0.05


class TestMockPredictorIntegration:
    """Integration tests using the mock_predictor fixture."""

    def test_mock_predictor_with_seq_len(self, mock_predictor):
        """Mock predictor handles seq_len parameter."""
        result = mock_predictor.predict("000001", seq_len=60)
        assert isinstance(result, dict)

    def test_mock_predictor_horizons_match(self, mock_predictor):
        """Mock predictor horizons match config horizons."""
        result = mock_predictor.predict("000001")
        assert set(result.keys()) == set(f"{h}d" for h in mock_predictor.horizons)

    def test_mock_predictor_device_is_cpu(self, mock_predictor):
        """Mock predictor device is set to cpu."""
        assert mock_predictor.device == "cpu"


# ---------------------------------------------------------------------------
# Mock helpers for use in other tests
# ---------------------------------------------------------------------------


def make_mock_predictor() -> PricePredictor:
    """Factory function to create a mock PricePredictor (used by other test modules)."""
    from tests.conftest import MockPredictor

    class _BareMockPredictor:
        """Minimal mock that simulates PricePredictor interface."""

        def __init__(self):
            self.cfg = MagicMock()
            self.cfg.seq_len = 60
            self.cfg.price_columns = [
                "open", "high", "low", "close", "volume", "amount",
                "turnover_rate", "volume_ratio", "pct_change", "amplitude", "change_amount",
            ]
            self.cfg.financial_columns = [
                "roe", "net_profit_margin", "gross_margin",
                "operating_cashflow_growth", "debt_to_asset",
                "eps", "operating_cashflow_per_share",
            ]
            self.cfg.target_mode = "close"
            self.scaler = {
                "mean": np.zeros(30, dtype="float32"),
                "std": np.ones(30, dtype="float32"),
            }
            self.horizons = [1, 3, 5, 7, 14, 20]
            self.feature_columns = (
                self.cfg.price_columns
                + self.cfg.financial_columns
                + [
                    "rsi_14", "macd_line", "macd_signal", "macd_hist",
                    "bb_position", "bb_width", "volume_ma5", "volume_ma20",
                    "atr_14", "stoch_k", "stoch_d", "obv_ma10", "roc_10", "momentum_10",
                ]
            )
            self.close_idx = 3
            self.device = "cpu"
            self.model = MagicMock()

            # Multi-head model returns 4 tensors
            def mock_model_forward(x):
                batch, seq_len, _ = x.shape
                num_horizons = len(self.horizons)
                reg_all = np.random.randn(num_horizons, batch, seq_len).astype("float32") * 0.01
                return (
                    np.random.randn(batch, seq_len, 2).astype("float32"),
                    reg_all,
                    np.random.randn(batch, seq_len, 64).astype("float32"),
                    np.random.randn(batch, 8, seq_len, seq_len).astype("float32"),
                )

            self.model.return_value = mock_model_forward(x)

        def predict(self, symbol, seq_len=None, base_dir=None, features=None, horizon=None):
            results = {
                "1d": 0.02,
                "3d": 0.06,
                "5d": 0.10,
                "7d": 0.13,
                "14d": 0.20,
                "20d": 0.25,
            }
            if horizon is not None:
                h_str = str(horizon) if isinstance(horizon, int) else horizon
                if not h_str.endswith("d"):
                    h_str = f"{h_str}d"
                return results.get(h_str, results["1d"])
            return results

    return _BareMockPredictor()
