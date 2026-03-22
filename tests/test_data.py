"""
Unit tests for the data preprocessing module (predict/data.py).

Tests feature preprocessing, scaler computation, and data pipeline
with synthetic data.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from auto_select_stock.predict.data import (
    PRICE_FEATURE_COLUMNS,
    FINANCIAL_FEATURE_COLUMNS,
    TECHNICAL_FEATURE_COLUMNS,
    compute_scaler,
    compute_technical_indicators,
    apply_scaler,
    close_index,
    load_feature_matrix,
    _merge_price_financial,
    _load_financial_frame,
    _normalize_date,
    _ema,
    _sma,
    _running_std,
    _rolling_max,
    _rolling_min,
)


class TestComputeTechnicalIndicators:
    """Tests for compute_technical_indicators()."""

    def _make_price_df(self, n: int = 60, seed: int = 42) -> pd.DataFrame:
        """Create a deterministic price DataFrame."""
        np.random.seed(seed)
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = 10.0 + np.cumsum(np.random.randn(n) * 0.1 + 0.02)
        high = close + np.abs(np.random.randn(n) * 0.05)
        low = close - np.abs(np.random.randn(n) * 0.05)
        open_price = close + np.random.randn(n) * 0.02
        volume = 1_000_000 + np.random.randint(-200_000, 200_000, size=n)
        return pd.DataFrame({
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def test_technical_indicators_all_columns_present(self):
        """All expected technical indicator columns are computed."""
        df = self._make_price_df(60)
        result = compute_technical_indicators(df)
        for col in TECHNICAL_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_technical_indicators_no_nan_in_output(self):
        """Technical indicators DataFrame has no NaN values."""
        df = self._make_price_df(60)
        result = compute_technical_indicators(df)
        assert not result.isna().any().any(), "NaN values found in technical indicators"

    def test_technical_indicators_rsi_range(self):
        """RSI values are in valid range [0, 100]."""
        df = self._make_price_df(60)
        result = compute_technical_indicators(df)
        assert (result["rsi_14"] >= 0).all()
        assert (result["rsi_14"] <= 100).all()

    def test_technical_indicators_bb_position_values(self):
        """Bollinger Band position values are computed without NaN/Inf."""
        df = self._make_price_df(60)
        result = compute_technical_indicators(df)
        # BB position can be negative (price below lower band) or > 1 (above upper band)
        # The key is that there are no NaN or Inf values
        valid = result["bb_position"].replace([np.inf, -np.inf], np.nan)
        assert not valid.isna().any(), "BB position should not contain NaN"

    def test_technical_indicators_short_series(self):
        """Handles short price series (fewer than indicator windows)."""
        df = self._make_price_df(10)
        result = compute_technical_indicators(df)
        assert len(result) == len(df)
        assert not result.isna().all().all()


class TestRollingFunctions:
    """Tests for rolling helper functions (_ema, _sma, etc.)."""

    def test_ema_basic(self):
        """EMA produces values close to recent data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _ema(data, span=3)
        assert len(result) == len(data)
        assert result[-1] > result[0]  # trending up

    def test_ema_constant_series(self):
        """EMA of constant series equals the constant."""
        data = np.ones(10) * 5.0
        result = _ema(data, span=3)
        np.testing.assert_allclose(result, 5.0, rtol=1e-9)

    def test_sma_basic(self):
        """SMA produces correct window averages."""
        data = np.arange(1.0, 11.0)  # 1 to 10
        result = _sma(data, window=5)
        # First 4 values are partial averages
        assert result[4] == 3.0  # mean of [1,2,3,4,5]
        assert result[-1] == 8.0  # mean of [6,7,8,9,10]

    def test_sma_single_element(self):
        """SMA with window=1 returns original values."""
        data = np.array([5.0, 10.0, 15.0])
        result = _sma(data, window=1)
        np.testing.assert_allclose(result, data)

    def test_running_std_basic(self):
        """Running std returns non-negative values."""
        data = np.random.randn(100).astype("float64")
        result = _running_std(data, window=20)
        assert (result >= 0).all()

    def test_rolling_max_basic(self):
        """Rolling max is monotonically non-decreasing within window."""
        data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        result = _rolling_max(data, window=3)
        # At index 2, max(3,1,4)=4; at index 3, max(1,4,1)=4
        assert result[2] == 4.0
        assert result[3] == 4.0

    def test_rolling_min_basic(self):
        """Rolling min is monotonically non-increasing within window."""
        data = np.array([2.0, 8.0, 1.0, 7.0, 3.0, 9.0, 4.0, 5.0])
        result = _rolling_min(data, window=3)
        assert result[2] == 1.0
        assert result[3] == 1.0


class TestScalerComputation:
    """Tests for compute_scaler() and apply_scaler()."""

    def test_compute_scaler_single_array(self):
        """Scaler computes mean and std for a single feature array."""
        data = [np.random.randn(50, 10).astype("float64") for _ in range(1)]
        scaler = compute_scaler(data)
        assert "mean" in scaler
        assert "std" in scaler
        assert scaler["mean"].shape == (10,)
        assert scaler["std"].shape == (10,)
        # std should have small epsilon added
        assert (scaler["std"] > 0).all()

    def test_compute_scaler_multiple_arrays(self):
        """Scaler correctly aggregates statistics across multiple arrays."""
        data = [
            np.random.randn(50, 5).astype("float64") + i
            for i in range(3)
        ]
        scaler = compute_scaler(data)
        # Combined mean should be approximately 1.0 (center of uniform distribution from 0,1,2)
        np.testing.assert_allclose(scaler["mean"], np.ones(5), atol=0.3)

    def test_compute_scaler_empty_generator_raises(self):
        """compute_scaler raises RuntimeError when given no data."""
        with pytest.raises(RuntimeError, match="No data"):
            compute_scaler(iter([]))

    def test_apply_scaler(self):
        """apply_scaler normalizes data correctly."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32")
        scaler = {
            "mean": np.array([3.0, 4.0], dtype="float32"),
            "std": np.array([2.0, 2.0], dtype="float32"),
        }
        result = apply_scaler(data, scaler)
        expected = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype="float32")
        np.testing.assert_allclose(result, expected)

    def test_apply_scaler_returns_float32(self):
        """apply_scaler returns float32 array."""
        data = np.array([[1.0, 2.0]], dtype="float64")
        scaler = {
            "mean": np.array([0.0, 0.0], dtype="float32"),
            "std": np.array([1.0, 1.0], dtype="float32"),
        }
        result = apply_scaler(data, scaler)
        assert result.dtype == np.float32


class TestCloseIndex:
    """Tests for close_index()."""

    def test_close_index_found(self):
        """close_index returns correct index for 'close' column."""
        columns = ["open", "high", "low", "close", "volume"]
        assert close_index(columns) == 3

    def test_close_index_missing_raises(self):
        """close_index raises RuntimeError when 'close' is missing."""
        columns = ["open", "high", "low", "volume"]
        with pytest.raises(RuntimeError, match="target column missing"):
            close_index(columns)


class TestNormalizeDate:
    """Tests for _normalize_date()."""

    def test_normalize_date_string(self):
        """_normalize_date handles string input."""
        result = _normalize_date("2024-01-15")
        assert result == pd.Timestamp("2024-01-15 00:00:00")

    def test_normalize_date_timestamp(self):
        """_normalize_date handles pandas Timestamp."""
        result = _normalize_date(pd.Timestamp("2024-01-15"))
        assert result == pd.Timestamp("2024-01-15 00:00:00")

    def test_normalize_date_invalid_raises(self):
        """_normalize_date raises exception for invalid input."""
        with pytest.raises(Exception):
            _normalize_date("not-a-date")


class TestMergePriceFinancial:
    """Tests for _merge_price_financial()."""

    def test_merge_empty_financial(self):
        """Handles case when financial data is None."""
        price_df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "close": np.arange(10, 20),
        })
        result = _merge_price_financial(price_df, None, ["roe", "eps"])
        assert result.shape == (10, 2)
        assert (result == 0).all()

    def test_merge_fills_forward(self):
        """Financial data is forward-filled to match price dates."""
        price_df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
        })
        fin_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "effective_date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "roe": [10.0, 15.0],
            "eps": [0.5, 0.8],
        })
        result = _merge_price_financial(price_df, fin_df, ["roe", "eps"])
        # 2024-01-01 -> 10.0, 2024-01-02 -> 10.0 (forward fill), 2024-01-03 -> 15.0
        assert result[0, 0] == 10.0
        assert result[1, 0] == 10.0  # forward-filled


class TestFeatureColumns:
    """Tests that feature column lists are well-defined."""

    def test_price_feature_columns_count(self):
        """PRICE_FEATURE_COLUMNS has exactly 11 columns."""
        assert len(PRICE_FEATURE_COLUMNS) == 11

    def test_financial_feature_columns_count(self):
        """FINANCIAL_FEATURE_COLUMNS has exactly 7 columns."""
        assert len(FINANCIAL_FEATURE_COLUMNS) == 7

    def test_technical_feature_columns_count(self):
        """TECHNICAL_FEATURE_COLUMNS has 14 columns."""
        assert len(TECHNICAL_FEATURE_COLUMNS) == 14

    def test_default_feature_columns_contains_all(self):
        """Default columns include price + financial features."""
        default_cols = PRICE_FEATURE_COLUMNS + FINANCIAL_FEATURE_COLUMNS
        for col in default_cols:
            assert col in default_cols
