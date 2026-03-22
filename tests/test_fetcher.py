"""
Unit tests for the data fetcher module (data/fetcher.py).

Tests price history fetching, date detection, and incremental update logic
with mocked network calls.
"""

from datetime import date
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from auto_select_stock.data import fetcher


class TestSymbolPrefix:
    """Tests for _symbol_with_prefix()."""

    def test_sh_prefix_for_6xxx(self):
        """6xxx symbols get sh prefix."""
        assert fetcher._symbol_with_prefix("600000") == "sh600000"

    def test_sz_prefix_for_0xxx(self):
        """0xxx symbols get sz prefix."""
        assert fetcher._symbol_with_prefix("000001") == "sz000001"

    def test_sz_prefix_for_3xxx(self):
        """3xxx symbols (创业板) get sz prefix."""
        assert fetcher._symbol_with_prefix("300001") == "sz300001"


class TestFinalizeHistoryDf:
    """Tests for _finalize_history_df()."""

    def _make_raw_df(self) -> pd.DataFrame:
        """Create a minimal raw DataFrame matching akshare output."""
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.5, 11.0],
            "high": [10.5, 11.0, 11.5],
            "low": [9.5, 10.0, 10.5],
            "close": [10.2, 10.8, 11.2],
            "volume": [1_000_000, 1_100_000, 1_200_000],
            "amount": [10_200_000, 11_000_000, 12_000_000],
        })

    def test_adds_derived_columns(self):
        """Derived columns (pct_change, amplitude, etc.) are computed."""
        raw = self._make_raw_df()
        result = fetcher._finalize_history_df(raw)
        assert "pct_change" in result.columns
        assert "amplitude" in result.columns
        assert "change_amount" in result.columns
        assert "turnover_rate" in result.columns
        assert "volume_ratio" in result.columns

    def test_date_column_normalized(self):
        """Date column is converted to datetime and sorted."""
        raw = self._make_raw_df()
        result = fetcher._finalize_history_df(raw)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result["date"].is_monotonic_increasing

    def test_handles_missing_turnover_column(self):
        """When 'turnover' column exists (not turnover_rate), it is renamed."""
        raw = self._make_raw_df()
        raw = raw.rename(columns={"volume": "turnover"})  # older akshare format
        raw["turnover"] = [1.5, 1.6, 1.7]
        result = fetcher._finalize_history_df(raw)
        assert "turnover_rate" in result.columns


class TestLastTradingDate:
    """Tests for _last_trading_date()."""

    @patch.object(fetcher, "_is_trading_day_sse", return_value=None)
    @patch.object(fetcher.ak, "tool_trade_date_hist_sina")
    def test_falls_back_to_sina_calendar(self, mock_sina, mock_is_trading):
        """Falls back to Sina trade calendar when SSE check returns None."""
        mock_sina.return_value = pd.DataFrame({
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-10"]),
        })
        ref = pd.Timestamp("2024-01-10")
        result = fetcher._last_trading_date(ref)
        assert result == pd.Timestamp("2024-01-10")

    @patch.object(fetcher, "_is_trading_day_sse", return_value=True)
    def test_uses_sse_fast_path(self, mock_is_trade):
        """Uses SSE daily check when available."""
        ref = pd.Timestamp("2024-01-10")
        result = fetcher._last_trading_date(ref)
        assert result == pd.Timestamp("2024-01-10")

    def test_saturday_adjusted_to_friday(self):
        """Saturday references are adjusted to previous Friday."""
        # 2024-01-06 is a Saturday
        ref = pd.Timestamp("2024-01-06")
        with patch.object(fetcher, "_is_trading_day_sse", return_value=None), \
             patch.object(fetcher.ak, "tool_trade_date_hist_sina", side_effect=Exception("no data")):
            result = fetcher._last_trading_date(ref)
        assert result.weekday() == 4  # Friday

    def test_sunday_adjusted_to_friday(self):
        """Sunday references are adjusted to previous Friday."""
        # 2024-01-07 is a Sunday
        ref = pd.Timestamp("2024-01-07")
        with patch.object(fetcher, "_is_trading_day_sse", return_value=None), \
             patch.object(fetcher.ak, "tool_trade_date_hist_sina", side_effect=Exception("no data")):
            result = fetcher._last_trading_date(ref)
        assert result.weekday() == 4  # Friday


class TestListAllSymbols:
    """Tests for list_all_symbols()."""

    @patch.object(fetcher.ak, "stock_info_a_code_name")
    def test_returns_symbols_from_code_name(self, mock_code_name):
        """Returns symbols from stock_info_a_code_name endpoint."""
        mock_code_name.return_value = pd.DataFrame({
            "code": ["600000", "000001", "300001"],
        })
        result = fetcher.list_all_symbols()
        assert result == ["600000", "000001", "300001"]

    @patch.object(fetcher.ak, "stock_info_a_code_name", side_effect=Exception("api error"))
    @patch.object(fetcher.ak, "stock_zh_a_spot_em")
    def test_falls_back_to_spot_em(self, mock_spot, mock_code_name):
        """Falls back to spot_em endpoint on error."""
        mock_spot.return_value = pd.DataFrame({
            "代码": ["600000", "000001"],
        })
        result = fetcher.list_all_symbols()
        assert result == ["600000", "000001"]

    @patch.object(fetcher.ak, "stock_info_a_code_name", side_effect=Exception("api error"))
    @patch.object(fetcher.ak, "stock_zh_a_spot_em", side_effect=Exception("api error"))
    @patch.object(fetcher, "_fallback_symbols", return_value=[])
    def test_raises_when_all_endpoints_fail(self, mock_fallback, mock_spot, mock_code_name):
        """Raises RuntimeError when all endpoints fail and no cached symbols."""
        with pytest.raises(RuntimeError, match="Unable to fetch symbol list"):
            fetcher.list_all_symbols()


class TestFetchHistory:
    """Tests for fetch_history()."""

    def _mock_raw_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.5, 11.0],
            "high": [10.5, 11.0, 11.5],
            "low": [9.5, 10.0, 10.5],
            "close": [10.2, 10.8, 11.2],
            "volume": [1_000_000, 1_100_000, 1_200_000],
            "amount": [10_200_000, 11_000_000, 12_000_000],
            "outstanding_share": [1e9, 1e9, 1e9],
            "turnover": [1.5, 1.6, 1.7],
        })

    @patch.object(fetcher, "_run_with_timeout")
    @patch.object(fetcher.ak, "stock_zh_a_daily")
    def test_returns_normalized_dataframe(self, mock_ak, mock_timeout):
        """Returns a properly normalized DataFrame with derived columns."""
        mock_timeout.return_value = self._mock_raw_df()
        result = fetcher.fetch_history("600000", start_date="2024-01-01")
        assert "pct_change" in result.columns
        assert "amplitude" in result.columns
        assert "change_amount" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    @patch.object(fetcher, "_run_with_timeout")
    @patch.object(fetcher.ak, "stock_zh_a_daily")
    def test_raises_when_empty_response(self, mock_ak, mock_timeout):
        """Raises RuntimeError when fetch returns empty data."""
        mock_timeout.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="Failed to fetch history"):
            fetcher.fetch_history("600000")

    @patch.object(fetcher, "_run_with_timeout")
    @patch.object(fetcher.ak, "stock_zh_a_daily")
    def test_retries_on_timeout(self, mock_ak, mock_timeout):
        """Retries up to configured number of attempts on timeout."""
        mock_timeout.side_effect = [TimeoutError("timeout"), TimeoutError("timeout"), self._mock_raw_df()]
        result = fetcher.fetch_history("600000", retries=3)
        assert len(result) > 0


class TestDetectTradingDates:
    """Tests for detect_trading_dates via _is_trading_day_sse."""

    def test_sse_returns_true_on_trading_day(self):
        """SSE check returns True for trading days."""
        with patch.object(fetcher.ak, "stock_sse_deal_daily") as mock_sse:
            mock_sse.return_value = pd.DataFrame({"date": ["20240102"]})
            result = fetcher._is_trading_day_sse(pd.Timestamp("2024-01-02"))
        assert result is True

    def test_sse_returns_false_on_non_trading_day(self):
        """SSE check returns False for non-trading days."""
        with patch.object(fetcher.ak, "stock_sse_deal_daily") as mock_sse:
            mock_sse.return_value = pd.DataFrame()  # empty
            result = fetcher._is_trading_day_sse(pd.Timestamp("2024-01-01"))
        assert result is False

    def test_sse_returns_none_on_exception(self):
        """SSE check returns None when API raises exception."""
        with patch.object(fetcher.ak, "stock_sse_deal_daily", side_effect=Exception("api error")):
            result = fetcher._is_trading_day_sse(pd.Timestamp("2024-01-01"))
        assert result is None
