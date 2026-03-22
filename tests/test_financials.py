"""
Unit tests for the financials module (data/financials.py).

Tests financial data fetching and parsing with mocked network calls.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from auto_select_stock.data import financials


class TestLatestQuarterEnd:
    """Tests for _latest_quarter_end()."""

    def test_q1_returns_march(self):
        """For date in Q2, returns March 31."""
        ref = date(2024, 4, 15)
        result = financials._latest_quarter_end(ref)
        assert result == date(2024, 3, 31)

    def test_q2_returns_june(self):
        """For date in Q3, returns June 30."""
        ref = date(2024, 8, 1)
        result = financials._latest_quarter_end(ref)
        assert result == date(2024, 6, 30)

    def test_q3_returns_september(self):
        """For date in Q4, returns September 30."""
        ref = date(2024, 11, 1)
        result = financials._latest_quarter_end(ref)
        assert result == date(2024, 9, 30)

    def test_q4_returns_december(self):
        """For date in January of next year, returns December 31."""
        ref = date(2025, 1, 15)
        result = financials._latest_quarter_end(ref)
        assert result == date(2024, 12, 31)

    def test_exact_quarter_end(self):
        """Returns the same date if ref is exactly a quarter end."""
        ref = date(2024, 6, 30)
        result = financials._latest_quarter_end(ref)
        assert result == date(2024, 6, 30)


class TestCleanNumeric:
    """Tests for _clean_numeric()."""

    def test_removes_percent_sign(self):
        """Removes % sign from values."""
        s = pd.Series(["10.5%", "20%", "15.5%"])
        result = financials._clean_numeric(s)
        assert result.iloc[0] == 10.5
        assert result.iloc[1] == 20.0

    def test_removes_commas(self):
        """Removes comma separators from values."""
        s = pd.Series(["1,000", "2,500", "3,750"])
        result = financials._clean_numeric(s)
        assert result.iloc[0] == 1000.0

    def test_converts_empty_string_to_zero(self):
        """Empty strings become 0.0."""
        s = pd.Series(["10", "", "20"])
        result = financials._clean_numeric(s)
        assert result.iloc[0] == 10.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == 20.0


class TestFormatEmSymbol:
    """Tests for _format_em_symbol()."""

    def test_adds_sh_for_6_prefix(self):
        """6xxxxx symbols get .SH suffix."""
        result = financials._format_em_symbol("600000")
        assert result == "600000.SH"

    def test_adds_sz_for_0_prefix(self):
        """0xxxxx symbols get .SZ suffix."""
        result = financials._format_em_symbol("000001")
        assert result == "000001.SZ"

    def test_adds_bj_for_4_or_8_prefix(self):
        """4xxxxx and 8xxxxx symbols get .BJ suffix."""
        assert financials._format_em_symbol("430001") == "430001.BJ"
        assert financials._format_em_symbol("830001") == "830001.BJ"

    def test_preserves_existing_suffix(self):
        """Already formatted symbols are unchanged."""
        result = financials._format_em_symbol("600000.SH")
        assert result == "600000.SH"


class TestTidyIndicatorEm:
    """Tests for _tidy_indicator_em()."""

    def _make_em_df(self) -> pd.DataFrame:
        """Create a mock Eastmoney indicator DataFrame."""
        return pd.DataFrame({
            "REPORT_DATE": ["2024-01-01", "2024-04-01", "2024-07-01"],
            "NOTICE_DATE": ["2024-01-15", "2024-04-20", "2024-07-15"],
            "ROEJQ": [15.0, 12.0, 14.0],
            "XSJLL": [10.0, 8.0, 9.0],
            "XSMLL": [30.0, 28.0, 29.0],
            "ZCFZL": [0.4, 0.5, 0.45],
            "EPSJB": [0.8, 0.6, 0.7],
            "MGJYXJJE": [1.0, 0.8, 0.9],
        })

    def test_renames_columns_correctly(self):
        """Maps Eastmoney column names to standard names."""
        raw = self._make_em_df()
        result = financials._tidy_indicator_em(raw, "600000")
        assert "roe" in result.columns
        assert "net_profit_margin" in result.columns
        assert "gross_margin" in result.columns
        assert "debt_to_asset" in result.columns
        assert "eps" in result.columns
        assert "operating_cashflow_per_share" in result.columns

    def test_adds_operating_cashflow_growth(self):
        """Adds operating_cashflow_growth column via pct_change."""
        raw = self._make_em_df()
        result = financials._tidy_indicator_em(raw, "600000")
        assert "operating_cashflow_growth" in result.columns
        # Growth from 1.0 to 0.8 is -20% (use pytest.approx for float precision)
        assert result["operating_cashflow_growth"].iloc[1] == pytest.approx(-20.0)

    def test_sorts_by_date(self):
        """Results are sorted by date ascending."""
        raw = self._make_em_df()
        result = financials._tidy_indicator_em(raw, "600000")
        assert result["date"].iloc[0] < result["date"].iloc[1]

    def test_raises_on_empty_dataframe(self):
        """Raises RuntimeError when DataFrame is empty."""
        raw = pd.DataFrame()
        with pytest.raises(RuntimeError, match="No Eastmoney indicator data"):
            financials._tidy_indicator_em(raw, "600000")

    def test_raises_on_none_dataframe(self):
        """Raises RuntimeError when DataFrame is None."""
        with pytest.raises(RuntimeError, match="No Eastmoney indicator data"):
            financials._tidy_indicator_em(None, "600000")


class TestFetchFinancials:
    """Tests for fetch_financials()."""

    def _make_em_result(self) -> pd.DataFrame:
        return pd.DataFrame({
            "REPORT_DATE": ["2024-01-01", "2024-04-01"],
            "NOTICE_DATE": ["2024-01-15", "2024-04-20"],
            "ROEJQ": [15.0, 12.0],
            "XSJLL": [10.0, 8.0],
            "XSMLL": [30.0, 28.0],
            "ZCFZL": [0.4, 0.5],
            "EPSJB": [0.8, 0.6],
            "MGJYXJJE": [1.0, 0.8],
        })

    @patch.object(financials, "_run_with_timeout")
    @patch.object(financials.ak, "stock_financial_analysis_indicator_em")
    def test_prefers_eastmoney_indicator(self, mock_em, mock_timeout):
        """Uses Eastmoney indicator endpoint as first choice."""
        mock_timeout.return_value = self._make_em_result()
        result = financials.fetch_financials("600000")
        assert len(result) == 2
        assert "roe" in result.columns

    @patch.object(financials, "_tidy_financial_abstract")
    @patch.object(financials, "_run_with_timeout")
    @patch.object(financials.ak, "stock_financial_analysis_indicator_em")
    def test_retries_on_timeout(self, mock_em, mock_timeout, mock_tidy_abstract):
        """Retries up to configured attempts on timeout."""
        mock_timeout.side_effect = [
            TimeoutError("timeout"),
            TimeoutError("timeout"),
            self._make_em_result(),
        ]
        # _tidy_financial_abstract receives Eastmoney-format data from _make_em_result
        # but expects Sina format, so we mock it to return a valid result
        mock_tidy_abstract.return_value = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "roe": [15.0, 12.0],
            "net_profit_margin": [10.0, 8.0],
            "gross_margin": [30.0, 28.0],
            "operating_cashflow_growth": [5.0, 3.0],
            "debt_to_asset": [0.4, 0.5],
            "eps": [0.8, 0.6],
            "operating_cashflow_per_share": [1.0, 0.8],
            "publish_date": pd.to_datetime(["2024-01-15", "2024-04-20"]),
            "report_date": pd.to_datetime(["2024-01-15", "2024-04-20"]),
        })
        result = financials.fetch_financials("600000", retries=3)
        assert len(result) > 0


class TestFetchAndStoreFinancials:
    """Tests for fetch_and_store_financials()."""

    def _make_financial_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "publish_date": pd.to_datetime(["2024-01-15", "2024-04-20"]),
            "report_date": pd.to_datetime(["2024-01-15", "2024-04-20"]),
            "roe": [15.0, 12.0],
            "net_profit_margin": [10.0, 8.0],
            "gross_margin": [30.0, 28.0],
            "operating_cashflow_growth": [5.0, 3.0],
            "debt_to_asset": [0.4, 0.5],
            "eps": [0.8, 0.6],
            "operating_cashflow_per_share": [1.0, 0.8],
        })

    @patch.object(financials, "load_financial")
    @patch.object(financials, "fetch_financials")
    def test_skips_when_up_to_date(self, mock_fetch, mock_load):
        """Skips fetch when latest quarter already exists."""
        # Must include Q4 2025 (2025-12-31) since today is 2026-03-22 and
        # latest_qe = Q4 2025 (2025-12-31). Existing must be >= latest_qe.
        existing = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]),
        })
        mock_load.return_value = existing

        result = financials.fetch_and_store_financials("600000")
        mock_fetch.assert_not_called()

    @patch.object(financials, "save_financial")
    @patch.object(financials, "load_financial")
    @patch.object(financials, "fetch_financials")
    def test_combines_existing_and_fresh(self, mock_fetch, mock_load, mock_save):
        """Combines existing data with newly fetched data."""
        existing = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "roe": [15.0],
        })
        mock_load.return_value = existing
        mock_fetch.return_value = self._make_financial_df()

        financials.fetch_and_store_financials("600000")
        # save_financial should have been called with combined data
        mock_save.assert_called_once()
        call_args = mock_save.call_args[0][0]
        assert len(call_args) >= 2  # combined rows
