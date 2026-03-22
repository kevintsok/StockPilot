"""
Unit tests for the storage module (storage.py).

Tests SQLite database operations: upsert, query, and utility functions.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_select_stock import storage


class TestEnsureDataDir:
    """Tests for ensure_data_dir()."""

    def test_creates_directory(self, tmp_path):
        """ensure_data_dir creates the directory if it doesn't exist."""
        target = tmp_path / "new_data_dir"
        result = storage.ensure_data_dir(target)
        assert result == target
        assert target.exists()
        assert target.is_dir()

    def test_returns_existing_directory(self, tmp_path):
        """ensure_data_dir returns existing directory unchanged."""
        target = tmp_path / "existing"
        target.mkdir()
        result = storage.ensure_data_dir(target)
        assert result == target


class TestDatabaseSchema:
    """Tests that database schema is initialized correctly."""

    def test_price_table_columns(self, temp_db):
        """Price table has all expected columns."""
        conn = sqlite3.connect(temp_db / "stock.db")
        cur = conn.execute("PRAGMA table_info(price)")
        columns = {row[1] for row in cur.fetchall()}
        expected = {
            "symbol", "date", "open", "high", "low", "close",
            "volume", "amount", "turnover_rate", "volume_ratio",
            "pct_change", "amplitude", "change_amount",
        }
        assert columns == expected
        conn.close()

    def test_financial_table_columns(self, temp_db):
        """Financial table has all expected columns."""
        conn = sqlite3.connect(temp_db / "stock.db")
        cur = conn.execute("PRAGMA table_info(financial)")
        columns = {row[1] for row in cur.fetchall()}
        expected = {
            "symbol", "date", "roe", "net_profit_margin", "gross_margin",
            "operating_cashflow_growth", "debt_to_asset",
            "eps", "operating_cashflow_per_share",
        }
        assert columns == expected
        conn.close()


class TestSaveStockHistory:
    """Tests for save_stock_history()."""

    def test_save_and_retrieve(self, temp_db):
        """Data saved via save_stock_history can be retrieved."""
        from auto_select_stock.data.storage import save_stock_history

        arr = np.array(
            [(f"2024-01-{i:02d}", 10.0 + i, 11.0 + i, 9.0 + i, 10.5 + i,
              1000000, 10500000, 1.5, 1.2, 0.5, 2.0, 0.5)
             for i in range(1, 32)],
            dtype=[
                ("date", "datetime64[D]"),
                ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
                ("volume", "f8"), ("amount", "f8"), ("turnover_rate", "f8"),
                ("volume_ratio", "f8"), ("pct_change", "f8"),
                ("amplitude", "f8"), ("change_amount", "f8"),
            ],
        )
        save_stock_history("000001", arr, base_dir=temp_db)

        # Verify by reading back
        conn = sqlite3.connect(temp_db / "stock.db")
        cur = conn.execute("SELECT COUNT(*) FROM price WHERE symbol=?", ("000001",))
        count = cur.fetchone()[0]
        conn.close()
        assert count == 31

    def test_upsert_updates_existing(self, temp_db):
        """save_stock_history updates existing rows on conflict."""
        from auto_select_stock.data.storage import save_stock_history

        arr1 = np.array(
            [("2024-01-15", 10.0, 11.0, 9.0, 10.5, 1000000, 10500000, 1.5, 1.2, 0.5, 2.0, 0.5)],
            dtype=[
                ("date", "datetime64[D]"), ("open", "f8"), ("high", "f8"),
                ("low", "f8"), ("close", "f8"), ("volume", "f8"), ("amount", "f8"),
                ("turnover_rate", "f8"), ("volume_ratio", "f8"), ("pct_change", "f8"),
                ("amplitude", "f8"), ("change_amount", "f8"),
            ],
        )
        arr2 = np.array(
            [("2024-01-15", 11.0, 12.0, 10.0, 11.5, 1100000, 11500000, 1.6, 1.3, 0.6, 2.1, 0.6)],
            dtype=[
                ("date", "datetime64[D]"), ("open", "f8"), ("high", "f8"),
                ("low", "f8"), ("close", "f8"), ("volume", "f8"), ("amount", "f8"),
                ("turnover_rate", "f8"), ("volume_ratio", "f8"), ("pct_change", "f8"),
                ("amplitude", "f8"), ("change_amount", "f8"),
            ],
        )
        save_stock_history("000001", arr1, base_dir=temp_db)
        save_stock_history("000001", arr2, base_dir=temp_db)

        conn = sqlite3.connect(temp_db / "stock.db")
        cur = conn.execute("SELECT close FROM price WHERE symbol=? AND date=?", ("000001", "2024-01-15"))
        row = cur.fetchone()
        conn.close()
        assert row[0] == 11.5  # Updated value, not original


class TestLoadStockHistory:
    """Tests for load_stock_history()."""

    def test_load_returns_structured_array(self, temp_db):
        """load_stock_history returns a numpy structured array."""
        from auto_select_stock.data.storage import save_stock_history, load_stock_history

        arr = np.array(
            [
                ("2024-01-15", 10.0, 11.0, 9.0, 10.5, 1000000, 10500000, 1.5, 1.2, 0.5, 2.0, 0.5),
                ("2024-01-16", 10.5, 11.5, 9.5, 11.0, 1100000, 11500000, 1.6, 1.3, 0.6, 2.1, 0.6),
            ],
            dtype=[
                ("date", "datetime64[D]"), ("open", "f8"), ("high", "f8"),
                ("low", "f8"), ("close", "f8"), ("volume", "f8"), ("amount", "f8"),
                ("turnover_rate", "f8"), ("volume_ratio", "f8"), ("pct_change", "f8"),
                ("amplitude", "f8"), ("change_amount", "f8"),
            ],
        )
        save_stock_history("000001", arr, base_dir=temp_db)
        result = load_stock_history("000001", base_dir=temp_db)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert result["close"][0] == 10.5
        assert result["close"][1] == 11.0

    def test_load_missing_symbol_raises(self, temp_db):
        """load_stock_history raises FileNotFoundError for unknown symbol."""
        from auto_select_stock.data.storage import load_stock_history

        with pytest.raises(FileNotFoundError, match="No price data"):
            load_stock_history("999999", base_dir=temp_db)


class TestPriceDateRange:
    """Tests for price_date_range()."""

    def test_returns_date_tuple(self, temp_db):
        """price_date_range returns (min_date, max_date) tuple."""
        from auto_select_stock.data.storage import save_stock_history, price_date_range

        arr = np.array(
            [
                ("2024-01-10", 10.0, 11.0, 9.0, 10.5, 1000000, 10500000, 1.5, 1.2, 0.5, 2.0, 0.5),
                ("2024-01-20", 10.5, 11.5, 9.5, 11.0, 1100000, 11500000, 1.6, 1.3, 0.6, 2.1, 0.6),
            ],
            dtype=[
                ("date", "datetime64[D]"), ("open", "f8"), ("high", "f8"),
                ("low", "f8"), ("close", "f8"), ("volume", "f8"), ("amount", "f8"),
                ("turnover_rate", "f8"), ("volume_ratio", "f8"), ("pct_change", "f8"),
                ("amplitude", "f8"), ("change_amount", "f8"),
            ],
        )
        save_stock_history("000001", arr, base_dir=temp_db)
        result = price_date_range("000001", base_dir=temp_db)

        assert result is not None
        assert len(result) == 2
        assert result[0] <= result[1]

    def test_missing_symbol_returns_none(self, temp_db):
        """price_date_range returns None for unknown symbol."""
        from auto_select_stock.data.storage import price_date_range

        result = price_date_range("999999", base_dir=temp_db)
        assert result is None


class TestListSymbols:
    """Tests for list_symbols()."""

    def test_returns_all_symbols(self, temp_db):
        """list_symbols returns all unique symbols in the table."""
        from auto_select_stock.data.storage import save_stock_history, list_symbols

        for sym in ["000001", "000002", "600000"]:
            arr = np.array(
                [("2024-01-15", 10.0, 11.0, 9.0, 10.5, 1000000, 10500000, 1.5, 1.2, 0.5, 2.0, 0.5)],
                dtype=[
                    ("date", "datetime64[D]"), ("open", "f8"), ("high", "f8"),
                    ("low", "f8"), ("close", "f8"), ("volume", "f8"), ("amount", "f8"),
                    ("turnover_rate", "f8"), ("volume_ratio", "f8"), ("pct_change", "f8"),
                    ("amplitude", "f8"), ("change_amount", "f8"),
                ],
            )
            save_stock_history(sym, arr, base_dir=temp_db)

        result = list_symbols("price", base_dir=temp_db)
        assert set(result) == {"000001", "000002", "600000"}

    def test_empty_database_returns_empty_list(self, temp_db):
        """list_symbols returns [] when database has no data."""
        from auto_select_stock.data.storage import list_symbols

        result = list_symbols("price", base_dir=temp_db)
        assert result == []


class TestSaveFinancial:
    """Tests for save_financial()."""

    def test_save_and_query(self, temp_db):
        """Financial data can be saved and queried."""
        from auto_select_stock.data.storage import save_financial

        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-04-01"],
            "roe": [15.0, 12.0],
            "eps": [0.8, 0.6],
            "net_profit_margin": [10.0, 8.0],
            "gross_margin": [30.0, 28.0],
            "operating_cashflow_growth": [5.0, 3.0],
            "debt_to_asset": [0.4, 0.5],
            "operating_cashflow_per_share": [1.0, 0.8],
        })
        save_financial("000001", df, base_dir=temp_db)

        conn = sqlite3.connect(temp_db / "stock.db")
        cur = conn.execute("SELECT COUNT(*) FROM financial WHERE symbol=?", ("000001",))
        count = cur.fetchone()[0]
        conn.close()
        assert count == 2


class TestLoadFinancial:
    """Tests for load_financial()."""

    def test_load_returns_dataframe(self, temp_db):
        """load_financial returns a pandas DataFrame."""
        from auto_select_stock.data.storage import save_financial, load_financial

        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-04-01"],
            "roe": [15.0, 12.0],
            "eps": [0.8, 0.6],
        })
        save_financial("000001", df, base_dir=temp_db)
        result = load_financial("000001", base_dir=temp_db)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "roe" in result.columns
        assert "eps" in result.columns

    def test_load_missing_symbol_raises(self, temp_db):
        """load_financial raises FileNotFoundError for unknown symbol."""
        from auto_select_stock.data.storage import load_financial

        with pytest.raises(FileNotFoundError, match="No financial data"):
            load_financial("999999", base_dir=temp_db)
