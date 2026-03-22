"""
Unit tests for the dashboard module (web/dashboard.py).

Tests dashboard rendering and row building with mocked database.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from auto_select_stock.web.dashboard import (
    StockRow,
    _classify_market,
    _price_snapshot,
    _latest_financial_snapshot,
    build_rows,
    render_dashboard,
)


class TestClassifyMarket:
    """Tests for _classify_market()."""

    def test_6xxx_is_shanghai(self):
        assert _classify_market("600000") == "沪市"

    def test_688xxx_is_tech_board(self):
        assert _classify_market("688001") == "科创板"

    def test_300xxx_is_growth_board(self):
        assert _classify_market("300001") == "创业板"

    def test_000xxx_is_shenzhen(self):
        assert _classify_market("000001") == "深市"

    def test_002xxx_is_shenzhen(self):
        assert _classify_market("002001") == "深市"

    def test_8xx_or_4xx_is_bj(self):
        assert _classify_market("830001") == "北交所"
        assert _classify_market("430001") == "北交所"

    def test_unknown_returns_other(self):
        assert _classify_market("999999") == "其他"


class TestPriceSnapshot:
    """Tests for _price_snapshot()."""

    def _setup_db(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "stock.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE price (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL, turnover_rate REAL, volume_ratio REAL,
                amplitude REAL, change_amount REAL, pct_change REAL,
                PRIMARY KEY(symbol, date)
            )
        """)
        conn.execute(
            "INSERT INTO price VALUES ('600000', '2024-01-15', 10.0, 1.5, 1.2, 2.0, 0.5, 1.0)"
        )
        conn.execute(
            "INSERT INTO price VALUES ('600000', '2024-01-16', 10.5, 1.6, 1.3, 2.1, 0.6, 1.5)"
        )
        conn.execute(
            "INSERT INTO price VALUES ('000001', '2024-01-15', 8.0, 0.8, 0.9, 1.5, 0.3, 0.5)"
        )
        conn.commit()
        conn.close()
        return tmp_path

    def test_returns_latest_price_per_symbol(self, tmp_path: Path):
        """Returns the most recent price row for each symbol."""
        base_dir = self._setup_db(tmp_path)
        result = _price_snapshot(["600000", "000001"], lookbacks=[], base_dir=base_dir)
        assert "600000" in result
        assert "000001" in result
        # Latest for 600000 should be 2024-01-16
        assert result["600000"]["last"]["date"] == "2024-01-16"
        assert result["600000"]["last"]["close"] == 10.5

    def test_includes_lookback_prices(self, tmp_path: Path):
        """Includes lookback close prices."""
        base_dir = self._setup_db(tmp_path)
        result = _price_snapshot(["600000"], lookbacks=[1, 2], base_dir=base_dir)
        assert 1 in result["600000"]["back"]
        assert 2 in result["600000"]["back"]

    def test_empty_db_returns_empty_dict(self, tmp_path: Path):
        """Returns empty dict when database has no data."""
        db_path = tmp_path / "stock.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE price (
                symbol TEXT, date TEXT, close REAL, turnover_rate REAL,
                volume_ratio REAL, amplitude REAL, change_amount REAL, pct_change REAL
            )
        """)
        conn.commit()
        conn.close()
        result = _price_snapshot(["600000"], lookbacks=[], base_dir=tmp_path)
        assert result == {}


class TestLatestFinancialSnapshot:
    """Tests for _latest_financial_snapshot()."""

    def _setup_db(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "stock.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE financial (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                roe REAL, net_profit_margin REAL, gross_margin REAL,
                operating_cashflow_growth REAL, debt_to_asset REAL, eps REAL,
                PRIMARY KEY(symbol, date)
            )
        """)
        conn.execute(
            "INSERT INTO financial VALUES ('600000', '2024-01-01', 15.0, 10.0, 30.0, 5.0, 0.4, 0.8)"
        )
        conn.execute(
            "INSERT INTO financial VALUES ('600000', '2024-04-01', 12.0, 8.0, 28.0, 3.0, 0.5, 0.6)"
        )
        conn.commit()
        conn.close()
        return tmp_path

    def test_returns_latest_financial_row(self, tmp_path: Path):
        """Returns the most recent financial data for each symbol."""
        base_dir = self._setup_db(tmp_path)
        result = _latest_financial_snapshot(["600000"], base_dir=base_dir)
        assert "600000" in result
        # Latest should be 2024-04-01
        assert result["600000"]["roe"] == 12.0

    def test_missing_symbol_returns_empty(self, tmp_path: Path):
        """Returns empty dict for symbols not in database."""
        base_dir = self._setup_db(tmp_path)
        result = _latest_financial_snapshot(["999999"], base_dir=base_dir)
        assert result == {}


class TestBuildRows:
    """Tests for build_rows()."""

    def _setup_db(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "stock.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE price (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL, turnover_rate REAL, volume_ratio REAL,
                amplitude REAL, change_amount REAL, pct_change REAL,
                PRIMARY KEY(symbol, date)
            )
        """)
        conn.execute(
            "INSERT INTO price VALUES ('600000', '2024-01-15', 10.0, 1.5, 1.2, 2.0, 0.5, 1.0)"
        )
        conn.execute(
            "INSERT INTO price VALUES ('600000', '2024-01-16', 10.5, 1.6, 1.3, 2.1, 0.6, 1.5)"
        )
        conn.execute("""
            CREATE TABLE financial (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                roe REAL, net_profit_margin REAL, gross_margin REAL,
                operating_cashflow_growth REAL, debt_to_asset REAL, eps REAL,
                PRIMARY KEY(symbol, date)
            )
        """)
        conn.execute(
            "INSERT INTO financial VALUES ('600000', '2024-01-01', 15.0, 10.0, 30.0, 5.0, 0.4, 0.8)"
        )
        conn.commit()
        conn.close()
        return tmp_path

    def test_builds_stock_rows(self, tmp_path: Path):
        """Returns list of StockRow objects."""
        base_dir = self._setup_db(tmp_path)
        rows = build_rows(["600000"], lookbacks=[20], base_dir=base_dir)
        assert len(rows) == 1
        row = rows[0]
        assert row.symbol == "600000"
        assert row.market == "沪市"
        assert row.price == 10.5  # latest close

    def test_computes_pe_ratio(self, tmp_path: Path):
        """Computes PE ratio from price and EPS."""
        base_dir = self._setup_db(tmp_path)
        rows = build_rows(["600000"], lookbacks=[], base_dir=base_dir)
        row = rows[0]
        # price=10.5, eps=0.8 -> PE should be 10.5/0.8 = 13.125
        assert row.pe is not None
        assert abs(row.pe - 13.125) < 0.01

    def test_handles_missing_financials(self, tmp_path: Path):
        """Handles case when no financial data exists."""
        db_path = tmp_path / "stock.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE price (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL, turnover_rate REAL, volume_ratio REAL,
                amplitude REAL, change_amount REAL, pct_change REAL,
                PRIMARY KEY(symbol, date)
            )
        """)
        conn.execute(
            "INSERT INTO price VALUES ('600000', '2024-01-15', 10.0, 1.5, 1.2, 2.0, 0.5, 1.0)"
        )
        conn.execute("CREATE TABLE financial (symbol TEXT, date TEXT)")
        conn.commit()
        conn.close()
        rows = build_rows(["600000"], lookbacks=[], base_dir=tmp_path)
        assert len(rows) == 1
        assert rows[0].roe is None


class TestRenderDashboard:
    """Tests for render_dashboard()."""

    def test_renders_html_with_rows_json(self, tmp_path: Path):
        """Renders HTML with rows data embedded as JSON."""
        rows = [
            StockRow(
                symbol="600000",
                market="沪市",
                last_date="2024-01-15",
                price=10.5,
                turnover_rate=1.5,
                pct_change=1.0,
                volume_ratio=1.2,
                amplitude=2.0,
                change_amount=0.5,
                chg_20d=None,
                chg_60d=None,
                pe=13.125,
                roe=15.0,
                net_profit_margin=10.0,
                gross_margin=30.0,
                ocf_growth=5.0,
                debt_to_asset=0.4,
            )
        ]
        output_path = tmp_path / "dashboard.html"
        result = render_dashboard(rows, output=output_path)
        assert result == output_path
        assert output_path.exists()
        html = output_path.read_text(encoding="utf-8")
        assert "rows_json" in html or "600000" in html

    def test_creates_output_directory(self, tmp_path: Path):
        """Creates parent directory if it doesn't exist."""
        rows = []
        output_path = tmp_path / "subdir" / "dashboard.html"
        render_dashboard(rows, output=output_path)
        assert output_path.parent.exists()

    def test_handles_empty_rows(self, tmp_path: Path):
        """Handles empty rows list gracefully."""
        output_path = tmp_path / "dashboard.html"
        result = render_dashboard([], output=output_path)
        assert result == output_path
        assert output_path.exists()
