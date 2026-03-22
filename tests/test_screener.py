"""
Unit tests for the web.screener module.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from auto_select_stock.web.screener import (
    ScreenCriteria,
    ScreenerRow,
    _criteria_to_str,
    parse_nl_query,
    render_screener_html,
    screen_stocks,
)


# ---------------------------------------------------------------------------
# parse_nl_query tests
# ---------------------------------------------------------------------------


class TestParseNlQuery:
    def test_default_lookback(self):
        criteria = parse_nl_query("ROE高于10%")
        assert criteria.lookback_days == 60
        assert criteria.min_roe == 10.0
        assert criteria.max_pct_change is None
        assert criteria.min_pct_change is None

    def test_months_conversion(self):
        criteria = parse_nl_query("3个月内涨幅不超过+10%")
        assert criteria.lookback_days == 90  # 3 * 30

    def test_days_direct(self):
        criteria = parse_nl_query("60日涨幅超过5%")
        assert criteria.lookback_days == 60

    def test_years_conversion(self):
        criteria = parse_nl_query("1年内涨幅大于0%")
        assert criteria.lookback_days == 365

    def test_max_pct_change(self):
        criteria = parse_nl_query("涨幅不超过+10%")
        assert criteria.max_pct_change == 10.0
        assert criteria.min_pct_change is None

    def test_max_pct_change_variants(self):
        for phrase in ["涨幅小于10%", "涨幅低于10%", "涨幅不多于10%"]:
            criteria = parse_nl_query(phrase)
            assert criteria.max_pct_change == 10.0, f"Failed for: {phrase}"

    def test_min_pct_change(self):
        criteria = parse_nl_query("涨幅超过20%")
        assert criteria.min_pct_change == 20.0
        assert criteria.max_pct_change is None

    def test_min_pct_change_variants(self):
        for phrase in ["涨幅大于20%", "涨幅高于20%"]:
            criteria = parse_nl_query(phrase)
            assert criteria.min_pct_change == 20.0, f"Failed for: {phrase}"

    def test_roe_min(self):
        criteria = parse_nl_query("ROE高于10%")
        assert criteria.min_roe == 10.0

    def test_eps_min(self):
        criteria = parse_nl_query("EPS高于0.5元")
        assert criteria.min_eps == 0.5

    def test_turnover_rate_max(self):
        criteria = parse_nl_query("换手率低于5%")
        assert criteria.max_turnover_rate == 5.0

    def test_turnover_rate_min(self):
        criteria = parse_nl_query("换手率高于2%")
        assert criteria.min_turnover_rate == 2.0

    def test_multiple_criteria(self):
        criteria = parse_nl_query("3个月内涨幅不超过+20%，ROE高于5%，换手率高于1%")
        assert criteria.lookback_days == 90
        assert criteria.max_pct_change == 20.0
        assert criteria.min_roe == 5.0
        assert criteria.min_turnover_rate == 1.0

    def test_plus_sign_optional(self):
        criteria = parse_nl_query("涨幅不超过+10%")
        assert criteria.max_pct_change == 10.0

    def test_chinese_punctuation_normalized(self):
        criteria = parse_nl_query("涨幅不超过10%，ROE高于5%")
        assert criteria.max_pct_change == 10.0
        assert criteria.min_roe == 5.0


# ---------------------------------------------------------------------------
# _criteria_to_str tests
# ---------------------------------------------------------------------------


class TestCriteriaToStr:
    def test_empty_criteria(self):
        s = _criteria_to_str(ScreenCriteria())
        assert "近60天" in s
        assert "ROE" not in s
        assert "涨幅" not in s

    def test_with_pct_change(self):
        criteria = ScreenCriteria(lookback_days=90, max_pct_change=10.0)
        s = _criteria_to_str(criteria)
        assert "近90天" in s
        assert "涨幅<10.0%" in s

    def test_with_roe(self):
        criteria = ScreenCriteria(min_roe=15.0)
        s = _criteria_to_str(criteria)
        assert "ROE>15.0%" in s


# ---------------------------------------------------------------------------
# screen_stocks tests (with temporary in-memory SQLite)
# ---------------------------------------------------------------------------


class TestScreenStocks:
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary SQLite database with test data at stock.db path."""
        db_path = tmp_path / "stock.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE price (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL,
                turnover_rate REAL,
                PRIMARY KEY(symbol, date)
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE financial (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                roe REAL,
                eps REAL,
                PRIMARY KEY(symbol, date)
            )
        """
        )

        # Insert test price data
        # latest date: 2025-03-01, lookback target: 2024-12-31 (60 days before)
        # lookback prices must be on or before 2024-12-31
        price_rows = [
            # Symbol 000001: latest=10.0 (2025-03-01), lookback=8.0 (2024-11-01) -> return = +25%
            ("000001", "2024-11-01", 8.0, 1.5),
            ("000001", "2025-01-15", 9.0, 2.0),
            ("000001", "2025-03-01", 10.0, 3.0),
            # Symbol 000002: latest=12.0 (2025-03-01), lookback=10.0 (2024-11-01) -> return = +20%
            ("000002", "2024-11-01", 10.0, 0.5),
            ("000002", "2025-03-01", 12.0, 1.0),
            # Symbol 000003: latest=6.0 (2025-03-01), lookback=10.0 (2024-11-01) -> return = -40%
            ("000003", "2024-11-01", 10.0, 4.0),
            ("000003", "2025-03-01", 6.0, 5.0),
        ]
        conn.executemany(
            "INSERT INTO price (symbol, date, close, turnover_rate) VALUES (?, ?, ?, ?)",
            price_rows,
        )

        # Insert test financial data
        fin_rows = [
            ("000001", "2025-01-01", 15.0, 0.8),
            ("000002", "2025-01-01", 8.0, 0.3),
            ("000003", "2025-01-01", 20.0, 1.2),
        ]
        conn.executemany(
            "INSERT INTO financial (symbol, date, roe, eps) VALUES (?, ?, ?, ?)",
            fin_rows,
        )
        conn.commit()
        conn.close()
        return tmp_path  # Return tmp_path so screen_stocks(base_dir=tmp_path) finds stock.db

    def test_roe_filter(self, temp_db):
        criteria = ScreenCriteria(lookback_days=60, min_roe=10.0)
        rows = screen_stocks(criteria, base_dir=temp_db)
        symbols = {r.symbol for r in rows}
        assert "000001" in symbols  # ROE=15 > 10
        assert "000003" in symbols  # ROE=20 > 10
        assert "000002" not in symbols  # ROE=8 < 10

    def test_pct_change_max_filter(self, temp_db):
        criteria = ScreenCriteria(lookback_days=60, max_pct_change=22.0)
        rows = screen_stocks(criteria, base_dir=temp_db)
        symbols = {r.symbol for r in rows}
        # 000001: +25% > 22% -> excluded
        # 000002: +20% < 22% -> included
        assert "000002" in symbols
        assert "000001" not in symbols

    def test_pct_change_min_filter(self, temp_db):
        criteria = ScreenCriteria(lookback_days=60, min_pct_change=0.0)
        rows = screen_stocks(criteria, base_dir=temp_db)
        symbols = {r.symbol for r in rows}
        # 000001: +25% > 0 -> included
        # 000002: +20% > 0 -> included
        # 000003: -40% < 0 -> excluded
        assert "000001" in symbols
        assert "000002" in symbols
        assert "000003" not in symbols

    def test_turnover_rate_filter(self, temp_db):
        criteria = ScreenCriteria(lookback_days=60, min_turnover_rate=2.0)
        rows = screen_stocks(criteria, base_dir=temp_db)
        symbols = {r.symbol for r in rows}
        # 000001: turnover=3.0 > 2 -> included
        # 000002: turnover=1.0 < 2 -> excluded
        # 000003: turnover=5.0 > 2 -> included
        assert "000001" in symbols
        assert "000003" in symbols
        assert "000002" not in symbols

    def test_combined_filters(self, temp_db):
        criteria = ScreenCriteria(
            lookback_days=60,
            min_pct_change=0.0,
            min_roe=10.0,
        )
        rows = screen_stocks(criteria, base_dir=temp_db)
        symbols = {r.symbol for r in rows}
        # 000001: +25% > 0, ROE=15 > 10 -> included
        # 000002: +20% > 0, ROE=8 < 10 -> excluded (roe)
        # 000003: -40% < 0 -> excluded (pct_change)
        assert "000001" in symbols
        assert "000002" not in symbols
        assert "000003" not in symbols

    def test_empty_result(self, temp_db):
        criteria = ScreenCriteria(lookback_days=60, min_pct_change=50.0)
        rows = screen_stocks(criteria, base_dir=temp_db)
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# render_screener_html tests
# ---------------------------------------------------------------------------


class TestRenderScreenerHtml:
    def test_renders_html_file(self, tmp_path):
        rows = [
            ScreenerRow(
                symbol="000001",
                name="平安银行",
                lookback_pct_change=25.5,
                last_close=10.0,
                last_date="2025-03-01",
                roe=15.0,
                eps=0.8,
                turnover_rate=3.0,
            )
        ]
        criteria = ScreenCriteria(lookback_days=60, min_pct_change=10.0)
        output = tmp_path / "screener.html"
        result = render_screener_html(rows, criteria, output)
        assert result == output
        assert output.exists()

    def test_html_contains_data(self, tmp_path):
        rows = [
            ScreenerRow(
                symbol="000001",
                name="平安银行",
                lookback_pct_change=25.5,
                last_close=10.0,
                last_date="2025-03-01",
                roe=15.0,
                eps=0.8,
                turnover_rate=3.0,
            )
        ]
        criteria = ScreenCriteria(lookback_days=60)
        output = tmp_path / "screener.html"
        render_screener_html(rows, criteria, output)
        content = output.read_text(encoding="utf-8")
        assert "000001" in content
        assert "平安银行" in content
        assert "25.50" in content
        assert "data-key" in content  # JS sort enabled via data-key attributes

    def test_empty_rows_renders(self, tmp_path):
        rows = []
        criteria = ScreenCriteria(lookback_days=60)
        output = tmp_path / "screener.html"
        render_screener_html(rows, criteria, output)
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "暂无符合条件的数据" in content or "0" in content
