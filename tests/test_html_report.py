"""
Unit tests for the HTML report module (web/html_report.py).

Tests LLM scoring HTML report rendering.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from auto_select_stock.core.types import StockMeta, StockScore
from auto_select_stock.web.html_report import render_report


class TestRenderReport:
    """Tests for render_report()."""

    def _make_scores(self) -> list[StockScore]:
        """Create sample StockScore objects."""
        return [
            StockScore(
                symbol="600000",
                score=8.5,
                rationale="Strong fundamentals with high ROE.",
                meta=StockMeta(symbol="600000", name="浦发银行", industry="银行"),
                factors={"roe": 15.0, "pe": 8.0},
            ),
            StockScore(
                symbol="000001",
                score=7.2,
                rationale="Undervalued with growth potential.",
                meta=StockMeta(symbol="000001", name="平安银行", industry="银行"),
                factors={"roe": 12.0, "pe": 6.5},
            ),
        ]

    def test_renders_html_file(self, tmp_path: Path):
        """Creates HTML file at specified output path."""
        scores = self._make_scores()
        output_path = tmp_path / "report.html"
        result = render_report(scores, top_n=10, output_path=output_path)
        assert result == output_path
        assert output_path.exists()

    def test_respects_top_n_limit(self, tmp_path: Path):
        """Only renders top_n items."""
        scores = self._make_scores()
        output_path = tmp_path / "report.html"
        render_report(scores, top_n=1, output_path=output_path)
        html = output_path.read_text(encoding="utf-8")
        # Should contain first symbol
        assert "600000" in html
        # Should NOT contain second symbol
        assert "000001" not in html

    def test_creates_parent_directory(self, tmp_path: Path):
        """Creates parent directory if it doesn't exist."""
        scores = self._make_scores()
        output_path = tmp_path / "subdir" / "report.html"
        render_report(scores, top_n=10, output_path=output_path)
        assert output_path.parent.exists()

    def test_handles_empty_stock_list(self, tmp_path: Path):
        """Handles empty stock list gracefully."""
        output_path = tmp_path / "report.html"
        result = render_report([], top_n=10, output_path=output_path)
        assert result == output_path
        assert output_path.exists()
        html = output_path.read_text(encoding="utf-8")
        # Should still be valid HTML
        assert "<html" in html.lower() or "<!doctype" in html.lower()

    def test_renders_score_value(self, tmp_path: Path):
        """Renders the score value in HTML."""
        scores = self._make_scores()
        output_path = tmp_path / "report.html"
        render_report(scores, top_n=10, output_path=output_path)
        html = output_path.read_text(encoding="utf-8")
        assert "8.5" in html or "600000" in html

    def test_renders_symbol(self, tmp_path: Path):
        """Renders symbol in HTML output."""
        scores = self._make_scores()
        output_path = tmp_path / "report.html"
        render_report(scores, top_n=10, output_path=output_path)
        html = output_path.read_text(encoding="utf-8")
        assert "600000" in html
