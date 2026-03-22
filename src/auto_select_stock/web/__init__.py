"""Web/display module: dashboards, reports, screening, and LLM scoring."""

from .dashboard import build_rows, render_dashboard, StockRow
from .html_report import render_report
from .scoring import score_symbols
from .screener import (
    ScreenCriteria,
    ScreenerRow,
    parse_nl_query,
    render_screener_html,
    screen_stocks,
)

__all__ = [
    "build_rows",
    "render_dashboard",
    "render_report",
    "score_symbols",
    "ScreenCriteria",
    "ScreenerRow",
    "StockRow",
    "parse_nl_query",
    "render_screener_html",
    "screen_stocks",
]
