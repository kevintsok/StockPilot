"""
Holdings diagnosis and recommendation module.

Provides position analysis combining latest price data with multi-horizon
model predictions to generate actionable recommendations (hold/add/reduce/stop-loss).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json

from ..config import DATA_DIR
from ..data.storage import _connect
from ..predict.inference import PricePredictor


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass
class Position:
    """A single stock holding."""
    symbol: str
    cost_price: float
    shares: int


@dataclass
class PositionAnalysis:
    """Analysis result for a single position."""
    symbol: str
    shares: int
    cost_price: float
    current_price: float
    cost_value: float          # cost_price * shares
    current_value: float       # current_price * shares
    unrealized_pnl: float      # current_value - cost_value
    unrealized_pct: float      # unrealized_pnl / cost_value
    pred_rets: Dict[str, float]  # {horizon: return}
    recommendation: str
    reason: str


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_holdings(path: Path) -> List[Position]:
    """Load holdings from JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        Position(symbol=p["symbol"], cost_price=float(p["cost_price"]), shares=int(p["shares"]))
        for p in data.get("positions", [])
    ]


# ----------------------------------------------------------------------
# Price fetching
# ----------------------------------------------------------------------


def get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch the latest closing price for each symbol from SQLite."""
    if not symbols:
        return {}
    conn = _connect()
    placeholders = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"SELECT symbol, close FROM price WHERE symbol IN ({placeholders}) "
        "AND date = (SELECT MAX(date) FROM price WHERE symbol = price.symbol)",
        symbols,
    ).fetchall()
    return {row[0]: float(row[1]) for row in rows}


# ----------------------------------------------------------------------
# Recommendation logic
# ----------------------------------------------------------------------


def _generate_recommendation(
    unrealized_pct: float,
    pred_5d: float,
) -> Tuple[str, str]:
    """
    Generate a recommendation based on unrealized P&L and 5-day prediction.

    Rules:
      - Loss > 8% AND 5d prediction continues to fall  -> STOP_LOSS
      - Loss > 5% AND 5d prediction continues to fall    -> REDUCE
      - Loss but 5d prediction rebounds > 2%            -> HOLD / WATCH
      - Gain AND 5d prediction rises > 3%               -> HOLD / ADD
      - Gain but 5d prediction falls > 3%               -> REDUCE / LOCK_PROFIT
      - Otherwise                                        -> HOLD
    """
    if unrealized_pct < -0.08 and pred_5d < 0:
        return "止损", f"浮亏{unrealized_pct*100:.1f}%且5日预测继续下跌({pred_5d*100:+.2f}%)"
    if unrealized_pct < -0.05 and pred_5d < 0:
        return "减仓", f"浮亏{unrealized_pct*100:.1f}%且5日预测下跌({pred_5d*100:+.2f}%)"
    if unrealized_pct < 0 and pred_5d > 0.02:
        return "持有/观望", f"浮亏{unrealized_pct*100:.1f}%但5日预测反弹({pred_5d*100:+.2f}%)"
    if unrealized_pct > 0 and pred_5d > 0.03:
        return "持有/加仓", f"浮盈{unrealized_pct*100:.1f}%且5日预测上涨({pred_5d*100:+.2f}%)"
    if unrealized_pct > 0 and pred_5d < -0.03:
        return "减仓锁利", f"浮盈{unrealized_pct*100:.1f}%但5日预测下跌({pred_5d*100:+.2f}%)"
    return "持有", f"浮盈{unrealized_pct*100:+.1f}%，5日预测平稳({pred_5d*100:+.2f}%)"


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------


def analyze_position(
    pos: Position,
    current_price: float,
    pred_rets: Dict[str, float],
) -> PositionAnalysis:
    """Analyze a single position and return a PositionAnalysis."""
    cost_value = pos.cost_price * pos.shares
    current_value = current_price * pos.shares
    unrealized_pnl = current_value - cost_value
    unrealized_pct = unrealized_pnl / cost_value if cost_value > 0 else 0.0
    pred_5d = pred_rets.get("5d", pred_rets.get("1d", 0.0))
    recommendation, reason = _generate_recommendation(unrealized_pct, pred_5d)
    return PositionAnalysis(
        symbol=pos.symbol,
        shares=pos.shares,
        cost_price=pos.cost_price,
        current_price=current_price,
        cost_value=cost_value,
        current_value=current_value,
        unrealized_pnl=unrealized_pnl,
        unrealized_pct=unrealized_pct,
        pred_rets=pred_rets,
        recommendation=recommendation,
        reason=reason,
    )


# ----------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------


def generate_holdings_report(
    checkpoint: str,
    holdings_path: Path,
    predictor: Optional[PricePredictor] = None,
) -> Tuple[str, List[PositionAnalysis]]:
    """
    Generate an HTML holdings diagnosis report.

    Args:
        checkpoint: path to model checkpoint
        holdings_path: path to holdings JSON file
        predictor: optional pre-constructed PricePredictor (avoids reloading)

    Returns:
        Tuple of (html_content, list of PositionAnalysis)
    """
    from datetime import date
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    # Load holdings
    positions = load_holdings(holdings_path)
    if not positions:
        return "<p>No positions found in holdings file.</p>", []

    symbols = [p.symbol for p in positions]

    # Load predictor
    if predictor is None:
        predictor = PricePredictor(checkpoint)

    # Get current prices
    prices = get_current_prices(symbols)

    # Run inference for each symbol
    analyses: List[PositionAnalysis] = []
    for pos in positions:
        sym = pos.symbol
        current_price = prices.get(sym)
        if current_price is None:
            # Try to get price from predictor's last known data
            try:
                result = predictor.predict(sym, horizon=None)
                if isinstance(result, dict):
                    # Use last close from feature loading
                    from ..predict.data import load_feature_matrix
                    feats = load_feature_matrix(
                        sym,
                        price_columns=predictor.cfg.price_columns,
                        financial_columns=predictor.cfg.financial_columns,
                    )
                    current_price = float(feats[-1, predictor.close_idx])
                else:
                    current_price = 0.0
            except Exception:
                current_price = 0.0

        if current_price <= 0:
            # Fallback: use cost price as approximation
            current_price = pos.cost_price

        # Get predictions
        try:
            pred_rets = predictor.predict(sym, horizon=None)
            if not isinstance(pred_rets, dict):
                pred_rets = {"1d": float(pred_rets)}
        except Exception:
            pred_rets = {"1d": 0.0}

        analysis = analyze_position(pos, current_price, pred_rets)
        analyses.append(analysis)

    # Render HTML
    template_dir = Path(__file__).resolve().parent.parent / "web" / "templates"
    env = Environment(
        loader=FileSystemLoader(searchpath=str(template_dir)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("holdings_report.html")

    today_str = date.today().isoformat()
    latest_db_date = analyses[0].symbol if analyses else ""

    # Sort by recommendation priority (止损 > 减仓 > 持有/观望 > 持有 > 持有/加仓)
    rec_order = {"止损": 0, "减仓": 1, "减仓锁利": 2, "持有/观望": 3, "持有": 4, "持有/加仓": 5}
    analyses.sort(key=lambda a: (rec_order.get(a.recommendation, 99), -a.unrealized_pct))

    html = template.render(
        today_str=today_str,
        analyses=analyses,
        total_cost=sum(a.cost_value for a in analyses),
        total_value=sum(a.current_value for a in analyses),
        total_pnl=sum(a.unrealized_pnl for a in analyses),
    )

    return html, analyses
