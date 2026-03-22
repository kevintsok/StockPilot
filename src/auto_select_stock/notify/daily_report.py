"""
Daily report generation: fetch data, run inference, apply strategy, build HTML.

Reuses the existing predict_pipeline.py infrastructure.
"""

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .pipeline import get_latest_price_date as _get_latest_date
from .pipeline import get_top_k_stocks
from ..predict.strategies.registry import StrategyRegistry


def _get_template_env() -> Environment:
    """Get Jinja2 environment with FileSystemLoader pointing to templates directory."""
    template_dir = Path(__file__).resolve().parent.parent / "web" / "templates"
    return Environment(
        loader=FileSystemLoader(searchpath=str(template_dir)),
        autoescape=select_autoescape(["html"]),
    )


def get_latest_price_date() -> str:
    """Proxy for predict_pipeline.get_latest_price_date."""
    return _get_latest_date()


def _load_strategy_config(strategy_name: str) -> Dict[str, Any]:
    """Load strategy config from registry JSON files."""
    try:
        registry = StrategyRegistry(
            Path(__file__).parent.parent / "predict" / "strategies" / "configs"
        )
        return registry.get(strategy_name)
    except Exception:
        return {}


def _format_weight(weight: float, capital: float = 100_000) -> str:
    """Convert weight ratio to RMB position amount."""
    amount = weight * capital
    if amount >= 10_000:
        return f"{amount / 10_000:.1f}万"
    return f"{amount:.0f}元"


def _strategy_params_summary(strategy_name: str, cfg: Dict[str, Any]) -> str:
    """Build a short string summarizing key strategy parameters."""
    p = cfg.get("params", {})
    stype = cfg.get("type", "")
    horizon = cfg.get("horizon", "1d")

    if stype == "risk_parity":
        vl = p.get("vol_lookback", "?")
        return f"波动率倒数加权(vol_LookBack={vl}) | {horizon}预测"
    elif stype == "topk":
        k = p.get("top_k", "?")
        allow_short = p.get("allow_short", False)
        side = "多空" if allow_short else "纯多"
        return f"Top-{k}等权({side}) | {horizon}预测"
    elif stype == "trailing_stop":
        pct = p.get("stop_pct", "?")
        return f"追踪止损({pct}%) | {horizon}预测"
    elif stype == "confidence":
        mc = p.get("min_confidence_bp", "?")
        return f"置信度加权(min={mc}bp) | {horizon}预测"
    elif stype == "momentum_filter":
        lb = p.get("lookback", "?")
        return f"动量过滤(LB={lb}) | {horizon}预测"
    else:
        return f"{stype} | {horizon}预测"


def generate_report(
    checkpoint: str,
    strategy: str = "confidence",
    top_k: int = 10,
    horizon: str = "5d",
    capital: float = 100_000,
) -> Tuple[str, List[Tuple[str, Dict[str, float], float]]]:
    """
    Generate the daily push report.

    Args:
        checkpoint: path to model checkpoint
        strategy: strategy name
        top_k: number of top stocks to include
        horizon: prediction horizon for ranking (default: 5d)
        capital: initial capital for position size calculation (default: 100,000 RMB)

    Returns:
        Tuple of (html_content, stock_list) where stock_list is
        [(symbol, {horizon: return, ...}, weight), ...]
    """
    today_str = date.today().isoformat()
    latest_db_date = get_latest_price_date()

    # Load strategy config to get actual top_k and horizon
    cfg = _load_strategy_config(strategy)
    actual_top_k = cfg.get("params", {}).get("top_k", top_k)
    strategy_horizon = cfg.get("horizon", horizon)  # strategy's own horizon (e.g. "1d")

    results = get_top_k_stocks(
        checkpoint=checkpoint,
        strategy=strategy,
        top_k=actual_top_k,
        horizon=horizon,  # ranking horizon
    )

    params_summary = _strategy_params_summary(strategy, cfg)

    # Table columns: Stock | [horizon] Return | Key Params | Weight/Position
    rows_html = ""
    for sym, pred_rets, weight in results:
        # Primary prediction (strategy's own horizon)
        primary_ret = pred_rets.get(strategy_horizon, 0.0) * 100
        color = "#2e7d32" if primary_ret > 0 else "#c62828" if primary_ret < 0 else "#666"

        # Build params column from config
        p = cfg.get("params", {})
        stype = cfg.get("type", "")
        if stype == "risk_parity":
            params_str = f"vol_LookBack={p.get('vol_lookback', '?')}"
        elif stype == "topk":
            params_str = f"K={p.get('top_k', '?')}"
        elif stype == "trailing_stop":
            params_str = f"止损{p.get('stop_pct', '?')}% | {p.get('horizon', '?')}"
        elif stype == "confidence":
            params_str = f"min={p.get('min_confidence_bp', '?')}bp"
        elif stype == "momentum_filter":
            params_str = f"lookback={p.get('lookback', '?')}天"
        else:
            params_str = ""

        # Position amount
        pos_amount = _format_weight(weight, capital)

        cells = f'<td><b>{sym}</b></td>'
        cells += f'<td style="color:{color};font-weight:bold">{primary_ret:+.2f}%</td>'
        cells += f'<td style="color:#555">{params_str}</td>'
        cells += f'<td><span style="color:#1565c0">{weight:.2%}</span><br><span style="color:#888;font-size:11px">≈{pos_amount}</span></td>'
        rows_html += f"<tr>{cells}</tr>\n"

    # Summary
    total_weight = sum(w for _, _, w in results)
    avg_primary = (
        sum(pred_rets.get(strategy_horizon, 0) for _, pred_rets, _ in results) / len(results)
        if results else 0
    )

    # Use Jinja2 template
    env = _get_template_env()
    template = env.get_template("daily_report.html")
    html = template.render(
        today_str=today_str,
        latest_db_date=latest_db_date,
        strategy=strategy,
        top_k=actual_top_k,
        params_summary=params_summary,
        rows_html=rows_html,
        horizon=strategy_horizon,
        result_count=len(results),
        total_weight=f"{total_weight:.0%}",
        avg_primary=f"{avg_primary*100:+.2f}",
        capital=f"{capital:,.0f}",
    )

    return html, results
