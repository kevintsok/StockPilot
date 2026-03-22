"""
Daily report generation: fetch data, run inference, apply strategy, build HTML.

Reuses the existing predict_pipeline.py infrastructure.
"""

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..predict_pipeline import get_latest_price_date as _get_latest_date
from ..predict_pipeline import get_top_k_stocks
from ..predict.strategies.registry import StrategyRegistry


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
    rows = ""
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
        rows += f"<tr>{cells}</tr>\n"

    # Summary
    total_weight = sum(w for _, _, w in results)
    avg_primary = (
        sum(pred_rets.get(strategy_horizon, 0) for _, pred_rets, _ in results) / len(results)
        if results else 0
    )

    html = f"""<html>
<head><meta charset="utf-8"></head>
<body>
<h3>StockPilot 每日推荐 ({today_str})</h3>
<p>
  <b>数据库最新:</b> {latest_db_date} &nbsp;|&nbsp;
  <b>策略:</b> {strategy} &nbsp;|&nbsp;
  <b>推送股数:</b> {actual_top_k}只
</p>
<p style="color:#666;font-size:13px">{params_summary}</p>

<table border="1" cellpadding="8" cellspacing="0">
<thead>
<tr>
  <th>股票代码</th>
  <th>{strategy_horizon}预测收益</th>
  <th>关键参数</th>
  <th>权重 / 仓位</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<p>
  <b>汇总:</b> {len(results)}只股票 | 总仓位{total_weight:.0%} |
  平均{strategy_horizon}收益{avg_primary*100:+.2f}% |
  假设本金{capital:,.0f}元
</p>

<p style="background:#f5f5f5;padding:10px;border-radius:6px">
<b>操作建议:</b> 按「权重」列分配买入金额（≈后的数字），T+1制度当日买入次日可卖。<br>
<b>风险提示:</b> 预测仅供参考，不构成投资建议。A股涨跌停时无法买卖。
</p>
<p style="color:#888;font-size:12px">由 StockPilot 自动生成</p>
</body>
</html>"""

    return html, results
