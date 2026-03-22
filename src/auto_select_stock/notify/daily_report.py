"""
Daily report generation: fetch data, run inference, apply strategy, build HTML.

Reuses the existing predict_pipeline.py infrastructure.
"""

from datetime import date
from typing import Dict, List, Tuple

from ..predict_pipeline import get_latest_price_date as _get_latest_date
from ..predict_pipeline import get_top_k_stocks


def get_latest_price_date() -> str:
    """Proxy for predict_pipeline.get_latest_price_date."""
    return _get_latest_date()


def _format_weight(weight: float, capital: float = 100_000) -> str:
    """Convert weight ratio to RMB position amount."""
    amount = weight * capital
    if amount >= 10_000:
        return f"{amount / 10_000:.1f}万"
    return f"{amount:.0f}元"


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
    results = get_top_k_stocks(
        checkpoint=checkpoint,
        strategy=strategy,
        top_k=top_k,
        horizon=horizon,
    )

    # Determine which horizons to display
    display_horizons = ["1d", "3d", "5d", "7d", "14d", "20d"]
    if results and results[0][1]:
        available = set(results[0][1].keys())
        display_horizons = [h for h in display_horizons if h in available]

    # Strategy description
    strategy_descriptions = {
        "RiskParity-VL10-1d": "波动率倒数加权（vol_lookback=10天，1日预测）",
        "RiskParity-VL20-5d": "波动率倒数加权（vol_lookback=20天，5日预测）",
        "TopK-K3-1d": "Top-3等权策略（1日预测）",
        "TopK-K10-1d": "Top-10等权策略（1日预测）",
        "StopLoss-3pct-5d": "3%止损追踪（5日预测）",
        "StopLoss-8pct-1d": "8%止损追踪（1日预测）",
        "Confidence-MC100bp-1d": "置信度加权（最低100bp，1日预测）",
    }
    strategy_desc = strategy_descriptions.get(strategy, strategy)

    # Build table rows
    horizon_headers = "".join(f"<th>{h}</th>" for h in display_horizons)
    rows = ""
    for sym, pred_rets, weight in results:
        # Color code: green for positive, red for negative
        def cell_class(h):
            v = pred_rets.get(h, 0.0)
            return "pos" if v > 0 else "neg" if v < 0 else ""

        cells = f'<td><b>{sym}</b></td>'
        for h in display_horizons:
            pct = pred_rets.get(h, 0.0) * 100
            cls = cell_class(h)
            color = "#2e7d32" if cls == "pos" else "#c62828" if cls == "neg" else "#666"
            cells += f'<td style="color:{color}">{pct:+.2f}%</td>'

        # Position size
        pos_amount = _format_weight(weight, capital)
        cells += f'<td><span style="color:#1565c0">{weight:.2%}</span><br><span style="color:#888;font-size:11px">≈{pos_amount}</span></td>'
        rows += f"<tr>{cells}</tr>\n"

    # Summary stats
    total_weight = sum(w for _, _, w in results)
    avg_1d = sum(pred_rets.get("1d", 0) for _, pred_rets, _ in results) / len(results) if results else 0
    avg_5d = sum(pred_rets.get("5d", 0) for _, pred_rets, _ in results) / len(results) if results else 0

    html = f"""<html>
<head><meta charset="utf-8"></head>
<body>
<h3>StockPilot 每日推荐 ({today_str})</h3>
<p><b>数据库最新:</b> {latest_db_date} &nbsp;|&nbsp; <b>策略:</b> {strategy}（{strategy_desc}）&nbsp;|&nbsp; <b>排序:</b> {horizon}</p>

<h4>📊 本次推荐持仓（风险平价加权）</h4>
<table border="1" cellpadding="6" cellspacing="0">
<thead>
<tr>
<th>股票代码</th>
{horizon_headers}
<th>权重 / 仓位</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<p><b>汇总:</b> 共{len(results)}只股票 | 总仓位{total_weight:.0%} | 平均1d收益{avg_1d*100:+.2f}% | 平均5d收益{avg_5d*100:+.2f}%</p>

<p style="background:#f5f5f5;padding:10px;border-radius:6px">
<b>策略说明:</b> {strategy_desc}<br>
<br>
<b>操作建议:</b> 按上述权重买入对应股票。假设初始资金{capital:,.0f}元，按权重分配每只股票的买入金额（见「仓位」列）。<br>
<b>风险提示:</b> A股T+1制度，当日买入次日才能卖出；预测仅供参考，不构成投资建议。
</p>
<p style="color:#888;font-size:12px">由 StockPilot 自动生成 · StockPilot</p>
</body>
</html>"""

    return html, results
