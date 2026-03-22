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


def generate_report(
    checkpoint: str,
    strategy: str = "confidence",
    top_k: int = 10,
    horizon: str = "5d",
) -> Tuple[str, List[Tuple[str, Dict[str, float], float]]]:
    """
    Generate the daily push report.

    Args:
        checkpoint: path to model checkpoint
        strategy: strategy name
        top_k: number of top stocks to include
        horizon: prediction horizon for ranking (default: 5d)

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

    # Determine which horizons to display (all available in first result)
    display_horizons = ["1d", "3d", "5d", "7d", "14d", "20d"]
    if results and results[0][1]:
        available = set(results[0][1].keys())
        display_horizons = [h for h in display_horizons if h in available]

    # Build header
    horizon_headers = "".join(f"<th>{h}</th>" for h in display_horizons)
    rows = ""
    for sym, pred_rets, weight in results:
        cells = f"<td>{sym}</td>"
        for h in display_horizons:
            pct = pred_rets.get(h, 0.0) * 100
            cells += f"<td>{pct:+.2f}%</td>"
        cells += f"<td>{weight:.4f}</td>"
        rows += f"<tr>{cells}</tr>\n"

    html = f"""<html>
<head><meta charset="utf-8"></head>
<body>
<h3>StockPilot 每日推荐 ({today_str})</h3>
<p>数据库最新日期: {latest_db_date} | 策略: {strategy} | 排序依据: {horizon}</p>
<table border="1" cellpadding="6" cellspacing="0">
<thead><tr><th>股票代码</th>{horizon_headers}<th>权重</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<p style="color:#888;font-size:12px">由 StockPilot 自动生成 · 数据仅供参考</p>
</body>
</html>"""

    return html, results
