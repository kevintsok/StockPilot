"""
Daily report generation: fetch data, run inference, apply strategy, build HTML.

Reuses the existing predict_pipeline.py infrastructure.
"""

from datetime import date
from typing import List, Tuple

from ..predict_pipeline import get_latest_price_date as _get_latest_date
from ..predict_pipeline import get_top_k_stocks


def get_latest_price_date() -> str:
    """Proxy for predict_pipeline.get_latest_price_date."""
    return _get_latest_date()


def generate_report(
    checkpoint: str,
    strategy: str = "confidence",
    top_k: int = 10,
) -> Tuple[str, List[Tuple[str, float, float]]]:
    """
    Generate the daily push report.

    Args:
        checkpoint: path to model checkpoint
        strategy: strategy name
        top_k: number of top stocks to include

    Returns:
        Tuple of (html_content, stock_list) where stock_list is
        [(symbol, predicted_return, weight), ...]
    """
    today_str = date.today().isoformat()
    latest_db_date = get_latest_price_date()
    results = get_top_k_stocks(
        checkpoint=checkpoint,
        strategy=strategy,
        top_k=top_k,
    )

    rows = ""
    for sym, pred_ret, weight in results:
        pct = pred_ret * 100
        rows += f"<tr><td>{sym}</td><td>{pct:+.2f}%</td><td>{weight:.4f}</td></tr>\n"

    html = f"""<html>
<head><meta charset="utf-8"></head>
<body>
<h3>StockPilot 每日推荐 ({today_str})</h3>
<p>数据库最新日期: {latest_db_date} | 策略: {strategy}</p>
<table border="1" cellpadding="6" cellspacing="0">
<thead><tr><th>股票代码</th><th>预测收益率</th><th>权重</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<p style="color:#888;font-size:12px">由 StockPilot 自动生成 · 数据仅供参考</p>
</body>
</html>"""

    return html, results
