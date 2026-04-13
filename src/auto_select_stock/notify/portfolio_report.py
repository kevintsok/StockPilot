"""
Portfolio report generator for virtual portfolio top-N strategies.

Generates an HTML report with strategy metrics, holdings, and mini equity curves.
"""

import base64
import io
from datetime import date
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from .virtual_portfolio import VirtualPortfolio, StrategyPortfolio


# ----------------------------------------------------------------------
# Mini chart helpers
# ----------------------------------------------------------------------


def _equity_curve_base64(sp: StrategyPortfolio) -> str:
    """Render equity curve as base64-encoded PNG."""
    if not sp.equity_curve or len(sp.equity_curve) < 2:
        return ""

    dates_str = [s.date for s in sp.equity_curve]
    values = [s.total_value for s in sp.equity_curve]

    fig, ax = plt.subplots(figsize=(4, 1.8))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2d2d2d")

    x = range(len(dates_str))
    ax.plot(x, values, color="#4fc3f7", lw=1.2)
    ax.fill_between(x, values, alpha=0.15, color="#4fc3f7")

    # Mark peak and trough
    peak_idx = int(max(range(len(values)), key=lambda i: values[i]))
    ax.scatter([peak_idx], [values[peak_idx]], color="#81c784", s=20, zorder=5)

    ax.axhline(y=100_000, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax.set_title(sp.name, fontsize=7, color="#cccccc", pad=2)
    ax.tick_params(labelsize=6, colors="#aaaaaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.xaxis.label.set_color("#aaaaaa")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


# ----------------------------------------------------------------------
# Main report generator
# ----------------------------------------------------------------------


def generate_portfolio_report(
    portfolio: VirtualPortfolio,
    top_n: int = 3,
    checkpoint: str = "",
) -> str:
    """Generate HTML report for top-N strategies in the virtual portfolio.

    Args:
        portfolio: VirtualPortfolio instance
        top_n: number of top strategies to include
        checkpoint: model checkpoint path (for display only)

    Returns:
        HTML string
    """
    today_str = date.today().isoformat()
    top_strategies = portfolio.get_top_n(top_n)

    strategy_rows = ""
    for rank, sp in enumerate(top_strategies, 1):
        m = sp.metrics
        curve_b64 = _equity_curve_base64(sp)

        # Top holdings (by weight = value / total_value)
        total_val = sp.equity_curve[-1].total_value if sp.equity_curve else 100_000
        holdings = []
        for sym, pos in sp.positions.items():
            cur_price = pos.entry_price  # already updated to latest close
            val = pos.shares * cur_price
            weight = val / total_val if total_val > 0 else 0
            # Get predicted return from trade log (most recent prediction context)
            pred_ret = 0.0
            for t in reversed(sp.trade_log):
                if t.symbol == sym:
                    pred_ret = 0.0  # no prediction stored; show as N/A
                    break
            holdings.append((sym, weight, pred_ret, val))

        holdings.sort(key=lambda x: -x[1])
        holdings_rows = ""
        for sym, weight, pred_ret, val in holdings[:5]:
            color = "#2e7d32" if pred_ret >= 0 else "#c62828" if pred_ret < 0 else "#666"
            holdings_rows += f"""
            <tr>
                <td><b>{sym}</b></td>
                <td>{val:,.0f}元</td>
                <td><span style="color:#1565c0">{weight:.1%}</span></td>
                <td style="color:{color}">{pred_ret*100:+.2f}%</td>
            </tr>"""

        curve_img = (
            f'<img src="data:image/png;base64,{curve_b64}" style="border-radius:4px"/>'
            if curve_b64 else "<em style='color:#666'>No data</em>"
        )

        # Tag color
        tag_color = {"1": "#e57373", "2": "#ffb74d", "3": "#81c784"}.get(str(rank), "#90a4ae")
        rank_badge = f"<span style='background:{tag_color};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:bold'>#{rank}</span>"

        strategy_rows += f"""
        <div style="background:#1a1a2e;border-radius:12px;padding:16px;margin-bottom:16px;border:1px solid #2d2d44">
            <div style="display:flex;align-items:center;margin-bottom:10px;gap:10px">
                {rank_badge}
                <span style="color:#e0e0e0;font-size:16px;font-weight:bold">{sp.name}</span>
                <span style="background:#333;color:#aaa;padding:2px 8px;border-radius:6px;font-size:11px">{sp.tag}</span>
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px">
                <div style="background:#16213e;padding:8px;border-radius:8px;text-align:center">
                    <div style="color:#888;font-size:10px">累计收益</div>
                    <div style="color:{'#2e7d32' if m.total_return >= 0 else '#c62828'};font-size:18px;font-weight:bold">{m.total_return*100:+.2f}%</div>
                </div>
                <div style="background:#16213e;padding:8px;border-radius:8px;text-align:center">
                    <div style="color:#888;font-size:10px">年化收益</div>
                    <div style="color:#4fc3f7;font-size:18px;font-weight:bold">{m.ann_return*100:+.2f}%</div>
                </div>
                <div style="background:#16213e;padding:8px;border-radius:8px;text-align:center">
                    <div style="color:#888;font-size:10px">夏普比率</div>
                    <div style="color:#ffb74d;font-size:18px;font-weight:bold">{m.sharpe_ratio:+.3f}</div>
                </div>
                <div style="background:#16213e;padding:8px;border-radius:8px;text-align:center">
                    <div style="color:#888;font-size:10px">最大回撤</div>
                    <div style="color:{'#c62828' if m.max_drawdown < 0 else '#81c784'};font-size:18px;font-weight:bold">{m.max_drawdown*100:+.1f}%</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start">
                <div>
                    <div style="color:#888;font-size:11px;margin-bottom:6px">持仓明细（当前{len(sp.positions)}只）</div>
                    <table style="width:100%;border-collapse:collapse;font-size:12px">
                        <thead>
                            <tr style="color:#888;border-bottom:1px solid #333">
                                <th style="text-align:left;padding:4px">股票</th>
                                <th style="text-align:right;padding:4px">市值</th>
                                <th style="text-align:right;padding:4px">权重</th>
                                <th style="text-align:right;padding:4px">预测</th>
                            </tr>
                        </thead>
                        <tbody>{holdings_rows or '<tr><td colspan=4 style="color:#666;text-align:center">无持仓</td></tr>'}</tbody>
                    </table>
                </div>
                <div style="text-align:center">
                    <div style="color:#888;font-size:11px;margin-bottom:4px">资金曲线</div>
                    {curve_img}
                </div>
            </div>
            <div style="margin-top:8px;font-size:10px;color:#555">
                累计交易 {m.num_trades} 笔 | 策略 tag={sp.tag}
            </div>
        </div>"""

    # Compose overall stats
    total_strategies = len(portfolio.portfolios)
    avg_return = (
        sum(p.metrics.total_return for p in portfolio.portfolios.values()) / total_strategies
        if total_strategies > 0 else 0
    )

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StockPilot 虚拟盘 Top-{top_n} {today_str}</title>
<style>
body{{background:#0f0f1a;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:16px}}
.header{{text-align:center;margin-bottom:20px}}
.header h1{{color:#4fc3f7;font-size:22px;margin:0;font-weight:bold}}
.header p{{color:#888;font-size:12px;margin:4px 0 0}}
.summary{{display:flex;gap:12px;justify-content:center;margin-bottom:20px}}
.summary-card{{background:#16213e;padding:10px 20px;border-radius:10px;text-align:center;min-width:120px}}
.summary-card .label{{color:#888;font-size:11px}}
.summary-card .value{{font-size:20px;font-weight:bold;color:#4fc3f7}}
.note{{text-align:center;color:#666;font-size:11px;margin-top:16px}}
</style>
</head>
<body>
<div class="header">
    <h1>StockPilot 虚拟盘</h1>
    <p>更新日期: {today_str} | 模型: {checkpoint} | 共追踪 {total_strategies} 个策略</p>
</div>
<div class="summary">
    <div class="summary-card">
        <div class="label">策略平均收益</div>
        <div class="value" style="color:{'#2e7d32' if avg_return >= 0 else '#c62828'}">{avg_return*100:+.2f}%</div>
    </div>
    <div class="summary-card">
        <div class="label">最佳策略</div>
        <div class="value" style="color:#81c784">{top_strategies[0].name if top_strategies else '-'}</div>
    </div>
</div>
<div class="strategies">
    {strategy_rows or '<p style="text-align:center;color:#666">暂无策略数据</p>'}
</div>
<div class="note">
    虚拟盘模拟 · 非真实交易 · 仅供参考<br>
    每日 15:05 自动更新
</div>
</body>
</html>"""

    return html
