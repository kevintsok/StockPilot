#!/usr/bin/env python3
"""
Plot strategy comparison from saved backtest JSON results.
Usage: python plot_backtest_results.py
"""

import json
from pathlib import Path
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

RESULTS_FILE = Path("models/price_transformer_2025-train20250331-val20260327_strategies_comparison.json")

with open(RESULTS_FILE) as f:
    data = json.load(f)

results = data["results"]

# Sort by total return
results.sort(key=lambda r: r['metrics']['total_return_gross'], reverse=True)

# Color palette
colors = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3e2cd", "#fdcdac", "#cbd5e8", "#f4cae4", "#e6f5c9",
    "#fff2ae", "#f1e2cc", "#cccccc", "#8dd3c7", "#ffffb3",
    "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69",
]

INITIAL = 100000.0

fig, axes = plt.subplots(3, 1, figsize=(18, 14),
                          gridspec_kw={"height_ratios": [3, 1, 1]})
fig.suptitle(
    f"StockPilot 回测  |  2025-04-01 → 2026-03-27  |  初始: {INITIAL:,.0f} RMB\n"
    f"Model: price_transformer_2025-train20250331-val20260327.pt",
    fontsize=14, fontweight="bold",
)

ax_cap, ax_ret, ax_dd = axes

# ── Capital curves ────────────────────────────────────────────────
ax_cap.set_title(f"30 策略资金曲线（按总收益排序）", fontsize=11, color="gray")
for i, r in enumerate(results):
    cumulative = r.get("cumulative")
    if cumulative is None or len(cumulative) == 0:
        continue
    # cumulative is a dict: {date_str: value}
    dates = [mdates.date2num(d) for d in cumulative.keys()]
    capitals = list(cumulative.values())
    color = colors[i % len(colors)]
    name = r['strategy_name']
    tag = r.get('tag', '')
    label = f"{name} [{tag}]" if tag else name
    total_ret = r['metrics']['total_return_gross'] * 100
    lw = 1.2 if i < 10 else 0.6
    alpha = 0.85 if i < 10 else 0.4
    ax_cap.plot(dates, capitals, color=color, lw=lw, alpha=alpha, label=label)
    ax_cap.scatter([dates[-1]], [capitals[-1]], color=color, s=8, zorder=5)
    if i < 10:
        ax_cap.annotate(
            f"{total_ret:+.0f}%",
            xy=(dates[-1], capitals[-1]),
            xytext=(3, 0), textcoords="offset points",
            fontsize=7, color=color, va="center",
        )

ax_cap.axhline(y=INITIAL, color="gray", lw=0.8, ls="--", alpha=0.5)
ax_cap.set_ylabel("组合价值 (RMB)", fontsize=11)
ax_cap.set_ylim(bottom=0)
ax_cap.grid(True, alpha=0.2)
ax_cap.legend(loc="upper left", fontsize=6, ncol=3, framealpha=0.85)

# ── Normalized return % curves ─────────────────────────────────
for i, r in enumerate(results):
    cumulative = r.get("cumulative")
    if cumulative is None or len(cumulative) == 0:
        continue
    dates = [mdates.date2num(d) for d in cumulative.keys()]
    capitals = list(cumulative.values())
    rets = (np.array(capitals) / INITIAL - 1) * 100
    color = colors[i % len(colors)]
    lw = 1.2 if i < 10 else 0.6
    alpha = 0.85 if i < 10 else 0.4
    ax_ret.plot(dates, rets, color=color, lw=lw, alpha=alpha)

ax_ret.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5)
ax_ret.set_ylabel("收益率 (%)", fontsize=11)
ax_ret.grid(True, alpha=0.2)
ax_ret.set_title("收益率曲线", fontsize=10, color="gray")

# ── Drawdown panel ──────────────────────────────────────────────
for i, r in enumerate(results):
    cumulative = r.get("cumulative")
    if cumulative is None or len(cumulative) == 0:
        continue
    dates = [mdates.date2num(d) for d in cumulative.keys()]
    capitals = np.array(list(cumulative.values()))
    peak = np.maximum.accumulate(capitals)
    dd = (capitals / peak - 1) * 100
    color = colors[i % len(colors)]
    lw = 1.2 if i < 10 else 0.6
    alpha = 0.85 if i < 10 else 0.4
    ax_dd.fill_between(dates, dd, 0, color=color, alpha=0.1, linewidth=0)
    ax_dd.plot(dates, dd, color=color, lw=lw, alpha=alpha)

ax_dd.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5)
ax_dd.set_ylabel("回撤 (%)", fontsize=11)
ax_dd.set_xlabel("日期", fontsize=11)
ax_dd.grid(True, alpha=0.2)
ax_dd.set_title("回撤曲线", fontsize=10, color="gray")

# Format x-axis
for ax in [ax_cap, ax_ret, ax_dd]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

plt.tight_layout()
out = "models/price_transformer_2025_strategies_comparison.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

# ── Bar chart: final metrics ─────────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 8))
fig2.suptitle("策略表现对比 (2025-04-01 至 2026-03-27)", fontsize=14, fontweight="bold")

names = [f"{r['strategy_name']} [{r.get('tag','')}]" for r in results]
total_rets = [r['metrics']['total_return_gross'] * 100 for r in results]
ann_rets = [r['metrics']['annual_return_gross'] * 100 for r in results]
sharpes = [r['metrics']['sharpe_gross'] for r in results]
max_dds = [r['metrics']['max_drawdown_gross'] * 100 for r in results]
colors_bar = [colors[i % len(colors)] for i in range(len(results))]

# Sort by total return for bar chart
sort_idx = np.argsort(total_rets)[::-1]
names = [names[i] for i in sort_idx]
total_rets = [total_rets[i] for i in sort_idx]
ann_rets = [ann_rets[i] for i in sort_idx]
sharpes = [sharpes[i] for i in sort_idx]
max_dds = [max_dds[i] for i in sort_idx]
colors_bar = [colors_bar[i] for i in sort_idx]

x = np.arange(len(names))
bar_w = 0.6

axes2[0].barh(x, total_rets, color=colors_bar, alpha=0.8)
axes2[0].set_yticks(x)
axes2[0].set_yticklabels(names, fontsize=6)
axes2[0].set_xlabel("总收益 (%)")
axes2[0].set_title("总收益")
axes2[0].axvline(x=0, color="gray", lw=0.8)
axes2[0].invert_yaxis()

axes2[1].barh(x, ann_rets, color=colors_bar, alpha=0.8)
axes2[1].set_yticks(x)
axes2[1].set_yticklabels(names, fontsize=6)
axes2[1].set_xlabel("年化收益 (%)")
axes2[1].set_title("年化收益")
axes2[1].axvline(x=0, color="gray", lw=0.8)
axes2[1].invert_yaxis()

axes2[2].barh(x, sharpes, color=colors_bar, alpha=0.8)
axes2[2].set_yticks(x)
axes2[2].set_yticklabels(names, fontsize=6)
axes2[2].set_xlabel("夏普比率")
axes2[2].set_title("夏普比率")
axes2[2].axvline(x=0, color="gray", lw=0.8)
axes2[2].invert_yaxis()

axes2[3].barh(x, max_dds, color=colors_bar, alpha=0.8)
axes2[3].set_yticks(x)
axes2[3].set_yticklabels(names, fontsize=6)
axes2[3].set_xlabel("最大回撤 (%)")
axes2[3].set_title("最大回撤")
axes2[3].axvline(x=0, color="gray", lw=0.8)
axes2[3].invert_yaxis()

plt.tight_layout()
out2 = "models/price_transformer_2025_strategy_metrics_bar.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

print("\n=== 策略排名 (按总收益) ===")
print(f"{'排名':<4} {'策略':<30} {'总收益':>10} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10}")
print("-" * 80)
for rank, r in enumerate(results, 1):
    m = r['metrics']
    tag = r.get('tag', '')
    name = r['strategy_name']
    label = f"{name} [{tag}]" if tag else name
    print(f"{rank:<4} {label:<30} "
          f"{m['total_return_gross']*100:>+9.1f}% "
          f"{m['annual_return_gross']*100:>+9.1f}% "
          f"{m['sharpe_gross']:>8.2f} "
          f"{m['max_drawdown_gross']*100:>+9.1f}%")
