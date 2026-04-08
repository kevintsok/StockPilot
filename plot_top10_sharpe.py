#!/usr/bin/env python3
"""
Plot top 10 strategies by Sharpe ratio from backtest results.
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

# Sort by Sharpe descending, filter out nan
valid_results = [r for r in results if not np.isnan(r['metrics']['sharpe_gross'])]
valid_results.sort(key=lambda r: r['metrics']['sharpe_gross'], reverse=True)
top10 = valid_results[:10]

# Color palette
colors = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
]

INITIAL = 100000.0

fig, axes = plt.subplots(3, 1, figsize=(18, 14),
                          gridspec_kw={"height_ratios": [3, 1, 1]})
fig.suptitle(
    f"StockPilot Top-10 Strategies by Sharpe | 2025-04-01 to 2026-03-27\n"
    f"Model: price_transformer_2025-train20250331-val20260327.pt",
    fontsize=14, fontweight="bold",
)

ax_cap, ax_ret, ax_dd = axes

# ── Capital curves ────────────────────────────────────────────────
ax_cap.set_title("Capital Curves (Top 10 by Sharpe)", fontsize=11, color="gray")
for i, r in enumerate(top10):
    cumulative = r.get("cumulative")
    if cumulative is None or len(cumulative) == 0:
        continue
    dates = [mdates.date2num(d) for d in cumulative.keys()]
    capitals = list(cumulative.values())
    color = colors[i % len(colors)]
    name = r['strategy_name']
    tag = r.get('tag', '')
    label = f"{name} [{tag}]" if tag else name
    total_ret = r['metrics']['total_return_gross'] * 100
    sharpe = r['metrics']['sharpe_gross']
    lw = 1.8
    alpha = 0.9
    ax_cap.plot(dates, capitals, color=color, lw=lw, alpha=alpha, label=label)
    ax_cap.scatter([dates[-1]], [capitals[-1]], color=color, s=12, zorder=5)
    ax_cap.annotate(
        f"{total_ret:+.0f}% (SR={sharpe:.2f})",
        xy=(dates[-1], capitals[-1]),
        xytext=(3, 0), textcoords="offset points",
        fontsize=8, color=color, va="center",
    )

ax_cap.axhline(y=INITIAL, color="gray", lw=0.8, ls="--", alpha=0.5)
ax_cap.set_ylabel("Portfolio Value (RMB)", fontsize=11)
ax_cap.set_ylim(bottom=0)
ax_cap.grid(True, alpha=0.2)
ax_cap.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)

# ── Normalized return % curves ─────────────────────────────────
for i, r in enumerate(top10):
    cumulative = r.get("cumulative")
    if cumulative is None or len(cumulative) == 0:
        continue
    dates = [mdates.date2num(d) for d in cumulative.keys()]
    capitals = list(cumulative.values())
    rets = (np.array(capitals) / INITIAL - 1) * 100
    color = colors[i % len(colors)]
    lw = 1.5
    alpha = 0.85
    ax_ret.plot(dates, rets, color=color, lw=lw, alpha=alpha)

ax_ret.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5)
ax_ret.set_ylabel("Return (%)", fontsize=11)
ax_ret.grid(True, alpha=0.2)
ax_ret.set_title("Return % Curves", fontsize=10, color="gray")

# ── Drawdown panel ──────────────────────────────────────────────
for i, r in enumerate(top10):
    cumulative = r.get("cumulative")
    if cumulative is None or len(cumulative) == 0:
        continue
    dates = [mdates.date2num(d) for d in cumulative.keys()]
    capitals = np.array(list(cumulative.values()))
    peak = np.maximum.accumulate(capitals)
    dd = (capitals / peak - 1) * 100
    color = colors[i % len(colors)]
    lw = 1.5
    alpha = 0.85
    ax_dd.fill_between(dates, dd, 0, color=color, alpha=0.1, linewidth=0)
    ax_dd.plot(dates, dd, color=color, lw=lw, alpha=alpha)

ax_dd.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5)
ax_dd.set_ylabel("Drawdown (%)", fontsize=11)
ax_dd.set_xlabel("Date", fontsize=11)
ax_dd.grid(True, alpha=0.2)
ax_dd.set_title("Drawdown Curves", fontsize=10, color="gray")

# Format x-axis
for ax in [ax_cap, ax_ret, ax_dd]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

plt.tight_layout()
out = "models/price_transformer_2025_top10_sharpe_comparison.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

# ── Bar chart: final metrics ─────────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 8))
fig2.suptitle("Top-10 Strategies by Sharpe (2025-04-01 to 2026-03-27)", fontsize=14, fontweight="bold")

names = [f"{r['strategy_name']} [{r.get('tag','')}]" for r in top10]
total_rets = [r['metrics']['total_return_gross'] * 100 for r in top10]
ann_rets = [r['metrics']['annual_return_gross'] * 100 for r in top10]
sharpes = [r['metrics']['sharpe_gross'] for r in top10]
max_dds = [r['metrics']['max_drawdown_gross'] * 100 for r in top10]

x = np.arange(len(names))
bar_w = 0.6

axes2[0].barh(x, total_rets, color=colors[:len(top10)], alpha=0.8)
axes2[0].set_yticks(x)
axes2[0].set_yticklabels(names, fontsize=8)
axes2[0].set_xlabel("Total Return (%)")
axes2[0].set_title("Total Return")
axes2[0].axvline(x=0, color="gray", lw=0.8)
axes2[0].invert_yaxis()

axes2[1].barh(x, ann_rets, color=colors[:len(top10)], alpha=0.8)
axes2[1].set_yticks(x)
axes2[1].set_yticklabels(names, fontsize=8)
axes2[1].set_xlabel("Annual Return (%)")
axes2[1].set_title("Annual Return")
axes2[1].axvline(x=0, color="gray", lw=0.8)
axes2[1].invert_yaxis()

axes2[2].barh(x, sharpes, color=colors[:len(top10)], alpha=0.8)
axes2[2].set_yticks(x)
axes2[2].set_yticklabels(names, fontsize=8)
axes2[2].set_xlabel("Sharpe Ratio")
axes2[2].set_title("Sharpe Ratio")
axes2[2].axvline(x=0, color="gray", lw=0.8)
axes2[2].invert_yaxis()

axes2[3].barh(x, max_dds, color=colors[:len(top10)], alpha=0.8)
axes2[3].set_yticks(x)
axes2[3].set_yticklabels(names, fontsize=8)
axes2[3].set_xlabel("Max Drawdown (%)")
axes2[3].set_title("Max Drawdown")
axes2[3].axvline(x=0, color="gray", lw=0.8)
axes2[3].invert_yaxis()

plt.tight_layout()
out2 = "models/price_transformer_2025_top10_sharpe_metrics_bar.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)
