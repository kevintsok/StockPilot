#!/usr/bin/env python3
"""
Usage:
    python plot_backtest.py <backtest_results_full.json> [output.png]

Generates capital curve chart from the full backtest JSON (with timeseries + trades).
  - Left axis: portfolio value (RMB)
  - Right axis: strategy return % + benchmark index return % (both in %)
  - Bottom panel: daily returns distribution
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Support both old single-dict format and new list-of-strategies format
    if isinstance(data, dict):
        return [data]
    return data


def fetch_index(symbol: str, start: str, end: str) -> tuple:
    """Fetch index daily data. Returns (dates, close_prices)."""
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol=symbol)
        df["date"] = df["date"].astype(str).str[:10]
        mask = (df["date"] >= start) & (df["date"] <= end)
        df = df[mask].sort_values("date")
        return list(df["date"]), list(df["close"])
    except Exception:
        return [], []


def normalize_to_pct(values: list, base_value: float) -> list:
    """Convert series to % return from base_value."""
    if not values or base_value == 0:
        return values
    return [(v / base_value - 1) * 100 for v in values]


def plot_backtest(json_path: str, output_path: str = None) -> None:
    strategies = load_json(json_path)

    if not strategies:
        print("No results found.")
        return

    # Infer start/end from first strategy
    start = strategies[0].get("start", "2023-01-01")
    end = strategies[0].get("end", "2024-12-31")
    initial = strategies[0].get("initial_capital", 100_000)

    # Filter out strategies with max_drawdown < -80% (too risky)
    MAX_DD_THRESHOLD = -0.80
    filtered = [r for r in strategies if r.get("max_drawdown", 0) > MAX_DD_THRESHOLD]
    print(f"Filtered {len(strategies) - len(filtered)} strategies with max_dd <= {MAX_DD_THRESHOLD*100:.0f}%; "
          f"showing {len(filtered)} strategies")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(
        f"StockPilot Backtest  |  {start} → {end}  |  Initial: {initial:,.0f} RMB",
        fontsize=13, fontweight="bold",
    )

    ax1_r = ax1.twinx()

    strategy_colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
        "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    ]

    # ── Fetch benchmark indices ──────────────────────────────────────
    index_configs = [
        ("sh000001", "SSE Composite"),
        ("sz399001", "SZSE Component"),
        ("sz399006", "ChiNext"),
    ]

    index_data = {}
    for symbol, name in index_configs:
        dates, closes = fetch_index(symbol, start, end)
        if dates and closes:
            rets = normalize_to_pct(closes, closes[0])
            # Normalize index to capital: index_capitals[i] = initial * (1 + rets[i]/100)
            index_capitals = [initial * (1 + r / 100) for r in rets]
            index_data[name] = (dates, index_capitals, rets)
            print(f"Loaded {name}: {dates[0]} → {dates[-1]}, {len(dates)} pts, "
                  f"return: {rets[-1]:+.1f}%")

    index_colors = {"SSE Composite": "#FF6B6B", "SZSE Component": "#4ECDC4", "ChiNext": "#FFD93D"}

    # ── Plot indices on left axis (as normalized capital starting at initial) ──
    for name, (dates, index_capitals, rets) in index_data.items():
        x = [mdates.datestr2num(d.replace("/", "-")) for d in dates]
        ax1.plot(x, index_capitals,
                 color=index_colors[name],
                 ls="--", lw=2.8, alpha=0.85,
                 label=f"Index: {name} ({rets[-1]:+.1f}%)", zorder=3)

    # ── Plot strategies on left axis (absolute capital, all starting at initial) ──
    for i, r in enumerate(filtered):
        ts = r.get("timeseries", [])
        if not ts:
            continue

        dates = [t["date"].replace("/", "-") for t in ts]
        x = [mdates.datestr2num(d) for d in dates]
        capitals = [t.get("capital") or initial for t in ts]
        daily_rets = [t.get("daily_return", 0) for t in ts]
        turnovers = [t.get("turnover", 0) for t in ts]

        total_ret = r.get("total_return", 0)
        color = strategy_colors[i % len(strategy_colors)]
        name = r.get("strategy_name") or r.get("strategy", f"Strategy {i+1}")
        tag = r.get("tag", "")
        label = f"{name} [{tag}]" if tag else name

        # Left axis: capital in RMB (all start at ~initial, showing growth)
        ax1.plot(x, capitals, color=color, lw=1.0, alpha=0.7, label=label, zorder=4)
        ax1.scatter([x[-1]], [capitals[-1]], color=color, s=10, zorder=6)
        ax1.annotate(
            f"{total_ret*100:+.0f}%",
            xy=(x[-1], capitals[-1]),
            xytext=(4, 0), textcoords="offset points",
            fontsize=6, color=color, va="center", zorder=7,
        )

        # Bottom panels
        ax2.fill_between(x, daily_rets, 0, color=color, alpha=0.15, linewidth=0)
        ax2.plot(x, daily_rets, color=color, lw=0.4, alpha=0.4)
        ax3.fill_between(x, turnovers, 0, color=color, alpha=0.1, linewidth=0)
        ax3.plot(x, turnovers, color=color, lw=0.4, alpha=0.5)

    # ── Left axis: capital in RMB (all lines start at initial) ───────────────────
    ax1.axhline(y=initial, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=2)
    ax1.set_ylabel("Portfolio Value (RMB)", fontsize=11)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.2)

    # ── Right axis: return % from initial ─────────────────────────────────────
    # Compute return % from initial for all lines
    all_rets = []
    for r in filtered:
        ts = r.get("timeseries", [])
        if ts:
            caps = [t.get("capital") or initial for t in ts]
            rets_pct = [(c / initial - 1) * 100 for c in caps]
            all_rets.extend(rets_pct)
    for name, (dates, index_capitals, rets_pct) in index_data.items():
        all_rets.extend(rets_pct)

    # Symlog scale to handle: indices at ±10%, top strategies at +2000%
    if all_rets:
        min_r = min(all_rets)
        max_r = max(all_rets)
        # Use symlog: linthreshx determines linear vs log boundary
        lin_thresh = max(abs(min_r) * 0.1, abs(max_r) * 0.01, 5)
        ax1_r.set_yscale('symlog', linthresh=lin_thresh)
        ax1_r.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax1_r.set_ylabel("Return from Initial (%)", fontsize=11)
    ax1_r.grid(False)

    # ── Title & legends ───────────────────────────────────────────────
    index_names = ", ".join(index_data.keys())
    ax1.set_title(
        f"{len(filtered)} strategies (max_dd > {MAX_DD_THRESHOLD*100:.0f}%)  |  "
        f"Indices: {index_names}  |  Top: Capital (RMB)  |  Middle: Daily Return  |  Bottom: Turnover",
        fontsize=9, color="gray",
    )

    handles_l, labels_l = ax1.get_legend_handles_labels()
    handles_r, labels_r = ax1_r.get_legend_handles_labels()
    ordered = [(h, l) for h, l in zip(handles_r, labels_r)] + \
              [(h, l) for h, l in zip(handles_l, labels_l)]
    ordered_h = [h for h, _ in ordered]
    ordered_l = [l for _, l in ordered]

    ax1.legend(ordered_h, ordered_l,
               loc="upper left", fontsize=6, ncol=2,
               framealpha=0.85, edgecolor="none")

    # ── Daily returns panel ─────────────────────────────────────────────
    ax2.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax2.set_ylabel("Daily Return", fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.set_title("Daily Returns", fontsize=9, color="gray")

    # ── Turnover panel ───────────────────────────────────────────────
    ax3.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax3.set_ylabel("Turnover", fontsize=10)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.grid(True, alpha=0.2)
    ax3.set_title("Daily Turnover Rate", fontsize=9, color="gray")

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    plt.tight_layout()

    out = output_path or str(
        Path(json_path).parent / (Path(json_path).stem + "_capital_curve.png")
    )
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_backtest.py <backtest_results_full.json> [output.png]")
        sys.exit(1)
    json_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    plot_backtest(json_path, out_path)
