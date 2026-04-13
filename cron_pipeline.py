#!/usr/bin/env python3
"""
Hourly cron pipeline:
  1. Auto-optimize (generate new strategy configs)
  2. Run full backtest of all strategies
  3. Sort by Sharpe, save top10
  4. Plot top 10
  5. Push via PushPlus
  6. Compact (self trim)
"""
import datetime
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_DIR = Path("/mnt/d/Projects/auto-select-stock")
SRC_DIR = REPO_DIR / "src"
CHECKPOINT = "models/price_transformer_2025-train20250331-val20260327.pt"
RESULTS_FILE = REPO_DIR / "models" / "price_transformer_2025-train20250331-val20260327_strategies_comparison.json"
TOP10_FILE = REPO_DIR / "models" / "top10_strategies.json"

# ── Step 0: Log start ─────────────────────────────────────────────────────────
now = datetime.datetime.now()
LOCK_FILE = REPO_DIR / "logs" / "cron_pipeline.lock"

print(f"\n{'='*60}")
print(f"CRON PIPELINE START {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

# Check for existing run (互斥锁)
if LOCK_FILE.exists():
    mtime = datetime.datetime.fromtimestamp(LOCK_FILE.stat().st_mtime)
    age = (now - mtime).total_seconds() / 3600
    if age < 3:  # Less than 3 hours old → previous run still active
        print(f"Previous run still active (lock age: {age:.1f}h). Exiting.")
        sys.exit(0)
    else:
        print(f"Stale lock found (age: {age:.1f}h). Removing and continuing.")
        LOCK_FILE.unlink()

# Create lock
LOCK_FILE.touch()

def run_cmd(cmd: str, timeout: int = 10800) -> subprocess.CompletedProcess:
    """Run command in the current WSL fin conda environment."""
    # We're already inside WSL from run_cron_pipeline.sh, so just run directly
    full_cmd = (
        f"source /home/julian/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate fin && "
        f"cd /mnt/d/Projects/auto-select-stock && "
        f"{cmd}"
    )
    result = subprocess.run(
        ["/bin/bash", "-c", full_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


# ── Step 1: Auto-optimize ─────────────────────────────────────────────────────
print("\n[1/5] Running auto-optimization...")
sys.path.insert(0, str(REPO_DIR))
try:
    import auto_optimize as _ao
    _ao.OPTIMIZED_V2 = REPO_DIR / "src" / "auto_select_stock" / "predict" / "strategies" / "configs" / "optimized_strategies_v2.json"
    _ao.TOP10_FILE = TOP10_FILE
    # Re-run the generation
    import importlib
    importlib.reload(_ao)
    print(f"  Generated {len(json.load(open(_ao.OPTIMIZED_V2)))} candidate configs")
except Exception as e:
    print(f"  WARNING: auto-optimize failed: {e}")


# ── Step 2: Run backtest ─────────────────────────────────────────────────────
print("\n[2/5] Running backtest (all strategies)...")
backtest_start = datetime.datetime.now()

# Check if we have a valid checkpoint
if not REPO_DIR.joinpath(CHECKPOINT).exists():
    print(f"  WARNING: checkpoint {CHECKPOINT} not found, skipping backtest")
else:
    result = run_cmd(
        f"PYTHONPATH=./src python -m auto_select_stock.cli backtest-strategies "
        f"--start 2025-04-01 --end 2026-03-27 "
        f"--checkpoint {CHECKPOINT} "
        f"--cost-bps 0 --slippage-bps 0 "
        f"--strategies-dir src/auto_select_stock/predict/strategies/configs "
        f"--output models/price_transformer_2025-train20250331-val20260327_strategies_comparison.json",
        timeout=7200,
    )
    backtest_duration = (datetime.datetime.now() - backtest_start).total_seconds() / 60
    print(f"  Backtest done in {backtest_duration:.1f} min, exit={result.returncode}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-1000:]}")
    else:
        print(f"  STDOUT (last 3 lines):")
        for line in result.stdout.strip().split('\n')[-3:]:
            print(f"    {line}")


# ── Step 3: Sort by Sharpe ────────────────────────────────────────────────────
print("\n[3/5] Sorting by Sharpe...")
if not RESULTS_FILE.exists():
    print(f"  ERROR: results file not found: {RESULTS_FILE}")
else:
    try:
        sys.path.insert(0, str(REPO_DIR))
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        results = data["results"]
        valid = [r for r in results if not (r["metrics"].get("sharpe_gross") != r["metrics"].get("sharpe_gross"))]  # filter nan
        import math
        valid = [r for r in results if not (isinstance(r["metrics"].get("sharpe_gross"), float) and math.isnan(r["metrics"].get("sharpe_gross", 0)))]
        valid.sort(key=lambda r: r["metrics"].get("sharpe_gross", 0), reverse=True)
        top10_names = [r["strategy_name"] for r in valid[:10]]
        with open(TOP10_FILE, "w") as f:
            json.dump(top10_names, f)
        print(f"  Top 10 strategies:")
        for i, r in enumerate(valid[:10], 1):
            m = r["metrics"]
            print(f"    {i:2d}. {r['strategy_name']:<40} sharpe={m.get('sharpe_gross', 0):.3f}  dd={m.get('max_drawdown_gross',0)*100:+6.1f}%")
    except Exception as e:
        print(f"  ERROR sorting: {e}")


# ── Step 4: Plot ──────────────────────────────────────────────────────────────
print("\n[4/5] Plotting top 10...")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np

    RESULTS_FILE = REPO_DIR / "models" / "price_transformer_2025-train20250331-val20260327_strategies_comparison.json"
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    results = data["results"]
    valid = [r for r in results if isinstance(r["metrics"].get("sharpe_gross"), float)]
    import math
    valid = [r for r in valid if not math.isnan(r["metrics"].get("sharpe_gross", 0))]
    valid.sort(key=lambda r: r["metrics"]["sharpe_gross"], reverse=True)
    top10 = valid[:10]

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
              "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62"]
    INITIAL = 100000.0

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(
        f"StockPilot Top-10 Strategies by Sharpe | 2025-04-01 to 2026-03-27\n"
        f"Updated: {now.strftime('%Y-%m-%d %H:%M')}",
        fontsize=14, fontweight="bold",
    )
    ax_cap, ax_ret, ax_dd = axes

    for i, r in enumerate(top10):
        cumulative = r.get("cumulative")
        if not cumulative:
            continue
        dates = [mdates.date2num(d) for d in cumulative.keys()]
        capitals = list(cumulative.values())
        color = colors[i % len(colors)]
        name = r["strategy_name"]
        tag = r.get("tag", "")
        label = f"{name} [{tag}]" if tag else name
        total_ret = r["metrics"]["total_return_gross"] * 100
        sharpe = r["metrics"]["sharpe_gross"]
        ax_cap.plot(dates, capitals, color=color, lw=1.8, alpha=0.9, label=label)
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

    for i, r in enumerate(top10):
        cumulative = r.get("cumulative")
        if not cumulative:
            continue
        dates = [mdates.date2num(d) for d in cumulative.keys()]
        capitals = list(cumulative.values())
        rets = (np.array(capitals) / INITIAL - 1) * 100
        color = colors[i % len(colors)]
        ax_ret.plot(dates, rets, color=color, lw=1.5, alpha=0.85)

    ax_ret.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_ret.set_ylabel("Return (%)", fontsize=11)
    ax_ret.grid(True, alpha=0.2)

    for i, r in enumerate(top10):
        cumulative = r.get("cumulative")
        if not cumulative:
            continue
        dates = [mdates.date2num(d) for d in cumulative.keys()]
        capitals = np.array(list(cumulative.values()))
        peak = np.maximum.accumulate(capitals)
        dd = (capitals / peak - 1) * 100
        color = colors[i % len(colors)]
        ax_dd.fill_between(dates, dd, 0, color=color, alpha=0.1, linewidth=0)
        ax_dd.plot(dates, dd, color=color, lw=1.5, alpha=0.85)

    ax_dd.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=11)
    ax_dd.set_xlabel("Date", fontsize=11)
    ax_dd.grid(True, alpha=0.2)

    for ax in [ax_cap, ax_ret, ax_dd]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    out = REPO_DIR / "models" / "price_transformer_2025_top10_sharpe_comparison.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # Bar chart
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 8))
    fig2.suptitle(f"Top-10 Strategies by Sharpe (Updated: {now.strftime('%Y-%m-%d %H:%M')})", fontsize=14, fontweight="bold")
    names = [f"{r['strategy_name']} [{r.get('tag','')}]" for r in top10]
    total_rets = [r["metrics"]["total_return_gross"] * 100 for r in top10]
    ann_rets = [r["metrics"]["annual_return_gross"] * 100 for r in top10]
    sharpes = [r["metrics"]["sharpe_gross"] for r in top10]
    max_dds = [r["metrics"]["max_drawdown_gross"] * 100 for r in top10]
    x = np.arange(len(names))
    for ax, vals, title in zip(axes2, [total_rets, ann_rets, sharpes, max_dds],
                                 ["Total Return", "Annual Return", "Sharpe Ratio", "Max Drawdown"]):
        ax.barh(x, vals, color=colors[:len(top10)], alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(title)
        ax.axvline(x=0, color="gray", lw=0.8)
        ax.invert_yaxis()
    plt.tight_layout()
    out2 = REPO_DIR / "models" / "price_transformer_2025_top10_sharpe_metrics_bar.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2}")

except Exception as e:
    import traceback
    print(f"  ERROR plotting: {e}")
    traceback.print_exc()


# ── Step 5: Push ──────────────────────────────────────────────────────────────
print("\n[5/5] Pushing via PushPlus...")
try:
    sys.path.insert(0, str(REPO_DIR / "src"))
    from auto_select_stock.notify.push_providers import PushPlusProvider
    from auto_select_stock.notify.daily_report import generate_report, get_latest_price_date

    TOKEN = os.getenv("PUSHPLUS_TOKEN", "183cae5e7d8148f0b85754a2912fc81c")

    # Read best strategy from top10
    if TOP10_FILE.exists():
        with open(TOP10_FILE) as f:
            top10 = json.load(f)
        strategy = top10[0] if top10 else "Conf-5d-K15-SL5pct-TP15pct"
    else:
        strategy = "Conf-5d-K15-SL5pct-TP15pct"

    html, results_list = generate_report(
        checkpoint=str(REPO_DIR / CHECKPOINT),
        strategy=strategy,
        top_k=10,
        horizon="5d",
    )
    latest_date = get_latest_price_date()
    provider = PushPlusProvider(token=TOKEN)
    provider.send(
        title=f"StockPilot {strategy} Top-10 {latest_date} SR={now.strftime('%H:%M')}",
        content=html,
    )
    print(f"  Pushed: {len(results_list)} stocks, strategy={strategy}")
except Exception as e:
    import traceback
    print(f"  ERROR pushing: {e}")
    traceback.print_exc()

# ── Done ─────────────────────────────────────────────────────────────────────
try:
    LOCK_FILE.unlink()
except Exception:
    pass
print(f"\n{'='*60}")
print(f"CRON PIPELINE DONE {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")
