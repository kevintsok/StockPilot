#!/usr/bin/env python3
"""
Auto-optimization: reads current best strategy, generates parameter variations,
writes new candidate configs to optimized_strategies_v2.json for the next backtest cycle.
"""
import json
import random
import sys
from pathlib import Path

# Strategy configs directory
CONFIGS_DIR = Path("src/auto_select_stock/predict/strategies/configs")
OPTIMIZED_V2 = CONFIGS_DIR / "optimized_strategies_v2.json"
TOP10_FILE = Path("models/top10_strategies.json")

# Load current top strategies
if not TOP10_FILE.exists():
    print("No top10_strategies.json found, using defaults")
    best_name = "Conf-5d-K15-SL5pct-TP15pct"
else:
    with open(TOP10_FILE) as f:
        top10 = json.load(f)
    best_name = top10[0] if top10 else "Conf-5d-K15-SL5pct-TP15pct"

print(f"Best strategy to optimize: {best_name}")

# Parse strategy name to extract params
# Format: Conf-[1d|5d]-K{N}-[SLNpct][-TPNpct][-HoldN]
import re

horizon = "5d"
top_k = 15
stop_loss_pct = 0.05
take_profit_pct = 0.15
max_holding_days = 10

parts = best_name.split("-")
for i, part in enumerate(parts):
    if part in ("1d", "3d", "5d", "14d"):
        horizon = part
    if part.startswith("K"):
        try:
            top_k = int(part[1:])
        except ValueError:
            pass
    if part.startswith("SL"):
        try:
            stop_loss_pct = float(part[2:].replace("pct", "")) / 100
        except ValueError:
            pass
    if part.startswith("TP"):
        try:
            take_profit_pct = float(part[2:].replace("pct", "")) / 100
        except ValueError:
            pass
    if part.startswith("Hold"):
        try:
            max_holding_days = int(part[4:])
        except ValueError:
            pass

print(f"Parsed: horizon={horizon}, K={top_k}, SL={stop_loss_pct*100:.0f}%, TP={take_profit_pct*100:.0f}%, Hold={max_holding_days}d")

# Generate variations
candidates = []

# ── 1. Vary top_k (capped at 10 — larger K makes stop-loss ineffective) ──
for k in [3, 5, 7, 10]:
    if k == top_k:
        continue
    name = f"Conf-{horizon}-K{k}-SL{int(stop_loss_pct*100)}pct-TP{int(take_profit_pct*100)}pct"
    candidates.append({
        "name": name,
        "description": f"Auto: K={k} from {best_name}",
        "type": "confidence",
        "horizon": horizon,
        "params": {"top_k": k, "min_confidence": 0.005, "stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct},
        "tag": f"auto_k{k}",
    })

# ── 2. Vary stop-loss (fewer key values — too many SL values cause repetition) ──
for sl in [0.03, 0.05, 0.08, 0.10]:
    if abs(sl - stop_loss_pct) < 0.01:
        continue
    name = f"Conf-{horizon}-K{top_k}-SL{int(sl*100)}pct-TP{int(take_profit_pct*100)}pct"
    candidates.append({
        "name": name,
        "description": f"Auto: SL={sl*100:.0f}% from {best_name}",
        "type": "confidence",
        "horizon": horizon,
        "params": {"top_k": top_k, "min_confidence": 0.005, "stop_loss_pct": sl, "take_profit_pct": take_profit_pct},
        "tag": f"auto_sl{int(sl*100)}",
    })

# ── 3. Vary take-profit ────────────────────────────────────────────────
for tp in [0.05, 0.10, 0.20, 0.25, 0.30]:
    if abs(tp - take_profit_pct) < 0.02:
        continue
    name = f"Conf-{horizon}-K{top_k}-SL{int(stop_loss_pct*100)}pct-TP{int(tp*100)}pct"
    candidates.append({
        "name": name,
        "description": f"Auto: TP={tp*100:.0f}% from {best_name}",
        "type": "confidence",
        "horizon": horizon,
        "params": {"top_k": top_k, "min_confidence": 0.005, "stop_loss_pct": stop_loss_pct, "take_profit_pct": tp},
        "tag": f"auto_tp{int(tp*100)}",
    })

# ── 4. Vary max_holding_days ──────────────────────────────────────────
for hold in [5, 7, 15, 20]:
    if hold == max_holding_days:
        continue
    name = f"ConfStop-{horizon}-K{top_k}-SL{int(stop_loss_pct*100)}-TP{int(take_profit_pct*100)}-Hold{hold}"
    candidates.append({
        "name": name,
        "description": f"Auto: Hold={hold}d from {best_name}",
        "type": "confidence_stop",
        "horizon": horizon,
        "params": {"top_k": top_k, "min_confidence": 0.005, "stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct, "max_holding_days": hold},
        "tag": f"auto_h{hold}",
    })

# ── 5. Switch horizon (test 1d vs 5d) ─────────────────────────────────
for h in ["1d", "3d"]:
    if h == horizon:
        continue
    name = f"Conf-{h}-K{top_k}-SL{int(stop_loss_pct*100)}pct-TP{int(take_profit_pct*100)}pct"
    candidates.append({
        "name": name,
        "description": f"Auto: horizon={h} from {best_name}",
        "type": "confidence",
        "horizon": h,
        "params": {"top_k": top_k, "min_confidence": 0.005, "stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct},
        "tag": f"auto_h{h}",
    })

# ── 6. Aggressive: tight stop + no take-profit (pure stop-loss focus) ──
for k in [5, 7, 10]:
    for sl in [0.03, 0.05]:
        name = f"Conf-{horizon}-K{k}-SL{int(sl*100)}pct"
        candidates.append({
            "name": name,
            "description": f"Auto: K={k} SL={sl*100:.0f}% no-TP",
            "type": "confidence",
            "horizon": horizon,
            "params": {"top_k": k, "min_confidence": 0.005, "stop_loss_pct": sl, "take_profit_pct": 0.0},
            "tag": f"auto_agg_k{k}_sl{int(sl*100)}",
        })

# ── 7. Very tight stop + take-profit combos ───────────────────────────
for sl in [0.02, 0.03]:
    for tp in [0.05, 0.08, 0.10, 0.15]:
        if tp <= sl:
            continue
        name = f"Conf-{horizon}-K{top_k}-SL{int(sl*100)}pct-TP{int(tp*100)}pct"
        candidates.append({
            "name": name,
            "description": f"Auto: tight SL+TP from {best_name}",
            "type": "confidence",
            "horizon": horizon,
            "params": {"top_k": top_k, "min_confidence": 0.005, "stop_loss_pct": sl, "take_profit_pct": tp},
            "tag": f"auto_tight_sl{int(sl*100)}_tp{int(tp*100)}",
        })

# ── 8. Volatility-adaptive: K=10 with very tight stop ──────────────
for sl in [0.03, 0.05]:
    name = f"Conf-{horizon}-K10-SL{int(sl*100)}pct"
    candidates.append({
        "name": name,
        "description": f"Auto: K=10 tight SL={sl*100:.0f}%",
        "type": "confidence",
        "horizon": horizon,
        "params": {"top_k": 10, "min_confidence": 0.005, "stop_loss_pct": sl, "take_profit_pct": 0.0},
        "tag": f"auto_div_k10_sl{int(sl*100)}",
    })

# Deduplicate by name
seen = set()
unique = []
for c in candidates:
    if c["name"] not in seen:
        seen.add(c["name"])
        unique.append(c)

print(f"Generated {len(unique)} candidate strategies")
for c in unique[:10]:
    print(f"  {c['name']} (SL={c['params']['stop_loss_pct']*100:.0f}%, TP={c['params']['take_profit_pct']*100:.0f}%, K={c['params']['top_k']})")
if len(unique) > 10:
    print(f"  ... and {len(unique) - 10} more")

# Write to v2 file
# If the file contains "# MANUALLY DESIGNED" marker, skip to preserve manual configs
MANUAL_MARKER = "# MANUALLY DESIGNED"
if OPTIMIZED_V2.exists():
    with open(OPTIMIZED_V2) as f:
        content = f.read()
    if MANUAL_MARKER in content:
        print(f"Skipping auto-optimize — {OPTIMIZED_V2} contains manual config marker")
        print(f"Manual configs preserved, {len(unique)} candidate strategies generated (not saved)")
        sys.exit(0)

with open(OPTIMIZED_V2, "w") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)
print(f"Saved {len(unique)} configs to {OPTIMIZED_V2}")
