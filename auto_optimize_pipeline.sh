#!/usr/bin/env bash
# Auto-optimize pipeline: optimize -> backtest -> top10 -> plot -> commit+push
set -e
cd /mnt/d/Projects/auto-select-stock

echo "========================================="
echo "[$(date)] Starting optimization pipeline"
echo "========================================="

echo "[Step1] Running auto_optimize.py..."
source /home/julian/miniconda3/etc/profile.d/conda.sh
conda activate fin
PYTHONPATH=./src python auto_optimize.py

echo "[Step2] Running backtest-strategies..."
PYTHONPATH=./src python -m auto_select_stock.cli backtest-strategies \
    --start 2025-04-01 \
    --end 2026-03-27 \
    --checkpoint models/price_transformer_2025-train20250331-val20260327.pt \
    --cost-bps 0 \
    --slippage-bps 0 \
    --strategies-dir src/auto_select_stock/predict/strategies/configs \
    --output models/price_transformer_2025-train20250331-val20260327_strategies_comparison.json

echo "[Step3] Running plot_strategies.py..."
PYTHONPATH=./src python plot_strategies.py

echo "[Step4] Checking git status..."
git add -A
if git diff --cached --quiet; then
    echo "No changes to commit."
else
    echo "[Step5] Committing..."
    git commit -m "$(cat <<'EOF'
chore(auto): strategy optimization pipeline run

- Run auto_optimize to generate new candidates
- Backtest all strategies
- Update top-10 Sharpe comparison charts

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
    echo "[Step6] Pushing..."
    git push
fi

echo "========================================="
echo "[$(date)] Pipeline complete"
echo "========================================="
