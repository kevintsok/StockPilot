#!/bin/bash
# StockPilot virtual portfolio daily update wrapper
# Called by cron at 15:05 Mon-Fri (after A-share market close)

source /home/julian/miniconda3/etc/profile.d/conda.sh
conda activate fin

cd /mnt/d/Projects/auto-select-stock
export NOTIFY_CHECKPOINT="models/price_transformer_2025-train20250331-val20260327.pt"
PYTHONPATH=./src python -m auto_select_stock.notify.runner --portfolio
