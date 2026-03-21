#!/bin/bash
# StockPilot daily notification wrapper
# Called by cron at 15:05 Mon-Fri (after A-share market close)

source /home/julian/miniconda3/etc/profile.d/conda.sh
conda activate fin

cd /mnt/d/Projects/auto-select-stock
PYTHONPATH=./src python -m auto_select_stock.notify.runner
