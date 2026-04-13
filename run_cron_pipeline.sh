#!/bin/bash
# Runs cron_pipeline.py in the fin conda environment (non-blocking via nohup)
source /home/julian/miniconda3/etc/profile.d/conda.sh
conda activate fin
cd /mnt/d/Projects/auto-select-stock
exec nohup python -u cron_pipeline.py >> logs/cron_pipeline.log 2>&1
