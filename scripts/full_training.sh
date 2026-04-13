#!/bin/bash
# Full training script: price_hfq + fund_flow, seq_len=252 (1 trading year)
# Run after migration completes
source /home/julian/miniconda3/etc/profile.d/conda.sh
conda activate fin
cd /mnt/d/Projects/auto-select-stock

PYTHONPATH=./src python -m auto_select_stock.cli train-transformer \
  --seq-len 252 \
  --epochs 20 \
  --batch-size 64 \
  --save-path models/price_transformer_full_hfq_ff.pt \
  --price-table price_hfq \
  --include-fund-flow \
  --device cuda \
  2>&1 | tee logs/full_training_$(date +%Y%m%d_%H%M%S).log
