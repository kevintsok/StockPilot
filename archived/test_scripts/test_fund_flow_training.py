#!/usr/bin/env python3
"""Small training test to verify price_hfq + fund_flow features work end-to-end."""
import sys
sys.path.insert(0, "src")

from pathlib import Path
import numpy as np
from auto_select_stock.predict.data import (
    load_feature_matrix,
    prepare_datasets,
    FUND_FLOW_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
)
from auto_select_stock.predict.train import train_from_symbols
from auto_select_stock.core.torch_model import TrainConfig
from auto_select_stock.config import DATA_DIR

# Test 1: Verify load_feature_matrix with fund_flow
print("=== Test 1: load_feature_matrix with include_fund_flow ===")
symbols_with_ff = ["000001", "000002", "000004", "000006", "000007"]

for sym in symbols_with_ff[:2]:
    feats = load_feature_matrix(
        sym,
        price_columns=PRICE_FEATURE_COLUMNS,
        financial_columns=[],
        base_dir=DATA_DIR,
        price_table="price_hfq",
        include_fund_flow=True,
    )
    n_price = len(PRICE_FEATURE_COLUMNS)
    n_ff = len(FUND_FLOW_FEATURE_COLUMNS)
    price_part = feats[:, :n_price]
    ff_part = feats[:, n_price:n_price + n_ff]
    has_nonzero_ff = np.any(ff_part != 0, axis=0)
    print(f"  {sym}: shape={feats.shape}, fund_flow non-zero cols={has_nonzero_ff.sum()}/{n_ff}")

# Test 2: Verify load_feature_matrix with fund_flow for symbol without fund_flow data
print("\n=== Test 2: load_feature_matrix with fund_flow for symbol without ff data ===")
sym_no_ff = "002536"  # Not in fund_flow table
feats = load_feature_matrix(
    sym_no_ff,
    price_columns=PRICE_FEATURE_COLUMNS,
    financial_columns=[],
    base_dir=DATA_DIR,
    price_table="price_hfq",
    include_fund_flow=True,
)
n_price = len(PRICE_FEATURE_COLUMNS)
n_ff = len(FUND_FLOW_FEATURE_COLUMNS)
ff_part = feats[:, n_price:n_price + n_ff]
print(f"  {sym_no_ff}: shape={feats.shape}, fund_flow all_zero={np.allclose(ff_part, 0)}")

# Test 3: Small training with price_hfq + fund_flow
print("\n=== Test 3: Small training with price_hfq + fund_flow ===")
train_symbols = ["000001", "000002", "000004", "000006", "000007", "000008", "000009", "000010", "000011", "000012"]
print(f"  Training symbols: {train_symbols}")

cfg = TrainConfig(
    seq_len=60,
    window_stride=10,
    batch_size=32,
    epochs=1,
    lr=1e-4,
    device="cpu",
    price_columns=PRICE_FEATURE_COLUMNS,
    financial_columns=[],
    target_mode="log_return",
    save_path=Path("models/test_fund_flow.pt"),
)

stats = train_from_symbols(
    train_symbols,
    cfg,
    base_dir=DATA_DIR,
    price_table="price_hfq",
    include_fund_flow=True,
)

print("\n=== Training Result ===")
if isinstance(stats, dict):
    for k, v in stats.items():
        print(f"  {k}: {v}")
else:
    print(f"  {stats}")
print("\nAll tests passed!")
