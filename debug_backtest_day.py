"""
Debug: trace through one day of backtest to understand the return calculation.
"""
import sys
sys.path.insert(0, 'src')

import math
from pathlib import Path
from auto_select_stock.predict.inference import PricePredictor
from auto_select_stock.predict.data import load_feature_matrix
from auto_select_stock.storage import list_symbols
import numpy as np

# Load model
p = PricePredictor('models/price_transformer_multihorizon_full.pt')
print(f"Model: target_mode={p.cfg.target_mode}, horizons={p.horizons}")

# Load all A-share symbols
symbols = list_symbols(base_dir=Path('data'))
# Filter to a few for quick test
test_symbols = ['600000', '600519', '000001', '000002', '000333']
print(f"Testing with {len(test_symbols)} symbols: {test_symbols}")

# Get features for each symbol
all_features = {}
for sym in test_symbols:
    feats = load_feature_matrix(sym,
        price_columns=p.cfg.price_columns,
        financial_columns=p.cfg.financial_columns,
        base_dir=Path('data'))
    all_features[sym] = feats
    print(f"  {sym}: features shape {feats.shape}, close[-5:]={feats[-5:, 3]}")

# Use scaler from checkpoint (or compute from all features)
scaler_mean = p.scaler['mean']
scaler_std = p.scaler['std']

# Normalize and run model for each symbol's last window
print("\n=== Model predictions for last window ===")
pred_results = {}
for sym in test_symbols:
    feats = all_features[sym]
    normed = (feats - scaler_mean) / scaler_std
    # Last 1024 windows
    context = normed[-1024:]
    import torch
    x = torch.tensor(context, dtype=torch.float32).unsqueeze(0).cuda()
    with torch.inference_mode():
        out = p.model(x)
    reg_all = out[2]  # (num_hor, 1, seq_len)
    last_close = feats[-1, 3]

    h_idx = 0  # 1d horizon
    pred_log_ret = reg_all[h_idx, 0, -1].item()
    pred_simple_ret = math.exp(pred_log_ret) - 1.0

    pred_results[sym] = {
        'pred_log_ret': pred_log_ret,
        'pred_simple_ret': pred_simple_ret,
        'last_close': last_close,
    }
    print(f"  {sym}: pred_log={pred_log_ret:.6f}, pred_simple={pred_simple_ret:.4%}, last_close={last_close:.2f}")

# Now compute what TopKStrategy would do
print("\n=== TopK Strategy (top_k=5, 1d) ===")
sorted_syms = sorted(pred_results.keys(), key=lambda s: pred_results[s]['pred_simple_ret'], reverse=True)
top5 = sorted_syms[:5]
print(f"Top 5 by predicted return: {[(s, pred_results[s]['pred_simple_ret']) for s in top5]}")

# Normalize to weights
total_pred = sum(pred_results[s]['pred_simple_ret'] for s in top5)
weights = {s: pred_results[s]['pred_simple_ret'] / total_pred for s in top5}
print(f"Weights (proportional): {weights}")
print(f"Sum of weights: {sum(weights.values()):.4f}")

# Now, what would the realized return be?
# In the backtest, realized_ret = nxt_c / cur_c - 1
# For the LAST window, we don't have nxt_c (future)
# But for previous windows we would.
# Let me use the PREVIOUS day's close as cur_c and predict the return

# Actually, let me compute for the SECOND-to-last window to check realized returns
print("\n=== Checking realized returns for second-to-last window ===")
for sym in test_symbols[:3]:
    feats = all_features[sym]
    normed = (feats - scaler_mean) / scaler_std
    # Second-to-last window
    context = normed[-1025:-1]  # 1024 window ending at penultimate
    cur_close = feats[-2, 3]  # close at end of context
    nxt_close = feats[-1, 3]  # next close (actual)

    import torch
    x = torch.tensor(context, dtype=torch.float32).unsqueeze(0).cuda()
    with torch.inference_mode():
        out = p.model(x)
    reg_all = out[2]
    pred_log_ret = reg_all[0, 0, -1].item()
    pred_simple_ret = math.exp(pred_log_ret) - 1.0
    actual_ret = nxt_close / cur_close - 1.0

    print(f"  {sym}:")
    print(f"    cur_close={cur_close:.2f}, nxt_close={nxt_close:.2f}")
    print(f"    predicted={pred_simple_ret:.4%}, actual={actual_ret:.4%}")
    print(f"    error={actual_ret - pred_simple_ret:.4%}")

# The key question: are actual returns much larger than predicted?
print("\n=== Summary ===")
for sym in test_symbols[:3]:
    pr = pred_results[sym]
    print(f"{sym}: model predicts {pr['pred_simple_ret']:.4%} but actual might be very different")