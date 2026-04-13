import sys
sys.path.insert(0, 'src')

from auto_select_stock.predict.inference import PricePredictor
from auto_select_stock.predict.data import load_feature_matrix
import numpy as np
from pathlib import Path

p = PricePredictor('models/price_transformer_multihorizon_full.pt')
print('=== Model Config ===')
print('target_mode:', p.cfg.target_mode)
print('horizons:', p.horizons)
print('close_idx:', p.close_idx)
print('feature_columns:', len(p.feature_columns))
print()

if p.scaler is not None:
    print('=== Scaler (from checkpoint) ===')
    sm = p.scaler['mean']
    ss = p.scaler['std']
    print('scaler mean shape:', sm.shape)
    print('scaler std shape:', ss.shape)
    print()
    print('scaler mean[close_idx=3]:', sm[p.close_idx])
    print('scaler std[close_idx=3]:', ss[p.close_idx])
    print()

# Load raw features for 600000
feats = load_feature_matrix('600000',
    price_columns=p.cfg.price_columns,
    financial_columns=p.cfg.financial_columns,
    base_dir=Path('data'))
print('=== Raw Features ===')
print('shape:', feats.shape)
print('close column (col 3) last 5:')
close_col = 3  # close is typically 4th column
print(feats[-5:, close_col])
print('mean of close col:', feats[:, close_col].mean())
print('std of close col:', feats[:, close_col].std())
print()

# Normalize using scaler
normed = (feats - sm) / ss
print('=== Normalized Features ===')
print('normalized close col last 5:', normed[-5:, p.close_idx])
print('normalized close col mean:', normed[:, p.close_idx].mean())
print('normalized close col std:', normed[:, p.close_idx].std())
print()

# What does the model predict for the last window?
import torch
x = torch.tensor(normed[-1024:], dtype=torch.float32).unsqueeze(0).cuda()
with torch.inference_mode():
    out = p.model(x)

if len(out) == 4:
    reg_all = out[2]
    print('=== Model Predictions (last window) ===')
    for h_i, h in enumerate(p.horizons):
        pred = reg_all[h_i, 0, -1].item()
        print(f'Horizon {h}d: raw={pred:.6f}, exp-1={np.exp(pred)-1:.6f} ({(np.exp(pred)-1)*100:.2f}%)')
else:
    pred = out[0][0, -1].item()
    print(f'1d: raw={pred:.6f}, exp-1={np.exp(pred)-1:.6f}')

# Check: what would the close price be if we denormalized?
last_normed_close = normed[-1, p.close_idx]
denorm_close = last_normed_close * ss[p.close_idx] + sm[p.close_idx]
print()
print('=== Denormalization Check ===')
print(f'Last normalized close: {last_normed_close:.6f}')
print(f'Denormalized close: {denorm_close:.2f}')
print(f'Actual raw close: {feats[-1, close_col]:.2f}')
