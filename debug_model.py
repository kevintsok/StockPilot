from auto_select_stock.predict.inference import PricePredictor
from auto_select_stock.predict.data import load_feature_matrix
import torch
import numpy as np
from pathlib import Path

p = PricePredictor('models/price_transformer_multihorizon_full.pt')
print('target_mode:', p.cfg.target_mode)
print('horizons:', p.horizons)
print('close_idx:', p.close_idx)
print('feature_columns:', len(p.feature_columns))
print('scaler mean shape:', p.scaler['mean'].shape if p.scaler else None)
print('scaler std shape:', p.scaler['std'].shape if p.scaler else None)

# Check scaler for close column
if p.scaler is not None:
    print('\nScaler mean[close_idx]:', p.scaler['mean'][p.close_idx])
    print('Scaler std[close_idx]:', p.scaler['std'][p.close_idx])

# Load features for 600000
feats = load_feature_matrix('600000',
    price_columns=p.cfg.price_columns,
    financial_columns=p.cfg.financial_columns,
    base_dir=Path('data'))
print('\nFeatures shape:', feats.shape)
print('Last 5 close prices from features (raw):')
# Find close column index in raw features
close_col_idx = p.cfg.price_columns.index('close') if 'close' in p.cfg.price_columns else 3
print('Close col idx in price_columns:', close_col_idx)
print('Last 5 rows, close column:', feats[-5:, close_col_idx] if close_col_idx < feats.shape[1] else 'N/A')

# Run model inference
x = torch.tensor(feats[-1024:], dtype=torch.float32).unsqueeze(0).cuda()
with torch.inference_mode():
    out = p.model(x)
print('\nModel output:')
print('out[0] shape:', out[0].shape)
if len(out) == 4:
    print('out[2] shape:', out[2].shape)
    reg_all = out[2]
    for h_i, h in enumerate(p.horizons):
        pred = reg_all[h_i, 0, -1].item()
        print(f'Horizon {h}d: raw={pred:.6f}, exp-1={np.exp(pred)-1:.6f}')
