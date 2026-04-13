import sys
sys.path.insert(0, 'src')
import torch

for model_path in ['models/price_transformer_full.pt', 'models/price_transformer_multihorizon_full.pt']:
    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        print(f'=== {model_path} ===')
        if hasattr(cfg, '__dict__'):
            for k in dir(cfg):
                if not k.startswith('_'):
                    v = getattr(cfg, k, None)
                    if not callable(v):
                        print(f'  {k}: {v}')
        else:
            for k, v in cfg.items():
                print(f'  {k}: {v}')
        print()
    except Exception as e:
        print(f'{model_path}: {e}')
