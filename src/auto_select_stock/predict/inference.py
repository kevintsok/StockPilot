import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..config import DATA_DIR
from ..torch_model import PriceTransformer, TrainConfig
from .data import close_index, load_feature_matrix


def load_model(
    checkpoint_path: Path, device: Optional[str] = None
) -> Tuple[PriceTransformer, Dict[str, float], TrainConfig, Optional[Dict[str, np.ndarray]]]:
    target_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        payload = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=target_device)
    cfg: TrainConfig = payload["config"]
    if not hasattr(cfg, "target_mode"):
        cfg.target_mode = "close"
    if not hasattr(cfg, "weight_decay"):
        cfg.weight_decay = 0.0
    if not hasattr(cfg, "grad_clip"):
        cfg.grad_clip = 0.0
    if not hasattr(cfg, "lambda_reg"):
        cfg.lambda_reg = 1.0
    if not hasattr(cfg, "lambda_cls"):
        cfg.lambda_cls = 1.0
    feature_columns = cfg.price_columns + cfg.financial_columns
    state_dict = payload["model_state"]
    d_model = state_dict["input_proj.weight"].shape[0]
    dim_feedforward = state_dict.get("encoder.layers.0.linear1.weight", torch.empty(0)).shape[0] or 256
    # Infer layer count from encoder layer keys to stay compatible with checkpoints trained using
    # different num_layers defaults.
    layer_keys = [k for k in state_dict.keys() if k.startswith("encoder.layers.")]
    num_layers = 0
    for k in layer_keys:
        parts = k.split(".")
        if len(parts) > 2 and parts[2].isdigit():
            num_layers = max(num_layers, int(parts[2]) + 1)
    num_layers = max(num_layers, 1)
    # Prefer config-provided nhead when available; otherwise pick a divisor of d_model.
    nhead = getattr(cfg, "nhead", 8)
    if nhead <= 0 or d_model % nhead != 0:
        for candidate in (8, 4, 2, 1):
            if d_model % candidate == 0:
                nhead = candidate
                break
    dropout = getattr(cfg, "dropout", 0.1)
    model = PriceTransformer(
        input_dim=len(feature_columns),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(target_device)
    model.eval()
    scaler = payload.get("scaler")
    return model, payload.get("metrics", {}), cfg, scaler


class PricePredictor:
    """
    Lightweight inference helper that loads a checkpoint once and reuses the model/scaler
    for multiple symbols to avoid重复反复反序列化。
    """

    def __init__(self, checkpoint_path: Path, device: Optional[str] = None):
        model, _, cfg, scaler = load_model(checkpoint_path, device=device)
        self.model = model
        self.cfg = cfg
        self.scaler = scaler
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_columns = cfg.price_columns + cfg.financial_columns
        self.close_idx = close_index(self.feature_columns)
        self.model.to(self.device)
        self.model.eval()

    def _validate_features(self, features: np.ndarray, symbol: str) -> None:
        if features.ndim != 2 or features.shape[1] != len(self.feature_columns):
            raise RuntimeError(
                f"Feature shape mismatch for {symbol}: expected (?,{len(self.feature_columns)}) "
                f"from checkpoint, got {features.shape}"
            )

    def predict(
        self,
        symbol: str,
        seq_len: Optional[int] = None,
        base_dir: Path = DATA_DIR,
        features: Optional[np.ndarray] = None,
    ) -> float:
        seq_len = seq_len or self.cfg.seq_len
        feats = features
        if feats is None:
            feats = load_feature_matrix(
                symbol,
                price_columns=self.cfg.price_columns,
                financial_columns=self.cfg.financial_columns,
                base_dir=base_dir,
            )
        self._validate_features(feats, symbol)

        if self.scaler is None:
            # Need an extra day to compute scaler from history.
            if len(feats) < seq_len + 1:
                raise RuntimeError(f"Not enough data for symbol {symbol} to build a {seq_len}-length context.")
            scaler_mean = feats[:-1].mean(axis=0)
            scaler_std = feats[:-1].std(axis=0) + 1e-6
        else:
            if len(feats) < seq_len:
                raise RuntimeError(f"Not enough data for symbol {symbol} to build a {seq_len}-length context.")
            scaler_mean = self.scaler["mean"]
            scaler_std = self.scaler["std"]

        normed = (feats - scaler_mean) / scaler_std
        context = normed[-seq_len:]
        x = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_seq = self.model(x)[0].detach().cpu().numpy()

        # Handle batch dimension consistently; we expect shape (batch, seq_len).
        if pred_seq.ndim == 2:
            pred_last = float(pred_seq[0, -1])
        elif pred_seq.ndim == 1:
            pred_last = float(pred_seq[-1])
        else:
            pred_last = float(np.array(pred_seq).reshape(-1)[-1])

        mode = getattr(self.cfg, "target_mode", "close")
        if mode == "log_return":
            last_close = float(feats[-1, self.close_idx])
            predicted_price = last_close * math.exp(pred_last)
        else:
            predicted_price = pred_last * scaler_std[self.close_idx] + scaler_mean[self.close_idx]
        return float(predicted_price)


def predict_next_close(
    symbol: str,
    checkpoint_path: Path,
    seq_len: Optional[int] = None,
    base_dir: Path = DATA_DIR,
    device: Optional[str] = None,
) -> float:
    predictor = PricePredictor(checkpoint_path, device=device)
    return predictor.predict(symbol, seq_len=seq_len, base_dir=base_dir)


__all__ = ["PricePredictor", "predict_next_close", "load_model"]
