import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from ..config import DATA_DIR
from ..core.torch_model import DEFAULT_HORIZONS, PriceTransformer, TrainConfig
from .data import TECHNICAL_FEATURE_COLUMNS, close_index, load_feature_matrix


class CheckpointMigrator:
    """
    Handles backward-compatibility migration for legacy checkpoint formats.

    Older checkpoints may be missing fields added in later versions, or may use a
    single-head model architecture instead of the current multi-head design.
    This class provides structured, documented helpers to bring them up to date.
    """

    def __init__(self, cfg: TrainConfig, state_dict: Dict):
        self.cfg = cfg
        self.state_dict = state_dict

    def migrate_config(self) -> None:
        """
        Backward-compat: fill in any TrainConfig fields that are missing from
        older checkpoints so that downstream code can rely on them being present.
        """
        if not hasattr(self.cfg, "target_mode"):
            self.cfg.target_mode = "close"
        if not hasattr(self.cfg, "weight_decay"):
            self.cfg.weight_decay = 0.0
        if not hasattr(self.cfg, "grad_clip"):
            self.cfg.grad_clip = 0.0
        if not hasattr(self.cfg, "lambda_reg"):
            self.cfg.lambda_reg = 0.1  # new default (was 1000)
        if not hasattr(self.cfg, "lambda_cls"):
            self.cfg.lambda_cls = 10.0  # new default (was 1.0)
        if not hasattr(self.cfg, "lambda_rank"):
            self.cfg.lambda_rank = 1.0
        if not hasattr(self.cfg, "horizons"):
            self.cfg.horizons = DEFAULT_HORIZONS.copy()

    def infer_architecture(self) -> Tuple[int, int, int, int, float]:
        """
        Infer model architecture dimensions from the checkpoint state_dict.

        Returns:
            Tuple of (d_model, num_layers, nhead, dim_feedforward, dropout)
        """
        d_model = self.state_dict["input_proj.weight"].shape[0]
        dim_feedforward = self.state_dict.get("encoder.layers.0.linear1.weight", torch.empty(0)).shape[0] or 256

        # Infer layer count from encoder layer keys to stay compatible with checkpoints trained using
        # different num_layers defaults.
        layer_keys = [k for k in self.state_dict.keys() if k.startswith("encoder.layers.")]
        num_layers = 0
        for k in layer_keys:
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                num_layers = max(num_layers, int(parts[2]) + 1)
        num_layers = max(num_layers, 1)

        # Prefer config-provided nhead when available; otherwise pick a divisor of d_model.
        nhead = getattr(self.cfg, "nhead", 8)
        if nhead <= 0 or d_model % nhead != 0:
            for candidate in (8, 4, 2, 1):
                if d_model % candidate == 0:
                    nhead = candidate
                    break
        dropout = getattr(self.cfg, "dropout", 0.1)
        return d_model, num_layers, nhead, dim_feedforward, dropout

    def is_legacy_single_head(self) -> bool:
        """Returns True if this checkpoint uses the old single-head architecture."""
        return not any(k.startswith("reg_heads.") for k in self.state_dict.keys())

    def migrate_legacy_heads(self, horizons: list) -> None:
        """
        Backward-compat: if loading a legacy single-head checkpoint into the current
        multi-head model, copy the old head weights to all new horizon heads so that
        all horizons produce the same prediction as the original single-head model.
        """
        if self.is_legacy_single_head() and "head.weight" in self.state_dict:
            old_reg_weight = self.state_dict.pop("head.weight")
            old_reg_bias = self.state_dict.pop("head.bias")
            old_cls_weight = self.state_dict.pop("cls_head.weight")
            old_cls_bias = self.state_dict.pop("cls_head.bias")
            for i in range(len(horizons)):
                self.state_dict[f"reg_heads.{i}.weight"] = old_reg_weight.clone()
                self.state_dict[f"reg_heads.{i}.bias"] = old_reg_bias.clone()
                self.state_dict[f"cls_heads.{i}.weight"] = old_cls_weight.clone()
                self.state_dict[f"cls_heads.{i}.bias"] = old_cls_bias.clone()


def load_model(
    checkpoint_path: Path, device: Optional[str] = None
) -> Tuple[PriceTransformer, Dict[str, float], TrainConfig, Optional[Dict[str, np.ndarray]]]:
    target_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compatibility: checkpoints saved with old module path after directory reorganization
    import sys
    if "auto_select_stock.torch_model" not in sys.modules:
        sys.modules["auto_select_stock.torch_model"] = sys.modules["auto_select_stock.core.torch_model"]

    try:
        payload = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=target_device)
    cfg: TrainConfig = payload["config"]
    state_dict = payload["model_state"]

    # Run backward-compat migrations
    migrator = CheckpointMigrator(cfg, state_dict)
    migrator.migrate_config()

    # Reconstruct feature columns including technical features
    tech_cols = getattr(cfg, "technical_columns", None) or TECHNICAL_FEATURE_COLUMNS
    feature_columns = cfg.price_columns + cfg.financial_columns + tech_cols

    d_model, num_layers, nhead, dim_feedforward, dropout = migrator.infer_architecture()

    # Check if this is a legacy single-head checkpoint
    horizons = cfg.horizons

    model = PriceTransformer(
        input_dim=len(feature_columns),
        horizons=horizons,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    # Migrate legacy single-head weights to multi-head format
    migrator.migrate_legacy_heads(horizons)

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
        self.horizons = cfg.horizons
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Infer actual input dim from model (includes technical features if trained with them)
        actual_input_dim = model.input_proj.weight.shape[1]
        # Include technical features (v3 models have them)
        self.feature_columns = cfg.price_columns + cfg.financial_columns + TECHNICAL_FEATURE_COLUMNS
        # Trim if model was trained with fewer features (backward compat)
        if len(self.feature_columns) > actual_input_dim:
            self.feature_columns = self.feature_columns[:actual_input_dim]
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
        horizon: Optional[Union[int, str]] = None,
    ) -> Union[Dict[str, float], float]:
        """
        Run inference for a symbol.

        Args:
            symbol: stock symbol
            seq_len: sequence length (default from config)
            base_dir: data directory
            features: optional pre-loaded features
            horizon: None -> return all horizons as dict {f"{h}d": ret, ...}
                     1/3/5/7/14/20 or "1d"/"3d"/etc -> return float for that horizon

        Returns:
            Predicted return(s) as float or dict of {horizon: return}.
        """
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
            out = self.model(x)
            # Multi-head returns 4 tensors; single-head returns 2
            if len(out) == 4:
                reg_all = out[2]  # (num_hor, batch=1, seq_len)
            else:
                reg_all = out[0].unsqueeze(0)  # wrap single head as 1-horizon

        mode = getattr(self.cfg, "target_mode", "close")
        last_close = float(feats[-1, self.close_idx])

        results: Dict[str, float] = {}
        for h_idx, h in enumerate(self.horizons):
            pred = reg_all[h_idx, 0, -1].item()  # last timestep
            if mode == "log_return":
                # Predicted h-day log return -> convert to simple return
                results[f"{h}d"] = math.exp(pred) - 1.0
            else:
                # Predicts close price directly
                pred_close = pred * scaler_std[self.close_idx] + scaler_mean[self.close_idx]
                results[f"{h}d"] = float(pred_close / last_close - 1.0)

        if horizon is not None:
            # Return specific horizon as float
            h_str = str(horizon) if isinstance(horizon, int) else horizon
            if not h_str.endswith("d"):
                h_str = f"{h_str}d"
            return results.get(h_str, results.get("1d", 0.0))

        return results


def predict_next_close(
    symbol: str,
    checkpoint_path: Path,
    seq_len: Optional[int] = None,
    base_dir: Path = DATA_DIR,
    device: Optional[str] = None,
    horizon: Optional[Union[int, str]] = None,
) -> Union[Dict[str, float], float]:
    predictor = PricePredictor(checkpoint_path, device=device)
    return predictor.predict(symbol, seq_len=seq_len, base_dir=base_dir, horizon=horizon)


__all__ = ["PricePredictor", "predict_next_close", "load_model"]
