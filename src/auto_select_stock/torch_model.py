import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn

from .config import MODEL_DIR, PREPROCESSED_DIR

# Feature column constants (duplicated here to avoid circular import with predict.data)
_PRICE_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover_rate",
    "volume_ratio",
    "pct_change",
    "amplitude",
    "change_amount",
]
_FINANCIAL_FEATURE_COLUMNS = [
    "roe",
    "net_profit_margin",
    "gross_margin",
    "operating_cashflow_growth",
    "debt_to_asset",
    "eps",
    "operating_cashflow_per_share",
]

DEFAULT_HORIZONS = [1, 3, 5, 7, 14, 20]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def _maybe_extend(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= self.pe.size(0):
            return self.pe
        # Dynamically extend positional encoding to cover longer sequences during inference.
        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        # Update cached buffer for future calls on the same device/length
        self.pe = pe  # type: ignore[assignment]
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        pe = self._maybe_extend(x.size(0), x.device)
        x = x + pe[: x.size(0)]
        return self.dropout(x)


class PriceTransformer(nn.Module):
    """
    Causal Transformer encoder that maps 历史特征到下一时刻收盘价（自回归形式）。
    Supports multi-horizon prediction via parallel regression/classification heads.
    """

    def __init__(
        self,
        input_dim: int,
        horizons: List[int] = None,
        d_model: int = 256,  # increased from 128 for more capacity
        nhead: int = 8,
        num_layers: int = 10,  # increased from 8 for deeper reasoning
        dim_feedforward: int = 512,  # increased from 256
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizons = horizons or DEFAULT_HORIZONS
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Multi-head architecture: each head predicts one horizon
        self.reg_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in self.horizons])
        self.cls_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in self.horizons])
        # Backward-compat aliases: first head = 1d prediction
        self.head = self.reg_heads[0]
        self.cls_head = self.cls_heads[0]
        self._cached_mask: Optional[torch.Tensor] = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._cached_mask is None or self._cached_mask.size(0) != seq_len or self._cached_mask.device != device:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            self._cached_mask = mask
        return self._cached_mask

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pe(x)
        mask = self._causal_mask(seq_len=x.size(0), device=x.device)
        encoded = self.encoder(x, mask=mask)  # (seq_len, batch, d_model)

        # Multi-head outputs: stack all horizon predictions
        reg_all = torch.stack([h(encoded).squeeze(-1) for h in self.reg_heads], dim=0)  # (num_hor, seq_len, batch)
        cls_all = torch.stack([h(encoded).squeeze(-1) for h in self.cls_heads], dim=0)  # (num_hor, seq_len, batch)

        # Backward compat: first head (1d) as single outputs
        reg = reg_all[0]  # (seq_len, batch)
        cls = cls_all[0]  # (seq_len, batch)
        return reg.transpose(0, 1), cls.transpose(0, 1), reg_all.transpose(1, 2), cls_all.transpose(1, 2)
        # Returns: (batch, seq_len), (batch, seq_len), (num_hor, batch, seq_len), (num_hor, batch, seq_len)


@dataclass
class TrainConfig:
    seq_len: int = 1024
    window_stride: int = 10
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    eval_every: int = 1
    device: Optional[str] = None
    num_workers: int = 0
    train_ratio: float = 0.8
    date_windows: List[str] = field(default_factory=list)
    price_columns: List[str] = field(default_factory=lambda: _PRICE_FEATURE_COLUMNS.copy())
    financial_columns: Optional[List[str]] = None
    technical_columns: Optional[List[str]] = None  # if None, defaults to TECHNICAL_FEATURE_COLUMNS
    target_mode: str = "log_return"  # "log_return" or "close"
    lambda_reg: float = 0.1  # regression weight (small - ranking signal is secondary)
    lambda_cls: float = 10.0  # classification weight (dominant - provides directional stability)
    lambda_rank: float = 1.0  # pairwise ranking loss weight (ListMLE-style)
    save_path: Path = MODEL_DIR / "price_transformer.pt"
    experiment_name: str = "experiment"
    checkpoint_steps: int = 10000
    resume_checkpoint: Optional[Path] = None
    preprocessed_dir: Path = PREPROCESSED_DIR
    keep_preprocessed_in_memory: bool = False
    base_seed: Optional[int] = None
    exact_resume: bool = True
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_mode: Optional[str] = None  # e.g. "offline"
    profile: bool = False
    lr_warmup_steps: int = 500  # warmup steps before cosine annealing
    lr_min: float = 1e-6  # minimum LR for cosine annealing
    horizons: List[int] = field(default_factory=lambda: DEFAULT_HORIZONS.copy())
__all__ = ["PriceTransformer", "TrainConfig", "DEFAULT_HORIZONS"]
