import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn

from .config import MODEL_DIR, PREPROCESSED_DIR
from .predict.data import FINANCIAL_FEATURE_COLUMNS, PRICE_FEATURE_COLUMNS


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
    Causal Transformer encoder that maps历史特征到下一时刻收盘价（自回归形式）。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
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
        self.head = nn.Linear(d_model, 1)
        self._cached_mask: Optional[torch.Tensor] = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._cached_mask is None or self._cached_mask.size(0) != seq_len or self._cached_mask.device != device:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            self._cached_mask = mask
        return self._cached_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pe(x)
        mask = self._causal_mask(seq_len=x.size(0), device=x.device)
        encoded = self.encoder(x, mask=mask)  # (seq_len, batch, d_model)
        preds = self.head(encoded).squeeze(-1)  # (seq_len, batch)
        return preds.transpose(0, 1)  # (batch, seq_len)


@dataclass
class TrainConfig:
    seq_len: int = 1024
    window_stride: int = 10
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    eval_every: int = 1
    device: Optional[str] = None
    num_workers: int = 0
    train_ratio: float = 0.8
    price_columns: List[str] = field(default_factory=lambda: PRICE_FEATURE_COLUMNS.copy())
    financial_columns: Optional[List[str]] = None
    save_path: Path = MODEL_DIR / "price_transformer.pt"
    experiment_name: str = "experiment"
    checkpoint_steps: int = 10000
    resume_checkpoint: Optional[Path] = None
    preprocessed_dir: Path = PREPROCESSED_DIR
    keep_preprocessed_in_memory: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_mode: Optional[str] = None  # e.g. "offline"
__all__ = ["PriceTransformer", "TrainConfig"]
