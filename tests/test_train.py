"""
Unit tests for the training module (predict/train.py).

Tests training loop basic sanity with minimal CPU-only configuration.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from auto_select_stock.predict.train import (
    TrainConfig,
    _compute_ranking_loss,
    _parse_date_window_strings,
    collate_multi_horizon,
)


class TestComputeRankingLoss:
    """Tests for _compute_ranking_loss()."""

    def test_empty_batch_returns_zero(self):
        """Returns zero loss for batch size < 2."""
        pred = torch.randn(1, 10)
        y = torch.randn(1, 10)
        loss = _compute_ranking_loss(pred, y)
        assert loss.item() == 0.0

    def test_same_predictions_returns_zero(self):
        """Identical predictions produce zero ranking loss."""
        pred = torch.zeros(4, 10)
        y = torch.tensor([[0.0, 0.1, 0.2, 0.3]] * 4)  # 4 samples, increasing returns
        loss = _compute_ranking_loss(pred, y)
        assert loss.item() == 0.0

    def test_correct_ranking_produces_zero_loss(self):
        """When higher predicted beats lower predicted, loss is zero."""
        # pred[i] > pred[j] when y[i] > y[j]
        pred = torch.tensor([[0.4], [0.3], [0.2], [0.1]], dtype=torch.float32)
        y = torch.tensor([[0.4], [0.3], [0.2], [0.1]], dtype=torch.float32)
        loss = _compute_ranking_loss(pred, y)
        assert loss.item() == 0.0

    def test_incorrect_ranking_produces_positive_loss(self):
        """When lower actual beats higher actual but prediction says otherwise, loss > 0."""
        # i=0 has highest actual return but lowest prediction -> should be penalized
        pred = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32)  # lowest to highest
        y = torch.tensor([[0.4], [0.3], [0.2], [0.1]], dtype=torch.float32)  # highest to lowest
        loss = _compute_ranking_loss(pred, y)
        assert loss.item() > 0.0


class TestParseDateWindowStrings:
    """Tests for _parse_date_window_strings()."""

    def test_parses_single_window(self):
        """Parses a single TRAIN_END:VAL_END window."""
        result = _parse_date_window_strings(["2023-01-01:2023-06-01"])
        assert len(result) == 1
        train_end, val_end = result[0]
        assert train_end == pd.Timestamp("2023-01-01")
        assert val_end == pd.Timestamp("2023-06-01")

    def test_parses_multiple_windows(self):
        """Parses multiple comma-separated windows."""
        result = _parse_date_window_strings(["2022-01-01:2022-06-01", "2023-01-01:2023-06-01"])
        assert len(result) == 2

    def test_accepts_comma_separator(self):
        """Accepts comma as separator between dates."""
        result = _parse_date_window_strings(["2023-01-01,2023-06-01"])
        assert len(result) == 1

    def test_raises_on_invalid_format(self):
        """Raises ValueError when format is invalid."""
        with pytest.raises(ValueError, match="Invalid date window"):
            _parse_date_window_strings(["invalid"])

    def test_raises_when_val_before_train(self):
        """Raises ValueError when val_end <= train_end."""
        with pytest.raises(ValueError, match="val_end must be after train_end"):
            _parse_date_window_strings(["2023-06-01:2023-01-01"])


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_default_values(self):
        """TrainConfig has sensible defaults."""
        cfg = TrainConfig()
        assert cfg.seq_len == 1024
        assert cfg.batch_size == 16
        assert cfg.epochs == 20
        assert cfg.lr == 1e-3
        assert cfg.horizons == [1, 3, 5, 7, 14, 20]

    def test_custom_values(self):
        """TrainConfig accepts custom values."""
        cfg = TrainConfig(seq_len=60, batch_size=32, epochs=5)
        assert cfg.seq_len == 60
        assert cfg.batch_size == 32
        assert cfg.epochs == 5

    def test_date_windows_default_empty(self):
        """date_windows defaults to empty list."""
        cfg = TrainConfig()
        assert cfg.date_windows == []


class TestCollateMultiHorizon:
    """Tests for collate_multi_horizon()."""

    def test_collate_pads_to_max_length(self):
        """Pads horizon tensors to the longest sequence in batch."""
        # Batch of 2 samples - x must be same size (function stacks, doesn't pad x)
        batch = [
            (
                torch.randn(10, 30),  # x: (seq_len, features) - same size
                {1: np.array([1.0, 2.0]), 3: np.array([1.0])},  # y_reg_dict with different lengths
                torch.tensor([1, 0]),  # y_cls
            ),
            (
                torch.randn(10, 30),  # same seq_len as first sample
                {1: np.array([1.5, 2.5, 3.5]), 3: np.array([1.5, 2.5])},
                torch.tensor([0, 1]),
            ),
        ]
        x, y_reg, y_cls = collate_multi_horizon(batch)
        assert x.shape[0] == 2  # batch size
        # 1d horizon: max len is 3
        assert y_reg[1].shape == (2, 3)
        # 3d horizon: max len is 2
        assert y_reg[3].shape == (2, 2)

    def test_collate_returns_tensors(self):
        """Returns proper torch tensors."""
        batch = [
            (
                torch.randn(10, 30),
                {1: np.array([1.0, 2.0])},
                torch.tensor([1]),
            ),
            (
                torch.randn(10, 30),  # same seq_len
                {1: np.array([1.5, 2.5, 3.5])},
                torch.tensor([0]),
            ),
        ]
        x, y_reg, y_cls = collate_multi_horizon(batch)
        assert isinstance(x, torch.Tensor)
        assert isinstance(y_reg[1], torch.Tensor)
        assert isinstance(y_cls, torch.Tensor)
