"""
Shared pytest fixtures for StockPilot tests.

Provides:
- temp_db: isolated SQLite database for storage tests
- sample_price_data: deterministic OHLCV data for testing
- mock_predictor: mock PricePredictor for backtest tests
- sample_signals: list of Signal objects for strategy tests
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from auto_select_stock.predict.strategies.base import Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with test schema."""
    db_path = tmp_path / "stock.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE price (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL, amount REAL, turnover_rate REAL, volume_ratio REAL,
            pct_change REAL, amplitude REAL, change_amount REAL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE financial (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            roe REAL, net_profit_margin REAL, gross_margin REAL,
            operating_cashflow_growth REAL, debt_to_asset REAL,
            eps REAL, operating_cashflow_per_share REAL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    conn.commit()
    conn.close()
    return tmp_path


@pytest.fixture
def sample_price_data() -> List[Dict]:
    """Generate deterministic OHLCV price data for a single symbol.

    Returns 60 days of trending price data where close goes from 10.0 to 12.0.
    """
    np.random.seed(42)
    data = []
    close = 10.0
    for i in range(60):
        day = 30 + i  # start from day 30
        date = f"2024-01-{day:02d}" if day <= 31 else f"2024-02-{(day-31):02d}"
        close += np.random.randn() * 0.05 + 0.02
        open_price = close + np.random.randn() * 0.02
        high = max(close, open_price) + abs(np.random.randn() * 0.03)
        low = min(close, open_price) - abs(np.random.randn() * 0.03)
        volume = 1_000_000 + np.random.randint(-100_000, 100_000)
        amount = volume * close
        turnover_rate = 1.5 + np.random.randn() * 0.3
        volume_ratio = 1.0 + np.random.randn() * 0.2
        pct_change = (close / (close - (close - open_price)) - 1) * 100 if close != open_price else 0.0
        amplitude = ((high - low) / low) * 100
        change_amount = close - open_price
        data.append({
            "date": date,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": int(volume),
            "amount": round(amount, 2),
            "turnover_rate": round(turnover_rate, 4),
            "volume_ratio": round(volume_ratio, 4),
            "pct_change": round(pct_change, 4),
            "amplitude": round(amplitude, 4),
            "change_amount": round(change_amount, 4),
        })
    return data


@pytest.fixture
def sample_price_df(sample_price_data: List[Dict]) -> pd.DataFrame:
    """Return sample price data as a pandas DataFrame."""
    return pd.DataFrame(sample_price_data)


@pytest.fixture
def mock_predictor(monkeypatch) -> "MockPredictor":
    """Return a mock PricePredictor that returns predictable values."""
    return MockPredictor()


class MockPredictor:
    """A mock PricePredictor for testing without loading real models."""

    def __init__(self):
        self.cfg = MockConfig()
        self.scaler = {
            "mean": np.zeros(30, dtype="float32"),
            "std": np.ones(30, dtype="float32"),
        }
        self.horizons = [1, 3, 5, 7, 14, 20]
        self.feature_columns = (
            ["open", "high", "low", "close", "volume", "amount",
             "turnover_rate", "volume_ratio", "pct_change", "amplitude", "change_amount"]
            + ["roe", "net_profit_margin", "gross_margin",
               "operating_cashflow_growth", "debt_to_asset", "eps", "operating_cashflow_per_share"]
            + ["rsi_14", "macd_line", "macd_signal", "macd_hist",
               "bb_position", "bb_width", "volume_ma5", "volume_ma20",
               "atr_14", "stoch_k", "stoch_d", "obv_ma10", "roc_10", "momentum_10"]
        )
        self.close_idx = 3  # close is at index 3 in price columns
        self.device = "cpu"
        self.model = MockModel()

    def predict(self, symbol: str, seq_len: Optional[int] = None,
                base_dir: Optional[Path] = None,
                features: Optional[np.ndarray] = None,
                horizon: Optional[str] = None) -> Dict[str, float]:
        """Return deterministic mock predictions."""
        results = {
            "1d": 0.02,
            "3d": 0.06,
            "5d": 0.10,
            "7d": 0.13,
            "14d": 0.20,
            "20d": 0.25,
        }
        if horizon is not None:
            h_str = str(horizon) if isinstance(horizon, int) else horizon
            if not h_str.endswith("d"):
                h_str = f"{h_str}d"
            return results.get(h_str, results["1d"])
        return results


class MockModel:
    """A mock torch model for testing."""

    def __init__(self):
        self.input_proj = MockLinear()

    def __call__(self, x):
        # Return 4 tensors like multi-head model: (cls_logits, reg_all, hidden, attn_weights)
        batch, seq_len, features = x.shape
        num_horizons = 6
        reg_all = np.random.randn(num_horizons, batch, seq_len).astype("float32") * 0.01
        return (
            np.random.randn(batch, seq_len, 2).astype("float32"),
            reg_all,
            np.random.randn(batch, seq_len, 64).astype("float32"),
            np.random.randn(batch, 8, seq_len, seq_len).astype("float32"),
        )


class MockLinear:
    def __init__(self):
        self.weight = np.random.randn(64, 30).astype("float32")
        self.bias = np.random.randn(64).astype("float32")


class MockConfig:
    """Mock TrainConfig for PricePredictor."""
    seq_len = 60
    price_columns = ["open", "high", "low", "close", "volume", "amount",
                     "turnover_rate", "volume_ratio", "pct_change", "amplitude", "change_amount"]
    financial_columns = ["roe", "net_profit_margin", "gross_margin",
                        "operating_cashflow_growth", "debt_to_asset",
                        "eps", "operating_cashflow_per_share"]
    target_mode = "close"
    weight_decay = 0.0
    grad_clip = 0.0
    lambda_reg = 0.1
    lambda_cls = 10.0
    lambda_rank = 1.0
    horizons = [1, 3, 5, 7, 14, 20]
    dropout = 0.1


@pytest.fixture
def sample_signals() -> List[Signal]:
    """Create a list of 10 sample Signal objects with deterministic values."""
    return [
        Signal(symbol=f"00{i:04d}", predicted_ret=0.01 * i, realized_ret=0.005 * i,
               industry="电子" if i % 2 == 0 else "医药", predicted_rets=None,
               entry_price=10.0 + i, auc_limit=0)
        for i in range(1, 11)
    ]


@pytest.fixture
def sample_signals_with_multi_horizon() -> List[Signal]:
    """Create signals with multi-horizon predicted returns."""
    return [
        Signal(
            symbol=f"00{i:04d}",
            predicted_ret=0.01 * i,
            realized_ret=0.005 * i,
            industry="电子" if i % 2 == 0 else "医药",
            predicted_rets={
                "1d": 0.01 * i,
                "3d": 0.03 * i,
                "5d": 0.05 * i,
                "7d": 0.07 * i,
                "14d": 0.10 * i,
                "20d": 0.12 * i,
            },
            entry_price=10.0 + i,
            auc_limit=0,
        )
        for i in range(1, 11)
    ]
