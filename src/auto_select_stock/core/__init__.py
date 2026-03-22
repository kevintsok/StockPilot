"""
Core module - contains fundamental types and the Transformer model.
"""
from .types import (
    NUMERIC_FIELDS,
    StockDailyRow,
    StockMeta,
    StockScore,
    StockSnapshot,
    to_structured_array,
)
from .features import (
    DEFAULT_FEATURE_COLUMNS,
    FINANCIAL_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    TECHNICAL_FEATURE_COLUMNS,
)
from .torch_model import DEFAULT_HORIZONS, PriceTransformer, TrainConfig

__all__ = [
    # types
    "NUMERIC_FIELDS",
    "StockDailyRow",
    "StockMeta",
    "StockScore",
    "StockSnapshot",
    "to_structured_array",
    # features
    "DEFAULT_FEATURE_COLUMNS",
    "FINANCIAL_FEATURE_COLUMNS",
    "PRICE_FEATURE_COLUMNS",
    "TECHNICAL_FEATURE_COLUMNS",
    # torch_model
    "DEFAULT_HORIZONS",
    "PriceTransformer",
    "TrainConfig",
]
