"""
Canonical feature column definitions shared across modules.
Kept here to avoid circular imports between core/torch_model.py and predict/data.py.
"""

# Price feature columns (12 columns)
PRICE_FEATURE_COLUMNS = [
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

# Financial feature columns (forward-filled onto trading days)
FINANCIAL_FEATURE_COLUMNS = [
    "roe",
    "net_profit_margin",
    "gross_margin",
    "operating_cashflow_growth",
    "debt_to_asset",
    "eps",
    "operating_cashflow_per_share",
]

DEFAULT_FEATURE_COLUMNS = PRICE_FEATURE_COLUMNS + FINANCIAL_FEATURE_COLUMNS

# Technical indicator feature columns - computed from price data (no lookahead)
TECHNICAL_FEATURE_COLUMNS = [
    "rsi_14",  # Relative Strength Index (14-day)
    "macd_line",  # MACD line (EMA12 - EMA26)
    "macd_signal",  # MACD signal line (9-day EMA of MACD)
    "macd_hist",  # MACD histogram (MACD - Signal)
    "bb_position",  # Bollinger Band position (0-1, price's position in bands)
    "bb_width",  # Bollinger Band width (normalized)
    "volume_ma5",  # Volume MA5 ratio
    "volume_ma20",  # Volume MA20 ratio
    "atr_14",  # Average True Range (14-day)
    "stoch_k",  # Stochastic %K
    "stoch_d",  # Stochastic %D
    "obv_ma10",  # OBV MA10 ratio
    "roc_10",  # Rate of change (10-day)
    "momentum_10",  # Momentum (10-day)
]
