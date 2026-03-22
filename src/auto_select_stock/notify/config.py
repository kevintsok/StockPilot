"""
Configuration for the daily notification service.

Environment variables:
    PUSHPLUS_TOKEN: PushPlus token (required)
    NOTIFY_CHECKPOINT: model checkpoint path
    NOTIFY_STRATEGY: strategy name (default: StopLoss-8pct-5d)
    NOTIFY_TOP_K: number of stocks to push (default: 10)
    NOTIFY_HORIZON: prediction horizon for ranking (default: 5d)
"""

import os

# PushPlus token (required)
PUSHPLUS_TOKEN = os.getenv("PUSHPLUS_TOKEN", "")

# Model checkpoint
DEFAULT_CHECKPOINT = "models/price_transformer-train20220101-val20230101.pt"
NOTIFY_CHECKPOINT = os.getenv("NOTIFY_CHECKPOINT", DEFAULT_CHECKPOINT)

# Strategy (top backtest performer: RiskParity-VL10-1d tag=469da)
NOTIFY_STRATEGY = os.getenv("NOTIFY_STRATEGY", "RiskParity-VL10-1d")

# Number of stocks to push
NOTIFY_TOP_K = int(os.getenv("NOTIFY_TOP_K", "10"))

# Prediction horizon for ranking (default: 5d)
NOTIFY_HORIZON = os.getenv("NOTIFY_HORIZON", "5d")
