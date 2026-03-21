"""
Configuration for the daily notification service.

Environment variables:
    PUSHPLUS_TOKEN: PushPlus token (required)
    NOTIFY_CHECKPOINT: model checkpoint path
    NOTIFY_STRATEGY: strategy name (default: confidence)
    NOTIFY_TOP_K: number of stocks to push (default: 10)
"""

import os
from pathlib import Path

# PushPlus token (required)
PUSHPLUS_TOKEN = os.getenv("PUSHPLUS_TOKEN", "")

# Model checkpoint
DEFAULT_CHECKPOINT = "models/price_transformer-train20220101-val20230101.pt"
NOTIFY_CHECKPOINT = os.getenv("NOTIFY_CHECKPOINT", DEFAULT_CHECKPOINT)

# Strategy
NOTIFY_STRATEGY = os.getenv("NOTIFY_STRATEGY", "confidence")

# Number of stocks to push
NOTIFY_TOP_K = int(os.getenv("NOTIFY_TOP_K", "10"))
