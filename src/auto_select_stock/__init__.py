"""Auto stock selector package."""

# Re-export from new locations for backward compatibility
from . import config

# data subpackage
from .data import fetcher as data_fetcher
from .data import financials as financials_fetcher
from .data import storage

# core subpackage
from .core import torch_model
from .core import types as stock_types
from .core import features

# web subpackage - re-export key components for backward compat
from .web import dashboard
from .web import html_report
from .web import ops_dashboard
from .web import ops_handlers
from .web import scoring
from .web import screener

# notify subpackage
from . import notify

# llm subpackage
from . import llm

# predict subpackage
from . import predict

__all__ = [
    "config",
    "data_fetcher",
    "financials_fetcher",
    "storage",
    "stock_types",
    "torch_model",
    "features",
    "dashboard",
    "html_report",
    "ops_dashboard",
    "ops_handlers",
    "scoring",
    "screener",
    "notify",
    "llm",
    "predict",
    "web",
]
