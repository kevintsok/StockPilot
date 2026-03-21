# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**StockPilot** - A-share automated stock screener using LLM-based scoring and a Transformer that predicts next-day close. Includes a web control panel and built-in long/short backtester.

## Setup

```bash
PYTHONPATH=./src python -m auto_select_stock.cli <command>
# or from repo root with venv activated:
python -m auto_select_stock.cli <command>
```

**Environment Variables** (override defaults):
- `AUTO_SELECT_STOCK_DATA_DIR` - price/financial data directory (default: `data/`)
- `AUTO_SELECT_MODEL_DIR` - model checkpoints (default: `models/`)
- `AUTO_SELECT_STOCK_PREPROCESSED_DIR` - cached preprocessed features (default: `data/preprocessed/`)
- `AUTO_SELECT_LLM_PROVIDER` - LLM provider (default: `openai`)
- `AUTO_SELECT_LLM_MODEL` - LLM model (default: `gpt-4o-mini`)
- `OPENAI_API_KEY` - required for LLM scoring

## Common Commands

### Data
```bash
python -m auto_select_stock.cli fetch-all --start 2018-01-01 [--limit N]  # initial history
python -m auto_select_stock.cli update-daily [symbols...]  # incremental update
python -m auto_select_stock.cli fetch-financials [--limit N]  # quarterly reports
```

### Training
```bash
python -m auto_select_stock.cli train-transformer \
  --seq-len 60 --epochs 20 --batch-size 64 --device cuda \
  --save-path models/price_transformer.pt \
  [--date-window 2022-01-01:2023-01-01]  # optional date-based splits
```

### Inference & Backtesting
```bash
python -m auto_select_stock.cli predict-transformer 600000 --checkpoint models/price_transformer.pt

# Long/short (top/bottom 10%)
python -m auto_select_stock.cli backtest-transformer \
  --start 2023-01-01 --end 2023-06-30 \
  --top-pct 0.1 --checkpoint models/price_transformer.pt \
  --cost-bps 0 --slippage-bps 0

# Top-K long-only strategy
python -m auto_select_stock.cli backtest-transformer --mode topk --top-k 5 ...
python -m auto_select_stock.cli backtest-per-symbol --workers 4 ...  # per-stock analysis
```

### Reports
```bash
python -m auto_select_stock.cli score --top 50 --provider openai
python -m auto_select_stock.cli render --top 50 --output reports/undervalued.html
python -m auto_select_stock.cli render-dashboard [--lookback-short 20 --lookback-long 60]
```

### Web UI
```bash
python -m auto_select_stock.ops_dashboard  # http://127.0.0.1:8000
```

## Architecture

### Data Pipeline
1. **SQLite** (`storage.py`) - `data/stock.db` with `price` and `financial` tables; price data fetched via `data_fetcher.py` (akshare); financial data via `financials_fetcher.py`
2. **Preprocessing** (`predict/data.py`) - `preprocess_symbol_features()` merges price + financial features; cached as `.npz` in `data/preprocessed/` with version-based invalidation
3. **Features**: 12 price columns (open/high/low/close/volume/amount/turnover_rate/volume_ratio/pct_change/amplitude/change_amount) + 7 financial columns (roe/net_profit_margin/gross_margin/operating_cashflow_growth/debt_to_asset/eps/operating_cashflow_per_share)

### Model
- **PriceTransformer** (`torch_model.py`) - Causal Transformer encoder with dual heads:
  - Regression head: predicts next-day close (or log return depending on `target_mode`)
  - Classification head: predicts up/down direction
  - Positional encoding extends dynamically for inference on longer sequences

### Training (`predict/train.py`)
- `train_from_symbols()` - main entry; supports date-window based train/val/test splits (prevents future data leakage)
- Streaming scaler computation to avoid memory blowup
- Checkpoints (`predict/checkpoints.py`) store: model state, scaler, TrainConfig, optimizer state, metrics
- Supports wandb logging, resume from checkpoint, gradient clipping

### Inference (`predict/inference.py`)
- `PricePredictor` class - loads checkpoint once, reusable for multiple symbols
- `predict_next_close()` - convenience function for single predictions

### Backtesting (`predict/backtest.py`)
- `run_backtest()` - long/short portfolio based on predicted returns (top/bottom N%)
- `run_topk_strategy()` - daily batch inference, rank by predicted return, buy top-K only
- Both support cost/slippage modeling and turnover tracking

### Multi-Strategy Backtest (`predict/strategies/`)
New in v0.0.2: JSON-driven strategy system with shared signal collection.
- `base.py` - `Signal` dataclass + `BaseStrategy` ABC
- `__init__.py` - 10 strategy implementations (TopK, Threshold, LongShort, MomentumFilter, RiskParity, MeanReversion, Confidence, SectorNeutral, TrailingStop, DualThresh)
- `registry.py` - `StrategyRegistry` loads and validates JSON configs
- `runner.py` - `run_all_strategies_shared()`: **one GPU pass for all stocks/dates, all strategies share the same signals**
- `configs/default_strategies.json` - 10 pre-defined strategy configs
- `run_all_strategies_shared()`: calls `_collect_signals_batched()` once, then iterates each strategy's `select_positions()` independently
- CLI: `backtest-strategies --list` to preview; `backtest-strategies --start ... --end ...` to run all

### LLM Scoring (`scoring.py`, `llm/`)
- `score_symbols()` - calls LLM to assess undervalued stocks
- `llm/base.py` defines interface; `llm/openai_client.py` and `llm/dummy.py` are implementations

## Key Design Notes

- **Date-window splits** are the preferred training approach - they prevent financial reports from leaking future information into training
- **Lazy torch imports** in CLI (`_lazy_torch_import()`) allow running data commands without GPU/torch installed
- **StreamingPriceDataset** avoids materializing all sequences in memory - loads features on demand with LRU cache
- Checkpoint format is forward-compatible: missing config attributes get sensible defaults during `load_model()`
- Backtest batch inference (`_collect_signals_batched`) groups windows across symbols for better GPU utilization
- **Shared signal collection**: `run_all_strategies_shared()` collects signals once (one GPU pass), then runs all strategies independently — ~10x faster than running each strategy separately
