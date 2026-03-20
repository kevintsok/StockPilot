# StockPilot

A-share automated stock screener combining LLM-based valuation scoring with a Transformer model that predicts next-day returns. Includes a web control panel and built-in backtester.

## Features

- **Data**: fetch daily prices and financials into SQLite, with cached npz preprocessed tensors
- **Model**: causal Transformer encoder with regression + classification heads; checkpoints include scaler and config for reproducible inference
- **Inference**: reusable `PricePredictor` API and CLI for single-stock predictions
- **Backtest**: long-only (top-K) or long/short (top/bottom N%) strategies with cost/slippage modeling, turnover and concentration tracking
- **Ops dashboard**: local web UI for data fetching, training, inference, and backtesting with live logs
- **Reports**: HTML leaderboard and sortable price/financial dashboard

## Quickstart

```bash
# Install
pip install -r requirements.txt

# Fetch history
python -m auto_select_stock.cli fetch-all --start 2018-01-01

# Train
python -m auto_select_stock.cli train-transformer \
  --seq-len 60 --epochs 20 --batch-size 64 --device cuda \
  --save-path models/price_transformer.pt

# Predict
python -m auto_select_stock.cli predict-transformer 600000 \
  --seq-len 60 --checkpoint models/price_transformer.pt

# Backtest (long/short)
python -m auto_select_stock.cli backtest-transformer \
  --start 2023-01-01 --end 2023-06-30 \
  --top-pct 0.1 --checkpoint models/price_transformer.pt

# Backtest (top-K long-only)
python -m auto_select_stock.cli backtest-transformer --mode topk --top-k 5 \
  --checkpoint models/price_transformer.pt

# Web UI
python -m auto_select_stock.ops_dashboard
# open http://127.0.0.1:8000
```

## Backtest Commands

| Command | Description |
|---------|-------------|
| `backtest-transformer` | Long/short portfolio from predicted returns (top/bottom N%) |
| `backtest-transformer --mode topk` | Daily batch inference, rank by predicted return, buy top-K only |
| `backtest-strategy` | Same as `--mode topk`, results saved to checkpoint directory |
| `backtest-per-symbol --workers N` | Per-stock backtest, CSV output; multi-process if workers > 1 |

## LLM Scoring

```bash
export OPENAI_API_KEY=your_key
python -m auto_select_stock.cli score --top 50 --provider openai
python -m auto_select_stock.cli render --top 50 --output reports/undervalued.html
```

## Layout

```
src/auto_select_stock/
  config.py              - paths and environment variable defaults
  storage.py             - SQLite I/O (price and financial tables)
  data_fetcher.py        - daily price data ingestion via akshare
  financials_fetcher.py   - quarterly financial report ingestion
  predict/
    data.py              - preprocessing, feature caching, datasets
    torch_model.py        - PriceTransformer architecture
    train.py              - training loop with date-window splits
    inference.py          - PricePredictor for batched inference
    backtest.py           - backtest strategies (longshort, topk)
    checkpoints.py        - checkpoint save/load utilities
    strategy.py           - portfolio construction helpers
  cli.py                 - all CLI commands
  ops_dashboard.py        - web control panel (port 8000)
  scoring.py              - LLM-based stock scoring
  llm/                    - LLM provider adapters
  html_report.py          - HTML leaderboard rendering
  dashboard.py            - sortable price/financial dashboard
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_SELECT_STOCK_DATA_DIR` | `data/` | price/financial data |
| `AUTO_SELECT_MODEL_DIR` | `models/` | checkpoint storage |
| `AUTO_SELECT_STOCK_PREPROCESSED_DIR` | `data/preprocessed/` | cached features |
| `AUTO_SELECT_LLM_PROVIDER` | `openai` | LLM provider |
| `AUTO_SELECT_LLM_MODEL` | `gpt-4o-mini` | LLM model |
| `OPENAI_API_KEY` | - | required for LLM scoring |

## Notes

- Always set `PYTHONPATH=./src` when running from the repo root
- CUDA warnings on CPU-only machines are benign; inference falls back to CPU automatically
- Date-window training splits (e.g. `--date-window 2022-01-01:2023-01-01`) prevent financial report data leakage into training
- Use `--provider dummy` with `score` to test without calling external APIs
