# StockPilot

> A-share automated stock screener — LLM valuation scoring + Transformer price prediction + multi-strategy backtester.

Predict next-day returns, rank stocks, and compare 10+ trading strategies in one command. Built for Chinese A-shares (沪深/创业板).

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)

## What it does

1. **Fetch** daily price + quarterly financials into SQLite
2. **Train** a Transformer model to predict next-day returns
3. **Backtest** 10+ trading strategies (shared signal, single GPU pass)
4. **Score** stocks with LLM for qualitative valuation
5. **Dashboard** — local web UI for everything

---

## QuickStart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Fetch 5 years of A-share history
python -m auto_select_stock.cli fetch-all --start 2018-01-01 --limit 100

# 3. Train (date-window split prevents financial data leakage)
python -m auto_select_stock.cli train-transformer \
  --seq-len 60 --epochs 20 --batch-size 64 --device cuda \
  --date-window 2022-01-01:2023-01-01 \
  --save-path models/price_transformer.pt

# 4. Backtest all 10 strategies at once (shared signal = 1 GPU pass)
python -m auto_select_stock.cli backtest-strategies \
  --start 2023-01-01 --end 2024-12-31 \
  --checkpoint models/price_transformer.pt \
  --cost-bps 15 --slippage-bps 10

# 5. Web dashboard
python -m auto_select_stock.ops_dashboard
# open http://127.0.0.1:8000
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        StockPilot Pipeline                       │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
  │  Data Fetch  │────▶│  Preprocessing  │────▶│  SQLite (.db)  │
  │  (akshare)   │     │  npz cache      │     │  price/fin     │
  └──────────────┘     └─────────────────┘     └───────┬────────┘
                                                       │
                       ┌────────────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                    Transformer Training                           │
  │  PriceTransformer: causal encoder + regression/classification heads│
  │  Date-window splits ──▶ train/val/test (no future data leakage)  │
  └────────────────────────────────────┬───────────────────────────┘
                                       │ checkpoint
                                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │              Multi-Strategy Backtest (shared signals)             │
  │                                                                  │
  │  _collect_signals_batched() ──▶ 1 GPU pass for all stocks/dates │
  │                    │                                             │
  │         ┌──────────┴──────────┬──────────┬──────────┐            │
  │         ▼                     ▼          ▼          ▼            │
  │   TopK-Proportional   Momentum   Risk-Parity  Sector-Neutral ... │
  │   (select_positions() per strategy, independent weights/cache)     │
  └──────────────────────────────────────────────────────────────────┘
```

**Key design: one model → one signal collection → all strategies compare fairly.**

---

## Backtest Results (2023-01-01 ~ 2024-12-31)

| Strategy | Total Ret (Net) | Sharpe (Net) | Max DD | Ann Ret (Net) | Avg Turnover |
|----------|----------------:|-------------:|-------:|--------------:|-------------:|
| **Confidence-Sized** | **+71.1%** | **1.08** | -27.6% | +32.3% | 65.0% |
| TopK-Proportional | +63.3% | 0.98 | -27.6% | +29.1% | 64.8% |
| Momentum-Filter | +47.2% | 0.76 | -30.7% | +22.3% | 67.6% |
| TopK-Threshold | +45.7% | 0.70 | -27.6% | +21.6% | 63.6% |
| Dual-Threshold | +31.3% | 0.71 | -22.5% | +15.2% | 28.1% |
| BottomK-Reversal | +18.9% | 0.50 | -20.5% | +9.4% | 50.2% |
| Risk-Parity | -8.3% | -0.18 | -41.6% | -4.4% | 64.8% |
| LongShort-Equal | -12.9% | -0.75 | -17.2% | -6.9% | 33.6% |
| Sector-Neutral | -12.9% | -0.75 | -17.2% | -6.9% | 33.6% |
| TopK-StopLoss | -100.0% | nan | -108.9% | nan | 1355.3% |

> Model: price_transformer (train: 2018-2021, val: 2022). Costs: 15bp commission + 10bp slippage.
> **TopK-StopLoss fails due to A-share T+1 constraint** — trailing stops need next-day execution.

**Top insight**: Confidence-weighted sizing (weight by |pred_ret|) beats naive equal-weight TopK. Momentum filter adds no benefit in this period. Long/short strategies underperform due to limited true shorting in A-shares.

---

## Available Strategies

| Type | Description |
|------|-------------|
| `topk` | Buy top-K proportional to predicted return |
| `threshold` | Buy top-K only when pred > threshold |
| `long_short` | Long top-N%, short bottom-N%, equal weight |
| `momentum_filter` | TopK only when pred > short-term MA of predictions |
| `risk_parity` | TopK weighted inversely by realized volatility |
| `mean_reversion` | Long losers (bottom-K), short winners (top-K) |
| `confidence` | Weight by \|pred_ret\|, long only |
| `sector_neutral` | Long/short equal weight, net-zero sector exposure |
| `trailing_stop` | TopK with stop-loss on realized loss |
| `dual_thresh` | Long when pred > upper, short when pred < lower |

Add new strategies as JSON files in `strategies/configs/`.

---

## All Commands

```bash
# Data
python -m auto_select_stock.cli fetch-all --start 2018-01-01 [--limit N]
python -m auto_select_stock.cli update-daily [symbols...]
python -m auto_select_stock.cli fetch-financials [--limit N]

# Training
python -m auto_select_stock.cli train-transformer \
  --seq-len 60 --epochs 20 --batch-size 64 --device cuda \
  --save-path models/price_transformer.pt \
  [--date-window 2022-01-01:2023-01-01]

# Inference
python -m auto_select_stock.cli predict-transformer 600000 \
  --checkpoint models/price_transformer.pt

# Backtest
python -m auto_select_stock.cli backtest-transformer --mode topk --top-k 5 ...
python -m auto_select_stock.cli backtest-per-symbol --workers 4 ...

# Multi-strategy (all 10 at once)
python -m auto_select_stock.cli backtest-strategies --list
python -m auto_select_stock.cli backtest-strategies \
  --start 2023-01-01 --end 2024-12-31 \
  --checkpoint models/price_transformer.pt \
  --cost-bps 15 --slippage-bps 10

# LLM Scoring
export OPENAI_API_KEY=your_key
python -m auto_select_stock.cli score --top 50 --provider openai
python -m auto_select_stock.cli render --top 50 --output reports/undervalued.html

# Web UI
python -m auto_select_stock.ops_dashboard
```

---

## Project Layout

```
src/auto_select_stock/
├── cli.py                 # All CLI commands
├── storage.py             # SQLite I/O (price & financial tables)
├── data_fetcher.py        # Daily price ingestion via akshare
├── financials_fetcher.py   # Quarterly report ingestion
├── scoring.py             # LLM-based stock scoring
├── ops_dashboard.py       # Web control panel (port 8000)
│
└── predict/
    ├── data.py            # Feature engineering, npz caching
    ├── torch_model.py     # PriceTransformer architecture
    ├── train.py           # Training loop with date-window splits
    ├── inference.py       # PricePredictor (batched, reusable)
    ├── backtest.py        # BacktestConfig, run_backtest, _collect_signals_batched
    ├── strategy.py        # build_long_short_portfolio helper
    ├── checkpoints.py     # Checkpoint save/load
    └── strategies/        # v0.0.2: JSON-driven strategy system
        ├── base.py        # Signal dataclass, BaseStrategy ABC
        ├── __init__.py    # 10 strategy implementations
        ├── registry.py    # StrategyRegistry (loads JSON configs)
        ├── runner.py     # run_all_strategies_shared (shared signal collection)
        └── configs/
            └── default_strategies.json  # 10 pre-defined strategies
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_SELECT_STOCK_DATA_DIR` | `data/` | Price/financial data |
| `AUTO_SELECT_MODEL_DIR` | `models/` | Checkpoint storage |
| `AUTO_SELECT_STOCK_PREPROCESSED_DIR` | `data/preprocessed/` | Cached features |
| `AUTO_SELECT_LLM_PROVIDER` | `openai` | LLM provider |
| `AUTO_SELECT_LLM_MODEL` | `gpt-4o-mini` | LLM model |
| `OPENAI_API_KEY` | — | Required for LLM scoring |

**Always set `PYTHONPATH=./src`** when running from repo root.

---

## Notes

- Date-window training splits (`--date-window 2022-01-01:2023-01-01`) prevent financial report data leakage into training
- A-share T+1 trading rule: strategies requiring same-day buy/sell (e.g. trailing stops) will fail
- CUDA warnings on CPU-only machines are benign; inference falls back to CPU automatically
- Use `--provider dummy` with `score` to test without calling external APIs
