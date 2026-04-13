# Debug Scripts

This directory contains standalone debugging scripts used during development.
These are not part of the test suite and should not be imported by production code.

## Purpose of Each Script

| Script | Purpose |
|--------|---------|
| `debug_backtest_day.py` | Traces through one day of backtest to understand the return calculation. Inspects model predictions and compares predicted vs realized returns for test symbols. |
| `debug_backtest_trace.py` | Checks model output structure and scaler configuration. Verifies that the PriceTransformer multi-head model returns the expected output format (4 tensors). |
| `debug_model.py` | Loads the actual model checkpoint and inspects its configuration, horizons, scaler values, and feature column indices. |
| `debug_scaler.py` | Validates scaler computation: loads features, normalizes them, runs model inference, and checks denormalization to verify scaler correctness. |
| `debug_topk_day1.py` | Simulates the TopK strategy over 6 trading days with a small symbol subset (50 stocks). Tracks portfolio value, cash, holdings, and daily returns. |
| `test_runner_fix.py` | Tests the strategy runner with 200 symbols over 10 days. Used to verify fix for `_LOT_SIZE` constant in the runner module. |
| `test_signal_fix.py` | Verifies signal collection for symbol 600000 over 5 days. Checks that signals include entry_price and auc_limit fields correctly. |
| `check_axis.py` | Analyzes backtest JSON output to check axis scaling in the dashboard visualization. Compares index returns vs strategy capital returns. |

## Usage

All debug scripts assume the working directory is the project root and the `src` subdirectory is on the Python path:

```bash
cd /path/to/auto-select-stock
PYTHONPATH=./src python debug_*.py
```

## Cleanup Recommendation

These scripts are candidates to be moved to `scripts/debug/` in a future refactoring pass to keep the project root cleaner.
