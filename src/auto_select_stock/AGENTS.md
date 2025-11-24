# Repository Guidelines

## Project Structure & Module Organization
- Root: `README.md` (usage), `requirements.txt` (deps), `start_ops_dashboard.sh` (ops UI), `data/` `models/` `reports/` `logs/` for artifacts; keep `PYTHONPATH=./src`.
- Core package `src/auto_select_stock/`: `cli.py` (entrypoint), `config.py` (paths/envs), `data_fetcher.py` + `financials_fetcher.py` (ingestion), `storage.py` (SQLite/NumPy I/O), `scoring.py` + `llm/` (LLM adapters), `torch_model.py` + `predict/` (training, inference, backtest), `dashboard.py` + `html_report.py` (HTML outputs), `ops_dashboard.py` (control panel).

## Build, Test, and Development Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`; export `PYTHONPATH=./src:${PYTHONPATH:-}`.
- Data: `python -m auto_select_stock.cli fetch-all --start 2018-01-01 [--limit N]`; incremental: `python -m auto_select_stock.cli update-daily`.
- LLM scoring/report: `python -m auto_select_stock.cli score --provider dummy --top 20`; render: `python -m auto_select_stock.cli render --provider dummy --top 50 --output reports/undervalued.html`.
- Modeling: `python -m auto_select_stock.cli train-transformer --seq-len 60 --epochs 1 --device cpu` (smoke); predict: `python -m auto_select_stock.cli predict-transformer 600000 --checkpoint models/price_transformer.pt`.
- Ops dashboard: `./start_ops_dashboard.sh` (serves at http://127.0.0.1:8000; sets envs).

## Coding Style & Naming Conventions
- Python 3 with PEP 8 defaults; 4-space indents, snake_case for funcs/vars, lowercase module names; keep docstrings succinct.
- Use type hints on public helpers; mirror the lazy-import pattern (see `_lazy_torch_import` in `cli.py`) for heavy deps.
- Add new adapters under `llm/` following `base.py`; keep CLI flags descriptive and grouped by command.

## Testing Guidelines
- No dedicated test suite yet; run the CLI flows above with `--provider dummy` and confirm artifacts under `reports/` and `logs/` are produced without errors.
- For model changes, run a quick epoch on a small symbol subset (`--limit 30`, `--epochs 1`) to verify training loop and checkpoint writing.
- When adding data transforms, validate one symbol end-to-end (`update-daily`, `train-transformer --epochs 1`, `predict-transformer`) and include commands/output notes in the PR.

## Commit & Pull Request Guidelines
- Use short, imperative subjects with scope prefixes when helpful (e.g., `cli: add resume flag`, `predict: clamp logits`); keep one concern per commit.
- PRs should state intent, list validation commands, mention new env vars or defaults, and attach paths or screenshots for generated dashboards/reports when applicable.
- Call out breaking changes (data schema, checkpoint format, report columns) and note any manual migration steps.

## Security & Configuration Tips
- Keep secrets in env vars (`OPENAI_API_KEY`, `AUTO_SELECT_LLM_MODEL`, etc.); never commit keys. Override storage roots via `AUTO_SELECT_STOCK_DATA_DIR`, `AUTO_SELECT_STOCK_REPORT_DIR`, `AUTO_SELECT_MODEL_DIR`, `AUTO_SELECT_STOCK_PREPROCESSED_DIR`.
- Large artifacts in `data/`, `models/`, and `reports/` should stay untracked; prefer external storage and only commit small samples needed for docs or tests.
