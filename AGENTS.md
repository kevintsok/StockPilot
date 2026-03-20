# Repository Guidelines

## Project Structure & Module Organization
- Root: `README.md` (usage), `requirements.txt` (deps), `start_ops_dashboard.sh` (ops launcher); artifacts live in `data/`, `models/`, `reports/`, `logs/`; export `PYTHONPATH=./src`.
- Package `src/auto_select_stock/`: `cli.py` (commands), `data_fetcher.py` + `financials_fetcher.py` (ingestion), `storage.py` (SQLite/NumPy I/O), `predict/` (train/infer/backtest), `scoring.py` + `llm/` (LLM adapters), `html_report.py` + `dashboard.py` (outputs), `ops_dashboard.py` (control panel).
- Generated datasets/checkpoints/reports stay out of git; commit only small samples that prove behavior.

## Build, Test, and Development Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`; keep `PYTHONPATH=./src:${PYTHONPATH:-}`.
- Data: `python -m auto_select_stock.cli fetch-all --start 2018-01-01 [--limit N]` for history; `python -m auto_select_stock.cli update-daily` for refreshes.
- Modeling: smoke train `python -m auto_select_stock.cli train-transformer --seq-len 60 --epochs 1 --device cpu`; predict `python -m auto_select_stock.cli predict-transformer 600000 --checkpoint models/price_transformer.pt`; backtest `python -m auto_select_stock.cli backtest-transformer ... --top-pct 0.1`.
- Ops dashboard: `./start_ops_dashboard.sh` or `python -m auto_select_stock.ops_dashboard` (serves http://127.0.0.1:8000).

## Coding Style & Naming Conventions
- Python 3, PEP 8, 4-space indents, snake_case, lowercase modules; keep docstrings concise on intent and edge cases.
- Use type hints on public helpers and CLI handlers; follow lazy-import patterns (see `_lazy_torch_import` in `cli.py`) for heavy deps.
- New LLM providers belong in `llm/` extending `base.py`; group CLI flags by command and favor descriptive names.

## Testing Guidelines
- No formal test suite; rely on CLI smoke runs. Use `--provider dummy` to avoid external calls and verify reports land in `reports/` without errors.
- For model changes, run a short train (`--limit 30 --epochs 1`) and ensure checkpoints write to `models/`.
- For data or storage changes, run `update-daily`, one short train, and one predict/backtest; summarize outputs in the PR.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative, optional scope (e.g., `cli: add resume flag`); one concern per commit.
- PRs should describe intent, list validation commands, mention new env vars/defaults, link issues, and attach report/dashboard paths or screenshots when relevant.
- Call out breaking changes (schema, checkpoint format, report columns) and note migration/backfill steps.

## Security & Configuration Tips
- Keep secrets in env vars (`OPENAI_API_KEY`, `AUTO_SELECT_LLM_MODEL`, etc.); never commit keys.
- Override storage/report roots with `AUTO_SELECT_STOCK_DATA_DIR`, `AUTO_SELECT_STOCK_REPORT_DIR`, `AUTO_SELECT_MODEL_DIR`, `AUTO_SELECT_STOCK_PREPROCESSED_DIR`; keep large artifacts untracked or stored externally.
