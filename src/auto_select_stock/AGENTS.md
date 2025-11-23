# Repository Guidelines

## Project Structure & Module Organization
- Root assets: `README.md` (usage), `requirements.txt` (Python deps), `start_ops_dashboard.sh` (control panel launcher), data outputs under `data/`, `models/`, `reports/`, and logs under `logs/`.
- Core code lives in `src/auto_select_stock/`: `cli.py` (entrypoint), `data_fetcher.py` and `financials_fetcher.py` (market/financial ingestion), `storage.py` (NumPy I/O), `scoring.py` + `llm/` (LLM provider adapters), `torch_model.py` (Transformer training/inference), `html_report.py` and `dashboard.py` (report rendering), `ops_dashboard.py` (ops UI).
- Config defaults and overrideable paths are in `config.py` (`AUTO_SELECT_*` env vars).

## Build, Test, and Development Commands
- Install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Run CLI (from repo root): `python -m auto_select_stock.cli fetch-all --start 2018-01-01` to bootstrap data; `python -m auto_select_stock.cli update-daily` for increments.
- Score and report (dummy LLM is deterministic and requires no keys): `python -m auto_select_stock.cli score --provider dummy --top 10`; render HTML: `python -m auto_select_stock.cli render --provider dummy --top 50 --output reports/undervalued.html`.
- Model workflows: `python -m auto_select_stock.cli train-transformer --device cuda` to train; `python -m auto_select_stock.cli predict-transformer 600000 --checkpoint models/price_transformer.pt` to infer.
- Ops dashboard: `./start_ops_dashboard.sh` (sets `PYTHONPATH` and starts at http://127.0.0.1:8000).

## Coding Style & Naming Conventions
- Python 3; follow PEP 8 with 4-space indents and descriptive snake_case names; keep public functions typed (type hints are already present in most modules).
- Prefer pure functions and small helpers; keep side effects at CLI/command boundaries.
- Organize new providers under `llm/` and keep adapter interfaces consistent with `base.py`.

## Testing Guidelines
- No formal test suite yet; validate changes by running CLI flows above with `--provider dummy` and inspecting generated outputs in `reports/`.
- For model changes, run a short epoch smoke test (`--epochs 1 --limit 20` on fetch) to confirm training still runs.
- Capture failures or sample outputs in PR descriptions when adding new data paths or models.

## Commit & Pull Request Guidelines
- Use imperative, scoped commit messages (e.g., `cli: add top-n filter`, `llm: handle rate limits`); one logical change per commit when possible.
- PRs should include: summary of intent, main commands run for validation, any new env vars/config defaults, and screenshots/paths for generated HTML dashboards when relevant.
- Link related issues/tasks and call out backward-incompatible changes (data schema, report format, model checkpoints).

## Security & Configuration Tips
- Keep API keys (e.g., `OPENAI_API_KEY`) in env vars; never check them into the repo. Override paths with `AUTO_SELECT_STOCK_DATA_DIR`, `AUTO_SELECT_STOCK_REPORT_DIR`, `AUTO_SELECT_MODEL_DIR` when needed.
- Large artifacts in `data/`, `models/`, `reports/` should stay out of version control; prefer `.gitignore` or external storage for datasets and checkpoints.
