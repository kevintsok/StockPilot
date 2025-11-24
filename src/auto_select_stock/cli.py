import argparse
import concurrent.futures
import json
import math
import sys
from pathlib import Path

import pandas as pd
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

from . import data_fetcher
from . import financials_fetcher
from .config import DATA_DIR, DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, MODEL_DIR, REPORT_DIR
from .dashboard import build_rows, render_dashboard
from .html_report import render_report
from .predict.backtest import BacktestConfig, filter_a_share_symbols, run_backtest, run_backtest_for_symbol, run_topk_strategy
from .scoring import score_symbols
from .storage import ensure_data_dir, list_symbols
def _lazy_torch_import():
    # Delay heavy deps (torch) so fetch/update can run without GPU/torch installed.
    from .predict.backtest import BacktestConfig, run_backtest
    from .predict.inference import predict_next_close
    from .predict.train import train_from_symbols
    from .torch_model import TrainConfig

    return TrainConfig, predict_next_close, train_from_symbols, BacktestConfig, run_backtest


def symbols_from_data_dir(path: Path) -> list[str]:
    if path.exists():
        cached = [p.stem for p in path.glob("*.npz")]
        if cached:
            return cached
    try:
        return list_symbols(base_dir=path)
    except Exception:
        return []


def _safe_number(value):
    try:
        num = float(value)
    except Exception:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _serialize_backtest_timeseries(result) -> list[dict]:
    rows = []
    for dt in result.daily_returns.index:
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "gross_ret": _safe_number(result.daily_returns.loc[dt]),
                "net_ret": _safe_number(result.daily_returns_net.loc[dt]),
                "turnover": _safe_number(result.turnover.loc[dt]),
                "industry_hhi": _safe_number(result.industry_hhi.loc[dt]),
                "cumulative": _safe_number(result.cumulative.loc[dt]),
                "cumulative_net": _safe_number(result.cumulative_net.loc[dt]),
            }
        )
    return rows


def _write_with_fallback(path: Path, writer) -> tuple[Path | None, Exception | None]:
    """
    Attempt to write using writer(path); if it fails (e.g., permission), retry in CWD and /tmp with same filename.
    """
    candidates = [path, Path.cwd() / path.name, Path("/tmp") / path.name]
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            writer(candidate)
            return candidate, None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    return None, last_exc


def _sanitize_label(value: str) -> str:
    return value.replace("/", "-").replace(":", "").replace(" ", "_")


def _build_backtest_paths(checkpoint: Path, start: str | None, end: str | None) -> tuple[Path, Path]:
    start_label = _sanitize_label(start) if start else "begin"
    end_label = _sanitize_label(end) if end else "end"
    stem = checkpoint.stem
    base_dir = checkpoint.parent
    metrics = base_dir / f"{stem}_{start_label}_to_{end_label}_strategy_backtest.json"
    trades = base_dir / f"{stem}_{start_label}_to_{end_label}_strategy_daily_trades.csv"
    return metrics, trades


def cmd_fetch_all(args):
    ensure_data_dir()
    symbols = data_fetcher.fetch_all(start_date=args.start, limit=args.limit, base_dir=DATA_DIR)
    print(f"Fetched {len(symbols)} symbols.")


def cmd_update_daily(args):
    ensure_data_dir()
    symbols = args.symbols or symbols_from_data_dir(DATA_DIR)
    for code in symbols:
        data_fetcher.append_latest(code, base_dir=DATA_DIR)
    print(f"Updated {len(symbols)} symbols.")


def cmd_score(args):
    symbols = args.symbols or symbols_from_data_dir(DATA_DIR)
    scores = score_symbols(symbols, provider=args.provider or DEFAULT_LLM_PROVIDER, model=args.model or DEFAULT_LLM_MODEL)
    top_n = args.top or len(scores)
    for item in scores[:top_n]:
        print(f"{item.symbol}\t{item.score:.2f}\t{item.rationale}")


def cmd_render(args):
    symbols = args.symbols or symbols_from_data_dir(DATA_DIR)
    scores = score_symbols(symbols, provider=args.provider or DEFAULT_LLM_PROVIDER, model=args.model or DEFAULT_LLM_MODEL)
    top_n = args.top or len(scores)
    output = Path(args.output) if args.output else REPORT_DIR / "undervalued.html"
    render_report(scores, top_n=top_n, output_path=output)
    print(f"Report written to {output}")


def cmd_train_transformer(args):
    TrainConfig, _, train_from_symbols, _, _ = _lazy_torch_import()
    ensure_data_dir()
    symbols = args.symbols or symbols_from_data_dir(DATA_DIR)
    cfg = TrainConfig(
        seq_len=args.seq_len,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        eval_every=args.eval_every,
        device=args.device,
        train_ratio=args.train_ratio,
        date_windows=args.date_window or [],
        save_path=Path(args.save_path) if args.save_path else MODEL_DIR / "price_transformer.pt",
        num_workers=args.num_workers,
        experiment_name=args.experiment_name,
        checkpoint_steps=args.checkpoint_steps,
        resume_checkpoint=Path(args.resume_from) if args.resume_from else None,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags or [],
        wandb_mode=args.wandb_mode,
        keep_preprocessed_in_memory=args.preprocess_in_memory,
        profile=getattr(args, "profile", False),
    )
    stats = train_from_symbols(symbols, cfg, base_dir=DATA_DIR)
    if isinstance(stats, dict) and "best_val_loss" not in stats:
        print("Training finished for date windows:")
        for name, window_stats in stats.items():
            print(
                f"  window={name} best_val_loss={window_stats.get('best_val_loss')} "
                f"best_val_last_mae={window_stats.get('best_val_last_mae')} "
                f"test_loss={window_stats.get('test_loss')} checkpoint={window_stats.get('checkpoint')}"
            )
    else:
        print(
            f"Training finished. best_val_loss={stats.get('best_val_loss')} "
            f"best_val_last_mae={stats.get('best_val_last_mae')} checkpoint={cfg.save_path}"
        )


def cmd_predict_transformer(args):
    TrainConfig, predict_next_close, _, _, _ = _lazy_torch_import()
    checkpoint = Path(args.checkpoint) if args.checkpoint else MODEL_DIR / "price_transformer.pt"
    price = predict_next_close(
        symbol=args.symbol,
        checkpoint_path=checkpoint,
        seq_len=args.seq_len,
        base_dir=DATA_DIR,
        device=args.device,
    )
    print(f"{args.symbol}\tnext_close={price:.4f}\tcheckpoint={checkpoint}")


def cmd_fetch_financials(args):
    symbols = args.symbols or symbols_from_data_dir(DATA_DIR)
    if not symbols:
        # Fall back to full universe when本地尚未有日线数据，便于一键抓取财报。
        symbols = data_fetcher.list_all_symbols(base_dir=DATA_DIR)
    written = financials_fetcher.fetch_financials_for_symbols(symbols, base_dir=DATA_DIR, limit=args.limit)
    print(f"Fetched financials for {len(written)} symbols.")


def cmd_render_dashboard(args):
    symbols = args.symbols or symbols_from_data_dir(DATA_DIR)
    lookbacks = [args.lookback_short, args.lookback_long]
    rows = build_rows(symbols, lookbacks, base_dir=DATA_DIR)
    output = Path(args.output) if args.output else REPORT_DIR / "dashboard.html"
    path = render_dashboard(rows, output=output)
    print(f"Dashboard written to {path} with {len(rows)} rows.")


def cmd_backtest_transformer(args):
    TrainConfig, _, _, BacktestConfig, run_backtest = _lazy_torch_import()
    ensure_data_dir()
    checkpoint = Path(args.checkpoint) if args.checkpoint else MODEL_DIR / "price_transformer.pt"
    symbols = args.symbols or filter_a_share_symbols(symbols_from_data_dir(DATA_DIR))
    allow_short = getattr(args, "allow_short", False)
    top_pct = getattr(args, "top_pct", 0.1)
    mode = getattr(args, "mode", "topk")
    cfg = BacktestConfig(
        checkpoint=checkpoint,
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
        top_pct=top_pct,
        allow_short=allow_short,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        base_dir=DATA_DIR,
        eval_batch_size=args.eval_batch_size,
    )
    if mode == "topk":
        top_k = getattr(args, "top_k", 5)
        result = run_topk_strategy(cfg, top_k=top_k, show_progress=True)
        print("Strategy backtest metrics (top-K long-only):")
        for k, v in result.metrics.items():
            print(f"  {k}: {v}")
        print("Last 5 days cumulative gross/net:")
        tail = pd.DataFrame({"cumulative": result.cumulative, "cumulative_net": result.cumulative_net}).tail(5)
        print(tail)
        out_path, trades_path = _build_backtest_paths(checkpoint, args.start, args.end)
        payload = {
            "checkpoint": str(checkpoint),
            "mode": "topk_long_only",
            "start": args.start,
            "end": args.end,
            "top_k": top_k,
            "cost_bps": args.cost_bps,
            "slippage_bps": args.slippage_bps,
            "eval_batch_size": args.eval_batch_size,
            "symbols": symbols,
            "metrics": result.metrics,
            "timeseries": _serialize_backtest_timeseries(result),
            "trades_path": str(trades_path),
        }
        wrote_path, err = _write_with_fallback(out_path, lambda p: p.write_text(json.dumps(payload, indent=2, ensure_ascii=False)))
        if result.trades:
            df_trades = pd.DataFrame(result.trades)
            trades_written, trade_err = _write_with_fallback(trades_path, lambda p: df_trades.to_csv(p, index=False))
            if trades_written:
                print(f"[Strategy backtest] Saved trades to {trades_written}")
            elif trade_err:
                print(f"[Strategy backtest] Failed to write trades CSV: {trade_err}")
        if wrote_path:
            print(f"[Strategy backtest] Saved metrics to {wrote_path}")
        elif err:
            print(f"[Strategy backtest] Failed to write metrics JSON: {err}")
        return

    result = run_backtest(cfg)
    print("Backtest metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")
    print("Last 5 days cumulative gross/net:")
    tail = pd.DataFrame({"cumulative": result.cumulative, "cumulative_net": result.cumulative_net}).tail(5)
    print(tail)


def cmd_backtest_strategy(args):
    # Reuse the transformer backtest command with a fixed mode to avoid duplicated logic.
    args.mode = "topk"
    # Provide defaults for fields not present on this sub-command to satisfy BacktestConfig.
    if not hasattr(args, "allow_short"):
        args.allow_short = False
    if not hasattr(args, "top_pct"):
        args.top_pct = 0.1
    return cmd_backtest_transformer(args)


def cmd_backtest_per_symbol(args):
    checkpoint = Path(args.checkpoint)
    ensure_data_dir()
    symbols = args.symbols or filter_a_share_symbols(symbols_from_data_dir(DATA_DIR))
    tasks = list(symbols)
    if not tasks:
        print("No symbols found for backtest.")
        return 1
    cfg = BacktestConfig(
        checkpoint=checkpoint,
        start_date=args.start,
        end_date=args.end,
        symbols=None,
        top_pct=args.top_pct,
        allow_short=args.allow_short,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        base_dir=DATA_DIR,
        eval_batch_size=args.eval_batch_size,
    )
    rows = []
    errors = []
    progress_desc = "Signal gen (stocks)"

    if args.workers > 1:
        # Each worker loads its own model; good for CPU-heavy runs or multiple GPUs.
        bar = tqdm(total=len(tasks), desc=progress_desc, unit="stock") if tqdm is not None else None
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                futs = [executor.submit(run_backtest_for_symbol, cfg, sym, None, False) for sym in tasks]
                for fut in concurrent.futures.as_completed(futs):
                    sym, metrics, err = fut.result()
                    if bar is not None:
                        bar.update(1)
                        bar.set_description(f"{progress_desc} {bar.n}/{bar.total}")
                    if err:
                        errors.append((sym, err))
                        rows.append({"symbol": sym, "error": err})
                        continue
                    row = {"symbol": sym}
                    row.update(metrics)
                    rows.append(row)
        finally:
            if bar is not None:
                bar.close()
    else:
        # Reuse a single predictor to leverage GPU batching for all symbols.
        predictor = None
        try:
            from auto_select_stock.predict.inference import PricePredictor

            predictor = PricePredictor(checkpoint)
        except Exception as exc:  # noqa: BLE001
            print(f"[Backtest per symbol] Failed to init predictor with shared model: {exc}")

        bar = tqdm(total=len(tasks), desc=progress_desc, unit="stock") if tqdm is not None else None
        try:
            for sym in tasks:
                sym_metrics, err = {}, None
                try:
                    _, sym_metrics, err = run_backtest_for_symbol(cfg, sym, predictor=predictor, show_progress=False)
                except Exception as exc:  # noqa: BLE001
                    err = str(exc)
                if bar is not None:
                    bar.update(1)
                    bar.set_description(f"{progress_desc} {bar.n}/{bar.total}")
                if err:
                    errors.append((sym, err))
                    rows.append({"symbol": sym, "error": err})
                    continue
                row = {"symbol": sym}
                row.update(sym_metrics)
                rows.append(row)
        finally:
            if bar is not None:
                bar.close()
    output_path = checkpoint.with_name(f"{checkpoint.stem}_per_symbol_returns.csv")
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[Backtest per symbol] wrote {len(rows)} rows to {output_path}")
    if errors:
        print(f"[Backtest per symbol] {len(errors)} symbols failed (see CSV 'error' column).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A股自动选股器 CLI")
    sub = parser.add_subparsers(dest="command")

    p_fetch = sub.add_parser("fetch-all", help="抓取全市场历史数据")
    p_fetch.add_argument("--start", default="2018-01-01")
    p_fetch.add_argument("--limit", type=int, default=None, help="只抓取前 N 只股票用于测试")
    p_fetch.set_defaults(func=cmd_fetch_all)

    p_update = sub.add_parser("update-daily", help="增量更新已有股票")
    p_update.add_argument("symbols", nargs="*", help="指定股票代码，不填则更新 data 目录下所有文件")
    p_update.set_defaults(func=cmd_update_daily)

    p_score = sub.add_parser("score", help="为股票打分（LLM）")
    p_score.add_argument("symbols", nargs="*", help="指定股票代码，不填则默认 data 目录下全部")
    p_score.add_argument("--provider", default=None, help="LLM provider (openai/dummy)")
    p_score.add_argument("--model", default=None, help="模型名称")
    p_score.add_argument("--top", type=int, default=None, help="只输出前 N 名")
    p_score.set_defaults(func=cmd_score)

    p_render = sub.add_parser("render", help="生成 HTML 榜单")
    p_render.add_argument("symbols", nargs="*", help="指定股票代码，不填则默认 data 目录下全部")
    p_render.add_argument("--provider", default=None)
    p_render.add_argument("--model", default=None)
    p_render.add_argument("--top", type=int, default=None)
    p_render.add_argument("--output", help="输出路径，默认 reports/undervalued.html")
    p_render.set_defaults(func=cmd_render)

    p_train = sub.add_parser("train-transformer", help="训练 Transformer 模型预测次日收盘价")
    p_train.add_argument("symbols", nargs="*", help="指定股票代码，默认 data 目录下全部")
    p_train.add_argument("--seq-len", type=int, default=1024, help="输入序列长度")
    p_train.add_argument("--window-stride", type=int, default=10, help="滑动窗口步长（天）")
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--eval-every", type=int, default=1, help="多少个 epoch 执行一次验证")
    p_train.add_argument("--device", default="cuda", help="指定设备，例如 cuda 或 cuda:0；默认自动检测")
    p_train.add_argument("--train-ratio", type=float, default=0.8, help="训练集占比")
    p_train.add_argument(
        "--date-window",
        action="append",
        default=None,
        help="按日期切分 train/val/test 的窗口，格式 TRAIN_END:VAL_END，可重复，例如 2022-01-01:2023-01-01",
    )
    p_train.add_argument("--num-workers", type=int, default=0, help="DataLoader workers 数量")
    p_train.add_argument("--save-path", default=None, help="模型检查点保存路径，默认 models/price_transformer.pt")
    p_train.add_argument("--experiment-name", default="experiment", help="实验名，用于中间 checkpoint 命名")
    p_train.add_argument("--checkpoint-steps", type=int, default=1000, help="每隔多少个 global step 保存 checkpoint；<=0 关闭")
    p_train.add_argument("--resume-from", default=None, help="从已有 checkpoint 继续训练")
    p_train.add_argument("--wandb-project", default=None, help="wandb 项目名，填写后开启日志上报")
    p_train.add_argument("--wandb-run-name", default=None, help="wandb run 名称")
    p_train.add_argument("--wandb-tags", nargs="*", default=None, help="wandb tags 列表")
    p_train.add_argument("--wandb-mode", default=None, help="wandb 模式，例如 offline/disabled")
    p_train.add_argument(
        "--preprocess-in-memory",
        action="store_true",
        help="预处理后把特征缓存留在内存中，加速组装数据集（占用更高内存）",
    )
    p_train.add_argument("--profile", action="store_true", help="使用 torch.profiler 记录第 2 个 step（保存到 reports/）")
    p_train.set_defaults(func=cmd_train_transformer)

    p_pred = sub.add_parser("predict-transformer", help="使用训练好的 Transformer 预测次日收盘价")
    p_pred.add_argument("symbol", help="股票代码")
    p_pred.add_argument("--seq-len", type=int, default=1024, help="输入序列长度，需与训练时一致")
    p_pred.add_argument("--checkpoint", default=None, help="模型检查点路径，默认 models/price_transformer.pt")
    p_pred.add_argument("--device", default=None, help="指定设备，例如 cuda 或 cpu")
    p_pred.set_defaults(func=cmd_predict_transformer)

    p_fin = sub.add_parser("fetch-financials", help="抓取财报指标并保存到 data/financials")
    p_fin.add_argument("symbols", nargs="*", help="指定股票代码，不填则默认 data 目录下全部")
    p_fin.add_argument("--limit", type=int, default=None, help="只抓取前 N 只股票用于测试")
    p_fin.set_defaults(func=cmd_fetch_financials)

    p_dash = sub.add_parser("render-dashboard", help="生成可排序的价格+财务看板 HTML")
    p_dash.add_argument("symbols", nargs="*", help="指定股票代码，不填则默认 data 目录下全部")
    p_dash.add_argument("--lookback-short", type=int, default=20, help="短周期涨跌幅天数")
    p_dash.add_argument("--lookback-long", type=int, default=60, help="长周期涨跌幅天数")
    p_dash.add_argument("--output", default=None, help="输出路径，默认 reports/dashboard.html")
    p_dash.set_defaults(func=cmd_render_dashboard)

    p_backtest = sub.add_parser("backtest-transformer", help="使用训练好的 Transformer 做批量推理的策略回测")
    p_backtest.add_argument("symbols", nargs="*", help="指定股票代码，默认 data 目录下全部")
    p_backtest.add_argument("--start", default=None, help="回测起始日期，格式 YYYY-MM-DD")
    p_backtest.add_argument("--end", default=None, help="回测结束日期，格式 YYYY-MM-DD")
    p_backtest.add_argument(
        "--mode",
        choices=["topk", "longshort"],
        default="topk",
        help="topk: 每日按预测涨幅买入 Top-N（默认）；longshort: 前/后分位多空",
    )
    p_backtest.add_argument("--top-k", type=int, default=5, help="mode=topk 时每日买入的前 N 只（按预测涨幅归一化）")
    p_backtest.add_argument("--top-pct", type=float, default=0.1, help="mode=longshort 时多空各取的比例，例如 0.1")
    p_backtest.add_argument("--allow-short", action="store_true", help="mode=longshort 时是否做空尾部组合")
    p_backtest.add_argument("--checkpoint", default=None, help="模型 checkpoint 路径，默认 models/price_transformer.pt")
    p_backtest.add_argument("--cost-bps", type=float, default=0.0, help="单边交易成本（bp）")
    p_backtest.add_argument("--slippage-bps", type=float, default=0.0, help="滑点（bp）")
    p_backtest.add_argument("--eval-batch-size", type=int, default=64, help="回测推理 batch 大小")
    p_backtest.set_defaults(func=cmd_backtest_transformer)

    p_strategy = sub.add_parser("backtest-strategy", help="按日 batch 推理+排序的策略回测，结果保存在 checkpoint 旁")
    p_strategy.add_argument("symbols", nargs="*", help="指定股票代码，默认 data 目录下全部")
    p_strategy.add_argument("--start", default=None, help="回测起始日期，格式 YYYY-MM-DD")
    p_strategy.add_argument("--end", default=None, help="回测结束日期，格式 YYYY-MM-DD")
    p_strategy.add_argument("--top-k", type=int, default=5, help="每日买入的前 K 只（按预测涨幅归一化）")
    p_strategy.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    p_strategy.add_argument("--cost-bps", type=float, default=0.0, help="单边交易成本（bp）")
    p_strategy.add_argument("--slippage-bps", type=float, default=0.0, help="滑点（bp）")
    p_strategy.add_argument("--eval-batch-size", type=int, default=64, help="推理 batch 大小（越大越能利用 GPU）")
    p_strategy.set_defaults(func=cmd_backtest_strategy)

    p_backtest_sym = sub.add_parser("backtest-per-symbol", help="并行按股票回测并输出每只股票的窗口收益")
    p_backtest_sym.add_argument("symbols", nargs="*", help="指定股票代码，默认 data 目录下全部")
    p_backtest_sym.add_argument("--start", default=None, help="回测起始日期，格式 YYYY-MM-DD")
    p_backtest_sym.add_argument("--end", default=None, help="回测结束日期，格式 YYYY-MM-DD")
    p_backtest_sym.add_argument("--top-pct", type=float, default=0.1, help="多空各取的比例")
    p_backtest_sym.add_argument("--allow-short", action="store_true", help="是否做空尾部组合")
    p_backtest_sym.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    p_backtest_sym.add_argument("--cost-bps", type=float, default=0.0, help="单边交易成本（bp）")
    p_backtest_sym.add_argument("--slippage-bps", type=float, default=0.0, help="滑点（bp）")
    p_backtest_sym.add_argument("--workers", type=int, default=4, help="并行 worker 数，默认 4")
    p_backtest_sym.add_argument("--eval-batch-size", type=int, default=64, help="回测推理 batch 大小")
    p_backtest_sym.set_defaults(func=cmd_backtest_per_symbol)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    debug_args = vars(args).copy()
    func = debug_args.pop("func", None)
    if func:
        debug_args["func"] = getattr(func, "__name__", str(func))
    print(f"[Args] {debug_args}")
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
