import argparse
import sys
from pathlib import Path

import pandas as pd

from . import data_fetcher
from . import financials_fetcher
from .dashboard import build_rows, render_dashboard
from .config import DATA_DIR, DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, MODEL_DIR, REPORT_DIR
from .html_report import render_report
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
    )
    stats = train_from_symbols(symbols, cfg, base_dir=DATA_DIR)
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
    checkpoint = Path(args.checkpoint) if args.checkpoint else MODEL_DIR / "price_transformer.pt"
    symbols = args.symbols or symbols_from_data_dir(DATA_DIR)
    cfg = BacktestConfig(
        checkpoint=checkpoint,
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
        top_pct=args.top_pct,
        allow_short=args.allow_short,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        base_dir=DATA_DIR,
    )
    result = run_backtest(cfg)
    print("Backtest metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")
    print("Last 5 days cumulative gross/net:")
    tail = pd.DataFrame({"cumulative": result.cumulative, "cumulative_net": result.cumulative_net}).tail(5)
    print(tail)


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
    p_train.add_argument("--seq-len", type=int, default=60, help="输入序列长度")
    p_train.add_argument("--window-stride", type=int, default=10, help="滑动窗口步长（天）")
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--eval-every", type=int, default=1, help="多少个 epoch 执行一次验证")
    p_train.add_argument("--device", default=None, help="指定设备，例如 cuda 或 cuda:0；默认自动检测")
    p_train.add_argument("--train-ratio", type=float, default=0.8, help="训练集占比")
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
    p_train.set_defaults(func=cmd_train_transformer)

    p_pred = sub.add_parser("predict-transformer", help="使用训练好的 Transformer 预测次日收盘价")
    p_pred.add_argument("symbol", help="股票代码")
    p_pred.add_argument("--seq-len", type=int, default=60, help="输入序列长度，需与训练时一致")
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

    p_backtest = sub.add_parser("backtest-transformer", help="使用训练好的 Transformer 进行简单多空回测")
    p_backtest.add_argument("symbols", nargs="*", help="指定股票代码，默认 data 目录下全部")
    p_backtest.add_argument("--start", default=None, help="回测起始日期，格式 YYYY-MM-DD")
    p_backtest.add_argument("--end", default=None, help="回测结束日期，格式 YYYY-MM-DD")
    p_backtest.add_argument("--top-pct", type=float, default=0.1, help="多空各取的比例，例如 0.1 代表前/后 10%")
    p_backtest.add_argument("--allow-short", action="store_true", help="是否做空尾部组合")
    p_backtest.add_argument("--checkpoint", default=None, help="模型 checkpoint 路径，默认 models/price_transformer.pt")
    p_backtest.add_argument("--cost-bps", type=float, default=0.0, help="单边交易成本（bp）")
    p_backtest.add_argument("--slippage-bps", type=float, default=0.0, help="滑点（bp）")
    p_backtest.set_defaults(func=cmd_backtest_transformer)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
