__all__ = [
    "TrainConfig",
    "PricePredictor",
    "predict_next_close",
    "train_from_symbols",
    "load_model",
    "run_backtest",
    "BacktestConfig",
    "BacktestResult",
    "build_long_short_portfolio",
]


def __getattr__(name):
    if name == "PricePredictor" or name == "predict_next_close" or name == "load_model":
        from .inference import PricePredictor, predict_next_close, load_model
        return {"PricePredictor": PricePredictor, "predict_next_close": predict_next_close, "load_model": load_model}[name]
    if name == "TrainConfig" or name == "train_from_symbols":
        from .train import TrainConfig, train_from_symbols
        return {"TrainConfig": TrainConfig, "train_from_symbols": train_from_symbols}[name]
    if name in {"run_backtest", "BacktestConfig", "BacktestResult"}:
        from .backtest import BacktestConfig, BacktestResult, run_backtest
        return {"run_backtest": run_backtest, "BacktestConfig": BacktestConfig, "BacktestResult": BacktestResult}[name]
    if name == "build_long_short_portfolio":
        from .strategy import build_long_short_portfolio
        return build_long_short_portfolio
    raise AttributeError(name)
