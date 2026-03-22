"""
Stock prediction pipeline: inference + strategy selection.

Provides a reusable interface for getting top-K stock recommendations
based on a trained Transformer model and a given strategy.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .predict.inference import PricePredictor
from .predict.strategies.base import Signal
from .predict.strategies.registry import StrategyRegistry, make_strategy
from .storage import _connect, list_symbols
from .predict.backtest import filter_a_share_symbols


def get_latest_price_date() -> str:
    """Return the most recent price date in the database."""
    conn = _connect()
    return conn.execute("SELECT MAX(date) FROM price").fetchone()[0]


def run_inference(
    symbols: List[str],
    predictor: PricePredictor,
) -> List[Tuple[str, float, float, Optional[str], Optional[Dict[str, float]]]]:
    """
    Run model inference on a list of symbols.

    Returns list of (symbol, predicted_ret_1d, realized_ret, industry, predicted_rets) tuples.
    predicted_rets is a dict of {horizon: return} when multi-horizon is available, else None.
    predicted_ret_1d is always a float (1d return) for backward compatibility.
    """
    all_signals = []
    for i, sym in enumerate(symbols):
        try:
            result = predictor.predict(sym, horizon=None)  # get all horizons
            conn = _connect()
            row = conn.execute(
                "SELECT close FROM price WHERE symbol=? ORDER BY date DESC LIMIT 1",
                (sym,),
            ).fetchone()
            if row:
                last_close = row[0]
                if isinstance(result, dict):
                    # Multi-horizon: use 1d for backward compat
                    pred_ret_1d = result.get("1d", 0.0)
                    predicted_rets = result
                else:
                    pred_ret_1d = float(result)
                    predicted_rets = None
                all_signals.append((sym, float(pred_ret_1d), 0.0, None, predicted_rets))
        except Exception:
            pass
    return all_signals


def apply_strategy(
    all_signals: List[Tuple[str, float, float, Optional[str], Optional[Dict[str, float]]]],
    strategy_name: str = "confidence",
    top_k: int = 10,
    **strategy_kwargs,
) -> List[Tuple[str, float, float]]:
    """
    Apply a named strategy to signals and return ranked results.

    Args:
        all_signals: list of (symbol, predicted_ret_1d, realized_ret, industry, predicted_rets)
        strategy_name: strategy name (tries registry first, then falls back to built-in types)
        top_k: number of top stocks to return
        **strategy_kwargs: passed to strategy constructor

    Returns:
        List of (symbol, predicted_ret_1d, weight), sorted by weight descending.
    """
    # Try to find strategy in registry first
    try:
        registry = StrategyRegistry(Path(__file__).parent / "predict" / "strategies" / "configs")
        cfg = registry.get(strategy_name)
        strategy = make_strategy(cfg)
    except Exception:
        # Fall back to built-in strategy types
        from .predict.strategies import ConfidenceStrategy
        if strategy_name == "confidence":
            strategy = ConfidenceStrategy(top_k=top_k, **strategy_kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    signals = [
        Signal(
            symbol=s[0],
            predicted_ret=s[1],
            realized_ret=s[2],
            industry=s[3],
            predicted_rets=s[4] if len(s) > 4 else None,
        )
        for s in all_signals
    ]
    weights = strategy.select_positions(signals, {}, {})

    # Build result with predicted_ret included
    pred_map = {s[0]: s[1] for s in all_signals}
    results = [(sym, pred_map[sym], w) for sym, w in weights.items()]
    results.sort(key=lambda x: -x[2])
    return results[:top_k]


def get_top_k_stocks(
    checkpoint: str = "models/price_transformer-train20220101-val20230101.pt",
    strategy: str = "confidence",
    top_k: int = 10,
    horizon: Optional[Union[int, str]] = None,
) -> List[Tuple[str, Dict[str, float], float]]:
    """
    Main entry point: load model, run inference, apply strategy, return top-K stocks.

    Args:
        checkpoint: path to model checkpoint
        strategy: strategy name
        top_k: number of stocks to return
        horizon: None -> use all horizons; specific horizon -> use only that horizon for ranking

    Returns:
        List of (symbol, {horizon: return, ...}, weight), sorted by weight descending.
    """
    symbols = filter_a_share_symbols(list_symbols())
    predictor = PricePredictor(checkpoint)
    all_signals = run_inference(symbols, predictor)

    # Build rets_map BEFORE any horizon filtering (keeps all horizons intact)
    rets_map = {s[0]: s[4] for s in all_signals if s[4]}

    # Create ranked signals using specific horizon for ranking (if provided)
    if horizon is not None:
        h_str = str(horizon) if isinstance(horizon, int) else horizon
        if not h_str.endswith("d"):
            h_str = f"{h_str}d"
        ranked_signals = []
        for s in all_signals:
            predicted_rets = s[4]
            # Use the ranking horizon's predicted value for sorting; keep ALL horizons in predicted_rets
            ranked_pred = predicted_rets.get(h_str, s[1]) if predicted_rets else s[1]
            ranked_signals.append((s[0], ranked_pred, s[2], s[3], s[4] if s[4] else {h_str: ranked_pred}))
        all_signals = ranked_signals

    strategy_results = apply_strategy(all_signals, strategy_name=strategy, top_k=top_k)

    # Build final results with full multi-horizon predicted_rets preserved
    results = []
    for sym, pred_ret, weight in strategy_results:
        pred_rets = rets_map.get(sym, {f"{1}d": pred_ret})
        results.append((sym, pred_rets, weight))
    return results
