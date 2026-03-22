from typing import Dict, List, Optional, Tuple

from .strategies.base import Signal


def build_long_short_portfolio(
    signals: List[Signal],
    top_pct: float,
    allow_short: bool,
    prev_weights: Dict[str, float],
    industry_map: Optional[Dict[str, str]],
    cost_rate: float,
) -> Tuple[float, float, float, Dict[str, float], float]:
    """
    Construct a simple long/short (or long-only) portfolio based on predicted returns.

    signals: list of Signal objects
    top_pct: fraction to allocate on each side (0-1)
    allow_short: whether to short the bottom bucket
    prev_weights: previous day's weights for turnover calculation
    industry_map: optional symbol->industry mapping; when provided compute HHI on abs weights
    cost_rate: per-unit transaction cost (e.g., (cost_bps+slippage_bps)/10000)

    Returns: (gross_ret, net_ret, turnover, weights, industry_hhi)
    """
    if not signals:
        return 0.0, 0.0, 0.0, {}, float("nan")

    signals_sorted = sorted(signals, key=lambda s: s.predicted_ret, reverse=True)
    n = len(signals_sorted)
    pct = min(max(top_pct, 0.0), 1.0)
    top_n = max(1, int(n * pct))
    long_bucket = signals_sorted[:top_n]
    short_bucket: List[Signal] = signals_sorted[-top_n:] if allow_short else []

    weights: Dict[str, float] = {}
    long_alloc = 1.0 if not allow_short else 0.5
    short_alloc = 0.5 if allow_short else 0.0
    if long_bucket:
        w = long_alloc / len(long_bucket)
        for sig in long_bucket:
            weights[sig.symbol] = w
    if short_bucket:
        w = -short_alloc / len(short_bucket)
        for sig in short_bucket:
            weights[sig.symbol] = w

    gross_ret = 0.0
    for sig in signals_sorted:
        w = weights.get(sig.symbol)
        if w is None:
            continue
        gross_ret += w * sig.realized_ret

    turnover = 0.0
    for sym, w in weights.items():
        turnover += abs(w - prev_weights.get(sym, 0.0))
    for sym, w_prev in prev_weights.items():
        if sym not in weights:
            turnover += abs(w_prev)
    net_ret = gross_ret - turnover * cost_rate

    if industry_map is None:
        industry_hhi = float("nan")
    else:
        by_industry: Dict[str, float] = {}
        for sig in signals_sorted:
            weight = abs(weights.get(sig.symbol, 0.0))
            if weight == 0.0:
                continue
            ind = industry_map.get(sig.symbol) or sig.industry or "UNKNOWN"
            by_industry[ind] = by_industry.get(ind, 0.0) + weight
        industry_hhi = sum(v * v for v in by_industry.values()) if by_industry else float("nan")

    return gross_ret, net_ret, turnover, weights, industry_hhi


__all__ = ["build_long_short_portfolio"]
