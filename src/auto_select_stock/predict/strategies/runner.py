"""
回测运行器：一次预测，所有策略共享信号，并行运行。

核心思路：
1. _collect_signals_batched() 一次性收集所有股票所有日期的信号（一次GPU推理）
2. 对每个策略，遍历每日信号调用 select_positions()，获得当日权重
3. 权重 × realized_ret = 当日组合收益
4. 所有策略共享同一次信号收集，避免重复推理

共享状态：
- 每日 signals（同一批预测结果）
- 每日 realized_ret、predicted_ret 完全相同
- 各策略独立维护 prev_weights 和 cache
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtest import BacktestConfig, _collect_signals_batched, _parse_date, filter_a_share_symbols
from ..inference import PricePredictor
from ...storage import list_symbols
from .base import BaseStrategy, Signal

_TRADING_DAYS = 252


# ----------------------------------------------------------------------
# 结果数据类
# ----------------------------------------------------------------------


@dataclass
class StrategyBacktestResult:
    strategy_name: str
    daily_returns: pd.Series
    daily_returns_net: pd.Series
    cumulative: pd.Series
    cumulative_net: pd.Series
    turnover: pd.Series
    industry_hhi: pd.Series
    weights: pd.Series
    metrics: Dict[str, float]
    trades: List[Dict[str, Any]] = field(default_factory=list)


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------


def _annualize_return(daily_ret: pd.Series) -> float:
    if daily_ret.empty:
        return float("nan")
    total_return = (1 + daily_ret).prod()
    periods = len(daily_ret)
    return total_return ** (_TRADING_DAYS / periods) - 1


def _annualize_vol(daily_ret: pd.Series) -> float:
    if daily_ret.empty:
        return float("nan")
    return daily_ret.std(ddof=0) * np.sqrt(_TRADING_DAYS)


def _max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return float("nan")
    peak = cumulative.cummax()
    dd = cumulative / peak - 1.0
    return dd.min()


def _compute_turnover_rate(
    prev_weights: Dict[str, float],
    curr_weights: Dict[str, float],
) -> float:
    """权重层面的换手率（sum |w_new - w_old| / 2）。"""
    all_syms = set(prev_weights.keys()) | set(curr_weights.keys())
    return sum(abs(curr_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_syms) / 2


def _build_industry_hhi(
    weights: Dict[str, float],
    industry_map: Dict[str, str],
) -> float:
    by_ind: Dict[str, float] = {}
    for sym, w in weights.items():
        if abs(w) < 1e-10:
            continue
        ind = industry_map.get(sym, "UNKNOWN")
        by_ind[ind] = by_ind.get(ind, 0.0) + abs(w)
    if not by_ind:
        return float("nan")
    return sum(v * v for v in by_ind.values())


# ----------------------------------------------------------------------
# 核心：一次预测，所有策略共享信号
# ----------------------------------------------------------------------


def run_all_strategies_shared(
    strategies_dir: Path,
    cfg: BacktestConfig,
    show_progress: bool = True,
) -> List[StrategyBacktestResult]:
    """高效回测：一次 GPU 推理，所有策略共享信号。

    流程：
    1. 一次性收集所有股票所有日期的信号（调用一次 _collect_signals_batched）
    2. 初始化所有策略的 prev_weights 和 cache
    3. 遍历每日信号，对每个策略调用 select_positions()
    4. 每个策略独立计算当日收益、换手率、累计收益
    """
    from .registry import StrategyRegistry, make_strategy

    start_date = _parse_date(cfg.start_date)
    end_date = _parse_date(cfg.end_date)

    # ── 1. 加载所有策略 ──────────────────────────────────────────────
    registry = StrategyRegistry(strategies_dir)
    strategy_metas = registry.list_strategies()
    if not strategy_metas:
        raise RuntimeError(f"No strategies found in {strategies_dir}")

    strategies: List[BaseStrategy] = []
    for meta in strategy_metas:
        cfg_dict = registry.get(meta["name"])
        strategies.append(make_strategy(cfg_dict))

    # ── 2. 一次推理，收集所有信号 ────────────────────────────────────
    predictor = PricePredictor(cfg.checkpoint)

    if cfg.symbols:
        symbols = list(cfg.symbols)
    else:
        symbols = filter_a_share_symbols(list_symbols(base_dir=cfg.base_dir))
    if not symbols:
        raise RuntimeError("No symbols available for backtest.")

    # daily_raw: {date: [(symbol, pred_ret, realized_ret, industry)]}
    daily_raw = _collect_signals_batched(
        symbols, predictor, cfg, start_date, end_date, show_progress=show_progress
    )

    dates_sorted = sorted(daily_raw.keys())
    if not dates_sorted:
        raise RuntimeError("No signals generated in the specified backtest window.")

    cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10000.0
    industry_map = cfg.industry_map or {}

    # ── 3. 为每个策略准备独立状态 ─────────────────────────────────────
    # per-strategy state
    prev_weights_list: List[Dict[str, float]] = [{} for _ in strategies]
    cache_list: List[Dict[str, Any]] = [{} for _ in strategies]
    # per-strategy results buffers
    gross_rets_list: List[List[float]] = [[] for _ in strategies]
    net_rets_list: List[List[float]] = [[] for _ in strategies]
    turnovers_list: List[List[float]] = [[] for _ in strategies]
    hhi_list: List[List[float]] = [[] for _ in strategies]
    weights_list: List[List[Dict[str, float]]] = [[] for _ in strategies]
    trades_list: List[List[Dict[str, Any]]] = [[] for _ in strategies]

    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None

    date_iter = dates_sorted
    if show_progress and tqdm is not None:
        date_iter = tqdm(dates_sorted, desc="Backtesting (shared signals)", unit="day")

    # ── 4. 每日循环：对所有策略用同一批信号 ───────────────────────────
    for dt in date_iter:
        raw_signals = daily_raw[dt]
        # 构建 Signal 对象列表
        signals = [
            Signal(symbol=s, predicted_ret=p, realized_ret=r, industry=i)
            for s, p, r, i in raw_signals
        ]
        # 当日 realized_ret 映射
        realized_map = {s.symbol: s.realized_ret for s in signals}
        # 权重归一化用总预测 ret
        pred_map = {s.symbol: s.predicted_ret for s in signals}

        # 对每个策略分别计算
        for strat_idx, strat in enumerate(strategies):
            prev_w = prev_weights_list[strat_idx]
            cache = cache_list[strat_idx]

            # 策略决策
            weights = strat.select_positions(signals, prev_w, cache)

            # gross return = sum(w_i * realized_ret_i)
            gross = sum(w * realized_map.get(sym, 0.0) for sym, w in weights.items())

            # 换手率
            turnover = _compute_turnover_rate(prev_w, weights)

            # net return
            net = gross - turnover * cost_rate

            # 行业 HHI
            hhi = _build_industry_hhi(weights, industry_map)

            # 记录
            gross_rets_list[strat_idx].append(gross)
            net_rets_list[strat_idx].append(net)
            turnovers_list[strat_idx].append(turnover)
            hhi_list[strat_idx].append(hhi)
            weights_list[strat_idx].append(weights)

            # 策略 on_day_end hook
            strat.on_day_end(dt.strftime("%Y-%m-%d"), weights, realized_map, cache)

            # 更新状态
            prev_weights_list[strat_idx] = weights

    # ── 5. 组装结果 ─────────────────────────────────────────────────
    results: List[StrategyBacktestResult] = []

    for strat_idx, strat in enumerate(strategies):
        gross_series = pd.Series(gross_rets_list[strat_idx], index=dates_sorted[:len(gross_rets_list[strat_idx])])
        net_series = pd.Series(net_rets_list[strat_idx], index=dates_sorted[:len(net_rets_list[strat_idx])])
        turnover_series = pd.Series(turnovers_list[strat_idx], index=dates_sorted[:len(turnovers_list[strat_idx])])
        hhi_series = pd.Series(hhi_list[strat_idx], index=dates_sorted[:len(hhi_list[strat_idx])])
        weights_series = pd.Series(weights_list[strat_idx], index=dates_sorted[:len(weights_list[strat_idx])])

        cumulative = (1 + gross_series).cumprod()
        cumulative_net = (1 + net_series).cumprod()

        total_ret_gross = float(cumulative.iloc[-1] - 1) if len(cumulative) > 0 else 0.0
        total_ret_net = float(cumulative_net.iloc[-1] - 1) if len(cumulative_net) > 0 else 0.0

        ann_ret_gross = _annualize_return(gross_series)
        ann_vol_gross = _annualize_vol(gross_series)
        ann_ret_net = _annualize_return(net_series)
        ann_vol_net = _annualize_vol(net_series)

        metrics = {
            "annual_return_gross": ann_ret_gross,
            "annual_vol_gross": ann_vol_gross,
            "sharpe_gross": ann_ret_gross / (ann_vol_gross or np.nan),
            "total_return_gross": total_ret_gross,
            "annual_return_net": ann_ret_net,
            "annual_vol_net": ann_vol_net,
            "sharpe_net": ann_ret_net / (ann_vol_net or np.nan),
            "total_return_net": total_ret_net,
            "max_drawdown_gross": _max_drawdown(cumulative),
            "max_drawdown_net": _max_drawdown(cumulative_net),
            "avg_turnover": turnover_series.mean(),
            "median_turnover": turnover_series.median(),
            "avg_industry_hhi": hhi_series.mean() if not hhi_series.isna().all() else float("nan"),
            "num_days": len(gross_series),
        }

        results.append(StrategyBacktestResult(
            strategy_name=strat.name,
            daily_returns=gross_series,
            daily_returns_net=net_series,
            cumulative=cumulative,
            cumulative_net=cumulative_net,
            turnover=turnover_series,
            industry_hhi=hhi_series,
            weights=weights_series,
            metrics=metrics,
            trades=[],  # 可扩展：记录交易流水
        ))

    return results


# ----------------------------------------------------------------------
# 向后兼容：单策略回测
# ----------------------------------------------------------------------


def run_strategy_backtest(
    strategy: BaseStrategy,
    cfg: BacktestConfig,
    predictor: Optional[PricePredictor] = None,
    show_progress: bool = True,
) -> StrategyBacktestResult:
    """单策略回测（向后兼容接口）。"""
    from .registry import make_strategy

    start_date = _parse_date(cfg.start_date)
    end_date = _parse_date(cfg.end_date)
    predictor = predictor or PricePredictor(cfg.checkpoint)

    if cfg.symbols:
        symbols = list(cfg.symbols)
    else:
        symbols = filter_a_share_symbols(list_symbols(base_dir=cfg.base_dir))

    if not symbols:
        raise RuntimeError("No symbols available for backtest.")

    daily_raw = _collect_signals_batched(
        symbols, predictor, cfg, start_date, end_date, show_progress=show_progress
    )
    dates_sorted = sorted(daily_raw.keys())

    prev_weights: Dict[str, float] = {}
    cache: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10000.0
    industry_map = cfg.industry_map or {}

    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None

    date_iter = dates_sorted
    if show_progress and tqdm is not None:
        date_iter = tqdm(dates_sorted, desc=f"Strategy: {strategy.name}", unit="day")

    for dt in date_iter:
        raw_signals = daily_raw[dt]
        signals = [Signal(symbol=s, predicted_ret=p, realized_ret=r, industry=i) for s, p, r, i in raw_signals]
        realized_map = {s.symbol: s.realized_ret for s in signals}

        weights = strategy.select_positions(signals, prev_weights, cache)
        gross = sum(w * realized_map.get(sym, 0.0) for sym, w in weights.items())
        turnover = _compute_turnover_rate(prev_weights, weights)
        net = gross - turnover * cost_rate
        hhi = _build_industry_hhi(weights, industry_map)

        rows.append({
            "date": dt, "gross_ret": gross, "net_ret": net,
            "turnover": turnover, "industry_hhi": hhi,
        })

        strategy.on_day_end(dt.strftime("%Y-%m-%d"), weights, realized_map, cache)
        prev_weights = weights

    if not rows:
        raise RuntimeError("No signals generated.")

    df = pd.DataFrame(rows).set_index("date").sort_index()
    cumulative = (1 + df["gross_ret"]).cumprod()
    cumulative_net = (1 + df["net_ret"]).cumprod()

    ann_ret_gross = _annualize_return(df["gross_ret"])
    ann_vol_gross = _annualize_vol(df["gross_ret"])
    ann_ret_net = _annualize_return(df["net_ret"])
    ann_vol_net = _annualize_vol(df["net_ret"])

    metrics = {
        "annual_return_gross": ann_ret_gross,
        "annual_vol_gross": ann_vol_gross,
        "sharpe_gross": ann_ret_gross / (ann_vol_gross or np.nan),
        "total_return_gross": float(cumulative.iloc[-1] - 1) if len(cumulative) > 0 else 0,
        "annual_return_net": ann_ret_net,
        "annual_vol_net": ann_vol_net,
        "sharpe_net": ann_ret_net / (ann_vol_net or np.nan),
        "total_return_net": float(cumulative_net.iloc[-1] - 1) if len(cumulative_net) > 0 else 0,
        "max_drawdown_gross": _max_drawdown(cumulative),
        "max_drawdown_net": _max_drawdown(cumulative_net),
        "avg_turnover": df["turnover"].mean(),
        "median_turnover": df["turnover"].median(),
        "avg_industry_hhi": df["industry_hhi"].mean() if not df["industry_hhi"].isna().all() else float("nan"),
        "num_days": len(df),
    }

    return StrategyBacktestResult(
        strategy_name=strategy.name,
        daily_returns=df["gross_ret"],
        daily_returns_net=df["net_ret"],
        cumulative=cumulative,
        cumulative_net=cumulative_net,
        turnover=df["turnover"],
        industry_hhi=df["industry_hhi"],
        weights=pd.Series(dtype=object),
        metrics=metrics,
    )


def run_all_strategies(
    strategies_dir: Path,
    cfg: BacktestConfig,
    show_progress: bool = True,
) -> List[StrategyBacktestResult]:
    """运行目录下所有策略，统一用一次推理共享信号。"""
    return run_all_strategies_shared(strategies_dir, cfg, show_progress=show_progress)
