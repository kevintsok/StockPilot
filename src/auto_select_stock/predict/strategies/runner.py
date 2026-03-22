"""
回测运行器：一次预测，所有策略共享信号，并行运行。

核心思路：
1. _collect_signals_batched() 一次性收集所有股票所有日期的信号（一次GPU推理）
2. 对每个策略，遍历每日信号调用 select_positions()，获得当日权重
3. 以仓位（股数）为单位追踪资金账户，买入/卖出执行于当日收盘价
4. 所有策略共享同一次信号收集，避免重复推理

资金模型：
- 起始资金：100,000 RMB
- 最小交易单位：100股（A股规则）
- 涨停股（开盘涨幅≥9.5%）无法买入
- 跌停股（开盘跌幅≤-9.5%）无法卖出
- 卖出价 = 当日收盘价，买入价 = 当日收盘价（同价格执行，清算当日持仓）
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtest import BacktestConfig, _collect_signals_batched, _parse_date, filter_a_share_symbols
from ..inference import PricePredictor
from ...data.storage import list_symbols
from .base import BaseStrategy, Signal

_TRADING_DAYS = 252
_INITIAL_CAPITAL = 100_000.0   # 起始资金 10万元
_LOT_SIZE = 100                 # A股最小买卖单位
_AUC_LIMIT_THRESHOLD = 0.095   # 涨/跌幅超过此值认为涨跌停


# ----------------------------------------------------------------------
# 结果数据类
# ----------------------------------------------------------------------


@dataclass
class Trade:
    """单笔交易记录"""
    date: str          # 交易日期（T日，执行于T日收盘）
    symbol: str        # 股票代码
    action: str        # "buy" 或 "sell"
    price: float       # 成交价格（当日收盘价）
    shares: int        # 成交股数（100的倍数）
    amount: float     # 成交金额 = price × shares
    reason: str = ""  # 附加说明（如 "auc_limit_up" 表示因涨停无法买入）


@dataclass
class StrategyBacktestResult:
    strategy_name: str
    tag: str = ""                     # 短标签，如 "H1d_k5" 用于唯一标识
    daily_returns: pd.Series = None  # 每日收益率（当日组合价值变化率）
    daily_returns_net: pd.Series = None  # 扣费后每日收益率
    cumulative: pd.Series = None     # 累计组合价值序列
    cumulative_net: pd.Series = None  # 扣费后累计组合价值序列
    turnover: pd.Series = None       # 每日换手率
    industry_hhi: pd.Series = None
    weights: pd.Series = None
    metrics: Dict[str, float] = None
    trades: List[Trade] = field(default_factory=list)  # 所有交易记录
    capital: pd.Series = field(default_factory=None)   # 每日总资产（cash + holdings）
    holdings_value: pd.Series = field(default_factory=None)  # 持仓市值
    cash_series: pd.Series = field(default_factory=None)      # 每日现金


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

    资金模型：
    - 起始资金 100,000 RMB，每日结算
    - 卖出执行于当日收盘价 → 即时更新现金
    - 买入执行于当日收盘价 → 即时更新持仓
    - 涨跌停股无法买入/卖出
    - 最小交易单位 100 股
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

    # daily_raw: {date: [(symbol, pred_ret, realized_ret, industry, predicted_rets, entry_price, auc_limit)]}
    daily_raw = _collect_signals_batched(
        symbols, predictor, cfg, start_date, end_date, show_progress=show_progress
    )

    dates_sorted = sorted(daily_raw.keys())
    if not dates_sorted:
        raise RuntimeError("No signals generated in the specified backtest window.")

    cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10000.0
    industry_map = cfg.industry_map or {}

    # ── 3. 为每个策略准备独立状态 ─────────────────────────────────────
    # per-strategy position state: symbol -> {shares, entry_price, entry_date}
    positions_list: List[Dict[str, Dict[str, Any]]] = [{} for _ in strategies]
    cash_state_list: List[float] = [_INITIAL_CAPITAL for _ in strategies]  # 每个策略的现金状态
    prev_weights_list: List[Dict[str, float]] = [{} for _ in strategies]
    cache_list: List[Dict[str, Any]] = [{} for _ in strategies]

    # per-strategy results buffers
    gross_rets_list: List[List[float]] = [[] for _ in strategies]
    net_rets_list: List[List[float]] = [[] for _ in strategies]
    turnovers_list: List[List[float]] = [[] for _ in strategies]
    hhi_list: List[List[float]] = [[] for _ in strategies]
    weights_list: List[List[Dict[str, float]]] = [[] for _ in strategies]
    trades_list: List[List[Trade]] = [[] for _ in strategies]
    capital_list: List[List[float]] = [[] for _ in strategies]
    holdings_val_list: List[List[float]] = [[] for _ in strategies]
    cash_list: List[List[float]] = [[] for _ in strategies]

    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None

    date_iter = dates_sorted
    if show_progress and tqdm is not None:
        date_iter = tqdm(dates_sorted, desc="Backtesting (shared signals)", unit="day")

    # ── 4. 每日循环 ─────────────────────────────────────────────────
    # Pre-compute price_map and auc_map once per date, avoiding a second O(n) scan.
    for dt in date_iter:
        raw_signals = daily_raw[dt]
        signals: List[Signal] = []
        price_map: Dict[str, float] = {}   # symbol -> T日收盘价（入场价）
        auc_map: Dict[str, int] = {}       # symbol -> 0=none, 1=limit_up, -1=limit_down
        for raw in raw_signals:
            # raw is already a Signal object from _collect_signals_batched
            price_map[raw.symbol] = raw.entry_price
            auc_map[raw.symbol] = raw.auc_limit
            signals.append(raw)

        # 对每个策略分别计算
        for strat_idx, strat in enumerate(strategies):
            positions = positions_list[strat_idx]
            cash = cash_state_list[strat_idx]  # 延续上日现金
            prev_w = prev_weights_list[strat_idx]
            cache = cache_list[strat_idx]
            trades = trades_list[strat_idx]

            # ── 4a. 卖出昨日持仓 ─────────────────────────────────────
            # 卖出执行于 T日收盘价（与昨日收盘价相同，即 T日收盘 = T-1日收盘 → 实质等于清算昨日持仓）
            sell_cash = 0.0  # 卖出所得现金
            sell_symbols = list(positions.keys())
            for sym in sell_symbols:
                pos = positions[sym]
                entry_p = pos["entry_price"]
                shares = pos["shares"]
                sell_price = price_map.get(sym, entry_p)  # 如果没有该股信号，用入场价
                sell_amount = shares * sell_price
                sell_cash += sell_amount
                trades.append(Trade(
                    date=dt.strftime("%Y-%m-%d"), symbol=sym,
                    action="sell", price=sell_price, shares=shares,
                    amount=sell_amount,
                ))
            # 清空所有持仓
            positions.clear()

            # ── 4b. 策略决策（基于所有信号，不受涨跌停影响预测）────────
            weights = strat.select_positions(signals, prev_w, cache)
            # 权重归一化（仅针对可交易的，正权的）
            pos_weights = {s: w for s, w in weights.items() if w > 0}
            total_w = sum(pos_weights.values())
            if total_w <= 0:
                # 没有可买信号，全仓现金
                target_weights = {}
            else:
                # 可用现金 = 现有现金 + 卖出所得
                alloc_cash = cash + sell_cash
                target_weights = {s: w / total_w * alloc_cash for s, w in pos_weights.items()}

            # ── 4c. 买入新持仓（贪心：能买就买，分配金额不足则跳过）────
            # 按分配金额降序处理（分配金额大的优先买）
            buy_cash = 0.0
            new_weights: Dict[str, float] = {}
            remaining_cash = cash + sell_cash  # 初始可用现金

            for sym, alloc in sorted(target_weights.items(), key=lambda x: x[1], reverse=True):
                auc = auc_map.get(sym, 0)
                price = price_map.get(sym, 0.0)
                if price <= 0 or auc == 1:  # 无价格或涨停：无法买入
                    trades.append(Trade(
                        date=dt.strftime("%Y-%m-%d"), symbol=sym,
                        action="buy", price=price, shares=0,
                        amount=0.0, reason="no_price_or_limit_up"
                    ))
                    continue
                max_possible = int(alloc // price // _LOT_SIZE) * _LOT_SIZE
                if max_possible >= _LOT_SIZE:
                    cost = max_possible * price
                    positions[sym] = {
                        "shares": max_possible,
                        "entry_price": price,
                        "entry_date": dt.strftime("%Y-%m-%d"),
                    }
                    buy_cash += cost
                    remaining_cash -= cost
                    new_weights[sym] = cost / ((cash + sell_cash) or 1.0)
                    trades.append(Trade(
                        date=dt.strftime("%Y-%m-%d"), symbol=sym,
                        action="buy", price=price, shares=max_possible,
                        amount=cost,
                    ))
                else:
                    # 买不起1手（分配金额不够）
                    trades.append(Trade(
                        date=dt.strftime("%Y-%m-%d"), symbol=sym,
                        action="buy", price=price, shares=0,
                        amount=0.0, reason="insufficient_capital"
                    ))

            # ── 4d. 当日结算 ─────────────────────────────────────────
            # 现金 = 上日现金 + 卖出所得 - 买入消耗
            cash = cash + sell_cash - buy_cash
            cash_state_list[strat_idx] = cash
            # 持仓市值 = sum(shares × T日收盘价)
            holdings_val = sum(
                pos["shares"] * price_map.get(sym, pos["entry_price"])
                for sym, pos in positions.items()
            )
            portfolio_val = cash + holdings_val
            # 计算当日收益率（相对昨日组合价值）
            prev_capital = capital_list[strat_idx][-1] if capital_list[strat_idx] else _INITIAL_CAPITAL
            gross = (portfolio_val / prev_capital - 1.0) if prev_capital > 0 else 0.0
            # 换手率 = |买入金额 + 卖出金额| / 2 / 昨日组合价值
            total_trade_val = sell_cash + buy_cash
            turnover = total_trade_val / (2.0 * prev_capital) if prev_capital > 0 else 0.0
            net = gross
            # 行业 HHI
            hhi = _build_industry_hhi({s: w for s, w in new_weights.items()}, industry_map)

            # 记录
            gross_rets_list[strat_idx].append(gross)
            net_rets_list[strat_idx].append(net)
            turnovers_list[strat_idx].append(turnover)
            hhi_list[strat_idx].append(hhi)
            weights_list[strat_idx].append(new_weights)
            capital_list[strat_idx].append(portfolio_val)
            holdings_val_list[strat_idx].append(holdings_val)
            cash_list[strat_idx].append(cash)

            # 策略 on_day_end hook（传入 realized_ret map）
            realized_map = {s.symbol: s.realized_ret for s in signals}
            strat.on_day_end(dt.strftime("%Y-%m-%d"), new_weights, realized_map, cache)

            prev_weights_list[strat_idx] = new_weights

    # ── 5. 组装结果 ─────────────────────────────────────────────────
    results: List[StrategyBacktestResult] = []

    for strat_idx, strat in enumerate(strategies):
        n = len(gross_rets_list[strat_idx])
        idx = dates_sorted[:n]

        gross_series = pd.Series(gross_rets_list[strat_idx], index=idx)
        net_series = pd.Series(net_rets_list[strat_idx], index=idx)
        turnover_series = pd.Series(turnovers_list[strat_idx], index=idx)
        hhi_series = pd.Series(hhi_list[strat_idx], index=idx)
        weights_series = pd.Series(weights_list[strat_idx], index=idx)
        capital_series = pd.Series(capital_list[strat_idx], index=idx)
        holdings_series = pd.Series(holdings_val_list[strat_idx], index=idx)
        cash_series = pd.Series(cash_list[strat_idx], index=idx)

        # 累计组合价值（单位：元）
        cumulative = capital_series
        cumulative_net = capital_series  # 暂不单独处理费用（已包含在 turnover 中）

        final_capital = float(cumulative.iloc[-1]) if len(cumulative) > 0 else _INITIAL_CAPITAL
        total_ret_gross = (final_capital / _INITIAL_CAPITAL - 1.0)
        total_ret_net = total_ret_gross  # 费用已从cash扣除

        ann_ret_gross = _annualize_return(gross_series)
        ann_vol_gross = _annualize_vol(gross_series)
        ann_ret_net = ann_ret_gross
        ann_vol_net = ann_vol_gross

        metrics = {
            "initial_capital": _INITIAL_CAPITAL,
            "final_capital": final_capital,
            "annual_return_gross": ann_ret_gross,
            "annual_vol_gross": ann_vol_gross,
            "sharpe_gross": ann_ret_gross / (ann_vol_gross or np.nan),
            "total_return_gross": total_ret_gross,
            "annual_return_net": ann_ret_net,
            "annual_vol_net": ann_vol_net,
            "sharpe_net": ann_ret_net / (ann_vol_net or np.nan),
            "total_return_net": total_ret_net,
            "max_drawdown_gross": _max_drawdown(capital_series / _INITIAL_CAPITAL),
            "max_drawdown_net": _max_drawdown(capital_series / _INITIAL_CAPITAL),
            "avg_turnover": turnover_series.mean(),
            "median_turnover": turnover_series.median(),
            "avg_industry_hhi": hhi_series.mean() if not hhi_series.isna().all() else float("nan"),
            "num_days": len(gross_series),
        }

        results.append(StrategyBacktestResult(
            strategy_name=strat.name,
            tag=strat.tag,
            daily_returns=gross_series,
            daily_returns_net=net_series,
            cumulative=cumulative,
            cumulative_net=cumulative_net,
            turnover=turnover_series,
            industry_hhi=hhi_series,
            weights=weights_series,
            metrics=metrics,
            trades=trades_list[strat_idx],
            capital=capital_series,
            holdings_value=holdings_series,
            cash_series=cash_series,
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
        signals = []
        for raw in raw_signals:
            # raw is already a Signal object
            signals.append(raw)
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
        tag=strategy.tag,
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
