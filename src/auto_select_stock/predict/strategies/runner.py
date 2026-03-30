"""
回测运行器：一次预测，所有策略共享信号，并行运行。

核心思路：
1. _collect_signals_batched() 一次性收集所有股票所有日期的信号（一次GPU推理）
2. 对每个策略，遍历每日信号调用 select_positions()，获得当日权重
3. 以仓位（股数）为单位追踪资金账户
4. 所有策略共享同一次信号收集，避免重复推理

资金模型（T+1 规则）：
- 起始资金：100,000 RMB
- 最小交易单位：100股（A股规则）
- 涨停股（T+1开盘涨幅≥9.5%）无法买入
- 跌停股（T+1开盘跌幅≤-9.5%）无法卖出
- 买入执行价 = T+1 开盘价
- 卖出执行价 = T+1 收盘价
- T+1 规则：今日买 → 明日才可卖（T日买的股票今日不可卖）
- 资金规则：T日收盘卖出所得 → T+1才可用（冻结一日）
- 持仓跨日保留：不在目标中的才卖出，不强制每日清仓
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
    date: str          # 交易日期（T+1）
    symbol: str        # 股票代码
    action: str        # "buy" 或 "sell"
    price: float       # 成交价格（买入=开盘价，卖出=收盘价）
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
    # frozen_cash_list: List[float] 每个策略的冻结资金（T日收盘卖出 → T+1才可用）
    frozen_cash_list: List[float] = [0.0 for _ in strategies]
    # position_entry_dates_list: 每个策略的持仓买入日期追踪 {symbol: entry_date_str}
    position_entry_dates_list: List[Dict[str, str]] = [{} for _ in strategies]

    for dt in date_iter:
        raw_signals = daily_raw[dt]
        signals: List[Signal] = []
        # price_map: symbol -> T日收盘价（卖出价），auc_map: symbol -> 涨跌停标志
        price_map: Dict[str, float] = {}
        auc_map: Dict[str, int] = {}
        # next_open_map: symbol -> T+1开盘价（买入价）
        next_open_map: Dict[str, float] = {}
        # next_close_map: symbol -> T+1收盘价（卖出价，用于更新持仓）
        next_close_map: Dict[str, float] = {}
        for raw in raw_signals:
            price_map[raw.symbol] = raw.entry_price   # T日收盘：卖出价
            auc_map[raw.symbol] = raw.auc_limit
            next_open_map[raw.symbol] = raw.next_open   # T+1开盘：买入价
            next_close_map[raw.symbol] = raw.next_close # T+1收盘：持仓股市值计算
            signals.append(raw)

        # 对每个策略分别计算
        for strat_idx, strat in enumerate(strategies):
            positions = positions_list[strat_idx]
            cash = cash_state_list[strat_idx]  # 延续上日现金（不含当日卖出所得）
            frozen = frozen_cash_list[strat_idx]  # 当日收盘卖出所得，T+1才可用
            entry_dates = position_entry_dates_list[strat_idx]
            prev_w = prev_weights_list[strat_idx]
            cache = cache_list[strat_idx]
            trades = trades_list[strat_idx]

            # ── 4a. 结算昨日卖出所得 ──────────────────────────────
            # 昨日卖出所得今日解冻
            available_cash = cash + frozen
            sell_cash_today = 0.0  # 今日卖出所得（冻结至明日）

            # ── 4b. 决定目标持仓 ─────────────────────────────────
            weights = strat.select_positions(signals, prev_w, cache)
            pos_weights = {s: w for s, w in weights.items() if w > 0}
            target_syms = set(pos_weights.keys())

            # ── 4c. 卖出不在目标中的持仓（遵守T+1限制）──────────
            # T+1规则：今天买的股票不能今天卖 → 检查 entry_date
            sell_symbols = list(positions.keys())
            for sym in sell_symbols:
                entry_date_str = entry_dates.get(sym)
                # 不能卖：今天买的（entry_date == 今天）
                if entry_date_str == dt.strftime("%Y-%m-%d"):
                    continue
                pos = positions[sym]
                shares = pos["shares"]
                # 卖出价：T日收盘价（price_map里有）
                sell_price = price_map.get(sym, pos["entry_price"])
                sell_amount = shares * sell_price
                sell_cash_today += sell_amount
                trades.append(Trade(
                    date=dt.strftime("%Y-%m-%d"), symbol=sym,
                    action="sell", price=sell_price, shares=shares,
                    amount=sell_amount,
                ))
                # 从持仓中移除
                del positions[sym]
                del entry_dates[sym]

            # ── 4d. 持仓更新（没被卖出的，entry_price更新为T+1收盘）─
            # 对于仍持有的仓位：用 next_close 更新 entry_price（下一日计算收益用）
            # 但分拆股票（auc_limit=2）不更新——next_close 是分拆价，
            # 更新会导致 entry_price 虚增，进而扭曲后续 realized_ret。
            for sym in list(positions.keys()):
                if sym in next_close_map and auc_map.get(sym) != 2:
                    positions[sym]["entry_price"] = next_close_map[sym]

            # ── 4e. 买入新目标持仓（用开盘价，可用现金中支出）────
            # 可用现金 = 昨日结余现金 + 已解冻的昨日卖出所得
            # 今日卖出所得暂时不可用
            buy_cash = 0.0
            new_weights: Dict[str, float] = {}
            remaining_cash = available_cash  # 初始可用现金

            for sym, weight in sorted(pos_weights.items(), key=lambda x: x[1], reverse=True):
                if sym in positions:
                    continue  # 已持有，跳过
                auc = auc_map.get(sym, 0)
                buy_price = next_open_map.get(sym, 0.0)  # 开盘价买入
                if buy_price <= 0 or auc == 1 or auc == 2:  # 无价格或涨停或分拆：无法买入
                    trades.append(Trade(
                        date=dt.strftime("%Y-%m-%d"), symbol=sym,
                        action="buy", price=buy_price, shares=0,
                        amount=0.0, reason="no_price_or_limit_up"
                    ))
                    continue
                # weight 是仓位权重（0~1），转成可用现金中的实际配额
                cash_for_pos = weight * remaining_cash
                max_possible = int(cash_for_pos // buy_price // _LOT_SIZE) * _LOT_SIZE
                if max_possible >= _LOT_SIZE and cash_for_pos >= max_possible * buy_price:
                    cost = max_possible * buy_price
                    positions[sym] = {
                        "shares": max_possible,
                        "entry_price": buy_price,      # 买入价（next_open）
                        "entry_date": dt.strftime("%Y-%m-%d"),  # 买入日期（明天才能卖）
                    }
                    entry_dates[sym] = dt.strftime("%Y-%m-%d")
                    buy_cash += cost
                    remaining_cash -= cost
                    new_weights[sym] = cost / (available_cash or 1.0)
                    trades.append(Trade(
                        date=dt.strftime("%Y-%m-%d"), symbol=sym,
                        action="buy", price=buy_price, shares=max_possible,
                        amount=cost,
                    ))
                else:
                    trades.append(Trade(
                        date=dt.strftime("%Y-%m-%d"), symbol=sym,
                        action="buy", price=buy_price, shares=0,
                        amount=0.0, reason="insufficient_capital"
                    ))

            # ── 4f. 当日结算 ──────────────────────────────────────
            # 现金变化：- 买入消耗 + 卖出所得
            # 今日卖出所得冻结，明日才解冻
            cash = available_cash - buy_cash
            frozen_cash_list[strat_idx] = sell_cash_today
            cash_state_list[strat_idx] = cash
            # DEBUG
            if strat_idx == 0 and len(capital_list[0]) < 5:
                holdings_val_debug = sum(
                    (pos["shares"] * (pos["entry_price"] if pos.get("entry_date") == dt.strftime("%Y-%m-%d") else price_map.get(sym, pos["entry_price"])))
                    for sym, pos in positions.items()
                )
                print(f"  [DEBUG] date={dt.date()} cash={cash:.0f} frozen={frozen:.0f} holdings_val={holdings_val_debug:.0f} "
                      f"portfolio_val={cash+frozen+holdings_val_debug:.0f} prev={capital_list[0][-1] if capital_list[0] else _INITIAL_CAPITAL:.0f} "
                      f"buy_cash={buy_cash:.0f} sell_cash={sell_cash_today:.0f} positions={len(positions)}")
            # 持仓市值计算：
            # - 原有持仓：用 T 日收盘价
            # - 当天新建仓（entry_date == 今天）：用 entry_price（成本价）
            # - 分拆股（auc_limit=2）：用 entry_price（分拆前的价格），避免市值虚增
            holdings_val = 0.0
            today_str = dt.strftime("%Y-%m-%d")
            for sym, pos in positions.items():
                if pos.get("entry_date") == today_str:
                    # 当天新建仓：用买入价（成本）估值
                    holdings_val += pos["shares"] * pos["entry_price"]
                elif auc_map.get(sym) == 2:
                    # 分拆股票：用 entry_price（未分拆价），不用 price_map（分拆后价格）
                    holdings_val += pos["shares"] * pos["entry_price"]
                else:
                    # 原有持仓：用 T 日收盘价
                    holdings_val += pos["shares"] * price_map.get(sym, pos["entry_price"])
            portfolio_val = cash + frozen + holdings_val
            # 计算当日收益率
            prev_capital = capital_list[strat_idx][-1] if capital_list[strat_idx] else _INITIAL_CAPITAL
            gross = (portfolio_val / prev_capital - 1.0) if prev_capital > 0 else 0.0
            # 换手率
            total_trade_val = sell_cash_today + buy_cash
            turnover = total_trade_val / (2.0 * prev_capital) if prev_capital > 0 else 0.0
            net = gross
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

            # 策略 on_day_end hook
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
