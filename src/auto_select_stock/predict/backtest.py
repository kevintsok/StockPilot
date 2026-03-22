import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

from ..config import DATA_DIR
from ..data.storage import list_symbols, load_stock_history
from .data import TECHNICAL_FEATURE_COLUMNS, _load_financial_frame, _merge_price_financial, compute_technical_indicators
from .inference import PricePredictor
from .strategy import build_long_short_portfolio
from .strategies.base import Signal

_A_SHARE_PREFIXES = ("0", "3", "6")  # 沪深主板/中小板/创业板/科创板常用前缀
_EXCLUDE_PREFIXES = ("688",)  # 科创板（如需排除）


@dataclass
class BacktestConfig:
    checkpoint: Path
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    symbols: Optional[Iterable[str]] = None
    top_pct: float = 0.1
    allow_short: bool = False
    cost_bps: float = 0.0  # explicit commission/fee
    slippage_bps: float = 0.0  # implicit price impact
    base_dir: Path = DATA_DIR
    industry_map: Optional[Dict[str, str]] = None  # symbol -> industry name
    eval_batch_size: int = 64
    trading_days: int = 252  # Annualization factor forSharpe/volatility calculations


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    daily_returns_net: pd.Series
    cumulative: pd.Series
    cumulative_net: pd.Series
    turnover: pd.Series
    industry_hhi: pd.Series
    metrics: Dict[str, float]
    trades: List[Dict[str, Any]]  # per-symbol daily actions/contributions for debugging


def _parse_date(value: Optional[str | datetime | pd.Timestamp]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    return pd.to_datetime(value).normalize()


def _annualize_return(daily_ret: pd.Series, trading_days: int = 252) -> float:
    if daily_ret.empty:
        return float("nan")
    total_return = (1 + daily_ret).prod()
    periods = len(daily_ret)
    return total_return ** (trading_days / periods) - 1


def _annualize_vol(daily_ret: pd.Series, trading_days: int = 252) -> float:
    if daily_ret.empty:
        return float("nan")
    return daily_ret.std(ddof=0) * np.sqrt(trading_days)


def _max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return float("nan")
    peak = cumulative.cummax()
    dd = cumulative / peak - 1.0
    return dd.min()


def _window_return(daily_ret: pd.Series, window: int) -> float:
    if daily_ret.empty:
        return float("nan")
    tail = daily_ret.tail(window)
    if tail.empty:
        return float("nan")
    return float((1 + tail).prod() - 1.0)


def _compute_turnover(prev_weights: Dict[str, float], weights: Dict[str, float]) -> float:
    turnover = 0.0
    for sym, w in weights.items():
        turnover += abs(w - prev_weights.get(sym, 0.0))
    for sym, w_prev in prev_weights.items():
        if sym not in weights:
            turnover += abs(w_prev)
    return turnover


def filter_a_share_symbols(symbols: Iterable[str]) -> List[str]:
    """
    Keep default A-share universe: Shanghai/Shenzhen + ChiNext (exclude 北交所等前缀).
    """
    allowed = []
    for sym in symbols:
        if not sym:
            continue
        s = str(sym)
        if s.startswith(_EXCLUDE_PREFIXES):
            continue
        if s.startswith(_A_SHARE_PREFIXES):
            allowed.append(sym)
    return allowed


def _progress(items: Iterable, desc: str, unit: str = "step"):
    seq = list(items)
    total = len(seq)
    if tqdm is None or total == 0:
        for item in seq:
            yield item
        return
    with tqdm(total=total, desc=desc, leave=False, unit=unit) as bar:
        for idx, item in enumerate(seq, 1):
            bar.set_description(f"{desc} {idx}/{total}")
            bar.update(1)
            yield item


def _progress_items(items: Iterable[str], desc: str):
    yield from _progress(items, desc, unit="stock")


def _build_feature_frame(
    symbol: str,
    price_columns: List[str],
    financial_columns: List[str],
    base_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    arr = load_stock_history(symbol, base_dir=base_dir)
    price_df = pd.DataFrame(arr)
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
    price_df.sort_values("date", inplace=True)
    price_features = price_df[price_columns].astype("float32").to_numpy()
    fin_df = _load_financial_frame(symbol, financial_columns, base_dir=base_dir)
    fin_features = _merge_price_financial(price_df, fin_df, financial_columns)
    # Include technical indicators to match training
    tech_df = compute_technical_indicators(price_df)
    tech_features = tech_df[TECHNICAL_FEATURE_COLUMNS].astype("float32").to_numpy()
    features = np.concatenate([price_features, fin_features, tech_features], axis=1)
    opens = price_df["open"].to_numpy(dtype="float32")
    closes = price_df["close"].to_numpy(dtype="float32")
    return price_df[["date"]], features, closes, opens


def _collect_signals_for_symbol(
    symbol: str,
    predictor: PricePredictor,
    cfg: BacktestConfig,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    base_dir: Path,
) -> List[Tuple[pd.Timestamp, float, float]]:
    """
    Collect daily predicted/realized returns for a single symbol using batched inference.
    """
    price_cols = predictor.cfg.price_columns
    fin_cols = predictor.cfg.financial_columns
    dates_df, features, closes, opens = _build_feature_frame(symbol, price_cols, fin_cols, base_dir)
    dates = dates_df["date"].to_numpy()
    seq_len = predictor.cfg.seq_len
    if len(features) < seq_len + 1:
        return []

    if predictor.scaler is None:
        scaler_mean = features[:-1].mean(axis=0)
        scaler_std = features[:-1].std(axis=0) + 1e-6
    else:
        scaler_mean = predictor.scaler["mean"]
        scaler_std = predictor.scaler["std"]

    normed = (features - scaler_mean) / scaler_std
    num_windows = len(normed) - seq_len
    if num_windows <= 0:
        return []

    current_close = closes[seq_len - 1 : -1]
    next_close = closes[seq_len:]
    next_open = opens[seq_len:]
    window_dates = dates[seq_len - 1 : -1]
    next_dates = dates[seq_len:]

    batch_size = max(1, getattr(cfg, "eval_batch_size", 64))
    preds: List[float] = []
    device = predictor.device
    close_idx = predictor.close_idx
    mode = getattr(predictor.cfg, "target_mode", "close")
    with torch.inference_mode():
        for start in range(0, num_windows, batch_size):
            end = min(start + batch_size, num_windows)
            batch_windows = np.stack([normed[i : i + seq_len] for i in range(start, end)])
            batch = torch.tensor(batch_windows, dtype=torch.float32, device=device)
            out = predictor.model(batch)[0]
            last_step = out[:, -1]
            preds.extend(last_step.detach().cpu().numpy().tolist())

    signals: List[Tuple[pd.Timestamp, float, float]] = []
    for idx, pred_last in enumerate(preds):
        dt = window_dates[idx]  # trade date (context end)
        target_dt = pd.Timestamp(next_dates[idx])  # date when return realizes
        if start_date is not None and target_dt < start_date:
            continue
        if end_date is not None and target_dt > end_date:
            break
        cur_c = float(current_close[idx])
        nxt_c = float(next_close[idx])
        nxt_o = float(next_open[idx])
        if mode == "log_return":
            predicted_ret = math.exp(pred_last) - 1.0
        else:
            pred_close = pred_last * scaler_std[close_idx] + scaler_mean[close_idx]
            predicted_ret = float(pred_close / cur_c - 1.0)
        realized_ret = float(nxt_c / max(cur_c, 1e-6) - 1.0)
        signals.append((pd.Timestamp(target_dt), predicted_ret, realized_ret))
    return signals


def _collect_signals_batched(
    symbols: Iterable[str],
    predictor: PricePredictor,
    cfg: BacktestConfig,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    show_progress: bool,
    horizon: str = "1d",
) -> Dict[pd.Timestamp, List[Signal]]:
    """
    Collect daily signals across symbols using cross-stock batching to better utilize the device.

    Args:
        horizon: prediction horizon for ranking (e.g. "1d", "5d"). Default "1d".

    Returns:
        Dictionary mapping each target date to a list of Signal objects, each containing
        the symbol, predicted return, realized return, industry, multi-horizon predictions,
        entry price, and auction limit flag.
    """
    batch_size = max(1, getattr(cfg, "eval_batch_size", 64))
    device = predictor.device
    close_idx = predictor.close_idx
    mode = getattr(predictor.cfg, "target_mode", "close")
    industry_map = cfg.industry_map or {}
    daily_signals: Dict[pd.Timestamp, List[Signal]] = {}

    # Determine which horizon index to use (for the primary pred_ret stored in tuple)
    h_int = int(horizon.replace("d", ""))
    if h_int in predictor.horizons:
        h_idx = predictor.horizons.index(h_int)
    else:
        h_idx = 0  # fallback to 1d

    pending_windows: List[np.ndarray] = []
    pending_meta: List[Tuple[str, int]] = []  # (symbol, window_idx)
    remaining_windows: Dict[str, int] = {}
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    done_symbols: set[str] = set()

    def _flush() -> None:
        if not pending_windows:
            return
        batch = torch.tensor(np.stack(pending_windows), dtype=torch.float32, device=device)
        with torch.inference_mode():
            out = predictor.model(batch)
            if len(out) == 4:
                reg_all = out[2]
                last_step = reg_all[h_idx, :, -1]
                all_preds = reg_all[:, :, -1].detach().cpu().numpy()
                horizons_list = predictor.horizons
            else:
                last_step = out[0][:, -1]
                all_preds = last_step.detach().cpu().numpy().reshape(1, -1)
                horizons_list = [1]
            preds = last_step.detach().cpu().numpy().tolist()
        for batch_i, (pred_last, (sym, idx)) in enumerate(zip(preds, pending_meta)):
            info = cache[sym]
            entry_c = float(info["entry_close"][idx])   # T's close (entry price)
            exit_c = float(info["exit_close"][idx])     # T+1's close (exit price)
            entry_o = float(info["entry_open"][idx])    # T's open (for auction check)
            prev_c = float(info["prev_close"][idx])     # T-1's close (for auction check)
            # Auction limit: A-share limit is ±10% from previous close
            open_ret = entry_o / prev_c - 1.0 if prev_c > 0 else 0.0
            auc_limit = 0  # 0=none, 1=limit_up (can't buy), -1=limit_down (can't sell)
            if open_ret >= 0.095:      # opened at limit-up (涨停)
                auc_limit = 1
            elif open_ret <= -0.095:   # opened at limit-down (跌停)
                auc_limit = -1
            target_dt = pd.Timestamp(info["next_dates"][idx])  # T+1
            scaler_mean = info["scaler_mean"]
            scaler_std = info["scaler_std"]
            if mode == "log_return":
                predicted_ret = math.exp(pred_last) - 1.0
                predicted_rets = {}
                for h_j, h in enumerate(horizons_list):
                    p = all_preds[h_j, batch_i]
                    predicted_rets[f"{h}d"] = math.exp(p) - 1.0
            else:
                pred_close = pred_last * scaler_std[close_idx] + scaler_mean[close_idx]
                predicted_ret = float(pred_close / entry_c - 1.0)
                predicted_rets = {}
                for h_j, h in enumerate(horizons_list):
                    p = all_preds[h_j, batch_i]
                    pc = p * scaler_std[close_idx] + scaler_mean[close_idx]
                    predicted_rets[f"{h}d"] = float(pc / entry_c - 1.0)
            # realized_ret: entry at T's close, exit at T+1's close (proper 1-day return)
            realized_ret = float(exit_c / max(entry_c, 1e-6) - 1.0)
            # Signal tuple: (symbol, predicted_ret, realized_ret, industry, predicted_rets, entry_price, auc_limit)
            daily_signals.setdefault(target_dt, []).append(
                Signal(
                    symbol=sym,
                    predicted_ret=predicted_ret,
                    realized_ret=realized_ret,
                    industry=industry_map.get(sym),
                    predicted_rets=predicted_rets,
                    entry_price=entry_c,
                    auc_limit=auc_limit,
                )
            )
            remaining_windows[sym] = remaining_windows.get(sym, 0) - 1
            if remaining_windows[sym] <= 0 and sym in done_symbols:
                cache.pop(sym, None)
        pending_windows.clear()
        pending_meta.clear()

    symbols_iter = _progress_items(symbols, "Collect signals") if show_progress else symbols
    for sym in symbols_iter:
        dates_df, features, closes, opens = _build_feature_frame(
            sym, predictor.cfg.price_columns, predictor.cfg.financial_columns, cfg.base_dir
        )
        dates = dates_df["date"].to_numpy()
        seq_len = predictor.cfg.seq_len
        if len(features) < seq_len + 1:
            continue
        if predictor.scaler is None:
            scaler_mean = features[:-1].mean(axis=0)
            scaler_std = features[:-1].std(axis=0) + 1e-6
        else:
            scaler_mean = predictor.scaler["mean"]
            scaler_std = predictor.scaler["std"]

        normed = (features - scaler_mean) / scaler_std
        num_windows = len(normed) - seq_len
        if num_windows <= 0:
            continue

        # FIX: entry price = T's close (not T-1's close).
        # context ends at T-1 (closes[seq_len-1+idx]), model predicts close[T]/close[T-1]-1.
        # We execute at T's close, hold to T+1 close: realized_ret = close[T+1]/close[T]-1.
        entry_close = closes[seq_len:-1]    # close at T, shape (num_windows,)
        exit_close = closes[seq_len + 1:]    # close at T+1, shape (num_windows,)
        prev_close = closes[seq_len - 1:-1]  # close at T-1, shape (num_windows,) — for auction check
        entry_open = opens[seq_len:-1]      # open at T (for auction limit check)
        window_dates = dates[seq_len - 1 : -1]   # T date (= context end = prediction date)
        next_dates = dates[seq_len + 1:]          # T+1 date (= target date = exit date)

        cache[sym] = {
            "entry_close": entry_close,
            "exit_close": exit_close,
            "prev_close": prev_close,
            "entry_open": entry_open,
            "dates": window_dates,
            "next_dates": next_dates,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
        }
        valid_windows = 0
        remaining_windows[sym] = remaining_windows.get(sym, 0)
        for idx in range(num_windows):
            dt = window_dates[idx]  # trade date (context end)
            target_dt = pd.Timestamp(next_dates[idx])
            if start_date is not None and target_dt < start_date:
                continue
            if end_date is not None and target_dt > end_date:
                break
            pending_windows.append(normed[idx : idx + seq_len])
            pending_meta.append((sym, idx))
            valid_windows += 1
            if len(pending_windows) >= batch_size:
                _flush()
            remaining_windows[sym] = remaining_windows.get(sym, 0) + 1
        if valid_windows == 0:
            cache.pop(sym, None)
        done_symbols.add(sym)
        if remaining_windows.get(sym, 0) <= 0:
            cache.pop(sym, None)
    _flush()
    return daily_signals


def run_backtest(
    cfg: BacktestConfig, predictor: Optional[PricePredictor] = None, show_progress: bool = True, horizon: str = "1d"
) -> BacktestResult:
    start_date = _parse_date(cfg.start_date)
    end_date = _parse_date(cfg.end_date)
    predictor = predictor or PricePredictor(cfg.checkpoint)
    if cfg.symbols:
        symbols = list(cfg.symbols)
    else:
        symbols = filter_a_share_symbols(list_symbols(base_dir=cfg.base_dir))
    if not symbols:
        raise RuntimeError("No symbols available for backtest after filtering to 沪深/创业板默认范围.")

    # Collect daily signals per symbol
    daily_signals = _collect_signals_batched(symbols, predictor, cfg, start_date, end_date, show_progress, horizon=horizon)

    dates_sorted = sorted(daily_signals.keys())
    prev_weights: Dict[str, float] = {}
    daily_rets: List[Tuple[pd.Timestamp, float, float, float, float]] = []  # date, gross, net, turnover, hhi

    cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10000.0
    date_iter = _progress(dates_sorted, "Rebalance by day", unit="day") if show_progress else dates_sorted
    for dt in date_iter:
        # daily_signals already contains Signal objects from _collect_signals_batched
        raw_list = daily_signals[dt]
        sigs = raw_list
        gross, net, turnover, weights, hhi = build_long_short_portfolio(
            signals=sigs,
            top_pct=cfg.top_pct,
            allow_short=cfg.allow_short,
            prev_weights=prev_weights,
            industry_map=cfg.industry_map,
            cost_rate=cost_rate,
        )
        daily_rets.append((dt, gross, net, turnover, hhi))
        prev_weights = weights

    if not daily_rets:
        raise RuntimeError("No signals generated in the specified backtest window.")

    df = pd.DataFrame(daily_rets, columns=["date", "gross_ret", "net_ret", "turnover", "industry_hhi"]).set_index("date").sort_index()
    cumulative = (1 + df["gross_ret"]).cumprod()
    cumulative_net = (1 + df["net_ret"]).cumprod()

    metrics = {
        "annual_return_gross": _annualize_return(df["gross_ret"], cfg.trading_days),
        "annual_vol_gross": _annualize_vol(df["gross_ret"], cfg.trading_days),
        "sharpe_gross": _annualize_return(df["gross_ret"], cfg.trading_days) / (_annualize_vol(df["gross_ret"], cfg.trading_days) or np.nan),
        "total_return_gross": float(cumulative.iloc[-1] - 1.0),
        "annual_return_net": _annualize_return(df["net_ret"], cfg.trading_days),
        "annual_vol_net": _annualize_vol(df["net_ret"], cfg.trading_days),
        "sharpe_net": _annualize_return(df["net_ret"], cfg.trading_days) / (_annualize_vol(df["net_ret"], cfg.trading_days) or np.nan),
        "total_return_net": float(cumulative_net.iloc[-1] - 1.0),
        "max_drawdown_gross": _max_drawdown(cumulative),
        "max_drawdown_net": _max_drawdown(cumulative_net),
        "avg_turnover": df["turnover"].mean(),
        "median_turnover": df["turnover"].median(),
        "avg_industry_hhi": df["industry_hhi"].mean(skipna=True),
        "ret_gross_5d": _window_return(df["gross_ret"], 5),
        "ret_gross_10d": _window_return(df["gross_ret"], 10),
        "ret_gross_20d": _window_return(df["gross_ret"], 20),
        "ret_net_5d": _window_return(df["net_ret"], 5),
        "ret_net_10d": _window_return(df["net_ret"], 10),
        "ret_net_20d": _window_return(df["net_ret"], 20),
    }

    return BacktestResult(
        daily_returns=df["gross_ret"],
        daily_returns_net=df["net_ret"],
        cumulative=cumulative,
        cumulative_net=cumulative_net,
        turnover=df["turnover"],
        industry_hhi=df["industry_hhi"],
        metrics=metrics,
        trades=[],
    )


def run_backtest_for_symbol(
    cfg: BacktestConfig, symbol: str, predictor: Optional[PricePredictor] = None, show_progress: bool = True, horizon: str = "1d"
) -> Tuple[str, Dict[str, float], Optional[str]]:
    """
    Convenience wrapper to run a single-symbol backtest and return metrics or an error message.
    """
    try:
        single_cfg = replace(cfg, symbols=[symbol])
        result = run_backtest(single_cfg, predictor=predictor, show_progress=show_progress, horizon=horizon)
        return symbol, result.metrics, None
    except Exception as exc:  # noqa: BLE001
        return symbol, {}, str(exc)


def run_topk_strategy(
    cfg: BacktestConfig,
    top_k: int,
    predictor: Optional[PricePredictor] = None,
    show_progress: bool = True,
    horizon: str = "1d",
) -> BacktestResult:
    """
    Daily workflow:
    - Batch infer predicted returns for all symbols.
    - Sort by predicted return, take top K positives.
    - Allocate weights proportional to predicted return (normalized); zero weight for predicted-down names.
    - Sell any previously held names not in today's top K (or predicted <= 0).
    """
    start_date = _parse_date(cfg.start_date)
    end_date = _parse_date(cfg.end_date)
    predictor = predictor or PricePredictor(cfg.checkpoint)
    if cfg.symbols:
        symbols = list(cfg.symbols)
    else:
        symbols = filter_a_share_symbols(list_symbols(base_dir=cfg.base_dir))
    if not symbols:
        raise RuntimeError("No symbols available for strategy backtest after filtering to 沪深/创业板默认范围.")

    daily_signals = _collect_signals_batched(symbols, predictor, cfg, start_date, end_date, show_progress=show_progress, horizon=horizon)

    dates_sorted = sorted(daily_signals.keys())
    prev_weights: Dict[str, float] = {}
    daily_rets: List[Tuple[pd.Timestamp, float, float, float, float]] = []
    trade_logs: List[Dict[str, Any]] = []
    cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10000.0
    cumulative_gross = 1.0
    cumulative_net = 1.0

    date_iter = _progress(dates_sorted, "Rebalance by day", unit="day") if show_progress else dates_sorted
    for dt in date_iter:
        sigs = sorted(daily_signals[dt], key=lambda x: x[1], reverse=True)
        top = [s for s in sigs if s[1] > 0][: max(1, top_k)]
        total_pred = sum(s[1] for s in top)
        weights: Dict[str, float] = {}
        if total_pred > 0:
            for sym, pred_ret, _, _ in top:
                weights[sym] = pred_ret / total_pred
        gross = 0.0
        for sym, _, realized_ret, _ in top:
            w = weights.get(sym, 0.0)
            gross += w * realized_ret
        turnover = _compute_turnover(prev_weights, weights)
        net = gross - turnover * cost_rate
        cumulative_gross *= 1.0 + gross
        cumulative_net *= 1.0 + net
        daily_rets.append((dt, gross, net, turnover, float("nan")))

        # Build pred/real maps lazily for O(k) symbols instead of O(n) all signals
        union_syms = set(prev_weights.keys()) | set(weights.keys())
        pred_map = {}
        real_map = {}
        for sym, pred, real, *_ in sigs:
            if sym in union_syms:
                pred_map[sym] = pred
                real_map[sym] = real
        for sym in sorted(union_syms):
            wb = prev_weights.get(sym, 0.0)
            wa = weights.get(sym, 0.0)
            if wb == 0.0 and wa == 0.0:
                continue
            if wb == 0.0 and wa > 0.0:
                action = "buy"
            elif wb > 0.0 and wa == 0.0:
                action = "sell"
            else:
                action = "hold"
            predicted_ret = pred_map.get(sym)
            realized_ret = real_map.get(sym)
            contribution = wa * realized_ret if realized_ret is not None else None
            trade_logs.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "action": action,
                    "weight_before": wb,
                    "weight_after": wa,
                    "predicted_ret": predicted_ret,
                    "realized_ret": realized_ret,
                    "contribution": contribution,
                    "daily_gross": gross,
                    "daily_net": net,
                    "daily_turnover": turnover,
                    "cumulative": cumulative_gross,
                    "cumulative_net": cumulative_net,
                }
            )
        prev_weights = weights

    if not daily_rets:
        raise RuntimeError("No signals generated in the specified backtest window.")

    df = pd.DataFrame(daily_rets, columns=["date", "gross_ret", "net_ret", "turnover", "industry_hhi"]).set_index("date").sort_index()
    cumulative = (1 + df["gross_ret"]).cumprod()
    cumulative_net = (1 + df["net_ret"]).cumprod()

    metrics = {
        "annual_return_gross": _annualize_return(df["gross_ret"], cfg.trading_days),
        "annual_vol_gross": _annualize_vol(df["gross_ret"], cfg.trading_days),
        "sharpe_gross": _annualize_return(df["gross_ret"], cfg.trading_days) / (_annualize_vol(df["gross_ret"], cfg.trading_days) or np.nan),
        "total_return_gross": float(cumulative.iloc[-1] - 1.0),
        "annual_return_net": _annualize_return(df["net_ret"], cfg.trading_days),
        "annual_vol_net": _annualize_vol(df["net_ret"], cfg.trading_days),
        "sharpe_net": _annualize_return(df["net_ret"], cfg.trading_days) / (_annualize_vol(df["net_ret"], cfg.trading_days) or np.nan),
        "total_return_net": float(cumulative_net.iloc[-1] - 1.0),
        "max_drawdown_gross": _max_drawdown(cumulative),
        "max_drawdown_net": _max_drawdown(cumulative_net),
        "avg_turnover": df["turnover"].mean(),
        "median_turnover": df["turnover"].median(),
        "avg_industry_hhi": df["industry_hhi"].mean(skipna=True),
        "ret_gross_5d": _window_return(df["gross_ret"], 5),
        "ret_gross_10d": _window_return(df["gross_ret"], 10),
        "ret_gross_20d": _window_return(df["gross_ret"], 20),
        "ret_net_5d": _window_return(df["net_ret"], 5),
        "ret_net_10d": _window_return(df["net_ret"], 10),
        "ret_net_20d": _window_return(df["net_ret"], 20),
    }

    return BacktestResult(
        daily_returns=df["gross_ret"],
        daily_returns_net=df["net_ret"],
        cumulative=cumulative,
        cumulative_net=cumulative_net,
        turnover=df["turnover"],
        industry_hhi=df["industry_hhi"],
        metrics=metrics,
        trades=trade_logs,
    )


__all__ = ["BacktestConfig", "BacktestResult", "run_backtest", "run_backtest_for_symbol", "run_topk_strategy", "filter_a_share_symbols", "_collect_signals_batched"]
