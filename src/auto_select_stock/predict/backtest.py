from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import DATA_DIR
from ..storage import list_symbols, load_stock_history
from .data import PRICE_FEATURE_COLUMNS, _load_financial_frame, _merge_price_financial
from .inference import PricePredictor
from .strategy import build_long_short_portfolio

_TRADING_DAYS = 252


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


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    daily_returns_net: pd.Series
    cumulative: pd.Series
    cumulative_net: pd.Series
    turnover: pd.Series
    industry_hhi: pd.Series
    metrics: Dict[str, float]


def _parse_date(value: Optional[str | datetime | pd.Timestamp]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    return pd.to_datetime(value).normalize()


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


def _build_feature_frame(
    symbol: str,
    price_columns: List[str],
    financial_columns: List[str],
    base_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray]:
    arr = load_stock_history(symbol, base_dir=base_dir)
    price_df = pd.DataFrame(arr)
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
    price_df.sort_values("date", inplace=True)
    price_features = price_df[price_columns].astype("float32").to_numpy()
    fin_df = _load_financial_frame(symbol, financial_columns, base_dir=base_dir)
    fin_features = _merge_price_financial(price_df, fin_df, financial_columns)
    features = np.concatenate([price_features, fin_features], axis=1)
    closes = price_df["close"].to_numpy(dtype="float32")
    return price_df[["date"]], features, closes


def _collect_signals_for_symbol(
    symbol: str,
    predictor: PricePredictor,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    base_dir: Path,
) -> List[Tuple[pd.Timestamp, float, float]]:
    price_cols = predictor.cfg.price_columns
    fin_cols = predictor.cfg.financial_columns
    dates_df, features, closes = _build_feature_frame(symbol, price_cols, fin_cols, base_dir)
    dates = dates_df["date"].to_numpy()
    if len(features) < predictor.cfg.seq_len + 1:
        return []

    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    if predictor.scaler is None:
        scaler_mean = features[:-1].mean(axis=0)
        scaler_std = features[:-1].std(axis=0) + 1e-6
    else:
        scaler_mean = predictor.scaler["mean"]
        scaler_std = predictor.scaler["std"]
    normed = (features - scaler_mean) / scaler_std

    x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(predictor.device)
    with torch.no_grad():
        pred_seq = predictor.model(x)[0].cpu().numpy()
    close_idx = predictor.close_idx

    signals: List[Tuple[pd.Timestamp, float, float]] = []
    for i in range(len(pred_seq) - 1):
        if i + 1 < predictor.cfg.seq_len:
            continue  # ensure至少有足够历史长度
        dt = dates[i]
        if start_date is not None and dt < start_date:
            continue
        if end_date is not None and dt > end_date:
            break
        pred_close = pred_seq[i] * scaler_std[close_idx] + scaler_mean[close_idx]
        predicted_ret = float(pred_close / closes[i] - 1.0)
        realized_ret = float(closes[i + 1] / closes[i] - 1.0)
        signals.append((pd.Timestamp(dt), predicted_ret, realized_ret))
    return signals


def run_backtest(cfg: BacktestConfig) -> BacktestResult:
    start_date = _parse_date(cfg.start_date)
    end_date = _parse_date(cfg.end_date)
    predictor = PricePredictor(cfg.checkpoint)
    symbols = list(cfg.symbols) if cfg.symbols else list_symbols(base_dir=cfg.base_dir)

    # Collect daily signals per symbol
    daily_signals: Dict[pd.Timestamp, List[Tuple[str, float, float, Optional[str]]]] = {}
    for sym in symbols:
        sym_signals = _collect_signals_for_symbol(sym, predictor, start_date, end_date, cfg.base_dir)
        for dt, pred_ret, realized_ret in sym_signals:
            daily_signals.setdefault(dt, []).append((sym, pred_ret, realized_ret, cfg.industry_map.get(sym) if cfg.industry_map else None))

    dates_sorted = sorted(daily_signals.keys())
    prev_weights: Dict[str, float] = {}
    daily_rets: List[Tuple[pd.Timestamp, float, float, float, float]] = []  # date, gross, net, turnover, hhi

    cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10000.0
    for dt in dates_sorted:
        gross, net, turnover, weights, hhi = build_long_short_portfolio(
            signals=daily_signals[dt],
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
        "annual_return_gross": _annualize_return(df["gross_ret"]),
        "annual_vol_gross": _annualize_vol(df["gross_ret"]),
        "sharpe_gross": _annualize_return(df["gross_ret"]) / (_annualize_vol(df["gross_ret"]) or np.nan),
        "annual_return_net": _annualize_return(df["net_ret"]),
        "annual_vol_net": _annualize_vol(df["net_ret"]),
        "sharpe_net": _annualize_return(df["net_ret"]) / (_annualize_vol(df["net_ret"]) or np.nan),
        "max_drawdown_gross": _max_drawdown(cumulative),
        "max_drawdown_net": _max_drawdown(cumulative_net),
        "avg_turnover": df["turnover"].mean(),
        "median_turnover": df["turnover"].median(),
        "avg_industry_hhi": df["industry_hhi"].mean(skipna=True),
    }

    return BacktestResult(
        daily_returns=df["gross_ret"],
        daily_returns_net=df["net_ret"],
        cumulative=cumulative,
        cumulative_net=cumulative_net,
        turnover=df["turnover"],
        industry_hhi=df["industry_hhi"],
        metrics=metrics,
    )


__all__ = ["BacktestConfig", "BacktestResult", "run_backtest"]
