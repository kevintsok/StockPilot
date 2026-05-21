#!/usr/bin/env python3
"""
Market Regime Detection and Regime-Based Position Scaling.

Detects market regime (Bull/Bear/Neutral/HighVol) from market-wide indicators
and applies dynamic position scaling to BC's output.

Regime Detection Features:
1. Realized Volatility: 20-day rolling std of market returns
2. Market Breadth: % of stocks above their 20-day MA
3. Trend Strength: slope of market portfolio returns
4. Mean Reversion: short-term vs medium-term return differential

Position Scaling:
- Bull + Low Vol: 1.0x (full position)
- Bull + High Vol: 0.7x (reduce slightly)
- Neutral: 0.5x (reduce exposure)
- Bear + High Vol: 0.25x (minimal exposure, ready to exit)
"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from auto_select_stock.config import DATA_DIR
from auto_select_stock.data.storage import list_symbols, load_stock_history
from auto_select_stock.predict.data import compute_technical_indicators, PRICE_FEATURE_COLUMNS
from .bc_pretrain_trainer import BCNetwork


class MarketRegimeDetector:
    """
    Detects market regime based on multiple indicators.
    """

    def __init__(
        self,
        vol_window: int = 20,
        ma_short: int = 20,
        ma_long: int = 60,
        vol_high_threshold: float = 0.02,
        vol_low_threshold: float = 0.01,
    ):
        self.vol_window = vol_window
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.vol_high_threshold = vol_high_threshold  # Daily vol > 2% = high vol
        self.vol_low_threshold = vol_low_threshold    # Daily vol < 1% = low vol

    def compute_market_features(
        self,
        all_sequences: List[np.ndarray],
        current_t: int,
    ) -> dict:
        """
        Compute market-wide features at time t across all stocks.
        Supports both 1D (close only) and 2D (full price array) sequences.
        """
        returns = []
        above_ma20_count = 0
        above_ma60_count = 0
        total_stocks = 0

        for seq in all_sequences:
            if len(seq) <= current_t or current_t < self.ma_long:
                continue

            total_stocks += 1

            # Handle both 1D (close only) and 2D (full features) arrays
            is_1d = seq.ndim == 1
            close_idx = 0 if is_1d else 3

            # Compute returns over lookback window
            if current_t >= self.vol_window:
                price_now = seq[current_t] if is_1d else seq[current_t, close_idx]
                price_start = seq[current_t - self.vol_window] if is_1d else seq[current_t - self.vol_window, close_idx]
                if price_start > 0:
                    ret = (price_now - price_start) / price_start
                    returns.append(ret)

            # Check if above MAs
            if current_t >= self.ma_short:
                ma20 = np.mean(seq[current_t - self.ma_short + 1:current_t + 1] if is_1d else seq[current_t - self.ma_short + 1:current_t + 1, close_idx])
                current_price = seq[current_t] if is_1d else seq[current_t, close_idx]
                if current_price > ma20:
                    above_ma20_count += 1

            if current_t >= self.ma_long:
                ma60 = np.mean(seq[current_t - self.ma_long + 1:current_t + 1] if is_1d else seq[current_t - self.ma_long + 1:current_t + 1, close_idx])
                current_price = seq[current_t] if is_1d else seq[current_t, close_idx]
                if current_price > ma60:
                    above_ma60_count += 1

        features = {
            'total_stocks': total_stocks,
            'returns': returns,
        }

        if len(returns) >= 10:
            # Realized volatility (annualized)
            features['realized_vol'] = np.std(returns) * np.sqrt(252)

            # Market breadth
            features['breadth_20'] = above_ma20_count / max(total_stocks, 1)
            features['breadth_60'] = above_ma60_count / max(total_stocks, 1)

            # Trend: short-term vs long-term momentum
            if current_t >= 5:
                ret_5d = []
                for seq in all_sequences:
                    if len(seq) > current_t >= 5:
                        p_now = seq[current_t, 3]
                        p_5d = seq[current_t - 5, 3]
                        if p_5d > 0:
                            ret_5d.append((p_now - p_5d) / p_5d)
                features['ret_5d_mean'] = np.mean(ret_5d) if ret_5d else 0.0
            else:
                features['ret_5d_mean'] = 0.0

        else:
            features['realized_vol'] = 0.02  # Default moderate vol
            features['breadth_20'] = 0.5
            features['breadth_60'] = 0.5
            features['ret_5d_mean'] = 0.0

        return features

    def classify_regime(self, features: dict) -> Tuple[str, float]:
        """
        Classify market regime and return (regime_name, position_multiplier).

        Binary filter approach: multipliers only apply in adverse regimes.
        Bull/Neutral markets: keep full exposure
        Bear markets: reduce significantly or skip

        Regime classification:
        - Bull: breadth_60 > 0.6, positive 5d momentum
        - Bear: breadth_60 < 0.4, negative 5d momentum
        - HighVol: realized_vol > vol_high_threshold
        - Neutral: everything else

        Position multiplier (binary filter):
        - Bull: 1.0
        - Bull_HighVol: 1.0 (bull momentum overrides high vol)
        - Neutral: 1.0
        - Neutral_HighVol: 0.9
        - Bear: 0.3 (reduce significantly in bear)
        - Bear_HighVol: 0.1 (almost skip in bear + high vol)
        """
        vol = features.get('realized_vol', 0.02)
        breadth_60 = features.get('breadth_60', 0.5)
        ret_5d = features.get('ret_5d_mean', 0.0)

        is_high_vol = vol > self.vol_high_threshold

        # Bull vs Bear classification
        is_bull = breadth_60 > 0.6 and ret_5d > 0
        is_bear = breadth_60 < 0.4 and ret_5d < 0

        if is_bull:
            regime = "Bull_HighVol" if is_high_vol else "Bull"
            multiplier = 1.0
        elif is_bear:
            if is_high_vol:
                regime = "Bear_HighVol"
                multiplier = 0.1  # Almost skip
            else:
                regime = "Bear"
                multiplier = 0.3  # Reduce
        else:
            if is_high_vol:
                regime = "Neutral_HighVol"
                multiplier = 0.9
            else:
                regime = "Neutral"
                multiplier = 1.0

        return regime, multiplier


def compute_state(seq, t, position=0.0, entry_price=0.0):
    """Compute BC state vector."""
    recent = seq[max(0, t - 20):t + 1, :]

    if len(recent) >= 20:
        ma20 = np.mean(recent[-20:, 3])
        price_level = recent[-1, 3] / ma20 if ma20 > 0 else 1.0
    else:
        price_level = 1.0

    if len(recent) >= 2:
        ret_1d = np.log(recent[-1, 3] / recent[-2, 3]) if recent[-2, 3] > 0 else 0.0
    else:
        ret_1d = 0.0

    if len(recent) >= 5:
        ret_5d = np.log(recent[-1, 3] / recent[-5, 3]) if recent[-5, 3] > 0 else 0.0
    else:
        ret_5d = 0.0

    unrealized_pnl = 0.0
    state = np.array([price_level - 1.0, ret_1d, ret_5d, position / 0.15, unrealized_pnl, 0.0], dtype=np.float32)
    return state


def load_stock_sequences_for_regime(
    symbols: List[str],
    start_date: str,
    end_date: str,
    min_len: int = 200,
) -> List[np.ndarray]:
    """Load stock sequences for regime detection."""
    sequences = []
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    for sym in tqdm(symbols, desc="Loading data"):
        try:
            arr = load_stock_history(sym, base_dir=DATA_DIR)
            if arr is None or len(arr) < min_len:
                continue

            df = pd.DataFrame(arr)
            df["date"] = pd.to_datetime(df["date"]).dt.floor("D").astype("datetime64[ns]")
            df.sort_values("date", inplace=True)
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

            if len(df) < min_len:
                continue

            # Only keep close prices for regime detection
            arr_out = df["close"].values.astype(np.float32)
            sequences.append(arr_out)
        except Exception:
            continue

    return sequences


def backtest_with_regime(
    model: BCNetwork,
    regime_sequences: List[np.ndarray],  # Market-wide sequences for regime
    test_sequences: List[np.ndarray],   # Individual stock sequences
    device: torch.device,
    trade_threshold: float = 0.05,
) -> dict:
    """
    Backtest BC with regime-based position scaling.
    Computes market-wide regime from ALL regime stocks at each timestep.
    """
    model.eval()

    commission = 0.0003
    stamp_tax = 0.001
    slippage = 0.0001

    regime_detector = MarketRegimeDetector()

    portfolio_values = []
    all_actions = []
    trade_count = 0
    regime_changes = 0
    prev_regime = None

    regime_counts = {}

    for seq_idx, seq in enumerate(tqdm(test_sequences, desc="Backtesting")):
        if len(seq) < 120:
            continue

        portfolio = 1.0
        position = 0.0
        entry_price = 0.0
        prev_position = 0.0

        for t in range(60, len(seq) - 1):
            # Get market regime at time t using ALL regime stocks
            # Extract prices at time t from all regime stocks that have data
            market_prices_at_t = []
            for regime_seq in regime_sequences:
                if len(regime_seq) > t:
                    market_prices_at_t.append(regime_seq[t])

            if len(market_prices_at_t) >= 10:
                # Build synthetic market sequence for this timestep
                # Use 2D format (seq_len, 4) with close at index 3
                # For regime detection, we need historical context
                # Use last vol_window prices from each stock
                vol_window = regime_detector.vol_window
                ma_long = regime_detector.ma_long

                all_histories = []
                for regime_seq in regime_sequences:
                    lookback = min(t + 1, vol_window + ma_long)
                    if lookback > 0 and len(regime_seq) >= lookback:
                        hist = regime_seq[t - lookback + 1:t + 1]
                        # Pad to vol_window + ma_long
                        if len(hist) < vol_window + ma_long:
                            pad_width = vol_window + ma_long - len(hist)
                            hist = np.pad(hist, (pad_width, 0), mode='edge')
                        all_histories.append(hist)

                if all_histories:
                    # Compute market features using the ensemble
                    features = _compute_market_features_from_histories(
                        all_histories, vol_window, ma_long
                    )
                    regime, regime_multiplier = regime_detector.classify_regime(features)
                else:
                    regime, regime_multiplier = "Neutral", 0.6
            else:
                regime, regime_multiplier = "Neutral", 0.6

            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            if regime != prev_regime:
                regime_changes += 1
                prev_regime = regime

            # Get BC state and prediction
            state = compute_state(seq, t, position, entry_price)
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(state_t).item()

            # Apply regime multiplier to position
            adjusted_action = action * regime_multiplier

            current_price = seq[t, 3]
            next_price = seq[t + 1, 3]

            # Trading with adjusted action
            is_new = (adjusted_action > trade_threshold and position < 0.01)
            is_close = (adjusted_action < trade_threshold and position > 0.01)

            if is_new:
                cost = adjusted_action * (1.0 + commission + slippage)
                portfolio *= (1.0 - cost)
                position = adjusted_action
                entry_price = current_price
                trade_count += 1
            elif is_close:
                rev = position * (1.0 - commission - stamp_tax - slippage)
                portfolio *= (1.0 + rev)
                position = 0.0
                entry_price = 0.0
            elif position > 0.01:
                delta = abs(adjusted_action - position)
                if delta > 0.001:
                    cost = delta * (commission + slippage)
                    portfolio *= (1.0 - cost)
                position = adjusted_action

            if position > 0.01:
                ret = (next_price - current_price) / current_price if current_price > 0 else 0.0
                portfolio *= (1.0 + ret * position)

        portfolio_values.append(portfolio)
        all_actions.append(position)

    returns = [v - 1.0 for v in portfolio_values]

    print(f"\n{'='*50}")
    print("BC + Market Regime Backtest Results")
    print(f"{'='*50}")
    print(f"Total Return:     {np.mean(returns)*100:.2f}%")
    print(f"Std of Return:    {np.std(returns)*100:.2f}%")
    print(f"Sharpe Ratio:    {np.mean(returns)/(np.std(returns)+1e-8)*np.sqrt(252):.4f}")
    print(f"Trade Count:     {trade_count}")
    print(f"Avg Position:    {np.mean([a for a in all_actions if a > 0.01]):.4f}")
    print(f"Regime Changes:  {regime_changes}")
    print(f"\nRegime Distribution:")
    for r, c in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c} ({c/sum(regime_counts.values())*100:.1f}%)")

    return {
        "return": np.mean(returns),
        "sharpe": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
        "trade_count": trade_count,
        "regime_counts": regime_counts,
    }


def _compute_market_features_from_histories(
    all_histories: List[np.ndarray],
    vol_window: int,
    ma_long: int,
) -> dict:
    """
    Compute market-wide features from ensemble of stock price histories.
    all_histories: list of 1D price arrays, each of length (vol_window + ma_long)
    """
    import numpy as np

    if not all_histories:
        return {'realized_vol': 0.02, 'breadth_20': 0.5, 'breadth_60': 0.5, 'ret_5d_mean': 0.0}

    # Stack into 2D: (num_stocks, seq_len)
    max_len = max(len(h) for h in all_histories)
    padded = []
    for h in all_histories:
        if len(h) < max_len:
            h = np.pad(h, (max_len - len(h), 0), mode='edge')
        padded.append(h)
    stacked = np.stack(padded)  # (num_stocks, seq_len)

    num_stocks = len(all_histories)

    # Compute realized volatility across all stocks at each timestep
    # Then average
    vol_per_stock = []
    for i in range(num_stocks):
        prices = stacked[i]
        if len(prices) >= vol_window:
            rets = np.diff(prices[-vol_window:]) / (prices[-vol_window:-1] + 1e-8)
            vol_per_stock.append(np.std(rets))
    realized_vol = np.mean(vol_per_stock) * np.sqrt(252) if vol_per_stock else 0.02

    # Compute breadth: % of stocks above their MA
    above_ma20_count = 0
    above_ma60_count = 0
    for i in range(num_stocks):
        prices = stacked[i]
        current_price = prices[-1]
        if len(prices) >= 20:
            ma20 = np.mean(prices[-20:])
            if current_price > ma20:
                above_ma20_count += 1
        if len(prices) >= 60:
            ma60 = np.mean(prices[-60:])
            if current_price > ma60:
                above_ma60_count += 1

    breadth_20 = above_ma20_count / max(num_stocks, 1)
    breadth_60 = above_ma60_count / max(num_stocks, 1)

    # Compute 5-day return momentum (mean across stocks)
    ret_5d_list = []
    for i in range(num_stocks):
        prices = stacked[i]
        if len(prices) >= 6:
            ret = (prices[-1] - prices[-6]) / (prices[-6] + 1e-8)
            ret_5d_list.append(ret)
    ret_5d_mean = np.mean(ret_5d_list) if ret_5d_list else 0.0

    return {
        'realized_vol': realized_vol,
        'breadth_20': breadth_20,
        'breadth_60': breadth_60,
        'ret_5d_mean': ret_5d_mean,
    }


def backtest_without_regime(
    model: BCNetwork,
    test_sequences: List[np.ndarray],
    device: torch.device,
    trade_threshold: float = 0.05,
) -> dict:
    """Baseline backtest WITHOUT regime adjustment."""
    model.eval()

    commission = 0.0003
    stamp_tax = 0.001
    slippage = 0.0001

    portfolio_values = []
    all_actions = []
    trade_count = 0

    for seq in tqdm(test_sequences, desc="Backtesting (baseline)"):
        if len(seq) < 120:
            continue

        portfolio = 1.0
        position = 0.0
        entry_price = 0.0

        for t in range(60, len(seq) - 1):
            state = compute_state(seq, t, position, entry_price)
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(state_t).item()

            current_price = seq[t, 3]
            next_price = seq[t + 1, 3]

            is_new = (action > trade_threshold and position < 0.01)
            is_close = (action < trade_threshold and position > 0.01)

            if is_new:
                cost = action * (1.0 + commission + slippage)
                portfolio *= (1.0 - cost)
                position = action
                entry_price = current_price
                trade_count += 1
            elif is_close:
                rev = position * (1.0 - commission - stamp_tax - slippage)
                portfolio *= (1.0 + rev)
                position = 0.0
                entry_price = 0.0
            elif position > 0.01:
                delta = abs(action - position)
                if delta > 0.001:
                    cost = delta * (commission + slippage)
                    portfolio *= (1.0 - cost)
                position = action

            if position > 0.01:
                ret = (next_price - current_price) / current_price if current_price > 0 else 0.0
                portfolio *= (1.0 + ret * position)

        portfolio_values.append(portfolio)
        all_actions.append(position)

    returns = [v - 1.0 for v in portfolio_values]

    print(f"\n{'='*50}")
    print("BC Baseline (No Regime) Backtest Results")
    print(f"{'='*50}")
    print(f"Total Return:     {np.mean(returns)*100:.2f}%")
    print(f"Std of Return:    {np.std(returns)*100:.2f}%")
    print(f"Sharpe Ratio:    {np.mean(returns)/(np.std(returns)+1e-8)*np.sqrt(252):.4f}")
    print(f"Trade Count:     {trade_count}")

    return {
        "return": np.mean(returns),
        "sharpe": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
        "trade_count": trade_count,
    }


def load_test_sequences(symbols, start_date="2024-01-01"):
    """Load full stock sequences with features."""
    sequences = []
    start_dt = pd.to_datetime(start_date)

    for sym in tqdm(symbols, desc="Loading test data"):
        try:
            arr = load_stock_history(sym, base_dir=DATA_DIR)
            if arr is None or len(arr) < 120:
                continue

            df = pd.DataFrame(arr)
            df["date"] = pd.to_datetime(df["date"]).dt.floor("D").astype("datetime64[ns]")
            df.sort_values("date", inplace=True)
            df = df[df["date"] >= start_dt]

            if len(df) < 120:
                continue

            price_df = df[PRICE_FEATURE_COLUMNS].copy()
            tech_df = compute_technical_indicators(df)
            combined = pd.concat([price_df, tech_df], axis=1)

            cols = list(PRICE_FEATURE_COLUMNS)
            for c in ["rsi_14", "macd_line", "macd_signal", "macd_hist",
                      "bb_position", "bb_width", "volume_ma5", "volume_ma20",
                      "atr_14", "stoch_k", "stoch_d", "obv_ma10", "roc_10", "momentum_10"]:
                if c in combined.columns:
                    cols.append(c)

            arr_out = combined[cols].values.astype(np.float32)
            arr_out = np.nan_to_num(arr_out, nan=0.0, posinf=0.0, neginf=0.0)

            for ci in range(arr_out.shape[1]):
                arr_out[:, ci] = np.clip(arr_out[:, ci], -1e6, 1e6)

            sequences.append(arr_out)
        except Exception:
            continue

    return sequences


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BC with Market Regime Detection")
    parser.add_argument("--checkpoint", type=str, default="models/bc_pretrain.pt")
    parser.add_argument("--test-symbols", type=int, default=50)
    parser.add_argument("--regime-symbols", type=int, default=100)
    parser.add_argument("--start", type=str, default="2024-01-01")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load BC model
    model = BCNetwork(input_dim=6, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded BC model from {args.checkpoint}")

    # Load symbols
    all_symbols = list_symbols(base_dir=DATA_DIR)
    print(f"Total symbols: {len(all_symbols)}")

    # Use first N symbols for market regime detection (market proxy)
    regime_symbols = all_symbols[:args.regime_symbols]
    # Use different symbols for actual trading (test set)
    test_symbols = all_symbols[200:200 + args.test_symbols]

    print(f"\nLoading regime detection data ({len(regime_symbols)} symbols)...")
    regime_sequences = load_stock_sequences_for_regime(
        regime_symbols, args.start, "2026-01-01"
    )
    print(f"Loaded {len(regime_sequences)} regime sequences")

    print(f"\nLoading test data ({len(test_symbols)} symbols)...")
    test_sequences = load_test_sequences(test_symbols, args.start)
    print(f"Loaded {len(test_sequences)} test sequences")

    if len(test_sequences) < 10:
        print("Not enough test data!")
        return

    # Run baseline (no regime)
    print("\n" + "=" * 60)
    print("RUNNING BASELINE (NO REGIME)")
    print("=" * 60)
    baseline_results = backtest_without_regime(model, test_sequences, device)

    # Run with regime
    print("\n" + "=" * 60)
    print("RUNNING WITH MARKET REGIME")
    print("=" * 60)
    regime_results = backtest_with_regime(
        model, regime_sequences, test_sequences, device
    )

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':<15} {'With Regime':<15}")
    print("-" * 50)
    print(f"{'Return':<20} {baseline_results['return']*100:.2f}%{'':<8} {regime_results['return']*100:.2f}%")
    print(f"{'Sharpe':<20} {baseline_results['sharpe']:.4f}{'':<8} {regime_results['sharpe']:.4f}")
    print(f"{'Trades':<20} {baseline_results['trade_count']}{'':<8} {regime_results['trade_count']}")

    improvement = regime_results['return'] - baseline_results['return']
    print(f"\n{'Improvement:':<20} {improvement*100:.2f}%")


if __name__ == "__main__":
    main()
