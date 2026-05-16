"""
Sector analysis: correlation matrix, cycle detection, momentum ranking.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Import storage functions directly to avoid circular imports
import auto_select_stock.data.storage as storage_module

load_sector_daily = storage_module.load_sector_daily
list_all_sectors = storage_module.list_all_sectors


def compute_correlation_matrix(
    sector_codes: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Dict:
    """
    Compute Pearson correlation matrix of daily returns across sectors.

    Returns:
        Dict with 'dates', 'sectors', and 'correlation' matrix (2D list)
    """
    # Collect daily returns for each sector
    sector_returns: Dict[str, pd.Series] = {}
    all_dates: set = set()

    for code in sector_codes:
        try:
            df = load_sector_daily(code, base_dir=base_dir)
        except FileNotFoundError:
            continue

        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end)]

        df = df.sort_values("date")
        if df.empty:
            continue

        returns = df["pct_change"].fillna(0) / 100  # convert to decimal
        returns.index = pd.to_datetime(df["date"])
        sector_returns[code] = returns
        all_dates.update(returns.index)

    if not sector_returns:
        return {"error": "No data available", "sectors": [], "dates": [], "correlation": []}

    # Align all series to common dates
    aligned = pd.DataFrame(sector_returns)
    aligned = aligned.sort_index()

    # Compute correlation matrix
    corr = aligned.corr(method="pearson")

    return {
        "sectors": list(corr.columns),
        "dates": [str(d) for d in aligned.index],
        "correlation": corr.values.tolist(),
    }


def detect_cycles(
    sector_code: str,
    threshold_pct: float = 5.0,
    base_dir: Optional[Path] = None,
) -> Dict:
    """
    Detect continuous up/down cycles for a sector.

    A cycle starts when cumulative return exceeds threshold_pct and ends
    when the direction reverses.

    Args:
        sector_code: sector code/name
        threshold_pct: minimum return to consider a cycle start (default 5%)
        base_dir: data directory

    Returns:
        Dict with 'cycles' list containing {start, end, direction, total_return, duration_days}
    """
    try:
        df = load_sector_daily(sector_code, base_dir=base_dir)
    except FileNotFoundError:
        return {"error": f"No data for sector {sector_code}", "cycles": []}

    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return {"cycles": [], "sector": sector_code}

    pct = df["pct_change"].fillna(0).values
    dates = pd.to_datetime(df["date"]).values

    cycles = []
    in_cycle = False
    cycle_start_idx = 0
    cycle_direction = 0  # 1 for up, -1 for down
    cumulative_return = 0.0

    for i in range(len(pct)):
        daily_ret = pct[i]

        if not in_cycle:
            # Start a new cycle if daily return exceeds threshold
            if abs(daily_ret) >= threshold_pct:
                in_cycle = True
                cycle_start_idx = i
                cycle_direction = 1 if daily_ret > 0 else -1
                cumulative_return = daily_ret
        else:
            cumulative_return += daily_ret

            # Check if cycle should end (direction reverses)
            if cycle_direction == 1 and cumulative_return <= -threshold_pct:
                # Bull cycle ended, start bear
                cycles.append({
                    "start": str(pd.Timestamp(dates[cycle_start_idx]).date()),
                    "end": str(pd.Timestamp(dates[i - 1]).date()),
                    "direction": "up",
                    "total_return": round(cumulative_return - daily_ret, 2),
                    "duration_days": i - cycle_start_idx,
                })
                cycle_start_idx = i
                cycle_direction = -1
                cumulative_return = daily_ret
            elif cycle_direction == -1 and cumulative_return >= threshold_pct:
                # Bear cycle ended, start bull
                cycles.append({
                    "start": str(pd.Timestamp(dates[cycle_start_idx]).date()),
                    "end": str(pd.Timestamp(dates[i - 1]).date()),
                    "direction": "down",
                    "total_return": round(cumulative_return - daily_ret, 2),
                    "duration_days": i - cycle_start_idx,
                })
                cycle_start_idx = i
                cycle_direction = 1
                cumulative_return = daily_ret

    # Close any open cycle
    if in_cycle and cycles:
        last_cycle = cycles[-1]
        if last_cycle["direction"] == "up" and cycle_direction == 1:
            last_cycle["end"] = str(pd.Timestamp(dates[-1]).date())
            last_cycle["total_return"] = round(cumulative_return, 2)
            last_cycle["duration_days"] = len(pct) - cycle_start_idx
        elif last_cycle["direction"] == "down" and cycle_direction == -1:
            last_cycle["end"] = str(pd.Timestamp(dates[-1]).date())
            last_cycle["total_return"] = round(cumulative_return, 2)
            last_cycle["duration_days"] = len(pct) - cycle_start_idx

    return {
        "sector": sector_code,
        "threshold_pct": threshold_pct,
        "cycles": cycles,
    }


def cycle_statistics(
    sector_code: str,
    base_dir: Optional[Path] = None,
) -> Dict:
    """
    Compute statistics about up/down cycles for a sector.

    Returns:
        Dict with avg_duration_months, cycle_count, up_cycles, down_cycles, etc.
    """
    result = detect_cycles(sector_code, threshold_pct=5.0, base_dir=base_dir)
    cycles = result.get("cycles", [])

    if not cycles:
        return {
            "sector": sector_code,
            "cycle_count": 0,
            "up_cycles": 0,
            "down_cycles": 0,
            "avg_up_duration_days": None,
            "avg_down_duration_days": None,
            "avg_up_return": None,
            "avg_down_return": None,
            "longest_up_cycle": None,
            "longest_down_cycle": None,
        }

    up_cycles = [c for c in cycles if c["direction"] == "up"]
    down_cycles = [c for c in cycles if c["direction"] == "down"]

    def avg_duration(cycle_list):
        if not cycle_list:
            return None
        return round(sum(c["duration_days"] for c in cycle_list) / len(cycle_list), 1)

    def avg_return(cycle_list):
        if not cycle_list:
            return None
        return round(sum(c["total_return"] for c in cycle_list) / len(cycle_list), 2)

    def longest(cycle_list):
        if not cycle_list:
            return None
        return max(c["duration_days"] for c in cycle_list)

    return {
        "sector": sector_code,
        "cycle_count": len(cycles),
        "up_cycles": len(up_cycles),
        "down_cycles": len(down_cycles),
        "avg_up_duration_days": avg_duration(up_cycles),
        "avg_down_duration_days": avg_duration(down_cycles),
        "avg_up_return": avg_return(up_cycles),
        "avg_down_return": avg_return(down_cycles),
        "longest_up_cycle": longest(up_cycles),
        "longest_down_cycle": longest(down_cycles),
        # Approximate months (assuming ~21 trading days per month)
        "avg_up_duration_months": round(avg_duration(up_cycles) / 21, 1) if avg_duration(up_cycles) else None,
        "avg_down_duration_months": round(avg_duration(down_cycles) / 21, 1) if avg_duration(down_cycles) else None,
    }


def rank_sectors_by_momentum(
    sector_codes: Optional[List[str]] = None,
    lookback_days: int = 20,
    category: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Rank sectors by momentum (recent return).

    Args:
        sector_codes: list of sector codes to analyze, or None for all sectors
        lookback_days: number of days for momentum calculation (default 20)
        category: filter by 'industry' or 'concept'
        base_dir: data directory

    Returns:
        List of dicts with sector info and momentum metrics, sorted by momentum descending
    """
    if sector_codes is None:
        all_sectors = list_all_sectors(base_dir=base_dir)
        if category:
            sector_codes = [code for code, name, cat in all_sectors if cat == category]
        else:
            sector_codes = [code for code, name, cat in all_sectors]

    results = []

    for code in sector_codes:
        try:
            df = load_sector_daily(code, base_dir=base_dir)
        except FileNotFoundError:
            continue

        df = df.sort_values("date")
        if len(df) < lookback_days:
            continue

        current_close = df.iloc[-1]["close"]
        past_close = df.iloc[-lookback_days]["close"]
        momentum = (current_close - past_close) / past_close * 100 if past_close != 0 else 0

        # Also calculate 60-day for medium-term momentum
        if len(df) >= 60:
            past_close_60d = df.iloc[-60]["close"]
            momentum_60d = (current_close - past_close_60d) / past_close_60d * 100 if past_close_60d != 0 else 0
        else:
            momentum_60d = None

        results.append({
            "code": code,
            "name": code,
            "momentum": round(momentum, 2),
            "momentum_60d": round(momentum_60d, 2) if momentum_60d is not None else None,
            "current_close": round(float(current_close), 2),
            "lookback_days": lookback_days,
            "data_points": len(df),
        })

    # Sort by momentum descending
    results.sort(key=lambda x: x["momentum"], reverse=True)
    return results


def sector_correlation_with_market(
    sector_code: str,
    market_symbol: str = "000001",
    base_dir: Optional[Path] = None,
) -> Dict:
    """
    Compute correlation between a sector and a market index (e.g., 000001 = 上证指数).
    """
    try:
        sector_df = load_sector_daily(sector_code, base_dir=base_dir)
    except FileNotFoundError:
        return {"error": f"No data for sector {sector_code}"}

    try:
        market_df = storage_module.load_stock_history(market_symbol, base_dir=base_dir)
        market_df = pd.DataFrame(market_df)
    except Exception:
        return {"error": f"No market data for {market_symbol}"}

    # Align by date
    sector_df["date"] = pd.to_datetime(sector_df["date"])
    market_df["date"] = pd.to_datetime(market_df["date"])

    merged = pd.merge(
        sector_df[["date", "pct_change"]].rename(columns={"pct_change": "sector_ret"}),
        market_df[["date", "pct_change"]].rename(columns={"pct_change": "market_ret"}),
        on="date",
        how="inner",
    )

    if len(merged) < 10:
        return {"error": "Insufficient overlapping data"}

    corr = merged["sector_ret"].corr(merged["market_ret"])

    return {
        "sector": sector_code,
        "market": market_symbol,
        "correlation": round(float(corr), 4),
        "data_points": len(merged),
        "date_range": {
            "start": str(merged["date"].min().date()),
            "end": str(merged["date"].max().date()),
        },
    }
