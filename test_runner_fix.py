import sys
sys.path.insert(0, 'src')
from pathlib import Path
import pandas as pd
from auto_select_stock.predict.backtest import BacktestConfig, _collect_signals_batched, _parse_date, filter_a_share_symbols
from auto_select_stock.predict.inference import PricePredictor
from auto_select_stock.predict.strategies.base import Signal
from auto_select_stock.predict.strategies import TopKStrategy
from auto_select_stock.storage import list_symbols

print("=== TopK strategy simulation (6 days) ===")

predictor = PricePredictor('models/price_transformer_multihorizon_full.pt')
symbols = filter_a_share_symbols(list_symbols(base_dir=Path('data')))[:50]

cfg = BacktestConfig(
    checkpoint=Path('models/price_transformer_multihorizon_full.pt'),
    start_date='2024-06-01', end_date='2024-06-10',
    symbols=symbols, cost_bps=0, slippage_bps=0, base_dir=Path('data'),
)

daily_raw = _collect_signals_batched(
    symbols, predictor, cfg,
    _parse_date(cfg.start_date), _parse_date(cfg.end_date),
    show_progress=False, horizon='1d',
)
dates = sorted(daily_raw.keys())
print(f"Days: {[d.strftime('%Y-%m-%d') for d in dates]}")

INITIAL = 100_000.0
LOT = 100
cash = float(INITIAL)
positions = {}
prev_weights = {}
strat = TopKStrategy(top_k=5, allow_short=False, horizon="1d")
portfolio_values = []

for dt in dates:
    raw = daily_raw[dt]
    sigs = []
    for raw_item in raw:
        if len(raw_item) >= 7:
            sym, pred, realized, ind, pred_rets, entry_price, auc_limit = raw_item[:7]
        else:
            sym, pred, realized, ind, pred_rets = raw_item[:5]
            entry_price, auc_limit = 0.0, 0
        sigs.append(Signal(symbol=sym, predicted_ret=pred, realized_ret=realized,
                          industry=ind, predicted_rets=pred_rets,
                          entry_price=entry_price, auc_limit=auc_limit))

    price_map = {s.symbol: s.entry_price for s in sigs}
    auc_map = {s.symbol: s.auc_limit for s in sigs}

    # Sell all existing positions at T's close
    sell_cash = sum(pos["shares"] * price_map.get(sym, pos["entry_price"])
                    for sym, pos in positions.items())
    positions.clear()

    # Strategy selects new target positions
    weights = strat.select_positions(sigs, prev_weights, {})
    prev_weights = weights

    # Buy new positions at T's close
    top_sigs = sorted([s for s in sigs if s.predicted_ret > 0],
                       key=lambda s: s.predicted_ret, reverse=True)[:5]
    total_pred = sum(s.predicted_ret for s in top_sigs)
    total_buy_cash = cash + sell_cash

    buy_cash = 0.0
    for s in top_sigs:
        if auc_map.get(s.symbol, 0) == 1:
            continue
        price = s.entry_price
        if price <= 0:
            continue
        alloc = total_buy_cash * s.predicted_ret / total_pred if total_pred > 0 else 0
        max_shares = int(alloc // price // LOT) * LOT
        if max_shares < LOT:
            continue
        positions[s.symbol] = {"shares": max_shares, "entry_price": price}
        buy_cash += max_shares * price

    cash = cash + sell_cash - buy_cash
    holdings_val = sum(pos["shares"] * price_map.get(sym, pos["entry_price"])
                      for sym, pos in positions.items())
    portfolio_val = cash + holdings_val

    prev_val = portfolio_values[-1] if portfolio_values else INITIAL
    daily_ret = portfolio_val / prev_val - 1 if prev_val > 0 else 0.0
    portfolio_values.append(portfolio_val)

    print(f"{dt.strftime('%Y-%m-%d')}: cash={cash:>12,.0f} holdings={holdings_val:>12,.0f} "
          f"total={portfolio_val:>12,.0f} ret={daily_ret:+.2%} pos={len(positions)}")

final_val = portfolio_values[-1]
print(f"\nFinal: {final_val:,.2f} RMB | Return: {(final_val/INITIAL-1):.2%}")
