import sys
sys.path.insert(0, 'src')
from pathlib import Path
import pandas as pd
from auto_select_stock.predict.backtest import _collect_signals_batched, _parse_date, BacktestConfig, filter_a_share_symbols
from auto_select_stock.predict.inference import PricePredictor
from auto_select_stock.predict.strategies.base import Signal
from auto_select_stock.predict.strategies import TopKStrategy
from auto_select_stock.storage import list_symbols
from auto_select_stock.predict.strategies.runner import _INITIAL_CAPITAL, _LOT_SIZE, _AUC_LIMIT_THRESHOLD

predictor = PricePredictor('models/price_transformer_multihorizon_full.pt')
symbols = filter_a_share_symbols(list_symbols(base_dir=Path('data')))[:200]

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
print("Days:", len(dates))

strat = TopKStrategy(top_k=5, allow_short=False, horizon="1d")
cash = float(_INITIAL_CAPITAL)
positions = {}
prev_weights = {}

for dt in dates:
    raw = daily_raw[dt]
    signals = []
    for raw_item in raw:
        if len(raw_item) >= 7:
            s, pred, realized, ind, pred_rets, entry_price, auc_limit = raw_item[:7]
        else:
            s, pred, realized, ind, pred_rets = raw_item[:5]
            entry_price, auc_limit = 0.0, 0
        signals.append(Signal(symbol=s, predicted_ret=pred, realized_ret=realized,
                              industry=ind, predicted_rets=pred_rets,
                              entry_price=entry_price, auc_limit=auc_limit))

    price_map = {sig.symbol: sig.entry_price for sig in signals}
    auc_map = {sig.symbol: sig.auc_limit for sig in signals}

    # Sell all existing positions
    sell_cash = 0.0
    for sym, pos in list(positions.items()):
        price = price_map.get(sym, pos['entry_price'])
        sell_amount = pos['shares'] * price
        sell_cash += sell_amount
    positions.clear()

    # Strategy selects new positions
    weights = strat.select_positions(signals, prev_weights, {})
    prev_weights = weights

    top_sigs = sorted([s for s in signals if s.predicted_ret > 0],
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
        max_shares = int(alloc // price // _LOT_SIZE) * _LOT_SIZE
        if max_shares < _LOT_SIZE:
            continue
        positions[s.symbol] = {"shares": max_shares, "entry_price": price}
        buy_cash += max_shares * price

    cash = cash + sell_cash - buy_cash
    holdings_val = sum(pos["shares"] * price_map.get(sym, pos["entry_price"])
                       for sym, pos in positions.items())
    portfolio_val = cash + holdings_val

    print(f"{str(dt)[:10]}: cash={cash:12,.0f} holdings={holdings_val:12,.0f} total={portfolio_val:12,.0f} pos={len(positions)}")

print(f"\nFinal capital: {cash + holdings_val:.2f}")
print(f"Return: {((cash + holdings_val) / _INITIAL_CAPITAL - 1) * 100:.2f}%")
