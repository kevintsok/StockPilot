content = open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/backtest.py').read()

# Fix the _collect_signals_for_symbol call site too
old = '''    dates_df, features, closes, opens = _build_feature_frame(symbol, price_cols, fin_cols, base_dir)'''
new = '''    dates_df, features, closes, opens = _build_feature_frame(
        symbol, price_cols, fin_cols, base_dir,
        table=getattr(predictor.cfg, "price_table", "price_hfq"),
    )'''
if old in content:
    content = content.replace(old, new)
    print('Fix applied')
else:
    print('Not found')

open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/backtest.py', 'w').write(content)
