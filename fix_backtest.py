content = open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/backtest.py').read()

# Fix 1: _build_feature_frame signature and load_stock_history call
old1 = '''def _build_feature_frame(
    symbol: str,
    price_columns: List[str],
    financial_columns: List[str],
    base_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    arr = load_stock_history(symbol, base_dir=base_dir)'''
new1 = '''def _build_feature_frame(
    symbol: str,
    price_columns: List[str],
    financial_columns: List[str],
    base_dir: Path,
    table: str = "price_hfq",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    arr = load_stock_history(symbol, base_dir=base_dir, table=table)'''
if old1 in content:
    content = content.replace(old1, new1)
    print('Fix 1 applied')
else:
    print('Fix 1 not found')

# Fix 2: call site in _collect_signals_batched
old2 = '''        dates_df, features, closes, opens = _build_feature_frame(
            sym, predictor.cfg.price_columns, predictor.cfg.financial_columns, cfg.base_dir
        )'''
new2 = '''        dates_df, features, closes, opens = _build_feature_frame(
            sym, predictor.cfg.price_columns, predictor.cfg.financial_columns, cfg.base_dir,
            table=getattr(predictor.cfg, "price_table", "price_hfq"),
        )'''
if old2 in content:
    content = content.replace(old2, new2)
    print('Fix 2 applied')
else:
    print('Fix 2 not found')

open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/backtest.py', 'w').write(content)
