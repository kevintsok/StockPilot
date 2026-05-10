content = open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/train.py').read()
old = '    cfg.price_columns = price_cols\n    cfg.financial_columns = fin_cols\n    parsed_windows'
new = '    cfg.price_columns = price_cols\n    cfg.financial_columns = fin_cols\n    cfg.price_table = price_table  # store in cfg so checkpoint captures it\n    parsed_windows'
if old in content:
    content = content.replace(old, new)
    print('Fixed')
else:
    print('Not found')
open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/train.py', 'w').write(content)
