content = open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/core/torch_model.py').read()
old = '    price_table: str = "price_hfq"  # price or price_hfq: price or price_hfq'
new = '    price_table: str = "price_hfq"  # price (qfq) or price_hfq (hfq)'
if old in content:
    content = content.replace(old, new)
    print('Fixed')
else:
    print('Not found')
open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/core/torch_model.py', 'w').write(content)
