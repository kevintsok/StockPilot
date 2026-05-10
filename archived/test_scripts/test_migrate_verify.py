#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
from auto_select_stock.data.storage import load_stock_history, load_fund_flow
import pandas as pd

# Check price tables
arr = load_stock_history('000001', table='price')
df = pd.DataFrame(arr)
print(f"price 000001: {len(df)} rows, latest close={df.iloc[-1]['close']:.2f}")

arr_h = load_stock_history('000001', table='price_hfq')
df_h = pd.DataFrame(arr_h)
print(f"price_hfq 000001: {len(df_h)} rows, latest close={df_h.iloc[-1]['close']:.2f}")

# Check fund_flow
ff = load_fund_flow('000001')
print(f"fund_flow 000001: {len(ff)} rows")
print("Latest 3 rows:")
print(ff.tail(3).to_string())
