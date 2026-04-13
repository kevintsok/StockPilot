content = open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/backtest.py').read()

# Add debug prints after building arrays
old = '''        normed = (features - scaler_mean) / scaler_std
        num_windows = len(normed) - seq_len
        if num_windows <= 0:
            continue

        # FIX: entry price = T's close'''
new = '''        normed = (features - scaler_mean) / scaler_std
        num_windows = len(normed) - seq_len
        if num_windows <= 0:
            continue
        # DEBUG: check lengths
        if len(dates) != len(closes) or len(features) != len(dates):
            print(f"[Debug] {sym}: dates={len(dates)} closes={len(closes)} features={len(features)}")
        if len(next_dates) < num_windows:
            print(f"[Debug] {sym}: num_windows={num_windows} but next_dates={len(next_dates)}")

        # FIX: entry price = T's close'''
if old in content:
    content = content.replace(old, new)
    print('Fix applied')
else:
    print('Pattern not found')

open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/backtest.py', 'w').write(content)
