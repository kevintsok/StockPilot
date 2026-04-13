content = open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/backtest.py').read()

# Add debug logging and defensive check after window_dates/next_dates assignment
old = '''        window_dates = dates[seq_len - 1 : -1]   # T date (= context end = prediction date)
        next_dates = dates[seq_len + 1:]          # T+1 date (= target date = exit date)

        cache[sym] = {'''

new = '''        window_dates = dates[seq_len - 1 : -1]   # T date (= context end = prediction date)
        next_dates = dates[seq_len + 1:]          # T+1 date (= target date = exit date)

        # Defensive: ensure num_windows matches next_dates length
        safe_num_windows = min(num_windows, len(next_dates))
        if safe_num_windows < num_windows:
            print(f"[Debug] {sym}: num_windows {num_windows} > len(next_dates) {len(next_dates)}, truncating")
        num_windows = safe_num_windows

        cache[sym] = {'''

if old in content:
    content = content.replace(old, new)
    print('Fix applied')
else:
    print('Pattern not found')

open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/backtest.py', 'w').write(content)
