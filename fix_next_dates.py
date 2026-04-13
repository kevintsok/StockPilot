content = open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/backtest.py').read()

# Fix the order: remove the early debug block that references next_dates before it's defined
old = '''        # DEBUG: check lengths
        if len(dates) != len(closes) or len(features) != len(dates):
            print(f"[Debug] {sym}: dates={len(dates)} closes={len(closes)} features={len(features)}")
        if len(next_dates) < num_windows:
            print(f"[Debug] {sym}: num_windows={num_windows} but next_dates={len(next_dates)}")

        # FIX: entry price = T's close (not T-1's close).'''

new = '''        # DEBUG: check lengths (after next_dates is defined below)
        # FIX: entry price = T's close (not T-1's close).'''

if old in content:
    content = content.replace(old, new)
    print('Removed early debug block')
else:
    print('Pattern not found, trying alternate')
    # Remove just the problematic lines
    old2 = '''        if len(next_dates) < num_windows:
            print(f"[Debug] {sym}: num_windows={num_windows} but next_dates={len(next_dates)}")

        # FIX: entry price'''
    new2 = '''        # FIX: entry price'''
    if old2 in content:
        content = content.replace(old2, new2)
        print('Removed problematic check')
    else:
        print('Alternate pattern not found either')

open('/mnt/d/Projects/auto-select-stock/src/auto_select_stock/predict/backtest.py', 'w').write(content)
