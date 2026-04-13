import sqlite3
import os

conn = sqlite3.connect('data/stock.db')
cur = conn.cursor()

# Check price counts per table
for table in ['price', 'price_hfq', 'price_qfq']:
    try:
        cur.execute(f'SELECT COUNT(*) FROM {table}')
        print(f'{table}: {cur.fetchone()[0]} rows')
    except Exception as e:
        print(f'{table}: ERROR - {e}')

# Check last symbol processed
try:
    cur.execute("SELECT symbol, MAX(date) FROM price_hfq GROUP BY symbol ORDER BY MAX(date) DESC LIMIT 5")
    print('Last 5 symbols by date:', cur.fetchall())
except Exception as e:
    print(f'Error: {e}')

try:
    cur.execute('SELECT COUNT(DISTINCT symbol) FROM price_hfq')
    print(f'Unique symbols in price_hfq: {cur.fetchone()[0]}')
except Exception as e:
    print(f'Error: {e}')

# Check fund_flow
try:
    cur.execute('SELECT COUNT(*) FROM fund_flow')
    print(f'fund_flow: {cur.fetchone()[0]} rows')
    cur.execute('SELECT COUNT(DISTINCT symbol) FROM fund_flow')
    print(f'Unique symbols in fund_flow: {cur.fetchone()[0]}')
except Exception as e:
    print(f'fund_flow error: {e}')

# Check migration progress
progress_file = 'data/migration_progress.txt'
if os.path.exists(progress_file):
    with open(progress_file) as f:
        lines = f.readlines()
    print(f'\nMigration progress file ({len(lines)} entries):')
    for l in lines[-5:]:
        print(f'  {l.strip()}')
else:
    print('\nNo migration progress file found')

conn.close()
