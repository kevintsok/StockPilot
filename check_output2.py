import os

path = '/mnt/c/Users/kevin/AppData/Local/Temp/claude/D--Projects-auto-select-stock/567209b8-5fe8-4510-ab54-59291379c388/tasks/bfxfvjagi.output'
if os.path.exists(path):
    with open(path, 'rb') as f:
        content = f.read().decode('utf-8', errors='ignore')
    lines = content.split('\n')
    for line in lines[-60:]:
        if any(x in line for x in ['Debug', 'Epoch', 'Strategy', 'Collect', 'batch', 'Traceback', 'Error', 'IndexError', 'signal', 'saved', 'results', 'total', 'Sharpe', 'return']):
            print(line[:300])
else:
    print(f'File not found: {path}')
    # Check if there's a newer output file
    import glob
    tasks_dir = '/mnt/c/Users/kevin/AppData/Local/Temp/claude/D--Projects-auto-select-stock/567209b8-5fe8-4510-ab54-59291379c388/tasks/'
    files = sorted(glob.glob(tasks_dir + '*.output'), key=os.path.getmtime)
    print('Recent files:', [os.path.basename(f) for f in files[-5:]])
