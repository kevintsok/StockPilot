import subprocess
result = subprocess.run(
    ['tail', '-50', '/mnt/c/Users/kevin/AppData/Local/Temp/claude/D--Projects-auto-select-stock/567209b8-5fe8-4510-ab54-59291379c388/tasks/bfxfvjagi.output'],
    capture_output=True, text=True
)
lines = result.stdout.split('\n')
for line in lines:
    if any(x in line for x in ['Debug', 'Epoch', 'Strategy', 'Collect', 'batch', 'Traceback', 'Error', 'IndexError', 'signal']):
        print(line[:200])
