import json

with open('/mnt/d/Projects/auto-select-stock/reports/JulianGame_392.1774974257452482222.pt.trace.json') as f:
    d = json.load(f)

trace_events = d.get('traceEvents', [])
print('Total events:', len(trace_events))

# Check for cuda kernel events
cuda_events = [e for e in trace_events if 'cuda' in e.get('name', '').lower() or e.get('cat') == 'cuda']
print(f'CUDA-related events: {len(cuda_events)}')

# Look for kernel launches
kernel_events = [e for e in trace_events if 'Kernel' in e.get('name', '') or 'kernel' in e.get('name', '').lower()]
print(f'Kernel events: {len(kernel_events)}')
for e in kernel_events[:5]:
    print(f'  {e.get("name")} dur={e.get("dur")}')

# Check for backward-related
bwd_names = set()
for e in trace_events:
    name = e.get('name', '')
    if any(x in name.lower() for x in ['backward', 'grad', 'bw', 'autograd']):
        bwd_names.add(name)
print(f'\nBackward-related event names ({len(bwd_names)}):')
for n in sorted(bwd_names)[:20]:
    print(f'  {n}')

# Look for duration info
durations = [(e.get('name'), e.get('dur')) for e in trace_events if e.get('dur')]
durations.sort(key=lambda x: x[1] if x[1] else 0, reverse=True)
print(f'\nTop 10 longest events by duration:')
for name, dur in durations[:10]:
    print(f'  {name}: {dur}us')

# Check for modules
modules = set()
for e in trace_events:
    m = e.get('module', '')
    if m:
        modules.add(m)
print(f'\nModules: {modules}')
