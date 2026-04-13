import json

with open('/mnt/d/Projects/auto-select-stock/reports/JulianGame_392.1774974257452482222.pt.trace.json') as f:
    d = json.load(f)

trace_events = d.get('traceEvents', [])
print('Total events:', len(trace_events))
print('Schema version:', d.get('schemaVersion'))

# Count different types of events
types = {}
for e in trace_events:
    cat = e.get('cat', 'unknown')
    types[cat] = types.get(cat, 0) + 1
for k, v in sorted(types.items()):
    print(f'  {k}: {v}')

# Print first 5 events
print('\nFirst 5 events:')
for e in trace_events[:5]:
    print(f'  name={e.get("name")} cat={e.get("cat")} pid={e.get("pid")} tid={e.get("tid")}')

# Print events with 'backward' in name
bwd = [e for e in trace_events if 'backward' in e.get('name', '').lower() or 'bw' in e.get('name', '').lower()]
print(f'\nBackward events: {len(bwd)}')
for e in bwd[:10]:
    print(f'  {e.get("name")}')
