from src.dssms.dssms_scheduler import DSSMSScheduler

s1 = DSSMSScheduler()
s1.positions['9984'] = {
    'symbol': '9984',
    'entry_price': 2000.0,
    'quantity': 200,
    'entry_date': '2026-04-01',
    'strategy': 'GCStrategy'
}
s1._save_positions()
print('[Instance1] positions saved:', list(s1.positions.keys()))

s2 = DSSMSScheduler()
print('[Instance2] positions loaded:', list(s2.positions.keys()))

if '9984' in s2.positions:
    p = s2.positions['9984']
    print(f'[Instance2] 9984 entry_price={p["entry_price"]} quantity={p["quantity"]}')
    if p['entry_price'] == 2000.0 and p['quantity'] == 200:
        print('PASS L2-5: positions persisted across restart')
    else:
        print('FAIL L2-5: values mismatch after restart')
else:
    print('FAIL L2-5: 9984 missing after restart')

history = s2.execution_history.get_recent_events(limit=5)
print(f'[Instance2] execution_history count: {len(history)}')
print('execution_history load OK' if isinstance(history, list) else 'execution_history load FAIL')

if '9984' in s2.positions:
    del s2.positions['9984']
s2._save_positions()
print('cleanup done')
