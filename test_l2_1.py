from src.dssms.dssms_scheduler import DSSMSScheduler
import json

s = DSSMSScheduler()

s.positions['7203'] = {
    'symbol': '7203',
    'entry_price': 1000,
    'shares': 100,
    'entry_date': '2026-04-01',
    'strategy': 'GCStrategy'
}
s._save_positions()

with open('logs/dssms/positions.json', 'r', encoding='utf-8') as f:
    before = json.load(f)
print('BEFORE positions keys:', list(before.keys()))

original_retry = s._execute_with_retry
original_debug_mode = s.kabu_integration.config.get('development_settings', {}).get('debug_mode', True)


def mock_fail_retry(*args, **kwargs):
    print('[MOCK] _execute_with_retry called -> forced failure')
    return {'success': False, 'error': 'mock_failure'}


s.kabu_integration.config.setdefault('development_settings', {})['debug_mode'] = False
s._execute_with_retry = mock_fail_retry

result = s._execute_sl_sell('7203', 100)
print('_execute_sl_sell result:', result)

with open('logs/dssms/positions.json', 'r', encoding='utf-8') as f:
    after = json.load(f)
print('AFTER positions keys:', list(after.keys()))

if '7203' in after:
    print('PASS L2-1: positions.json unchanged on SELL failure')
else:
    print('FAIL L2-1: 7203 disappeared after SELL failure')

s._execute_with_retry = original_retry
s.kabu_integration.config.setdefault('development_settings', {})['debug_mode'] = original_debug_mode
if '7203' in s.positions:
    del s.positions['7203']
s._save_positions()
print('cleanup done')
