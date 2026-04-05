from src.dssms.dssms_scheduler import DSSMSScheduler
import json

s = DSSMSScheduler()
test_symbol = 'L2201'

s._add_position(test_symbol, entry_price=2345.0, quantity=100, strategy='GCStrategy')

with open('logs/dssms/positions.json', 'r', encoding='utf-8') as f:
    positions = json.load(f)

if test_symbol in positions:
    recorded_price = positions[test_symbol].get('entry_price')
    print(f'recorded entry_price: {recorded_price}')
    if recorded_price == 2345.0:
        print('PASS L2-2: entry_price matches executed_price 2345.0')
    else:
        print(f'FAIL L2-2: entry_price mismatch ({recorded_price} != 2345.0)')
else:
    print(f'FAIL L2-2: {test_symbol} not found in positions.json')

if test_symbol in s.positions:
    del s.positions[test_symbol]
s._save_positions()
print('cleanup done')
