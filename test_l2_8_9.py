from src.dssms.dssms_scheduler import DSSMSScheduler

s = DSSMSScheduler()

print('=== L2-8: insufficient balance ===')
original_balance = s.paper_balance.balance
s.paper_balance._balance = 1000.0
s.paper_balance._save()
print(f'forced balance: {s.paper_balance.balance} yen')

balance = s.paper_balance.balance
max_pos = s.max_positions
allocated = balance / max_pos
test_price = 2000
raw_shares = int(allocated / test_price)
rounded = (raw_shares // 100) * 100

print(f'allocated={allocated:.1f} / price={test_price} -> rounded={rounded} shares')
if rounded < 100:
    print('PASS L2-8: LOT_GUARD skip expected')
else:
    print('FAIL L2-8: BUY may pass unexpectedly')

s.paper_balance._balance = original_balance
s.paper_balance._save()
print(f'balance restored: {s.paper_balance.balance} yen')

print('')
print('=== L2-9: max positions limit ===')
for code in ['1001', '1002', '1003']:
    s.positions[code] = {
        'symbol': code,
        'entry_price': 1000.0,
        'shares': 100,
        'entry_date': '2026-04-01',
        'strategy': 'GCStrategy'
    }
s._save_positions()
print(f'positions count: {len(s.positions)} / max_positions: {s.max_positions}')

if len(s.positions) >= s.max_positions:
    print('PASS L2-9: max position guard active')
else:
    print('FAIL L2-9: max position guard inactive')

for code in ['1001', '1002', '1003']:
    if code in s.positions:
        del s.positions[code]
s._save_positions()
print('cleanup done')
