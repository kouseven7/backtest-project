from src.dssms.dssms_scheduler import DSSMSScheduler
import json

s = DSSMSScheduler()

# 3銘柄のポジションをセット（キーはquantity・entry_time）
s.positions['7203'] = {
    'symbol': '7203',
    'entry_price': 1000.0,
    'quantity': 100,
    'entry_time': '2026-04-01',
    'strategy': 'GCStrategy'
}
s.positions['9984'] = {
    'symbol': '9984',
    'entry_price': 2000.0,
    'quantity': 100,
    'entry_time': '2026-04-01',
    'strategy': 'GCStrategy'
}
s.positions['6758'] = {
    'symbol': '6758',
    'entry_price': 3000.0,
    'quantity': 100,
    'entry_time': '2026-04-01',
    'strategy': 'GCStrategy'
}
s._save_positions()
print('セットアップ完了 positions:', list(s.positions.keys()))

# is_market_open() をモック（深夜・時間外でも通過させる）
original_is_open = s.market_time_manager.is_market_open
s.market_time_manager.is_market_open = lambda: True

# emergency_detector.check_emergency_conditions をモック
# 7203: SL発動（immediate_exit）、9984/6758: 正常
def mock_check_emergency(symbol):
    if symbol == '7203':
        return {
            'is_emergency': True,
            'emergency_level': 1,
            'recommended_action': 'immediate_exit',
            'stop_loss_details': {
                'entry_price': 1000.0,
                'current_price': 960.0,
                'loss_percentage': -0.04
            }
        }
    else:
        return {'is_emergency': False}

original_check = s.emergency_detector.check_emergency_conditions
s.emergency_detector.check_emergency_conditions = mock_check_emergency

# SLチェック実行
print('\nhandle_emergency_switch_check 実行...')
s.handle_emergency_switch_check()

# 結果確認
print('\n実行後 positions:', list(s.positions.keys()))

with open('logs/dssms/positions.json', 'r', encoding='utf-8') as f:
    saved = json.load(f)
print('保存済み positions:', list(saved.keys()))

if '7203' not in saved and '9984' in saved and '6758' in saved:
    print('L3-1 PASS: SL発動銘柄(7203)のみSELL、残り2銘柄はhold維持')
elif '7203' in saved:
    print('L3-1 FAIL: SL発動銘柄(7203)がpositionsに残存（SELLされていない）')
else:
    print('L3-1 FAIL: 意図しない銘柄もpositionsから消えている')

# 後片付け
s.market_time_manager.is_market_open = original_is_open
s.emergency_detector.check_emergency_conditions = original_check
s.positions = {}
s._save_positions()
print('後片付け完了')
