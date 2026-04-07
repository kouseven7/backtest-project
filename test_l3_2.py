from src.dssms.dssms_scheduler import DSSMSScheduler
from src.dssms import dssms_scheduler as scheduler_module
import pandas as pd
import numpy as np
import datetime
import json

s = DSSMSScheduler()

# Step1: BUY相当の状態をセット
# ※ キーはquantity・entry_time（sharesやentry_dateではない）
print('=== Step1: BUY（ポジション追加） ===')
s.positions['7203'] = {
    'symbol': '7203',
    'entry_price': 1000.0,
    'quantity': 100,
    'entry_time': '2026-04-01T09:00:00',
    'entry_idx': None,
    'strategy': 'GCStrategy'
}
s._save_positions()
print('positions追加:', list(s.positions.keys()))

# Step2: デスクロス状態のデータを準備
# short_window=5, long_window=75 のため200本用意
print('\n=== Step2: EXIT判定 ===')

n = 200
prices = np.concatenate([
    np.linspace(1000, 1400, 130),  # 上昇（GC形成）
    np.linspace(1400,  700,  70)   # 急落（DC形成）
])
dates = pd.date_range(end=datetime.date.today(), periods=n, freq='B')
df = pd.DataFrame({
    'Open':      prices,
    'High':      prices * 1.01,
    'Low':       prices * 0.99,
    'Close':     prices,
    'Adj Close': prices,
    'Volume':    100000
}, index=dates)

# _check_exit_for_positions は get_parameters_and_data() を呼び出すため
# モジュールレベルで置換する
def mock_get_parameters_and_data(ticker, start_date, end_date, warmup_days=150):
    return ticker, start_date, end_date, df, None

original_gpd = scheduler_module.get_parameters_and_data
scheduler_module.get_parameters_and_data = mock_get_parameters_and_data

# EXIT判定実行
s._check_exit_for_positions()

# Step3: 結果確認
print('\n=== Step3: SELL確認 ===')
with open('logs/dssms/positions.json', 'r', encoding='utf-8') as f:
    saved = json.load(f)
print('実行後 positions:', list(saved.keys()))

if '7203' not in saved:
    print('L3-2 PASS: BUY→EXIT判定→SELLの一連フロー正常完結')
else:
    print('L3-2 CONDITIONAL: positionsに7203が残存（holdが返った可能性）')
    print('  → ログでbacktest_daily結果のaction=を確認してください')

# 後片付け
scheduler_module.get_parameters_and_data = original_gpd
if '7203' in s.positions:
    del s.positions['7203']
    s._save_positions()
print('後片付け完了')
