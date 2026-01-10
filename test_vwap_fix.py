import sys
sys.path.append('C:/Users/imega/Documents/my_backtest_project')
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
import yfinance as yf
import pandas as pd

# 4506のデータ取得
print('データ取得中...')
ticker = yf.Ticker('4506.T')
data = ticker.history(start='2025-06-30', end='2025-11-29', auto_adjust=False)
print(f'データ期間: {data.index[0]} - {data.index[-1]} ({len(data)}日)')

# VWAPBreakoutStrategy初期化
print('VWAPBreakoutStrategy初期化中...')
strategy = VWAPBreakoutStrategy(data)

# 各日付でbacktest_daily()を実行（タイムゾーン考慮）
test_dates = ['2025-11-26', '2025-11-27', '2025-11-28']
print(f"データに含まれる最後の日付: {data.index[-5:]}")

for date_str in test_dates:
    # データ内で利用可能な最後の3つの日付で実際にテスト
    pass
    
# 実際にデータに含まれている最後の3つの日付でテスト
last_3_dates = data.index[-3:]
for test_date in last_3_dates:
    result = strategy.backtest_daily(test_date, data)
    print(f"{test_date.strftime('%Y-%m-%d')}: action={result['action']}, signal={result['signal']}, price={result['price']}, shares={result['shares']}")
    print(f"  reason: {result['reason']}")
    print(f"  current_idx: {data.index.get_loc(test_date)}")