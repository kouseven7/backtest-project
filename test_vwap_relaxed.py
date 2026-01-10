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

# VWAPBreakoutStrategy初期化（緩い条件）
print('VWAPBreakoutStrategy初期化中（緩い条件）...')
relaxed_params = {
    'volume_threshold': 0.8,  # 出来高減少も許可
    'breakout_min_percent': 0.001,  # 最小ブレイク率を小さく
    'market_filter_method': 'none',  # 市場フィルター無効
    'confirmation_bars': 0  # 確認バー数なし（即時エントリー）
}
strategy = VWAPBreakoutStrategy(data, params=relaxed_params)

# 実際にデータに含まれている最後の5つの日付でテスト
last_5_dates = data.index[-5:]
for test_date in last_5_dates:
    result = strategy.backtest_daily(test_date, data)
    print(f"{test_date.strftime('%Y-%m-%d')}: action={result['action']}, signal={result['signal']}, price={result['price']:.2f}, shares={result['shares']}")
    if result['action'] != 'hold':
        print(f"  [SUCCESS] エントリー/エグジット発生！ reason: {result['reason']}")
    else:
        print(f"  reason: {result['reason']}")