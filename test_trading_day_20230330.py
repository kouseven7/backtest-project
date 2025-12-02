"""
2023-03-30の取引可否を確認
"""
import yfinance as yf
from datetime import datetime
import pandas as pd

symbol = "7267.T"
target_date = datetime(2023, 3, 30)

print("=" * 60)
print("2023-03-30 取引可否確認")
print("=" * 60)

# 広めの範囲でデータ取得
ticker = yf.Ticker(symbol)
data = ticker.history(
    start=datetime(2023, 3, 27), 
    end=datetime(2023, 4, 3), 
    auto_adjust=False
)

print(f"データ範囲: 2023-03-27 ~ 2023-04-02")
print(f"取得件数: {len(data)}")
print()
print("取得された日付:")
for idx in data.index:
    date_str = idx.strftime('%Y-%m-%d (%a)')
    print(f"  {date_str}")

print()
print("【結論】")
target_date_ts = pd.Timestamp(target_date).tz_localize('Asia/Tokyo')
if target_date_ts in data.index:
    print(f"2023-03-30は取引日です")
else:
    print(f"2023-03-30は取引日ではありません（休業日または未来）")
    
# 曜日確認
print(f"2023-03-30は{target_date.strftime('%A')}（{target_date.strftime('%a')}）")
