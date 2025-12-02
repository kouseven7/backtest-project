"""
yfinanceのデータ取得動作を確認するテストスクリプト
"""
import yfinance as yf
from datetime import datetime, timedelta

# テスト条件
symbol = "7267.T"
target_date = datetime(2023, 3, 30)
start_date = target_date - timedelta(days=60)

print("=" * 60)
print("yfinanceデータ取得テスト")
print("=" * 60)
print(f"銘柄: {symbol}")
print(f"target_date: {target_date.strftime('%Y-%m-%d')}")
print(f"start_date: {start_date.strftime('%Y-%m-%d')}")
print()

# パターン1: end_date = target_date
print("【パターン1】end_date = target_date")
ticker = yf.Ticker(symbol)
data1 = ticker.history(start=start_date, end=target_date, auto_adjust=False)
print(f"  データ件数: {len(data1)}")
if len(data1) > 0:
    print(f"  最終日: {data1.index[-1]}")
    print(f"  最終日 < target_date: {data1.index[-1] < target_date}")
print()

# パターン2: end_date = target_date + timedelta(days=1)
print("【パターン2】end_date = target_date + 1日")
data2 = ticker.history(start=start_date, end=target_date + timedelta(days=1), auto_adjust=False)
print(f"  データ件数: {len(data2)}")
if len(data2) > 0:
    print(f"  最終日: {data2.index[-1]}")
    print(f"  最終日 >= target_date: {data2.index[-1].date() >= target_date.date()}")
print()

# パターン3: end_date = target_date + timedelta(days=3) (現在の実装)
print("【パターン3】end_date = target_date + 3日 (現在の実装)")
data3 = ticker.history(start=start_date, end=target_date + timedelta(days=3), auto_adjust=False)
print(f"  データ件数: {len(data3)}")
if len(data3) > 0:
    print(f"  最終日: {data3.index[-1]}")
    print(f"  最終日 >= target_date: {data3.index[-1].date() >= target_date.date()}")
print()

print("【結論】")
print("yfinanceのhistory()は、end_dateを「含まない」（exclusive）仕様です。")
print("target_date当日のデータを取得するには、end_date = target_date + 1日が必要です。")
print()
print("ただし、2023-03-30（木曜日）が市場休業日または未来の日付の場合、")
print("その日のデータは取得できません。")
