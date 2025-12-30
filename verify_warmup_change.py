"""
ウォームアップ期間変更の影響検証スクリプト
2025-12-29
"""
from datetime import datetime, timedelta

print("=" * 60)
print("Option 3 + warmup_days変更の影響検証")
print("=" * 60)

# テストケース: 2025-01-30（実際のエラー発生日）
target_date = datetime(2025, 1, 30)
backtest_start_date = target_date - timedelta(days=1)  # Option 3

print(f"\n[基準日]")
print(f"target_date:          {target_date.date()}")
print(f"backtest_start_date:  {backtest_start_date.date()} (Option 3: -1日)")

print(f"\n[warmup_days=150の場合]")
warmup_150 = backtest_start_date - timedelta(days=150)
print(f"warmup_start:         {warmup_150.date()}")
print(f"計算: {backtest_start_date.date()} - 150日 = {warmup_150.date()}")

print(f"\n[warmup_days=149の場合]")
warmup_149 = backtest_start_date - timedelta(days=149)
print(f"warmup_start:         {warmup_149.date()}")
print(f"計算: {backtest_start_date.date()} - 149日 = {warmup_149.date()}")

print(f"\n[_get_symbol_data()のデータ取得（現状）]")
data_start_current = target_date - timedelta(days=150)  # target_date基準
print(f"データ取得開始日:     {data_start_current.date()}")
print(f"計算: {target_date.date()} - 150日 = {data_start_current.date()}")

print(f"\n[エラー状況の比較]")
print(f"現状（warmup_days=150）:")
print(f"  Required:  {warmup_150.date()}")
print(f"  Available: {data_start_current.date()}")
print(f"  差分:      {(data_start_current - warmup_150).days}日不足 ❌")

print(f"\n変更後（warmup_days=149）:")
print(f"  Required:  {warmup_149.date()}")
print(f"  Available: {data_start_current.date()}")
if warmup_149 == data_start_current:
    print(f"  差分:      0日（一致） ✅")
elif warmup_149 < data_start_current:
    print(f"  差分:      {(data_start_current - warmup_149).days}日余裕 ✅")
else:
    print(f"  差分:      {(warmup_149 - data_start_current).days}日不足 ❌")

print(f"\n[全期間の検証]")
print(f"2025-01-15 ~ 2025-01-31の各日で検証:")
for day in range(15, 32):
    test_date = datetime(2025, 1, day)
    test_backtest_start = test_date - timedelta(days=1)
    test_warmup_149 = test_backtest_start - timedelta(days=149)
    test_data_start = test_date - timedelta(days=150)
    
    if test_warmup_149 <= test_data_start:
        status = "✅"
    else:
        status = "❌"
    
    print(f"  {test_date.date()}: warmup={test_warmup_149.date()}, data={test_data_start.date()} {status}")

print("\n" + "=" * 60)
print("検証完了")
print("=" * 60)
