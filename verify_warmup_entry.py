"""
ウォームアップ期間エントリー検証スクリプト（Issue調査報告20260210対応）

バックテスト期間外のエントリーが記録されていないことを検証する。
"""
import pandas as pd
from datetime import datetime
from pathlib import Path

# CSVファイルパス
csv_path = Path("output/dssms_integration/dssms_20260210_223512/all_transactions.csv")

if not csv_path.exists():
    print(f"❌ エラー: CSVファイルが見つかりません: {csv_path}")
    exit(1)

# CSV読み込み
df = pd.read_csv(csv_path)
df['entry_date'] = pd.to_datetime(df['entry_date'])

# バックテスト開始日
start_date = datetime(2024, 1, 1)

print("=" * 60)
print("ウォームアップ期間エントリー検証")
print("=" * 60)
print(f"バックテスト開始日: {start_date.strftime('%Y-%m-%d')}")
print(f"検証対象: {csv_path}")
print()

# 期間外エントリー確認
out_of_range_entries = df[df['entry_date'] < start_date]

if len(out_of_range_entries) > 0:
    print(f"❌ 失敗: 期間外エントリーが{len(out_of_range_entries)}件検出")
    print()
    print("期間外エントリー詳細:")
    print(out_of_range_entries[['symbol', 'entry_date', 'entry_price', 'strategy_name']])
    print()
else:
    print("✅ 成功: 期間外エントリー0件")
    print()

# 通常期間のエントリー確認
normal_entries = df[df['entry_date'] >= start_date]
print(f"通常期間エントリー: {len(normal_entries)}件")
print()

if len(normal_entries) > 0:
    print("エントリー詳細:")
    for idx, row in normal_entries.iterrows():
        entry_date = row['entry_date'].strftime('%Y-%m-%d')
        print(f"  - {row['symbol']}: {entry_date} ({row['strategy_name']})")

print()
print("=" * 60)
print("検証完了")
print("=" * 60)
