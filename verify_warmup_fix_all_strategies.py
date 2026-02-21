"""
ウォームアップ期間エントリー問題の検証スクリプト（全戦略対応版）

修正完了後の検証:
1. 期間外エントリー（2024-01-01より前）が0件
2. 戦略別エントリー数の確認
3. 各戦略が正常動作しているか確認
"""

import pandas as pd
from datetime import datetime
import glob
import os

# 最新のDSSMS出力ディレクトリを取得
output_dirs = glob.glob("output/dssms_integration/dssms_*")
if not output_dirs:
    print("❌ エラー: DSSMS出力ディレクトリが見つかりません")
    exit(1)

latest_dir = max(output_dirs, key=os.path.getmtime)
csv_path = os.path.join(latest_dir, "all_transactions.csv")

if not os.path.exists(csv_path):
    print(f"❌ エラー: {csv_path}が見つかりません")
    exit(1)

print(f"検証対象: {csv_path}")
print("=" * 80)

# CSVロード
df = pd.read_csv(csv_path)
df['entry_date'] = pd.to_datetime(df['entry_date'])

# バックテスト期間
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 31)

# 1. 期間外エントリー確認
print("\n[検証1] 期間外エントリー確認")
print("=" * 80)
out_of_range = df[df['entry_date'] < start_date]

if len(out_of_range) > 0:
    print(f"❌ 失敗: 期間外エントリーが{len(out_of_range)}件検出されました")
    print("\n期間外エントリー詳細:")
    print(out_of_range[['symbol', 'entry_date', 'entry_price', 'strategy_name', 'exit_date', 'pnl']])
    print("\n【重要】期間外エントリーが存在します。修正が不完全です。")
else:
    print("✅ 成功: 期間外エントリー0件")
    print(f"全{len(df)}件のエントリーが{start_date.strftime('%Y-%m-%d')}以降です。")

# 2. 戦略別エントリー数確認
print("\n[検証2] 戦略別エントリー数")
print("=" * 80)
strategy_counts = df['strategy_name'].value_counts()
print(strategy_counts)

if len(strategy_counts) == 0:
    print("\n⚠️ 警告: エントリーが1件もありません。戦略が動作していない可能性があります。")
else:
    print(f"\n✅ {len(strategy_counts)}種類の戦略が動作中")

# 3. 期間内エントリー確認
print("\n[検証3] 期間内エントリー詳細")
print("=" * 80)
in_range = df[(df['entry_date'] >= start_date) & (df['entry_date'] <= end_date)]
print(f"期間内エントリー: {len(in_range)}件")

if len(in_range) > 0:
    print("\nエントリー日時範囲:")
    print(f"  最初: {in_range['entry_date'].min()}")
    print(f"  最後: {in_range['entry_date'].max()}")
    
    print("\n銘柄別エントリー数:")
    print(in_range['symbol'].value_counts().head(10))
    
    print("\n戦略別統計:")
    for strategy in in_range['strategy_name'].unique():
        strategy_df = in_range[in_range['strategy_name'] == strategy]
        avg_pnl = strategy_df['pnl'].mean() if 'pnl' in strategy_df.columns else 0
        print(f"  {strategy}: {len(strategy_df)}件, 平均損益={avg_pnl:,.0f}円")

# 4. 最終判定
print("\n" + "=" * 80)
print("[最終判定]")
print("=" * 80)

all_passed = True
if len(out_of_range) > 0:
    print("❌ 失敗: 期間外エントリーが存在します")
    all_passed = False
else:
    print("✅ 期間外エントリーチェック: PASS")

if len(df) == 0:
    print("⚠️ 警告: エントリーが1件もありません")
    all_passed = False
else:
    print(f"✅ エントリー件数チェック: PASS（{len(df)}件）")

if all_passed and len(df) > 0:
    print("\n🎉 全ての検証に合格しました！")
    print("ウォームアップ期間フィルタリングが正常に動作しています。")
else:
    print("\n⚠️ 一部の検証に合格していません。")
    print("詳細を確認してください。")
