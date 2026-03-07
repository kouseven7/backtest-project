import pandas as pd

# CSVファイルを読み込み
df = pd.read_csv('output/dssms_integration/dssms_20260307_071151/all_transactions.csv')

# 統計を計算
total_trades = len(df)
total_pnl = df['pnl'].sum()

# プロフィットファクター計算
wins = df[df['pnl'] > 0]['pnl'].sum()
losses = abs(df[df['pnl'] < 0]['pnl'].sum())
profit_factor = wins / losses if losses > 0 else float('inf')

# 平均利益
avg_pnl = df['pnl'].mean()

# 結果を出力
print("=" * 60)
print("before-force-strategy-main バックテスト結果 (2015-2024)")
print("=" * 60)
print(f"総取引数: {total_trades}")
print(f"純利益: {total_pnl:,.0f}円")
print(f"プロフィットファクター: {profit_factor:.3f}")
print(f"平均利益: {avg_pnl:,.0f}円")
print("=" * 60)
