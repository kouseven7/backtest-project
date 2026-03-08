"""VWAPBreakout最新バックテスト結果の分析"""
import pandas as pd
import glob
import os
from pathlib import Path

# 最新のDSSMSフォルダを取得
output_dir = Path("output/dssms_integration")
folders = sorted(output_dir.glob("dssms_*"), key=os.path.getmtime, reverse=True)

if not folders:
    print("No DSSMS output folders found")
    exit(1)

latest_folder = folders[0]
print(f"=== Latest folder: {latest_folder.name} ===\n")

# all_transactions.csvを読み込み
csv_path = latest_folder / "all_transactions.csv"
if not csv_path.exists():
    print(f"CSV not found: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

# 戦略名の確認
print("=== Strategy Usage ===")
strategy_counts = df['strategy_name'].value_counts()
print(strategy_counts)
print()

# 基本統計
print("=== Summary Statistics ===")
print(f"Total Trades: {len(df)}")
print(f"Total PnL: {df['pnl'].sum():,.2f} yen")

wins = (df['pnl'] > 0).sum()
losses = (df['pnl'] < 0).sum()
print(f"Win count: {wins}")
print(f"Loss count: {losses}")
print(f"Win rate: {wins/len(df)*100:.2f}%")

# プロフィットファクター
gross_profit = df[df['pnl'] > 0]['pnl'].sum()
gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
if gross_loss > 0:
    pf = gross_profit / gross_loss
    print(f"Profit Factor: {pf:.2f}")
else:
    print("Profit Factor: N/A (no losses)")

print(f"\nGross Profit: {gross_profit:,.2f} yen")
print(f"Gross Loss: {-gross_loss:,.2f} yen")
print()

# 年別損益
print("=== Yearly PnL (by entry_date) ===")
df['year'] = pd.to_datetime(df['entry_date']).dt.year
yearly_pnl = df.groupby('year')['pnl'].agg(['sum', 'count'])
yearly_pnl.columns = ['PnL (yen)', 'Trade Count']

for year, row in yearly_pnl.iterrows():
    print(f"{year}: {row['PnL (yen)']:>12,.2f} yen ({int(row['Trade Count'])} trades)")

print(f"\nTotal: {yearly_pnl['PnL (yen)'].sum():>12,.2f} yen ({int(yearly_pnl['Trade Count'].sum())} trades)")
