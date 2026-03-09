"""最新DSSMS結果の簡易分析"""
import pandas as pd
from pathlib import Path
import os

# 最新フォルダ取得
output_dir = Path("output/dssms_integration")
folders = sorted(output_dir.glob("dssms_*"), key=os.path.getmtime, reverse=True)
latest = folders[0]
csv_path = latest / "all_transactions.csv"

print(f"Folder: {latest.name}")
print()

df = pd.read_csv(csv_path)

print("=== Strategy Counts ===")
print(df['strategy_name'].value_counts())
print()

print(f"Total Trades: {len(df)}")
print(f"Total PnL: {df['pnl'].sum():,.2f} yen")
print(f"Win Rate: {(df['pnl'] > 0).sum() / len(df) * 100:.2f}%")

gross_profit = df[df['pnl'] > 0]['pnl'].sum()
gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
print(f"Profit Factor: {pf:.2f}")
print()

print("=== Yearly PnL ===")
df['year'] = pd.to_datetime(df['entry_date']).dt.year
yearly = df.groupby('year')['pnl'].agg(['sum', 'count'])
for year, row in yearly.iterrows():
    print(f"{year}: {row['sum']:>12,.2f} yen ({int(row['count'])} trades)")
