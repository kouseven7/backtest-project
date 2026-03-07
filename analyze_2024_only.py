import pandas as pd

# 2024年の取引データを集計
df = pd.read_csv('output/dssms_integration/dssms_20260307_071151/all_transactions.csv')
df['entry_date'] = pd.to_datetime(df['entry_date'])
df_2024 = df[df['entry_date'].dt.year == 2024]

print(f"2024年取引数: {len(df_2024)}件")
print(f"2024年総損益: {df_2024['pnl'].sum():,.0f}円")
print(f"2024年総利益: {df_2024[df_2024['pnl'] > 0]['pnl'].sum():,.0f}円")
print(f"2024年総損失: {df_2024[df_2024['pnl'] < 0]['pnl'].sum():,.0f}円")
print(f"2024年勝ちトレード: {(df_2024['pnl'] > 0).sum()}件")
print(f"2024年負けトレード: {(df_2024['pnl'] < 0).sum()}件")
print(f"2024年勝率: {(df_2024['pnl'] > 0).sum() / len(df_2024) * 100:.2f}%")
print(f"\n2024年使用戦略:")
print(df_2024['strategy_name'].value_counts())
