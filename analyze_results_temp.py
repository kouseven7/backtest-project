import pandas as pd
import sys

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

total_trades = len(df)
wins = (df['pnl'] > 0).sum()
losses = (df['pnl'] < 0).sum()
win_rate = wins / total_trades if total_trades > 0 else 0
total_profit = df[df['pnl'] > 0]['pnl'].sum()
total_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
profit_factor = total_profit / total_loss if total_loss > 0 else 0
total_pnl = df['pnl'].sum()

print(f'総取引数: {total_trades}')
print(f'勝ちトレード: {wins}')
print(f'負けトレード: {losses}')
print(f'勝率: {win_rate:.2%}')
print(f'総損益: {total_pnl:,.0f}円')
print(f'総利益: {total_profit:,.0f}円')
print(f'総損失: {total_loss:,.0f}円')
print(f'プロフィットファクター: {profit_factor:.2f}')
print(f'\n使用された戦略:')
print(df['strategy_name'].value_counts())
