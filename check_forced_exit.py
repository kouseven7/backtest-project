import pandas as pd

df = pd.read_csv('output/comprehensive_reports/5803.T_20251120_135502/5803.T_trades.csv')
forced = df[df['is_forced_exit'] == True]

print(f'Total trades: {len(df)}')
print(f'Forced exits: {len(forced)}')
print()
print('Forced exit details:')
print(f'Exit date: {forced.iloc[0]["exit_date"]}')
print(f'Entry date: {forced.iloc[0]["entry_date"]}')
print(f'Strategy: {forced.iloc[0]["strategy"]}')
print(f'Holding period: {int(forced.iloc[0]["holding_period_days"])} days')
print(f'PnL: {forced.iloc[0]["pnl"]:.2f}')
