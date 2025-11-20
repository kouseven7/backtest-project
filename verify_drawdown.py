import pandas as pd

df = pd.read_csv('output/comprehensive_reports/9101.T_20251120_114716/portfolio_equity_curve.csv')

max_dd = df['drawdown_pct'].max()
max_dd_idx = df['drawdown_pct'].idxmax()
max_dd_date = df.loc[max_dd_idx, 'date']
portfolio_value = df.loc[max_dd_idx, 'portfolio_value']
peak_value = df.loc[max_dd_idx, 'peak_value']

print(f'Max Drawdown: {max_dd*100:.2f}%')
print(f'Date: {max_dd_date}')
print(f'Portfolio Value: {portfolio_value:,.2f}')
print(f'Peak Value: {peak_value:,.2f}')
print(f'\n前回（18日分データ時）: 2.48%')
print(f'今回（245日分データ時）: {max_dd*100:.2f}%')
print(f'差分: {(2.48 - max_dd*100):.2f}%')
