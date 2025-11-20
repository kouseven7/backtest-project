import pandas as pd

df = pd.read_csv('output/comprehensive_reports/5803.T_20251120_135502/5803.T_trades.csv')

# Calculate win/loss stats
winning_trades = df[df['pnl'] > 0]
losing_trades = df[df['pnl'] < 0]

print('=== Win Rate Verification ===')
print(f'Total trades: {len(df)}')
print(f'Winning trades: {len(winning_trades)}')
print(f'Losing trades: {len(losing_trades)}')
calculated_win_rate = len(winning_trades) / len(df)
print(f'Calculated win rate: {calculated_win_rate:.4f} ({calculated_win_rate*100:.2f}%)')
print()

# Verify against performance_summary.csv
perf = pd.read_csv('output/comprehensive_reports/5803.T_20251120_135502/5803.T_performance_summary.csv')
reported_win_rate = perf[perf['Metric'] == 'win_rate']['Value'].values[0]
print(f'Reported win rate: {reported_win_rate:.4f} ({reported_win_rate*100:.2f}%)')
print(f'Match: {abs(calculated_win_rate - reported_win_rate) < 0.0001}')
print()

# Calculate total profit/loss
total_profit = winning_trades['pnl'].sum()
total_loss = abs(losing_trades['pnl'].sum())
net_profit = df['pnl'].sum()

print('=== Profit/Loss Verification ===')
print(f'Total profit: {total_profit:.2f}')
print(f'Total loss: {total_loss:.2f}')
print(f'Net profit: {net_profit:.2f}')
print()

reported_total_profit = perf[perf['Metric'] == 'total_profit']['Value'].values[0]
reported_total_loss = perf[perf['Metric'] == 'total_loss']['Value'].values[0]
reported_net_profit = perf[perf['Metric'] == 'net_profit']['Value'].values[0]

print(f'Reported total profit: {reported_total_profit:.2f}')
print(f'Reported total loss: {reported_total_loss:.2f}')
print(f'Reported net profit: {reported_net_profit:.2f}')
print()

print(f'Profit match: {abs(total_profit - reported_total_profit) < 0.01}')
print(f'Loss match: {abs(total_loss - reported_total_loss) < 0.01}')
print(f'Net profit match: {abs(net_profit - reported_net_profit) < 0.01}')
print()

# Calculate profit factor
calculated_pf = total_profit / total_loss if total_loss > 0 else 0
reported_pf = perf[perf['Metric'] == 'profit_factor']['Value'].values[0]

print('=== Profit Factor Verification ===')
print(f'Calculated: {calculated_pf:.4f}')
print(f'Reported: {reported_pf:.4f}')
print(f'Match: {abs(calculated_pf - reported_pf) < 0.0001}')
print()

# Max drawdown verification
print('=== Max Drawdown Verification ===')
reported_max_dd = perf[perf['Metric'] == 'max_loss']['Value'].values[0]
calculated_max_loss = losing_trades['pnl'].min()
print(f'Calculated max loss: {calculated_max_loss:.2f}')
print(f'Reported max loss: {reported_max_dd:.2f}')
print(f'Match: {abs(calculated_max_loss - reported_max_dd) < 0.01}')
