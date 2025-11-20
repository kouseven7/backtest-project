import pandas as pd

df = pd.read_csv('output/comprehensive_reports/5803.T_20251120_135502/5803.T_trades.csv')

print('=== Trade 1 Verification ===')
print(f'Entry: {df.iloc[0]["entry_price"]:.2f} x {int(df.iloc[0]["shares"])} shares')
print(f'Exit: {df.iloc[0]["exit_price"]:.2f} x {int(df.iloc[0]["shares"])} shares')
print(f'PnL (CSV): {df.iloc[0]["pnl"]:.2f}')
manual_pnl = (df.iloc[0]['exit_price'] - df.iloc[0]['entry_price']) * df.iloc[0]['shares']
print(f'PnL (manual): {manual_pnl:.2f}')
print(f'Difference: {abs(df.iloc[0]["pnl"] - manual_pnl):.6f}')
print()

print('=== Trade 2 Verification ===')
print(f'Entry: {df.iloc[1]["entry_price"]:.2f} x {int(df.iloc[1]["shares"])} shares')
print(f'Exit: {df.iloc[1]["exit_price"]:.2f} x {int(df.iloc[1]["shares"])} shares')
print(f'PnL (CSV): {df.iloc[1]["pnl"]:.2f}')
manual_pnl2 = (df.iloc[1]['exit_price'] - df.iloc[1]['entry_price']) * df.iloc[1]['shares']
print(f'PnL (manual): {manual_pnl2:.2f}')
print(f'Difference: {abs(df.iloc[1]["pnl"] - manual_pnl2):.6f}')
print()

print('=== Last Trade (Forced Close) Verification ===')
print(f'Entry: {df.iloc[-1]["entry_price"]:.2f} x {int(df.iloc[-1]["shares"])} shares')
print(f'Exit: {df.iloc[-1]["exit_price"]:.2f} x {int(df.iloc[-1]["shares"])} shares')
print(f'PnL (CSV): {df.iloc[-1]["pnl"]:.2f}')
manual_pnl_last = (df.iloc[-1]['exit_price'] - df.iloc[-1]['entry_price']) * df.iloc[-1]['shares']
print(f'PnL (manual): {manual_pnl_last:.2f}')
print(f'Difference: {abs(df.iloc[-1]["pnl"] - manual_pnl_last):.6f}')
print(f'Is forced exit: {df.iloc[-1]["is_forced_exit"]}')
