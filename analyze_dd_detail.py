import pandas as pd

df = pd.read_csv('output/comprehensive_reports/9101.T_20251120_121506/portfolio_equity_curve.csv')

print("=" * 80)
print("ドローダウン詳細分析")
print("=" * 80)

# 最大ドローダウン箇所
max_dd_idx = df['drawdown_pct'].idxmax()
print(f"\n最大ドローダウン箇所（行{max_dd_idx}）:")
print(f"  Date: {df.loc[max_dd_idx, 'date']}")
print(f"  Portfolio Value: {df.loc[max_dd_idx, 'portfolio_value']:,.2f}円")
print(f"  Cash Balance: {df.loc[max_dd_idx, 'cash_balance']:,.2f}円")
print(f"  Position Value: {df.loc[max_dd_idx, 'position_value']:,.2f}円")
print(f"  Peak Value: {df.loc[max_dd_idx, 'peak_value']:,.2f}円")
print(f"  Drawdown: {df.loc[max_dd_idx, 'drawdown_pct']*100:.2f}%")
print(f"  Snapshot Type: {df.loc[max_dd_idx, 'snapshot_type']}")

# Peak値の推移を確認
print("\n\nPeak値の変遷:")
peak_changes = df[df['peak_value'].diff() != 0].head(15)
for idx, row in peak_changes.iterrows():
    print(f"  [{idx}] {row['date']}: Peak={row['peak_value']:,.2f}, Portfolio={row['portfolio_value']:,.2f}, Cash={row['cash_balance']:,.2f}")

# ドローダウン上位5件
print("\n\nドローダウン上位5件:")
top_dd = df.nlargest(5, 'drawdown_pct')
for idx, row in top_dd.iterrows():
    print(f"  {row['date']}: {row['drawdown_pct']*100:.2f}% (Portfolio={row['portfolio_value']:,.2f}, Peak={row['peak_value']:,.2f})")
