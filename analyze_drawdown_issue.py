import pandas as pd

df = pd.read_csv('output/comprehensive_reports/9101.T_20251120_114716/portfolio_equity_curve.csv')

# 最大ドローダウン発生箇所の詳細調査
max_dd_idx = df['drawdown_pct'].idxmax()

print("=== 最大ドローダウン発生箇所 ===")
print(f"Index: {max_dd_idx}")
print(f"Date: {df.loc[max_dd_idx, 'date']}")
print(f"Drawdown: {df.loc[max_dd_idx, 'drawdown_pct']*100:.2f}%")
print(f"Portfolio Value: {df.loc[max_dd_idx, 'portfolio_value']:,.2f}")
print(f"Peak Value: {df.loc[max_dd_idx, 'peak_value']:,.2f}")
print(f"Snapshot Type: {df.loc[max_dd_idx, 'snapshot_type']}")
print()

# 前後10行を確認
print("=== 前後10行の状況 ===")
start_idx = max(0, max_dd_idx - 10)
end_idx = min(len(df), max_dd_idx + 11)

for i in range(start_idx, end_idx):
    row = df.loc[i]
    marker = ">>> " if i == max_dd_idx else "    "
    print(f"{marker}[{i}] {row['date']}: Value={row['portfolio_value']:,.2f}, Peak={row['peak_value']:,.2f}, DD={row['drawdown_pct']*100:.2f}%, Type={row['snapshot_type']}")

print()

# Peak値が異常に高い理由を調査
print("=== Peak値の推移 ===")
peak_changes = df[df['peak_value'].diff() != 0].head(20)
for idx, row in peak_changes.iterrows():
    print(f"[{idx}] {row['date']}: Peak={row['peak_value']:,.2f}, Portfolio={row['portfolio_value']:,.2f}, Type={row['snapshot_type']}")
