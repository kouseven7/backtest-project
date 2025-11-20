import pandas as pd

# portfolio_equity_curve.csvを読み込み
csv_path = 'output/comprehensive_reports/9101.T_20251120_115359/portfolio_equity_curve.csv'
df = pd.read_csv(csv_path)

print("=" * 80)
print("タスク4: CSV出力値との整合性確認")
print("=" * 80)

# 4/23と4/30のデータを確認
target_dates = ['2024-04-23', '2024-04-30']

print("\n4月の該当日付のCSV実データ:")
for date_str in target_dates:
    matching_rows = df[df['date'].str.contains(date_str, na=False)]
    
    if not matching_rows.empty:
        for idx, row in matching_rows.iterrows():
            print(f"\n{row['date']}:")
            print(f"  portfolio_value: {row['portfolio_value']:,.2f}")
            print(f"  cash_balance: {row['cash_balance']:,.2f}")
            print(f"  position_value: {row['position_value']:,.2f}")
            print(f"  peak_value: {row['peak_value']:,.2f}")
            print(f"  drawdown_pct: {row['drawdown_pct']*100:.2f}%")
            print(f"  snapshot_type: {row['snapshot_type']}")

# 4/24-4/26も確認（Peak値推移）
print("\n4/24-4/26のCSV実データ（Peak値推移）:")
peak_dates = ['2024-04-24', '2024-04-25', '2024-04-26']
for date_str in peak_dates:
    matching_rows = df[df['date'].str.contains(date_str, na=False)]
    
    if not matching_rows.empty:
        for idx, row in matching_rows.iterrows():
            print(f"\n{row['date']}:")
            print(f"  portfolio_value: {row['portfolio_value']:,.2f}")
            print(f"  cash_balance: {row['cash_balance']:,.2f}")
            print(f"  position_value: {row['position_value']:,.2f}")
            print(f"  peak_value: {row['peak_value']:,.2f}")

print("\n" + "=" * 80)
print("検証: cash_balanceが1,000,000のまま変わらないか?")
print("=" * 80)

# 全行のcash_balanceをユニーク値で確認
unique_cash = df['cash_balance'].unique()
print(f"\ncash_balanceのユニーク値: {unique_cash.tolist()}")
print(f"全て1,000,000か?: {(unique_cash == 1000000.0).all()}")
