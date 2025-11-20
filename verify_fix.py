import pandas as pd

df = pd.read_csv('output/comprehensive_reports/9101.T_20251120_121506/portfolio_equity_curve.csv')

max_dd = df['drawdown_pct'].max()
max_dd_date = df.loc[df['drawdown_pct'].idxmax(), 'date']

print("=" * 80)
print("修正後の検証結果")
print("=" * 80)
print(f"\n最大ドローダウン: {max_dd*100:.2f}%")
print(f"発生日: {max_dd_date}")
print(f"\nCash balance変動確認:")
print(f"  Min: {df['cash_balance'].min():,.2f}円")
print(f"  Max: {df['cash_balance'].max():,.2f}円")
print(f"  Unique values: {len(df['cash_balance'].unique())}種類")
print(f"\n期待値: 約2.36-2.38%")
print(f"実績値: {max_dd*100:.2f}%")
print(f"判定: {'OK' if 2.3 <= max_dd*100 <= 2.4 else 'NG'}")
