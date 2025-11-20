import pandas as pd

df = pd.read_csv('output/comprehensive_reports/9101.T_20251120_121506/portfolio_equity_curve.csv')

# CSV実測値
max_dd_pct_csv = df['drawdown_pct'].max() * 100
max_dd_idx = df['drawdown_pct'].idxmax()
max_dd_date = df.loc[max_dd_idx, 'date']
portfolio_at_max_dd = df.loc[max_dd_idx, 'portfolio_value']
peak_at_max_dd = df.loc[max_dd_idx, 'peak_value']
dd_amount = peak_at_max_dd - portfolio_at_max_dd

# ログ表示から逆算（バックテスト実行時のターミナル出力）
# 最大ドローダウン: 2.36%
log_dd_pct = 2.36

print("=" * 80)
print("ドローダウン比較分析")
print("=" * 80)

print(f"\n【CSV実測値】（portfolio_equity_curve.csv）")
print(f"  最大ドローダウン: {max_dd_pct_csv:.2f}%")
print(f"  発生日: {max_dd_date}")
print(f"  Portfolio Value: {portfolio_at_max_dd:,.2f}円")
print(f"  Peak Value: {peak_at_max_dd:,.2f}円")
print(f"  ドローダウン金額: {dd_amount:,.2f}円")

print(f"\n【ログ表示値】（バックテスト実行時）")
print(f"  最大ドローダウン: {log_dd_pct:.2f}%")
print(f"  （ComprehensivePerformanceAnalyzerの計算結果）")

print(f"\n【TXT表示値】（main_comprehensive_report）")
print(f"  最大ドローダウン: ¥23,635")
print(f"  （金額ベース）")

print(f"\n【差分分析】")
print(f"  CSV - ログ: {max_dd_pct_csv - log_dd_pct:.2f}%")
print(f"  CSV金額: {dd_amount:,.2f}円")
print(f"  TXT金額: ¥23,635")
print(f"  金額差分: {dd_amount - 23635:,.2f}円")

print(f"\n【計算検証】")
print(f"  CSV ドローダウン率: ({peak_at_max_dd:,.2f} - {portfolio_at_max_dd:,.2f}) / {peak_at_max_dd:,.2f} = {dd_amount / peak_at_max_dd * 100:.4f}%")
print(f"  TXT ドローダウン率: 23,635 / {peak_at_max_dd:,.2f} = {23635 / peak_at_max_dd * 100:.4f}%")

print(f"\n【結論】")
if abs(dd_amount - 23635) < 10:
    print(f"  金額は一致（差分{dd_amount - 23635:,.2f}円）")
    print(f"  パーセント計算に差異あり（CSV={max_dd_pct_csv:.2f}%, TXT={23635/peak_at_max_dd*100:.2f}%）")
else:
    print(f"  金額が異なる：CSV={dd_amount:,.2f}円 vs TXT=¥23,635")
    print(f"  異なるドローダウンポイントを参照している可能性")

# Peak値の履歴を確認
print(f"\n\n【Peak値の履歴確認】")
peak_changes = df[df['peak_value'].diff() != 0]
print(f"  Peak値の更新回数: {len(peak_changes)}回")
print(f"  初期Peak: {df.loc[0, 'peak_value']:,.2f}円")
print(f"  最終Peak: {df.loc[len(df)-1, 'peak_value']:,.2f}円")
print(f"  最大Peak: {df['peak_value'].max():,.2f}円")
