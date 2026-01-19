"""
Task 4代替: GC戦略取引パターン簡易分析
yfinanceのMA計算を使わず、既存CSVから統計を抽出
"""

import pandas as pd

# all_transactions.csvを読み込み
df = pd.read_csv('output/dssms_integration/dssms_20260116_133050/all_transactions.csv')

# 未決済取引を除外
df = df[df['exit_date'].notna()].copy()

# entry_date, exit_dateをdatetime変換
df['entry_date'] = pd.to_datetime(df['entry_date'])
df['exit_date'] = pd.to_datetime(df['exit_date'])

# 月別に分類
df['entry_month'] = df['entry_date'].dt.to_period('M')

print('=== GC戦略エントリー条件緩和効果検証 ===\n')
print(f'総取引数: {len(df)}件（決済済み）')
print(f'総損益: {df["pnl"].sum():.0f}円')
print(f'平均損益: {df["pnl"].mean():.0f}円')
print(f'平均リターン: {df["return_pct"].mean()*100:.2f}%')
print(f'勝率: {(df["pnl"] > 0).sum() / len(df) * 100:.1f}%')

print('\n=== 月別取引数 ===')
print(df.groupby('entry_month').size())

print('\n=== 銘柄別取引数 ===')
print(df['symbol'].value_counts().head(10))

print('\n=== 銘柄別平均損益 ===')
symbol_pnl = df.groupby('symbol')['pnl'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print(symbol_pnl.head(10))

print('\n=== 保有期間別統計 ===')
print(df['holding_period_days'].describe())

print('\n=== 損益分布 ===')
profit_trades = df[df['pnl'] > 0]
loss_trades = df[df['pnl'] < 0]
print(f'利益取引: {len(profit_trades)}件（平均+{profit_trades["pnl"].mean():.0f}円）')
print(f'損失取引: {len(loss_trades)}件（平均{loss_trades["pnl"].mean():.0f}円）')
if len(loss_trades) > 0:
    print(f'損益比: {abs(profit_trades["pnl"].mean() / loss_trades["pnl"].mean()):.2f}')

print('\n=== スイッチ強制決済の影響 ===')
forced = df[df['is_forced_exit']]
normal = df[~df['is_forced_exit']]
print(f'スイッチ決済: {len(forced)}件（平均{forced["pnl"].mean():.0f}円）')
print(f'通常決済: {len(normal)}件（平均{normal["pnl"].mean():.0f}円）')

print('\n=== 大損失取引トップ5 ===')
worst_trades = df.nsmallest(5, 'pnl')[['symbol', 'entry_date', 'exit_date', 'pnl', 'return_pct', 'holding_period_days', 'is_forced_exit']]
print(worst_trades.to_string())

print('\n=== 大利益取引トップ5 ===')
best_trades = df.nlargest(5, 'pnl')[['symbol', 'entry_date', 'exit_date', 'pnl', 'return_pct', 'holding_period_days', 'is_forced_exit']]
print(best_trades.to_string())

# CSV出力
output_summary = pd.DataFrame({
    '項目': ['総取引数', '総損益', '平均損益', '勝率', '損益比'],
    '値': [
        f'{len(df)}件',
        f'{df["pnl"].sum():.0f}円',
        f'{df["pnl"].mean():.0f}円',
        f'{(df["pnl"] > 0).sum() / len(df) * 100:.1f}%',
        f'{abs(profit_trades["pnl"].mean() / loss_trades["pnl"].mean()):.2f}' if len(loss_trades) > 0 else 'N/A'
    ]
})
output_summary.to_csv('output/dssms_integration/dssms_20260116_133050/task4_summary.csv', index=False, encoding='utf-8-sig')
print('\n[OUTPUT] task4_summary.csv 生成完了')
