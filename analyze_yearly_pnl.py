import pandas as pd

# CSVファイルを読み込み
df = pd.read_csv('output/dssms_integration/dssms_20260307_220239/all_transactions.csv')

# exit_dateを日付型に変換
df['exit_date'] = pd.to_datetime(df['exit_date'])

# 年を抽出
df['year'] = df['exit_date'].dt.year

# 年ごとの集計
yearly_stats = df.groupby('year').agg({
    'pnl': ['sum', 'count', 'mean'],
    'symbol': 'count'
}).round(0)

yearly_stats.columns = ['純利益', '取引数', '平均利益', '取引数2']
yearly_stats = yearly_stats[['純利益', '取引数', '平均利益']]

print("=" * 80)
print("年別損益レポート (2015-2024)")
print("=" * 80)
print()

# 年ごとの結果を表示
for year in sorted(df['year'].unique()):
    year_data = df[df['year'] == year]
    total_pnl = year_data['pnl'].sum()
    trade_count = len(year_data)
    avg_pnl = year_data['pnl'].mean()
    
    # プラスかマイナスかを判定
    if total_pnl >= 0:
        status = "[+] プラス"
        symbol = "+"
    else:
        status = "[-] マイナス"
        symbol = ""
    
    print(f"{year}年: {status}")
    print(f"  純利益: {symbol}{total_pnl:>12,.0f}円")
    print(f"  取引数: {trade_count:>5}件")
    print(f"  平均利益: {symbol}{avg_pnl:>10,.0f}円")
    print()

print("=" * 80)
print("総合統計")
print("=" * 80)
total_pnl = df['pnl'].sum()
total_trades = len(df)
avg_pnl = df['pnl'].mean()

print(f"総純利益: {total_pnl:>15,.0f}円")
print(f"総取引数: {total_trades:>5}件")
print(f"平均利益: {avg_pnl:>15,.0f}円")
print()

# プラスとマイナスの年数をカウント
positive_years = len(yearly_stats[yearly_stats['純利益'] >= 0])
negative_years = len(yearly_stats[yearly_stats['純利益'] < 0])

print(f"プラスの年: {positive_years}年")
print(f"マイナスの年: {negative_years}年")
print("=" * 80)
