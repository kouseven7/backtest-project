import json
import pandas as pd

# 9101.T_execution_results.jsonからbacktest_signalsを抽出
with open('output/comprehensive_reports/9101.T_20251120_115359/9101.T_execution_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# GCStrategyのbacktest_signalsを取得
gc_strategy_result = data['execution_results'][0]
backtest_signals = gc_strategy_result['backtest_signals']

# DataFrameに変換
signals_df = pd.DataFrame(backtest_signals)

print("=" * 80)
print("タスク2: signals_dfのClose価格確認")
print("=" * 80)

# Date列がある場合は確認
if 'Date' in signals_df.columns:
    print(f"\nDate列の型: {signals_df['Date'].dtype}")
    print(f"先頭5行のDate: {signals_df['Date'].head().tolist()}")

# 4/23と4/30のデータを抽出
target_dates = ['2024-04-23', '2024-04-30']

print("\n4月の該当日付のClose価格（backtest_signals）:")
for date_str in target_dates:
    # Date列から該当日を検索
    matching_rows = signals_df[signals_df['Date'].str.contains(date_str, na=False)]
    
    if not matching_rows.empty:
        for idx, row in matching_rows.iterrows():
            close_price = row.get('Close', None)
            adj_close = row.get('Adj Close', None)
            print(f"  {row['Date']}: Close={close_price:.2f}, Adj Close={adj_close:.2f}")
    else:
        print(f"  {date_str}: データなし")

# 4/24-4/26も確認（Peak値検証用）
print("\n4/24-4/26のClose価格（Peak値推移確認）:")
peak_dates = ['2024-04-24', '2024-04-25', '2024-04-26']
for date_str in peak_dates:
    matching_rows = signals_df[signals_df['Date'].str.contains(date_str, na=False)]
    
    if not matching_rows.empty:
        for idx, row in matching_rows.iterrows():
            close_price = row.get('Close', None)
            print(f"  {row['Date']}: Close={close_price:.2f}")
