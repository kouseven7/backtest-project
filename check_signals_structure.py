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
print("タスク2: signals_dfの構造確認")
print("=" * 80)

print(f"\n行数: {len(signals_df)}")
print(f"\nカラム名: {signals_df.columns.tolist()}")
print(f"\nデータ型:\n{signals_df.dtypes}")
print(f"\n先頭5行:\n{signals_df.head()}")
print(f"\nインデックス型: {type(signals_df.index)}")
print(f"\nインデックス先頭5個: {signals_df.index[:5].tolist()}")
