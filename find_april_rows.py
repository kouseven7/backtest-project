import json
import pandas as pd

# 9101.T_execution_results.jsonからbacktest_signalsを抽出
with open('output/comprehensive_reports/9101.T_20251120_115359/9101.T_execution_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# GCStrategyのbacktest_signalsを取得（最初の10行を確認）
gc_strategy_result = data['execution_results'][0]
backtest_signals = gc_strategy_result['backtest_signals']

print("=" * 80)
print("タスク2: backtest_signals（JSON）の生データ確認")
print("=" * 80)

print(f"\n総行数: {len(backtest_signals)}")
print(f"\n最初の3行（JSON形式）:")
for i in range(min(3, len(backtest_signals))):
    print(f"\n行{i}:")
    for key, value in backtest_signals[i].items():
        print(f"  {key}: {value}")

# Entry_Signal==1の行を検索（4/23がどの行か特定）
print("\n\nEntry_Signal==1の行:")
for i, row in enumerate(backtest_signals):
    if row.get('Entry_Signal') == 1:
        print(f"\n行{i}: Entry_Signal==1")
        print(f"  Close: {row.get('Close'):.2f}")
        print(f"  Adj Close: {row.get('Adj Close'):.2f}")
        print(f"  ExecutedPrice: {row.get('ExecutedPrice'):.2f}")
        print(f"  ExecutedQuantity: {row.get('ExecutedQuantity'):.2f}")
        print(f"  Position: {row.get('Position')}")
        
        # この行が4/23か判定（ExecutedPrice==3812.93なら4/23）
        if abs(row.get('ExecutedPrice', 0) - 3812.93) < 1:
            print("  >>> これが4/23のBUY行と推定")

# Exit_Signal==-1の行を検索（4/30がどの行か特定）
print("\n\nExit_Signal==-1の行:")
for i, row in enumerate(backtest_signals):
    if row.get('Exit_Signal') == -1:
        print(f"\n行{i}: Exit_Signal==-1")
        print(f"  Close: {row.get('Close'):.2f}")
        print(f"  Adj Close: {row.get('Adj Close'):.2f}")
        print(f"  ExecutedPrice: {row.get('ExecutedPrice'):.2f}")
        print(f"  ExecutedQuantity: {row.get('ExecutedQuantity'):.2f}")
        print(f"  Position: {row.get('Position')}")
        
        # この行が4/30か判定（ExecutedPrice==4104.72なら4/30）
        if abs(row.get('ExecutedPrice', 0) - 4104.72) < 1:
            print("  >>> これが4/30のSELL行と推定")
