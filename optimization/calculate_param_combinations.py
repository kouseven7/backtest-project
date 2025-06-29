"""
パラメータ組み合わせ数を計算するスクリプト
"""
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# 新しい設定ファイルをインポート
from optimization.configs.vwap_breakout_optimization_reduced import PARAM_GRID

# パラメータ数を計算
total_combinations = 1
for param_name, param_values in PARAM_GRID.items():
    print(f"{param_name}: {len(param_values)} values - {param_values}")
    total_combinations *= len(param_values)

print(f"\n総パラメータ組み合わせ数: {total_combinations:,}通り")

# パラメータの要素数を確認
param_counts = {}
for param, values in PARAM_GRID.items():
    param_counts[param] = len(values)

print("\nパラメータごとの組み合わせ数:")
for param, count in param_counts.items():
    print(f"{param}: {count}通り")
