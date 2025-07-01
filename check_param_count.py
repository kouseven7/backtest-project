"""
パラメータ組み合わせ数を確認するスクリプト
"""
from optimization.configs.vwap_breakout_optimization import PARAM_GRID

# パラメータ組み合わせ数を計算
total = 1
param_counts = {}

for param, values in PARAM_GRID.items():
    param_count = len(values)
    total *= param_count
    param_counts[param] = param_count
    print(f"パラメータ '{param}': {param_count}通りの値")

print(f"\nパラメータ組み合わせ総数: {total}通り")
