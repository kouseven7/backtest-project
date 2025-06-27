"""
パラメータグリッドの組み合わせ総数を計算するスクリプト
"""
import sys
import math
from optimization.configs.vwap_breakout_optimization import PARAM_GRID

def calculate_combinations(grid):
    """
    パラメータグリッドの全組み合わせ数を計算する
    """
    total_combinations = 1
    for param_name, param_values in grid.items():
        num_values = len(param_values)
        total_combinations *= num_values
        print(f"{param_name}: {num_values}通り - {param_values}")
    
    return total_combinations

if __name__ == "__main__":
    # 組み合わせ総数を計算
    combinations = calculate_combinations(PARAM_GRID)
    print(f"\n総組み合わせ数: {combinations:,}")
    
    # 10,000以下であることを確認
    if combinations <= 10000:
        print("✓ 組み合わせ数は10,000以下です")
    else:
        print(f"! 組み合わせ数が10,000を超えています。削減してください ({combinations:,} > 10,000)")
        
    # 高重要度パラメータのみの組み合わせ数
    high_importance_params = {
        k: v for k, v in PARAM_GRID.items() 
        if k in ["stop_loss", "take_profit", "sma_short", "sma_long", 
                "volume_threshold", "breakout_min_percent", "trailing_stop"]
    }
    
    high_importance_combinations = 1
    for param_values in high_importance_params.values():
        high_importance_combinations *= len(param_values)
        
    print(f"\n高重要度パラメータのみの組み合わせ数: {high_importance_combinations:,}")
    print(f"高重要度パラメータの比率: {high_importance_combinations / combinations * 100:.1f}%")
