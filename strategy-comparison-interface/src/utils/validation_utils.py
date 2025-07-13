# filepath: c:\Users\imega\Documents\my_backtest_project\strategy_comparator.py
"""
Module: Strategy Comparator
File: strategy_comparator.py
Description: 
  複数の戦略を比較するためのモジュールです。
"""

import pandas as pd

class StrategyComparator:
    def __init__(self, strategies: list):
        self.strategies = strategies

    def compare(self, metric: str) -> pd.DataFrame:
        """指定された指標に基づいて戦略を比較します。"""
        results = {}
        for strategy in self.strategies:
            # 各戦略のパフォーマンスを取得
            performance = strategy.get_performance_metrics()
            results[strategy.name] = performance[metric]
        
        return pd.DataFrame(results, index=[metric])