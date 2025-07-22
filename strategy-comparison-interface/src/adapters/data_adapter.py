# filepath: c:\Users\imega\Documents\my_backtest_project\strategy_comparator.py
"""
Module: Strategy Comparator
File: strategy_comparator.py
Description: 
  複数の戦略を比較するためのモジュール。
"""

import pandas as pd

class StrategyComparator:
    def __init__(self, strategies: list):
        self.strategies = strategies

    def compare(self, metric: str) -> pd.DataFrame:
        """指定された指標に基づいて戦略を比較する"""
        results = {}
        for strategy in self.strategies:
            results[strategy.__class__.__name__] = strategy.evaluate(metric)
        
        return pd.DataFrame(results)