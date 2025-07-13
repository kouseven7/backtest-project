# filepath: c:\Users\imega\Documents\my_backtest_project\strategy_comparator.py
"""
Module: Strategy Comparator
File: strategy_comparator.py
Description: 
  戦略のパフォーマンスを比較するためのクラス
"""

import pandas as pd

class StrategyComparator:
    def __init__(self, strategies: list):
        self.strategies = strategies

    def compare(self, metric: str) -> pd.DataFrame:
        """
        指定されたメトリックに基づいて戦略を比較する
        :param metric: 比較するメトリック名
        :return: 戦略のパフォーマンスを含むデータフレーム
        """
        results = {}
        for strategy in self.strategies:
            # 各戦略のパフォーマンスを取得
            performance = strategy.get_performance(metric)
            results[strategy.name] = performance
        
        return pd.DataFrame(results)