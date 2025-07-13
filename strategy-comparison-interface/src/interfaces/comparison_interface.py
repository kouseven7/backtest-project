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
        results = {}
        for strategy in self.strategies:
            # 各戦略のバックテスト結果を取得
            results[strategy.__name__] = strategy.backtest()  # 例: 戦略のバックテストメソッドを呼び出す

        # 指標に基づいて結果を整理
        comparison_df = pd.DataFrame(results).T
        return comparison_df[metric]