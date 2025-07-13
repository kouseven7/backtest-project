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

    def compare(self):
        results = {}
        for strategy in self.strategies:
            # 各戦略のパフォーマンスを計算
            performance = self.evaluate_strategy(strategy)
            results[strategy.__name__] = performance
        return results

    def evaluate_strategy(self, strategy):
        # 戦略のバックテスト結果を取得し、必要な指標を計算
        # ここでは仮のデータを使用
        return {
            "sharpe_ratio": 1.5,  # 例: シャープレシオ
            "win_rate": 0.6,      # 例: 勝率
            "max_drawdown": 0.2   # 例: 最大ドローダウン
        }