# filepath: c:\Users\imega\Documents\my_backtest_project\strategy_comparator.py
"""
Module: Strategy Comparator
File: strategy_comparator.py
Description: 
  異なる戦略のパフォーマンスを比較するためのクラスです。
"""

import pandas as pd

class StrategyComparator:
    def __init__(self):
        self.results = {}

    def add_strategy_results(self, strategy_name: str, performance_metrics: dict):
        """戦略のパフォーマンス指標を追加"""
        self.results[strategy_name] = performance_metrics

    def compare_strategies(self):
        """戦略を比較し、最もパフォーマンスが良い戦略を返す"""
        best_strategy = None
        best_score = float('-inf')

        for strategy, metrics in self.results.items():
            score = metrics.get('sharpe_ratio', 0)  # 例としてシャープレシオを使用
            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy, best_score