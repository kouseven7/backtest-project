# filepath: c:\Users\imega\Documents\my_backtest_project\strategy_comparator.py
"""
Module: Strategy Comparator
File: strategy_comparator.py
Description: 
  異なる戦略のパフォーマンスを比較するためのクラス
"""

from typing import Dict, Any

class StrategyComparator:
    def __init__(self):
        self.results = {}

    def add_strategy_result(self, strategy_name: str, performance_metrics: Dict[str, Any]):
        """戦略のパフォーマンス結果を追加"""
        self.results[strategy_name] = performance_metrics

    def compare_strategies(self):
        """戦略を比較し、最も優れた戦略を返す"""
        best_strategy = None
        best_score = float('-inf')

        for strategy, metrics in self.results.items():
            score = self.calculate_score(metrics)
            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy, best_score

    def calculate_score(self, metrics: Dict[str, Any]) -> float:
        """パフォーマンス指標に基づいてスコアを計算"""
        # 例: シャープレシオを重視する場合
        return metrics.get('sharpe_ratio', 0)  # デフォルトは0