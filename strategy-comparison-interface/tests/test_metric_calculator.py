# filepath: c:\Users\imega\Documents\my_backtest_project\strategy_comparator.py
"""
Module: Strategy Comparator
File: strategy_comparator.py
Description: 
  複数の戦略のパフォーマンスを比較するためのモジュールです。
"""

from metrics.performance_metrics import calculate_sharpe_ratio, calculate_sortino_ratio

class StrategyComparator:
    def __init__(self, strategies_results):
        """
        コンストラクタ
        :param strategies_results: 各戦略のバックテスト結果を含む辞書
        """
        self.strategies_results = strategies_results

    def compare_strategies(self):
        """
        戦略のパフォーマンスを比較し、最も優れた戦略を特定します。
        """
        comparison_results = {}
        
        for strategy_name, results in self.strategies_results.items():
            sharpe_ratio = calculate_sharpe_ratio(results['returns'])
            sortino_ratio = calculate_sortino_ratio(results['returns'])
            win_rate = results['win_rate']
            max_drawdown = results['max_drawdown']
            
            comparison_results[strategy_name] = {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown
            }
        
        return comparison_results