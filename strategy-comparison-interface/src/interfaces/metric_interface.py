# filepath: c:\Users\imega\Documents\my_backtest_project\comparison_interface.py
"""
Module: Comparison Interface
File: comparison_interface.py
Description: 
  異なる戦略のパフォーマンスを比較するためのインターフェース。
"""

import pandas as pd

class StrategyComparison:
    def __init__(self, strategies: dict):
        """
        :param strategies: 戦略名とその結果の辞書
        """
        self.strategies = strategies

    def compare_performance(self):
        """
        戦略のパフォーマンスを比較し、結果を表示します。
        """
        comparison_results = {}
        
        for strategy_name, results in self.strategies.items():
            # 各戦略のパフォーマンス指標を計算
            sharpe_ratio = self.calculate_sharpe_ratio(results['returns'])
            win_rate = self.calculate_win_rate(results['trades'])
            total_return = results['total_return']
            
            comparison_results[strategy_name] = {
                'Sharpe Ratio': sharpe_ratio,
                'Win Rate': win_rate,
                'Total Return': total_return
            }
        
        return pd.DataFrame(comparison_results).T

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """シャープレシオを計算するメソッド"""
        return returns.mean() / returns.std()

    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """勝率を計算するメソッド"""
        return (trades['result'] > 0).mean()