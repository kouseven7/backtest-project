# filepath: c:\Users\imega\Documents\my_backtest_project\comparison_strategy.py
"""
Module: Comparison Strategy
File: comparison_strategy.py
Description: 
  複数の戦略を比較するためのクラス
"""

import pandas as pd

class StrategyComparison:
    def __init__(self, strategies: list):
        self.strategies = strategies

    def compare(self):
        results = {}
        for strategy in self.strategies:
            results[strategy.__class__.__name__] = self.evaluate_strategy(strategy)
        return results

    def evaluate_strategy(self, strategy):
        # 戦略のバックテスト結果を取得
        result = strategy.backtest()
        
        # 必要な指標を計算
        sharpe_ratio = self.calculate_sharpe_ratio(result['returns'])
        win_rate = self.calculate_win_rate(result['trades'])
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_return': result['total_return']
        }

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        # シャープレシオの計算
        return returns.mean() / returns.std()

    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        # 勝率の計算
        return (trades['result'] > 0).mean()