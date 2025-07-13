# filepath: c:\Users\imega\Documents\my_backtest_project\strategy_comparison.py
from strategies.base_strategy import BaseStrategy
from metrics.performance_metrics import calculate_sharpe_ratio, calculate_sortino_ratio

class StrategyComparisonInterface:
    def __init__(self):
        self.strategies = []

    def add_strategy(self, strategy: BaseStrategy):
        self.strategies.append(strategy)

    def run_comparison(self):
        results = []
        for strategy in self.strategies:
            # バックテストを実行
            backtest_result = strategy.backtest()
            # パフォーマンス指標を計算
            sharpe_ratio = calculate_sharpe_ratio(backtest_result['returns'])
            sortino_ratio = calculate_sortino_ratio(backtest_result['returns'])
            results.append({
                'strategy_name': strategy.__class__.__name__,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'win_rate': (backtest_result['win_count'] / backtest_result['total_trades']),
            })
        return results

    def get_results(self):
        return self.run_comparison()