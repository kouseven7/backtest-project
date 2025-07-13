   class StrategyComparator:
       def __init__(self, strategies: List[BaseStrategy]):
           self.strategies = strategies

       def compare(self):
           results = {}
           for strategy in self.strategies:
               performance = self.evaluate_strategy(strategy)
               results[strategy.__class__.__name__] = performance
           return results

       def evaluate_strategy(self, strategy: BaseStrategy):
           # 戦略のバックテストを実行し、パフォーマンス指標を計算
           result = strategy.backtest()
           sharpe_ratio = calculate_sharpe_ratio(result['returns'])
           sortino_ratio = calculate_sortino_ratio(result['returns'])
           return {
               'sharpe_ratio': sharpe_ratio,
               'sortino_ratio': sortino_ratio,
               'win_rate': (result['win_count'] / result['total_trades']) if result['total_trades'] > 0 else 0
           }