   class StrategyComparator:
       def __init__(self, strategies: List[BaseStrategy]):
           self.strategies = strategies

       def compare_performance(self) -> Dict[str, Any]:
           results = {}
           for strategy in self.strategies:
               performance = self.evaluate_strategy(strategy)
               results[strategy.name] = performance
           return results

       def evaluate_strategy(self, strategy: BaseStrategy) -> Dict[str, float]:
           # 戦略のパフォーマンスを評価するロジックを実装
           return {
               "sharpe_ratio": calculate_sharpe_ratio(strategy.results),
               "max_drawdown": calculate_max_drawdown(strategy.results),
               "win_rate": calculate_win_rate(strategy.results),
           }