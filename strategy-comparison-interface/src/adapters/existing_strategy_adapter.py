   # filepath: c:\Users\imega\Documents\my_backtest_project\strategy_comparator.py
   from metrics.performance_metrics import calculate_sharpe_ratio, calculate_sortino_ratio

   class StrategyComparator:
       def __init__(self, strategy_results: dict):
           self.strategy_results = strategy_results

       def compare_strategies(self):
           comparison_results = {}
           for strategy_name, results in self.strategy_results.items():
               sharpe_ratio = calculate_sharpe_ratio(results['returns'])
               sortino_ratio = calculate_sortino_ratio(results['returns'])
               comparison_results[strategy_name] = {
                   'sharpe_ratio': sharpe_ratio,
                   'sortino_ratio': sortino_ratio,
                   'total_return': results['total_return'],
                   'win_rate': results['win_rate']
               }
           return comparison_results