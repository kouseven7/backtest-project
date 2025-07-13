   # filepath: c:\Users\imega\Documents\my_backtest_project\comparison_interface.py
   class StrategyComparison:
       def __init__(self, strategy_results: Dict[str, pd.DataFrame]):
           self.strategy_results = strategy_results

       def compare_strategies(self):
           # 比較ロジックを実装
           comparison_results = {}
           for strategy_name, results in self.strategy_results.items():
               # 各戦略のパフォーマンス指標を計算
               sharpe_ratio = self.calculate_sharpe_ratio(results['returns'])
               comparison_results[strategy_name] = sharpe_ratio
           return comparison_results

       def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
           # シャープレシオの計算
           return returns.mean() / returns.std()