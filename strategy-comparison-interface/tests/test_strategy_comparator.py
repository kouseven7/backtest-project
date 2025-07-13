   # filepath: c:\Users\imega\Documents\my_backtest_project\optimization\strategy_comparator.py
   class StrategyComparator:
       def __init__(self, strategies: List[str]):
           self.strategies = strategies
           self.results = {}

       def compare(self, metric: str):
           for strategy in self.strategies:
               # 各戦略のパフォーマンスを取得
               performance = self.get_performance(strategy)
               self.results[strategy] = performance

           # 比較結果を表示
           self.display_results(metric)

       def get_performance(self, strategy: str):
           # 戦略のパフォーマンスを取得するロジックを実装
           pass

       def display_results(self, metric: str):
           # 比較結果を表示するロジックを実装
           pass