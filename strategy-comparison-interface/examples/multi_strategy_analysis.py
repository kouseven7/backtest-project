# filepath: c:\Users\imega\Documents\my_backtest_project\metrics\performance_metrics.py

class StrategyComparison:
    def __init__(self, strategies_data: List[Dict[str, Any]]):
        self.strategies_data = strategies_data

    def compare_strategies(self, metric: str) -> Dict[str, Any]:
        """
        指定された指標に基づいて戦略を比較します。
        """
        comparison_results = {}
        for strategy in self.strategies_data:
            comparison_results[strategy['name']] = strategy[metric]
        
        return comparison_results