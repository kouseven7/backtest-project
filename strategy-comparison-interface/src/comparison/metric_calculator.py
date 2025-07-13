class StrategyComparison:
    def __init__(self, strategies: List[BaseStrategy]):
        self.strategies = strategies

    def compare_performance(self):
        results = {}
        for strategy in self.strategies:
            performance_metrics = strategy.get_performance_metrics()  # 戦略のパフォーマンスメトリクスを取得
            results[strategy.__class__.__name__] = performance_metrics
        return results