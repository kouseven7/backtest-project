class StrategyComparison:
    def __init__(self):
        self.results = []

    def add_result(self, strategy_name: str, performance_metrics: Dict[str, float]):
        self.results.append({
            "strategy_name": strategy_name,
            "metrics": performance_metrics
        })

    def compare(self):
        # 比較ロジックを実装
        # 例えば、シャープレシオでソートするなど
        sorted_results = sorted(self.results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
        return sorted_results