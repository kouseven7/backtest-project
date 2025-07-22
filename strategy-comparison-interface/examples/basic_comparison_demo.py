# filepath: c:\Users\imega\Documents\my_backtest_project\metrics\performance_metrics.py

class StrategyComparison:
    def __init__(self, strategies_data: List[Dict[str, Any]]):
        self.strategies_data = strategies_data

    def compare_strategies(self) -> pd.DataFrame:
        """戦略のパフォーマンスを比較する"""
        comparison_results = []
        for strategy in self.strategies_data:
            # 各戦略のパフォーマンス指標を計算
            performance_metrics = {
                'strategy_name': strategy['name'],
                'sharpe_ratio': calculate_sharpe_ratio(strategy['returns']),
                'win_rate': strategy['win_rate'],
                'max_drawdown': strategy['max_drawdown'],
            }
            comparison_results.append(performance_metrics)

        return pd.DataFrame(comparison_results)