"""
共通パフォーマンス指標計算ユーティリティ
全戦略で使えるようにラップ
"""
import pandas as pd
from metrics import performance_metrics

class PerformanceMetricsCalculator:
    @staticmethod
    def calculate_all(trade_results: pd.DataFrame, cumulative_pnl: pd.Series = None, risk_free_rate: float = 0.0) -> dict:
        """
        主要なパフォーマンス指標をまとめて計算してdictで返す
        """
        if cumulative_pnl is None and '累積損益' in trade_results.columns:
            cumulative_pnl = trade_results['累積損益']
        elif cumulative_pnl is None:
            cumulative_pnl = trade_results['取引結果'].cumsum()
        returns = cumulative_pnl.diff().fillna(0)
        metrics = {
            'sharpe_ratio': performance_metrics.calculate_sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': performance_metrics.calculate_sortino_ratio(returns, risk_free_rate),
            'win_rate': performance_metrics.calculate_win_rate(trade_results),
            'total_return': cumulative_pnl.iloc[-1] if len(cumulative_pnl) > 0 else 0.0,
            'max_drawdown': performance_metrics.calculate_max_drawdown(cumulative_pnl),
            'profit_factor': (trade_results[trade_results['取引結果'] > 0]['取引結果'].sum() / abs(trade_results[trade_results['取引結果'] < 0]['取引結果'].sum())) if (trade_results[trade_results['取引結果'] < 0]['取引結果'].sum() != 0) else float('inf'),
            'total_trades': performance_metrics.calculate_total_trades(trade_results),
            'expectancy': performance_metrics.calculate_expectancy(trade_results),
            'max_consecutive_losses': performance_metrics.calculate_max_consecutive_losses(trade_results),
            'max_consecutive_wins': performance_metrics.calculate_max_consecutive_wins(trade_results),
            'avg_consecutive_losses': performance_metrics.calculate_avg_consecutive_losses(trade_results),
            'avg_consecutive_wins': performance_metrics.calculate_avg_consecutive_wins(trade_results),
            'max_drawdown_amount': performance_metrics.calculate_max_drawdown_amount(cumulative_pnl),
            'max_profit': performance_metrics.calculate_max_profit(trade_results),
            'max_loss': performance_metrics.calculate_max_loss(trade_results),
        }
        return metrics
