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
        # データフレームが空または列がない場合のチェック
        if trade_results is None or trade_results.empty:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'expectancy': 0.0
            }
        
        if cumulative_pnl is None and '累積損益' in trade_results.columns:
            cumulative_pnl = trade_results['累積損益']
        elif cumulative_pnl is None and '取引結果' in trade_results.columns:
            cumulative_pnl = trade_results['取引結果'].cumsum()
        elif cumulative_pnl is None:
            # 取引結果列がない場合、scoreを使用して簡易的なデータを作成
            if 'score' in trade_results.columns:
                cumulative_pnl = pd.Series([trade_results['score'].iloc[0]] * len(trade_results))
            else:
                # どの列も存在しない場合は、ゼロのシリーズを作成
                cumulative_pnl = pd.Series([0.0] * len(trade_results))
        returns = cumulative_pnl.diff().fillna(0)
        # 基本の指標を設定する
        metrics = {
            'sharpe_ratio': performance_metrics.calculate_sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': performance_metrics.calculate_sortino_ratio(returns, risk_free_rate),
            'total_return': cumulative_pnl.iloc[-1] if len(cumulative_pnl) > 0 else 0.0,
            'max_drawdown': performance_metrics.calculate_max_drawdown(cumulative_pnl),
            'max_drawdown_amount': performance_metrics.calculate_max_drawdown_amount(cumulative_pnl)
        }
        
        # '取引結果' カラムが存在する場合のみ、依存する指標を計算する
        if '取引結果' in trade_results.columns:
            # 利益ファクター（勝ちトレードの合計 / 負けトレードの合計の絶対値）
            loss_sum = abs(trade_results[trade_results['取引結果'] < 0]['取引結果'].sum())
            profit_factor = (trade_results[trade_results['取引結果'] > 0]['取引結果'].sum() / loss_sum) if loss_sum != 0 else float('inf')
            
            # その他の取引結果に依存する指標
            win_rate = performance_metrics.calculate_win_rate(trade_results)
            total_trades = performance_metrics.calculate_total_trades(trade_results)
            expectancy = performance_metrics.calculate_expectancy(trade_results)
            max_consecutive_losses = performance_metrics.calculate_max_consecutive_losses(trade_results)
            max_consecutive_wins = performance_metrics.calculate_max_consecutive_wins(trade_results)
            avg_consecutive_losses = performance_metrics.calculate_avg_consecutive_losses(trade_results)
            avg_consecutive_wins = performance_metrics.calculate_avg_consecutive_wins(trade_results)
            max_profit = performance_metrics.calculate_max_profit(trade_results)
            max_loss = performance_metrics.calculate_max_loss(trade_results)
            
            # 計算した指標を辞書に追加
            metrics.update({
                'profit_factor': profit_factor,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'expectancy': expectancy,
                'max_consecutive_losses': max_consecutive_losses,
                'max_consecutive_wins': max_consecutive_wins,
                'avg_consecutive_losses': avg_consecutive_losses,
                'avg_consecutive_wins': avg_consecutive_wins,
                'max_profit': max_profit,
                'max_loss': max_loss
            })
        else:
            # '取引結果'カラムがない場合はデフォルト値を設定
            metrics.update({
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'expectancy': 0.0,
                'max_consecutive_losses': 0,
                'max_consecutive_wins': 0,
                'avg_consecutive_losses': 0.0,
                'avg_consecutive_wins': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0
            })
        
        return metrics
