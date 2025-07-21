"""
Module: Performance Evaluator
File: performance_evaluator.py
Description: 
  5-1-3「リスク調整後リターンの最適化」
  拡張パフォーマンス評価システム

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 既存モジュールのインポート
try:
    from metrics.performance_metrics import (
        calculate_sharpe_ratio, calculate_sortino_ratio, 
        calculate_expectancy, calculate_max_drawdown_during_losses
    )
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error in performance_evaluator: {e}")
    # フォールバック実装
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0, trading_days=252):
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess_returns = returns - (risk_free_rate / trading_days)
        return np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(returns, risk_free_rate=0.0, trading_days=252):
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - (risk_free_rate / trading_days)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        downside_std = np.sqrt((downside_returns ** 2).mean())
        return np.sqrt(trading_days) * excess_returns.mean() / downside_std
    
    def calculate_expectancy(trade_results):
        if len(trade_results) == 0:
            return 0.0
        return trade_results['pnl'].mean()
    
    def calculate_max_drawdown_during_losses(trade_results):
        if len(trade_results) == 0:
            return 0.0
        cumulative_pnl = trade_results['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdowns = (cumulative_pnl - running_max) / running_max.abs()
        return abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricCategory(Enum):
    """指標カテゴリ"""
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    RISK_METRICS = "risk_metrics"
    RETURN_METRICS = "return_metrics"
    DIVERSIFICATION = "diversification"
    STABILITY = "stability"
    EFFICIENCY = "efficiency"

@dataclass
class PerformanceMetric:
    """パフォーマンス指標"""
    name: str
    category: MetricCategory
    value: float
    description: str
    benchmark_value: Optional[float] = None
    percentile_rank: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ComprehensivePerformanceReport:
    """包括的パフォーマンスレポート"""
    portfolio_returns: pd.Series
    benchmark_returns: Optional[pd.Series]
    strategy_returns: pd.DataFrame
    weights: Dict[str, float]
    metrics: Dict[str, PerformanceMetric]
    risk_adjusted_metrics: Dict[str, float]
    diversification_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    comparison_metrics: Dict[str, float]
    optimization_improvement: Dict[str, float]
    confidence_scores: Dict[str, float]
    report_timestamp: datetime = field(default_factory=datetime.now)

class EnhancedPerformanceEvaluator:
    """拡張パフォーマンス評価器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.trading_days = self.config.get('trading_days', 252)
        self.benchmark_returns = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def calculate_comprehensive_metrics(
        self,
        portfolio_returns: pd.Series,
        strategy_returns: pd.DataFrame,
        weights: Dict[str, float],
        benchmark_returns: Optional[pd.Series] = None,
        previous_weights: Optional[Dict[str, float]] = None
    ) -> ComprehensivePerformanceReport:
        """包括的パフォーマンス指標を計算"""
        
        try:
            metrics = {}
            
            # 1. リスク調整済みリターン指標
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(portfolio_returns, benchmark_returns)
            for name, value in risk_adjusted_metrics.items():
                metrics[name] = PerformanceMetric(
                    name=name,
                    category=MetricCategory.RISK_ADJUSTED_RETURN,
                    value=value,
                    description=self._get_metric_description(name)
                )
            
            # 2. リスク指標
            risk_metrics = self._calculate_risk_metrics(portfolio_returns)
            for name, value in risk_metrics.items():
                metrics[name] = PerformanceMetric(
                    name=name,
                    category=MetricCategory.RISK_METRICS,
                    value=value,
                    description=self._get_metric_description(name)
                )
            
            # 3. リターン指標
            return_metrics = self._calculate_return_metrics(portfolio_returns, benchmark_returns)
            for name, value in return_metrics.items():
                metrics[name] = PerformanceMetric(
                    name=name,
                    category=MetricCategory.RETURN_METRICS,
                    value=value,
                    description=self._get_metric_description(name)
                )
            
            # 4. 分散投資指標
            diversification_metrics = self._calculate_diversification_metrics(
                strategy_returns, weights
            )
            for name, value in diversification_metrics.items():
                metrics[name] = PerformanceMetric(
                    name=name,
                    category=MetricCategory.DIVERSIFICATION,
                    value=value,
                    description=self._get_metric_description(name)
                )
            
            # 5. 安定性指標
            stability_metrics = self._calculate_stability_metrics(
                portfolio_returns, strategy_returns, weights
            )
            for name, value in stability_metrics.items():
                metrics[name] = PerformanceMetric(
                    name=name,
                    category=MetricCategory.STABILITY,
                    value=value,
                    description=self._get_metric_description(name)
                )
            
            # 6. 効率性指標
            efficiency_metrics = self._calculate_efficiency_metrics(
                portfolio_returns, weights, previous_weights
            )
            for name, value in efficiency_metrics.items():
                metrics[name] = PerformanceMetric(
                    name=name,
                    category=MetricCategory.EFFICIENCY,
                    value=value,
                    description=self._get_metric_description(name)
                )
            
            # 7. 比較指標
            comparison_metrics = self._calculate_comparison_metrics(
                portfolio_returns, benchmark_returns, strategy_returns
            )
            
            # 8. 最適化改善指標
            optimization_improvement = self._calculate_optimization_improvement(
                weights, previous_weights, risk_adjusted_metrics
            )
            
            # 9. 信頼度スコア
            confidence_scores = self._calculate_confidence_scores(
                portfolio_returns, strategy_returns, weights
            )
            
            return ComprehensivePerformanceReport(
                portfolio_returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
                strategy_returns=strategy_returns,
                weights=weights,
                metrics=metrics,
                risk_adjusted_metrics=risk_adjusted_metrics,
                diversification_metrics=diversification_metrics,
                stability_metrics=stability_metrics,
                comparison_metrics=comparison_metrics,
                optimization_improvement=optimization_improvement,
                confidence_scores=confidence_scores
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            # エラー時は基本的なレポートを返す
            return self._create_fallback_report(portfolio_returns, strategy_returns, weights)
    
    def _calculate_risk_adjusted_metrics(
        self, 
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """リスク調整済みリターン指標を計算"""
        
        metrics = {}
        
        try:
            # シャープレシオ
            metrics['sharpe_ratio'] = calculate_sharpe_ratio(
                portfolio_returns, self.risk_free_rate, self.trading_days
            )
            
            # ソルティノレシオ
            metrics['sortino_ratio'] = calculate_sortino_ratio(
                portfolio_returns, self.risk_free_rate, self.trading_days
            )
            
            # カルマーレシオ
            metrics['calmar_ratio'] = self._calculate_calmar_ratio(portfolio_returns)
            
            # インフォメーションレシオ
            if benchmark_returns is not None:
                metrics['information_ratio'] = self._calculate_information_ratio(
                    portfolio_returns, benchmark_returns
                )
            
            # トレイナーレシオ（ベンチマークとの相関が必要）
            if benchmark_returns is not None:
                metrics['treynor_ratio'] = self._calculate_treynor_ratio(
                    portfolio_returns, benchmark_returns
                )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted metrics: {e}")
        
        return metrics
    
    def _calculate_risk_metrics(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """リスク指標を計算"""
        
        metrics = {}
        
        try:
            # ボラティリティ
            metrics['volatility'] = portfolio_returns.std() * np.sqrt(self.trading_days)
            
            # 最大ドローダウン
            metrics['max_drawdown'] = self._calculate_max_drawdown(portfolio_returns)
            
            # VaR (Value at Risk)
            metrics['var_95'] = self._calculate_var(portfolio_returns, 0.05)
            metrics['var_99'] = self._calculate_var(portfolio_returns, 0.01)
            
            # CVaR (Conditional VaR)
            metrics['cvar_95'] = self._calculate_cvar(portfolio_returns, 0.05)
            metrics['cvar_99'] = self._calculate_cvar(portfolio_returns, 0.01)
            
            # 下方偏差
            metrics['downside_deviation'] = self._calculate_downside_deviation(portfolio_returns)
            
            # 歪度と尖度
            metrics['skewness'] = portfolio_returns.skew()
            metrics['kurtosis'] = portfolio_returns.kurtosis()
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
        
        return metrics
    
    def _calculate_return_metrics(
        self, 
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """リターン指標を計算"""
        
        metrics = {}
        
        try:
            # 年率リターン
            metrics['annual_return'] = portfolio_returns.mean() * self.trading_days
            
            # 累積リターン
            metrics['cumulative_return'] = (1 + portfolio_returns).prod() - 1
            
            # 最大リターン
            metrics['max_return'] = portfolio_returns.max()
            
            # 最小リターン
            metrics['min_return'] = portfolio_returns.min()
            
            # 勝率（正のリターンの割合）
            positive_returns = portfolio_returns > 0
            metrics['win_rate'] = positive_returns.sum() / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
            
            # 平均勝ち負け比率
            winning_returns = portfolio_returns[portfolio_returns > 0]
            losing_returns = portfolio_returns[portfolio_returns < 0]
            
            if len(winning_returns) > 0 and len(losing_returns) > 0:
                metrics['win_loss_ratio'] = winning_returns.mean() / abs(losing_returns.mean())
            else:
                metrics['win_loss_ratio'] = 0.0
            
            # ベンチマーク比較
            if benchmark_returns is not None:
                metrics['excess_return'] = (portfolio_returns - benchmark_returns).mean() * self.trading_days
                metrics['tracking_error'] = (portfolio_returns - benchmark_returns).std() * np.sqrt(self.trading_days)
            
        except Exception as e:
            self.logger.error(f"Error calculating return metrics: {e}")
        
        return metrics
    
    def _calculate_diversification_metrics(
        self, 
        strategy_returns: pd.DataFrame, 
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """分散投資指標を計算"""
        
        metrics = {}
        
        try:
            # 重み配列の作成
            weight_values = list(weights.values())
            
            # ハーフィンダール指数（集中度）
            metrics['herfindahl_index'] = sum(w**2 for w in weight_values)
            
            # 有効戦略数
            metrics['effective_number_strategies'] = 1 / metrics['herfindahl_index'] if metrics['herfindahl_index'] > 0 else 0
            
            # 最大重み
            metrics['max_weight'] = max(weight_values) if weight_values else 0
            
            # 重み分散
            metrics['weight_variance'] = np.var(weight_values) if weight_values else 0
            
            # 分散比率
            if len(strategy_returns.columns) > 1:
                metrics['diversification_ratio'] = self._calculate_diversification_ratio(
                    strategy_returns, weights
                )
            
            # 相関構造の分析
            correlation_matrix = strategy_returns.corr()
            if not correlation_matrix.empty:
                # 平均相関
                upper_triangle = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                metrics['average_correlation'] = upper_triangle.stack().mean()
                
                # 最大相関
                metrics['max_correlation'] = upper_triangle.stack().max()
                
                # 相関リスク（重み付き相関）
                metrics['correlation_risk'] = self._calculate_correlation_risk(correlation_matrix, weights)
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification metrics: {e}")
        
        return metrics
    
    def _calculate_stability_metrics(
        self, 
        portfolio_returns: pd.Series,
        strategy_returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """安定性指標を計算"""
        
        metrics = {}
        
        try:
            # リターンの安定性（変動係数）
            if portfolio_returns.mean() != 0:
                metrics['coefficient_of_variation'] = portfolio_returns.std() / abs(portfolio_returns.mean())
            else:
                metrics['coefficient_of_variation'] = float('inf')
            
            # ローリング相関の安定性
            if len(portfolio_returns) > 60:  # 最低3ヶ月のデータが必要
                rolling_std = portfolio_returns.rolling(window=30).std()
                metrics['volatility_stability'] = 1.0 - rolling_std.std() / rolling_std.mean() if rolling_std.mean() > 0 else 0
            
            # パフォーマンスの一貫性
            if len(portfolio_returns) > 12:
                monthly_returns = portfolio_returns.resample('M').sum() if hasattr(portfolio_returns, 'resample') else portfolio_returns
                positive_months = (monthly_returns > 0).sum()
                total_months = len(monthly_returns)
                metrics['performance_consistency'] = positive_months / total_months if total_months > 0 else 0
            
            # 重み安定性（前期比較が必要な場合）
            metrics['weight_stability'] = self._calculate_weight_stability(weights)
            
        except Exception as e:
            self.logger.error(f"Error calculating stability metrics: {e}")
        
        return metrics
    
    def _calculate_efficiency_metrics(
        self, 
        portfolio_returns: pd.Series,
        weights: Dict[str, float],
        previous_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """効率性指標を計算"""
        
        metrics = {}
        
        try:
            # 回転率（ポートフォリオの変更度合い）
            if previous_weights is not None:
                metrics['turnover_rate'] = self._calculate_turnover_rate(weights, previous_weights)
                
                # 取引コスト推定
                metrics['estimated_transaction_cost'] = metrics['turnover_rate'] * self.config.get('transaction_cost_rate', 0.001)
            
            # リターン/リスク効率性
            if portfolio_returns.std() > 0:
                metrics['return_risk_efficiency'] = portfolio_returns.mean() / portfolio_returns.std()
            else:
                metrics['return_risk_efficiency'] = 0
            
            # 資本効率性（レバレッジ調整後）
            total_weights = sum(abs(w) for w in weights.values())
            metrics['capital_efficiency'] = portfolio_returns.mean() / total_weights if total_weights > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency metrics: {e}")
        
        return metrics
    
    def _calculate_comparison_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        strategy_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """比較指標を計算"""
        
        metrics = {}
        
        try:
            if benchmark_returns is not None:
                # アルファ
                metrics['alpha'] = self._calculate_alpha(portfolio_returns, benchmark_returns)
                
                # ベータ
                metrics['beta'] = self._calculate_beta(portfolio_returns, benchmark_returns)
                
                # R-squared
                metrics['r_squared'] = self._calculate_r_squared(portfolio_returns, benchmark_returns)
                
                # アクティブリターン
                active_returns = portfolio_returns - benchmark_returns
                metrics['active_return'] = active_returns.mean() * self.trading_days
                
                # トラッキングエラー
                metrics['tracking_error'] = active_returns.std() * np.sqrt(self.trading_days)
            
            # 個別戦略との比較
            if len(strategy_returns.columns) > 0:
                best_strategy_return = strategy_returns.mean().max() * self.trading_days
                portfolio_annual_return = portfolio_returns.mean() * self.trading_days
                metrics['outperformance_vs_best_strategy'] = portfolio_annual_return - best_strategy_return
                
                # 戦略間の分散
                strategy_annual_returns = strategy_returns.mean() * self.trading_days
                metrics['strategy_return_dispersion'] = strategy_annual_returns.std()
            
        except Exception as e:
            self.logger.error(f"Error calculating comparison metrics: {e}")
        
        return metrics
    
    def _calculate_optimization_improvement(
        self,
        weights: Dict[str, float],
        previous_weights: Optional[Dict[str, float]],
        risk_adjusted_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """最適化改善指標を計算"""
        
        improvements = {}
        
        try:
            if previous_weights is not None:
                # 重み変更度合い
                weight_changes = {}
                for strategy in set(weights.keys()) | set(previous_weights.keys()):
                    current_w = weights.get(strategy, 0.0)
                    previous_w = previous_weights.get(strategy, 0.0)
                    weight_changes[strategy] = abs(current_w - previous_w)
                
                improvements['total_weight_change'] = sum(weight_changes.values()) / 2
                improvements['max_weight_change'] = max(weight_changes.values()) if weight_changes else 0
            
            # リスク調整後リターンの改善
            current_sharpe = risk_adjusted_metrics.get('sharpe_ratio', 0)
            improvements['sharpe_improvement'] = current_sharpe  # ベースラインがない場合は現在値
            
            # 分散投資の改善
            current_hhi = sum(w**2 for w in weights.values())
            optimal_hhi = 1.0 / len(weights) if len(weights) > 0 else 1.0  # 等重み
            improvements['diversification_improvement'] = 1.0 - (current_hhi / optimal_hhi) if optimal_hhi > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating optimization improvement: {e}")
        
        return improvements
    
    def _calculate_confidence_scores(
        self,
        portfolio_returns: pd.Series,
        strategy_returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """信頼度スコアを計算"""
        
        scores = {}
        
        try:
            # データ量に基づく信頼度
            data_points = len(portfolio_returns)
            scores['data_sufficiency'] = min(1.0, data_points / 252)  # 1年分を基準
            
            # リターンの統計的有意性
            if portfolio_returns.std() > 0:
                t_stat = portfolio_returns.mean() / (portfolio_returns.std() / np.sqrt(len(portfolio_returns)))
                scores['statistical_significance'] = min(1.0, abs(t_stat) / 2.0)
            else:
                scores['statistical_significance'] = 0.0
            
            # 重みの妥当性
            weight_values = list(weights.values())
            weight_sum = sum(weight_values)
            scores['weight_validity'] = 1.0 if abs(weight_sum - 1.0) < 0.01 else max(0.0, 1.0 - abs(weight_sum - 1.0))
            
            # 総合信頼度
            scores['overall_confidence'] = np.mean(list(scores.values()))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence scores: {e}")
            scores['overall_confidence'] = 0.5
        
        return scores
    
    # ヘルパーメソッド
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """カルマーレシオを計算"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * self.trading_days
        max_drawdown = self._calculate_max_drawdown(returns)
        
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_drawdown
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """最大ドローダウンを計算"""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """VaRを計算"""
        if len(returns) == 0:
            return 0.0
        return abs(np.percentile(returns, confidence_level * 100))
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """CVaRを計算"""
        if len(returns) == 0:
            return 0.0
        
        var_threshold = -self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var_threshold]
        
        return abs(tail_returns.mean()) if len(tail_returns) > 0 else 0.0
    
    def _calculate_information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """インフォメーションレシオを計算"""
        active_returns = portfolio_returns - benchmark_returns
        if active_returns.std() == 0:
            return 0.0
        return (active_returns.mean() * self.trading_days) / (active_returns.std() * np.sqrt(self.trading_days))
    
    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """下方偏差を計算"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        return downside_returns.std() * np.sqrt(self.trading_days)
    
    def _calculate_turnover_rate(self, current_weights: Dict[str, float], previous_weights: Dict[str, float]) -> float:
        """回転率を計算"""
        turnover = 0.0
        for strategy in set(current_weights.keys()) | set(previous_weights.keys()):
            current_w = current_weights.get(strategy, 0.0)
            previous_w = previous_weights.get(strategy, 0.0)
            turnover += abs(current_w - previous_w)
        return turnover / 2
    
    def _get_metric_description(self, metric_name: str) -> str:
        """指標の説明を取得"""
        descriptions = {
            'sharpe_ratio': 'リスク調整後リターンの効率性指標',
            'sortino_ratio': '下方リスクのみを考慮したリスク調整後リターン',
            'calmar_ratio': '最大ドローダウンに対する年率リターンの比率',
            'information_ratio': 'ベンチマークに対するアクティブリターンの効率性',
            'treynor_ratio': 'システマティックリスクに対する超過リターン',
            'volatility': '年率換算したリターンの標準偏差',
            'max_drawdown': '最大ドローダウン率',
            'var_95': '95%信頼水準でのValue at Risk',
            'var_99': '99%信頼水準でのValue at Risk',
            'cvar_95': '95%信頼水準でのConditional VaR',
            'cvar_99': '99%信頼水準でのConditional VaR',
            'annual_return': '年率換算リターン',
            'cumulative_return': '累積リターン',
            'win_rate': '正のリターンを記録した期間の割合',
            'herfindahl_index': 'ポートフォリオの集中度指標',
            'diversification_ratio': '分散投資効果の測定',
            'turnover_rate': 'ポートフォリオの回転率'
        }
        return descriptions.get(metric_name, f'{metric_name}の説明')
    
    def _create_fallback_report(
        self,
        portfolio_returns: pd.Series,
        strategy_returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> ComprehensivePerformanceReport:
        """エラー時のフォールバックレポート"""
        
        basic_metrics = {
            'sharpe_ratio': PerformanceMetric(
                name='sharpe_ratio',
                category=MetricCategory.RISK_ADJUSTED_RETURN,
                value=0.0,
                description='シャープレシオ'
            )
        }
        
        return ComprehensivePerformanceReport(
            portfolio_returns=portfolio_returns,
            benchmark_returns=None,
            strategy_returns=strategy_returns,
            weights=weights,
            metrics=basic_metrics,
            risk_adjusted_metrics={'sharpe_ratio': 0.0},
            diversification_metrics={},
            stability_metrics={},
            comparison_metrics={},
            optimization_improvement={},
            confidence_scores={'overall_confidence': 0.0}
        )

    def _calculate_diversification_ratio(self, strategy_returns: pd.DataFrame, weights: Dict[str, float]) -> float:
        """分散比率を計算"""
        try:
            # 個別戦略のボラティリティ
            individual_vols = strategy_returns.std() * np.sqrt(self.trading_days)
            
            # 重み付き平均ボラティリティ
            weight_vol_sum = sum(weights.get(strategy, 0) * individual_vols.get(strategy, 0) 
                               for strategy in strategy_returns.columns)
            
            # ポートフォリオボラティリティ
            portfolio_returns = sum(strategy_returns[strategy] * weights.get(strategy, 0) 
                                  for strategy in strategy_returns.columns)
            portfolio_vol = portfolio_returns.std() * np.sqrt(self.trading_days)
            
            if portfolio_vol == 0:
                return 1.0
            
            return weight_vol_sum / portfolio_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification ratio: {e}")
            return 1.0

    def _calculate_correlation_risk(self, correlation_matrix: pd.DataFrame, weights: Dict[str, float]) -> float:
        """相関リスクを計算"""
        try:
            risk_score = 0.0
            total_weight_pairs = 0
            
            for strategy1 in correlation_matrix.index:
                for strategy2 in correlation_matrix.columns:
                    if strategy1 != strategy2:
                        correlation = correlation_matrix.loc[strategy1, strategy2]
                        weight1 = weights.get(strategy1, 0)
                        weight2 = weights.get(strategy2, 0)
                        
                        risk_contribution = abs(correlation) * weight1 * weight2
                        risk_score += risk_contribution
                        total_weight_pairs += weight1 * weight2
            
            return risk_score / total_weight_pairs if total_weight_pairs > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0

    def _calculate_weight_stability(self, weights: Dict[str, float]) -> float:
        """重み安定性を計算（分散度合いに基づく）"""
        try:
            weight_values = list(weights.values())
            if len(weight_values) <= 1:
                return 1.0
            
            # 重みの分散が小さいほど安定
            weight_variance = np.var(weight_values)
            max_variance = 0.25  # 等重み時の最大分散を仮定
            
            stability = 1.0 - min(weight_variance / max_variance, 1.0)
            return max(0.0, stability)
            
        except Exception as e:
            self.logger.error(f"Error calculating weight stability: {e}")
            return 0.5

    def _calculate_treynor_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """トレイナーレシオを計算"""
        try:
            beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            if beta == 0:
                return 0.0
            
            excess_return = (portfolio_returns.mean() - self.risk_free_rate / self.trading_days) * self.trading_days
            return excess_return / beta
            
        except Exception as e:
            self.logger.error(f"Error calculating Treynor ratio: {e}")
            return 0.0

    def _calculate_alpha(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """アルファを計算"""
        try:
            beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            
            portfolio_mean = portfolio_returns.mean() * self.trading_days
            benchmark_mean = benchmark_returns.mean() * self.trading_days
            
            alpha = portfolio_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
            return alpha
            
        except Exception as e:
            self.logger.error(f"Error calculating alpha: {e}")
            return 0.0

    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """ベータを計算"""
        try:
            if len(portfolio_returns) != len(benchmark_returns):
                return 1.0
            
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            if benchmark_variance == 0:
                return 1.0
            
            return covariance / benchmark_variance
            
        except Exception as e:
            self.logger.error(f"Error calculating beta: {e}")
            return 1.0

    def _calculate_r_squared(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """R-squaredを計算"""
        try:
            if len(portfolio_returns) != len(benchmark_returns):
                return 0.0
            
            correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
            return correlation ** 2 if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating R-squared: {e}")
            return 0.0


# テスト用のメイン関数
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    logger.info("Testing Enhanced Performance Evaluator...")
    
    # テストデータの生成
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # 戦略リターンデータ
    strategy_returns = pd.DataFrame({
        'strategy1': np.random.normal(0.001, 0.02, len(dates)),
        'strategy2': np.random.normal(0.0015, 0.025, len(dates)),
        'strategy3': np.random.normal(0.0008, 0.018, len(dates))
    }, index=dates)
    
    # ポートフォリオ重み
    weights = {
        'strategy1': 0.4,
        'strategy2': 0.35,
        'strategy3': 0.25
    }
    
    # ポートフォリオリターンの計算
    portfolio_returns = (strategy_returns * pd.Series(weights)).sum(axis=1)
    
    # ベンチマークリターン（市場インデックス）
    benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)
    
    # パフォーマンス評価器のテスト
    evaluator = EnhancedPerformanceEvaluator()
    
    report = evaluator.calculate_comprehensive_metrics(
        portfolio_returns=portfolio_returns,
        strategy_returns=strategy_returns,
        weights=weights,
        benchmark_returns=benchmark_returns
    )
    
    logger.info("Performance Evaluation Results:")
    logger.info(f"Sharpe Ratio: {report.risk_adjusted_metrics.get('sharpe_ratio', 0):.4f}")
    logger.info(f"Sortino Ratio: {report.risk_adjusted_metrics.get('sortino_ratio', 0):.4f}")
    logger.info(f"Max Drawdown: {report.metrics.get('max_drawdown', {}).value if 'max_drawdown' in report.metrics else 0:.4f}")
    logger.info(f"Annual Return: {report.metrics.get('annual_return', {}).value if 'annual_return' in report.metrics else 0:.4f}")
    logger.info(f"Volatility: {report.metrics.get('volatility', {}).value if 'volatility' in report.metrics else 0:.4f}")
    logger.info(f"Diversification Ratio: {report.diversification_metrics.get('diversification_ratio', 0):.4f}")
    logger.info(f"Overall Confidence: {report.confidence_scores.get('overall_confidence', 0):.4f}")
    
    logger.info("Enhanced Performance Evaluator test completed successfully!")
