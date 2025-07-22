"""
Module: Enhanced Performance Calculator
File: enhanced_performance_calculator.py
Description: 
  4-2-2「複合戦略バックテスト機能実装」- Enhanced Performance Calculator
  期待値重視のパフォーマンス計算と詳細な分析機能

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 期待値重視パフォーマンス計算
  - 複合戦略対応パフォーマンス分析
  - ベンチマーク比較機能
  - リスク調整リターン計算
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ロガーの設定
logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """パフォーマンス指標"""
    EXPECTED_VALUE = "expected_value"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    EXPECTED_SHORTFALL = "expected_shortfall"
    VALUE_AT_RISK = "value_at_risk"
    INFORMATION_RATIO = "information_ratio"

class RiskMetric(Enum):
    """リスク指標"""
    VOLATILITY = "volatility"
    DOWNSIDE_DEVIATION = "downside_deviation"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    CORRELATION = "correlation"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"

@dataclass
class ExpectedValueCalculation:
    """期待値計算結果"""
    expected_return: float
    probability_weighted_return: float
    worst_case_scenario: float
    best_case_scenario: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    scenario_contributions: Dict[str, float]
    risk_adjusted_expected_value: float
    calculation_method: str
    uncertainty_level: float

@dataclass
class PerformanceAnalysis:
    """パフォーマンス分析結果"""
    period_start: datetime
    period_end: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    expected_value_metrics: ExpectedValueCalculation
    risk_metrics: Dict[str, float]
    benchmark_comparison: Dict[str, float]
    monthly_returns: pd.Series
    drawdown_series: pd.Series
    rolling_metrics: pd.DataFrame

@dataclass
class CompositeStrategyPerformance:
    """複合戦略パフォーマンス"""
    strategy_combination_id: str
    individual_strategy_performance: Dict[str, PerformanceAnalysis]
    combined_performance: PerformanceAnalysis
    correlation_matrix: pd.DataFrame
    contribution_analysis: Dict[str, float]
    diversification_ratio: float
    rebalancing_cost: float
    alpha_generation: Dict[str, float]
    risk_decomposition: Dict[str, Dict[str, float]]

class EnhancedPerformanceCalculator:
    """強化パフォーマンス計算器"""
    
    def __init__(self, config_path: Optional[str] = None, risk_free_rate: float = 0.02):
        """計算器の初期化"""
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = risk_free_rate
        
        # 設定の読み込み
        self.config = self._load_performance_config(config_path)
        self.calculation_settings = self.config.get('calculation_settings', {})
        self.expected_value_config = self.config.get('expected_value_calculation', {})
        
        # キャッシュの初期化
        self.performance_cache = {}
        self.calculation_stats = {
            "total_calculations": 0,
            "cache_hits": 0,
            "calculation_time": 0.0
        }
        
        self.logger.info("EnhancedPerformanceCalculator initialized")
    
    def _load_performance_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """パフォーマンス設定の読み込み"""
        
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "backtest", 
                "performance_expectations.json"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Performance config loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load performance config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        
        return {
            "calculation_settings": {
                "confidence_levels": [0.90, 0.95, 0.99],
                "var_method": "historical",
                "expected_value_method": "monte_carlo",
                "benchmark_symbol": "SPY"
            },
            "expected_value_calculation": {
                "monte_carlo_iterations": 10000,
                "scenario_probabilities": {
                    "bull_market": 0.3,
                    "normal_market": 0.4,
                    "bear_market": 0.3
                },
                "risk_adjustment_factor": 0.1
            }
        }
    
    def calculate_expected_value_metrics(self, 
                                       returns: pd.Series, 
                                       scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> ExpectedValueCalculation:
        """期待値重視メトリクスの計算"""
        
        try:
            # 基本統計の計算
            mean_return = returns.mean()
            std_return = returns.std()
            
            # 信頼区間の計算
            confidence_levels = self.calculation_settings.get('confidence_levels', [0.90, 0.95, 0.99])
            confidence_intervals = {}
            
            for level in confidence_levels:
                alpha = 1 - level
                t_value = stats.t.ppf(1 - alpha/2, len(returns) - 1)
                margin_of_error = t_value * std_return / np.sqrt(len(returns))
                
                confidence_intervals[f'{int(level*100)}%'] = (
                    mean_return - margin_of_error,
                    mean_return + margin_of_error
                )
            
            # シナリオベース期待値計算
            if scenarios:
                scenario_contributions = {}
                probability_weighted_return = 0.0
                
                scenario_probs = self.expected_value_config.get('scenario_probabilities', {})
                
                for scenario_name, scenario_data in scenarios.items():
                    prob = scenario_probs.get(scenario_name, 1.0 / len(scenarios))
                    scenario_return = scenario_data.get('expected_return', mean_return)
                    contribution = prob * scenario_return
                    
                    scenario_contributions[scenario_name] = contribution
                    probability_weighted_return += contribution
            else:
                scenario_contributions = {"historical": mean_return}
                probability_weighted_return = mean_return
            
            # モンテカルロシミュレーション
            mc_iterations = self.expected_value_config.get('monte_carlo_iterations', 10000)
            np.random.seed(42)
            
            simulated_returns = np.random.normal(mean_return, std_return, mc_iterations)
            worst_case = np.percentile(simulated_returns, 5)
            best_case = np.percentile(simulated_returns, 95)
            
            # リスク調整期待値
            risk_adjustment = self.expected_value_config.get('risk_adjustment_factor', 0.1)
            risk_adjusted_expected_value = probability_weighted_return - (risk_adjustment * std_return)
            
            # 不確実性レベル
            uncertainty_level = std_return / abs(mean_return) if mean_return != 0 else float('inf')
            
            return ExpectedValueCalculation(
                expected_return=mean_return,
                probability_weighted_return=probability_weighted_return,
                worst_case_scenario=worst_case,
                best_case_scenario=best_case,
                confidence_intervals=confidence_intervals,
                scenario_contributions=scenario_contributions,
                risk_adjusted_expected_value=risk_adjusted_expected_value,
                calculation_method="monte_carlo_with_scenarios",
                uncertainty_level=uncertainty_level
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate expected value metrics: {e}")
            # フォールバック計算
            return ExpectedValueCalculation(
                expected_return=returns.mean(),
                probability_weighted_return=returns.mean(),
                worst_case_scenario=returns.min(),
                best_case_scenario=returns.max(),
                confidence_intervals={"95%": (returns.mean() - 2*returns.std(), returns.mean() + 2*returns.std())},
                scenario_contributions={"fallback": returns.mean()},
                risk_adjusted_expected_value=returns.mean() - 0.1 * returns.std(),
                calculation_method="fallback",
                uncertainty_level=returns.std() / abs(returns.mean()) if returns.mean() != 0 else float('inf')
            )
    
    def calculate_comprehensive_performance(self, 
                                          returns: pd.Series, 
                                          prices: Optional[pd.Series] = None,
                                          benchmark_returns: Optional[pd.Series] = None,
                                          scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> PerformanceAnalysis:
        """包括的パフォーマンス分析"""
        
        start_time = datetime.now()
        
        try:
            # キャッシュチェック
            cache_key = f"{returns.index[0]}_{returns.index[-1]}_{len(returns)}"
            if cache_key in self.performance_cache:
                self.calculation_stats["cache_hits"] += 1
                return self.performance_cache[cache_key]
            
            # 基本統計
            total_return = (1 + returns).prod() - 1
            n_years = len(returns) / 252.0  # 営業日ベース
            annualized_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else total_return
            volatility = returns.std() * np.sqrt(252)
            
            # リスク調整指標
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # ダウンサイドリスク
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
            
            # ドローダウン分析
            if prices is not None:
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown_series = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown_series.min()
            else:
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown_series = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown_series.min()
            
            # 勝率と利益因子
            win_rate = (returns > 0).mean()
            winners = returns[returns > 0].sum()
            losers = abs(returns[returns < 0].sum())
            profit_factor = winners / losers if losers > 0 else float('inf')
            
            # 期待値メトリクス
            expected_value_metrics = self.calculate_expected_value_metrics(returns, scenarios)
            
            # リスクメトリクス
            risk_metrics = {
                RiskMetric.VOLATILITY.value: volatility,
                RiskMetric.DOWNSIDE_DEVIATION.value: downside_deviation,
                RiskMetric.SKEWNESS.value: returns.skew(),
                RiskMetric.KURTOSIS.value: returns.kurtosis()
            }
            
            # VaR and ES
            for confidence_level in [0.95, 0.99]:
                var = returns.quantile(1 - confidence_level)
                es = returns[returns <= var].mean()
                risk_metrics[f'var_{int(confidence_level*100)}'] = var
                risk_metrics[f'es_{int(confidence_level*100)}'] = es
            
            # ベンチマーク比較
            benchmark_comparison = {}
            if benchmark_returns is not None:
                # アルファとベータ
                covariance = returns.cov(benchmark_returns)
                benchmark_variance = benchmark_returns.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                alpha = returns.mean() - beta * benchmark_returns.mean()
                
                # 相関とトラッキングエラー
                correlation = returns.corr(benchmark_returns)
                tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
                information_ratio = alpha / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
                
                benchmark_comparison = {
                    'alpha': alpha * 252,  # 年率
                    'beta': beta,
                    'correlation': correlation,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio
                }
                
                risk_metrics[RiskMetric.BETA.value] = beta
                risk_metrics[RiskMetric.CORRELATION.value] = correlation
                risk_metrics[RiskMetric.TRACKING_ERROR.value] = tracking_error
            
            # 月次リターン
            returns_monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # ローリング指標
            rolling_window = min(60, len(returns) // 4)  # 最低60日、データの1/4
            if rolling_window >= 30:
                rolling_metrics = pd.DataFrame(index=returns.index)
                rolling_metrics['rolling_sharpe'] = excess_returns.rolling(window=rolling_window).apply(
                    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
                )
                rolling_metrics['rolling_volatility'] = returns.rolling(window=rolling_window).std() * np.sqrt(252)
            else:
                rolling_metrics = pd.DataFrame()
            
            # 結果の作成
            performance_analysis = PerformanceAnalysis(
                period_start=returns.index[0],
                period_end=returns.index[-1],
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                expected_value_metrics=expected_value_metrics,
                risk_metrics=risk_metrics,
                benchmark_comparison=benchmark_comparison,
                monthly_returns=returns_monthly,
                drawdown_series=drawdown_series,
                rolling_metrics=rolling_metrics
            )
            
            # キャッシュに保存
            self.performance_cache[cache_key] = performance_analysis
            
            # 統計更新
            self.calculation_stats["total_calculations"] += 1
            self.calculation_stats["calculation_time"] += (datetime.now() - start_time).total_seconds()
            
            return performance_analysis
            
        except Exception as e:
            self.logger.error(f"Performance calculation failed: {e}")
            raise
    
    def calculate_composite_strategy_performance(self, 
                                               strategy_returns: Dict[str, pd.Series],
                                               combination_weights: Dict[str, float],
                                               rebalancing_dates: Optional[List[datetime]] = None,
                                               benchmark_returns: Optional[pd.Series] = None) -> CompositeStrategyPerformance:
        """複合戦略パフォーマンスの計算"""
        
        try:
            # 個別戦略のパフォーマンス計算
            individual_performance = {}
            for strategy_name, returns in strategy_returns.items():
                performance = self.calculate_comprehensive_performance(
                    returns=returns, 
                    benchmark_returns=benchmark_returns
                )
                individual_performance[strategy_name] = performance
            
            # 複合リターンの計算
            combined_returns = self._calculate_weighted_returns(
                strategy_returns, combination_weights, rebalancing_dates
            )
            
            # 複合パフォーマンス
            combined_performance = self.calculate_comprehensive_performance(
                returns=combined_returns,
                benchmark_returns=benchmark_returns
            )
            
            # 相関行列
            returns_df = pd.DataFrame(strategy_returns)
            correlation_matrix = returns_df.corr()
            
            # 寄与度分析
            contribution_analysis = self._calculate_contribution_analysis(
                strategy_returns, combination_weights, combined_returns
            )
            
            # 分散効果の計算
            diversification_ratio = self._calculate_diversification_ratio(
                strategy_returns, combination_weights, correlation_matrix
            )
            
            # リバランスコスト
            rebalancing_cost = self._calculate_rebalancing_cost(
                strategy_returns, combination_weights, rebalancing_dates
            )
            
            # アルファ生成分析
            alpha_generation = {}
            for strategy_name, performance in individual_performance.items():
                if benchmark_returns is not None:
                    alpha_generation[strategy_name] = performance.benchmark_comparison.get('alpha', 0)
            
            # リスク分解
            risk_decomposition = self._calculate_risk_decomposition(
                strategy_returns, combination_weights, individual_performance
            )
            
            return CompositeStrategyPerformance(
                strategy_combination_id=f"composite_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                individual_strategy_performance=individual_performance,
                combined_performance=combined_performance,
                correlation_matrix=correlation_matrix,
                contribution_analysis=contribution_analysis,
                diversification_ratio=diversification_ratio,
                rebalancing_cost=rebalancing_cost,
                alpha_generation=alpha_generation,
                risk_decomposition=risk_decomposition
            )
            
        except Exception as e:
            self.logger.error(f"Composite strategy performance calculation failed: {e}")
            raise
    
    def _calculate_weighted_returns(self, 
                                   strategy_returns: Dict[str, pd.Series], 
                                   weights: Dict[str, float],
                                   rebalancing_dates: Optional[List[datetime]] = None) -> pd.Series:
        """加重リターンの計算"""
        
        # 全ての戦略の共通日付を取得
        common_dates = None
        for returns in strategy_returns.values():
            if common_dates is None:
                common_dates = set(returns.index)
            else:
                common_dates &= set(returns.index)
        
        if not common_dates:
            raise ValueError("No common dates found across strategies")
        
        common_dates = sorted(list(common_dates))
        
        # リバランシングを考慮した加重リターン計算
        if rebalancing_dates is None:
            # 固定ウェイトの場合
            weighted_returns = pd.Series(0.0, index=common_dates)
            for strategy_name, returns in strategy_returns.items():
                weight = weights.get(strategy_name, 0.0)
                strategy_returns_aligned = returns.reindex(common_dates).fillna(0)
                weighted_returns += weight * strategy_returns_aligned
        else:
            # 動的リバランシングの場合
            weighted_returns = pd.Series(0.0, index=common_dates)
            current_weights = weights.copy()
            
            for date in common_dates:
                # リバランシング日付をチェック
                if date in rebalancing_dates:
                    current_weights = weights.copy()  # ウェイトをリセット
                
                # 当日のリターンを計算
                daily_return = 0.0
                for strategy_name, returns in strategy_returns.items():
                    if date in returns.index:
                        weight = current_weights.get(strategy_name, 0.0)
                        daily_return += weight * returns[date]
                
                weighted_returns[date] = daily_return
        
        return weighted_returns
    
    def _calculate_contribution_analysis(self, 
                                       strategy_returns: Dict[str, pd.Series],
                                       weights: Dict[str, float],
                                       combined_returns: pd.Series) -> Dict[str, float]:
        """寄与度分析"""
        
        contributions = {}
        total_return = combined_returns.sum()
        
        for strategy_name, returns in strategy_returns.items():
            weight = weights.get(strategy_name, 0.0)
            aligned_returns = returns.reindex(combined_returns.index).fillna(0)
            strategy_contribution = (weight * aligned_returns).sum()
            
            if total_return != 0:
                contribution_ratio = strategy_contribution / total_return
            else:
                contribution_ratio = weight
            
            contributions[strategy_name] = contribution_ratio
        
        return contributions
    
    def _calculate_diversification_ratio(self, 
                                       strategy_returns: Dict[str, pd.Series],
                                       weights: Dict[str, float],
                                       correlation_matrix: pd.DataFrame) -> float:
        """分散効果の計算"""
        
        try:
            # 個別戦略のボラティリティ
            individual_vols = {}
            for strategy_name, returns in strategy_returns.items():
                individual_vols[strategy_name] = returns.std() * np.sqrt(252)
            
            # ウェイト配列
            strategy_names = list(strategy_returns.keys())
            weights_array = np.array([weights.get(name, 0.0) for name in strategy_names])
            vols_array = np.array([individual_vols[name] for name in strategy_names])
            
            # ポートフォリオボラティリティの理論値
            weighted_avg_vol = np.dot(weights_array, vols_array)
            
            # 実際のポートフォリオボラティリティ
            combined_returns = self._calculate_weighted_returns(strategy_returns, weights)
            actual_vol = combined_returns.std() * np.sqrt(252)
            
            # 分散効果 = 加重平均ボラティリティ / 実際のボラティリティ
            diversification_ratio = weighted_avg_vol / actual_vol if actual_vol > 0 else 1.0
            
            return max(1.0, diversification_ratio)  # 最小値は1.0
            
        except Exception as e:
            self.logger.warning(f"Diversification ratio calculation failed: {e}")
            return 1.0
    
    def _calculate_rebalancing_cost(self, 
                                  strategy_returns: Dict[str, pd.Series],
                                  weights: Dict[str, float],
                                  rebalancing_dates: Optional[List[datetime]],
                                  transaction_cost: float = 0.001) -> float:
        """リバランシングコストの計算"""
        
        if not rebalancing_dates:
            return 0.0
        
        total_cost = 0.0
        
        try:
            # 各リバランシング日でのコスト計算
            for rebal_date in rebalancing_dates:
                # 各戦略の時価ウェイト変化を推定
                weight_changes = 0.0
                for strategy_name, target_weight in weights.items():
                    # 簡単な推定: 目標ウェイトからの偏差
                    weight_changes += abs(target_weight - (1.0 / len(weights)))
                
                # トランザクションコスト = 変化量 × コスト率
                daily_cost = weight_changes * transaction_cost
                total_cost += daily_cost
            
            # 年率コストに変換
            days_span = (max(rebalancing_dates) - min(rebalancing_dates)).days
            if days_span > 0:
                annual_cost = total_cost * (365.0 / days_span)
            else:
                annual_cost = total_cost
            
            return annual_cost
            
        except Exception as e:
            self.logger.warning(f"Rebalancing cost calculation failed: {e}")
            return 0.0
    
    def _calculate_risk_decomposition(self, 
                                    strategy_returns: Dict[str, pd.Series],
                                    weights: Dict[str, float],
                                    individual_performance: Dict[str, PerformanceAnalysis]) -> Dict[str, Dict[str, float]]:
        """リスク分解分析"""
        
        risk_decomposition = {}
        
        try:
            # 各戦略のリスク寄与度
            total_risk_contribution = 0.0
            
            for strategy_name, performance in individual_performance.items():
                weight = weights.get(strategy_name, 0.0)
                strategy_vol = performance.volatility
                
                # リスク寄与度 = ウェイト × ボラティリティ
                risk_contribution = weight * strategy_vol
                total_risk_contribution += risk_contribution
                
                risk_decomposition[strategy_name] = {
                    'volatility_contribution': risk_contribution,
                    'weight': weight,
                    'individual_volatility': strategy_vol
                }
            
            # 相対的なリスク寄与度を計算
            if total_risk_contribution > 0:
                for strategy_name in risk_decomposition:
                    vol_contrib = risk_decomposition[strategy_name]['volatility_contribution']
                    risk_decomposition[strategy_name]['relative_risk_contribution'] = (
                        vol_contrib / total_risk_contribution
                    )
            
        except Exception as e:
            self.logger.warning(f"Risk decomposition calculation failed: {e}")
        
        return risk_decomposition
    
    def generate_performance_summary(self, performance: PerformanceAnalysis) -> Dict[str, Any]:
        """パフォーマンスサマリーの生成"""
        
        summary = {
            "期間": f"{performance.period_start.strftime('%Y-%m-%d')} ~ {performance.period_end.strftime('%Y-%m-%d')}",
            "総リターン": f"{performance.total_return:.2%}",
            "年率リターン": f"{performance.annualized_return:.2%}",
            "ボラティリティ": f"{performance.volatility:.2%}",
            "シャープレシオ": f"{performance.sharpe_ratio:.3f}",
            "ソルティノレシオ": f"{performance.sortino_ratio:.3f}",
            "最大ドローダウン": f"{performance.max_drawdown:.2%}",
            "勝率": f"{performance.win_rate:.2%}",
            "プロフィットファクター": f"{performance.profit_factor:.2f}",
            "期待値": {
                "期待リターン": f"{performance.expected_value_metrics.expected_return:.2%}",
                "確率重み付リターン": f"{performance.expected_value_metrics.probability_weighted_return:.2%}",
                "リスク調整期待値": f"{performance.expected_value_metrics.risk_adjusted_expected_value:.2%}",
                "ワーストケース": f"{performance.expected_value_metrics.worst_case_scenario:.2%}",
                "ベストケース": f"{performance.expected_value_metrics.best_case_scenario:.2%}"
            }
        }
        
        if performance.benchmark_comparison:
            summary["ベンチマーク比較"] = {
                "アルファ": f"{performance.benchmark_comparison.get('alpha', 0):.2%}",
                "ベータ": f"{performance.benchmark_comparison.get('beta', 0):.3f}",
                "相関": f"{performance.benchmark_comparison.get('correlation', 0):.3f}",
                "情報レシオ": f"{performance.benchmark_comparison.get('information_ratio', 0):.3f}"
            }
        
        return summary
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """計算統計の取得"""
        
        stats = self.calculation_stats.copy()
        if stats["total_calculations"] > 0:
            stats["average_calculation_time"] = stats["calculation_time"] / stats["total_calculations"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_calculations"]
        else:
            stats["average_calculation_time"] = 0.0
            stats["cache_hit_rate"] = 0.0
        
        return stats

# テスト関数
def test_enhanced_performance_calculator():
    """テスト関数"""
    logger.info("Testing EnhancedPerformanceCalculator")
    
    # サンプルデータの生成
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    
    # 計算器の初期化
    calculator = EnhancedPerformanceCalculator()
    
    # パフォーマンス計算
    performance = calculator.calculate_comprehensive_performance(returns)
    
    logger.info(f"Total return: {performance.total_return:.2%}")
    logger.info(f"Sharpe ratio: {performance.sharpe_ratio:.3f}")
    logger.info(f"Max drawdown: {performance.max_drawdown:.2%}")
    logger.info(f"Expected value: {performance.expected_value_metrics.expected_return:.2%}")
    
    # 複合戦略テスト
    strategy_returns = {
        'strategy_a': returns,
        'strategy_b': pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)
    }
    weights = {'strategy_a': 0.6, 'strategy_b': 0.4}
    
    composite_performance = calculator.calculate_composite_strategy_performance(
        strategy_returns, weights
    )
    
    logger.info(f"Composite Sharpe ratio: {composite_performance.combined_performance.sharpe_ratio:.3f}")
    logger.info(f"Diversification ratio: {composite_performance.diversification_ratio:.3f}")
    
    # サマリー生成
    summary = calculator.generate_performance_summary(performance)
    logger.info(f"Performance summary: {summary}")
    
    # 統計表示
    stats = calculator.get_calculation_stats()
    logger.info(f"Calculation stats: {stats}")
    
    return performance

if __name__ == "__main__":
    # テスト実行
    result = test_enhanced_performance_calculator()
    print(f"Test completed successfully")
