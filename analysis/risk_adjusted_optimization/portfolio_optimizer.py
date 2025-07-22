"""
Module: Portfolio Optimizer
File: portfolio_optimizer.py
Description: 
  5-1-3「リスク調整後リターンの最適化」
  ポートフォリオ特化最適化クラス

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
from pathlib import Path
import warnings

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 内部モジュールのインポート
try:
    from .risk_return_optimizer import (
        RiskAdjustedOptimizationEngine, OptimizationContext, RiskAdjustedOptimizationResult
    )
    from .objective_function_builder import OptimizationObjective
    from .constraint_manager import ConstraintType
    from .performance_evaluator import ComprehensivePerformanceReport
except ImportError:
    # 絶対インポートで再試行
    from analysis.risk_adjusted_optimization.risk_return_optimizer import (
        RiskAdjustedOptimizationEngine, OptimizationContext, RiskAdjustedOptimizationResult
    )
    from analysis.risk_adjusted_optimization.objective_function_builder import OptimizationObjective
    from analysis.risk_adjusted_optimization.constraint_manager import ConstraintType
    from analysis.risk_adjusted_optimization.performance_evaluator import ComprehensivePerformanceReport

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioOptimizationProfile:
    """ポートフォリオ最適化プロファイル"""
    profile_name: str
    risk_tolerance: str  # 'conservative', 'moderate', 'aggressive'
    return_target: Optional[float] = None
    max_drawdown_tolerance: float = 0.15
    min_diversification: float = 0.3
    rebalancing_frequency: str = "monthly"
    optimization_objectives: List[OptimizationObjective] = field(default_factory=list)
    custom_constraints: Dict[str, Any] = field(default_factory=dict)
    performance_benchmark: str = "equal_weight"

@dataclass
class MultiPeriodOptimizationRequest:
    """マルチ期間最適化リクエスト"""
    optimization_horizons: List[int] = field(default_factory=lambda: [63, 126, 252])  # 3M, 6M, 1Y
    weight_decay_factor: float = 0.95  # 古いデータの重み減衰
    rolling_window: int = 252  # ローリングウィンドウ
    min_data_points: int = 63  # 最小データポイント
    confidence_threshold: float = 0.6  # 最適化採用の信頼度しきい値

@dataclass
class PortfolioOptimizationResult:
    """ポートフォリオ最適化統合結果"""
    primary_result: RiskAdjustedOptimizationResult
    alternative_allocations: Dict[str, Dict[str, float]]
    multi_period_analysis: Dict[int, RiskAdjustedOptimizationResult]
    risk_profile_analysis: Dict[str, float]
    optimization_robustness: Dict[str, float]
    recommendation_summary: List[str]
    confidence_assessment: Dict[str, float]
    execution_plan: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedPortfolioOptimizer:
    """高度なポートフォリオ最適化クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_advanced_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 基本最適化エンジン
        self.base_optimizer = RiskAdjustedOptimizationEngine(config_path)
        
        # 最適化履歴とプロファイル
        self.optimization_profiles = {}
        self.optimization_history = []
        
    def _load_advanced_config(self) -> Dict[str, Any]:
        """高度な設定をロード"""
        
        base_config = self.base_optimizer._load_config() if hasattr(self, 'base_optimizer') else {}
        
        # 拡張設定
        advanced_config = {
            'risk_profiles': {
                'conservative': {
                    'max_volatility': 0.12,
                    'max_drawdown': 0.08,
                    'min_diversification': 0.5,
                    'objective_weights': {
                        'sharpe': 0.2,
                        'sortino': 0.2,
                        'calmar': 0.4,
                        'drawdown': 0.2
                    }
                },
                'moderate': {
                    'max_volatility': 0.20,
                    'max_drawdown': 0.15,
                    'min_diversification': 0.3,
                    'objective_weights': {
                        'sharpe': 0.4,
                        'sortino': 0.3,
                        'calmar': 0.2,
                        'drawdown': 0.1
                    }
                },
                'aggressive': {
                    'max_volatility': 0.35,
                    'max_drawdown': 0.25,
                    'min_diversification': 0.2,
                    'objective_weights': {
                        'sharpe': 0.5,
                        'sortino': 0.3,
                        'calmar': 0.1,
                        'drawdown': 0.1
                    }
                }
            },
            'multi_period': {
                'enable_multi_period': True,
                'horizons': [63, 126, 252],  # 3M, 6M, 1Y
                'horizon_weights': [0.3, 0.4, 0.3]
            },
            'robustness': {
                'bootstrap_samples': 100,
                'monte_carlo_scenarios': 500,
                'sensitivity_analysis': True
            },
            'execution': {
                'minimum_weight_change': 0.02,
                'transaction_cost': 0.001,
                'implementation_shortfall': 0.0005
            }
        }
        
        # 基本設定とマージ
        base_config.update(advanced_config)
        return base_config
    
    def optimize_portfolio_comprehensive(
        self,
        context: OptimizationContext,
        profile: PortfolioOptimizationProfile,
        multi_period_request: Optional[MultiPeriodOptimizationRequest] = None
    ) -> PortfolioOptimizationResult:
        """包括的ポートフォリオ最適化"""
        
        self.logger.info(f"Starting comprehensive portfolio optimization with profile: {profile.profile_name}")
        
        try:
            # 1. プライマリ最適化の実行
            primary_context = self._adapt_context_to_profile(context, profile)
            primary_result = self.base_optimizer.optimize_portfolio_allocation(primary_context)
            
            # 2. 代替配分の生成
            alternative_allocations = self._generate_alternative_allocations(
                primary_context, profile, primary_result
            )
            
            # 3. マルチ期間分析
            multi_period_analysis = {}
            if multi_period_request:
                multi_period_analysis = self._perform_multi_period_analysis(
                    context, profile, multi_period_request
                )
            
            # 4. リスクプロファイル分析
            risk_profile_analysis = self._analyze_risk_profile_fit(
                primary_result, profile
            )
            
            # 5. 最適化頑健性評価
            robustness_metrics = self._evaluate_optimization_robustness(
                primary_result, context, profile
            )
            
            # 6. 統合推奨事項の生成
            integrated_recommendations = self._generate_integrated_recommendations(
                primary_result, alternative_allocations, multi_period_analysis,
                risk_profile_analysis, profile
            )
            
            # 7. 信頼度評価
            confidence_assessment = self._assess_overall_confidence(
                primary_result, alternative_allocations, robustness_metrics
            )
            
            # 8. 実行プランの生成
            execution_plan = self._create_execution_plan(
                primary_result, context, profile, confidence_assessment
            )
            
            # 統合結果の構築
            comprehensive_result = PortfolioOptimizationResult(
                primary_result=primary_result,
                alternative_allocations=alternative_allocations,
                multi_period_analysis=multi_period_analysis,
                risk_profile_analysis=risk_profile_analysis,
                optimization_robustness=robustness_metrics,
                recommendation_summary=integrated_recommendations,
                confidence_assessment=confidence_assessment,
                execution_plan=execution_plan
            )
            
            # 履歴に追加
            self.optimization_history.append(comprehensive_result)
            
            self.logger.info("Comprehensive portfolio optimization completed successfully")
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive portfolio optimization failed: {e}")
            raise
    
    def _adapt_context_to_profile(
        self, 
        context: OptimizationContext, 
        profile: PortfolioOptimizationProfile
    ) -> OptimizationContext:
        """コンテキストをプロファイルに適応"""
        
        adapted_context = OptimizationContext(
            strategy_returns=context.strategy_returns,
            current_weights=context.current_weights,
            previous_weights=context.previous_weights,
            benchmark_returns=context.benchmark_returns,
            market_volatility=context.market_volatility,
            trend_strength=context.trend_strength,
            market_regime=context.market_regime,
            optimization_horizon=context.optimization_horizon,
            rebalancing_frequency=profile.rebalancing_frequency,
            timestamp=context.timestamp
        )
        
        # プロファイルに基づく調整
        risk_config = self.config['risk_profiles'].get(profile.risk_tolerance, {})
        
        if risk_config:
            # マーケットボラティリティの調整
            max_volatility = risk_config.get('max_volatility', 0.25)
            if adapted_context.market_volatility > max_volatility:
                self.logger.info(f"Adjusting market volatility from {adapted_context.market_volatility} to {max_volatility}")
                adapted_context.market_volatility = max_volatility
        
        return adapted_context
    
    def _generate_alternative_allocations(
        self,
        context: OptimizationContext,
        profile: PortfolioOptimizationProfile,
        primary_result: RiskAdjustedOptimizationResult
    ) -> Dict[str, Dict[str, float]]:
        """代替配分を生成"""
        
        alternatives = {}
        
        try:
            # 1. 等重み配分
            equal_weights = {
                strategy: 1.0 / len(context.current_weights)
                for strategy in context.current_weights.keys()
            }
            alternatives['equal_weight'] = equal_weights
            
            # 2. 逆ボラティリティ重み付け
            if len(context.strategy_returns) > 30:
                volatilities = context.strategy_returns.std()
                inv_vol_weights = 1.0 / volatilities
                inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
                alternatives['inverse_volatility'] = inv_vol_weights.to_dict()
            
            # 3. リスク許容度を変更した最適化
            for alt_risk_tolerance in ['conservative', 'moderate', 'aggressive']:
                if alt_risk_tolerance != profile.risk_tolerance:
                    alt_profile = PortfolioOptimizationProfile(
                        profile_name=f"alt_{alt_risk_tolerance}",
                        risk_tolerance=alt_risk_tolerance
                    )
                    
                    alt_context = self._adapt_context_to_profile(context, alt_profile)
                    
                    # 簡単な最適化を実行
                    try:
                        alt_result = self.base_optimizer.optimize_portfolio_allocation(alt_context)
                        if alt_result.optimization_success:
                            alternatives[f'risk_{alt_risk_tolerance}'] = alt_result.optimal_weights
                    except Exception as e:
                        self.logger.warning(f"Failed to generate alternative for {alt_risk_tolerance}: {e}")
            
            # 4. 最小分散ポートフォリオ
            try:
                min_var_weights = self._calculate_minimum_variance_weights(context.strategy_returns)
                if min_var_weights:
                    alternatives['minimum_variance'] = min_var_weights
            except Exception as e:
                self.logger.warning(f"Failed to calculate minimum variance weights: {e}")
            
            # 5. 現在の重みとの混合
            if primary_result.optimization_success:
                mixed_weights = {}
                for strategy in context.current_weights.keys():
                    current_weight = context.current_weights[strategy]
                    optimal_weight = primary_result.optimal_weights.get(strategy, 0)
                    mixed_weights[strategy] = 0.7 * optimal_weight + 0.3 * current_weight
                
                alternatives['conservative_transition'] = mixed_weights
            
        except Exception as e:
            self.logger.error(f"Error generating alternative allocations: {e}")
        
        return alternatives
    
    def _calculate_minimum_variance_weights(self, returns: pd.DataFrame) -> Optional[Dict[str, float]]:
        """最小分散重みを計算"""
        
        try:
            # 共分散行列の計算
            cov_matrix = returns.cov().values
            
            # 正則化（特異値対応）
            regularization = 1e-6
            cov_matrix += np.eye(len(cov_matrix)) * regularization
            
            # 最小分散重みの計算
            ones = np.ones((len(cov_matrix), 1))
            inv_cov = np.linalg.inv(cov_matrix)
            
            weights = inv_cov @ ones
            weights = weights / np.sum(weights)
            
            # 辞書形式に変換
            weight_dict = {}
            for i, strategy in enumerate(returns.columns):
                weight_dict[strategy] = float(weights[i, 0])
            
            return weight_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating minimum variance weights: {e}")
            return None
    
    def _perform_multi_period_analysis(
        self,
        context: OptimizationContext,
        profile: PortfolioOptimizationProfile,
        request: MultiPeriodOptimizationRequest
    ) -> Dict[int, RiskAdjustedOptimizationResult]:
        """マルチ期間分析を実行"""
        
        multi_period_results = {}
        
        try:
            for horizon in request.optimization_horizons:
                # データの期間制限
                if len(context.strategy_returns) >= horizon:
                    horizon_returns = context.strategy_returns.tail(horizon)
                    
                    # ホライゾン特化コンテキスト
                    horizon_context = OptimizationContext(
                        strategy_returns=horizon_returns,
                        current_weights=context.current_weights,
                        previous_weights=context.previous_weights,
                        benchmark_returns=context.benchmark_returns.tail(horizon) if context.benchmark_returns is not None else None,
                        market_volatility=context.market_volatility,
                        trend_strength=context.trend_strength,
                        market_regime=context.market_regime,
                        optimization_horizon=horizon,
                        rebalancing_frequency=profile.rebalancing_frequency
                    )
                    
                    # プロファイルに適応
                    adapted_context = self._adapt_context_to_profile(horizon_context, profile)
                    
                    # 最適化実行
                    try:
                        horizon_result = self.base_optimizer.optimize_portfolio_allocation(adapted_context)
                        multi_period_results[horizon] = horizon_result
                        
                        self.logger.info(f"Multi-period optimization for {horizon} days completed")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed multi-period optimization for {horizon} days: {e}")
                
                else:
                    self.logger.warning(f"Insufficient data for {horizon} day horizon: {len(context.strategy_returns)} available")
        
        except Exception as e:
            self.logger.error(f"Error in multi-period analysis: {e}")
        
        return multi_period_results
    
    def _analyze_risk_profile_fit(
        self,
        result: RiskAdjustedOptimizationResult,
        profile: PortfolioOptimizationProfile
    ) -> Dict[str, float]:
        """リスクプロファイル適合性を分析"""
        
        analysis = {}
        
        try:
            risk_config = self.config['risk_profiles'].get(profile.risk_tolerance, {})
            
            # ボラティリティ適合性
            portfolio_volatility = result.performance_report.metrics.get('portfolio_volatility', 0)
            max_volatility = risk_config.get('max_volatility', 0.25)
            analysis['volatility_fit'] = min(1.0, max_volatility / max(portfolio_volatility, 0.01))
            
            # ドローダウン適合性
            max_drawdown = result.performance_report.metrics.get('max_drawdown', 0)
            max_dd_tolerance = risk_config.get('max_drawdown', 0.15)
            analysis['drawdown_fit'] = min(1.0, max_dd_tolerance / max(abs(max_drawdown), 0.01))
            
            # 分散投資適合性
            hhi = sum(w**2 for w in result.optimal_weights.values())
            min_diversification = risk_config.get('min_diversification', 0.3)
            current_diversification = 1.0 - hhi
            analysis['diversification_fit'] = min(1.0, current_diversification / min_diversification)
            
            # 総合適合性
            analysis['overall_fit'] = np.mean([
                analysis['volatility_fit'],
                analysis['drawdown_fit'],
                analysis['diversification_fit']
            ])
            
            # リターン目標適合性（設定されている場合）
            if profile.return_target is not None:
                portfolio_returns = result.performance_report.metrics.get('annualized_return', 0)
                if portfolio_returns > 0:
                    analysis['return_target_fit'] = min(1.0, portfolio_returns / profile.return_target)
                else:
                    analysis['return_target_fit'] = 0.0
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk profile fit: {e}")
            analysis['overall_fit'] = 0.5  # デフォルト値
        
        return analysis
    
    def _evaluate_optimization_robustness(
        self,
        result: RiskAdjustedOptimizationResult,
        context: OptimizationContext,
        profile: PortfolioOptimizationProfile
    ) -> Dict[str, float]:
        """最適化の頑健性を評価"""
        
        robustness = {}
        
        try:
            # 基本頑健性指標
            robustness['convergence_quality'] = result.optimization_result.confidence_score
            robustness['constraint_satisfaction'] = 1.0 if result.constraint_result.is_satisfied else 0.0
            
            # 重み変化の安定性
            if context.previous_weights:
                weight_changes = []
                for strategy in result.optimal_weights.keys():
                    current_weight = result.optimal_weights[strategy]
                    previous_weight = context.previous_weights.get(strategy, 0.0)
                    weight_changes.append(abs(current_weight - previous_weight))
                
                max_change = max(weight_changes) if weight_changes else 0
                robustness['weight_stability'] = max(0.0, 1.0 - max_change * 2)  # 50%変化で0になる
            else:
                robustness['weight_stability'] = 1.0
            
            # データ品質による頑健性
            data_length = len(context.strategy_returns)
            min_data_requirement = 126  # 6ヶ月
            optimal_data_length = 504  # 2年
            
            if data_length >= optimal_data_length:
                robustness['data_sufficiency'] = 1.0
            elif data_length >= min_data_requirement:
                robustness['data_sufficiency'] = data_length / optimal_data_length
            else:
                robustness['data_sufficiency'] = 0.5 * (data_length / min_data_requirement)
            
            # パフォーマンス一貫性
            sharpe_ratio = result.performance_report.risk_adjusted_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio > 1.5:
                robustness['performance_consistency'] = 1.0
            elif sharpe_ratio > 1.0:
                robustness['performance_consistency'] = 0.8
            elif sharpe_ratio > 0.5:
                robustness['performance_consistency'] = 0.6
            else:
                robustness['performance_consistency'] = 0.4
            
            # 総合頑健性スコア
            robustness['overall_robustness'] = np.mean([
                robustness['convergence_quality'],
                robustness['constraint_satisfaction'],
                robustness['weight_stability'],
                robustness['data_sufficiency'],
                robustness['performance_consistency']
            ])
            
        except Exception as e:
            self.logger.error(f"Error evaluating robustness: {e}")
            robustness['overall_robustness'] = 0.5
        
        return robustness
    
    def _generate_integrated_recommendations(
        self,
        primary_result: RiskAdjustedOptimizationResult,
        alternatives: Dict[str, Dict[str, float]],
        multi_period: Dict[int, RiskAdjustedOptimizationResult],
        risk_analysis: Dict[str, float],
        profile: PortfolioOptimizationProfile
    ) -> List[str]:
        """統合推奨事項を生成"""
        
        recommendations = []
        
        try:
            # プライマリ結果からの推奨
            recommendations.extend(primary_result.recommendations[:2])  # 上位2件
            
            # リスクプロファイル適合性からの推奨
            overall_fit = risk_analysis.get('overall_fit', 0.5)
            if overall_fit < 0.7:
                recommendations.append(
                    f"リスクプロファイル（{profile.risk_tolerance}）への適合度が低いです（{overall_fit:.1%}）。"
                    "プロファイル設定の見直しを検討してください。"
                )
            elif overall_fit > 0.9:
                recommendations.append(
                    f"リスクプロファイルへの適合度が優秀です（{overall_fit:.1%}）。"
                    "現在の設定を維持することを推奨します。"
                )
            
            # 代替配分からの推奨
            if alternatives:
                best_alternative = None
                best_score = 0
                
                for alt_name, alt_weights in alternatives.items():
                    # 簡易スコア計算（実装依存）
                    diversification = 1 - sum(w**2 for w in alt_weights.values())
                    if diversification > best_score:
                        best_score = diversification
                        best_alternative = alt_name
                
                if best_alternative and best_alternative != 'equal_weight':
                    recommendations.append(
                        f"代替配分「{best_alternative}」も検討に値します（分散化スコア: {best_score:.3f}）。"
                    )
            
            # マルチ期間分析からの推奨
            if multi_period:
                successful_horizons = [h for h, r in multi_period.items() if r.optimization_success]
                if len(successful_horizons) >= 2:
                    recommendations.append(
                        f"マルチ期間分析では {len(successful_horizons)}/{len(multi_period)} の期間で最適化が成功しています。"
                        "期間横断的な一貫性が確認されました。"
                    )
                elif len(successful_horizons) == 1:
                    recommendations.append(
                        "マルチ期間分析で一部の期間のみ最適化が成功しました。"
                        "時間軸を考慮した慎重な適用を推奨します。"
                    )
            
            # 実行に関する推奨
            confidence = primary_result.confidence_level
            if confidence > 0.8:
                recommendations.append(
                    "高い信頼度での最適化です。段階的な実装（例：50%ずつ2回に分けて）を推奨します。"
                )
            elif confidence > 0.6:
                recommendations.append(
                    "中程度の信頼度です。小規模なテスト実装から開始することを推奨します。"
                )
            else:
                recommendations.append(
                    "信頼度が低いです。追加分析とパラメータ調整後に再実行を推奨します。"
                )
            
            # 重複除去
            unique_recommendations = list(dict.fromkeys(recommendations))
            
        except Exception as e:
            self.logger.error(f"Error generating integrated recommendations: {e}")
            recommendations = ["統合推奨事項の生成中にエラーが発生しました。個別結果を参照してください。"]
        
        return unique_recommendations[:8]  # 最大8件に制限
    
    def _assess_overall_confidence(
        self,
        primary_result: RiskAdjustedOptimizationResult,
        alternatives: Dict[str, Dict[str, float]],
        robustness: Dict[str, float]
    ) -> Dict[str, float]:
        """総合信頼度を評価"""
        
        confidence = {}
        
        try:
            # プライマリ結果の信頼度
            confidence['primary_optimization'] = primary_result.confidence_level
            
            # 頑健性スコア
            confidence['robustness_score'] = robustness.get('overall_robustness', 0.5)
            
            # 代替配分の一貫性
            if alternatives and len(alternatives) >= 2:
                # 重み分布の類似度を計算
                primary_weights = primary_result.optimal_weights
                similarities = []
                
                for alt_name, alt_weights in alternatives.items():
                    if alt_name != 'equal_weight':  # 等重みは除外
                        similarity = self._calculate_weight_similarity(primary_weights, alt_weights)
                        similarities.append(similarity)
                
                if similarities:
                    confidence['alternative_consistency'] = np.mean(similarities)
                else:
                    confidence['alternative_consistency'] = 0.5
            else:
                confidence['alternative_consistency'] = 0.5
            
            # データ品質信頼度
            data_confidence = robustness.get('data_sufficiency', 0.5)
            performance_confidence = robustness.get('performance_consistency', 0.5)
            confidence['data_quality'] = (data_confidence + performance_confidence) / 2
            
            # 総合信頼度（重み付き平均）
            weights = [0.4, 0.3, 0.2, 0.1]  # プライマリ、頑健性、代替一貫性、データ品質
            values = [
                confidence['primary_optimization'],
                confidence['robustness_score'],
                confidence['alternative_consistency'],
                confidence['data_quality']
            ]
            
            confidence['overall_confidence'] = sum(w * v for w, v in zip(weights, values))
            
        except Exception as e:
            self.logger.error(f"Error assessing overall confidence: {e}")
            confidence['overall_confidence'] = 0.5
        
        return confidence
    
    def _calculate_weight_similarity(
        self, 
        weights1: Dict[str, float], 
        weights2: Dict[str, float]
    ) -> float:
        """重み分布の類似度を計算"""
        
        try:
            # 共通の戦略を取得
            common_strategies = set(weights1.keys()) & set(weights2.keys())
            
            if not common_strategies:
                return 0.0
            
            # コサイン類似度を計算
            vec1 = np.array([weights1.get(s, 0) for s in common_strategies])
            vec2 = np.array([weights2.get(s, 0) for s in common_strategies])
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return max(0.0, similarity)  # 負の値を0にクリップ
            
        except Exception as e:
            self.logger.error(f"Error calculating weight similarity: {e}")
            return 0.0
    
    def _create_execution_plan(
        self,
        result: RiskAdjustedOptimizationResult,
        context: OptimizationContext,
        profile: PortfolioOptimizationProfile,
        confidence: Dict[str, float]
    ) -> Dict[str, Any]:
        """実行プランを作成"""
        
        plan = {}
        
        try:
            overall_confidence = confidence.get('overall_confidence', 0.5)
            
            # 実行戦略の決定
            if overall_confidence > 0.8:
                plan['execution_strategy'] = 'full_implementation'
                plan['implementation_phases'] = ['immediate']
                plan['phase_allocations'] = [1.0]
            elif overall_confidence > 0.6:
                plan['execution_strategy'] = 'phased_implementation'
                plan['implementation_phases'] = ['phase1', 'phase2']
                plan['phase_allocations'] = [0.6, 0.4]
            else:
                plan['execution_strategy'] = 'conservative_implementation'
                plan['implementation_phases'] = ['test', 'evaluation', 'gradual']
                plan['phase_allocations'] = [0.2, 0.0, 0.8]  # テスト、評価、段階実装
            
            # 重み変更の大きさに基づく調整
            if context.previous_weights:
                total_change = sum(
                    abs(result.optimal_weights.get(s, 0) - context.previous_weights.get(s, 0))
                    for s in set(result.optimal_weights.keys()) | set(context.previous_weights.keys())
                )
                
                plan['total_weight_change'] = total_change / 2  # 正規化
                
                if plan['total_weight_change'] > 0.5:  # 大幅変更
                    plan['recommended_phases'] = max(3, len(plan['implementation_phases']))
                    plan['monitoring_frequency'] = 'daily'
                elif plan['total_weight_change'] > 0.2:  # 中程度変更
                    plan['recommended_phases'] = max(2, len(plan['implementation_phases']))
                    plan['monitoring_frequency'] = 'weekly'
                else:  # 小幅変更
                    plan['recommended_phases'] = 1
                    plan['monitoring_frequency'] = 'monthly'
            else:
                plan['total_weight_change'] = 0.0
                plan['recommended_phases'] = 1
                plan['monitoring_frequency'] = 'weekly'
            
            # トランザクションコスト推定
            transaction_cost_rate = self.config.get('execution', {}).get('transaction_cost', 0.001)
            plan['estimated_transaction_costs'] = plan['total_weight_change'] * transaction_cost_rate
            
            # リバランシング頻度の推奨
            plan['rebalancing_frequency'] = profile.rebalancing_frequency
            
            # 次回見直し時期
            if profile.rebalancing_frequency == 'daily':
                plan['next_review_date'] = datetime.now() + timedelta(days=1)
            elif profile.rebalancing_frequency == 'weekly':
                plan['next_review_date'] = datetime.now() + timedelta(weeks=1)
            elif profile.rebalancing_frequency == 'monthly':
                plan['next_review_date'] = datetime.now() + timedelta(days=30)
            else:  # quarterly
                plan['next_review_date'] = datetime.now() + timedelta(days=90)
            
            # 停止条件
            plan['stop_loss_conditions'] = {
                'max_drawdown_limit': profile.max_drawdown_tolerance * 1.2,
                'volatility_limit': self.config['risk_profiles'][profile.risk_tolerance].get('max_volatility', 0.25) * 1.1,
                'consecutive_losses': 5
            }
            
        except Exception as e:
            self.logger.error(f"Error creating execution plan: {e}")
            plan = {'execution_strategy': 'manual_review_required'}
        
        return plan
    
    def create_optimization_profile(
        self,
        profile_name: str,
        risk_tolerance: str,
        **kwargs
    ) -> PortfolioOptimizationProfile:
        """最適化プロファイルを作成"""
        
        profile = PortfolioOptimizationProfile(
            profile_name=profile_name,
            risk_tolerance=risk_tolerance,
            **kwargs
        )
        
        self.optimization_profiles[profile_name] = profile
        return profile
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化サマリーを取得"""
        
        if not self.optimization_history:
            return {'total_optimizations': 0}
        
        successful_optimizations = [
            r for r in self.optimization_history 
            if r.primary_result.optimization_success
        ]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'success_rate': len(successful_optimizations) / len(self.optimization_history),
            'average_confidence': np.mean([r.confidence_assessment['overall_confidence'] for r in self.optimization_history]),
            'average_robustness': np.mean([r.optimization_robustness['overall_robustness'] for r in self.optimization_history]),
            'profiles_used': len(self.optimization_profiles)
        }


# テスト用のメイン関数
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    logger.info("Testing Advanced Portfolio Optimizer...")
    
    # テストデータの生成
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    strategy_returns = pd.DataFrame({
        'strategy1': np.random.normal(0.001, 0.02, len(dates)),
        'strategy2': np.random.normal(0.0015, 0.025, len(dates)),
        'strategy3': np.random.normal(0.0008, 0.018, len(dates)),
        'strategy4': np.random.normal(0.0012, 0.022, len(dates))
    }, index=dates)
    
    # 初期重み
    current_weights = {
        'strategy1': 0.3,
        'strategy2': 0.3,
        'strategy3': 0.2,
        'strategy4': 0.2
    }
    
    # 最適化コンテキスト
    context = OptimizationContext(
        strategy_returns=strategy_returns,
        current_weights=current_weights,
        market_volatility=0.20,
        trend_strength=0.05,
        market_regime="normal"
    )
    
    # 高度なポートフォリオオプティマイザーのテスト
    optimizer = AdvancedPortfolioOptimizer()
    
    # プロファイルの作成
    profile = optimizer.create_optimization_profile(
        profile_name="test_moderate",
        risk_tolerance="moderate",
        return_target=0.08,
        max_drawdown_tolerance=0.15
    )
    
    # マルチ期間リクエスト
    multi_period_request = MultiPeriodOptimizationRequest(
        optimization_horizons=[63, 126, 252],
        confidence_threshold=0.6
    )
    
    # 包括的最適化の実行
    result = optimizer.optimize_portfolio_comprehensive(
        context, profile, multi_period_request
    )
    
    logger.info("Comprehensive Optimization Results:")
    logger.info(f"Primary Success: {result.primary_result.optimization_success}")
    logger.info(f"Overall Confidence: {result.confidence_assessment['overall_confidence']:.4f}")
    logger.info(f"Risk Profile Fit: {result.risk_profile_analysis.get('overall_fit', 0):.4f}")
    logger.info(f"Robustness Score: {result.optimization_robustness['overall_robustness']:.4f}")
    logger.info(f"Execution Strategy: {result.execution_plan.get('execution_strategy', 'unknown')}")
    logger.info(f"Alternative Allocations: {len(result.alternative_allocations)}")
    logger.info(f"Multi-period Results: {len(result.multi_period_analysis)}")
    logger.info(f"Recommendations: {len(result.recommendation_summary)}")
    
    # サマリーの表示
    summary = optimizer.get_optimization_summary()
    logger.info(f"Optimizer Summary: {summary}")
    
    logger.info("Advanced Portfolio Optimizer test completed successfully!")
