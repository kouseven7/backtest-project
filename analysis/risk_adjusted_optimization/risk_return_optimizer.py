"""
Module: Risk Return Optimizer
File: risk_return_optimizer.py
Description: 
  5-1-3「リスク調整後リターンの最適化」
  メインリスク調整後リターン最適化エンジン

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
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import warnings

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 内部モジュールのインポート
try:
    from .objective_function_builder import (
        CompositeObjectiveFunction, ObjectiveFunctionBuilder, 
        OptimizationObjective, CompositeScoreResult
    )
    from .constraint_manager import (
        RiskConstraintManager, ConstraintResult, AdaptiveConstraintAdjuster
    )
    from .optimization_algorithms import (
        OptimizationEngine, OptimizationMethod, OptimizationConfig, OptimizationResult
    )
    from .performance_evaluator import (
        EnhancedPerformanceEvaluator, ComprehensivePerformanceReport
    )
except ImportError:
    # 絶対インポートで再試行
    from analysis.risk_adjusted_optimization.objective_function_builder import (
        CompositeObjectiveFunction, ObjectiveFunctionBuilder, 
        OptimizationObjective, CompositeScoreResult
    )
    from analysis.risk_adjusted_optimization.constraint_manager import (
        RiskConstraintManager, ConstraintResult, AdaptiveConstraintAdjuster
    )
    from analysis.risk_adjusted_optimization.optimization_algorithms import (
        OptimizationEngine, OptimizationMethod, OptimizationConfig, OptimizationResult
    )
    from analysis.risk_adjusted_optimization.performance_evaluator import (
        EnhancedPerformanceEvaluator, ComprehensivePerformanceReport
    )

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationContext:
    """最適化コンテキスト"""
    strategy_returns: pd.DataFrame
    current_weights: Dict[str, float]
    previous_weights: Optional[Dict[str, float]] = None
    benchmark_returns: Optional[pd.Series] = None
    market_volatility: float = 0.2
    trend_strength: float = 0.0
    market_regime: str = "normal"
    optimization_horizon: int = 252  # 1年
    rebalancing_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RiskAdjustedOptimizationResult:
    """リスク調整最適化結果"""
    optimization_success: bool
    optimal_weights: Dict[str, float]
    original_weights: Dict[str, float]
    optimization_result: OptimizationResult
    performance_report: ComprehensivePerformanceReport
    constraint_result: ConstraintResult
    objective_score: CompositeScoreResult
    improvement_metrics: Dict[str, float]
    recommendations: List[str]
    confidence_level: float
    execution_time: float
    optimization_context: OptimizationContext
    timestamp: datetime = field(default_factory=datetime.now)

class RiskAdjustedOptimizationEngine:
    """リスク調整後リターン最適化エンジン"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # コンポーネントの初期化
        self.objective_builder = ObjectiveFunctionBuilder(config_path)
        self.constraint_manager = RiskConstraintManager(self.config.get('constraints', {}))
        self.optimization_engine = OptimizationEngine(self.config.get('optimization', {}))
        self.performance_evaluator = EnhancedPerformanceEvaluator(self.config.get('performance', {}))
        self.adaptive_adjuster = AdaptiveConstraintAdjuster(self.config.get('adaptive', {}))
        
        # 最適化履歴
        self.optimization_history = []
        
    def _load_config(self) -> Dict[str, Any]:
        """設定をロード"""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        # デフォルト設定
        return {
            'objective_weights': {
                'sharpe': 0.4,
                'sortino': 0.3,
                'calmar': 0.2,
                'drawdown': 0.1
            },
            'constraints': {
                'weight_constraint': {
                    'type': 'weight',
                    'severity': 'soft',
                    'min_weight': 0.05,
                    'max_weight': 0.6,
                    'max_single_weight': 0.4,
                    'enabled': True
                },
                'volatility_constraint': {
                    'type': 'volatility',
                    'severity': 'soft',
                    'max_portfolio_volatility': 0.25,
                    'enabled': True
                },
                'drawdown_constraint': {
                    'type': 'drawdown',
                    'severity': 'soft',
                    'max_drawdown': 0.15,
                    'enabled': True
                }
            },
            'optimization': {
                'method': 'differential_evolution',
                'max_iterations': 1000,
                'population_size': 50,
                'tolerance': 1e-6
            },
            'performance': {
                'risk_free_rate': 0.02,
                'trading_days': 252
            },
            'adaptive': {
                'enable_adaptive_constraints': True,
                'volatility_sensitivity': 0.5,
                'trend_sensitivity': 0.3
            }
        }
    
    def optimize_portfolio_allocation(
        self,
        context: OptimizationContext
    ) -> RiskAdjustedOptimizationResult:
        """ポートフォリオ配分最適化のメイン実行"""
        
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting portfolio allocation optimization...")
            
            # 1. データ前処理と検証
            validated_context = self._validate_and_preprocess_context(context)
            
            # 2. 適応的制約調整
            if self.config.get('adaptive', {}).get('enable_adaptive_constraints', True):
                self._apply_adaptive_constraints(validated_context)
            
            # 3. 目的関数の構築
            objective_function = self._build_objective_function(validated_context)
            
            # 4. 最適化実行
            optimization_result = self._execute_optimization(objective_function, validated_context)
            
            # 5. 結果検証と評価
            performance_report = self._evaluate_optimization_result(
                optimization_result, validated_context
            )
            
            # 6. 制約チェック
            constraint_result = self._check_final_constraints(
                optimization_result.optimal_weights, performance_report, validated_context
            )
            
            # 7. 改善指標の計算
            improvement_metrics = self._calculate_improvement_metrics(
                optimization_result, performance_report, validated_context
            )
            
            # 8. 推奨事項の生成
            recommendations = self._generate_recommendations(
                optimization_result, performance_report, constraint_result, validated_context
            )
            
            # 9. 信頼度レベルの計算
            confidence_level = self._calculate_overall_confidence(
                optimization_result, performance_report, constraint_result
            )
            
            # 10. 目的関数スコアの計算
            objective_score = objective_function.calculate(
                self._calculate_portfolio_returns(optimization_result.optimal_weights, validated_context)
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 結果の構築
            final_result = RiskAdjustedOptimizationResult(
                optimization_success=optimization_result.success and constraint_result.is_satisfied,
                optimal_weights=optimization_result.optimal_weights,
                original_weights=validated_context.current_weights,
                optimization_result=optimization_result,
                performance_report=performance_report,
                constraint_result=constraint_result,
                objective_score=objective_score,
                improvement_metrics=improvement_metrics,
                recommendations=recommendations,
                confidence_level=confidence_level,
                execution_time=execution_time,
                optimization_context=validated_context
            )
            
            # 履歴に追加
            self.optimization_history.append(final_result)
            
            self.logger.info(f"Optimization completed successfully in {execution_time:.2f} seconds")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # エラー時のフォールバック結果
            return self._create_fallback_result(context, execution_time, str(e))
    
    def _validate_and_preprocess_context(self, context: OptimizationContext) -> OptimizationContext:
        """コンテキストの検証と前処理"""
        
        # データの整合性チェック
        if context.strategy_returns.empty:
            raise ValueError("Strategy returns data is empty")
        
        if not context.current_weights:
            raise ValueError("Current weights not provided")
        
        # 重みの正規化
        total_weight = sum(context.current_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"Normalizing weights: total was {total_weight:.4f}")
            normalized_weights = {
                k: v / total_weight for k, v in context.current_weights.items()
            }
            context.current_weights = normalized_weights
        
        # 戦略名の整合性チェック
        strategy_names = set(context.strategy_returns.columns)
        weight_names = set(context.current_weights.keys())
        
        if not strategy_names.issubset(weight_names):
            missing_strategies = strategy_names - weight_names
            self.logger.warning(f"Missing weights for strategies: {missing_strategies}")
            for strategy in missing_strategies:
                context.current_weights[strategy] = 0.0
        
        # データの期間チェック
        min_data_points = 30  # 最低1ヶ月分のデータ
        if len(context.strategy_returns) < min_data_points:
            self.logger.warning(f"Limited data: {len(context.strategy_returns)} points, minimum recommended: {min_data_points}")
        
        return context
    
    def _apply_adaptive_constraints(self, context: OptimizationContext):
        """適応的制約調整を適用"""
        
        try:
            adjustments = self.adaptive_adjuster.adjust_constraints_for_market_conditions(
                self.constraint_manager,
                context.market_volatility,
                context.trend_strength,
                context.market_regime
            )
            
            if adjustments:
                self.logger.info(f"Applied adaptive constraint adjustments: {len(adjustments)} constraints modified")
                for constraint_name, new_config in adjustments.items():
                    self.constraint_manager.update_constraint_config(constraint_name, new_config)
                    
        except Exception as e:
            self.logger.error(f"Failed to apply adaptive constraints: {e}")
    
    def _build_objective_function(self, context: OptimizationContext) -> CompositeObjectiveFunction:
        """目的関数を構築"""
        
        objective_weights = self.config.get('objective_weights', {
            'sharpe': 0.4,
            'sortino': 0.3,
            'calmar': 0.2,
            'drawdown': 0.1
        })
        
        # 市場環境に応じた重み調整
        if context.market_regime == "volatile":
            # ボラティル環境では安定性を重視
            objective_weights['drawdown'] *= 1.5
            objective_weights['sharpe'] *= 0.8
        elif context.market_regime == "trending":
            # トレンド環境ではリターンを重視
            objective_weights['sharpe'] *= 1.3
            objective_weights['sortino'] *= 1.2
        
        # 重みの正規化
        total_weight = sum(objective_weights.values())
        objective_weights = {k: v / total_weight for k, v in objective_weights.items()}
        
        return self.objective_builder.build_composite_objective(objective_weights)
    
    def _execute_optimization(
        self, 
        objective_function: CompositeObjectiveFunction,
        context: OptimizationContext
    ) -> OptimizationResult:
        """最適化実行"""
        
        # 最適化用の目的関数ラッパー
        def optimization_objective(weights_dict: Dict[str, float]) -> float:
            try:
                # ポートフォリオリターンの計算
                portfolio_returns = self._calculate_portfolio_returns(weights_dict, context)
                
                # 複合目的関数スコアの計算
                result = objective_function.calculate(portfolio_returns)
                
                # 制約ペナルティの追加
                metrics = self._calculate_portfolio_metrics(portfolio_returns, weights_dict, context)
                constraint_result = self.constraint_manager.check_all_constraints(
                    weights_dict, metrics, previous_weights=context.previous_weights
                )
                
                # 複合スコアからペナルティを差し引き
                final_score = result.composite_score - constraint_result.total_penalty
                
                return final_score
                
            except Exception as e:
                self.logger.error(f"Error in optimization objective: {e}")
                return 0.0
        
        # 最適化設定
        optimization_config = OptimizationConfig(
            method=OptimizationMethod(self.config.get('optimization', {}).get('method', 'differential_evolution')),
            max_iterations=self.config.get('optimization', {}).get('max_iterations', 1000),
            population_size=self.config.get('optimization', {}).get('population_size', 50),
            tolerance=self.config.get('optimization', {}).get('tolerance', 1e-6),
            seed=self.config.get('optimization', {}).get('seed')
        )
        
        # 境界条件の設定
        bounds = {}
        for strategy in context.current_weights.keys():
            bounds[strategy] = (0.0, 0.6)  # 各戦略の重み範囲
        
        return self.optimization_engine.run_optimization(
            objective_function=optimization_objective,
            initial_weights=context.current_weights,
            method=optimization_config.method,
            bounds=bounds,
            optimization_config=optimization_config
        )
    
    def _calculate_portfolio_returns(
        self, 
        weights: Dict[str, float], 
        context: OptimizationContext
    ) -> pd.Series:
        """ポートフォリオリターンを計算"""
        
        try:
            portfolio_returns = pd.Series(0.0, index=context.strategy_returns.index)
            
            for strategy, weight in weights.items():
                if strategy in context.strategy_returns.columns:
                    portfolio_returns += context.strategy_returns[strategy] * weight
            
            return portfolio_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio returns: {e}")
            return pd.Series(dtype=float)
    
    def _calculate_portfolio_metrics(
        self, 
        portfolio_returns: pd.Series,
        weights: Dict[str, float],
        context: OptimizationContext
    ) -> Dict[str, float]:
        """ポートフォリオ指標を計算"""
        
        metrics = {}
        
        try:
            # 基本的な指標
            metrics['portfolio_volatility'] = portfolio_returns.std() * np.sqrt(252) if len(portfolio_returns) > 0 else 0.0
            
            # 最大ドローダウン
            if len(portfolio_returns) > 0:
                cumulative_returns = (1 + portfolio_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - running_max) / running_max
                metrics['max_drawdown'] = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
            else:
                metrics['max_drawdown'] = 0.0
            
            # 相関エクスポージャー（簡易版）
            correlation_matrix = context.strategy_returns.corr()
            correlation_risk = 0.0
            total_pairs = 0
            
            for i, strategy1 in enumerate(correlation_matrix.index):
                for j, strategy2 in enumerate(correlation_matrix.columns):
                    if i < j:  # 上三角のみ
                        correlation = correlation_matrix.loc[strategy1, strategy2]
                        weight1 = weights.get(strategy1, 0)
                        weight2 = weights.get(strategy2, 0)
                        
                        if not np.isnan(correlation):
                            correlation_risk += abs(correlation) * weight1 * weight2
                            total_pairs += 1
            
            metrics['correlation_exposure'] = correlation_risk / max(total_pairs, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
        
        return metrics
    
    def _evaluate_optimization_result(
        self,
        optimization_result: OptimizationResult,
        context: OptimizationContext
    ) -> ComprehensivePerformanceReport:
        """最適化結果の評価"""
        
        # 最適化後のポートフォリオリターン
        optimal_portfolio_returns = self._calculate_portfolio_returns(
            optimization_result.optimal_weights, context
        )
        
        # 包括的パフォーマンス評価
        return self.performance_evaluator.calculate_comprehensive_metrics(
            portfolio_returns=optimal_portfolio_returns,
            strategy_returns=context.strategy_returns,
            weights=optimization_result.optimal_weights,
            benchmark_returns=context.benchmark_returns,
            previous_weights=context.previous_weights
        )
    
    def _check_final_constraints(
        self,
        optimal_weights: Dict[str, float],
        performance_report: ComprehensivePerformanceReport,
        context: OptimizationContext
    ) -> ConstraintResult:
        """最終制約チェック"""
        
        metrics = {}
        for name, metric in performance_report.metrics.items():
            metrics[name] = metric.value
        
        return self.constraint_manager.check_all_constraints(
            optimal_weights, 
            metrics, 
            previous_weights=context.previous_weights
        )
    
    def _calculate_improvement_metrics(
        self,
        optimization_result: OptimizationResult,
        performance_report: ComprehensivePerformanceReport,
        context: OptimizationContext
    ) -> Dict[str, float]:
        """改善指標の計算"""
        
        improvements = {}
        
        try:
            # 重み変更の計算
            if context.previous_weights:
                weight_changes = []
                for strategy in optimization_result.optimal_weights.keys():
                    current_weight = optimization_result.optimal_weights[strategy]
                    previous_weight = context.previous_weights.get(strategy, 0.0)
                    weight_changes.append(abs(current_weight - previous_weight))
                
                improvements['total_weight_change'] = sum(weight_changes) / 2
                improvements['max_weight_change'] = max(weight_changes) if weight_changes else 0
            
            # パフォーマンス改善
            if 'sharpe_ratio' in performance_report.risk_adjusted_metrics:
                improvements['sharpe_improvement'] = performance_report.risk_adjusted_metrics['sharpe_ratio']
            
            # 分散投資改善
            hhi = sum(w**2 for w in optimization_result.optimal_weights.values())
            optimal_hhi = 1.0 / len(optimization_result.optimal_weights)
            improvements['diversification_improvement'] = 1.0 - (hhi / optimal_hhi) if optimal_hhi > 0 else 0
            
            # 最適化品質
            improvements['optimization_convergence'] = optimization_result.confidence_score
            
        except Exception as e:
            self.logger.error(f"Error calculating improvement metrics: {e}")
        
        return improvements
    
    def _generate_recommendations(
        self,
        optimization_result: OptimizationResult,
        performance_report: ComprehensivePerformanceReport,
        constraint_result: ConstraintResult,
        context: OptimizationContext
    ) -> List[str]:
        """推奨事項の生成"""
        
        recommendations = []
        
        try:
            # 最適化成功・失敗に基づく推奨
            if optimization_result.success:
                if optimization_result.confidence_score > 0.8:
                    recommendations.append("高い信頼度で最適化が完了しました。推奨重みの適用を検討してください。")
                elif optimization_result.confidence_score > 0.6:
                    recommendations.append("最適化は完了しましたが、信頼度は中程度です。段階的な適用を検討してください。")
                else:
                    recommendations.append("最適化の信頼度が低いです。パラメータの見直しや追加データの取得を検討してください。")
            else:
                recommendations.append("最適化が収束しませんでした。制約条件や初期値の調整を検討してください。")
            
            # 制約違反に基づく推奨
            if not constraint_result.is_satisfied:
                recommendations.append(f"制約違反が {len(constraint_result.violations)} 件検出されました。制約設定の見直しを検討してください。")
                
                for violation in constraint_result.violations[:3]:  # 上位3件
                    recommendations.append(f"制約違反: {violation.description}")
            
            # パフォーマンスに基づく推奨
            sharpe_ratio = performance_report.risk_adjusted_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio > 1.5:
                recommendations.append("優秀なシャープレシオです。現在の配分戦略を維持することを推奨します。")
            elif sharpe_ratio > 1.0:
                recommendations.append("良好なシャープレシオです。リスク管理に注意しながら運用を継続してください。")
            elif sharpe_ratio > 0.5:
                recommendations.append("シャープレシオが中程度です。リスク調整の見直しを検討してください。")
            else:
                recommendations.append("シャープレシオが低いです。戦略の見直しやリバランシング頻度の調整を検討してください。")
            
            # 分散投資に基づく推奨
            if 'herfindahl_index' in performance_report.diversification_metrics:
                hhi = performance_report.diversification_metrics['herfindahl_index']
                if hhi > 0.6:
                    recommendations.append("ポートフォリオの集中度が高いです。分散投資の強化を検討してください。")
                elif hhi < 0.2:
                    recommendations.append("分散投資が十分に行われています。現在の分散レベルを維持してください。")
            
            # 市場環境に基づく推奨
            if context.market_regime == "volatile":
                recommendations.append("市場が不安定です。リスク制限の強化と頻繁なリバランシングを検討してください。")
            elif context.market_regime == "trending":
                recommendations.append("トレンド相場です。トレンドフォロー戦略の重みを増やすことを検討してください。")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("推奨事項の生成中にエラーが発生しました。")
        
        return recommendations
    
    def _calculate_overall_confidence(
        self,
        optimization_result: OptimizationResult,
        performance_report: ComprehensivePerformanceReport,
        constraint_result: ConstraintResult
    ) -> float:
        """総合信頼度の計算"""
        
        try:
            confidence_factors = []
            
            # 最適化の信頼度
            confidence_factors.append(optimization_result.confidence_score)
            
            # パフォーマンスレポートの信頼度
            overall_perf_confidence = performance_report.confidence_scores.get('overall_confidence', 0.5)
            confidence_factors.append(overall_perf_confidence)
            
            # 制約満足度
            constraint_confidence = 1.0 if constraint_result.is_satisfied else 0.3
            confidence_factors.append(constraint_confidence)
            
            # データ品質
            data_confidence = performance_report.confidence_scores.get('data_sufficiency', 0.5)
            confidence_factors.append(data_confidence)
            
            # 重み付き平均（最適化の信頼度を重視）
            weights = [0.4, 0.3, 0.2, 0.1]
            overall_confidence = sum(c * w for c, w in zip(confidence_factors, weights))
            
            return min(1.0, max(0.0, overall_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating overall confidence: {e}")
            return 0.5
    
    def _create_fallback_result(
        self, 
        context: OptimizationContext, 
        execution_time: float, 
        error_message: str
    ) -> RiskAdjustedOptimizationResult:
        """エラー時のフォールバック結果"""
        
        # 基本的なフォールバックオブジェクトを作成
        from .optimization_algorithms import OptimizationResult
        from .performance_evaluator import ComprehensivePerformanceReport, PerformanceMetric, MetricCategory
        from .constraint_manager import ConstraintResult
        from .objective_function_builder import CompositeScoreResult
        
        fallback_optimization = OptimizationResult(
            success=False,
            optimal_weights=context.current_weights,
            optimal_value=0.0,
            iterations=0,
            function_evaluations=0,
            convergence_message=error_message,
            execution_time=execution_time,
            confidence_score=0.0
        )
        
        fallback_performance = ComprehensivePerformanceReport(
            portfolio_returns=pd.Series(dtype=float),
            benchmark_returns=None,
            strategy_returns=context.strategy_returns,
            weights=context.current_weights,
            metrics={'error': PerformanceMetric('error', MetricCategory.RISK_METRICS, 0.0, 'Error occurred')},
            risk_adjusted_metrics={'sharpe_ratio': 0.0},
            diversification_metrics={},
            stability_metrics={},
            comparison_metrics={},
            optimization_improvement={},
            confidence_scores={'overall_confidence': 0.0}
        )
        
        fallback_constraint = ConstraintResult(
            is_satisfied=False,
            total_penalty=float('inf'),
            violations=[],
            checked_constraints=[]
        )
        
        fallback_objective = CompositeScoreResult(
            composite_score=0.0,
            individual_scores={},
            optimization_direction="maximize",
            total_weight=0.0,
            score_breakdown={},
            confidence_level=0.0
        )
        
        return RiskAdjustedOptimizationResult(
            optimization_success=False,
            optimal_weights=context.current_weights,
            original_weights=context.current_weights,
            optimization_result=fallback_optimization,
            performance_report=fallback_performance,
            constraint_result=fallback_constraint,
            objective_score=fallback_objective,
            improvement_metrics={},
            recommendations=[f"最適化に失敗しました: {error_message}"],
            confidence_level=0.0,
            execution_time=execution_time,
            optimization_context=context
        )
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化サマリーを取得"""
        
        if not self.optimization_history:
            return {
                'total_optimizations': 0,
                'success_rate': 0.0,
                'average_confidence': 0.0,
                'recent_trends': {}
            }
        
        successful_optimizations = [r for r in self.optimization_history if r.optimization_success]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'success_rate': len(successful_optimizations) / len(self.optimization_history),
            'average_confidence': np.mean([r.confidence_level for r in self.optimization_history]),
            'average_execution_time': np.mean([r.execution_time for r in self.optimization_history]),
            'recent_trends': self._analyze_recent_trends(),
            'constraint_violation_rate': sum(1 for r in self.optimization_history if not r.constraint_result.is_satisfied) / len(self.optimization_history)
        }
    
    def _analyze_recent_trends(self) -> Dict[str, Any]:
        """最近の傾向を分析"""
        
        if len(self.optimization_history) < 5:
            return {}
        
        recent_results = self.optimization_history[-5:]
        
        trends = {
            'confidence_trend': 'stable',
            'performance_trend': 'stable',
            'constraint_violations_trend': 'stable'
        }
        
        try:
            # 信頼度の傾向
            confidences = [r.confidence_level for r in recent_results]
            if len(confidences) >= 3:
                if confidences[-1] > confidences[0] + 0.1:
                    trends['confidence_trend'] = 'improving'
                elif confidences[-1] < confidences[0] - 0.1:
                    trends['confidence_trend'] = 'declining'
            
            # パフォーマンスの傾向
            sharpe_ratios = [r.performance_report.risk_adjusted_metrics.get('sharpe_ratio', 0) for r in recent_results]
            if len(sharpe_ratios) >= 3:
                if sharpe_ratios[-1] > sharpe_ratios[0] + 0.2:
                    trends['performance_trend'] = 'improving'
                elif sharpe_ratios[-1] < sharpe_ratios[0] - 0.2:
                    trends['performance_trend'] = 'declining'
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
        
        return trends


# テスト用のメイン関数
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    logger.info("Testing Risk Adjusted Optimization Engine...")
    
    # テストデータの生成
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    strategy_returns = pd.DataFrame({
        'strategy1': np.random.normal(0.001, 0.02, len(dates)),
        'strategy2': np.random.normal(0.0015, 0.025, len(dates)),
        'strategy3': np.random.normal(0.0008, 0.018, len(dates))
    }, index=dates)
    
    # 初期重み
    current_weights = {
        'strategy1': 0.4,
        'strategy2': 0.3,
        'strategy3': 0.3
    }
    
    # 最適化コンテキスト
    context = OptimizationContext(
        strategy_returns=strategy_returns,
        current_weights=current_weights,
        market_volatility=0.22,
        trend_strength=0.1,
        market_regime="normal"
    )
    
    # 最適化エンジンのテスト
    engine = RiskAdjustedOptimizationEngine()
    
    result = engine.optimize_portfolio_allocation(context)
    
    logger.info("Optimization Results:")
    logger.info(f"Success: {result.optimization_success}")
    logger.info(f"Confidence Level: {result.confidence_level:.4f}")
    logger.info(f"Execution Time: {result.execution_time:.2f} seconds")
    logger.info(f"Optimal Weights: {result.optimal_weights}")
    logger.info(f"Sharpe Ratio: {result.performance_report.risk_adjusted_metrics.get('sharpe_ratio', 0):.4f}")
    logger.info(f"Constraints Satisfied: {result.constraint_result.is_satisfied}")
    logger.info(f"Number of Recommendations: {len(result.recommendations)}")
    
    # サマリーの表示
    summary = engine.get_optimization_summary()
    logger.info(f"Optimization Summary: {summary}")
    
    logger.info("Risk Adjusted Optimization Engine test completed successfully!")
