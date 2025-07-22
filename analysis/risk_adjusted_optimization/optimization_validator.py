"""
Module: Optimization Validator
File: optimization_validator.py
Description: 
  5-1-3「リスク調整後リターンの最適化」
  最適化結果の検証と品質保証システム

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
from scipy import stats
from sklearn.metrics import mean_squared_error

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 内部モジュールのインポート
try:
    from .risk_return_optimizer import RiskAdjustedOptimizationResult, OptimizationContext
    from .portfolio_optimizer import PortfolioOptimizationResult, AdvancedPortfolioOptimizer
    from .performance_evaluator import ComprehensivePerformanceReport
    from .constraint_manager import ConstraintResult
except ImportError:
    # 絶対インポートで再試行
    from analysis.risk_adjusted_optimization.risk_return_optimizer import RiskAdjustedOptimizationResult, OptimizationContext
    from analysis.risk_adjusted_optimization.portfolio_optimizer import PortfolioOptimizationResult, AdvancedPortfolioOptimizer
    from analysis.risk_adjusted_optimization.performance_evaluator import ComprehensivePerformanceReport
    from analysis.risk_adjusted_optimization.constraint_manager import ConstraintResult

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationTest:
    """個別検証テスト"""
    test_name: str
    test_category: str  # 'statistical', 'constraint', 'performance', 'robustness'
    test_result: bool
    test_score: float  # 0.0-1.0
    test_details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ValidationReport:
    """包括的検証レポート"""
    validation_success: bool
    overall_score: float  # 0.0-1.0
    category_scores: Dict[str, float]
    individual_tests: List[ValidationTest]
    critical_failures: List[str]
    warnings: List[str]
    validation_summary: Dict[str, Any]
    improvement_suggestions: List[str]
    validation_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BacktestValidationConfig:
    """バックテスト検証設定"""
    out_of_sample_periods: List[int] = field(default_factory=lambda: [30, 60, 90])  # 日数
    monte_carlo_simulations: int = 1000
    bootstrap_samples: int = 500
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    statistical_tests: List[str] = field(default_factory=lambda: ['normality', 'stationarity', 'autocorrelation'])
    performance_benchmarks: List[str] = field(default_factory=lambda: ['equal_weight', 'market_cap'])

class OptimizationValidator:
    """最適化結果検証システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_validation_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 検証履歴
        self.validation_history = []
        
    def _load_validation_config(self) -> Dict[str, Any]:
        """検証設定をロード"""
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load validation config from {self.config_path}: {e}")
        
        # デフォルト設定
        return {
            'validation_thresholds': {
                'minimum_overall_score': 0.6,
                'minimum_category_scores': {
                    'statistical': 0.5,
                    'constraint': 0.8,
                    'performance': 0.6,
                    'robustness': 0.5
                },
                'critical_failure_thresholds': {
                    'max_constraint_violations': 2,
                    'min_sharpe_ratio': -0.5,
                    'max_drawdown': 0.5
                }
            },
            'test_weights': {
                'statistical': 0.2,
                'constraint': 0.3,
                'performance': 0.3,
                'robustness': 0.2
            },
            'backtest': {
                'out_of_sample_periods': [30, 60, 90],
                'monte_carlo_simulations': 500,
                'bootstrap_samples': 200,
                'confidence_levels': [0.95, 0.99]
            }
        }
    
    def validate_optimization_result(
        self,
        optimization_result: Union[RiskAdjustedOptimizationResult, PortfolioOptimizationResult],
        context: OptimizationContext,
        backtest_config: Optional[BacktestValidationConfig] = None
    ) -> ValidationReport:
        """最適化結果の包括的検証"""
        
        self.logger.info("Starting comprehensive optimization validation...")
        
        try:
            # 結果の正規化（PortfolioOptimizationResultの場合はprimary_resultを使用）
            if isinstance(optimization_result, PortfolioOptimizationResult):
                primary_result = optimization_result.primary_result
                is_comprehensive = True
            else:
                primary_result = optimization_result
                is_comprehensive = False
            
            # 個別検証テストの実行
            validation_tests = []
            
            # 1. 統計的検証
            statistical_tests = self._run_statistical_validation(primary_result, context)
            validation_tests.extend(statistical_tests)
            
            # 2. 制約検証
            constraint_tests = self._run_constraint_validation(primary_result, context)
            validation_tests.extend(constraint_tests)
            
            # 3. パフォーマンス検証
            performance_tests = self._run_performance_validation(primary_result, context)
            validation_tests.extend(performance_tests)
            
            # 4. 頑健性検証
            robustness_tests = self._run_robustness_validation(primary_result, context, backtest_config)
            validation_tests.extend(robustness_tests)
            
            # 5. 包括的結果の追加検証（該当する場合）
            if is_comprehensive:
                comprehensive_tests = self._run_comprehensive_validation(optimization_result)
                validation_tests.extend(comprehensive_tests)
            
            # カテゴリ別スコアの計算
            category_scores = self._calculate_category_scores(validation_tests)
            
            # 総合スコアの計算
            overall_score = self._calculate_overall_score(category_scores)
            
            # 重要な失敗と警告の識別
            critical_failures, warnings = self._identify_critical_issues(validation_tests, primary_result)
            
            # バリデーション成功判定
            validation_success = self._determine_validation_success(
                overall_score, category_scores, critical_failures
            )
            
            # サマリーの生成
            validation_summary = self._generate_validation_summary(
                validation_tests, overall_score, category_scores
            )
            
            # 改善提案の生成
            improvement_suggestions = self._generate_improvement_suggestions(
                validation_tests, critical_failures, warnings
            )
            
            # 検証レポートの構築
            validation_report = ValidationReport(
                validation_success=validation_success,
                overall_score=overall_score,
                category_scores=category_scores,
                individual_tests=validation_tests,
                critical_failures=critical_failures,
                warnings=warnings,
                validation_summary=validation_summary,
                improvement_suggestions=improvement_suggestions
            )
            
            # 履歴に追加
            self.validation_history.append(validation_report)
            
            self.logger.info(f"Validation completed. Overall score: {overall_score:.3f}, Success: {validation_success}")
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Validation failed with error: {e}")
            return self._create_error_validation_report(str(e))
    
    def _run_statistical_validation(
        self, 
        result: RiskAdjustedOptimizationResult, 
        context: OptimizationContext
    ) -> List[ValidationTest]:
        """統計的検証テストを実行"""
        
        tests = []
        
        try:
            # ポートフォリオリターンの計算
            portfolio_returns = self._calculate_portfolio_returns(result.optimal_weights, context)
            
            # 正規性検定
            if len(portfolio_returns) > 20:
                try:
                    stat, p_value = stats.normaltest(portfolio_returns.dropna())
                    is_normal = p_value > 0.05
                    
                    tests.append(ValidationTest(
                        test_name="normality_test",
                        test_category="statistical",
                        test_result=is_normal,
                        test_score=min(1.0, p_value * 10),  # p値を0-1スケールに変換
                        test_details={"p_value": p_value, "statistic": stat},
                        error_message=None if is_normal else f"Returns are not normally distributed (p={p_value:.4f})",
                        recommendations=[] if is_normal else ["非正規分布を考慮したリスク管理手法の採用を検討してください"]
                    ))
                except Exception as e:
                    tests.append(self._create_failed_test("normality_test", "statistical", str(e)))
            
            # 自己相関検定
            if len(portfolio_returns) > 30:
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    ljung_box = acorr_ljungbox(portfolio_returns.dropna(), lags=10, return_df=True)
                    no_autocorr = (ljung_box['lb_pvalue'] > 0.05).all()
                    avg_p_value = ljung_box['lb_pvalue'].mean()
                    
                    tests.append(ValidationTest(
                        test_name="autocorrelation_test",
                        test_category="statistical",
                        test_result=no_autocorr,
                        test_score=min(1.0, avg_p_value * 5),
                        test_details={"avg_p_value": avg_p_value, "ljung_box_results": ljung_box.to_dict()},
                        error_message=None if no_autocorr else "Significant autocorrelation detected",
                        recommendations=[] if no_autocorr else ["リターン系列の自己相関を考慮したモデル調整を検討してください"]
                    ))
                except ImportError:
                    self.logger.warning("statsmodels not available for autocorrelation test")
                except Exception as e:
                    tests.append(self._create_failed_test("autocorrelation_test", "statistical", str(e)))
            
            # 定常性検定
            if len(portfolio_returns) > 50:
                try:
                    from statsmodels.tsa.stattools import adfuller
                    adf_result = adfuller(portfolio_returns.dropna())
                    is_stationary = adf_result[1] < 0.05
                    
                    tests.append(ValidationTest(
                        test_name="stationarity_test",
                        test_category="statistical",
                        test_result=is_stationary,
                        test_score=1.0 if is_stationary else (1.0 - adf_result[1]),
                        test_details={"p_value": adf_result[1], "adf_statistic": adf_result[0]},
                        error_message=None if is_stationary else f"Returns are not stationary (p={adf_result[1]:.4f})",
                        recommendations=[] if is_stationary else ["非定常性を考慮した最適化手法の検討が必要です"]
                    ))
                except ImportError:
                    self.logger.warning("statsmodels not available for stationarity test")
                except Exception as e:
                    tests.append(self._create_failed_test("stationarity_test", "statistical", str(e)))
            
            # 外れ値検定
            try:
                Q1 = portfolio_returns.quantile(0.25)
                Q3 = portfolio_returns.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = portfolio_returns[(portfolio_returns < lower_bound) | (portfolio_returns > upper_bound)]
                outlier_rate = len(outliers) / len(portfolio_returns)
                
                no_excessive_outliers = outlier_rate < 0.05  # 5%以下
                
                tests.append(ValidationTest(
                    test_name="outlier_detection",
                    test_category="statistical",
                    test_result=no_excessive_outliers,
                    test_score=max(0.0, 1.0 - outlier_rate * 10),
                    test_details={"outlier_rate": outlier_rate, "outlier_count": len(outliers)},
                    error_message=None if no_excessive_outliers else f"Excessive outliers detected: {outlier_rate:.1%}",
                    recommendations=[] if no_excessive_outliers else ["外れ値対応の強化を検討してください"]
                ))
            except Exception as e:
                tests.append(self._create_failed_test("outlier_detection", "statistical", str(e)))
                
        except Exception as e:
            self.logger.error(f"Error in statistical validation: {e}")
        
        return tests
    
    def _run_constraint_validation(
        self, 
        result: RiskAdjustedOptimizationResult, 
        context: OptimizationContext
    ) -> List[ValidationTest]:
        """制約検証テストを実行"""
        
        tests = []
        
        try:
            constraint_result = result.constraint_result
            
            # 制約満足度テスト
            tests.append(ValidationTest(
                test_name="constraint_satisfaction",
                test_category="constraint",
                test_result=constraint_result.is_satisfied,
                test_score=1.0 if constraint_result.is_satisfied else 0.0,
                test_details={
                    "violation_count": len(constraint_result.violations),
                    "total_penalty": constraint_result.total_penalty,
                    "violations": [v.description for v in constraint_result.violations[:5]]
                },
                error_message=None if constraint_result.is_satisfied else f"{len(constraint_result.violations)} constraint violations",
                recommendations=[] if constraint_result.is_satisfied else ["制約違反の解決が必要です"]
            ))
            
            # 重み制約テスト
            optimal_weights = result.optimal_weights
            
            # 重み合計テスト
            weight_sum = sum(optimal_weights.values())
            weight_sum_ok = abs(weight_sum - 1.0) < 0.01
            
            tests.append(ValidationTest(
                test_name="weight_sum_constraint",
                test_category="constraint",
                test_result=weight_sum_ok,
                test_score=max(0.0, 1.0 - abs(weight_sum - 1.0) * 50),
                test_details={"weight_sum": weight_sum, "deviation": abs(weight_sum - 1.0)},
                error_message=None if weight_sum_ok else f"Weight sum deviation: {weight_sum - 1.0:.4f}",
                recommendations=[] if weight_sum_ok else ["重み合計の正規化が必要です"]
            ))
            
            # 負の重みテスト
            negative_weights = {k: v for k, v in optimal_weights.items() if v < 0}
            no_negative_weights = len(negative_weights) == 0
            
            tests.append(ValidationTest(
                test_name="non_negative_weights",
                test_category="constraint",
                test_result=no_negative_weights,
                test_score=1.0 if no_negative_weights else 0.0,
                test_details={"negative_weights": negative_weights},
                error_message=None if no_negative_weights else f"Negative weights detected: {negative_weights}",
                recommendations=[] if no_negative_weights else ["負の重みの制約設定を確認してください"]
            ))
            
            # 集中度制約テスト
            hhi = sum(w**2 for w in optimal_weights.values())
            max_concentration = 0.5  # 設定可能
            concentration_ok = hhi < max_concentration
            
            tests.append(ValidationTest(
                test_name="concentration_constraint",
                test_category="constraint",
                test_result=concentration_ok,
                test_score=max(0.0, (max_concentration - hhi) / max_concentration),
                test_details={"herfindahl_index": hhi, "max_allowed": max_concentration},
                error_message=None if concentration_ok else f"Portfolio too concentrated: HHI={hhi:.3f}",
                recommendations=[] if concentration_ok else ["ポートフォリオの分散化が必要です"]
            ))
            
        except Exception as e:
            self.logger.error(f"Error in constraint validation: {e}")
            tests.append(self._create_failed_test("constraint_validation_error", "constraint", str(e)))
        
        return tests
    
    def _run_performance_validation(
        self, 
        result: RiskAdjustedOptimizationResult, 
        context: OptimizationContext
    ) -> List[ValidationTest]:
        """パフォーマンス検証テストを実行"""
        
        tests = []
        
        try:
            performance_report = result.performance_report
            
            # シャープレシオテスト
            sharpe_ratio = performance_report.risk_adjusted_metrics.get('sharpe_ratio', 0)
            min_sharpe = self.config['validation_thresholds']['critical_failure_thresholds']['min_sharpe_ratio']
            sharpe_ok = sharpe_ratio > min_sharpe
            
            # シャープレシオスコア（0-2を0-1にマッピング）
            sharpe_score = min(1.0, max(0.0, (sharpe_ratio + 1.0) / 3.0))
            
            tests.append(ValidationTest(
                test_name="sharpe_ratio_test",
                test_category="performance",
                test_result=sharpe_ok,
                test_score=sharpe_score,
                test_details={"sharpe_ratio": sharpe_ratio, "minimum_threshold": min_sharpe},
                error_message=None if sharpe_ok else f"Poor Sharpe ratio: {sharpe_ratio:.3f}",
                recommendations=[] if sharpe_ok else ["リスク調整後リターンの改善が必要です"]
            ))
            
            # 最大ドローダウンテスト
            max_drawdown = abs(performance_report.metrics.get('max_drawdown', 0))
            max_dd_threshold = self.config['validation_thresholds']['critical_failure_thresholds']['max_drawdown']
            drawdown_ok = max_drawdown < max_dd_threshold
            
            tests.append(ValidationTest(
                test_name="max_drawdown_test",
                test_category="performance",
                test_result=drawdown_ok,
                test_score=max(0.0, 1.0 - max_drawdown / max_dd_threshold),
                test_details={"max_drawdown": max_drawdown, "threshold": max_dd_threshold},
                error_message=None if drawdown_ok else f"Excessive drawdown: {max_drawdown:.1%}",
                recommendations=[] if drawdown_ok else ["ドローダウン制御の強化が必要です"]
            ))
            
            # ボラティリティテスト
            portfolio_volatility = performance_report.metrics.get('portfolio_volatility', 0)
            if portfolio_volatility > 0:
                reasonable_vol = portfolio_volatility < 0.4  # 40%以下
                vol_score = max(0.0, (0.4 - portfolio_volatility) / 0.4)
                
                tests.append(ValidationTest(
                    test_name="volatility_test",
                    test_category="performance",
                    test_result=reasonable_vol,
                    test_score=vol_score,
                    test_details={"portfolio_volatility": portfolio_volatility},
                    error_message=None if reasonable_vol else f"High volatility: {portfolio_volatility:.1%}",
                    recommendations=[] if reasonable_vol else ["ボラティリティの削減を検討してください"]
                ))
            
            # リターン安定性テスト
            portfolio_returns = self._calculate_portfolio_returns(result.optimal_weights, context)
            if len(portfolio_returns) > 30:
                rolling_returns = portfolio_returns.rolling(window=30).mean()
                return_stability = 1.0 - (rolling_returns.std() / max(abs(rolling_returns.mean()), 0.001))
                return_stability = max(0.0, min(1.0, return_stability))
                
                tests.append(ValidationTest(
                    test_name="return_stability_test",
                    test_category="performance",
                    test_result=return_stability > 0.5,
                    test_score=return_stability,
                    test_details={"stability_score": return_stability},
                    error_message=None if return_stability > 0.5 else "Unstable returns detected",
                    recommendations=[] if return_stability > 0.5 else ["リターンの安定性向上が必要です"]
                ))
            
            # ベンチマーク比較テスト（該当する場合）
            if context.benchmark_returns is not None and len(context.benchmark_returns) > 0:
                portfolio_returns = self._calculate_portfolio_returns(result.optimal_weights, context)
                common_dates = portfolio_returns.index.intersection(context.benchmark_returns.index)
                
                if len(common_dates) > 30:
                    port_ret = portfolio_returns.loc[common_dates]
                    bench_ret = context.benchmark_returns.loc[common_dates]
                    
                    # 情報比率
                    active_return = port_ret - bench_ret
                    information_ratio = active_return.mean() / max(active_return.std(), 0.001)
                    
                    ir_positive = information_ratio > 0
                    ir_score = min(1.0, max(0.0, (information_ratio + 2.0) / 4.0))
                    
                    tests.append(ValidationTest(
                        test_name="information_ratio_test",
                        test_category="performance",
                        test_result=ir_positive,
                        test_score=ir_score,
                        test_details={"information_ratio": information_ratio},
                        error_message=None if ir_positive else f"Negative information ratio: {information_ratio:.3f}",
                        recommendations=[] if ir_positive else ["ベンチマーク対比でのパフォーマンス向上が必要です"]
                    ))
            
        except Exception as e:
            self.logger.error(f"Error in performance validation: {e}")
            tests.append(self._create_failed_test("performance_validation_error", "performance", str(e)))
        
        return tests
    
    def _run_robustness_validation(
        self, 
        result: RiskAdjustedOptimizationResult, 
        context: OptimizationContext,
        backtest_config: Optional[BacktestValidationConfig]
    ) -> List[ValidationTest]:
        """頑健性検証テストを実行"""
        
        tests = []
        
        try:
            # 最適化収束テスト
            convergence_ok = result.optimization_result.success
            confidence_score = result.optimization_result.confidence_score
            
            tests.append(ValidationTest(
                test_name="optimization_convergence",
                test_category="robustness",
                test_result=convergence_ok,
                test_score=confidence_score,
                test_details={
                    "iterations": result.optimization_result.iterations,
                    "confidence_score": confidence_score,
                    "convergence_message": result.optimization_result.convergence_message
                },
                error_message=None if convergence_ok else "Optimization did not converge",
                recommendations=[] if convergence_ok else ["最適化パラメータの調整が必要です"]
            ))
            
            # データ十分性テスト
            data_length = len(context.strategy_returns)
            min_data_points = 126  # 6ヶ月
            sufficient_data = data_length >= min_data_points
            data_score = min(1.0, data_length / min_data_points)
            
            tests.append(ValidationTest(
                test_name="data_sufficiency",
                test_category="robustness",
                test_result=sufficient_data,
                test_score=data_score,
                test_details={"data_points": data_length, "minimum_required": min_data_points},
                error_message=None if sufficient_data else f"Insufficient data: {data_length} points",
                recommendations=[] if sufficient_data else ["より多くのヒストリカルデータの取得を推奨します"]
            ))
            
            # アウトオブサンプルテスト（十分なデータがある場合）
            if backtest_config and data_length > 200:
                oos_results = self._perform_out_of_sample_validation(result, context, backtest_config)
                tests.extend(oos_results)
            
            # 重み安定性テスト
            if context.previous_weights:
                weight_changes = []
                for strategy in result.optimal_weights.keys():
                    current = result.optimal_weights[strategy]
                    previous = context.previous_weights.get(strategy, 0)
                    weight_changes.append(abs(current - previous))
                
                max_change = max(weight_changes) if weight_changes else 0
                avg_change = np.mean(weight_changes) if weight_changes else 0
                
                # 適度な変化（大きすぎず小さすぎず）
                reasonable_change = 0.05 < avg_change < 0.3
                stability_score = 1.0 - min(1.0, max(0, (max_change - 0.2) / 0.3))
                
                tests.append(ValidationTest(
                    test_name="weight_stability",
                    test_category="robustness",
                    test_result=reasonable_change,
                    test_score=stability_score,
                    test_details={
                        "max_weight_change": max_change,
                        "avg_weight_change": avg_change
                    },
                    error_message=None if reasonable_change else f"Weight changes too extreme: max={max_change:.1%}",
                    recommendations=[] if reasonable_change else ["重み変化の安定性を改善してください"]
                ))
            
            # 多様性テスト
            strategy_count = len([w for w in result.optimal_weights.values() if w > 0.01])
            min_strategies = max(2, len(result.optimal_weights) // 3)
            sufficient_diversity = strategy_count >= min_strategies
            
            tests.append(ValidationTest(
                test_name="portfolio_diversity",
                test_category="robustness",
                test_result=sufficient_diversity,
                test_score=min(1.0, strategy_count / len(result.optimal_weights)),
                test_details={
                    "active_strategies": strategy_count,
                    "total_strategies": len(result.optimal_weights)
                },
                error_message=None if sufficient_diversity else f"Low diversity: {strategy_count} active strategies",
                recommendations=[] if sufficient_diversity else ["ポートフォリオの多様性を向上させてください"]
            ))
            
        except Exception as e:
            self.logger.error(f"Error in robustness validation: {e}")
            tests.append(self._create_failed_test("robustness_validation_error", "robustness", str(e)))
        
        return tests
    
    def _run_comprehensive_validation(
        self, 
        result: PortfolioOptimizationResult
    ) -> List[ValidationTest]:
        """包括的結果の追加検証"""
        
        tests = []
        
        try:
            # 代替配分の品質テスト
            alt_count = len(result.alternative_allocations)
            sufficient_alternatives = alt_count >= 3
            
            tests.append(ValidationTest(
                test_name="alternative_allocations_count",
                test_category="robustness",
                test_result=sufficient_alternatives,
                test_score=min(1.0, alt_count / 5.0),
                test_details={"alternative_count": alt_count},
                error_message=None if sufficient_alternatives else f"Few alternatives: {alt_count}",
                recommendations=[] if sufficient_alternatives else ["代替配分オプションの拡充を検討してください"]
            ))
            
            # 信頼度評価の一貫性テスト
            overall_confidence = result.confidence_assessment.get('overall_confidence', 0)
            primary_confidence = result.primary_result.confidence_level
            
            confidence_consistent = abs(overall_confidence - primary_confidence) < 0.2
            
            tests.append(ValidationTest(
                test_name="confidence_consistency",
                test_category="robustness",
                test_result=confidence_consistent,
                test_score=1.0 - abs(overall_confidence - primary_confidence),
                test_details={
                    "overall_confidence": overall_confidence,
                    "primary_confidence": primary_confidence
                },
                error_message=None if confidence_consistent else "Confidence scores inconsistent",
                recommendations=[] if confidence_consistent else ["信頼度評価の整合性を確認してください"]
            ))
            
            # 実行プランの合理性テスト
            execution_plan = result.execution_plan
            has_execution_strategy = 'execution_strategy' in execution_plan
            
            tests.append(ValidationTest(
                test_name="execution_plan_completeness",
                test_category="robustness",
                test_result=has_execution_strategy,
                test_score=1.0 if has_execution_strategy else 0.0,
                test_details={"execution_strategy": execution_plan.get('execution_strategy', 'missing')},
                error_message=None if has_execution_strategy else "Incomplete execution plan",
                recommendations=[] if has_execution_strategy else ["実行プランの詳細化が必要です"]
            ))
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive validation: {e}")
            tests.append(self._create_failed_test("comprehensive_validation_error", "robustness", str(e)))
        
        return tests
    
    def _perform_out_of_sample_validation(
        self, 
        result: RiskAdjustedOptimizationResult, 
        context: OptimizationContext,
        backtest_config: BacktestValidationConfig
    ) -> List[ValidationTest]:
        """アウトオブサンプル検証"""
        
        tests = []
        
        try:
            for oos_period in backtest_config.out_of_sample_periods:
                if len(context.strategy_returns) > oos_period + 60:  # 最低限のインサンプル期間
                    
                    # データ分割
                    in_sample = context.strategy_returns.iloc[:-oos_period]
                    out_of_sample = context.strategy_returns.iloc[-oos_period:]
                    
                    # アウトオブサンプルでのパフォーマンス計算
                    oos_portfolio_returns = self._calculate_portfolio_returns_from_data(
                        result.optimal_weights, out_of_sample
                    )
                    
                    if len(oos_portfolio_returns) > 0:
                        # シャープレシオの計算
                        oos_sharpe = (oos_portfolio_returns.mean() * 252) / (oos_portfolio_returns.std() * np.sqrt(252))
                        
                        # インサンプルとの比較
                        is_portfolio_returns = self._calculate_portfolio_returns_from_data(
                            result.optimal_weights, in_sample
                        )
                        is_sharpe = (is_portfolio_returns.mean() * 252) / (is_portfolio_returns.std() * np.sqrt(252))
                        
                        # パフォーマンス維持テスト
                        performance_maintained = oos_sharpe > is_sharpe * 0.7  # 30%以下の低下まで許容
                        performance_score = min(1.0, max(0.0, oos_sharpe / max(is_sharpe, 0.1)))
                        
                        tests.append(ValidationTest(
                            test_name=f"out_of_sample_{oos_period}d",
                            test_category="robustness",
                            test_result=performance_maintained,
                            test_score=performance_score,
                            test_details={
                                "in_sample_sharpe": is_sharpe,
                                "out_of_sample_sharpe": oos_sharpe,
                                "period_days": oos_period
                            },
                            error_message=None if performance_maintained else f"Poor OOS performance: {oos_sharpe:.3f}",
                            recommendations=[] if performance_maintained else [f"{oos_period}日間のアウトオブサンプルパフォーマンスが低下しています"]
                        ))
        
        except Exception as e:
            self.logger.error(f"Error in out-of-sample validation: {e}")
            tests.append(self._create_failed_test("out_of_sample_validation_error", "robustness", str(e)))
        
        return tests
    
    def _calculate_portfolio_returns(
        self, 
        weights: Dict[str, float], 
        context: OptimizationContext
    ) -> pd.Series:
        """ポートフォリオリターンを計算"""
        return self._calculate_portfolio_returns_from_data(weights, context.strategy_returns)
    
    def _calculate_portfolio_returns_from_data(
        self, 
        weights: Dict[str, float], 
        strategy_returns: pd.DataFrame
    ) -> pd.Series:
        """データからポートフォリオリターンを計算"""
        
        portfolio_returns = pd.Series(0.0, index=strategy_returns.index)
        
        for strategy, weight in weights.items():
            if strategy in strategy_returns.columns:
                portfolio_returns += strategy_returns[strategy] * weight
        
        return portfolio_returns
    
    def _create_failed_test(self, test_name: str, category: str, error_msg: str) -> ValidationTest:
        """失敗したテストを作成"""
        
        return ValidationTest(
            test_name=test_name,
            test_category=category,
            test_result=False,
            test_score=0.0,
            error_message=error_msg,
            recommendations=[f"{test_name}で問題が発生しました"]
        )
    
    def _calculate_category_scores(self, tests: List[ValidationTest]) -> Dict[str, float]:
        """カテゴリ別スコアを計算"""
        
        categories = ['statistical', 'constraint', 'performance', 'robustness']
        category_scores = {}
        
        for category in categories:
            category_tests = [t for t in tests if t.test_category == category]
            
            if category_tests:
                category_scores[category] = np.mean([t.test_score for t in category_tests])
            else:
                category_scores[category] = 0.0
        
        return category_scores
    
    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """総合スコアを計算"""
        
        weights = self.config.get('test_weights', {
            'statistical': 0.2,
            'constraint': 0.3,
            'performance': 0.3,
            'robustness': 0.2
        })
        
        overall_score = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = weights.get(category, 0.25)
            overall_score += score * weight
            total_weight += weight
        
        return overall_score / max(total_weight, 1.0)
    
    def _identify_critical_issues(
        self, 
        tests: List[ValidationTest], 
        result: RiskAdjustedOptimizationResult
    ) -> Tuple[List[str], List[str]]:
        """重要な問題と警告を識別"""
        
        critical_failures = []
        warnings = []
        
        try:
            # 制約違反の重要度チェック
            constraint_violations = len(result.constraint_result.violations)
            max_violations = self.config['validation_thresholds']['critical_failure_thresholds']['max_constraint_violations']
            
            if constraint_violations > max_violations:
                critical_failures.append(f"過多な制約違反: {constraint_violations} > {max_violations}")
            elif constraint_violations > 0:
                warnings.append(f"制約違反あり: {constraint_violations}")
            
            # テスト失敗の重要度チェック
            failed_tests = [t for t in tests if not t.test_result]
            
            for test in failed_tests:
                if test.test_category == 'constraint' or test.test_score < 0.3:
                    critical_failures.append(f"{test.test_name}: {test.error_message}")
                elif test.test_score < 0.6:
                    warnings.append(f"{test.test_name}: {test.error_message}")
            
            # パフォーマンスの重要度チェック
            sharpe_ratio = result.performance_report.risk_adjusted_metrics.get('sharpe_ratio', 0)
            min_sharpe = self.config['validation_thresholds']['critical_failure_thresholds']['min_sharpe_ratio']
            
            if sharpe_ratio < min_sharpe:
                critical_failures.append(f"シャープレシオが低すぎます: {sharpe_ratio:.3f} < {min_sharpe}")
            elif sharpe_ratio < 0.5:
                warnings.append(f"シャープレシオが低めです: {sharpe_ratio:.3f}")
        
        except Exception as e:
            self.logger.error(f"Error identifying critical issues: {e}")
            critical_failures.append("重要問題の識別中にエラーが発生しました")
        
        return critical_failures, warnings
    
    def _determine_validation_success(
        self, 
        overall_score: float, 
        category_scores: Dict[str, float], 
        critical_failures: List[str]
    ) -> bool:
        """検証成功を判定"""
        
        # 重要な失敗がある場合は失敗
        if critical_failures:
            return False
        
        # 全体スコアのチェック
        min_overall = self.config['validation_thresholds']['minimum_overall_score']
        if overall_score < min_overall:
            return False
        
        # カテゴリ別スコアのチェック
        min_category_scores = self.config['validation_thresholds']['minimum_category_scores']
        
        for category, min_score in min_category_scores.items():
            if category_scores.get(category, 0) < min_score:
                return False
        
        return True
    
    def _generate_validation_summary(
        self, 
        tests: List[ValidationTest], 
        overall_score: float, 
        category_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """検証サマリーを生成"""
        
        passed_tests = [t for t in tests if t.test_result]
        failed_tests = [t for t in tests if not t.test_result]
        
        return {
            'total_tests': len(tests),
            'passed_tests': len(passed_tests),
            'failed_tests': len(failed_tests),
            'pass_rate': len(passed_tests) / max(len(tests), 1),
            'overall_score': overall_score,
            'category_scores': category_scores,
            'highest_scoring_category': max(category_scores.items(), key=lambda x: x[1]),
            'lowest_scoring_category': min(category_scores.items(), key=lambda x: x[1]),
            'test_categories': list(set(t.test_category for t in tests))
        }
    
    def _generate_improvement_suggestions(
        self, 
        tests: List[ValidationTest], 
        critical_failures: List[str], 
        warnings: List[str]
    ) -> List[str]:
        """改善提案を生成"""
        
        suggestions = []
        
        # 重要な失敗からの提案
        if critical_failures:
            suggestions.append("重要な問題の解決を最優先で実施してください:")
            for failure in critical_failures[:3]:
                suggestions.append(f"  - {failure}")
        
        # テスト結果からの提案
        failed_tests = [t for t in tests if not t.test_result and t.recommendations]
        
        category_suggestions = {}
        for test in failed_tests:
            if test.test_category not in category_suggestions:
                category_suggestions[test.test_category] = []
            category_suggestions[test.test_category].extend(test.recommendations)
        
        for category, recs in category_suggestions.items():
            unique_recs = list(dict.fromkeys(recs))  # 重複除去
            if unique_recs:
                suggestions.append(f"{category.title()}カテゴリの改善点:")
                for rec in unique_recs[:2]:  # 上位2件
                    suggestions.append(f"  - {rec}")
        
        # 警告からの提案
        if warnings and len(suggestions) < 8:
            suggestions.append("注意事項:")
            for warning in warnings[:2]:
                suggestions.append(f"  - {warning}")
        
        return suggestions[:10]  # 最大10件
    
    def _create_error_validation_report(self, error_message: str) -> ValidationReport:
        """エラー時の検証レポートを作成"""
        
        return ValidationReport(
            validation_success=False,
            overall_score=0.0,
            category_scores={},
            individual_tests=[],
            critical_failures=[f"検証実行エラー: {error_message}"],
            warnings=[],
            validation_summary={'error': error_message},
            improvement_suggestions=["検証システムの設定を確認し、再実行してください"]
        )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """検証サマリーを取得"""
        
        if not self.validation_history:
            return {'total_validations': 0}
        
        successful_validations = [v for v in self.validation_history if v.validation_success]
        
        return {
            'total_validations': len(self.validation_history),
            'success_rate': len(successful_validations) / len(self.validation_history),
            'average_overall_score': np.mean([v.overall_score for v in self.validation_history]),
            'average_category_scores': {
                category: np.mean([v.category_scores.get(category, 0) for v in self.validation_history])
                for category in ['statistical', 'constraint', 'performance', 'robustness']
            }
        }


# テスト用のメイン関数
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    logger.info("Testing Optimization Validator...")
    
    # テストデータとダミー結果の作成（実際の最適化結果が必要）
    from analysis.risk_adjusted_optimization.risk_return_optimizer import RiskAdjustedOptimizationEngine
    
    # テストデータの生成
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    strategy_returns = pd.DataFrame({
        'strategy1': np.random.normal(0.001, 0.02, len(dates)),
        'strategy2': np.random.normal(0.0015, 0.025, len(dates)),
        'strategy3': np.random.normal(0.0008, 0.018, len(dates))
    }, index=dates)
    
    current_weights = {
        'strategy1': 0.4,
        'strategy2': 0.3,
        'strategy3': 0.3
    }
    
    # 最適化コンテキスト
    from analysis.risk_adjusted_optimization.risk_return_optimizer import OptimizationContext
    context = OptimizationContext(
        strategy_returns=strategy_returns,
        current_weights=current_weights,
        market_volatility=0.20,
        trend_strength=0.05,
        market_regime="normal"
    )
    
    # 最適化実行（テスト用）
    engine = RiskAdjustedOptimizationEngine()
    optimization_result = engine.optimize_portfolio_allocation(context)
    
    # 検証システムのテスト
    validator = OptimizationValidator()
    
    validation_report = validator.validate_optimization_result(
        optimization_result, 
        context
    )
    
    logger.info("Validation Results:")
    logger.info(f"Validation Success: {validation_report.validation_success}")
    logger.info(f"Overall Score: {validation_report.overall_score:.3f}")
    logger.info(f"Category Scores: {validation_report.category_scores}")
    logger.info(f"Critical Failures: {len(validation_report.critical_failures)}")
    logger.info(f"Warnings: {len(validation_report.warnings)}")
    logger.info(f"Total Tests: {len(validation_report.individual_tests)}")
    logger.info(f"Improvement Suggestions: {len(validation_report.improvement_suggestions)}")
    
    # サマリーの表示
    summary = validator.get_validation_summary()
    logger.info(f"Validator Summary: {summary}")
    
    logger.info("Optimization Validator test completed successfully!")
