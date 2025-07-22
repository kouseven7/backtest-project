"""
VaRバックテスティングエンジン

VaR予測精度の検証
モデル性能の評価と改善
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .advanced_var_engine import AdvancedVaREngine, VaRCalculationConfig, VaRResult
from .hybrid_var_calculator import HybridVaRCalculator

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """バックテスト結果"""
    test_period_start: datetime
    test_period_end: datetime
    total_observations: int
    
    # VaR違反統計
    var_95_violations: int
    var_99_violations: int
    var_95_violation_rate: float
    var_99_violation_rate: float
    
    # 期待違反率
    expected_var_95_violations: float = 0.05
    expected_var_99_violations: float = 0.01
    
    # 統計的検定結果
    kupiec_test_95: Dict[str, float] = None
    kupiec_test_99: Dict[str, float] = None
    christoffersen_test_95: Dict[str, float] = None
    christoffersen_test_99: Dict[str, float] = None
    
    # 損失関数
    average_violation_magnitude: float = 0.0
    max_violation_magnitude: float = 0.0
    
    # モデル性能指標
    model_accuracy_score: float = 0.0
    calibration_quality: str = "unknown"
    
    # 推奨事項
    recommendations: List[str] = None

@dataclass
class BacktestConfig:
    """バックテスト設定"""
    # 期間設定
    lookback_window: int = 252  # 1年
    rolling_window: int = 60    # 2ヶ月のローリングウィンドウ
    min_observations: int = 30  # 最小観測数
    
    # 検定設定
    confidence_level: float = 0.05  # 統計検定の信頼水準
    enable_kupiec_test: bool = True
    enable_christoffersen_test: bool = True
    
    # 性能評価設定
    enable_model_comparison: bool = True
    enable_regime_analysis: bool = True
    
    # 出力設定
    save_detailed_results: bool = True
    generate_plots: bool = False  # プロット生成（必要に応じて）

class VaRBacktestingEngine:
    """VaRバックテスティングエンジン"""
    
    def __init__(self, 
                 var_engine: AdvancedVaREngine,
                 hybrid_calculator: Optional[HybridVaRCalculator] = None,
                 config: Optional[BacktestConfig] = None):
        
        self.var_engine = var_engine
        self.hybrid_calculator = hybrid_calculator
        self.config = config or BacktestConfig()
        
        self.logger = self._setup_logger()
        self.backtest_history: List[BacktestResult] = []
        
        self.logger.info("VaR Backtesting Engine initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.VaRBacktestingEngine")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def run_comprehensive_backtest(self, 
                                  historical_data: pd.DataFrame,
                                  weights_history: Dict[datetime, Dict[str, float]],
                                  test_start_date: Optional[datetime] = None,
                                  test_end_date: Optional[datetime] = None) -> BacktestResult:
        """包括的バックテスト実行"""
        try:
            self.logger.info("Starting comprehensive VaR backtest")
            
            # テスト期間の設定
            if test_start_date is None:
                test_start_date = historical_data.index[self.config.lookback_window]
            if test_end_date is None:
                test_end_date = historical_data.index[-1]
            
            # バックテストデータの準備
            backtest_data = self._prepare_backtest_data(
                historical_data, weights_history, test_start_date, test_end_date
            )
            
            if backtest_data.empty:
                raise ValueError("No valid backtest data available")
            
            # VaR予測とリターンの計算
            predictions = self._calculate_var_predictions(
                historical_data, weights_history, backtest_data.index
            )
            
            actual_returns = self._calculate_actual_returns(
                historical_data, weights_history, backtest_data.index
            )
            
            # バックテスト実行
            result = self._execute_backtest_analysis(
                predictions, actual_returns, test_start_date, test_end_date
            )
            
            # 結果を履歴に保存
            self.backtest_history.append(result)
            
            self.logger.info(f"Backtest completed: {result.total_observations} observations")
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive backtest error: {e}")
            raise
    
    def _prepare_backtest_data(self, 
                              historical_data: pd.DataFrame,
                              weights_history: Dict[datetime, Dict[str, float]],
                              start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
        """バックテストデータの準備"""
        try:
            # 期間フィルタリング
            mask = (historical_data.index >= start_date) & (historical_data.index <= end_date)
            test_data = historical_data.loc[mask].copy()
            
            # 重み履歴の整合性チェック
            valid_dates = []
            for date in test_data.index:
                # 最も近い重み日付を探す
                available_dates = [d for d in weights_history.keys() if d <= date]
                if available_dates:
                    valid_dates.append(date)
            
            # 有効な日付のみを保持
            if valid_dates:
                test_data = test_data.loc[valid_dates]
            
            self.logger.info(f"Prepared {len(test_data)} observations for backtesting")
            return test_data
            
        except Exception as e:
            self.logger.error(f"Backtest data preparation error: {e}")
            return pd.DataFrame()
    
    def _calculate_var_predictions(self,
                                  historical_data: pd.DataFrame,
                                  weights_history: Dict[datetime, Dict[str, float]],
                                  test_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """VaR予測の計算"""
        try:
            predictions = []
            
            for date in test_dates:
                try:
                    # 当該日付までのデータを使用
                    train_data = historical_data.loc[historical_data.index < date].copy()
                    
                    # 最新のlookback_window分のデータを使用
                    if len(train_data) > self.config.lookback_window:
                        train_data = train_data.tail(self.config.lookback_window)
                    
                    if len(train_data) < self.config.min_observations:
                        continue
                    
                    # 対応する重みを取得
                    available_weight_dates = [d for d in weights_history.keys() if d <= date]
                    if not available_weight_dates:
                        continue
                    
                    weight_date = max(available_weight_dates)
                    weights = weights_history[weight_date]
                    
                    # VaR計算
                    if self.hybrid_calculator:
                        var_result = self.hybrid_calculator.calculate_hybrid_var(train_data, weights)
                    else:
                        var_result = self.var_engine.calculate_comprehensive_var(train_data, weights)
                    
                    predictions.append({
                        'date': date,
                        'var_95': var_result.get_var_95(),
                        'var_99': var_result.get_var_99(),
                        'calculation_method': var_result.calculation_method,
                        'market_regime': var_result.market_regime
                    })
                    
                except Exception as e:
                    self.logger.warning(f"VaR prediction error for {date}: {e}")
                    continue
            
            predictions_df = pd.DataFrame(predictions)
            predictions_df.set_index('date', inplace=True)
            
            self.logger.info(f"Calculated {len(predictions_df)} VaR predictions")
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"VaR predictions calculation error: {e}")
            return pd.DataFrame()
    
    def _calculate_actual_returns(self,
                                 historical_data: pd.DataFrame,
                                 weights_history: Dict[datetime, Dict[str, float]],
                                 test_dates: pd.DatetimeIndex) -> pd.Series:
        """実際のポートフォリオリターン計算"""
        try:
            actual_returns = []
            
            for date in test_dates:
                try:
                    # 対応する重みを取得
                    available_weight_dates = [d for d in weights_history.keys() if d <= date]
                    if not available_weight_dates:
                        continue
                    
                    weight_date = max(available_weight_dates)
                    weights = weights_history[weight_date]
                    
                    # 当日のリターンを計算
                    if date in historical_data.index:
                        daily_returns = historical_data.loc[date]
                        
                        # ポートフォリオリターンの計算
                        portfolio_return = 0.0
                        total_weight = 0.0
                        
                        for symbol, weight in weights.items():
                            if symbol in daily_returns and not pd.isna(daily_returns[symbol]):
                                portfolio_return += daily_returns[symbol] * weight
                                total_weight += weight
                        
                        if total_weight > 0:
                            portfolio_return = portfolio_return / total_weight
                            actual_returns.append((date, portfolio_return))
                    
                except Exception as e:
                    self.logger.warning(f"Actual return calculation error for {date}: {e}")
                    continue
            
            if actual_returns:
                dates, returns = zip(*actual_returns)
                actual_returns_series = pd.Series(returns, index=dates)
            else:
                actual_returns_series = pd.Series(dtype=float)
            
            self.logger.info(f"Calculated {len(actual_returns_series)} actual returns")
            return actual_returns_series
            
        except Exception as e:
            self.logger.error(f"Actual returns calculation error: {e}")
            return pd.Series(dtype=float)
    
    def _execute_backtest_analysis(self,
                                  predictions: pd.DataFrame,
                                  actual_returns: pd.Series,
                                  start_date: datetime,
                                  end_date: datetime) -> BacktestResult:
        """バックテスト分析実行"""
        try:
            # データの整合性確保
            common_dates = predictions.index.intersection(actual_returns.index)
            if len(common_dates) == 0:
                raise ValueError("No common dates between predictions and actual returns")
            
            pred_aligned = predictions.loc[common_dates]
            returns_aligned = actual_returns.loc[common_dates]
            
            # VaR違反の計算
            var_95_violations = self._calculate_violations(returns_aligned, -pred_aligned['var_95'])
            var_99_violations = self._calculate_violations(returns_aligned, -pred_aligned['var_99'])
            
            total_obs = len(common_dates)
            var_95_violation_rate = len(var_95_violations) / total_obs
            var_99_violation_rate = len(var_99_violations) / total_obs
            
            # 統計検定
            kupiec_95 = self._kupiec_test(len(var_95_violations), total_obs, 0.05)
            kupiec_99 = self._kupiec_test(len(var_99_violations), total_obs, 0.01)
            
            christoffersen_95 = self._christoffersen_test(
                returns_aligned, -pred_aligned['var_95'], 0.05
            )
            christoffersen_99 = self._christoffersen_test(
                returns_aligned, -pred_aligned['var_99'], 0.01
            )
            
            # 違反の大きさ分析
            violation_magnitudes_95 = self._calculate_violation_magnitudes(
                returns_aligned, -pred_aligned['var_95']
            )
            violation_magnitudes_99 = self._calculate_violation_magnitudes(
                returns_aligned, -pred_aligned['var_99']
            )
            
            avg_violation_magnitude = np.mean(list(violation_magnitudes_95.values()) + 
                                            list(violation_magnitudes_99.values())) if \
                                    (violation_magnitudes_95 or violation_magnitudes_99) else 0.0
            
            max_violation_magnitude = max(
                max(violation_magnitudes_95.values()) if violation_magnitudes_95 else 0,
                max(violation_magnitudes_99.values()) if violation_magnitudes_99 else 0
            )
            
            # モデル精度スコア
            accuracy_score = self._calculate_model_accuracy(
                var_95_violation_rate, var_99_violation_rate, kupiec_95, kupiec_99
            )
            
            # キャリブレーション品質
            calibration_quality = self._assess_calibration_quality(
                var_95_violation_rate, var_99_violation_rate
            )
            
            # 推奨事項生成
            recommendations = self._generate_backtest_recommendations(
                var_95_violation_rate, var_99_violation_rate, kupiec_95, kupiec_99,
                christoffersen_95, christoffersen_99
            )
            
            # 結果の構築
            result = BacktestResult(
                test_period_start=start_date,
                test_period_end=end_date,
                total_observations=total_obs,
                var_95_violations=len(var_95_violations),
                var_99_violations=len(var_99_violations),
                var_95_violation_rate=var_95_violation_rate,
                var_99_violation_rate=var_99_violation_rate,
                kupiec_test_95=kupiec_95,
                kupiec_test_99=kupiec_99,
                christoffersen_test_95=christoffersen_95,
                christoffersen_test_99=christoffersen_99,
                average_violation_magnitude=avg_violation_magnitude,
                max_violation_magnitude=max_violation_magnitude,
                model_accuracy_score=accuracy_score,
                calibration_quality=calibration_quality,
                recommendations=recommendations
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest analysis execution error: {e}")
            raise
    
    def _calculate_violations(self, 
                            actual_returns: pd.Series, 
                            var_thresholds: pd.Series) -> Dict[datetime, float]:
        """VaR違反の計算"""
        try:
            violations = {}
            
            for date in actual_returns.index:
                if date in var_thresholds.index:
                    actual_return = actual_returns[date]
                    var_threshold = var_thresholds[date]
                    
                    # 損失が VaR を超える場合（actual_return < -var_threshold）
                    if actual_return < var_threshold:
                        violations[date] = actual_return - var_threshold
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Violation calculation error: {e}")
            return {}
    
    def _calculate_violation_magnitudes(self,
                                      actual_returns: pd.Series,
                                      var_thresholds: pd.Series) -> Dict[datetime, float]:
        """違反の大きさ計算"""
        try:
            violations = self._calculate_violations(actual_returns, var_thresholds)
            magnitudes = {date: abs(violation) for date, violation in violations.items()}
            return magnitudes
            
        except Exception as e:
            self.logger.error(f"Violation magnitude calculation error: {e}")
            return {}
    
    def _kupiec_test(self, 
                    num_violations: int, 
                    total_observations: int, 
                    expected_rate: float) -> Dict[str, float]:
        """Kupiec比率検定"""
        try:
            if total_observations == 0:
                return {'statistic': 0, 'p_value': 1, 'critical_value': 0, 'reject_null': False}
            
            observed_rate = num_violations / total_observations
            
            # 尤度比統計量
            if observed_rate == 0:
                lr_statistic = -2 * total_observations * np.log(1 - expected_rate)
            elif observed_rate == 1:
                lr_statistic = -2 * total_observations * np.log(expected_rate)
            else:
                lr_statistic = -2 * (
                    total_observations * np.log(1 - expected_rate) +
                    num_violations * np.log(expected_rate / observed_rate) +
                    (total_observations - num_violations) * np.log((1 - expected_rate) / (1 - observed_rate))
                )
            
            # p値の計算（カイ二乗分布）
            p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)
            
            # 臨界値（5%水準）
            critical_value = stats.chi2.ppf(0.95, df=1)
            
            # 帰無仮説の棄却判定
            reject_null = lr_statistic > critical_value
            
            return {
                'statistic': lr_statistic,
                'p_value': p_value,
                'critical_value': critical_value,
                'reject_null': reject_null,
                'observed_rate': observed_rate,
                'expected_rate': expected_rate
            }
            
        except Exception as e:
            self.logger.error(f"Kupiec test error: {e}")
            return {'error': str(e)}
    
    def _christoffersen_test(self,
                           actual_returns: pd.Series,
                           var_thresholds: pd.Series,
                           expected_rate: float) -> Dict[str, float]:
        """Christoffersen条件付きカバレッジ検定"""
        try:
            # 違反系列の生成
            violations = []
            for date in actual_returns.index:
                if date in var_thresholds.index:
                    is_violation = actual_returns[date] < var_thresholds[date]
                    violations.append(1 if is_violation else 0)
            
            if len(violations) < 2:
                return {'error': 'Insufficient data for Christoffersen test'}
            
            violations = np.array(violations)
            
            # 遷移確率の計算
            n00 = n01 = n10 = n11 = 0
            
            for i in range(len(violations) - 1):
                if violations[i] == 0 and violations[i + 1] == 0:
                    n00 += 1
                elif violations[i] == 0 and violations[i + 1] == 1:
                    n01 += 1
                elif violations[i] == 1 and violations[i + 1] == 0:
                    n10 += 1
                elif violations[i] == 1 and violations[i + 1] == 1:
                    n11 += 1
            
            # 無条件確率
            n_violations = np.sum(violations)
            n_total = len(violations)
            pi = n_violations / n_total
            
            # 条件付き確率
            pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
            pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
            
            # 尤度比統計量
            if pi01 > 0 and pi11 > 0 and pi > 0 and (1 - pi) > 0:
                lr_uc = -2 * np.log((1 - expected_rate)**(n_total - n_violations) * expected_rate**n_violations)
                lr_cc = -2 * np.log(
                    ((1 - pi)**(n_total - n_violations) * pi**n_violations) /
                    ((1 - pi01)**n00 * pi01**n01 * (1 - pi11)**n10 * pi11**n11)
                )
                lr_ind = lr_cc - lr_uc
            else:
                lr_cc = lr_ind = 0
            
            # p値の計算
            p_value_cc = 1 - stats.chi2.cdf(lr_cc, df=2) if lr_cc > 0 else 1
            p_value_ind = 1 - stats.chi2.cdf(lr_ind, df=1) if lr_ind > 0 else 1
            
            # 臨界値
            critical_value_cc = stats.chi2.ppf(0.95, df=2)
            critical_value_ind = stats.chi2.ppf(0.95, df=1)
            
            return {
                'lr_cc': lr_cc,
                'lr_ind': lr_ind,
                'p_value_cc': p_value_cc,
                'p_value_ind': p_value_ind,
                'critical_value_cc': critical_value_cc,
                'critical_value_ind': critical_value_ind,
                'reject_cc': lr_cc > critical_value_cc,
                'reject_independence': lr_ind > critical_value_ind,
                'pi01': pi01,
                'pi11': pi11
            }
            
        except Exception as e:
            self.logger.error(f"Christoffersen test error: {e}")
            return {'error': str(e)}
    
    def _calculate_model_accuracy(self,
                                var_95_rate: float,
                                var_99_rate: float,
                                kupiec_95: Dict[str, float],
                                kupiec_99: Dict[str, float]) -> float:
        """モデル精度スコアの計算"""
        try:
            accuracy_score = 0.0
            
            # 違反率の精度（期待値に近いほど高得点）
            var_95_accuracy = 1 - abs(var_95_rate - 0.05) / 0.05
            var_99_accuracy = 1 - abs(var_99_rate - 0.01) / 0.01
            
            accuracy_score += 0.4 * max(0, var_95_accuracy)
            accuracy_score += 0.4 * max(0, var_99_accuracy)
            
            # 統計検定の結果（帰無仮説を棄却しないほど高得点）
            if 'reject_null' in kupiec_95:
                accuracy_score += 0.1 * (0 if kupiec_95['reject_null'] else 1)
            
            if 'reject_null' in kupiec_99:
                accuracy_score += 0.1 * (0 if kupiec_99['reject_null'] else 1)
            
            return min(1.0, max(0.0, accuracy_score))
            
        except Exception as e:
            self.logger.error(f"Model accuracy calculation error: {e}")
            return 0.0
    
    def _assess_calibration_quality(self,
                                  var_95_rate: float,
                                  var_99_rate: float) -> str:
        """キャリブレーション品質の評価"""
        try:
            # 期待値からの偏差を評価
            var_95_deviation = abs(var_95_rate - 0.05) / 0.05
            var_99_deviation = abs(var_99_rate - 0.01) / 0.01
            
            max_deviation = max(var_95_deviation, var_99_deviation)
            
            if max_deviation <= 0.2:  # 20%以内の偏差
                return "excellent"
            elif max_deviation <= 0.5:  # 50%以内の偏差
                return "good"
            elif max_deviation <= 1.0:  # 100%以内の偏差
                return "acceptable"
            else:
                return "poor"
                
        except Exception as e:
            self.logger.error(f"Calibration quality assessment error: {e}")
            return "unknown"
    
    def _generate_backtest_recommendations(self,
                                         var_95_rate: float,
                                         var_99_rate: float,
                                         kupiec_95: Dict[str, float],
                                         kupiec_99: Dict[str, float],
                                         christoffersen_95: Dict[str, float],
                                         christoffersen_99: Dict[str, float]) -> List[str]:
        """バックテスト推奨事項の生成"""
        try:
            recommendations = []
            
            # 違反率に基づく推奨事項
            if var_95_rate > 0.08:  # 8%以上
                recommendations.append("var_95_overestimating_risk_consider_less_conservative_approach")
            elif var_95_rate < 0.02:  # 2%以下
                recommendations.append("var_95_underestimating_risk_increase_risk_sensitivity")
            
            if var_99_rate > 0.02:  # 2%以上
                recommendations.append("var_99_overestimating_risk_adjust_extreme_tail_modeling")
            elif var_99_rate < 0.005:  # 0.5%以下
                recommendations.append("var_99_underestimating_risk_enhance_tail_risk_modeling")
            
            # Kupiec検定に基づく推奨事項
            if kupiec_95.get('reject_null', False):
                recommendations.append("kupiec_test_95_failed_review_var_model_calibration")
            
            if kupiec_99.get('reject_null', False):
                recommendations.append("kupiec_test_99_failed_improve_tail_risk_estimation")
            
            # Christoffersen検定に基づく推奨事項
            if christoffersen_95.get('reject_cc', False):
                recommendations.append("christoffersen_test_95_failed_address_conditional_coverage")
            
            if christoffersen_95.get('reject_independence', False):
                recommendations.append("var_violations_clustered_improve_volatility_modeling")
            
            # 全般的な推奨事項
            if len(recommendations) == 0:
                recommendations.append("model_performance_acceptable_continue_monitoring")
            elif len(recommendations) >= 3:
                recommendations.append("multiple_issues_detected_comprehensive_model_review_recommended")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Backtest recommendations generation error: {e}")
            return ["error_generating_recommendations"]
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """バックテストサマリー取得"""
        try:
            if not self.backtest_history:
                return {"message": "No backtest results available"}
            
            latest_result = self.backtest_history[-1]
            
            summary = {
                "latest_backtest": {
                    "period": f"{latest_result.test_period_start.date()} to {latest_result.test_period_end.date()}",
                    "total_observations": latest_result.total_observations,
                    "var_95_violation_rate": f"{latest_result.var_95_violation_rate:.2%}",
                    "var_99_violation_rate": f"{latest_result.var_99_violation_rate:.2%}",
                    "model_accuracy_score": f"{latest_result.model_accuracy_score:.2f}",
                    "calibration_quality": latest_result.calibration_quality,
                    "recommendations_count": len(latest_result.recommendations or [])
                },
                "historical_performance": {
                    "total_backtests": len(self.backtest_history),
                    "average_accuracy": np.mean([r.model_accuracy_score for r in self.backtest_history]),
                    "best_accuracy": max([r.model_accuracy_score for r in self.backtest_history]),
                    "worst_accuracy": min([r.model_accuracy_score for r in self.backtest_history])
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Backtest summary error: {e}")
            return {"error": str(e)}
