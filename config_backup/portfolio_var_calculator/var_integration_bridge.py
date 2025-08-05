"""
VaR統合ブリッジ

既存のポートフォリオリスク管理システムとの統合
portfolio_risk_manager.pyとの連携
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .advanced_var_engine import AdvancedVaREngine, VaRCalculationConfig, VaRResult
from .hybrid_var_calculator import HybridVaRCalculator
from .real_time_var_monitor import RealTimeVaRMonitor, MonitoringConfig

# 既存のポートフォリオリスク管理システムのインポート
try:
    from portfolio_risk_manager import PortfolioRiskManager, VaRCalculator
    PORTFOLIO_RISK_AVAILABLE = True
except ImportError:
    PORTFOLIO_RISK_AVAILABLE = False
    logging.warning("portfolio_risk_manager.py not available for integration")

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntegrationResult:
    """統合結果"""
    success: bool
    advanced_var_result: Optional[VaRResult] = None
    legacy_var_result: Optional[Dict[str, float]] = None
    comparison_metrics: Optional[Dict[str, float]] = None
    recommendations: List[str] = None
    error_message: Optional[str] = None

@dataclass
class BridgeConfig:
    """ブリッジ設定"""
    enable_legacy_comparison: bool = True
    enable_advanced_override: bool = False
    comparison_tolerance: float = 0.1  # 10%の差異まで許容
    prefer_advanced_when_available: bool = True
    
    # ログ設定
    log_comparisons: bool = True
    log_discrepancies: bool = True
    
    # 統合設定
    merge_calculation_methods: bool = True
    use_hybrid_recommendations: bool = True

class VaRIntegrationBridge:
    """VaR統合ブリッジ"""
    
    def __init__(self, 
                 advanced_engine: AdvancedVaREngine,
                 hybrid_calculator: Optional[HybridVaRCalculator] = None,
                 monitor: Optional[RealTimeVaRMonitor] = None,
                 bridge_config: Optional[BridgeConfig] = None):
        
        self.advanced_engine = advanced_engine
        self.hybrid_calculator = hybrid_calculator or HybridVaRCalculator(advanced_engine)
        self.monitor = monitor
        self.config = bridge_config or BridgeConfig()
        
        # レガシーシステムの初期化
        self.legacy_portfolio_risk: Optional[Any] = None
        self.legacy_var_calculator: Optional[Any] = None
        
        if PORTFOLIO_RISK_AVAILABLE:
            self._initialize_legacy_systems()
        
        self.logger = self._setup_logger()
        self.integration_history: List[IntegrationResult] = []
        
        self.logger.info("VaR Integration Bridge initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.VaRIntegrationBridge")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_legacy_systems(self) -> None:
        """レガシーシステムの初期化"""
        try:
            if PORTFOLIO_RISK_AVAILABLE:
                # PortfolioRiskManagerの初期化
                self.legacy_portfolio_risk = PortfolioRiskManager(
                    lookback_days=252,  # 1年のルックバック
                    var_confidence_levels=[0.95, 0.99]
                )
                
                # VaRCalculatorの初期化
                self.legacy_var_calculator = VaRCalculator(
                    confidence_levels=[0.95, 0.99]
                )
                
                self.logger.info("Legacy systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Legacy systems initialization failed: {e}")
            self.legacy_portfolio_risk = None
            self.legacy_var_calculator = None
    
    def calculate_integrated_var(self, 
                                returns_data: pd.DataFrame, 
                                weights: Dict[str, float],
                                use_hybrid: bool = True) -> IntegrationResult:
        """統合VaR計算"""
        try:
            result = IntegrationResult(
                success=False,
                recommendations=[]
            )
            
            # 高度VaR計算
            if use_hybrid and self.hybrid_calculator:
                advanced_result = self.hybrid_calculator.calculate_hybrid_var(returns_data, weights)
            else:
                advanced_result = self.advanced_engine.calculate_comprehensive_var(returns_data, weights)
            
            result.advanced_var_result = advanced_result
            
            # レガシーシステムとの比較（利用可能な場合）
            if self.config.enable_legacy_comparison and self.legacy_var_calculator:
                legacy_result = self._calculate_legacy_var(returns_data, weights)
                result.legacy_var_result = legacy_result
                
                # 比較メトリクスの計算
                if legacy_result:
                    comparison = self._compare_var_results(advanced_result, legacy_result)
                    result.comparison_metrics = comparison
            
            # 統合推奨事項の生成
            recommendations = self._generate_integrated_recommendations(
                advanced_result, 
                result.legacy_var_result, 
                result.comparison_metrics
            )
            result.recommendations = recommendations
            
            result.success = True
            
            # 履歴に記録
            self.integration_history.append(result)
            
            # ログ出力
            if self.config.log_comparisons:
                self._log_integration_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Integrated VaR calculation error: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    def _calculate_legacy_var(self, 
                             returns_data: pd.DataFrame, 
                             weights: Dict[str, float]) -> Optional[Dict[str, float]]:
        """レガシーVaR計算"""
        try:
            if not self.legacy_var_calculator:
                return None
            
            # ポートフォリオリターンの計算
            portfolio_returns = self._calculate_portfolio_returns(returns_data, weights)
            
            if portfolio_returns.empty:
                return None
            
            # レガシーVaR計算
            var_results = {}
            
            # VaR 95%
            var_95 = self.legacy_var_calculator.calculate_var(portfolio_returns, confidence_level=0.95)
            if var_95 is not None:
                var_results['var_95'] = float(var_95)
            
            # VaR 99%
            var_99 = self.legacy_var_calculator.calculate_var(portfolio_returns, confidence_level=0.99)
            if var_99 is not None:
                var_results['var_99'] = float(var_99)
            
            return var_results if var_results else None
            
        except Exception as e:
            self.logger.error(f"Legacy VaR calculation error: {e}")
            return None
    
    def _calculate_portfolio_returns(self, 
                                   returns_data: pd.DataFrame, 
                                   weights: Dict[str, float]) -> pd.Series:
        """ポートフォリオリターン計算"""
        try:
            # 共通銘柄の抽出
            common_symbols = set(returns_data.columns) & set(weights.keys())
            
            if not common_symbols:
                return pd.Series(dtype=float)
            
            # 重みの正規化
            total_weight = sum(weights[symbol] for symbol in common_symbols)
            if total_weight <= 0:
                return pd.Series(dtype=float)
            
            normalized_weights = {
                symbol: weights[symbol] / total_weight 
                for symbol in common_symbols
            }
            
            # ポートフォリオリターンの計算
            portfolio_returns = pd.Series(0.0, index=returns_data.index)
            
            for symbol in common_symbols:
                if symbol in returns_data.columns:
                    portfolio_returns += returns_data[symbol] * normalized_weights[symbol]
            
            return portfolio_returns
            
        except Exception as e:
            self.logger.error(f"Portfolio returns calculation error: {e}")
            return pd.Series(dtype=float)
    
    def _compare_var_results(self, 
                           advanced_result: VaRResult, 
                           legacy_result: Dict[str, float]) -> Dict[str, float]:
        """VaR結果の比較"""
        try:
            comparison = {}
            
            # VaR 95%比較
            advanced_var_95 = advanced_result.get_var_95()
            legacy_var_95 = legacy_result.get('var_95')
            
            if advanced_var_95 is not None and legacy_var_95 is not None:
                diff_95 = abs(advanced_var_95 - legacy_var_95)
                relative_diff_95 = diff_95 / max(abs(legacy_var_95), 1e-6)
                
                comparison['var_95_absolute_diff'] = diff_95
                comparison['var_95_relative_diff'] = relative_diff_95
                comparison['var_95_advanced'] = advanced_var_95
                comparison['var_95_legacy'] = legacy_var_95
            
            # VaR 99%比較
            advanced_var_99 = advanced_result.get_var_99()
            legacy_var_99 = legacy_result.get('var_99')
            
            if advanced_var_99 is not None and legacy_var_99 is not None:
                diff_99 = abs(advanced_var_99 - legacy_var_99)
                relative_diff_99 = diff_99 / max(abs(legacy_var_99), 1e-6)
                
                comparison['var_99_absolute_diff'] = diff_99
                comparison['var_99_relative_diff'] = relative_diff_99
                comparison['var_99_advanced'] = advanced_var_99
                comparison['var_99_legacy'] = legacy_var_99
            
            # 不一致の検出
            discrepancies = []
            
            if 'var_95_relative_diff' in comparison and \
               comparison['var_95_relative_diff'] > self.config.comparison_tolerance:
                discrepancies.append('var_95_significant_difference')
            
            if 'var_99_relative_diff' in comparison and \
               comparison['var_99_relative_diff'] > self.config.comparison_tolerance:
                discrepancies.append('var_99_significant_difference')
            
            comparison['discrepancies'] = discrepancies
            comparison['has_discrepancies'] = len(discrepancies) > 0
            
            # 不一致のログ出力
            if self.config.log_discrepancies and discrepancies:
                self._log_discrepancies(comparison)
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"VaR comparison error: {e}")
            return {}
    
    def _generate_integrated_recommendations(self,
                                          advanced_result: VaRResult,
                                          legacy_result: Optional[Dict[str, float]],
                                          comparison: Optional[Dict[str, float]]) -> List[str]:
        """統合推奨事項の生成"""
        try:
            recommendations = []
            
            # 高度システムの推奨事項
            if advanced_result.market_regime == 'high_volatility':
                recommendations.append('consider_risk_reduction_due_to_high_volatility')
            
            if advanced_result.diversification_benefit < 0.1:
                recommendations.append('improve_portfolio_diversification')
            
            # VaRレベルに基づく推奨事項
            var_95 = advanced_result.get_var_95()
            var_99 = advanced_result.get_var_99()
            
            if var_95 and var_95 > 0.05:  # 5%以上
                recommendations.append('var_95_exceeds_typical_threshold')
            
            if var_99 and var_99 > 0.08:  # 8%以上
                recommendations.append('var_99_exceeds_conservative_threshold')
            
            # 比較に基づく推奨事項
            if comparison and comparison.get('has_discrepancies', False):
                recommendations.append('investigate_calculation_method_differences')
                
                # 高度システムがより保守的な場合
                if 'var_95_advanced' in comparison and 'var_95_legacy' in comparison:
                    if comparison['var_95_advanced'] > comparison['var_95_legacy'] * 1.2:
                        recommendations.append('advanced_system_more_conservative_consider_risk_factors')
            
            # ハイブリッド計算の推奨事項
            if self.config.use_hybrid_recommendations and self.hybrid_calculator:
                hybrid_recommendations = self._get_hybrid_recommendations(advanced_result)
                recommendations.extend(hybrid_recommendations)
            
            return list(set(recommendations))  # 重複を除去
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return ['error_in_recommendation_generation']
    
    def _get_hybrid_recommendations(self, var_result: VaRResult) -> List[str]:
        """ハイブリッド計算からの推奨事項"""
        try:
            recommendations = []
            
            # 計算方法に基づく推奨事項
            if var_result.calculation_method == 'monte_carlo':
                recommendations.append('monte_carlo_used_consider_scenario_analysis')
            elif var_result.calculation_method == 'historical':
                recommendations.append('historical_method_used_monitor_regime_changes')
            elif var_result.calculation_method == 'parametric':
                recommendations.append('parametric_method_used_validate_distribution_assumptions')
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Hybrid recommendations error: {e}")
            return []
    
    def _log_integration_result(self, result: IntegrationResult) -> None:
        """統合結果のログ出力"""
        try:
            self.logger.info("=== VaR Integration Result ===")
            
            if result.advanced_var_result:
                var_95 = result.advanced_var_result.get_var_95()
                var_99 = result.advanced_var_result.get_var_99()
                self.logger.info(f"Advanced VaR - 95%: {var_95:.4f}, 99%: {var_99:.4f}")
                self.logger.info(f"Market Regime: {result.advanced_var_result.market_regime}")
                self.logger.info(f"Calculation Method: {result.advanced_var_result.calculation_method}")
            
            if result.legacy_var_result:
                legacy_95 = result.legacy_var_result.get('var_95')
                legacy_99 = result.legacy_var_result.get('var_99')
                self.logger.info(f"Legacy VaR - 95%: {legacy_95:.4f}, 99%: {legacy_99:.4f}")
            
            if result.comparison_metrics:
                if 'var_95_relative_diff' in result.comparison_metrics:
                    diff = result.comparison_metrics['var_95_relative_diff']
                    self.logger.info(f"VaR 95% Relative Difference: {diff:.2%}")
                
                if 'var_99_relative_diff' in result.comparison_metrics:
                    diff = result.comparison_metrics['var_99_relative_diff']
                    self.logger.info(f"VaR 99% Relative Difference: {diff:.2%}")
            
            if result.recommendations:
                self.logger.info(f"Recommendations: {', '.join(result.recommendations)}")
            
            self.logger.info("=== End Integration Result ===")
            
        except Exception as e:
            self.logger.error(f"Log integration result error: {e}")
    
    def _log_discrepancies(self, comparison: Dict[str, float]) -> None:
        """不一致のログ出力"""
        try:
            self.logger.warning("=== VaR Calculation Discrepancies Detected ===")
            
            for discrepancy in comparison.get('discrepancies', []):
                if discrepancy == 'var_95_significant_difference':
                    diff = comparison.get('var_95_relative_diff', 0)
                    self.logger.warning(f"VaR 95% significant difference: {diff:.2%}")
                    self.logger.warning(f"Advanced: {comparison.get('var_95_advanced', 0):.4f}, "
                                      f"Legacy: {comparison.get('var_95_legacy', 0):.4f}")
                
                elif discrepancy == 'var_99_significant_difference':
                    diff = comparison.get('var_99_relative_diff', 0)
                    self.logger.warning(f"VaR 99% significant difference: {diff:.2%}")
                    self.logger.warning(f"Advanced: {comparison.get('var_99_advanced', 0):.4f}, "
                                      f"Legacy: {comparison.get('var_99_legacy', 0):.4f}")
            
            self.logger.warning("=== End Discrepancies ===")
            
        except Exception as e:
            self.logger.error(f"Log discrepancies error: {e}")
    
    def get_integration_summary(self, hours: int = 24) -> Dict[str, Any]:
        """統合サマリー取得"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_integrations = [
                integration for integration in self.integration_history
                if integration.advanced_var_result and 
                   integration.advanced_var_result.timestamp >= cutoff_time
            ]
            
            summary = {
                'integration_period_hours': hours,
                'total_integrations': len(recent_integrations),
                'successful_integrations': len([i for i in recent_integrations if i.success]),
                'legacy_system_available': PORTFOLIO_RISK_AVAILABLE and self.legacy_var_calculator is not None
            }
            
            if recent_integrations:
                # 不一致統計
                discrepancy_count = len([
                    i for i in recent_integrations 
                    if i.comparison_metrics and i.comparison_metrics.get('has_discrepancies', False)
                ])
                
                summary['discrepancy_rate'] = discrepancy_count / len(recent_integrations)
                
                # VaR統計
                var_95_values = [
                    i.advanced_var_result.get_var_95() 
                    for i in recent_integrations 
                    if i.advanced_var_result and i.advanced_var_result.get_var_95() is not None
                ]
                
                if var_95_values:
                    summary['var_95_stats'] = {
                        'mean': sum(var_95_values) / len(var_95_values),
                        'max': max(var_95_values),
                        'min': min(var_95_values)
                    }
                
                # 最頻出推奨事項
                all_recommendations = []
                for integration in recent_integrations:
                    if integration.recommendations:
                        all_recommendations.extend(integration.recommendations)
                
                if all_recommendations:
                    recommendation_counts = {}
                    for rec in all_recommendations:
                        recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
                    
                    summary['top_recommendations'] = sorted(
                        recommendation_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Integration summary error: {e}")
            return {'error': str(e)}
    
    def create_unified_var_report(self, 
                                 returns_data: pd.DataFrame, 
                                 weights: Dict[str, float]) -> Dict[str, Any]:
        """統一VaRレポート作成"""
        try:
            # 統合VaR計算
            integration_result = self.calculate_integrated_var(returns_data, weights)
            
            if not integration_result.success:
                return {'error': integration_result.error_message}
            
            # 統一レポートの構築
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'portfolio_summary': {
                    'total_assets': len(weights),
                    'total_weight': sum(weights.values()),
                    'largest_position': max(weights.values()) if weights else 0,
                    'smallest_position': min(weights.values()) if weights else 0
                }
            }
            
            # 高度VaR結果
            if integration_result.advanced_var_result:
                advanced = integration_result.advanced_var_result
                report['advanced_var'] = {
                    'var_95': advanced.get_var_95(),
                    'var_99': advanced.get_var_99(),
                    'expected_shortfall_95': advanced.expected_shortfall.get(0.95),
                    'expected_shortfall_99': advanced.expected_shortfall.get(0.99),
                    'market_regime': advanced.market_regime,
                    'calculation_method': advanced.calculation_method,
                    'diversification_benefit': advanced.diversification_benefit,
                    'component_var': dict(advanced.component_var) if advanced.component_var else {}
                }
            
            # レガシーVaR結果
            if integration_result.legacy_var_result:
                report['legacy_var'] = integration_result.legacy_var_result
            
            # 比較メトリクス
            if integration_result.comparison_metrics:
                report['comparison'] = integration_result.comparison_metrics
            
            # 推奨事項
            report['recommendations'] = integration_result.recommendations or []
            
            # リアルタイム監視情報（利用可能な場合）
            if self.monitor:
                monitoring_status = self.monitor.get_monitoring_status()
                report['monitoring'] = {
                    'is_active': monitoring_status.get('is_monitoring', False),
                    'last_alert_summary': self.monitor.get_alert_summary(hours=1)
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Unified VaR report error: {e}")
            return {'error': str(e)}
    
    def setup_monitoring_integration(self, 
                                   data_provider: Callable[[], pd.DataFrame],
                                   weight_provider: Callable[[], Dict[str, float]]) -> bool:
        """監視システムとの統合設定"""
        try:
            if not self.monitor:
                self.logger.warning("Real-time monitor not available for integration")
                return False
            
            # ドローダウンコントローラーとの統合コールバック
            def integrated_callback(drawdown_signal: Dict[str, Any]) -> None:
                """統合コールバック"""
                try:
                    self.logger.info(f"Integrated drawdown signal received: {drawdown_signal['signal_type']}")
                    
                    # 追加の統合処理をここに実装
                    # 例：レガシーシステムへの通知、統合アラート生成等
                    
                except Exception as e:
                    self.logger.error(f"Integrated callback error: {e}")
            
            # 監視システムにコールバックを設定
            self.monitor.set_drawdown_controller_callback(integrated_callback)
            
            # 監視開始
            success = self.monitor.start_monitoring(data_provider, weight_provider)
            
            if success:
                self.logger.info("Monitoring integration setup completed successfully")
            else:
                self.logger.error("Failed to start integrated monitoring")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Monitoring integration setup error: {e}")
            return False
