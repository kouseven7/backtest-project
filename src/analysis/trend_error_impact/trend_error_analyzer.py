"""
Module: Trend Error Analyzer
File: trend_error_analyzer.py
Description: 
  5-1-2「トレンド判定エラーの影響分析」
  統合トレンド判定エラー分析エンジン

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 警告を抑制
warnings.filterwarnings('ignore')

# ロガーの設定
logger = logging.getLogger(__name__)

try:
    from .trend_error_detector import TrendErrorDetector, ErrorDetectionResult
    from .error_impact_calculator import ErrorImpactCalculator, ErrorImpactResult, BatchErrorImpactAnalyzer
    from .error_classification_engine import TrendErrorInstance, TrendErrorType, ErrorSeverity
except ImportError:
    from trend_error_detector import TrendErrorDetector, ErrorDetectionResult
    from error_impact_calculator import ErrorImpactCalculator, ErrorImpactResult, BatchErrorImpactAnalyzer
    from error_classification_engine import TrendErrorInstance, TrendErrorType, ErrorSeverity

@dataclass
class ComprehensiveAnalysisResult:
    """包括的分析結果"""
    analysis_timestamp: datetime
    analysis_period: Tuple[datetime, datetime]
    
    # 検出結果
    detection_result: ErrorDetectionResult
    
    # 影響分析結果
    impact_results: List[ErrorImpactResult]
    
    # 統合メトリクス
    total_impact_score: float
    average_error_severity: float
    risk_adjusted_total_impact: float
    
    # 推奨事項
    priority_recommendations: List[str]
    immediate_actions: List[str]
    long_term_improvements: List[str]
    
    # 統計サマリー
    analysis_summary: Dict[str, Any]

class TrendErrorAnalyzer:
    """統合トレンド判定エラー分析エンジン"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Parameters:
            config_path: 設定ファイルのパス
        """
        self.logger = logger
        self.config_path = config_path or self._get_default_config_path()
        self.analysis_config = self._load_analysis_config()
        
        # コンポーネントの初期化
        self.error_detector = TrendErrorDetector()
        self.impact_calculator = ErrorImpactCalculator()
        self.batch_analyzer = BatchErrorImpactAnalyzer(self.impact_calculator)
        
        # 分析パラメータ
        self.analysis_window = self.analysis_config.get("analysis_window", 30)
        self.impact_threshold = self.analysis_config.get("impact_threshold", 0.05)
        self.severity_weights = self.analysis_config.get("severity_weights", {
            "LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 1.0
        })
        
        # 既存システム統合
        self._initialize_system_integrations()
    
    def _get_default_config_path(self) -> str:
        """デフォルト設定ファイルパスを取得"""
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "trend_error_analysis", "analysis_config.json"
        )
    
    def _load_analysis_config(self) -> Dict[str, Any]:
        """分析設定を読み込み"""
        try:
            import json
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return self._get_default_analysis_config()
        except Exception as e:
            self.logger.error(f"Failed to load analysis configuration: {e}")
            return self._get_default_analysis_config()
    
    def _get_default_analysis_config(self) -> Dict[str, Any]:
        """デフォルト分析設定を取得"""
        return {
            "analysis_window": 30,
            "impact_threshold": 0.05,
            "severity_weights": {
                "LOW": 0.25,
                "MEDIUM": 0.5,
                "HIGH": 0.75,
                "CRITICAL": 1.0
            },
            "integration": {
                "drawdown_correlation": True,
                "strategy_switching_analysis": True,
                "portfolio_risk_integration": True
            },
            "reporting": {
                "detailed_reports": True,
                "visualization": True,
                "export_formats": ["json", "csv"]
            }
        }
    
    def _initialize_system_integrations(self):
        """既存システムとの統合を初期化"""
        try:
            # DrawdownControllerとの統合
            from config.drawdown_controller import DrawdownController
            self.drawdown_controller = DrawdownController
            self.logger.info("Drawdown controller integration initialized")
        except ImportError:
            self.logger.warning("Drawdown controller not available")
            self.drawdown_controller = None
        
        try:
            # StrategySwitchingAnalyzerとの統合
            from analysis.strategy_switching.strategy_switching_analyzer import StrategySwitchingAnalyzer
            self.strategy_switching_analyzer = StrategySwitchingAnalyzer
            self.logger.info("Strategy switching analyzer integration initialized")
        except ImportError:
            self.logger.warning("Strategy switching analyzer not available")
            self.strategy_switching_analyzer = None
        
        try:
            # PortfolioRiskManagerとの統合
            from config.portfolio_risk_manager import PortfolioRiskManager
            self.portfolio_risk_manager = PortfolioRiskManager
            self.logger.info("Portfolio risk manager integration initialized")
        except ImportError:
            self.logger.warning("Portfolio risk manager not available")
            self.portfolio_risk_manager = None
    
    def analyze_trend_errors(self, 
                           market_data: pd.DataFrame,
                           trend_predictions: pd.DataFrame,
                           portfolio_context: Optional[Dict[str, Any]] = None,
                           analysis_period: Optional[Tuple[datetime, datetime]] = None) -> ComprehensiveAnalysisResult:
        """
        包括的なトレンド判定エラー分析
        
        Parameters:
            market_data: 市場データ
            trend_predictions: トレンド予測結果
            portfolio_context: ポートフォリオコンテキスト
            analysis_period: 分析期間
        
        Returns:
            ComprehensiveAnalysisResult: 包括的分析結果
        """
        try:
            self.logger.info("Starting comprehensive trend error analysis")
            
            # デフォルト分析期間の設定
            if analysis_period is None:
                end_date = market_data.index[-1]
                start_date = end_date - timedelta(days=self.analysis_window)
                analysis_period = (start_date, end_date)
            
            # 1. エラー検出
            self.logger.info("Phase 1: Error Detection")
            detection_result = self.error_detector.detect_trend_errors(
                market_data, trend_predictions, analysis_period
            )
            
            # 2. 影響分析
            self.logger.info("Phase 2: Impact Analysis")
            impact_results = []
            if detection_result.error_instances:
                impact_results = self.batch_analyzer.analyze_error_batch(
                    detection_result.error_instances,
                    market_data,
                    portfolio_context
                )
            
            # 3. 統合メトリクスの計算
            self.logger.info("Phase 3: Integrated Metrics Calculation")
            integrated_metrics = self._calculate_integrated_metrics(
                detection_result, impact_results
            )
            
            # 4. 既存システムとの相関分析
            self.logger.info("Phase 4: System Correlation Analysis")
            system_correlations = self._analyze_system_correlations(
                detection_result, impact_results, market_data, portfolio_context
            )
            
            # 5. 推奨事項の生成
            self.logger.info("Phase 5: Recommendations Generation")
            recommendations = self._generate_comprehensive_recommendations(
                detection_result, impact_results, integrated_metrics, system_correlations
            )
            
            # 6. 統計サマリーの作成
            analysis_summary = self._create_analysis_summary(
                detection_result, impact_results, integrated_metrics, system_correlations
            )
            
            # 包括的結果の作成
            result = ComprehensiveAnalysisResult(
                analysis_timestamp=datetime.now(),
                analysis_period=analysis_period,
                detection_result=detection_result,
                impact_results=impact_results,
                total_impact_score=integrated_metrics['total_impact_score'],
                average_error_severity=integrated_metrics['average_error_severity'],
                risk_adjusted_total_impact=integrated_metrics['risk_adjusted_total_impact'],
                priority_recommendations=recommendations['priority'],
                immediate_actions=recommendations['immediate'],
                long_term_improvements=recommendations['long_term'],
                analysis_summary=analysis_summary
            )
            
            self.logger.info(f"Comprehensive analysis completed. Total impact score: {result.total_impact_score:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive trend error analysis failed: {e}")
            raise
    
    def _calculate_integrated_metrics(self, 
                                    detection_result: ErrorDetectionResult,
                                    impact_results: List[ErrorImpactResult]) -> Dict[str, float]:
        """統合メトリクスを計算"""
        
        if not impact_results:
            return {
                'total_impact_score': 0.0,
                'average_error_severity': 0.0,
                'risk_adjusted_total_impact': 0.0,
                'error_concentration_index': 0.0,
                'systemic_risk_factor': 0.0
            }
        
        # 基本的な統合メトリクス
        composite_scores = [r.impact_metrics.composite_score for r in impact_results]
        total_impact_score = sum(composite_scores)
        
        # 平均エラー深刻度の計算
        severity_scores = []
        for result in impact_results:
            severity = result.error_instance.severity
            weight = self.severity_weights.get(severity.value.upper(), 0.5)
            severity_scores.append(weight)
        
        average_error_severity = np.mean(severity_scores) if severity_scores else 0.0
        
        # リスク調整後総合影響
        risk_adjustments = [r.impact_metrics.risk_adjusted_impact for r in impact_results]
        risk_adjusted_total_impact = sum(risk_adjustments)
        
        # エラー集中度指数（エラーが特定の期間に集中しているかの指標）
        timestamps = [r.error_instance.timestamp for r in impact_results]
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
            error_concentration_index = np.std(time_diffs) / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
        else:
            error_concentration_index = 0.0
        
        # システミックリスク要因
        systemic_impacts = [r.impact_metrics.systemic_impact for r in impact_results]
        systemic_risk_factor = max(systemic_impacts) if systemic_impacts else 0.0
        
        return {
            'total_impact_score': total_impact_score,
            'average_error_severity': average_error_severity,
            'risk_adjusted_total_impact': risk_adjusted_total_impact,
            'error_concentration_index': error_concentration_index,
            'systemic_risk_factor': systemic_risk_factor
        }
    
    def _analyze_system_correlations(self, 
                                   detection_result: ErrorDetectionResult,
                                   impact_results: List[ErrorImpactResult],
                                   market_data: pd.DataFrame,
                                   portfolio_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """既存システムとの相関分析"""
        
        correlations = {}
        
        # ドローダウン相関の分析
        if self.drawdown_controller and self.analysis_config.get("integration", {}).get("drawdown_correlation", True):
            try:
                drawdown_correlation = self._analyze_drawdown_correlation(
                    impact_results, market_data
                )
                correlations['drawdown'] = drawdown_correlation
            except Exception as e:
                self.logger.warning(f"Drawdown correlation analysis failed: {e}")
                correlations['drawdown'] = {'correlation': 0.0, 'significance': 'low'}
        
        # 戦略切替との相関分析
        if self.strategy_switching_analyzer and self.analysis_config.get("integration", {}).get("strategy_switching_analysis", True):
            try:
                switching_correlation = self._analyze_strategy_switching_correlation(
                    detection_result, market_data
                )
                correlations['strategy_switching'] = switching_correlation
            except Exception as e:
                self.logger.warning(f"Strategy switching correlation analysis failed: {e}")
                correlations['strategy_switching'] = {'correlation': 0.0, 'events_overlap': 0}
        
        # ポートフォリオリスクとの統合
        if self.portfolio_risk_manager and self.analysis_config.get("integration", {}).get("portfolio_risk_integration", True):
            try:
                portfolio_integration = self._analyze_portfolio_risk_integration(
                    impact_results, portfolio_context or {}
                )
                correlations['portfolio_risk'] = portfolio_integration
            except Exception as e:
                self.logger.warning(f"Portfolio risk integration analysis failed: {e}")
                correlations['portfolio_risk'] = {'risk_amplification': 1.0, 'diversification_impact': 0.0}
        
        return correlations
    
    def _analyze_drawdown_correlation(self, 
                                    impact_results: List[ErrorImpactResult],
                                    market_data: pd.DataFrame) -> Dict[str, Any]:
        """ドローダウンとの相関を分析"""
        
        # 簡易的なドローダウン計算
        prices = market_data['Adj Close']
        cumulative_returns = (prices / prices.iloc[0])
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / running_max - 1) * 100
        
        # エラー発生時点でのドローダウンを確認
        error_drawdowns = []
        for result in impact_results:
            error_date = result.error_instance.timestamp
            closest_date = min(drawdown.index, key=lambda x: abs((x - error_date).days))
            error_drawdowns.append(drawdown.loc[closest_date])
        
        # 相関係数の計算
        impact_scores = [r.impact_metrics.composite_score for r in impact_results]
        
        if len(error_drawdowns) > 1 and len(impact_scores) > 1:
            correlation = np.corrcoef(error_drawdowns, impact_scores)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # 重要性の判定
        significance = 'high' if abs(correlation) > 0.7 else 'medium' if abs(correlation) > 0.4 else 'low'
        
        return {
            'correlation': correlation,
            'significance': significance,
            'average_drawdown_at_error': np.mean(error_drawdowns) if error_drawdowns else 0.0,
            'max_drawdown_at_error': min(error_drawdowns) if error_drawdowns else 0.0
        }
    
    def _analyze_strategy_switching_correlation(self, 
                                              detection_result: ErrorDetectionResult,
                                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """戦略切替との相関を分析"""
        
        # エラー発生タイミングと戦略切替の時間的近接性を分析
        error_dates = [e.timestamp for e in detection_result.error_instances]
        
        # 簡易的な戦略切替イベントの推定（価格変動率の変化から）
        returns = market_data['Adj Close'].pct_change()
        volatility_changes = returns.rolling(10).std().diff()
        
        # 大きな変動率変化を戦略切替候補とする
        switching_candidates = volatility_changes[abs(volatility_changes) > volatility_changes.std()]
        switching_dates = switching_candidates.index.tolist()
        
        # 時間的近接性の分析
        overlaps = 0
        for error_date in error_dates:
            for switch_date in switching_dates:
                if abs((error_date - switch_date).days) <= 3:  # 3日以内
                    overlaps += 1
                    break
        
        correlation = overlaps / len(error_dates) if error_dates else 0.0
        
        return {
            'correlation': correlation,
            'events_overlap': overlaps,
            'total_errors': len(error_dates),
            'switching_events_detected': len(switching_dates)
        }
    
    def _analyze_portfolio_risk_integration(self, 
                                          impact_results: List[ErrorImpactResult],
                                          portfolio_context: Dict[str, Any]) -> Dict[str, Any]:
        """ポートフォリオリスクとの統合分析"""
        
        # ポートフォリオレベルでのリスク増幅効果を分析
        total_portfolio_value = portfolio_context.get('total_portfolio_value', 1000000)
        active_strategies = portfolio_context.get('active_strategies', 3)
        
        # 個別エラーのポートフォリオへの影響
        individual_impacts = [r.impact_metrics.composite_score for r in impact_results]
        portfolio_level_impact = sum(individual_impacts) * (total_portfolio_value / 1000000)
        
        # 多様化効果の分析
        error_types = [r.error_instance.error_type for r in impact_results]
        unique_error_types = len(set(error_types))
        diversification_impact = max(0, 1 - (unique_error_types / len(TrendErrorType)))
        
        # リスク増幅係数
        risk_amplification = 1.0 + (len(individual_impacts) / active_strategies * 0.1)
        
        return {
            'portfolio_level_impact': portfolio_level_impact,
            'risk_amplification': risk_amplification,
            'diversification_impact': diversification_impact,
            'error_type_diversity': unique_error_types
        }
    
    def _generate_comprehensive_recommendations(self, 
                                              detection_result: ErrorDetectionResult,
                                              impact_results: List[ErrorImpactResult],
                                              integrated_metrics: Dict[str, float],
                                              system_correlations: Dict[str, Any]) -> Dict[str, List[str]]:
        """包括的な推奨事項を生成"""
        
        priority_recommendations = []
        immediate_actions = []
        long_term_improvements = []
        
        # 優先度の高い推奨事項
        if integrated_metrics['total_impact_score'] > self.impact_threshold:
            priority_recommendations.append("Critical: High total impact detected - immediate intervention required")
        
        if integrated_metrics['average_error_severity'] > 0.7:
            priority_recommendations.append("Severe error patterns detected - review trend detection algorithms")
        
        if integrated_metrics['error_concentration_index'] > 2.0:
            priority_recommendations.append("Error clustering detected - investigate market regime changes")
        
        # 即座に実行すべきアクション
        error_rate = detection_result.error_rate
        if error_rate > 0.3:
            immediate_actions.append("Reduce position sizes due to high error rate")
            immediate_actions.append("Implement additional confirmation filters")
        
        if integrated_metrics['systemic_risk_factor'] > 0.05:
            immediate_actions.append("Activate enhanced risk monitoring")
            immediate_actions.append("Consider temporary trading halt for affected strategies")
        
        # ドローダウン相関が高い場合
        if system_correlations.get('drawdown', {}).get('significance') == 'high':
            immediate_actions.append("Strengthen drawdown control mechanisms")
        
        # 長期的改善提案
        most_common_error = self._get_most_common_error_type(impact_results)
        if most_common_error:
            long_term_improvements.append(f"Focus improvement on {most_common_error} error reduction")
        
        if system_correlations.get('strategy_switching', {}).get('correlation', 0) > 0.5:
            long_term_improvements.append("Integrate trend error analysis with strategy switching system")
        
        long_term_improvements.append("Implement adaptive trend detection thresholds based on market volatility")
        long_term_improvements.append("Develop predictive error detection models")
        
        # デフォルト推奨事項
        if not priority_recommendations:
            priority_recommendations.append("Continue monitoring trend detection performance")
        if not immediate_actions:
            immediate_actions.append("Maintain current risk management protocols")
        if not long_term_improvements:
            long_term_improvements.append("Regular system performance review and optimization")
        
        return {
            'priority': priority_recommendations[:5],
            'immediate': immediate_actions[:5],
            'long_term': long_term_improvements[:5]
        }
    
    def _get_most_common_error_type(self, impact_results: List[ErrorImpactResult]) -> Optional[str]:
        """最も多いエラータイプを取得"""
        if not impact_results:
            return None
        
        error_counts = {}
        for result in impact_results:
            error_type = result.error_instance.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
    
    def _create_analysis_summary(self, 
                               detection_result: ErrorDetectionResult,
                               impact_results: List[ErrorImpactResult],
                               integrated_metrics: Dict[str, float],
                               system_correlations: Dict[str, Any]) -> Dict[str, Any]:
        """分析サマリーを作成"""
        
        summary = {
            "detection_summary": {
                "total_predictions": detection_result.total_predictions,
                "errors_detected": detection_result.errors_detected,
                "error_rate": detection_result.error_rate,
                "analysis_period_days": (detection_result.period_analyzed[1] - detection_result.period_analyzed[0]).days
            },
            "impact_summary": {
                "total_errors_analyzed": len(impact_results),
                "total_impact_score": integrated_metrics['total_impact_score'],
                "average_error_severity": integrated_metrics['average_error_severity'],
                "highest_individual_impact": max([r.impact_metrics.composite_score for r in impact_results]) if impact_results else 0.0
            },
            "system_integration": {
                "drawdown_correlation": system_correlations.get('drawdown', {}).get('correlation', 0.0),
                "strategy_switching_overlap": system_correlations.get('strategy_switching', {}).get('correlation', 0.0),
                "portfolio_risk_amplification": system_correlations.get('portfolio_risk', {}).get('risk_amplification', 1.0)
            },
            "risk_assessment": {
                "overall_risk_level": self._assess_overall_risk_level(integrated_metrics),
                "systemic_risk_factor": integrated_metrics['systemic_risk_factor'],
                "error_concentration": integrated_metrics['error_concentration_index']
            }
        }
        
        return summary
    
    def _assess_overall_risk_level(self, integrated_metrics: Dict[str, float]) -> str:
        """全体的リスクレベルを評価"""
        
        total_impact = integrated_metrics['total_impact_score']
        avg_severity = integrated_metrics['average_error_severity']
        systemic_risk = integrated_metrics['systemic_risk_factor']
        
        # 複合スコアによるリスク評価
        composite_risk = (total_impact * 0.4 + avg_severity * 0.3 + systemic_risk * 0.3)
        
        if composite_risk >= 0.2:
            return "CRITICAL"
        elif composite_risk >= 0.1:
            return "HIGH"
        elif composite_risk >= 0.05:
            return "MEDIUM"
        else:
            return "LOW"

# テスト用のメイン関数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # サンプルデータでのテスト
    dates = pd.date_range('2023-01-01', periods=100)
    
    # サンプル市場データ
    market_data = pd.DataFrame({
        'Adj Close': 100 + np.cumsum(np.random.randn(100) * 2),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # サンプル予測データ
    trend_predictions = pd.DataFrame({
        'predicted_trend': np.random.choice(['uptrend', 'downtrend', 'range-bound'], 100),
        'confidence': np.random.uniform(0.3, 0.9, 100)
    }, index=dates)
    
    # ポートフォリオコンテキスト
    portfolio_context = {
        'total_portfolio_value': 5000000,
        'active_strategies': 5
    }
    
    # 包括的分析の実行
    analyzer = TrendErrorAnalyzer()
    result = analyzer.analyze_trend_errors(
        market_data, trend_predictions, portfolio_context
    )
    
    print(f"Comprehensive Analysis Completed:")
    print(f"Total Impact Score: {result.total_impact_score:.4f}")
    print(f"Average Error Severity: {result.average_error_severity:.4f}")
    print(f"Risk Level: {result.analysis_summary['risk_assessment']['overall_risk_level']}")
    print(f"Priority Recommendations: {len(result.priority_recommendations)}")
    print(f"Immediate Actions: {len(result.immediate_actions)}")
    print(f"Long-term Improvements: {len(result.long_term_improvements)}")
