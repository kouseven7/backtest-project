"""
Module: Error Impact Calculator
File: error_impact_calculator.py
Description: 
  5-1-2「トレンド判定エラーの影響分析」
  トレンド判定エラーの影響度計算エンジン

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
from enum import Enum
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
    from .error_classification_engine import TrendErrorInstance, TrendErrorType, ErrorSeverity
except ImportError:
    from error_classification_engine import TrendErrorInstance, TrendErrorType, ErrorSeverity

@dataclass
class ImpactMetrics:
    """影響度指標"""
    direct_loss: float                    # 直接的損失
    opportunity_cost: float              # 機会損失
    risk_adjusted_impact: float          # リスク調整後影響
    systemic_impact: float              # システム影響
    composite_score: float              # 複合スコア
    confidence_interval: Tuple[float, float]  # 信頼区間
    calculation_timestamp: datetime

@dataclass
class ErrorImpactResult:
    """エラー影響分析結果"""
    error_instance: TrendErrorInstance
    impact_metrics: ImpactMetrics
    contributing_factors: Dict[str, float]
    mitigation_suggestions: List[str]

class ErrorImpactCalculator:
    """エラー影響度計算エンジン"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Parameters:
            config_path: 設定ファイルのパス
        """
        self.logger = logger
        self.config_path = config_path or self._get_default_config_path()
        self.impact_config = self._load_impact_config()
        
        # 計算パラメータ
        self.risk_free_rate = self.impact_config.get("risk_free_rate", 0.02)
        self.impact_weights = self.impact_config.get("impact_weights", {
            'direct': 0.4,
            'opportunity': 0.3,
            'risk': 0.2,
            'systemic': 0.1
        })
        self.position_size = self.impact_config.get("default_position_size", 0.02)
        
        # パフォーマンス指標統合
        self._initialize_performance_metrics()
    
    def _get_default_config_path(self) -> str:
        """デフォルト設定ファイルパスを取得"""
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "trend_error_analysis", "impact_calculation_config.json"
        )
    
    def _load_impact_config(self) -> Dict[str, Any]:
        """影響計算設定を読み込み"""
        try:
            import json
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return self._get_default_impact_config()
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self._get_default_impact_config()
    
    def _get_default_impact_config(self) -> Dict[str, Any]:
        """デフォルト影響計算設定を取得"""
        return {
            "risk_free_rate": 0.02,
            "impact_weights": {
                'direct': 0.4,
                'opportunity': 0.3,
                'risk': 0.2,
                'systemic': 0.1
            },
            "default_position_size": 0.02,
            "volatility_adjustment": 1.5,
            "confidence_level": 0.95,
            "lookback_days": 30
        }
    
    def _initialize_performance_metrics(self):
        """パフォーマンス指標計算システムを初期化"""
        try:
            # 既存のperformance_metricsモジュールと統合
            from metrics import performance_metrics
            self.performance_calculator = performance_metrics
            self.logger.info("Performance metrics integration initialized")
        except ImportError:
            self.logger.warning("Performance metrics module not available, using fallback")
            self.performance_calculator = None
    
    def calculate_error_impact(self, 
                             error_instance: TrendErrorInstance,
                             market_data: pd.DataFrame,
                             portfolio_context: Optional[Dict[str, Any]] = None) -> ErrorImpactResult:
        """
        エラーの影響度を計算
        
        Parameters:
            error_instance: エラーインスタンス
            market_data: 市場データ
            portfolio_context: ポートフォリオコンテキスト
        
        Returns:
            ErrorImpactResult: 影響分析結果
        """
        try:
            self.logger.info(f"Calculating impact for error at {error_instance.timestamp}")
            
            # 市場データの準備
            relevant_data = self._prepare_market_data(market_data, error_instance.timestamp)
            
            # 各種影響指標の計算
            direct_loss = self._calculate_direct_loss(error_instance, relevant_data)
            opportunity_cost = self._calculate_opportunity_cost(error_instance, relevant_data)
            risk_adjusted_impact = self._calculate_risk_adjusted_impact(
                error_instance, relevant_data, direct_loss, opportunity_cost
            )
            systemic_impact = self._calculate_systemic_impact(
                error_instance, portfolio_context or {}
            )
            
            # 複合スコアの計算
            composite_score = self._calculate_composite_score(
                direct_loss, opportunity_cost, risk_adjusted_impact, systemic_impact
            )
            
            # 信頼区間の計算
            confidence_interval = self._calculate_confidence_interval(
                composite_score, error_instance, relevant_data
            )
            
            # 影響指標の作成
            impact_metrics = ImpactMetrics(
                direct_loss=direct_loss,
                opportunity_cost=opportunity_cost,
                risk_adjusted_impact=risk_adjusted_impact,
                systemic_impact=systemic_impact,
                composite_score=composite_score,
                confidence_interval=confidence_interval,
                calculation_timestamp=datetime.now()
            )
            
            # 寄与要因の分析
            contributing_factors = self._analyze_contributing_factors(
                error_instance, impact_metrics, relevant_data
            )
            
            # 軽減提案の生成
            mitigation_suggestions = self._generate_mitigation_suggestions(
                error_instance, impact_metrics
            )
            
            result = ErrorImpactResult(
                error_instance=error_instance,
                impact_metrics=impact_metrics,
                contributing_factors=contributing_factors,
                mitigation_suggestions=mitigation_suggestions
            )
            
            self.logger.info(f"Impact calculation completed: composite score = {composite_score:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error impact calculation failed: {e}")
            raise
    
    def _prepare_market_data(self, 
                           market_data: pd.DataFrame,
                           error_timestamp: datetime) -> pd.DataFrame:
        """エラー時点前後の市場データを準備"""
        
        lookback_days = self.impact_config.get("lookback_days", 30)
        lookforward_days = 10  # エラー後の影響を見る期間
        
        start_date = error_timestamp - timedelta(days=lookback_days)
        end_date = error_timestamp + timedelta(days=lookforward_days)
        
        # 期間でフィルタ
        mask = (market_data.index >= start_date) & (market_data.index <= end_date)
        relevant_data = market_data[mask].copy()
        
        if len(relevant_data) == 0:
            raise ValueError(f"No market data available for period {start_date} to {end_date}")
        
        # 必要な指標を計算
        relevant_data['returns'] = relevant_data['Adj Close'].pct_change()
        relevant_data['volatility'] = relevant_data['returns'].rolling(window=20, min_periods=5).std()
        relevant_data['volume_sma'] = relevant_data['Volume'].rolling(window=10, min_periods=3).mean()
        
        return relevant_data
    
    def _calculate_direct_loss(self, 
                             error_instance: TrendErrorInstance,
                             market_data: pd.DataFrame) -> float:
        """直接的損失を計算"""
        
        try:
            # エラー時点の価格を取得
            error_date = error_instance.timestamp
            if error_date not in market_data.index:
                # 最も近い日付を探す
                closest_date = min(market_data.index, key=lambda x: abs((x - error_date).days))
                error_price = market_data.loc[closest_date, 'Adj Close']
            else:
                error_price = market_data.loc[error_date, 'Adj Close']
            
            # エラー後の実際の価格変動
            future_dates = market_data.index[market_data.index > error_date]
            if len(future_dates) == 0:
                return 0.0
            
            # 5日後の価格での影響を計算
            target_date = future_dates[min(4, len(future_dates)-1)]
            actual_price = market_data.loc[target_date, 'Adj Close']
            
            # 実際のリターン
            actual_return = (actual_price / error_price) - 1
            
            # エラーによる損失計算
            if error_instance.error_type == TrendErrorType.FALSE_POSITIVE:
                # 上昇トレンドと誤判定した場合
                if error_instance.predicted_trend == "uptrend" and actual_return < 0:
                    direct_loss = abs(actual_return) * self.position_size
                else:
                    direct_loss = 0.0
            elif error_instance.error_type == TrendErrorType.FALSE_NEGATIVE:
                # トレンドを見逃した場合
                if actual_return > 0:
                    direct_loss = actual_return * self.position_size
                else:
                    direct_loss = 0.0
            elif error_instance.error_type == TrendErrorType.DIRECTION_WRONG:
                # 方向を間違えた場合
                expected_return = 0.02 if error_instance.predicted_trend == "uptrend" else -0.02
                direct_loss = abs(actual_return - expected_return) * self.position_size
            else:
                # その他のエラー
                direct_loss = abs(actual_return) * self.position_size * 0.5
            
            # 深刻度による調整
            severity_multiplier = {
                ErrorSeverity.LOW: 0.5,
                ErrorSeverity.MEDIUM: 1.0,
                ErrorSeverity.HIGH: 1.5,
                ErrorSeverity.CRITICAL: 2.0
            }.get(error_instance.severity, 1.0)
            
            return direct_loss * severity_multiplier
            
        except Exception as e:
            self.logger.error(f"Direct loss calculation failed: {e}")
            return 0.0
    
    def _calculate_opportunity_cost(self, 
                                  error_instance: TrendErrorInstance,
                                  market_data: pd.DataFrame) -> float:
        """機会損失を計算"""
        
        try:
            error_date = error_instance.timestamp
            
            # 正しい判定をした場合に得られたであろう利益を推定
            if error_instance.actual_trend == "uptrend":
                expected_return = 0.03  # 上昇トレンドでの期待リターン
            elif error_instance.actual_trend == "downtrend":
                expected_return = -0.02  # 下降トレンドでの期待リターン（ショート）
            else:
                expected_return = 0.0  # レンジ相場
            
            # 実際の判定での期待リターン
            if error_instance.predicted_trend == "uptrend":
                predicted_return = 0.025
            elif error_instance.predicted_trend == "downtrend":
                predicted_return = -0.015
            else:
                predicted_return = 0.0
            
            # 機会損失 = 正しい判定での利益 - 間違った判定での利益
            opportunity_cost = max(0, expected_return - predicted_return) * self.position_size
            
            # 信頼度による調整（高い信頼度で間違えた場合は機会損失が大きい）
            confidence_multiplier = 1 + (error_instance.confidence_level - 0.5)
            
            return opportunity_cost * confidence_multiplier
            
        except Exception as e:
            self.logger.error(f"Opportunity cost calculation failed: {e}")
            return 0.0
    
    def _calculate_risk_adjusted_impact(self, 
                                      error_instance: TrendErrorInstance,
                                      market_data: pd.DataFrame,
                                      direct_loss: float,
                                      opportunity_cost: float) -> float:
        """リスク調整後影響を計算"""
        
        try:
            error_date = error_instance.timestamp
            
            # 市場ボラティリティを取得
            if error_date in market_data.index:
                volatility = market_data.loc[error_date, 'volatility']
            else:
                volatility = market_data['volatility'].fillna(0.2).iloc[-1]  # デフォルト値
            
            if pd.isna(volatility) or volatility <= 0:
                volatility = 0.2  # デフォルトボラティリティ
            
            # ボラティリティ調整
            volatility_adjustment = self.impact_config.get("volatility_adjustment", 1.5)
            adjusted_volatility = min(volatility * volatility_adjustment, 1.0)  # 上限設定
            
            # リスク調整後の影響計算
            base_impact = direct_loss + opportunity_cost
            risk_adjusted_impact = base_impact * (1 + adjusted_volatility)
            
            # シャープレシオ的な調整を適用
            if self.performance_calculator:
                try:
                    # 既存のシャープレシオ計算を利用
                    returns_series = market_data['returns'].dropna()
                    if len(returns_series) > 10:
                        sharpe_ratio = self.performance_calculator.calculate_sharpe_ratio(
                            returns_series, self.risk_free_rate
                        )
                        if sharpe_ratio > 0:
                            risk_adjusted_impact *= (1 / max(sharpe_ratio, 0.1))
                except:
                    pass  # Fallback to basic calculation
            
            return risk_adjusted_impact
            
        except Exception as e:
            self.logger.error(f"Risk adjusted impact calculation failed: {e}")
            return direct_loss + opportunity_cost
    
    def _calculate_systemic_impact(self, 
                                 error_instance: TrendErrorInstance,
                                 portfolio_context: Dict[str, Any]) -> float:
        """システム全体への影響を計算"""
        
        try:
            # ポートフォリオレベルでの影響を評価
            base_systemic_impact = 0.01  # ベース値
            
            # エラータイプによる影響度
            error_type_multipliers = {
                TrendErrorType.DIRECTION_WRONG: 2.0,      # 方向間違いは影響大
                TrendErrorType.CONFIDENCE_MISMATCH: 1.5,  # 信頼度ミスマッチ
                TrendErrorType.FALSE_POSITIVE: 1.2,       # 偽陽性
                TrendErrorType.FALSE_NEGATIVE: 1.3,       # 偽陰性
                TrendErrorType.REGIME_MISMATCH: 1.8,      # レジーム誤判定
                TrendErrorType.TIMING_EARLY: 0.8,         # 早すぎるタイミング
                TrendErrorType.TIMING_LATE: 1.0           # 遅すぎるタイミング
            }
            
            multiplier = error_type_multipliers.get(error_instance.error_type, 1.0)
            
            # ポートフォリオサイズによる調整
            portfolio_size = portfolio_context.get('total_portfolio_value', 100000)
            size_adjustment = min(portfolio_size / 1000000, 2.0)  # 大きなポートフォリオほど影響大
            
            # 他の戦略への波及影響
            strategy_count = portfolio_context.get('active_strategies', 3)
            cascade_impact = min(strategy_count / 10.0, 1.0)
            
            systemic_impact = (
                base_systemic_impact * 
                multiplier * 
                size_adjustment * 
                (1 + cascade_impact)
            )
            
            # 深刻度による最終調整
            severity_multiplier = {
                ErrorSeverity.LOW: 0.5,
                ErrorSeverity.MEDIUM: 1.0,
                ErrorSeverity.HIGH: 2.0,
                ErrorSeverity.CRITICAL: 3.0
            }.get(error_instance.severity, 1.0)
            
            return systemic_impact * severity_multiplier
            
        except Exception as e:
            self.logger.error(f"Systemic impact calculation failed: {e}")
            return 0.01
    
    def _calculate_composite_score(self, 
                                 direct_loss: float,
                                 opportunity_cost: float,
                                 risk_adjusted_impact: float,
                                 systemic_impact: float) -> float:
        """複合影響スコアを計算"""
        
        weights = self.impact_weights
        
        composite_score = (
            direct_loss * weights['direct'] +
            opportunity_cost * weights['opportunity'] +
            risk_adjusted_impact * weights['risk'] +
            systemic_impact * weights['systemic']
        )
        
        return composite_score
    
    def _calculate_confidence_interval(self, 
                                     composite_score: float,
                                     error_instance: TrendErrorInstance,
                                     market_data: pd.DataFrame) -> Tuple[float, float]:
        """信頼区間を計算"""
        
        try:
            # 市場ボラティリティに基づく不確実性
            volatility = market_data['volatility'].fillna(0.2).mean()
            if pd.isna(volatility) or volatility <= 0:
                volatility = 0.2
            
            # 信頼区間の幅を計算
            confidence_level = self.impact_config.get("confidence_level", 0.95)
            z_score = 1.96 if confidence_level >= 0.95 else 1.64  # 95%または90%
            
            # 標準誤差の推定
            standard_error = composite_score * volatility * 0.5
            
            margin_of_error = z_score * standard_error
            
            lower_bound = max(0, composite_score - margin_of_error)
            upper_bound = composite_score + margin_of_error
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.error(f"Confidence interval calculation failed: {e}")
            return (composite_score * 0.8, composite_score * 1.2)
    
    def _analyze_contributing_factors(self, 
                                    error_instance: TrendErrorInstance,
                                    impact_metrics: ImpactMetrics,
                                    market_data: pd.DataFrame) -> Dict[str, float]:
        """寄与要因を分析"""
        
        factors = {}
        
        # 各要因の寄与度を計算
        total_impact = impact_metrics.composite_score
        if total_impact > 0:
            factors['direct_loss_contribution'] = impact_metrics.direct_loss / total_impact
            factors['opportunity_cost_contribution'] = impact_metrics.opportunity_cost / total_impact
            factors['risk_contribution'] = impact_metrics.risk_adjusted_impact / total_impact
            factors['systemic_contribution'] = impact_metrics.systemic_impact / total_impact
        
        # 市場要因
        try:
            error_date = error_instance.timestamp
            if error_date in market_data.index:
                volatility = market_data.loc[error_date, 'volatility']
                volume_ratio = (market_data.loc[error_date, 'Volume'] / 
                              market_data['volume_sma'].loc[error_date])
                factors['volatility_factor'] = min(volatility, 1.0) if not pd.isna(volatility) else 0.2
                factors['volume_factor'] = min(volume_ratio, 3.0) if not pd.isna(volume_ratio) else 1.0
        except:
            factors['volatility_factor'] = 0.2
            factors['volume_factor'] = 1.0
        
        # エラー特性要因
        factors['confidence_factor'] = error_instance.confidence_level
        factors['severity_factor'] = {
            ErrorSeverity.LOW: 0.25,
            ErrorSeverity.MEDIUM: 0.5,
            ErrorSeverity.HIGH: 0.75,
            ErrorSeverity.CRITICAL: 1.0
        }.get(error_instance.severity, 0.5)
        
        return factors
    
    def _generate_mitigation_suggestions(self, 
                                       error_instance: TrendErrorInstance,
                                       impact_metrics: ImpactMetrics) -> List[str]:
        """軽減提案を生成"""
        
        suggestions = []
        
        # エラータイプ別の提案
        if error_instance.error_type == TrendErrorType.FALSE_POSITIVE:
            suggestions.append("Consider implementing stricter confirmation filters before trend signals")
            suggestions.append("Reduce position size during uncertain market conditions")
        
        elif error_instance.error_type == TrendErrorType.FALSE_NEGATIVE:
            suggestions.append("Review trend detection sensitivity settings")
            suggestions.append("Implement early warning system for missed trend opportunities")
        
        elif error_instance.error_type == TrendErrorType.DIRECTION_WRONG:
            suggestions.append("Enhance directional trend detection algorithms")
            suggestions.append("Add multiple timeframe confirmation")
        
        elif error_instance.error_type == TrendErrorType.CONFIDENCE_MISMATCH:
            suggestions.append("Recalibrate confidence scoring methodology")
            suggestions.append("Implement confidence-based position sizing")
        
        # 深刻度別の提案
        if error_instance.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            suggestions.append("Implement emergency override procedures")
            suggestions.append("Consider temporary reduction in automated trading")
        
        # 影響度が高い場合の提案
        if impact_metrics.composite_score > 0.05:
            suggestions.append("Implement real-time error monitoring system")
            suggestions.append("Consider diversification to reduce single-point impact")
        
        # デフォルト提案
        if not suggestions:
            suggestions.append("Continue monitoring and gradual system improvements")
        
        return suggestions[:5]  # 最大5つの提案

# バッチ処理用のユーティリティクラス
class BatchErrorImpactAnalyzer:
    """バッチエラー影響分析器"""
    
    def __init__(self, impact_calculator: ErrorImpactCalculator):
        self.calculator = impact_calculator
        self.logger = logger
    
    def analyze_error_batch(self, 
                          error_instances: List[TrendErrorInstance],
                          market_data: pd.DataFrame,
                          portfolio_context: Optional[Dict[str, Any]] = None) -> List[ErrorImpactResult]:
        """複数のエラーをバッチで分析"""
        
        results = []
        
        for i, error_instance in enumerate(error_instances):
            try:
                self.logger.info(f"Processing error {i+1}/{len(error_instances)}")
                result = self.calculator.calculate_error_impact(
                    error_instance, market_data, portfolio_context
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process error {i+1}: {e}")
                continue
        
        self.logger.info(f"Batch processing completed: {len(results)}/{len(error_instances)} errors processed")
        return results
    
    def generate_batch_summary(self, results: List[ErrorImpactResult]) -> Dict[str, Any]:
        """バッチ処理結果のサマリーを生成"""
        
        if not results:
            return {"summary": "No results to analyze"}
        
        # 統計的サマリー
        composite_scores = [r.impact_metrics.composite_score for r in results]
        direct_losses = [r.impact_metrics.direct_loss for r in results]
        
        summary = {
            "total_errors_analyzed": len(results),
            "average_composite_score": np.mean(composite_scores),
            "total_direct_loss": sum(direct_losses),
            "max_impact_error": max(composite_scores),
            "most_common_error_type": self._get_most_common_error_type(results),
            "high_impact_errors": len([s for s in composite_scores if s > 0.05]),
            "total_impact_estimation": sum(composite_scores)
        }
        
        return summary
    
    def _get_most_common_error_type(self, results: List[ErrorImpactResult]) -> str:
        """最も多いエラータイプを取得"""
        error_counts = {}
        for result in results:
            error_type = result.error_instance.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        if error_counts:
            return max(error_counts.items(), key=lambda x: x[1])[0]
        return "unknown"

# テスト用のメイン関数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # サンプルデータでのテスト
    from error_classification_engine import TrendErrorInstance, TrendErrorType, ErrorSeverity
    
    # サンプル市場データ
    dates = pd.date_range('2023-01-01', periods=100)
    market_data = pd.DataFrame({
        'Adj Close': 100 + np.cumsum(np.random.randn(100) * 2),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # サンプルエラーインスタンス
    error_instance = TrendErrorInstance(
        timestamp=dates[50],
        error_type=TrendErrorType.FALSE_POSITIVE,
        severity=ErrorSeverity.MEDIUM,
        predicted_trend="uptrend",
        actual_trend="range-bound",
        confidence_level=0.7,
        market_context={"volatility": 0.15, "volume_ratio": 1.2}
    )
    
    # 影響計算の実行
    calculator = ErrorImpactCalculator()
    result = calculator.calculate_error_impact(error_instance, market_data)
    
    print(f"Impact calculation completed:")
    print(f"Composite Score: {result.impact_metrics.composite_score:.4f}")
    print(f"Direct Loss: {result.impact_metrics.direct_loss:.4f}")
    print(f"Opportunity Cost: {result.impact_metrics.opportunity_cost:.4f}")
    print(f"Mitigation Suggestions: {len(result.mitigation_suggestions)}")
