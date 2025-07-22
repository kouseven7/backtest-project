"""
2-2-3 統合意思決定システム (Integrated Decision System)

既存のConfidenceThresholdManagerと統合して、
包括的な意思決定ロジックを提供するシステム。
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import existing components
try:
    from confidence_threshold_manager import (
        ConfidenceThresholdManager, 
        ConfidenceThreshold, 
        DecisionResult, 
        ActionType, 
        ConfidenceLevel,
        create_confidence_threshold_manager
    )
    from indicators.unified_trend_detector import UnifiedTrendDetector
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")


class MarketCondition(Enum):
    """市場状況分類"""
    TRENDING = "trending"           # トレンド相場
    RANGE_BOUND = "range_bound"     # レンジ相場
    VOLATILE = "volatile"           # 高ボラティリティ
    STABLE = "stable"              # 安定状況
    UNKNOWN = "unknown"            # 不明


class RiskLevel(Enum):
    """リスクレベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class MarketContext:
    """市場コンテキスト情報"""
    volatility: float               # ボラティリティ
    volume_trend: str              # 出来高トレンド
    market_condition: MarketCondition
    risk_level: RiskLevel
    time_of_day: str               # 時間帯
    day_of_week: str               # 曜日
    additional_factors: Dict[str, Any]
    
    def __post_init__(self):
        if self.additional_factors is None:
            self.additional_factors = {}


class IntegratedDecisionSystem:
    """
    統合意思決定システム
    
    ConfidenceThresholdManagerと連携して、
    市場コンテキストを考慮した包括的な意思決定を提供
    """
    
    def __init__(self,
                 confidence_manager: ConfidenceThresholdManager,
                 risk_tolerance: float = 0.5,
                 max_position_size: float = 1.0,
                 logger: logging.Logger = None):
        """
        Parameters:
            confidence_manager: 信頼度閾値マネージャー
            risk_tolerance: リスク許容度（0-1）
            max_position_size: 最大ポジションサイズ
            logger: ロガー
        """
        self.confidence_manager = confidence_manager
        self.risk_tolerance = risk_tolerance
        self.max_position_size = max_position_size
        self.logger = logger or logging.getLogger(__name__)
        
        # 決定履歴と統計
        self.integrated_decisions: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # 市場分析キャッシュ
        self._market_analysis_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)  # 5分キャッシュ
    
    def analyze_market_context(self, data: pd.DataFrame) -> MarketContext:
        """
        市場コンテキスト分析
        
        Parameters:
            data: 市場データ
            
        Returns:
            MarketContext: 市場コンテキスト情報
        """
        # キャッシュチェック
        current_time = datetime.now()
        if (self._cache_timestamp and 
            current_time - self._cache_timestamp < self._cache_duration and
            self._market_analysis_cache):
            return self._market_analysis_cache.get('market_context')
        
        try:
            # 基本統計計算
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 年率ボラティリティ
            
            # 出来高トレンド分析
            if 'Volume' in data.columns:
                volume_sma_short = data['Volume'].rolling(5).mean()
                volume_sma_long = data['Volume'].rolling(20).mean()
                if len(volume_sma_short) > 0 and len(volume_sma_long) > 0:
                    volume_trend = "increasing" if volume_sma_short.iloc[-1] > volume_sma_long.iloc[-1] else "decreasing"
                else:
                    volume_trend = "stable"
            else:
                volume_trend = "unknown"
            
            # 市場状況判定
            if volatility > 0.3:
                if abs(returns.mean()) > 0.01:
                    market_condition = MarketCondition.TRENDING
                else:
                    market_condition = MarketCondition.VOLATILE
            elif volatility < 0.1:
                market_condition = MarketCondition.STABLE
            else:
                # 中程度のボラティリティ
                trend_strength = abs(returns.mean()) / returns.std() if returns.std() > 0 else 0
                market_condition = MarketCondition.TRENDING if trend_strength > 0.5 else MarketCondition.RANGE_BOUND
            
            # リスクレベル判定
            if volatility > 0.4:
                risk_level = RiskLevel.EXTREME
            elif volatility > 0.25:
                risk_level = RiskLevel.HIGH
            elif volatility > 0.15:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # 時間情報
            current_time = datetime.now()
            time_of_day = "market_hours"  # 簡略化
            day_of_week = current_time.strftime("%A")
            
            # 追加要因
            additional_factors = {
                "returns_skewness": returns.skew() if len(returns) > 10 else 0,
                "price_momentum": (data['Close'].iloc[-1] / data['Close'].iloc[-10] - 1) if len(data) >= 10 else 0,
                "volatility_regime": "high" if volatility > 0.2 else "normal"
            }
            
            market_context = MarketContext(
                volatility=volatility,
                volume_trend=volume_trend,
                market_condition=market_condition,
                risk_level=risk_level,
                time_of_day=time_of_day,
                day_of_week=day_of_week,
                additional_factors=additional_factors
            )
            
            # キャッシュ更新
            self._market_analysis_cache = {
                'market_context': market_context,
                'raw_data_analysis': {
                    'volatility': volatility,
                    'returns_mean': returns.mean(),
                    'returns_std': returns.std(),
                    'volume_trend': volume_trend
                }
            }
            self._cache_timestamp = current_time
            
            return market_context
            
        except Exception as e:
            self.logger.error(f"Error analyzing market context: {e}")
            # エラー時のデフォルト
            return MarketContext(
                volatility=0.2,
                volume_trend="unknown",
                market_condition=MarketCondition.UNKNOWN,
                risk_level=RiskLevel.MEDIUM,
                time_of_day="unknown",
                day_of_week="unknown",
                additional_factors={}
            )
    
    def adjust_decision_for_market_context(self,
                                         base_decision: DecisionResult,
                                         market_context: MarketContext) -> DecisionResult:
        """
        市場コンテキストに基づく意思決定調整
        
        Parameters:
            base_decision: 基本意思決定
            market_context: 市場コンテキスト
            
        Returns:
            DecisionResult: 調整後の意思決定
        """
        try:
            # コピーを作成して調整
            adjusted_action = base_decision.action
            adjusted_position_factor = base_decision.position_size_factor
            adjusted_reasoning = base_decision.reasoning
            
            # 市場状況による調整
            if market_context.market_condition == MarketCondition.VOLATILE:
                # 高ボラティリティ時はポジションサイズ削減
                adjusted_position_factor *= 0.5
                adjusted_reasoning += " [高ボラティリティ→ポジション半減]"
                
            elif market_context.market_condition == MarketCondition.RANGE_BOUND:
                # レンジ相場では保守的に
                if base_decision.action in [ActionType.BUY, ActionType.SELL]:
                    adjusted_position_factor *= 0.7
                    adjusted_reasoning += " [レンジ相場→保守的ポジション]"
            
            # リスクレベルによる調整
            risk_multipliers = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 0.8,
                RiskLevel.HIGH: 0.5,
                RiskLevel.EXTREME: 0.2
            }
            risk_multiplier = risk_multipliers.get(market_context.risk_level, 0.5)
            
            if risk_multiplier < self.risk_tolerance:
                adjusted_position_factor *= risk_multiplier
                adjusted_reasoning += f" [リスク調整:{market_context.risk_level.value}]"
            
            # 出来高トレンドによる調整
            if market_context.volume_trend == "decreasing" and adjusted_action in [ActionType.BUY, ActionType.SELL]:
                adjusted_position_factor *= 0.8
                adjusted_reasoning += " [出来高減少→慎重対応]"
            
            # 最大ポジションサイズ制限
            adjusted_position_factor = min(adjusted_position_factor, self.max_position_size)
            
            # アクションの再評価
            if adjusted_position_factor < 0.1 and adjusted_action in [ActionType.BUY, ActionType.SELL]:
                adjusted_action = ActionType.NO_ACTION
                adjusted_reasoning += " [ポジション係数低下→アクション取り消し]"
            
            # 調整された決定結果を作成
            adjusted_decision = DecisionResult(
                action=adjusted_action,
                confidence_score=base_decision.confidence_score,
                confidence_level=base_decision.confidence_level,
                position_size_factor=adjusted_position_factor,
                reasoning=adjusted_reasoning,
                additional_info={
                    **base_decision.additional_info,
                    "market_context": market_context,
                    "original_position_factor": base_decision.position_size_factor,
                    "market_adjustment_applied": True
                },
                timestamp=datetime.now()
            )
            
            return adjusted_decision
            
        except Exception as e:
            self.logger.error(f"Error adjusting decision for market context: {e}")
            return base_decision  # エラー時は元の決定を返す
    
    def make_integrated_decision(self,
                               data: pd.DataFrame,
                               current_position: float = 0.0,
                               unrealized_pnl: float = 0.0,
                               additional_context: Dict[str, Any] = None) -> DecisionResult:
        """
        統合意思決定実行
        
        Parameters:
            data: 市場データ
            current_position: 現在のポジション
            unrealized_pnl: 未実現損益
            additional_context: 追加コンテキスト
            
        Returns:
            DecisionResult: 最終意思決定結果
        """
        try:
            # 市場コンテキスト分析
            market_context = self.analyze_market_context(data)
            
            # 追加コンテキストを市場コンテキストに統合
            if additional_context:
                market_context.additional_factors.update(additional_context)
            
            # 基本意思決定取得
            base_decision = self.confidence_manager.make_comprehensive_decision(
                data=data,
                current_position=current_position,
                unrealized_pnl=unrealized_pnl,
                market_context=market_context.additional_factors
            )
            
            # 市場コンテキストによる調整
            final_decision = self.adjust_decision_for_market_context(
                base_decision=base_decision,
                market_context=market_context
            )
            
            # 統合決定履歴に追加
            decision_record = {
                "timestamp": final_decision.timestamp,
                "base_decision": base_decision,
                "market_context": market_context,
                "final_decision": final_decision,
                "input_data": {
                    "current_position": current_position,
                    "unrealized_pnl": unrealized_pnl,
                    "data_length": len(data)
                }
            }
            self.integrated_decisions.append(decision_record)
            
            # ログ出力
            self.logger.info(
                f"統合決定: {final_decision.action.value} "
                f"(信頼度: {final_decision.confidence_score:.3f}, "
                f"ポジション: {final_decision.position_size_factor:.2f}, "
                f"市場: {market_context.market_condition.value})"
            )
            
            return final_decision
            
        except Exception as e:
            self.logger.error(f"Error in integrated decision making: {e}")
            # エラー時のフォールバック
            return DecisionResult(
                action=ActionType.NO_ACTION,
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.UNRELIABLE,
                position_size_factor=0.0,
                reasoning=f"統合決定エラー: {str(e)}",
                additional_info={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def get_performance_summary(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        パフォーマンス要約取得
        
        Parameters:
            lookback_days: 分析対象期間（日）
            
        Returns:
            Dict[str, Any]: パフォーマンス要約
        """
        if not self.integrated_decisions:
            return {"error": "No decision history available"}
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_decisions = [
            d for d in self.integrated_decisions 
            if d["timestamp"] >= cutoff_date
        ]
        
        if not recent_decisions:
            return {"error": f"No decisions in last {lookback_days} days"}
        
        # 基本統計
        total_decisions = len(recent_decisions)
        actionable_decisions = sum(
            1 for d in recent_decisions 
            if d["final_decision"].is_actionable()
        )
        
        # 信頼度統計
        confidences = [d["final_decision"].confidence_score for d in recent_decisions]
        
        # 市場状況分析
        market_conditions = [d["market_context"].market_condition.value for d in recent_decisions]
        condition_counts = {}
        for condition in MarketCondition:
            condition_counts[condition.value] = market_conditions.count(condition.value)
        
        # アクション分析
        actions = [d["final_decision"].action.value for d in recent_decisions]
        action_counts = {}
        for action in ActionType:
            action_counts[action.value] = actions.count(action.value)
        
        # ポジションサイズ分析
        position_factors = [d["final_decision"].position_size_factor for d in recent_decisions]
        
        summary = {
            "period_days": lookback_days,
            "total_decisions": total_decisions,
            "actionable_decisions": actionable_decisions,
            "actionable_ratio": actionable_decisions / total_decisions if total_decisions > 0 else 0,
            
            "confidence_stats": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "high_confidence_ratio": sum(1 for c in confidences if c >= 0.8) / len(confidences)
            },
            
            "market_condition_distribution": condition_counts,
            "action_distribution": action_counts,
            
            "position_sizing_stats": {
                "mean_factor": np.mean(position_factors),
                "std_factor": np.std(position_factors),
                "full_position_ratio": sum(1 for f in position_factors if f >= 0.9) / len(position_factors)
            },
            
            "risk_management": {
                "conservative_decisions": sum(
                    1 for d in recent_decisions 
                    if d["final_decision"].position_size_factor < d["base_decision"].position_size_factor
                ),
                "market_context_adjustments": sum(
                    1 for d in recent_decisions 
                    if d["final_decision"].additional_info.get("market_adjustment_applied", False)
                )
            }
        }
        
        return summary
    
    def update_risk_parameters(self, 
                             new_risk_tolerance: float = None,
                             new_max_position_size: float = None):
        """リスクパラメータ更新"""
        if new_risk_tolerance is not None:
            self.risk_tolerance = max(0.0, min(1.0, new_risk_tolerance))
            
        if new_max_position_size is not None:
            self.max_position_size = max(0.0, min(1.0, new_max_position_size))
        
        self.logger.info(
            f"Risk parameters updated: tolerance={self.risk_tolerance:.2f}, "
            f"max_position={self.max_position_size:.2f}"
        )
    
    def clear_cache_and_history(self):
        """キャッシュと履歴をクリア"""
        self._market_analysis_cache = {}
        self._cache_timestamp = None
        self.integrated_decisions = []
        self.confidence_manager.clear_history()
        self.logger.info("Cache and history cleared")


def create_integrated_decision_system(strategy_name: str,
                                    data: pd.DataFrame,
                                    trend_method: str = "advanced",
                                    custom_thresholds: ConfidenceThreshold = None,
                                    risk_tolerance: float = 0.5) -> IntegratedDecisionSystem:
    """
    統合意思決定システムの簡単生成ヘルパー
    
    Parameters:
        strategy_name: 戦略名
        data: 市場データ
        trend_method: トレンド検出メソッド
        custom_thresholds: カスタム閾値設定
        risk_tolerance: リスク許容度
        
    Returns:
        IntegratedDecisionSystem: 設定済みシステム
    """
    
    # ConfidenceThresholdManagerの作成
    confidence_manager = create_confidence_threshold_manager(
        strategy_name=strategy_name,
        data=data,
        trend_method=trend_method,
        custom_thresholds=custom_thresholds
    )
    
    # 統合システムの作成
    return IntegratedDecisionSystem(
        confidence_manager=confidence_manager,
        risk_tolerance=risk_tolerance
    )


if __name__ == "__main__":
    # 使用例とテスト
    import sys
    import os
    
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("=== 2-2-3 統合意思決定システム テスト ===")
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    volumes = np.random.randint(1000, 10000, 100)
    
    test_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volumes
    })
    
    # 統合意思決定システムの作成
    integrated_system = create_integrated_decision_system(
        strategy_name="VWAP",
        data=test_data,
        trend_method="advanced",
        risk_tolerance=0.6
    )
    
    # 統合意思決定テスト
    print("\n--- 統合意思決定テスト ---")
    for i in range(5):
        decision = integrated_system.make_integrated_decision(
            data=test_data.iloc[:80+i*4],
            current_position=0.0 if i % 2 == 0 else 0.5,
            unrealized_pnl=np.random.uniform(-100, 100),
            additional_context={"test_iteration": i}
        )
        
        print(f"統合決定 {i+1}: {decision.action.value}")
        print(f"  信頼度: {decision.confidence_score:.3f} ({decision.confidence_level.value})")
        print(f"  ポジション係数: {decision.position_size_factor:.2f}")
        print(f"  理由: {decision.reasoning}")
        print()
    
    # パフォーマンス要約表示
    print("--- パフォーマンス要約 ---")
    summary = integrated_system.get_performance_summary()
    if "error" not in summary:
        print(f"総決定数: {summary['total_decisions']}")
        print(f"アクション可能比率: {summary['actionable_ratio']:.2%}")
        print(f"高信頼度比率: {summary['confidence_stats']['high_confidence_ratio']:.2%}")
        print(f"保守的決定数: {summary['risk_management']['conservative_decisions']}")
        print(f"市場調整適用数: {summary['risk_management']['market_context_adjustments']}")
    
    print("\n=== テスト完了 ===")
