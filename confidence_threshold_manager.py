"""
2-2-3 信頼度閾値に基づく意思決定ロジック (Confidence Threshold-based Decision Logic)

This module implements confidence threshold-based decision logic for trading strategies.
It integrates with the existing unified trend detector to provide decision support
based on configurable confidence thresholds.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime

# Import existing unified trend detector
try:
    from indicators.unified_trend_detector import UnifiedTrendDetector
except ImportError:
    # Fallback for direct imports
    from unified_trend_detector import UnifiedTrendDetector


class ConfidenceLevel(Enum):
    """信頼度レベル定義"""
    UNRELIABLE = "unreliable"  # 0.0 - 0.39
    LOW = "low"                # 0.4 - 0.59
    MEDIUM = "medium"          # 0.6 - 0.79
    HIGH = "high"              # 0.8 - 1.0


class ActionType(Enum):
    """アクション種別"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"
    REDUCE_POSITION = "reduce_position"
    INCREASE_POSITION = "increase_position"
    NO_ACTION = "no_action"


@dataclass
class ConfidenceThreshold:
    """信頼度閾値設定"""
    # 基本閾値
    entry_threshold: float = 0.7    # エントリー最小信頼度
    exit_threshold: float = 0.5     # 損切り/利確実行最小信頼度
    hold_threshold: float = 0.6     # ホールド継続最小信頼度
    
    # 高度な閾値
    high_confidence_threshold: float = 0.8      # 高信頼度アクション閾値
    low_confidence_threshold: float = 0.4       # 低信頼度警告閾値
    position_sizing_threshold: float = 0.75     # ポジションサイジング判定閾値
    
    # 戦略別調整
    strategy_multipliers: Dict[str, float] = None  # 戦略別の信頼度倍率
    
    def __post_init__(self):
        if self.strategy_multipliers is None:
            self.strategy_multipliers = {
                "VWAP": 1.0,
                "Golden_Cross": 1.1,
                "Mean_Reversion": 0.9,
                "Momentum": 1.05
            }


@dataclass
class DecisionResult:
    """意思決定結果"""
    action: ActionType
    confidence_score: float
    confidence_level: ConfidenceLevel
    position_size_factor: float  # ポジションサイズ調整係数（0.0-1.0）
    reasoning: str
    additional_info: Dict[str, Any]
    timestamp: datetime
    
    def is_actionable(self) -> bool:
        """アクション実行可能かどうか"""
        return self.action != ActionType.NO_ACTION
    
    def get_risk_level(self) -> str:
        """リスクレベル取得"""
        if self.confidence_score >= 0.8:
            return "low"
        elif self.confidence_score >= 0.6:
            return "medium"
        else:
            return "high"


class ConfidenceThresholdManager:
    """
    信頼度閾値に基づく意思決定マネージャー
    
    既存のUnifiedTrendDetectorと統合して、信頼度に基づく
    包括的な意思決定ロジックを提供します。
    """
    
    def __init__(self, 
                 trend_detector: UnifiedTrendDetector,
                 thresholds: ConfidenceThreshold = None,
                 logger: logging.Logger = None):
        """
        Parameters:
            trend_detector: 統合トレンド検出器
            thresholds: 信頼度閾値設定
            logger: ロガー
        """
        self.trend_detector = trend_detector
        self.thresholds = thresholds or ConfidenceThreshold()
        self.logger = logger or logging.getLogger(__name__)
        
        # 意思決定履歴
        self.decision_history: List[DecisionResult] = []
        
        # 戦略固有の調整値
        self.strategy_name = trend_detector.strategy_name
        self.confidence_multiplier = self.thresholds.strategy_multipliers.get(
            self.strategy_name, 1.0)
    
    def get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """信頼度スコアからレベルを判定"""
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNRELIABLE
    
    def adjust_confidence_for_strategy(self, raw_confidence: float) -> float:
        """戦略別信頼度調整"""
        adjusted = raw_confidence * self.confidence_multiplier
        return min(max(adjusted, 0.0), 1.0)  # 0-1範囲にクリップ
    
    def calculate_position_size_factor(self, confidence_score: float) -> float:
        """
        信頼度に基づくポジションサイズ係数計算
        
        Returns:
            float: ポジションサイズ係数（0.0-1.0）
        """
        if confidence_score >= self.thresholds.high_confidence_threshold:
            # 高信頼度：フルポジション
            return 1.0
        elif confidence_score >= self.thresholds.position_sizing_threshold:
            # 中高信頼度：75%ポジション
            return 0.75
        elif confidence_score >= self.thresholds.entry_threshold:
            # 中信頼度：50%ポジション
            return 0.5
        elif confidence_score >= self.thresholds.hold_threshold:
            # 低中信頼度：25%ポジション
            return 0.25
        else:
            # 低信頼度：ポジション取らない
            return 0.0
    
    def make_entry_decision(self, 
                           current_trend: str,
                           current_position: float = 0.0,
                           market_context: Dict[str, Any] = None) -> DecisionResult:
        """
        エントリー意思決定
        
        Parameters:
            current_trend: 現在のトレンド
            current_position: 現在のポジション（正=買い、負=売り）
            market_context: 市場コンテキスト情報
            
        Returns:
            DecisionResult: 意思決定結果
        """
        market_context = market_context or {}
        
        # 信頼度取得と調整
        raw_confidence = self.trend_detector.get_confidence_score()
        adjusted_confidence = self.adjust_confidence_for_strategy(raw_confidence)
        confidence_level = self.get_confidence_level(adjusted_confidence)
        
        # ポジションサイズ計算
        position_size_factor = self.calculate_position_size_factor(adjusted_confidence)
        
        # 意思決定ロジック
        action = ActionType.NO_ACTION
        reasoning = ""
        
        if adjusted_confidence >= self.thresholds.entry_threshold:
            if current_trend == "uptrend" and current_position <= 0:
                action = ActionType.BUY
                reasoning = f"上昇トレンド確認、信頼度{adjusted_confidence:.2f}でエントリー条件満たす"
            elif current_trend == "downtrend" and current_position >= 0:
                action = ActionType.SELL
                reasoning = f"下降トレンド確認、信頼度{adjusted_confidence:.2f}でエントリー条件満たす"
            else:
                action = ActionType.HOLD
                reasoning = f"トレンド確認済みだが、既存ポジションあり。信頼度{adjusted_confidence:.2f}"
        elif adjusted_confidence >= self.thresholds.low_confidence_threshold:
            action = ActionType.HOLD
            reasoning = f"信頼度{adjusted_confidence:.2f}は低いが、様子見継続"
        else:
            action = ActionType.NO_ACTION
            reasoning = f"信頼度{adjusted_confidence:.2f}が低すぎる。アクション見送り"
        
        # 追加情報
        additional_info = {
            "raw_confidence": raw_confidence,
            "adjusted_confidence": adjusted_confidence,
            "strategy_multiplier": self.confidence_multiplier,
            "current_trend": current_trend,
            "current_position": current_position,
            "market_context": market_context
        }
        
        return DecisionResult(
            action=action,
            confidence_score=adjusted_confidence,
            confidence_level=confidence_level,
            position_size_factor=position_size_factor,
            reasoning=reasoning,
            additional_info=additional_info,
            timestamp=datetime.now()
        )
    
    def make_exit_decision(self,
                          current_trend: str,
                          current_position: float,
                          unrealized_pnl: float = 0.0,
                          market_context: Dict[str, Any] = None) -> DecisionResult:
        """
        エグジット意思決定
        
        Parameters:
            current_trend: 現在のトレンド
            current_position: 現在のポジション
            unrealized_pnl: 未実現損益
            market_context: 市場コンテキスト
            
        Returns:
            DecisionResult: 意思決定結果
        """
        market_context = market_context or {}
        
        # 信頼度取得と調整
        raw_confidence = self.trend_detector.get_confidence_score()
        adjusted_confidence = self.adjust_confidence_for_strategy(raw_confidence)
        confidence_level = self.get_confidence_level(adjusted_confidence)
        
        # ポジションサイズ計算
        position_size_factor = self.calculate_position_size_factor(adjusted_confidence)
        
        # 意思決定ロジック
        action = ActionType.NO_ACTION
        reasoning = ""
        
        if current_position == 0:
            action = ActionType.NO_ACTION
            reasoning = "ポジションなし"
        elif adjusted_confidence >= self.thresholds.exit_threshold:
            # トレンド継続信号が強い場合
            if (current_position > 0 and current_trend == "uptrend") or \
               (current_position < 0 and current_trend == "downtrend"):
                action = ActionType.HOLD
                reasoning = f"トレンド継続確認、信頼度{adjusted_confidence:.2f}でホールド継続"
            else:
                # トレンド転換の可能性
                action = ActionType.EXIT
                reasoning = f"トレンド転換可能性、信頼度{adjusted_confidence:.2f}でエグジット"
        else:
            # 信頼度が低い場合
            if unrealized_pnl < 0:  # 含み損がある場合
                action = ActionType.EXIT
                reasoning = f"信頼度{adjusted_confidence:.2f}低下、含み損あり、損切り実行"
            else:
                action = ActionType.REDUCE_POSITION
                reasoning = f"信頼度{adjusted_confidence:.2f}低下、ポジション削減"
        
        # 追加情報
        additional_info = {
            "raw_confidence": raw_confidence,
            "adjusted_confidence": adjusted_confidence,
            "current_trend": current_trend,
            "current_position": current_position,
            "unrealized_pnl": unrealized_pnl,
            "market_context": market_context
        }
        
        return DecisionResult(
            action=action,
            confidence_score=adjusted_confidence,
            confidence_level=confidence_level,
            position_size_factor=position_size_factor,
            reasoning=reasoning,
            additional_info=additional_info,
            timestamp=datetime.now()
        )
    
    def make_comprehensive_decision(self,
                                  data: pd.DataFrame,
                                  current_position: float = 0.0,
                                  unrealized_pnl: float = 0.0,
                                  market_context: Dict[str, Any] = None) -> DecisionResult:
        """
        包括的意思決定（エントリー・エグジット総合判定）
        
        Parameters:
            data: 市場データ
            current_position: 現在のポジション
            unrealized_pnl: 未実現損益
            market_context: 市場コンテキスト
            
        Returns:
            DecisionResult: 意思決定結果
        """
        try:
            # トレンド検出
            current_trend, raw_confidence = self.trend_detector.detect_trend_with_confidence()
            
            # エントリー・エグジット判定の統合
            if current_position == 0:
                # ポジションなし：エントリー判定
                decision = self.make_entry_decision(
                    current_trend=current_trend,
                    current_position=current_position,
                    market_context=market_context
                )
            else:
                # ポジションあり：エグジット判定
                decision = self.make_exit_decision(
                    current_trend=current_trend,
                    current_position=current_position,
                    unrealized_pnl=unrealized_pnl,
                    market_context=market_context
                )
            
            # 決定履歴に追加
            self.decision_history.append(decision)
            
            # ログ出力
            self.logger.info(f"Decision: {decision.action.value}, "
                           f"Confidence: {decision.confidence_score:.3f}, "
                           f"Reason: {decision.reasoning}")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive decision: {e}")
            # エラー時のフォールバック決定
            return DecisionResult(
                action=ActionType.NO_ACTION,
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.UNRELIABLE,
                position_size_factor=0.0,
                reasoning=f"意思決定エラー: {str(e)}",
                additional_info={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def get_decision_statistics(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        意思決定統計取得
        
        Parameters:
            lookback_days: 統計対象期間（日）
            
        Returns:
            Dict[str, Any]: 統計情報
        """
        if not self.decision_history:
            return {"error": "No decision history available"}
        
        cutoff_date = datetime.now() - pd.Timedelta(days=lookback_days)
        recent_decisions = [d for d in self.decision_history if d.timestamp >= cutoff_date]
        
        if not recent_decisions:
            return {"error": f"No decisions in last {lookback_days} days"}
        
        # 統計計算
        actions = [d.action for d in recent_decisions]
        confidences = [d.confidence_score for d in recent_decisions]
        
        action_counts = {}
        for action in ActionType:
            action_counts[action.value] = sum(1 for a in actions if a == action)
        
        stats = {
            "total_decisions": len(recent_decisions),
            "action_counts": action_counts,
            "confidence_stats": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "median": np.median(confidences)
            },
            "high_confidence_ratio": sum(1 for c in confidences if c >= 0.8) / len(confidences),
            "actionable_ratio": sum(1 for d in recent_decisions if d.is_actionable()) / len(recent_decisions),
            "period_days": lookback_days
        }
        
        return stats
    
    def update_thresholds(self, new_thresholds: ConfidenceThreshold):
        """信頼度閾値の更新"""
        self.thresholds = new_thresholds
        self.confidence_multiplier = self.thresholds.strategy_multipliers.get(
            self.strategy_name, 1.0)
        self.logger.info(f"Updated confidence thresholds for {self.strategy_name}")
    
    def clear_history(self):
        """意思決定履歴をクリア"""
        self.decision_history = []
        self.logger.info("Decision history cleared")


def create_confidence_threshold_manager(strategy_name: str,
                                      data: pd.DataFrame,
                                      trend_method: str = "advanced",
                                      custom_thresholds: ConfidenceThreshold = None) -> ConfidenceThresholdManager:
    """
    ConfidenceThresholdManagerの簡単生成ヘルパー
    
    Parameters:
        strategy_name: 戦略名
        data: 市場データ
        trend_method: トレンド検出メソッド
        custom_thresholds: カスタム閾値設定
        
    Returns:
        ConfidenceThresholdManager: 設定済みマネージャー
    """
    
    # UnifiedTrendDetectorの作成
    trend_detector = UnifiedTrendDetector(
        strategy_name=strategy_name,
        method=trend_method,
        data=data
    )
    
    # ConfidenceThresholdManagerの作成
    return ConfidenceThresholdManager(
        trend_detector=trend_detector,
        thresholds=custom_thresholds or ConfidenceThreshold()
    )


if __name__ == "__main__":
    # 使用例とテスト
    import sys
    import os
    
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("=== 2-2-3 信頼度閾値に基づく意思決定ロジック テスト ===")
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    test_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    # ConfidenceThresholdManagerの作成
    manager = create_confidence_threshold_manager(
        strategy_name="VWAP",
        data=test_data,
        trend_method="advanced"
    )
    
    # 包括的意思決定テスト
    print("\n--- 包括的意思決定テスト ---")
    for i in range(5):
        decision = manager.make_comprehensive_decision(
            data=test_data.iloc[:80+i*4],  # データを段階的に増やす
            current_position=0.0,
            market_context={"test_iteration": i}
        )
        
        print(f"決定 {i+1}: {decision.action.value}")
        print(f"  信頼度: {decision.confidence_score:.3f} ({decision.confidence_level.value})")
        print(f"  ポジション係数: {decision.position_size_factor:.2f}")
        print(f"  理由: {decision.reasoning}")
        print()
    
    # 統計情報表示
    print("--- 意思決定統計 ---")
    stats = manager.get_decision_statistics()
    if "error" not in stats:
        print(f"総決定数: {stats['total_decisions']}")
        print(f"高信頼度比率: {stats['high_confidence_ratio']:.2%}")
        print(f"アクション可能比率: {stats['actionable_ratio']:.2%}")
        print(f"平均信頼度: {stats['confidence_stats']['mean']:.3f}")
    
    print("\n=== テスト完了 ===")
