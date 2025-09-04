"""
Decision Context for Hierarchical Switch Decision System
決定コンテキスト：階層化切替決定システムで使用される全体的な状況情報
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime


@dataclass
class DecisionContext:
    """
    階層化切替決定システムで使用される決定コンテキスト
    
    Attributes:
        strategies_data: 各戦略のデータとパフォーマンス情報
        risk_metrics: リスク管理メトリクス
        market_conditions: 市場状況データ
        portfolio_state: ポートフォリオ状態
        timestamp: 決定時点のタイムスタンプ
        emergency_signals: 緊急シグナル情報
    """
    
    # 戦略関連データ
    strategies_data: Dict[str, Dict[str, Any]]
    
    # リスク管理メトリクス
    risk_metrics: Dict[str, float]
    
    # 市場状況
    market_conditions: Dict[str, Any]
    
    # ポートフォリオ状態
    portfolio_state: Dict[str, Any]
    
    # タイムスタンプ
    timestamp: datetime
    
    # 緊急シグナル
    emergency_signals: Optional[Dict[str, Any]] = None
    
    # 追加メタデータ
    metadata: Optional[Dict[str, Any]] = None
    
    def get_strategy_score(self, strategy_name: str) -> float:
        """戦略のスコアを取得"""
        if strategy_name not in self.strategies_data:
            return 0.0
        return self.strategies_data[strategy_name].get('score', 0.0)
    
    def get_strategy_weight(self, strategy_name: str) -> float:
        """戦略の重みを取得"""
        if strategy_name not in self.strategies_data:
            return 0.0
        return self.strategies_data[strategy_name].get('weight', 0.0)
    
    def get_risk_metric(self, metric_name: str) -> float:
        """リスクメトリクスを取得"""
        return self.risk_metrics.get(metric_name, 0.0)
    
    def has_emergency_signal(self, signal_type: str = None) -> bool:
        """緊急シグナルの存在確認"""
        if not self.emergency_signals:
            return False
        if signal_type is None:
            return bool(self.emergency_signals)
        return signal_type in self.emergency_signals
    
    def get_market_indicator(self, indicator_name: str) -> Any:
        """市場指標を取得"""
        return self.market_conditions.get(indicator_name)
    
    def get_portfolio_metric(self, metric_name: str) -> Any:
        """ポートフォリオメトリクスを取得"""
        return self.portfolio_state.get(metric_name)


@dataclass
class HierarchicalDecisionResult:
    """
    階層化決定結果
    
    Attributes:
        decision_level: 決定レベル (1, 2, 3)
        decision_type: 決定タイプ ('switch', 'maintain', 'emergency_stop')
        target_strategy: 切替先戦略名（None = 現状維持）
        confidence: 決定の信頼度 (0.0-1.0)
        reasoning: 決定理由
        override_conditions: 上位レベルの条件を無視するかどうか
        metadata: 追加メタデータ
    """
    
    decision_level: int
    decision_type: str
    target_strategy: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    override_conditions: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def is_switch_decision(self) -> bool:
        """切替決定かどうか"""
        return self.decision_type == 'switch' and self.target_strategy is not None
    
    def is_emergency_decision(self) -> bool:
        """緊急決定かどうか"""
        return self.decision_type == 'emergency_stop'
    
    def is_maintain_decision(self) -> bool:
        """現状維持決定かどうか"""
        return self.decision_type == 'maintain'
