"""
Base Switch Decision Level for Hierarchical Switch Decision System
基底切替決定レベル：階層化切替決定システムの抽象基底クラス
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from ..decision_context import DecisionContext, HierarchicalDecisionResult


class BaseSwitchDecisionLevel(ABC):
    """
    階層化切替決定システムの基底クラス
    
    各決定レベル（Level1, Level2, Level3）は、このクラスを継承して実装される
    """
    
    def __init__(self, level_number: int, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Args:
            level_number: レベル番号 (1, 2, 3)
            config: レベル固有の設定
        """
        self.level_number = level_number
        self.config = config or {}
        self.logger = logging.getLogger(f"HierarchicalSwitch.Level{level_number}")
        
        # レベル固有の設定を読み込み
        self._load_level_config()
    
    @abstractmethod
    def _load_level_config(self) -> None:
        """レベル固有の設定を読み込み（各レベルで実装）"""
        pass
    
    @abstractmethod
    def evaluate_switch_condition(self, context: DecisionContext) -> HierarchicalDecisionResult:
        """
        切替条件を評価して決定結果を返す（各レベルで実装）
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            HierarchicalDecisionResult: 決定結果
        """
        pass
    
    def should_activate(self, context: DecisionContext) -> bool:
        """
        このレベルがアクティベートするべきかどうかを判定
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            bool: アクティベートするべきかどうか
        """
        # デフォルトでは常にアクティブ
        return True
    
    def get_priority_score(self, context: DecisionContext) -> float:
        """
        このレベルの優先度スコアを計算
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            float: 優先度スコア（高いほど優先）
        """
        # デフォルトではレベル番号の逆順（Level3が最優先）
        return 10.0 - self.level_number
    
    def validate_decision(self, decision: HierarchicalDecisionResult, context: DecisionContext) -> bool:
        """
        決定結果の妥当性を検証
        
        Args:
            decision: 決定結果
            context: 決定コンテキスト
            
        Returns:
            bool: 妥当性フラグ
        """
        # 基本的な妥当性チェック
        if decision.decision_level != self.level_number:
            self.logger.warning(f"Decision level mismatch: expected {self.level_number}, got {decision.decision_level}")
            return False
        
        if decision.confidence < 0.0 or decision.confidence > 1.0:
            self.logger.warning(f"Invalid confidence value: {decision.confidence}")
            return False
        
        if decision.decision_type not in ['switch', 'maintain', 'emergency_stop']:
            self.logger.warning(f"Invalid decision type: {decision.decision_type}")
            return False
        
        return True
    
    def log_decision(self, decision: HierarchicalDecisionResult, context: DecisionContext) -> None:
        """
        決定結果をログに記録
        
        Args:
            decision: 決定結果
            context: 決定コンテキスト
        """
        self.logger.info(
            f"Level {self.level_number} Decision: "
            f"Type={decision.decision_type}, "
            f"Target={decision.target_strategy}, "
            f"Confidence={decision.confidence:.3f}, "
            f"Reasoning={decision.reasoning}"
        )
    
    def get_level_name(self) -> str:
        """レベル名を取得"""
        level_names = {
            1: "OptimizationRules",
            2: "DailyTarget", 
            3: "EmergencyConstraints"
        }
        return level_names.get(self.level_number, f"Level{self.level_number}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(level={self.level_number})"
