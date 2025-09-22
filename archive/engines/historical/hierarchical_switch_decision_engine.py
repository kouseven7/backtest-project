"""
Hierarchical Switch Decision Engine
階層化切替決定エンジン：3つのレベルを統合したメインエンジン
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import json

from .decision_context import DecisionContext, HierarchicalDecisionResult
from .switch_decision_levels.base_level import BaseSwitchDecisionLevel
from .switch_decision_levels.level1_optimization_rules import Level1OptimizationRules
from .switch_decision_levels.level2_daily_target import Level2DailyTarget
from .switch_decision_levels.level3_emergency_constraints import Level3EmergencyConstraints


class HierarchicalSwitchDecisionEngine:
    """
    階層化切替決定エンジン
    
    3つのレベル（最適化ルール、日次ターゲット、緊急制約）を統合し、
    優先度に基づいて最適な戦略切替決定を行う
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
            config: 設定辞書（config_pathより優先）
        """
        self.logger = logging.getLogger("HierarchicalSwitchDecisionEngine")
        
        # 設定の読み込み
        self.config = self._load_config(config_path, config)
        
        # 決定レベルの初期化
        self.decision_levels = self._initialize_decision_levels()
        
        # エンジン状態
        self.last_decision = None
        self.decision_history = []
        self.engine_stats = {
            'total_decisions': 0,
            'level1_decisions': 0,
            'level2_decisions': 0,
            'level3_decisions': 0,
            'switch_decisions': 0,
            'maintain_decisions': 0,
            'emergency_decisions': 0
        }
        
        self.logger.info("HierarchicalSwitchDecisionEngine initialized")
    
    def make_decision(self, context: DecisionContext) -> HierarchicalDecisionResult:
        """
        階層化決定を実行
        
        Args:
            context: 決定コンテキスト
            
        Returns:
            HierarchicalDecisionResult: 最終決定結果
        """
        try:
            self.logger.info(f"Starting hierarchical decision process at {context.timestamp}")
            
            # アクティブレベルの特定
            active_levels = self._identify_active_levels(context)
            
            if not active_levels:
                return self._create_default_decision("No active decision levels")
            
            # 各レベルの決定を取得
            level_decisions = self._collect_level_decisions(context, active_levels)
            
            # 最終決定の選択
            final_decision = self._select_final_decision(context, level_decisions)
            
            # 決定の記録と統計更新
            self._record_decision(final_decision, context)
            
            self.logger.info(f"Final decision: {final_decision.decision_type} to {final_decision.target_strategy} "
                           f"(Level {final_decision.decision_level}, confidence: {final_decision.confidence:.3f})")
            
            return final_decision
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical decision making: {e}")
            return self._create_error_decision(str(e))
    
    def _load_config(self, config_path: Optional[str], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """設定を読み込み"""
        if config:
            return config
        
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # デフォルト設定
        return {
            'optimization_rules': {
                'performance_threshold': 0.05,
                'confidence_threshold': 0.7,
                'lookback_period': 20
            },
            'daily_target': {
                'target_threshold': 0.02,
                'achievement_ratio': 0.8,
                'time_pressure_factor': 0.5
            },
            'emergency_constraints': {
                'max_drawdown_threshold': 0.05,
                'var_threshold': 0.03,
                'volatility_threshold': 0.25
            },
            'engine_settings': {
                'max_history_length': 1000,
                'decision_timeout': 30,
                'enable_level_override': True
            }
        }
    
    def _initialize_decision_levels(self) -> Dict[int, BaseSwitchDecisionLevel]:
        """決定レベルを初期化"""
        levels = {}
        
        try:
            # Level 1: 最適化ルール
            levels[1] = Level1OptimizationRules(config=self.config)
            
            # Level 2: 日次ターゲット
            levels[2] = Level2DailyTarget(config=self.config)
            
            # Level 3: 緊急制約
            levels[3] = Level3EmergencyConstraints(config=self.config)
            
            self.logger.info(f"Initialized {len(levels)} decision levels")
            
        except Exception as e:
            self.logger.error(f"Error initializing decision levels: {e}")
            raise
        
        return levels
    
    def _identify_active_levels(self, context: DecisionContext) -> List[BaseSwitchDecisionLevel]:
        """アクティブな決定レベルを特定"""
        active_levels = []
        
        for level_num in sorted(self.decision_levels.keys(), reverse=True):  # Level3から評価
            level = self.decision_levels[level_num]
            if level.should_activate(context):
                active_levels.append(level)
                self.logger.debug(f"Level {level_num} activated")
        
        return active_levels
    
    def _collect_level_decisions(self, context: DecisionContext, 
                               active_levels: List[BaseSwitchDecisionLevel]) -> List[HierarchicalDecisionResult]:
        """各レベルの決定を収集"""
        level_decisions = []
        
        for level in active_levels:
            try:
                decision = level.evaluate_switch_condition(context)
                if decision:
                    level_decisions.append(decision)
                    self.logger.debug(f"Level {level.level_number} decision: {decision.decision_type}")
            except Exception as e:
                self.logger.error(f"Error in Level {level.level_number} evaluation: {e}")
        
        return level_decisions
    
    def _select_final_decision(self, context: DecisionContext, 
                             level_decisions: List[HierarchicalDecisionResult]) -> HierarchicalDecisionResult:
        """最終決定を選択"""
        if not level_decisions:
            return self._create_default_decision("No level decisions available")
        
        # 緊急停止決定が含まれている場合は最優先
        emergency_decisions = [d for d in level_decisions if d.is_emergency_decision()]
        if emergency_decisions:
            return emergency_decisions[0]  # 最初の緊急決定を採用
        
        # override_conditionsがTrueの決定があるかチェック
        override_decisions = [d for d in level_decisions if d.override_conditions]
        if override_decisions:
            # Level番号が高い（より緊急な）決定を優先
            return max(override_decisions, key=lambda d: d.decision_level)
        
        # 通常の優先度ベースの選択
        return self._select_by_priority(context, level_decisions)
    
    def _select_by_priority(self, context: DecisionContext, 
                           level_decisions: List[HierarchicalDecisionResult]) -> HierarchicalDecisionResult:
        """優先度ベースの決定選択"""
        # 切替決定を優先
        switch_decisions = [d for d in level_decisions if d.is_switch_decision()]
        
        if switch_decisions:
            # 信頼度と決定レベルを組み合わせた優先度計算
            def calculate_priority(decision):
                level_weight = {1: 1.0, 2: 1.2, 3: 1.5}  # レベルによる重み
                return decision.confidence * level_weight.get(decision.decision_level, 1.0)
            
            return max(switch_decisions, key=calculate_priority)
        
        # 切替決定がない場合は現状維持
        maintain_decisions = [d for d in level_decisions if d.is_maintain_decision()]
        if maintain_decisions:
            # Level番号が高い決定を優先
            return max(maintain_decisions, key=lambda d: d.decision_level)
        
        # フォールバック
        return level_decisions[0] if level_decisions else self._create_default_decision("No valid decisions")
    
    def _record_decision(self, decision: HierarchicalDecisionResult, context: DecisionContext) -> None:
        """決定を記録し統計を更新"""
        # 決定履歴に追加
        decision_record = {
            'timestamp': context.timestamp,
            'decision': decision,
            'context_summary': self._create_context_summary(context)
        }
        
        self.decision_history.append(decision_record)
        
        # 履歴サイズ制限
        max_history = self.config.get('engine_settings', {}).get('max_history_length', 1000)
        if len(self.decision_history) > max_history:
            self.decision_history = self.decision_history[-max_history:]
        
        # 統計更新
        self.engine_stats['total_decisions'] += 1
        self.engine_stats[f'level{decision.decision_level}_decisions'] += 1
        
        if decision.is_switch_decision():
            self.engine_stats['switch_decisions'] += 1
        elif decision.is_maintain_decision():
            self.engine_stats['maintain_decisions'] += 1
        elif decision.is_emergency_decision():
            self.engine_stats['emergency_decisions'] += 1
        
        # 最新決定を保存
        self.last_decision = decision
    
    def _create_context_summary(self, context: DecisionContext) -> Dict[str, Any]:
        """コンテキストサマリーを作成"""
        return {
            'strategies_count': len(context.strategies_data),
            'risk_metrics_count': len(context.risk_metrics),
            'has_emergency_signals': context.has_emergency_signal(),
            'market_conditions_count': len(context.market_conditions),
            'portfolio_metrics_count': len(context.portfolio_state)
        }
    
    def _create_default_decision(self, reason: str) -> HierarchicalDecisionResult:
        """デフォルト決定を作成"""
        return HierarchicalDecisionResult(
            decision_level=1,
            decision_type='maintain',
            confidence=0.5,
            reasoning=f"Default decision: {reason}"
        )
    
    def _create_error_decision(self, error_message: str) -> HierarchicalDecisionResult:
        """エラー時の決定を作成"""
        return HierarchicalDecisionResult(
            decision_level=3,
            decision_type='emergency_stop',
            confidence=1.0,
            reasoning=f"Error in decision engine: {error_message}"
        )
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """エンジン統計を取得"""
        stats = self.engine_stats.copy()
        stats['decision_history_length'] = len(self.decision_history)
        
        # last_decision_timeの安全な取得
        if self.last_decision and hasattr(self.last_decision, 'metadata') and self.last_decision.metadata:
            stats['last_decision_time'] = self.last_decision.metadata.get('timestamp')
        else:
            stats['last_decision_time'] = None
            
        return stats
    
    def get_recent_decisions(self, count: int = 10) -> List[Dict[str, Any]]:
        """最近の決定履歴を取得"""
        return self.decision_history[-count:] if self.decision_history else []
    
    def reset_stats(self) -> None:
        """統計をリセット"""
        self.engine_stats = {key: 0 for key in self.engine_stats.keys()}
        self.logger.info("Engine statistics reset")
    
    def validate_configuration(self) -> bool:
        """設定の妥当性を検証"""
        try:
            required_sections = ['optimization_rules', 'daily_target', 'emergency_constraints']
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"Missing required config section: {section}")
                    return False
            
            # 各レベルの設定妥当性チェック
            for level_num, level in self.decision_levels.items():
                if not hasattr(level, 'evaluate_switch_condition'):
                    self.logger.error(f"Level {level_num} missing required method")
                    return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def __repr__(self) -> str:
        return (f"HierarchicalSwitchDecisionEngine("
                f"levels={len(self.decision_levels)}, "
                f"decisions={self.engine_stats['total_decisions']})")
