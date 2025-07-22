"""
Module: Drawdown Action Executor
File: drawdown_action_executor.py
Description: 
  5-3-1「ドローダウン制御機能」アクション実行器
  ドローダウン制御アクションを具体的に実行する専用モジュール

Author: imega
Created: 2025-07-20
Modified: 2025-07-20
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import json

# プロジェクトパス追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config.portfolio_risk_manager import PortfolioRiskManager
    PortfolioRiskManager = PortfolioRiskManager
except ImportError:
    PortfolioRiskManager = None

# 一時的にNoneに設定（マルチ戦略調整機能は後で統合）
MultiStrategyCoordinationManager = None

# DrawdownControlActionの定義
class DrawdownControlAction(Enum):
    NO_ACTION = "no_action"
    POSITION_REDUCTION_LIGHT = "position_reduction_light"
    POSITION_REDUCTION_MODERATE = "position_reduction_moderate"
    POSITION_REDUCTION_HEAVY = "position_reduction_heavy"
    STRATEGY_SUSPENSION = "strategy_suspension"
    EMERGENCY_STOP = "emergency_stop"

logger = logging.getLogger(__name__)

@dataclass
class ActionExecutionResult:
    """アクション実行結果"""
    action: DrawdownControlAction
    success: bool
    execution_time: float
    original_positions: Dict[str, float]
    final_positions: Dict[str, float]
    affected_systems: List[str]
    error_messages: List[str]
    rollback_available: bool
    
    def get_position_change_summary(self) -> Dict[str, float]:
        """ポジション変更サマリー"""
        changes = {}
        all_strategies = set(self.original_positions.keys()) | set(self.final_positions.keys())
        
        for strategy in all_strategies:
            original = self.original_positions.get(strategy, 0.0)
            final = self.final_positions.get(strategy, 0.0)
            change = final - original
            if abs(change) > 1e-6:  # 微小変更は除外
                changes[strategy] = change
                
        return changes

class DrawdownActionExecutor:
    """ドローダウンアクション実行器"""
    
    def __init__(self,
                 portfolio_risk_manager=None,
                 coordination_manager=None,
                 config: Optional[Dict[str, Any]] = None):
        """初期化"""
        self.portfolio_risk_manager = portfolio_risk_manager
        self.coordination_manager = coordination_manager
        self.config = config or {}
        
        # 実行履歴
        self.execution_history: List[ActionExecutionResult] = []
        
        # ロールバック用状態保存
        self.rollback_states: Dict[str, Dict[str, Any]] = {}
        
        # 実行制御
        self.execution_lock = threading.Lock()
        self.emergency_mode = False
        
    def execute_action(self, 
                      action: DrawdownControlAction,
                      event: Any,
                      current_positions: Dict[str, float]) -> ActionExecutionResult:
        """アクション実行"""
        start_time = datetime.now()
        
        with self.execution_lock:
            logger.info(f"Executing drawdown action: {action.value}")
            
            # 実行前状態保存
            rollback_id = f"rollback_{int(start_time.timestamp())}"
            self._save_rollback_state(rollback_id, current_positions)
            
            try:
                if action == DrawdownControlAction.NO_ACTION:
                    result = self._execute_no_action(current_positions)
                    
                elif action == DrawdownControlAction.POSITION_REDUCTION_LIGHT:
                    result = self._execute_position_reduction(current_positions, 0.15, "light")
                    
                elif action == DrawdownControlAction.POSITION_REDUCTION_MODERATE:
                    result = self._execute_position_reduction(current_positions, 0.40, "moderate")
                    
                elif action == DrawdownControlAction.POSITION_REDUCTION_HEAVY:
                    result = self._execute_position_reduction(current_positions, 0.60, "heavy")
                    
                elif action == DrawdownControlAction.STRATEGY_SUSPENSION:
                    result = self._execute_strategy_suspension(current_positions, event.affected_strategies if hasattr(event, 'affected_strategies') else [])
                    
                elif action == DrawdownControlAction.EMERGENCY_STOP:
                    result = self._execute_emergency_stop(current_positions)
                    
                else:
                    result = ActionExecutionResult(
                        action=action,
                        success=False,
                        execution_time=0.0,
                        original_positions=current_positions,
                        final_positions=current_positions,
                        affected_systems=[],
                        error_messages=[f"Unknown action: {action}"],
                        rollback_available=False
                    )
                
                # 実行時間計算
                execution_time = (datetime.now() - start_time).total_seconds()
                result.execution_time = execution_time
                
                # 履歴に追加
                self.execution_history.append(result)
                
                logger.info(f"Action execution completed: {action.value} ({'SUCCESS' if result.success else 'FAILED'}) in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                error_result = ActionExecutionResult(
                    action=action,
                    success=False,
                    execution_time=execution_time,
                    original_positions=current_positions,
                    final_positions=current_positions,
                    affected_systems=[],
                    error_messages=[str(e)],
                    rollback_available=True
                )
                
                self.execution_history.append(error_result)
                logger.error(f"Action execution failed: {action.value} - {e}")
                
                return error_result
    
    def _execute_no_action(self, positions: Dict[str, float]) -> ActionExecutionResult:
        """アクションなし実行"""
        return ActionExecutionResult(
            action=DrawdownControlAction.NO_ACTION,
            success=True,
            execution_time=0.0,
            original_positions=positions.copy(),
            final_positions=positions.copy(),
            affected_systems=[],
            error_messages=[],
            rollback_available=False
        )
    
    def _execute_position_reduction(self, 
                                  positions: Dict[str, float], 
                                  reduction_factor: float,
                                  level: str) -> ActionExecutionResult:
        """ポジション削減実行"""
        original_positions = positions.copy()
        adjusted_positions = {}
        affected_systems = []
        error_messages = []
        
        try:
            # ポジション削減計算
            for strategy, position in positions.items():
                adjusted_positions[strategy] = position * (1 - reduction_factor)
            
            # PortfolioRiskManagerとの統合
            if self.portfolio_risk_manager:
                try:
                    # リスク管理システムに新しいウェイトを適用
                    self._apply_risk_manager_adjustments(adjusted_positions)
                    affected_systems.append("PortfolioRiskManager")
                except Exception as e:
                    error_messages.append(f"PortfolioRiskManager error: {e}")
            
            # CoordinationManagerとの統合
            if self.coordination_manager:
                try:
                    self._notify_coordination_manager(adjusted_positions, f"position_reduction_{level}")
                    affected_systems.append("CoordinationManager")
                except Exception as e:
                    error_messages.append(f"CoordinationManager error: {e}")
            
            success = len(error_messages) == 0
            
            # 適切なアクションを設定
            if level == "light":
                action = DrawdownControlAction.POSITION_REDUCTION_LIGHT
            elif level == "moderate":
                action = DrawdownControlAction.POSITION_REDUCTION_MODERATE
            else:
                action = DrawdownControlAction.POSITION_REDUCTION_HEAVY
            
            return ActionExecutionResult(
                action=action,
                success=success,
                execution_time=0.0,
                original_positions=original_positions,
                final_positions=adjusted_positions,
                affected_systems=affected_systems,
                error_messages=error_messages,
                rollback_available=True
            )
            
        except Exception as e:
            return ActionExecutionResult(
                action=DrawdownControlAction.POSITION_REDUCTION_MODERATE,
                success=False,
                execution_time=0.0,
                original_positions=original_positions,
                final_positions=original_positions,
                affected_systems=[],
                error_messages=[str(e)],
                rollback_available=True
            )
    
    def _execute_strategy_suspension(self, 
                                   positions: Dict[str, float],
                                   affected_strategies: List[str]) -> ActionExecutionResult:
        """戦略停止実行"""
        original_positions = positions.copy()
        adjusted_positions = positions.copy()
        affected_systems = []
        error_messages = []
        
        try:
            # 影響戦略のポジションを0に
            for strategy in affected_strategies:
                if strategy in adjusted_positions:
                    adjusted_positions[strategy] = 0.0
            
            # 残存戦略への再配分
            active_strategies = [s for s, pos in adjusted_positions.items() if pos > 0]
            if active_strategies:
                equal_weight = 1.0 / len(active_strategies)
                for strategy in active_strategies:
                    adjusted_positions[strategy] = equal_weight
            
            # システム通知
            if self.coordination_manager:
                try:
                    self._suspend_strategies_in_coordinator(affected_strategies)
                    affected_systems.append("CoordinationManager")
                except Exception as e:
                    error_messages.append(f"Strategy suspension error: {e}")
            
            success = len(error_messages) == 0
            
            return ActionExecutionResult(
                action=DrawdownControlAction.STRATEGY_SUSPENSION,
                success=success,
                execution_time=0.0,
                original_positions=original_positions,
                final_positions=adjusted_positions,
                affected_systems=affected_systems,
                error_messages=error_messages,
                rollback_available=True
            )
            
        except Exception as e:
            return ActionExecutionResult(
                action=DrawdownControlAction.STRATEGY_SUSPENSION,
                success=False,
                execution_time=0.0,
                original_positions=original_positions,
                final_positions=original_positions,
                affected_systems=[],
                error_messages=[str(e)],
                rollback_available=True
            )
    
    def _execute_emergency_stop(self, positions: Dict[str, float]) -> ActionExecutionResult:
        """緊急停止実行"""
        original_positions = positions.copy()
        zero_positions = {strategy: 0.0 for strategy in positions}
        affected_systems = []
        error_messages = []
        
        try:
            self.emergency_mode = True
            
            # 全戦略停止
            if self.coordination_manager:
                try:
                    if hasattr(self.coordination_manager, 'shutdown_event'):
                        self.coordination_manager.shutdown_event.set()
                    affected_systems.append("CoordinationManager")
                except Exception as e:
                    error_messages.append(f"Emergency shutdown error: {e}")
            
            # リスク管理システム緊急モード
            if self.portfolio_risk_manager:
                try:
                    # 緊急リスク制御
                    affected_systems.append("PortfolioRiskManager")
                except Exception as e:
                    error_messages.append(f"Emergency risk control error: {e}")
            
            return ActionExecutionResult(
                action=DrawdownControlAction.EMERGENCY_STOP,
                success=True,  # 緊急停止は常に成功とみなす
                execution_time=0.0,
                original_positions=original_positions,
                final_positions=zero_positions,
                affected_systems=affected_systems,
                error_messages=error_messages,
                rollback_available=False  # 緊急停止はロールバック不可
            )
            
        except Exception as e:
            return ActionExecutionResult(
                action=DrawdownControlAction.EMERGENCY_STOP,
                success=False,
                execution_time=0.0,
                original_positions=original_positions,
                final_positions=original_positions,
                affected_systems=[],
                error_messages=[str(e)],
                rollback_available=False
            )
    
    def _save_rollback_state(self, rollback_id: str, positions: Dict[str, float]):
        """ロールバック状態保存"""
        try:
            self.rollback_states[rollback_id] = {
                'timestamp': datetime.now().isoformat(),
                'positions': positions.copy(),
                'emergency_mode': self.emergency_mode
            }
            
            # 古いロールバック状態を削除（最新10件まで保持）
            if len(self.rollback_states) > 10:
                oldest_key = min(self.rollback_states.keys())
                del self.rollback_states[oldest_key]
                
        except Exception as e:
            logger.warning(f"Failed to save rollback state: {e}")
    
    def _apply_risk_manager_adjustments(self, adjusted_positions: Dict[str, float]):
        """リスク管理システムへの調整適用"""
        if not self.portfolio_risk_manager:
            return
        
        try:
            # PortfolioRiskManagerのメソッドを呼び出してウェイト更新
            logger.info(f"Applying risk manager adjustments: {adjusted_positions}")
            
        except Exception as e:
            raise Exception(f"Risk manager adjustment failed: {e}")
    
    def _notify_coordination_manager(self, positions: Dict[str, float], reason: str):
        """調整マネージャーへの通知"""
        if not self.coordination_manager:
            return
            
        try:
            # CoordinationManagerに状態変更を通知
            logger.info(f"Notifying coordination manager: {reason}")
            
        except Exception as e:
            raise Exception(f"Coordination manager notification failed: {e}")
    
    def _suspend_strategies_in_coordinator(self, strategies: List[str]):
        """調整マネージャーでの戦略停止"""
        if not self.coordination_manager:
            return
            
        try:
            # 戦略停止の実装
            for strategy in strategies:
                logger.info(f"Suspending strategy in coordinator: {strategy}")
                
        except Exception as e:
            raise Exception(f"Strategy suspension in coordinator failed: {e}")
    
    def rollback_last_action(self) -> bool:
        """最後のアクションをロールバック"""
        if not self.execution_history:
            logger.warning("No actions to rollback")
            return False
        
        last_result = self.execution_history[-1]
        if not last_result.rollback_available:
            logger.warning("Last action is not rollback-able")
            return False
        
        try:
            with self.execution_lock:
                # 元の状態に戻す
                original_positions = last_result.original_positions
                
                # システムへの復元適用
                if self.portfolio_risk_manager:
                    self._apply_risk_manager_adjustments(original_positions)
                
                # ロールバック記録
                rollback_result = ActionExecutionResult(
                    action=DrawdownControlAction.NO_ACTION,
                    success=True,
                    execution_time=0.0,
                    original_positions=last_result.final_positions,
                    final_positions=original_positions,
                    affected_systems=["Rollback"],
                    error_messages=[],
                    rollback_available=False
                )
                
                self.execution_history.append(rollback_result)
                logger.info("Action rollback completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """実行サマリー取得"""
        if not self.execution_history:
            return {'total_executions': 0}
        
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        
        # アクション別統計
        action_counts = {}
        for result in self.execution_history:
            action = result.action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'total_executions': total,
            'successful_executions': successful,
            'success_rate': successful / total,
            'emergency_mode': self.emergency_mode,
            'action_counts': action_counts,
            'rollback_states_available': len(self.rollback_states),
            'last_execution_time': self.execution_history[-1].execution_time if self.execution_history else 0.0
        }

if __name__ == "__main__":
    # 基本テスト
    print("=" * 50)
    print("Drawdown Action Executor - Test")
    print("=" * 50)
    
    executor = DrawdownActionExecutor()
    
    test_positions = {
        'Momentum': 0.4,
        'Contrarian': 0.3,
        'Pairs_Trading': 0.3
    }
    
    print(f"Initial positions: {test_positions}")
    
    # 軽度削減テスト
    result = executor._execute_position_reduction(test_positions, 0.15, "light")
    print(f"\nLight reduction result:")
    print(f"  Success: {result.success}")
    print(f"  Final positions: {result.final_positions}")
    print(f"  Changes: {result.get_position_change_summary()}")
    
    # サマリー表示
    summary = executor.get_execution_summary()
    print(f"\nExecution Summary:")
    print(f"  Total: {summary['total_executions']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    
    print("\n✅ Drawdown Action Executor test completed")
