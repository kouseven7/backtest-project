"""
DSSMS 緊急修正コーディネーター
Task 3.4: 危機的パフォーマンス状況での自動修正システム
"""
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from .performance_target_manager import TargetResult, AchievementLevel, TargetPhase
from .comprehensive_evaluator import ComprehensiveEvaluationResult, DimensionScore

class FixPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FixCategory(Enum):
    EMERGENCY_STOP = "emergency_stop"
    POSITION_SIZING = "position_sizing"
    RISK_PARAMETERS = "risk_parameters"
    STRATEGY_WEIGHTS = "strategy_weights"
    EXECUTION_SETTINGS = "execution_settings"

@dataclass
class FixAction:
    """修正アクション情報"""
    action_id: str
    category: FixCategory
    priority: FixPriority
    description: str
    target_parameter: str
    current_value: Any
    recommended_value: Any
    confidence: float
    estimated_impact: float
    execution_timestamp: Optional[datetime] = None
    execution_result: Optional[str] = None

@dataclass
class EmergencyFixResult:
    """緊急修正結果"""
    trigger_condition: str
    actions_executed: List[FixAction]
    actions_pending: List[FixAction]
    overall_success: bool
    execution_summary: str
    post_fix_metrics: Optional[Dict[str, float]] = None

class EmergencyFixCoordinator:
    """緊急修正システムの統合管理"""
    
    def __init__(self, config_path: str = "config/dssms/emergency_fix_config.json"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.fix_config = self._load_fix_config()
        self.fix_history = []
        self.emergency_thresholds = self._initialize_emergency_thresholds()
        
    def _load_fix_config(self) -> Dict[str, Any]:
        """緊急修正設定の読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"緊急修正設定ファイルが見つかりません: {self.config_path}")
            return self._get_default_fix_config()
        except Exception as e:
            self.logger.error(f"緊急修正設定の読み込みエラー: {e}")
            return self._get_default_fix_config()
    
    def _get_default_fix_config(self) -> Dict[str, Any]:
        """デフォルト緊急修正設定"""
        return {
            "emergency_thresholds": {
                "portfolio_loss_limit": -30.0,
                "drawdown_limit": 40.0,
                "var_limit": 10.0,
                "consecutive_losses": 5
            },
            "auto_fix_enabled": True,
            "fix_escalation_rules": {
                "level_1": {"max_position_size": 0.05, "stop_loss_tight": 0.02},
                "level_2": {"max_position_size": 0.03, "stop_loss_tight": 0.015},
                "level_3": {"emergency_stop": True}
            },
            "notification_settings": {
                "email_alerts": True,
                "log_level": "CRITICAL"
            }
        }
    
    def _initialize_emergency_thresholds(self) -> Dict[str, float]:
        """緊急時閾値の初期化"""
        return self.fix_config.get("emergency_thresholds", {
            "portfolio_loss_limit": -30.0,
            "drawdown_limit": 40.0,
            "var_limit": 10.0,
            "overall_score_limit": 30.0,
            "risk_adjusted_score_limit": 25.0
        })
    
    def evaluate_emergency_conditions(
        self, 
        evaluation_result: ComprehensiveEvaluationResult,
        current_portfolio_metrics: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """緊急事態の評価"""
        
        emergency_conditions = []
        is_emergency = False
        
        # 総合スコア基準
        if evaluation_result.overall_score < self.emergency_thresholds.get("overall_score_limit", 30.0):
            emergency_conditions.append(f"総合スコアが危険レベル: {evaluation_result.overall_score:.1f}")
            is_emergency = True
        
        # リスク調整後スコア基準
        if evaluation_result.risk_adjusted_score < self.emergency_thresholds.get("risk_adjusted_score_limit", 25.0):
            emergency_conditions.append(f"リスク調整後スコアが危険レベル: {evaluation_result.risk_adjusted_score:.1f}")
            is_emergency = True
        
        # ポートフォリオ損失基準
        if current_portfolio_metrics.get("total_return", 0) < self.emergency_thresholds.get("portfolio_loss_limit", -30.0):
            emergency_conditions.append(f"ポートフォリオ損失が限界値を超過: {current_portfolio_metrics['total_return']:.1f}%")
            is_emergency = True
        
        # ドローダウン基準
        if current_portfolio_metrics.get("max_drawdown", 0) > self.emergency_thresholds.get("drawdown_limit", 40.0):
            emergency_conditions.append(f"最大ドローダウンが限界値を超過: {current_portfolio_metrics['max_drawdown']:.1f}%")
            is_emergency = True
        
        # VaR基準
        if current_portfolio_metrics.get("value_at_risk", 0) > self.emergency_thresholds.get("var_limit", 10.0):
            emergency_conditions.append(f"VaRが危険レベル: {current_portfolio_metrics['value_at_risk']:.1f}%")
            is_emergency = True
        
        # 次元別危険状態
        critical_dimensions = [
            score for score in evaluation_result.dimension_scores 
            if score.score < 25.0
        ]
        if critical_dimensions:
            dim_names = [d.dimension_name for d in critical_dimensions]
            emergency_conditions.append(f"クリティカル次元: {', '.join(dim_names)}")
            is_emergency = True
        
        return is_emergency, emergency_conditions
    
    def execute_emergency_fixes(
        self, 
        emergency_conditions: List[str],
        evaluation_result: ComprehensiveEvaluationResult,
        current_portfolio_metrics: Dict[str, float]
    ) -> EmergencyFixResult:
        """緊急修正の実行"""
        
        if not self.fix_config.get("auto_fix_enabled", True):
            self.logger.warning("自動修正が無効化されています")
            return EmergencyFixResult(
                trigger_condition="; ".join(emergency_conditions),
                actions_executed=[],
                actions_pending=[],
                overall_success=False,
                execution_summary="自動修正が無効化されているため実行されませんでした"
            )
        
        # 修正アクションの生成
        fix_actions = self._generate_fix_actions(
            emergency_conditions, evaluation_result, current_portfolio_metrics
        )
        
        # 優先度順にソート
        fix_actions.sort(key=lambda x: self._get_priority_weight(x.priority), reverse=True)
        
        # 修正アクションの実行
        executed_actions = []
        pending_actions = []
        
        for action in fix_actions:
            if action.priority in [FixPriority.CRITICAL, FixPriority.HIGH]:
                success = self._execute_fix_action(action)
                if success:
                    executed_actions.append(action)
                else:
                    pending_actions.append(action)
            else:
                pending_actions.append(action)
        
        # 実行結果の評価
        overall_success = len(executed_actions) > 0 and all(
            action.execution_result == "success" for action in executed_actions
        )
        
        execution_summary = self._generate_execution_summary(
            executed_actions, pending_actions
        )
        
        result = EmergencyFixResult(
            trigger_condition="; ".join(emergency_conditions),
            actions_executed=executed_actions,
            actions_pending=pending_actions,
            overall_success=overall_success,
            execution_summary=execution_summary
        )
        
        # 履歴に記録
        self.fix_history.append(result)
        
        return result
    
    def _generate_fix_actions(
        self, 
        emergency_conditions: List[str],
        evaluation_result: ComprehensiveEvaluationResult,
        current_portfolio_metrics: Dict[str, float]
    ) -> List[FixAction]:
        """修正アクションの生成"""
        
        actions = []
        
        # ポートフォリオ全体の緊急停止判定
        severe_loss = current_portfolio_metrics.get("total_return", 0) < -40.0
        severe_drawdown = current_portfolio_metrics.get("max_drawdown", 0) > 60.0
        
        if severe_loss or severe_drawdown:
            actions.append(FixAction(
                action_id="emergency_stop_001",
                category=FixCategory.EMERGENCY_STOP,
                priority=FixPriority.CRITICAL,
                description="全取引の緊急停止",
                target_parameter="trading_enabled",
                current_value=True,
                recommended_value=False,
                confidence=0.95,
                estimated_impact=0.8
            ))
        
        # ポジションサイズの縮小
        if current_portfolio_metrics.get("max_drawdown", 0) > 25.0:
            current_pos_size = current_portfolio_metrics.get("max_position_size", 0.1)
            new_pos_size = max(0.01, current_pos_size * 0.5)
            actions.append(FixAction(
                action_id="pos_size_001",
                category=FixCategory.POSITION_SIZING,
                priority=FixPriority.HIGH,
                description="ポジションサイズを50%削減",
                target_parameter="max_position_size",
                current_value=current_pos_size,
                recommended_value=new_pos_size,
                confidence=0.85,
                estimated_impact=0.6
            ))
        
        # リスクパラメータの調整
        if current_portfolio_metrics.get("value_at_risk", 0) > 8.0:
            actions.append(FixAction(
                action_id="risk_param_001",
                category=FixCategory.RISK_PARAMETERS,
                priority=FixPriority.HIGH,
                description="ストップロス幅を狭める",
                target_parameter="stop_loss_percent",
                current_value=current_portfolio_metrics.get("stop_loss_percent", 0.05),
                recommended_value=0.02,
                confidence=0.75,
                estimated_impact=0.5
            ))
        
        # 戦略重みの調整
        poor_dimensions = [
            score for score in evaluation_result.dimension_scores 
            if score.score < 40.0
        ]
        if poor_dimensions:
            actions.append(FixAction(
                action_id="strategy_weight_001",
                category=FixCategory.STRATEGY_WEIGHTS,
                priority=FixPriority.MEDIUM,
                description="低パフォーマンス戦略の重み削減",
                target_parameter="strategy_weights",
                current_value="current_weights",
                recommended_value="rebalanced_weights",
                confidence=0.65,
                estimated_impact=0.4
            ))
        
        return actions
    
    def _execute_fix_action(self, action: FixAction) -> bool:
        """個別修正アクションの実行"""
        try:
            action.execution_timestamp = datetime.now()
            
            if action.category == FixCategory.EMERGENCY_STOP:
                result = self._execute_emergency_stop(action)
            elif action.category == FixCategory.POSITION_SIZING:
                result = self._execute_position_sizing_fix(action)
            elif action.category == FixCategory.RISK_PARAMETERS:
                result = self._execute_risk_parameter_fix(action)
            elif action.category == FixCategory.STRATEGY_WEIGHTS:
                result = self._execute_strategy_weight_fix(action)
            else:
                result = False
                action.execution_result = "unknown_category"
            
            if result:
                action.execution_result = "success"
                self.logger.info(f"修正アクション実行成功: {action.action_id}")
            else:
                action.execution_result = "failed"
                self.logger.error(f"修正アクション実行失敗: {action.action_id}")
            
            return result
            
        except Exception as e:
            action.execution_result = f"error: {str(e)}"
            self.logger.error(f"修正アクション実行エラー {action.action_id}: {e}")
            return False
    
    def _execute_emergency_stop(self, action: FixAction) -> bool:
        """緊急停止の実行"""
        # 実際の実装では、取引システムの停止処理を行う
        self.logger.critical(f"【緊急停止】{action.description}")
        # ここでは設定ファイルのフラグ更新をシミュレート
        return True
    
    def _execute_position_sizing_fix(self, action: FixAction) -> bool:
        """ポジションサイズ修正の実行"""
        # 実際の実装では、ポジションサイズ設定の更新を行う
        self.logger.warning(f"【ポジションサイズ修正】{action.description}")
        # 設定更新のシミュレート
        return True
    
    def _execute_risk_parameter_fix(self, action: FixAction) -> bool:
        """リスクパラメータ修正の実行"""
        # 実際の実装では、リスク管理パラメータの更新を行う
        self.logger.warning(f"【リスクパラメータ修正】{action.description}")
        return True
    
    def _execute_strategy_weight_fix(self, action: FixAction) -> bool:
        """戦略重み修正の実行"""
        # 実際の実装では、戦略重みの再配分を行う
        self.logger.info(f"【戦略重み修正】{action.description}")
        return True
    
    def _get_priority_weight(self, priority: FixPriority) -> int:
        """優先度の重み値を取得"""
        weights = {
            FixPriority.CRITICAL: 4,
            FixPriority.HIGH: 3,
            FixPriority.MEDIUM: 2,
            FixPriority.LOW: 1
        }
        return weights.get(priority, 0)
    
    def _generate_execution_summary(
        self, 
        executed_actions: List[FixAction],
        pending_actions: List[FixAction]
    ) -> str:
        """実行結果サマリーの生成"""
        
        summary_parts = []
        
        if executed_actions:
            executed_count = len(executed_actions)
            successful_count = sum(
                1 for action in executed_actions 
                if action.execution_result == "success"
            )
            summary_parts.append(
                f"実行済みアクション: {successful_count}/{executed_count}件成功"
            )
        
        if pending_actions:
            pending_count = len(pending_actions)
            critical_pending = sum(
                1 for action in pending_actions 
                if action.priority == FixPriority.CRITICAL
            )
            summary_parts.append(f"保留アクション: {pending_count}件")
            if critical_pending > 0:
                summary_parts.append(f"（緊急保留: {critical_pending}件）")
        
        return "; ".join(summary_parts) if summary_parts else "アクションなし"
    
    def get_fix_recommendations(
        self, 
        evaluation_result: ComprehensiveEvaluationResult
    ) -> List[str]:
        """手動修正推奨事項の取得"""
        
        recommendations = []
        
        # 次元別推奨事項
        for dimension_score in evaluation_result.dimension_scores:
            if dimension_score.score < 50.0:
                if dimension_score.dimension_name == "profitability":
                    recommendations.append(
                        "収益性改善: パラメータ最適化の実行を推奨します"
                    )
                elif dimension_score.dimension_name == "risk_management":
                    recommendations.append(
                        "リスク管理強化: ポジションサイズとストップロス設定の見直しを推奨します"
                    )
                elif dimension_score.dimension_name == "stability":
                    recommendations.append(
                        "安定性向上: 戦略多様化と切り替えロジックの改善を推奨します"
                    )
        
        # 全体スコア基準の推奨事項
        if evaluation_result.overall_score < 40.0:
            recommendations.append(
                "緊急対応: システム全体の見直しとリスク管理の強化が必要です"
            )
        elif evaluation_result.overall_score < 60.0:
            recommendations.append(
                "改善対応: 主要パフォーマンス指標の向上に焦点を当ててください"
            )
        
        return recommendations
    
    def save_fix_config(self) -> bool:
        """緊急修正設定の保存"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.fix_config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"緊急修正設定の保存エラー: {e}")
            return False
