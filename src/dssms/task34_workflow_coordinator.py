"""
DSSMS Task 3.4 ワークフローコーディネーター
Task 3.4: パフォーマンス目標達成確認の統合ワークフロー管理
"""
import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from .performance_target_manager import PerformanceTargetManager, TargetResult, TargetPhase
from .comprehensive_evaluator import ComprehensiveEvaluator, ComprehensiveEvaluationResult
from .emergency_fix_coordinator import EmergencyFixCoordinator, EmergencyFixResult
from .performance_achievement_reporter import PerformanceAchievementReporter, ReportConfig

@dataclass
class Task34WorkflowConfig:
    """Task 3.4 ワークフロー設定"""
    enable_auto_phase_transition: bool = True
    enable_emergency_fixes: bool = True
    enable_detailed_reporting: bool = True
    report_formats: Optional[List[str]] = None
    performance_data_source: str = "dssms_performance_calculator"
    
    def __post_init__(self):
        if self.report_formats is None:
            self.report_formats = ['excel', 'json', 'text']

@dataclass
class Task34ExecutionResult:
    """Task 3.4 実行結果"""
    execution_id: str
    start_time: datetime
    end_time: datetime
    success: bool
    target_results: List[TargetResult]
    evaluation_result: ComprehensiveEvaluationResult
    emergency_fix_result: Optional[EmergencyFixResult]
    report_files: Dict[str, str]
    phase_transition_recommended: bool
    next_recommended_phase: Optional[TargetPhase]
    execution_summary: str
    errors: List[str]

class Task34WorkflowCoordinator:
    """DSSMS Task 3.4 統合ワークフロー管理システム"""
    
    def __init__(self, config: Optional[Task34WorkflowConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or Task34WorkflowConfig()
        
        # コンポーネント初期化
        self.target_manager = PerformanceTargetManager()
        self.evaluator = ComprehensiveEvaluator()
        self.emergency_coordinator = EmergencyFixCoordinator()
        self.reporter = PerformanceAchievementReporter()
        
        # 実行履歴
        self.execution_history: List[Task34ExecutionResult] = []
        
        self.logger.info("Task 3.4 ワークフローコーディネーター初期化完了")
    
    def execute_full_workflow(
        self, 
        performance_data: Dict[str, float],
        risk_metrics: Optional[Dict[str, float]] = None,
        execution_id: Optional[str] = None
    ) -> Task34ExecutionResult:
        """完全ワークフローの実行"""
        
        if execution_id is None:
            execution_id = f"task34_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        start_time = datetime.now()
        errors = []
        
        self.logger.info(f"Task 3.4 フルワークフロー開始: {execution_id}")
        
        try:
            # Step 1: パフォーマンス目標評価
            self.logger.info("Step 1: パフォーマンス目標評価")
            target_results = self.target_manager.evaluate_metrics(performance_data)
            
            if not target_results:
                errors.append("目標評価結果が取得できませんでした")
                return self._create_error_result(execution_id, start_time, errors)
            
            # Step 2: 総合評価実行
            self.logger.info("Step 2: 総合パフォーマンス評価")
            evaluation_result = self.evaluator.evaluate_comprehensive_performance(
                target_results, performance_data, risk_metrics
            )
            
            # Step 3: 緊急事態評価と修正
            emergency_fix_result = None
            if self.config.enable_emergency_fixes:
                self.logger.info("Step 3: 緊急事態評価と修正")
                is_emergency, emergency_conditions = self.emergency_coordinator.evaluate_emergency_conditions(
                    evaluation_result, performance_data
                )
                
                if is_emergency:
                    self.logger.warning(f"緊急事態検出: {emergency_conditions}")
                    emergency_fix_result = self.emergency_coordinator.execute_emergency_fixes(
                        emergency_conditions, evaluation_result, performance_data
                    )
            
            # Step 4: フェーズ移行評価
            phase_transition_recommended = False
            next_recommended_phase = None
            if self.config.enable_auto_phase_transition:
                self.logger.info("Step 4: フェーズ移行評価")
                phase_transition_recommended, transition_message = self.target_manager.suggest_next_phase(target_results)
                if phase_transition_recommended:
                    # 次のフェーズを特定
                    progression = ["emergency", "basic", "optimization"]
                    current_index = progression.index(self.target_manager.current_phase.value)
                    if current_index < len(progression) - 1:
                        next_recommended_phase = TargetPhase(progression[current_index + 1])
                    self.logger.info(transition_message)
            
            # Step 5: レポート生成
            report_files = {}
            if self.config.enable_detailed_reporting:
                self.logger.info("Step 5: 詳細レポート生成")
                report_config = ReportConfig(
                    output_directory=self.reporter.output_dir,
                    formats=self.config.report_formats or ['excel', 'json', 'text'],
                    include_charts=True,
                    detailed_analysis=True,
                    executive_summary=True
                )
                
                report_files = self.reporter.generate_comprehensive_report(
                    target_results, evaluation_result, emergency_fix_result, report_config
                )
            
            # 実行サマリー生成
            execution_summary = self._generate_execution_summary(
                target_results, evaluation_result, emergency_fix_result, 
                phase_transition_recommended, next_recommended_phase
            )
            
            end_time = datetime.now()
            
            # 結果オブジェクト作成
            result = Task34ExecutionResult(
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                success=True,
                target_results=target_results,
                evaluation_result=evaluation_result,
                emergency_fix_result=emergency_fix_result,
                report_files=report_files,
                phase_transition_recommended=phase_transition_recommended,
                next_recommended_phase=next_recommended_phase,
                execution_summary=execution_summary,
                errors=errors
            )
            
            # 履歴に追加
            self.execution_history.append(result)
            
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"Task 3.4 フルワークフロー完了: {execution_id} (実行時間: {execution_time:.2f}秒)")
            
            return result
            
        except Exception as e:
            errors.append(f"ワークフロー実行エラー: {str(e)}")
            self.logger.error(f"Task 3.4 ワークフロー実行エラー: {e}")
            return self._create_error_result(execution_id, start_time, errors)
    
    def execute_monitoring_workflow(
        self, 
        performance_data: Dict[str, float],
        monitoring_interval_minutes: int = 30
    ) -> Task34ExecutionResult:
        """監視モードでのワークフロー実行"""
        
        execution_id = f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 軽量版の評価のみ実行
        config = Task34WorkflowConfig(
            enable_auto_phase_transition=False,
            enable_emergency_fixes=True,
            enable_detailed_reporting=False,
            report_formats=['json']
        )
        
        # 一時的に設定を変更
        original_config = self.config
        self.config = config
        
        try:
            result = self.execute_full_workflow(performance_data, execution_id=execution_id)
            return result
        finally:
            # 設定を復元
            self.config = original_config
    
    def get_current_phase_status(self) -> Dict[str, Any]:
        """現在のフェーズ状況取得"""
        return {
            "current_phase": self.target_manager.current_phase.value,
            "phase_targets": self.target_manager._get_current_phase_targets(),
            "last_execution": self.execution_history[-1] if self.execution_history else None,
            "execution_count": len(self.execution_history)
        }
    
    def transition_to_phase(self, new_phase: TargetPhase) -> bool:
        """フェーズの手動移行"""
        try:
            success = self.target_manager.update_current_phase(new_phase)
            if success:
                self.logger.info(f"フェーズを '{new_phase.value}' に手動移行しました")
            return success
        except Exception as e:
            self.logger.error(f"フェーズ移行エラー: {e}")
            return False
    
    def get_performance_trends(self, last_n_executions: int = 10) -> Dict[str, Any]:
        """パフォーマンストレンド分析"""
        
        if len(self.execution_history) < 2:
            return {"message": "十分な履歴データがありません"}
        
        recent_executions = self.execution_history[-last_n_executions:]
        
        # 総合スコアのトレンド
        overall_scores = [e.evaluation_result.overall_score for e in recent_executions]
        risk_adjusted_scores = [e.evaluation_result.risk_adjusted_score for e in recent_executions]
        
        # 緊急修正頻度
        emergency_fix_count = sum(1 for e in recent_executions if e.emergency_fix_result is not None)
        
        # フェーズ移行頻度
        phase_transition_count = sum(1 for e in recent_executions if e.phase_transition_recommended)
        
        return {
            "analysis_period": {
                "executions_analyzed": len(recent_executions),
                "start_time": recent_executions[0].start_time.isoformat(),
                "end_time": recent_executions[-1].end_time.isoformat()
            },
            "overall_score_trend": {
                "current": overall_scores[-1],
                "average": sum(overall_scores) / len(overall_scores),
                "trend": "improving" if overall_scores[-1] > overall_scores[0] else "declining",
                "volatility": max(overall_scores) - min(overall_scores)
            },
            "risk_adjusted_score_trend": {
                "current": risk_adjusted_scores[-1],
                "average": sum(risk_adjusted_scores) / len(risk_adjusted_scores),
                "trend": "improving" if risk_adjusted_scores[-1] > risk_adjusted_scores[0] else "declining"
            },
            "operational_metrics": {
                "emergency_fix_rate": emergency_fix_count / len(recent_executions),
                "phase_transition_rate": phase_transition_count / len(recent_executions),
                "average_execution_time": sum(
                    (e.end_time - e.start_time).total_seconds() 
                    for e in recent_executions
                ) / len(recent_executions)
            }
        }
    
    def export_execution_history(self, output_path: Optional[str] = None) -> str:
        """実行履歴のエクスポート"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/dssms_reports/task34_execution_history_{timestamp}.json"
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            export_data = {
                "export_metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_executions": len(self.execution_history),
                    "workflow_config": {
                        "enable_auto_phase_transition": self.config.enable_auto_phase_transition,
                        "enable_emergency_fixes": self.config.enable_emergency_fixes,
                        "enable_detailed_reporting": self.config.enable_detailed_reporting
                    }
                },
                "execution_history": [
                    {
                        "execution_id": result.execution_id,
                        "start_time": result.start_time.isoformat(),
                        "end_time": result.end_time.isoformat(),
                        "success": result.success,
                        "overall_score": result.evaluation_result.overall_score,
                        "risk_adjusted_score": result.evaluation_result.risk_adjusted_score,
                        "emergency_fix_executed": result.emergency_fix_result is not None,
                        "phase_transition_recommended": result.phase_transition_recommended,
                        "execution_summary": result.execution_summary,
                        "errors": result.errors
                    }
                    for result in self.execution_history
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"実行履歴エクスポート完了: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"実行履歴エクスポートエラー: {e}")
            return ""
    
    def _generate_execution_summary(
        self,
        target_results: List[TargetResult],
        evaluation_result: ComprehensiveEvaluationResult,
        emergency_fix_result: Optional[EmergencyFixResult],
        phase_transition_recommended: bool,
        next_recommended_phase: Optional[TargetPhase]
    ) -> str:
        """実行サマリーの生成"""
        
        summary_parts = []
        
        # 基本評価結果
        success_rate = sum(1 for r in target_results if r.achievement_level.value != "failed") / len(target_results)
        summary_parts.append(f"目標達成率: {success_rate:.1%} ({len(target_results)}指標中)")
        summary_parts.append(f"総合スコア: {evaluation_result.overall_score:.1f}")
        summary_parts.append(f"リスク調整後スコア: {evaluation_result.risk_adjusted_score:.1f}")
        
        # 緊急修正情報
        if emergency_fix_result:
            if emergency_fix_result.overall_success:
                summary_parts.append(f"緊急修正実行: 成功 ({len(emergency_fix_result.actions_executed)}件)")
            else:
                summary_parts.append(f"緊急修正実行: 失敗/部分的成功")
        
        # フェーズ移行情報
        if phase_transition_recommended and next_recommended_phase:
            summary_parts.append(f"フェーズ移行推奨: {next_recommended_phase.value}")
        
        # アラート数
        if evaluation_result.alerts:
            summary_parts.append(f"アラート: {len(evaluation_result.alerts)}件")
        
        return "; ".join(summary_parts)
    
    def _create_error_result(
        self, 
        execution_id: str, 
        start_time: datetime, 
        errors: List[str]
    ) -> Task34ExecutionResult:
        """エラー時の結果オブジェクト作成"""
        
        from .comprehensive_evaluator import ComprehensiveEvaluationResult
        
        return Task34ExecutionResult(
            execution_id=execution_id,
            start_time=start_time,
            end_time=datetime.now(),
            success=False,
            target_results=[],
            evaluation_result=ComprehensiveEvaluationResult(
                overall_score=0.0,
                dimension_scores=[],
                risk_adjusted_score=0.0,
                confidence_level=0.0,
                evaluation_timestamp=datetime.now(),
                recommendations=["システムエラーにより評価を完了できませんでした"],
                alerts=["ワークフロー実行に失敗しました"]
            ),
            emergency_fix_result=None,
            report_files={},
            phase_transition_recommended=False,
            next_recommended_phase=None,
            execution_summary="実行失敗: " + "; ".join(errors),
            errors=errors
        )
    
    def cleanup_old_executions(self, keep_last_n: int = 100):
        """古い実行履歴のクリーンアップ"""
        if len(self.execution_history) > keep_last_n:
            removed_count = len(self.execution_history) - keep_last_n
            self.execution_history = self.execution_history[-keep_last_n:]
            self.logger.info(f"古い実行履歴をクリーンアップ: {removed_count}件削除")
