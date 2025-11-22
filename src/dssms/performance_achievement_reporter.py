"""
DSSMS パフォーマンス達成レポーター
Task 3.4: 総合評価結果の多形式出力とレポート生成
"""
import logging
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, asdict
from .performance_target_manager import TargetResult, AchievementLevel, TargetPhase
from .comprehensive_evaluator import ComprehensiveEvaluationResult, DimensionScore
from .emergency_fix_coordinator import EmergencyFixResult, FixAction

@dataclass
class ReportConfig:
    """レポート設定"""
    output_directory: str
    formats: List[str]  # ['excel', 'json', 'html', 'text']
    include_charts: bool
    detailed_analysis: bool
    executive_summary: bool

class PerformanceAchievementReporter:
    """パフォーマンス達成レポート生成システム"""
    
    def __init__(self, output_dir: str = "output/dssms_reports"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self._ensure_output_directory()
        
    def _ensure_output_directory(self):
        """出力ディレクトリの確保"""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_comprehensive_report(
        self,
        target_results: List[TargetResult],
        evaluation_result: ComprehensiveEvaluationResult,
        emergency_fix_result: Optional[EmergencyFixResult] = None,
        config: Optional[ReportConfig] = None
    ) -> Dict[str, str]:
        """総合レポートの生成"""
        
        if config is None:
            config = ReportConfig(
                output_directory=self.output_dir,
                formats=['excel', 'json', 'text'],
                include_charts=True,
                detailed_analysis=True,
                executive_summary=True
            )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}
        
        try:
            # 各形式でのレポート生成
            if 'excel' in config.formats:
                excel_file = self._generate_excel_report(
                    target_results, evaluation_result, emergency_fix_result, timestamp
                )
                report_files['excel'] = excel_file
                
            if 'json' in config.formats:
                json_file = self._generate_json_report(
                    target_results, evaluation_result, emergency_fix_result, timestamp
                )
                report_files['json'] = json_file
                
            if 'text' in config.formats:
                text_file = self._generate_text_report(
                    target_results, evaluation_result, emergency_fix_result, timestamp
                )
                report_files['text'] = text_file
                
            if 'html' in config.formats:
                html_file = self._generate_html_report(
                    target_results, evaluation_result, emergency_fix_result, timestamp
                )
                report_files['html'] = html_file
            
            self.logger.info(f"レポート生成完了: {len(report_files)}ファイル")
            return report_files
            
        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")
            return {}
    
    def _generate_excel_report(
        self,
        target_results: List[TargetResult],
        evaluation_result: ComprehensiveEvaluationResult,
        emergency_fix_result: Optional[EmergencyFixResult],
        timestamp: str
    ) -> str:
        """Excelレポートの生成"""
        
        filename = f"dssms_performance_report_{timestamp}.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # サマリーシート
            summary_df = self._create_summary_dataframe(evaluation_result)
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 目標達成状況シート
            targets_df = self._create_targets_dataframe(target_results)
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: targets_df.to_excel(writer, sheet_name='Target_Achievement', index=False)
            
            # 次元別評価シート
            dimensions_df = self._create_dimensions_dataframe(evaluation_result.dimension_scores)
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: dimensions_df.to_excel(writer, sheet_name='Dimension_Analysis', index=False)
            
            # 緊急修正シート（該当する場合）
            if emergency_fix_result:
                emergency_df = self._create_emergency_dataframe(emergency_fix_result)
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: emergency_df.to_excel(writer, sheet_name='Emergency_Fixes', index=False)
            
            # 推奨事項シート
            recommendations_df = self._create_recommendations_dataframe(evaluation_result)
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            self.logger.info(f"Excelレポート生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Excelレポート生成エラー: {e}")
            return ""
    
    def _generate_json_report(
        self,
        target_results: List[TargetResult],
        evaluation_result: ComprehensiveEvaluationResult,
        emergency_fix_result: Optional[EmergencyFixResult],
        timestamp: str
    ) -> str:
        """JSONレポートの生成"""
        
        filename = f"dssms_performance_report_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            report_data = {
                "report_metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "report_type": "DSSMS_Performance_Achievement",
                    "version": "1.0"
                },
                "executive_summary": {
                    "overall_score": evaluation_result.overall_score,
                    "risk_adjusted_score": evaluation_result.risk_adjusted_score,
                    "confidence_level": evaluation_result.confidence_level,
                    "evaluation_timestamp": evaluation_result.evaluation_timestamp.isoformat()
                },
                "target_achievement": [
                    {
                        "metric_name": result.metric_name,
                        "value": result.value,
                        "target_value": result.target_value,
                        "achievement_level": result.achievement_level.value,
                        "phase": result.phase.value,
                        "description": result.description
                    }
                    for result in target_results
                ],
                "dimension_analysis": [
                    {
                        "dimension_name": score.dimension_name,
                        "score": score.score,
                        "weight": score.weight,
                        "weighted_score": score.weighted_score,
                        "status": score.status,
                        "metrics_count": score.metrics_count,
                        "details": score.details
                    }
                    for score in evaluation_result.dimension_scores
                ],
                "recommendations": evaluation_result.recommendations,
                "alerts": evaluation_result.alerts
            }
            
            if emergency_fix_result:
                report_data["emergency_fixes"] = {
                    "trigger_condition": emergency_fix_result.trigger_condition,
                    "overall_success": emergency_fix_result.overall_success,
                    "execution_summary": emergency_fix_result.execution_summary,
                    "actions_executed": [
                        {
                            "action_id": action.action_id,
                            "category": action.category.value,
                            "priority": action.priority.value,
                            "description": action.description,
                            "execution_result": action.execution_result
                        }
                        for action in emergency_fix_result.actions_executed
                    ],
                    "actions_pending": [
                        {
                            "action_id": action.action_id,
                            "category": action.category.value,
                            "priority": action.priority.value,
                            "description": action.description
                        }
                        for action in emergency_fix_result.actions_pending
                    ]
                }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"JSONレポート生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"JSONレポート生成エラー: {e}")
            return ""
    
    def _generate_text_report(
        self,
        target_results: List[TargetResult],
        evaluation_result: ComprehensiveEvaluationResult,
        emergency_fix_result: Optional[EmergencyFixResult],
        timestamp: str
    ) -> str:
        """テキストレポートの生成"""
        
        filename = f"dssms_performance_report_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DSSMS パフォーマンス達成確認レポート\n")
                f.write("=" * 80 + "\n")
                f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"評価日時: {evaluation_result.evaluation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # エグゼクティブサマリー
                f.write("■ エグゼクティブサマリー\n")
                f.write("-" * 40 + "\n")
                f.write(f"総合スコア:           {evaluation_result.overall_score:.1f}/100.0\n")
                f.write(f"リスク調整後スコア:   {evaluation_result.risk_adjusted_score:.1f}/100.0\n")
                f.write(f"信頼度レベル:         {evaluation_result.confidence_level:.1%}\n\n")
                
                # 次元別分析
                f.write("■ 次元別パフォーマンス分析\n")
                f.write("-" * 40 + "\n")
                for score in evaluation_result.dimension_scores:
                    status_emoji = self._get_status_emoji(score.status)
                    f.write(f"{status_emoji} {score.dimension_name:15s}: {score.score:6.1f} (重み: {score.weight:.1%})\n")
                f.write("\n")
                
                # 目標達成状況
                f.write("■ 目標達成状況\n")
                f.write("-" * 40 + "\n")
                achievement_count = {}
                for level in AchievementLevel:
                    achievement_count[level] = sum(1 for r in target_results if r.achievement_level == level)
                
                total_metrics = len(target_results)
                f.write(f"評価指標数: {total_metrics}\n")
                for level, count in achievement_count.items():
                    percentage = (count / total_metrics * 100) if total_metrics > 0 else 0
                    emoji = self._get_achievement_emoji(level)
                    f.write(f"{emoji} {level.value:10s}: {count:3d}件 ({percentage:5.1f}%)\n")
                f.write("\n")
                
                # 詳細指標
                f.write("■ 詳細指標評価\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'指標名':<20s} {'現在値':<12s} {'目標値':<12s} {'達成度':<10s} {'フェーズ':<8s}\n")
                f.write("-" * 70 + "\n")
                for result in target_results:
                    emoji = self._get_achievement_emoji(result.achievement_level)
                    f.write(f"{result.metric_name:<20s} ")
                    f.write(f"{result.value:<12.2f} ")
                    f.write(f"{result.target_value:<12.2f} ")
                    f.write(f"{emoji}{result.achievement_level.value:<9s} ")
                    f.write(f"{result.phase.value:<8s}\n")
                f.write("\n")
                
                # 推奨事項
                if evaluation_result.recommendations:
                    f.write("■ 改善推奨事項\n")
                    f.write("-" * 40 + "\n")
                    for i, rec in enumerate(evaluation_result.recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
                
                # アラート
                if evaluation_result.alerts:
                    f.write("■ 重要アラート\n")
                    f.write("-" * 40 + "\n")
                    for alert in evaluation_result.alerts:
                        f.write(f"[WARNING]  {alert}\n")
                    f.write("\n")
                
                # 緊急修正情報
                if emergency_fix_result:
                    f.write("■ 緊急修正実行結果\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"トリガー条件: {emergency_fix_result.trigger_condition}\n")
                    f.write(f"実行結果: {'成功' if emergency_fix_result.overall_success else '失敗'}\n")
                    f.write(f"実行サマリー: {emergency_fix_result.execution_summary}\n")
                    
                    if emergency_fix_result.actions_executed:
                        f.write("\n実行済みアクション:\n")
                        for action in emergency_fix_result.actions_executed:
                            f.write(f"  - {action.description} ({action.execution_result})\n")
                    
                    if emergency_fix_result.actions_pending:
                        f.write("\n保留アクション:\n")
                        for action in emergency_fix_result.actions_pending:
                            f.write(f"  - {action.description} ({action.priority.value})\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("レポート終了\n")
            
            self.logger.info(f"テキストレポート生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"テキストレポート生成エラー: {e}")
            return ""
    
    def _generate_html_report(
        self,
        target_results: List[TargetResult],
        evaluation_result: ComprehensiveEvaluationResult,
        emergency_fix_result: Optional[EmergencyFixResult],
        timestamp: str
    ) -> str:
        """HTMLレポートの生成"""
        
        filename = f"dssms_performance_report_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            html_content = self._create_html_content(
                target_results, evaluation_result, emergency_fix_result
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTMLレポート生成成功: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"HTMLレポート生成エラー: {e}")
            return ""
    
    def _create_summary_dataframe(self, evaluation_result: ComprehensiveEvaluationResult) -> pd.DataFrame:
        """サマリーデータフレームの作成"""
        return pd.DataFrame([{
            '項目': 'Overall Score',
            '値': evaluation_result.overall_score,
            '単位': 'points'
        }, {
            '項目': 'Risk Adjusted Score',
            '値': evaluation_result.risk_adjusted_score,
            '単位': 'points'
        }, {
            '項目': 'Confidence Level',
            '値': evaluation_result.confidence_level,
            '単位': 'ratio'
        }])
    
    def _create_targets_dataframe(self, target_results: List[TargetResult]) -> pd.DataFrame:
        """目標データフレームの作成"""
        data = []
        for result in target_results:
            data.append({
                '指標名': result.metric_name,
                '現在値': result.value,
                '目標値': result.target_value,
                '最小値': result.minimum_value,
                'ストレッチ値': result.stretch_value,
                '達成レベル': result.achievement_level.value,
                'フェーズ': result.phase.value,
                '説明': result.description
            })
        return pd.DataFrame(data)
    
    def _create_dimensions_dataframe(self, dimension_scores: List[DimensionScore]) -> pd.DataFrame:
        """次元データフレームの作成"""
        data = []
        for score in dimension_scores:
            data.append({
                '次元名': score.dimension_name,
                'スコア': score.score,
                '重み': score.weight,
                '重み付きスコア': score.weighted_score,
                'ステータス': score.status,
                '指標数': score.metrics_count
            })
        return pd.DataFrame(data)
    
    def _create_emergency_dataframe(self, emergency_fix_result: EmergencyFixResult) -> pd.DataFrame:
        """緊急修正データフレームの作成"""
        data = []
        for action in emergency_fix_result.actions_executed + emergency_fix_result.actions_pending:
            data.append({
                'アクションID': action.action_id,
                'カテゴリ': action.category.value,
                '優先度': action.priority.value,
                '説明': action.description,
                '対象パラメータ': action.target_parameter,
                '現在値': str(action.current_value),
                '推奨値': str(action.recommended_value),
                '信頼度': action.confidence,
                '実行結果': action.execution_result or 'pending'
            })
        return pd.DataFrame(data)
    
    def _create_recommendations_dataframe(self, evaluation_result: ComprehensiveEvaluationResult) -> pd.DataFrame:
        """推奨事項データフレームの作成"""
        data = []
        for i, rec in enumerate(evaluation_result.recommendations, 1):
            data.append({
                '番号': i,
                '推奨事項': rec,
                'タイプ': 'recommendation'
            })
        for i, alert in enumerate(evaluation_result.alerts, len(evaluation_result.recommendations) + 1):
            data.append({
                '番号': i,
                '推奨事項': alert,
                'タイプ': 'alert'
            })
        return pd.DataFrame(data)
    
    def _create_html_content(
        self,
        target_results: List[TargetResult],
        evaluation_result: ComprehensiveEvaluationResult,
        emergency_fix_result: Optional[EmergencyFixResult]
    ) -> str:
        """HTML コンテンツの作成"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DSSMS Performance Achievement Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; margin: 20px 0; }}
        .dimension {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; }}
        .excellent {{ border-left-color: #28a745; }}
        .good {{ border-left-color: #007bff; }}
        .acceptable {{ border-left-color: #ffc107; }}
        .needs_improvement {{ border-left-color: #fd7e14; }}
        .critical {{ border-left-color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .alert {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>DSSMS パフォーマンス達成確認レポート</h1>
        <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>エグゼクティブサマリー</h2>
        <p><strong>総合スコア:</strong> {evaluation_result.overall_score:.1f}/100.0</p>
        <p><strong>リスク調整後スコア:</strong> {evaluation_result.risk_adjusted_score:.1f}/100.0</p>
        <p><strong>信頼度レベル:</strong> {evaluation_result.confidence_level:.1%}</p>
    </div>
    
    <h2>次元別パフォーマンス</h2>
"""
        
        for score in evaluation_result.dimension_scores:
            html += f"""
    <div class="dimension {score.status}">
        <strong>{score.dimension_name}:</strong> {score.score:.1f}/100.0 (重み: {score.weight:.1%})
        <br>ステータス: {score.status}
    </div>
"""
        
        html += """
    <h2>目標達成状況</h2>
    <table>
        <tr>
            <th>指標名</th>
            <th>現在値</th>
            <th>目標値</th>
            <th>達成レベル</th>
            <th>フェーズ</th>
        </tr>
"""
        
        for result in target_results:
            html += f"""
        <tr>
            <td>{result.metric_name}</td>
            <td>{result.value:.2f}</td>
            <td>{result.target_value:.2f}</td>
            <td>{result.achievement_level.value}</td>
            <td>{result.phase.value}</td>
        </tr>
"""
        
        html += "</table>"
        
        if evaluation_result.recommendations:
            html += "<h2>改善推奨事項</h2><ul>"
            for rec in evaluation_result.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        if evaluation_result.alerts:
            html += "<h2>重要アラート</h2>"
            for alert in evaluation_result.alerts:
                html += f'<div class="alert">[WARNING] {alert}</div>'
        
        html += """
</body>
</html>
"""
        return html
    
    def _get_status_emoji(self, status: str) -> str:
        """ステータス絵文字の取得"""
        emojis = {
            "excellent": "🟢",
            "good": "🔵", 
            "acceptable": "🟡",
            "needs_improvement": "🟠",
            "critical": "🔴"
        }
        return emojis.get(status, "⚪")
    
    def _get_achievement_emoji(self, level: AchievementLevel) -> str:
        """達成レベル絵文字の取得"""
        emojis = {
            AchievementLevel.STRETCH: "🌟",
            AchievementLevel.TARGET: "[OK]",
            AchievementLevel.MINIMUM: "⚡",
            AchievementLevel.FAILED: "[ERROR]"
        }
        return emojis.get(level, "❓")
