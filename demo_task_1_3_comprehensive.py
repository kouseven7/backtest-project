"""
DSSMS Task 1.3: 統合実行デモスクリプト
Dynamic Stock Selection Multi-Strategy System - Comprehensive Demo Script

Task 1.3: クイック修正版の作成と動作確認

統合実行フロー:
1. 問題箇所特定・修正実行
2. 統合マネージャーでハイブリッド統合
3. 段階的検証実行
4. 総合評価・レポート生成

すべてのコンポーネントを統合し、動作確認を実行
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
import traceback
import time

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

# Task 1.3コンポーネントインポート
try:
    from src.dssms.dssms_quick_fix_integration_manager import DSSMSQuickFixIntegrationManager
    from src.dssms.dssms_staged_validator import DSSMSStagedValidator
    from src.dssms.dssms_issue_detector_fixer import DSSMSIssueDetectorFixer
except ImportError as e:
    print(f"[WARNING] Task 1.3コンポーネントインポートエラー: {e}")
    print("個別にコンポーネントを実行してください")

class DemoPhase(Enum):
    """デモフェーズ"""
    SETUP = "setup"                      # セットアップ
    ISSUE_DETECTION = "issue_detection"  # 問題検出
    INTEGRATION = "integration"          # 統合実行
    VALIDATION = "validation"            # 検証実行
    REPORTING = "reporting"              # レポート生成
    CLEANUP = "cleanup"                  # クリーンアップ

@dataclass
class DemoResult:
    """デモ結果"""
    phase: DemoPhase
    success: bool
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class ComprehensiveDemoReport:
    """包括デモレポート"""
    overall_success: bool
    total_execution_time: float
    phase_results: List[DemoResult]
    system_health: float
    integration_score: float
    validation_score: float
    final_recommendations: List[str]
    summary: Dict[str, Any]

class DSSMSTask13ComprehensiveDemo:
    """
    DSSMS Task 1.3 包括デモシステム
    
    Task 1.3のすべてのコンポーネントを統合し、包括的な動作確認を実行
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Args:
            config: デモ実行設定
        """
        self.logger = self._setup_logger()
        self.config = config or self._get_default_config()
        
        # 実行履歴
        self.phase_results: List[DemoResult] = []
        self.start_time = datetime.now()
        
        # コンポーネント
        self.issue_detector: Optional[DSSMSIssueDetectorFixer] = None
        self.integration_manager: Optional[DSSMSQuickFixIntegrationManager] = None
        self.staged_validator: Optional[DSSMSStagedValidator] = None
        
        self.logger.info("DSSMS Task 1.3 包括デモシステム初期化完了")
    
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger('dssms.comprehensive_demo')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定取得"""
        return {
            'run_issue_detection': True,
            'run_integration': True,
            'run_validation': True,
            'detailed_logging': True,
            'save_results': True,
            'continue_on_error': True,
            'timeout_seconds': 300
        }
    
    def run_comprehensive_demo(self) -> ComprehensiveDemoReport:
        """
        包括デモ実行
        
        Returns:
            ComprehensiveDemoReport: 包括デモレポート
        """
        self.logger.info("DSSMS Task 1.3 包括デモ開始")
        
        try:
            # Phase 1: セットアップ
            self._execute_phase(DemoPhase.SETUP, self._run_setup_phase)
            
            # Phase 2: 問題検出・修正
            if self.config['run_issue_detection']:
                self._execute_phase(DemoPhase.ISSUE_DETECTION, self._run_issue_detection_phase)
            
            # Phase 3: 統合実行
            if self.config['run_integration']:
                self._execute_phase(DemoPhase.INTEGRATION, self._run_integration_phase)
            
            # Phase 4: 検証実行
            if self.config['run_validation']:
                self._execute_phase(DemoPhase.VALIDATION, self._run_validation_phase)
            
            # Phase 5: レポート生成
            self._execute_phase(DemoPhase.REPORTING, self._run_reporting_phase)
            
            # Phase 6: クリーンアップ
            self._execute_phase(DemoPhase.CLEANUP, self._run_cleanup_phase)
            
            # 総合評価
            report = self._generate_comprehensive_report()
            
            self.logger.info(f"包括デモ完了: 成功={report.overall_success}, 時間={report.total_execution_time:.2f}s")
            return report
            
        except Exception as e:
            self.logger.error(f"包括デモエラー: {e}")
            return self._generate_error_report(e)
    
    def _execute_phase(self, phase: DemoPhase, phase_function) -> DemoResult:
        """フェーズ実行"""
        self.logger.info(f"フェーズ開始: {phase.value}")
        start_time = time.time()
        
        try:
            details = phase_function()
            execution_time = time.time() - start_time
            
            result = DemoResult(
                phase=phase,
                success=True,
                execution_time=execution_time,
                details=details
            )
            
            self.phase_results.append(result)
            self.logger.info(f"フェーズ完了: {phase.value} ({execution_time:.2f}s)")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            result = DemoResult(
                phase=phase,
                success=False,
                execution_time=execution_time,
                details={'error': error_msg},
                error_message=error_msg
            )
            
            self.phase_results.append(result)
            self.logger.error(f"フェーズエラー: {phase.value} - {error_msg}")
            
            if not self.config['continue_on_error']:
                raise
            
            return result
    
    def _run_setup_phase(self) -> Dict[str, Any]:
        """セットアップフェーズ実行"""
        self.logger.info("セットアップフェーズ開始")
        
        # 必要ディレクトリ作成
        directories = [
            'backtest_results',
            'backtest_results/dssms_results',
            'backtest_results/task_1_3_results',
            'logs'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # 環境チェック
        import_status = {}
        required_modules = [
            'pandas', 'numpy', 'yfinance', 'matplotlib', 'seaborn'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
                import_status[module] = True
            except ImportError:
                import_status[module] = False
        
        return {
            'directories_created': len(directories),
            'import_status': import_status,
            'python_version': sys.version,
            'start_time': self.start_time.isoformat()
        }
    
    def _run_issue_detection_phase(self) -> Dict[str, Any]:
        """問題検出フェーズ実行"""
        self.logger.info("問題検出フェーズ開始")
        
        # 問題検出・修正システム初期化
        self.issue_detector = DSSMSIssueDetectorFixer({
            'auto_fix_enabled': True,
            'critical_fix_only': False
        })
        
        # 包括的問題検出・修正実行
        health_report = self.issue_detector.run_comprehensive_issue_detection()
        
        return {
            'system_health': health_report.overall_health,
            'detected_issues': len(health_report.detected_issues),
            'fixed_issues': len([f for f in health_report.applied_fixes if f.status.value == 'fixed']),
            'remaining_issues': len(health_report.remaining_issues),
            'execution_time': health_report.execution_summary.get('execution_time', 0)
        }
    
    def _run_integration_phase(self) -> Dict[str, Any]:
        """統合フェーズ実行"""
        self.logger.info("統合フェーズ開始")
        
        # 統合マネージャー初期化
        self.integration_manager = DSSMSQuickFixIntegrationManager({
            'integration_mode': 'hybrid',
            'enable_task_1_1': True,
            'enable_task_1_2': True,
            'auto_fix_enabled': True
        })
        
        # ハイブリッド統合実行
        integration_report = self.integration_manager.run_comprehensive_integration()
        
        return {
            'integration_success': integration_report.overall_success,
            'task_1_1_status': integration_report.task_1_1_results.get('success', False),
            'task_1_2_status': integration_report.task_1_2_results.get('success', False),
            'performance_score': integration_report.performance_metrics.get('integration_score', 0),
            'execution_time': integration_report.execution_summary.get('total_time', 0)
        }
    
    def _run_validation_phase(self) -> Dict[str, Any]:
        """検証フェーズ実行"""
        self.logger.info("検証フェーズ開始")
        
        # 段階的検証システム初期化
        self.staged_validator = DSSMSStagedValidator({
            'target_success_rates': {
                'basic': 0.8,
                'integration': 0.75,
                'performance': 0.7
            },
            'enable_detailed_analysis': True
        })
        
        # 包括的検証実行
        validation_report = self.staged_validator.run_comprehensive_validation()
        
        return {
            'overall_score': validation_report.overall_score,
            'basic_score': validation_report.basic_results.get('score', 0),
            'integration_score': validation_report.integration_results.get('score', 0),
            'performance_score': validation_report.performance_results.get('score', 0),
            'recommendation_count': len(validation_report.recommendations),
            'execution_time': validation_report.execution_summary.get('total_time', 0)
        }
    
    def _run_reporting_phase(self) -> Dict[str, Any]:
        """レポートフェーズ実行"""
        self.logger.info("レポートフェーズ開始")
        
        # 統合レポート生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(f'backtest_results/task_1_3_results/comprehensive_demo_report_{timestamp}.txt')
        
        report_content = self._generate_text_report()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # JSONサマリー生成
        json_file = Path(f'backtest_results/task_1_3_results/demo_summary_{timestamp}.json')
        summary_data = self._generate_json_summary()
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        return {
            'report_file': str(report_file),
            'json_file': str(json_file),
            'report_size': report_file.stat().st_size if report_file.exists() else 0,
            'timestamp': timestamp
        }
    
    def _run_cleanup_phase(self) -> Dict[str, Any]:
        """クリーンアップフェーズ実行"""
        self.logger.info("クリーンアップフェーズ開始")
        
        # 一時ファイル削除（必要に応じて）
        temp_files_removed = 0
        
        # ログファイル整理
        log_files = list(Path('.').glob('*.log'))
        old_logs = [f for f in log_files if f.stat().st_mtime < time.time() - 86400]  # 1日以上古い
        
        return {
            'temp_files_removed': temp_files_removed,
            'old_logs_found': len(old_logs),
            'cleanup_completed': True
        }
    
    def _generate_comprehensive_report(self) -> ComprehensiveDemoReport:
        """包括レポート生成"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        # 成功判定
        overall_success = all(result.success for result in self.phase_results)
        
        # スコア計算
        system_health = 0.0
        integration_score = 0.0
        validation_score = 0.0
        
        for result in self.phase_results:
            if result.phase == DemoPhase.ISSUE_DETECTION:
                system_health = result.details.get('system_health', 0.0)
            elif result.phase == DemoPhase.INTEGRATION:
                integration_score = result.details.get('performance_score', 0.0)
            elif result.phase == DemoPhase.VALIDATION:
                validation_score = result.details.get('overall_score', 0.0)
        
        # 最終推奨事項
        recommendations = self._generate_final_recommendations()
        
        # サマリー
        summary = {
            'total_phases': len(self.phase_results),
            'successful_phases': len([r for r in self.phase_results if r.success]),
            'failed_phases': len([r for r in self.phase_results if not r.success]),
            'average_phase_time': np.mean([r.execution_time for r in self.phase_results]) if self.phase_results else 0
        }
        
        return ComprehensiveDemoReport(
            overall_success=overall_success,
            total_execution_time=total_time,
            phase_results=self.phase_results.copy(),
            system_health=system_health,
            integration_score=integration_score,
            validation_score=validation_score,
            final_recommendations=recommendations,
            summary=summary
        )
    
    def _generate_error_report(self, error: Exception) -> ComprehensiveDemoReport:
        """エラーレポート生成"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        return ComprehensiveDemoReport(
            overall_success=False,
            total_execution_time=total_time,
            phase_results=self.phase_results.copy(),
            system_health=0.0,
            integration_score=0.0,
            validation_score=0.0,
            final_recommendations=[f"致命的エラー: {error}"],
            summary={'error': str(error)}
        )
    
    def _generate_final_recommendations(self) -> List[str]:
        """最終推奨事項生成"""
        recommendations = []
        
        # フェーズ別評価
        failed_phases = [r for r in self.phase_results if not r.success]
        
        if not failed_phases:
            recommendations.extend([
                "[OK] すべてのフェーズが正常完了しました",
                "[ROCKET] Task 1.3クイック修正版の動作確認完了",
                "[UP] Phase 2への移行準備が整いました"
            ])
        else:
            recommendations.append("[WARNING] 以下のフェーズで問題が発生しました:")
            for failed in failed_phases:
                recommendations.append(f"  - {failed.phase.value}: {failed.error_message}")
        
        # スコア別推奨事項
        if hasattr(self, 'issue_detector') and self.issue_detector:
            recommendations.append("[TOOL] 問題検出・修正システムが実行されました")
        
        if hasattr(self, 'integration_manager') and self.integration_manager:
            recommendations.append("🔗 ハイブリッド統合が実行されました")
        
        if hasattr(self, 'staged_validator') and self.staged_validator:
            recommendations.append("[OK] 段階的検証が実行されました")
        
        recommendations.append("[CHART] 詳細レポートを確認してください")
        
        return recommendations
    
    def _generate_text_report(self) -> str:
        """テキストレポート生成"""
        lines = [
            "=" * 80,
            "DSSMS Task 1.3: クイック修正版 包括デモレポート",
            "=" * 80,
            f"実行日時: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"総実行時間: {(datetime.now() - self.start_time).total_seconds():.2f}秒",
            ""
        ]
        
        # フェーズ別結果
        lines.append("🔄 フェーズ実行結果:")
        for result in self.phase_results:
            status = "[OK]" if result.success else "[ERROR]"
            lines.append(f"  {status} {result.phase.value}: {result.execution_time:.2f}s")
            if result.error_message:
                lines.append(f"    エラー: {result.error_message}")
        
        lines.append("")
        
        # 詳細結果
        lines.append("[CHART] 詳細結果:")
        for result in self.phase_results:
            lines.append(f"\n[{result.phase.value.upper()}]")
            for key, value in result.details.items():
                lines.append(f"  {key}: {value}")
        
        # 推奨事項
        lines.append("\n[IDEA] 推奨事項:")
        recommendations = self._generate_final_recommendations()
        for rec in recommendations:
            lines.append(f"  {rec}")
        
        lines.extend([
            "",
            "=" * 80,
            "レポート終了",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    def _generate_json_summary(self) -> Dict[str, Any]:
        """JSONサマリー生成"""
        return {
            'demo_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_execution_time': (datetime.now() - self.start_time).total_seconds(),
                'config': self.config
            },
            'phase_results': [
                {
                    'phase': result.phase.value,
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'details': result.details,
                    'error_message': result.error_message
                }
                for result in self.phase_results
            ],
            'summary': {
                'overall_success': all(r.success for r in self.phase_results),
                'successful_phases': len([r for r in self.phase_results if r.success]),
                'failed_phases': len([r for r in self.phase_results if not r.success]),
                'total_phases': len(self.phase_results)
            }
        }

def demo_comprehensive_task_1_3():
    """Task 1.3包括デモ"""
    print("=== DSSMS Task 1.3: クイック修正版 包括デモ ===")
    
    try:
        # デモシステム初期化
        demo_system = DSSMSTask13ComprehensiveDemo({
            'run_issue_detection': True,
            'run_integration': True,
            'run_validation': True,
            'continue_on_error': True,
            'detailed_logging': True
        })
        
        # 包括デモ実行
        report = demo_system.run_comprehensive_demo()
        
        print(f"\n[CHART] 包括デモ結果:")
        print(f"全体成功: {'[OK]' if report.overall_success else '[ERROR]'}")
        print(f"総実行時間: {report.total_execution_time:.2f}秒")
        print(f"成功フェーズ: {report.summary['successful_phases']}/{report.summary['total_phases']}")
        
        print(f"\n[TARGET] スコア:")
        print(f"システム健全性: {report.system_health:.1%}")
        print(f"統合スコア: {report.integration_score:.1%}")
        print(f"検証スコア: {report.validation_score:.1%}")
        
        print(f"\n🔄 フェーズ結果:")
        for result in report.phase_results:
            status = "[OK]" if result.success else "[ERROR]"
            print(f"  {status} {result.phase.value}: {result.execution_time:.2f}s")
        
        print(f"\n[IDEA] 最終推奨事項:")
        for rec in report.final_recommendations[:5]:  # 最初の5つ
            print(f"  {rec}")
        
        return report.overall_success
        
    except Exception as e:
        print(f"[ERROR] 包括デモエラー: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    demo_comprehensive_task_1_3()
