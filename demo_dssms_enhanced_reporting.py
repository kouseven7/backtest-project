"""
DSSMS Phase 3 Task 3.1: レポート生成システム改良
統合レポート生成メインスクリプト

エラー診断、パフォーマンス分析、リスク評価を統合した
包括的なレポート生成を実行します。
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import argparse
import json
from typing import Dict, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
from src.reports.error_diagnostic_reporter import ErrorDiagnosticReporter
from src.reports.report_integration_manager import ReportIntegrationManager, ReportGenerationRequest
from src.reports.dssms_enhanced_reporter import DSSMSEnhancedReporter

logger = setup_logger(__name__)

class DSSMSReportGenerator:
    """DSSMS統合レポート生成クラス"""
    
    def __init__(self, config_path: str = "config/reporting/report_config.json"):
        """
        初期化
        
        Parameters:
            config_path (str): 設定ファイルパス
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # レポーターの初期化
        self.error_reporter = ErrorDiagnosticReporter()
        self.integration_manager = ReportIntegrationManager(config_path)
        self.dssms_reporter = DSSMSEnhancedReporter(config_path)
        
        # 出力ディレクトリの確保
        self.output_dir = Path(self.config.get("report_generation", {}).get("default_output_directory", "output"))
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        default_config = {
            "report_generation": {
                "default_output_directory": "output",
                "enable_error_diagnostics": True,
                "enable_performance_metrics": True,
                "enable_risk_analysis": True
            }
        }
        
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"設定ファイルを読み込み: {self.config_path}")
            except Exception as e:
                logger.warning(f"設定ファイル読み込みエラー: {e}")
                
        return default_config
        
    def generate_full_report(self, report_type: str = "COMBINED", 
                           analysis_hours: int = 24,
                           emergency_mode: bool = False) -> Dict[str, Any]:
        """
        完全なDSSMSレポートを生成
        
        Parameters:
            report_type (str): レポートタイプ (HTML, EXCEL, COMBINED)
            analysis_hours (int): 分析対象時間（時間）
            emergency_mode (bool): 緊急モード
            
        Returns:
            dict: 生成結果
        """
        logger.info(f"完全なDSSMSレポート生成開始: {report_type}, {analysis_hours}時間")
        
        start_time = datetime.now()
        results = {
            "success": False,
            "generated_files": [],
            "errors": [],
            "warnings": [],
            "execution_summary": {}
        }
        
        try:
            # 1. エラー診断の実行
            logger.info("1. エラー診断を実行中...")
            diagnostics, health_report = self.error_reporter.analyze_logs(hours_back=analysis_hours)
            logger.info(f"エラー診断完了: {len(diagnostics)}件の問題を検出")
            
            # 2. DSSMSパフォーマンス分析
            logger.info("2. DSSMSパフォーマンス分析を実行中...")
            dssms_performance = self.dssms_reporter.analyze_dssms_performance(hours_back=analysis_hours)
            strategy_analyses = self.dssms_reporter.analyze_strategy_performance(hours_back=analysis_hours)
            portfolio_diagnostic = self.dssms_reporter.diagnose_portfolio_health(hours_back=analysis_hours)
            logger.info(f"DSSMS分析完了: 成功率{dssms_performance.switching_success_rate:.1f}%")
            
            # 3. 緊急モード判定
            if not emergency_mode:
                emergency_mode = self._should_activate_emergency_mode(diagnostics, dssms_performance, health_report)
                if emergency_mode:
                    results["warnings"].append("緊急モードが自動的に有効化されました")
                    
            # 4. レポート生成リクエストの構築
            request = ReportGenerationRequest(
                report_type=report_type,
                include_error_diagnostics=True,
                include_performance_metrics=True,
                include_risk_analysis=True,
                output_directory=str(self.output_dir),
                analysis_period_hours=analysis_hours,
                emergency_mode=emergency_mode
            )
            
            # 5. 統合レポートの生成
            logger.info("3. 統合レポートを生成中...")
            integration_result = self.integration_manager.generate_comprehensive_report(request)
            
            if integration_result.success:
                results["generated_files"].extend(integration_result.generated_files)
                results["warnings"].extend(integration_result.warnings)
                
            # 6. DSSMS専用レポートの生成
            logger.info("4. DSSMS専用レポートを生成中...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dssms_html_path = self.output_dir / f"dssms_detailed_report_{timestamp}.html"
            
            dssms_html_result = self.dssms_reporter.generate_dssms_html_report(
                str(dssms_html_path), hours_back=analysis_hours
            )
            
            if dssms_html_result:
                results["generated_files"].append(dssms_html_result)
                
            # 7. エラー診断HTMLレポートの生成
            if diagnostics:
                logger.info("5. エラー診断HTMLレポートを生成中...")
                error_html_path = self.output_dir / f"error_diagnostic_report_{timestamp}.html"
                error_html_result = self.error_reporter.generate_html_report(
                    diagnostics, health_report, str(error_html_path)
                )
                
                if error_html_result:
                    results["generated_files"].append(error_html_result)
                    
            # 8. 実行サマリーの構築
            execution_time = (datetime.now() - start_time).total_seconds()
            results["execution_summary"] = {
                "execution_time_seconds": execution_time,
                "analysis_period_hours": analysis_hours,
                "emergency_mode": emergency_mode,
                "total_errors_detected": len(diagnostics),
                "critical_errors": len([d for d in diagnostics if d.severity == 'CRITICAL']),
                "system_health_score": health_report.overall_health_score if health_report else 0.0,
                "switching_success_rate": dssms_performance.switching_success_rate,
                "portfolio_value": dssms_performance.portfolio_value,
                "portfolio_health": portfolio_diagnostic.health_status,
                "strategies_analyzed": len(strategy_analyses)
            }
            
            results["success"] = len(results["generated_files"]) > 0
            
            logger.info(f"レポート生成完了: {len(results['generated_files'])}ファイル, {execution_time:.2f}秒")
            
        except Exception as e:
            error_msg = f"レポート生成中にエラーが発生: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            results["success"] = False
            
        return results
        
    def _should_activate_emergency_mode(self, diagnostics, dssms_performance, health_report) -> bool:
        """緊急モード有効化判定"""
        emergency_triggers = self.config.get("emergency_mode", {}).get("triggers", [])
        
        # 診断結果から緊急トリガーをチェック
        for diagnostic in diagnostics:
            if diagnostic.error_type in emergency_triggers and diagnostic.severity == 'CRITICAL':
                logger.warning(f"緊急モードトリガー検出: {diagnostic.error_type}")
                return True
                
        # パフォーマンス指標から判定
        if dssms_performance.switching_success_rate <= 5.0:
            logger.warning("切り替え成功率が極度に低いため緊急モードを有効化")
            return True
            
        if dssms_performance.portfolio_value < 1000:
            logger.warning("ポートフォリオ価値が極度に低いため緊急モードを有効化")
            return True
            
        # システム健康度から判定
        if health_report and health_report.overall_health_score < 10.0:
            logger.warning("システム健康度が極度に低いため緊急モードを有効化")
            return True
            
        return False
        
    def generate_summary_report(self, analysis_hours: int = 24) -> Dict[str, Any]:
        """
        サマリーレポートを生成（軽量版）
        
        Parameters:
            analysis_hours (int): 分析対象時間（時間）
            
        Returns:
            dict: サマリー結果
        """
        logger.info(f"サマリーレポート生成開始: {analysis_hours}時間")
        
        try:
            # 基本的な指標のみ収集
            dssms_performance = self.dssms_reporter.analyze_dssms_performance(hours_back=analysis_hours)
            diagnostics, health_report = self.error_reporter.analyze_logs(hours_back=analysis_hours)
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "analysis_period_hours": analysis_hours,
                "system_health_score": health_report.overall_health_score if health_report else 0.0,
                "switching_success_rate": dssms_performance.switching_success_rate,
                "portfolio_value": dssms_performance.portfolio_value,
                "total_return": dssms_performance.total_return,
                "max_drawdown": dssms_performance.max_drawdown,
                "total_errors": len(diagnostics),
                "critical_errors": len([d for d in diagnostics if d.severity == 'CRITICAL']),
                "active_strategies": dssms_performance.active_strategies,
                "emergency_mode_required": self._should_activate_emergency_mode(diagnostics, dssms_performance, health_report)
            }
            
            logger.info("サマリーレポート生成完了")
            return summary
            
        except Exception as e:
            logger.error(f"サマリーレポート生成エラー: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "emergency_mode_required": True
            }
            
    def cleanup_old_reports(self, max_age_hours: int = 168) -> int:
        """
        古いレポートファイルをクリーンアップ
        
        Parameters:
            max_age_hours (int): 保持期間（時間）
            
        Returns:
            int: 削除されたファイル数
        """
        logger.info(f"古いレポートのクリーンアップ開始: {max_age_hours}時間以前")
        
        deleted_count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        try:
            for file_path in self.output_dir.glob("*report*.html"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"削除: {file_path}")
                    
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: for file_path in self.output_dir.glob("*report*.xlsx"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"削除: {file_path}")
                    
            logger.info(f"クリーンアップ完了: {deleted_count}ファイルを削除")
            
        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")
            
        return deleted_count

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="DSSMS統合レポート生成")
    parser.add_argument("--type", choices=["HTML", "EXCEL", "COMBINED"], default="COMBINED",
                       help="レポートタイプ")
    parser.add_argument("--hours", type=int, default=24,
                       help="分析対象時間（時間）")
    parser.add_argument("--emergency", action="store_true",
                       help="緊急モードを強制有効化")
    parser.add_argument("--summary-only", action="store_true",
                       help="サマリーレポートのみ生成")
    parser.add_argument("--cleanup", action="store_true",
                       help="古いレポートをクリーンアップ")
    parser.add_argument("--config", default="config/reporting/report_config.json",
                       help="設定ファイルパス")
    
    args = parser.parse_args()
    
    # レポート生成器の初期化
    generator = DSSMSReportGenerator(config_path=args.config)
    
    print("=" * 60)
    print("🔄 DSSMS Phase 3 Task 3.1: レポート生成システム改良")
    print("=" * 60)
    
    try:
        # クリーンアップ実行
        if args.cleanup:
            print("🧹 古いレポートをクリーンアップ中...")
            deleted_count = generator.cleanup_old_reports()
            print(f"✅ クリーンアップ完了: {deleted_count}ファイルを削除")
            
        # サマリーレポート生成
        if args.summary_only:
            print(f"📊 サマリーレポートを生成中... (過去{args.hours}時間)")
            summary = generator.generate_summary_report(analysis_hours=args.hours)
            
            print("\n=== サマリーレポート ===")
            print(f"システム健康度: {summary.get('system_health_score', 0):.1f}/100.0")
            print(f"切り替え成功率: {summary.get('switching_success_rate', 0):.1f}%")
            print(f"ポートフォリオ価値: ¥{summary.get('portfolio_value', 0):,.0f}")
            print(f"総リターン: {summary.get('total_return', 0):.2f}%")
            print(f"エラー数: {summary.get('total_errors', 0)} (重大: {summary.get('critical_errors', 0)})")
            print(f"緊急モード推奨: {'はい' if summary.get('emergency_mode_required', False) else 'いいえ'}")
            
        else:
            # 完全レポート生成
            print(f"📊 完全レポートを生成中... ({args.type}, 過去{args.hours}時間)")
            if args.emergency:
                print("⚠️ 緊急モードが有効化されています")
                
            results = generator.generate_full_report(
                report_type=args.type,
                analysis_hours=args.hours,
                emergency_mode=args.emergency
            )
            
            print("\n=== レポート生成結果 ===")
            print(f"成功: {'はい' if results['success'] else 'いいえ'}")
            print(f"生成ファイル数: {len(results['generated_files'])}")
            print(f"実行時間: {results['execution_summary'].get('execution_time_seconds', 0):.2f}秒")
            
            if results["generated_files"]:
                print("\n生成されたファイル:")
                for file_path in results["generated_files"]:
                    print(f"  📄 {file_path}")
                    
            if results["warnings"]:
                print("\n⚠️ 警告:")
                for warning in results["warnings"]:
                    print(f"  - {warning}")
                    
            if results["errors"]:
                print("\n❌ エラー:")
                for error in results["errors"]:
                    print(f"  - {error}")
                    
            # 実行サマリー表示
            summary = results.get("execution_summary", {})
            if summary:
                print("\n=== 実行サマリー ===")
                print(f"システム健康度: {summary.get('system_health_score', 0):.1f}/100.0")
                print(f"切り替え成功率: {summary.get('switching_success_rate', 0):.1f}%")
                print(f"ポートフォリオ価値: ¥{summary.get('portfolio_value', 0):,.0f}")
                print(f"ポートフォリオ健康度: {summary.get('portfolio_health', 'UNKNOWN')}")
                print(f"エラー検出数: {summary.get('total_errors', 0)} (重大: {summary.get('critical_errors', 0)})")
                print(f"分析された戦略数: {summary.get('strategies_analyzed', 0)}")
                print(f"緊急モード: {'有効' if summary.get('emergency_mode', False) else '無効'}")
                
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生: {e}")
        logger.error(f"メイン処理エラー: {e}")
        
    print("\n🏁 処理完了")

if __name__ == "__main__":
    main()
