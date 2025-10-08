"""
DSSMS Phase 3 Task 3.3: 自動検証フレームワーク デモ実行
包括的な動作確認とテスト実行

Author: GitHub Copilot Agent
Created: 2025-08-28
Phase: 3 Task 3.3
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
import json
from typing import List

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.testing.dssms_validation_framework import DSSMSValidationFramework, ValidationLevel, ValidationResult
from config.logger_config import setup_logger

def main():
    """メインデモ実行"""
    print("="*80)
    print("DSSMS Phase 3 Task 3.3: 自動検証フレームワーク - デモ実行")
    print("="*80)
    print()
    
    logger = setup_logger("DSSMSValidationDemo")
    
    try:
        # 1. フレームワーク初期化
        print("[TOOL] フレームワーク初期化中...")
        config_path = project_root / "config" / "validation" / "validation_config.json"
        framework = DSSMSValidationFramework(str(config_path))
        print("[OK] フレームワーク初期化完了")
        print()
        
        # 2. 基本検証レベルテスト
        print("[LIST] 基本検証レベルテスト実行...")
        basic_results = framework.run_validation([ValidationLevel.BASIC])
        print(f"[OK] 基本検証完了: {len(basic_results)}件の結果")
        _print_level_summary("BASIC", basic_results)
        print()
        
        # 3. 単体検証レベルテスト
        print("[SEARCH] 単体検証レベルテスト実行...")
        try:
            unit_results = framework.run_validation([ValidationLevel.UNIT])
            print(f"[OK] 単体検証完了: {len(unit_results)}件の結果")
            _print_level_summary("UNIT", unit_results)
        except Exception as e:
            print(f"[WARNING] 単体検証スキップ: {e}")
            unit_results = []
        print()
        
        # 4. 統合検証レベルテスト
        print("🔗 統合検証レベルテスト実行...")
        try:
            integration_results = framework.run_validation([ValidationLevel.INTEGRATION])
            print(f"[OK] 統合検証完了: {len(integration_results)}件の結果")
            _print_level_summary("INTEGRATION", integration_results)
        except Exception as e:
            print(f"[WARNING] 統合検証スキップ: {e}")
            integration_results = []
        print()
        
        # 5. パフォーマンス検証レベルテスト
        print("[ROCKET] パフォーマンス検証レベルテスト実行...")
        performance_results = framework.run_validation([ValidationLevel.PERFORMANCE])
        print(f"[OK] パフォーマンス検証完了: {len(performance_results)}件の結果")
        _print_level_summary("PERFORMANCE", performance_results)
        print()
        
        # 6. 本番環境検証レベルテスト
        print("🏭 本番環境検証レベルテスト実行...")
        production_results = framework.run_validation([ValidationLevel.PRODUCTION])
        print(f"[OK] 本番環境検証完了: {len(production_results)}件の結果")
        _print_level_summary("PRODUCTION", production_results)
        print()
        
        # 7. 全体結果集計
        all_results = basic_results + unit_results + integration_results + performance_results + production_results
        
        if all_results:
            print("[CHART] 全体結果サマリー")
            print("-" * 50)
            
            overall_score = framework.get_overall_score(all_results)
            production_ready = framework.is_production_ready(all_results)
            
            total_tests = len(all_results)
            successful_tests = sum(1 for r in all_results if r.success)
            success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
            
            print(f"実行テスト数: {total_tests}")
            print(f"成功テスト数: {successful_tests}")
            print(f"成功率: {success_rate:.1%}")
            print(f"総合スコア: {overall_score:.1%}")
            print(f"本番準備完了: {'はい' if production_ready else 'いいえ'}")
            print()
            
            # 8. レポート生成
            print("📝 検証レポート生成中...")
            report_path = framework.generate_report(all_results)
            if report_path:
                print(f"[OK] レポート生成完了: {report_path}")
                
                # JSON版とExcel版のパスも表示
                json_path = report_path.replace('.html', '.json')
                excel_path = report_path.replace('.html', '.xlsx')
                
                if Path(json_path).exists():
                    print(f"📄 JSON レポート: {json_path}")
                if Path(excel_path).exists():
                    print(f"[CHART] Excel レポート: {excel_path}")
            print()
            
            # 9. 修正提案生成
            print("[TOOL] 自動修正提案生成中...")
            try:
                suggestions = framework.suggest_fixes(all_results)
                if suggestions:
                    print(f"[OK] 修正提案生成完了: {len(suggestions)}件")
                    print("\n[TARGET] 主要な修正提案:")
                    for i, suggestion in enumerate(suggestions[:5], 1):
                        print(f"  {i}. [{suggestion.priority.value.upper()}] {suggestion.title}")
                        print(f"     カテゴリ: {suggestion.category}")
                        print(f"     推定時間: {suggestion.estimated_time}")
                        print(f"     影響: {suggestion.impact}")
                    
                    if len(suggestions) > 5:
                        print(f"     ... 他 {len(suggestions) - 5} 件の提案")
                else:
                    print("[OK] 修正が必要な問題は見つかりませんでした")
            except Exception as e:
                print(f"[WARNING] 修正提案生成スキップ: {e}")
            print()
            
            # 10. 最終評価
            print("[TARGET] 最終評価")
            print("-" * 50)
            
            if overall_score >= 0.90:
                status = "優秀"
                emoji = "🌟"
            elif overall_score >= 0.80:
                status = "良好"
                emoji = "[OK]"
            elif overall_score >= 0.70:
                status = "合格"
                emoji = "👍"
            elif overall_score >= 0.60:
                status = "要改善"
                emoji = "[WARNING]"
            else:
                status = "要修正"
                emoji = "[ERROR]"
            
            print(f"{emoji} 総合評価: {status}")
            print(f"スコア: {overall_score:.1%}")
            
            if production_ready:
                print("[ROCKET] 本番環境でのデプロイが可能です")
            else:
                print("[TOOL] 本番環境デプロイ前に修正が必要です")
            
        else:
            print("[ERROR] 検証結果が取得できませんでした")
        
        print()
        print("="*80)
        print("DSSMS自動検証フレームワーク デモ実行完了")
        print("="*80)
        
    except Exception as e:
        logger.error(f"デモ実行エラー: {e}")
        print(f"[ERROR] デモ実行エラー: {e}")
        return 1
    
    return 0

def _print_level_summary(level_name: str, results: List[ValidationResult]):
    """レベル別結果サマリー表示"""
    if not results:
        print(f"   {level_name}: 結果なし")
        return
    
    result = results[0]  # 各レベル1つの結果と仮定
    
    status = "[OK] 成功" if result.success else "[ERROR] 失敗"
    score = result.score
    execution_time = result.execution_time
    
    print(f"   {level_name}: {status} (スコア: {score:.1%}, 実行時間: {execution_time:.2f}秒)")
    
    if result.errors:
        print(f"   エラー数: {len(result.errors)}")
    if result.warnings:
        print(f"   警告数: {len(result.warnings)}")

def run_specific_level_demo(level: ValidationLevel):
    """特定レベルのデモ実行"""
    print(f"\n{level.value.upper()} レベル単独テスト実行")
    print("-" * 40)
    
    logger = setup_logger(f"DSSMS{level.value.title()}Demo")
    
    try:
        config_path = project_root / "config" / "validation" / "validation_config.json"
        framework = DSSMSValidationFramework(str(config_path))
        
        results = framework.run_validation([level])
        
        if results:
            result = results[0]
            print(f"結果: {'成功' if result.success else '失敗'}")
            print(f"スコア: {result.score:.1%}")
            print(f"実行時間: {result.execution_time:.2f}秒")
            
            if result.details:
                print("\n詳細情報:")
                for key, value in result.details.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"    {sub_key}: {sub_value}")
                    else:
                        print(f"  {key}: {value}")
            
            if result.errors:
                print(f"\nエラー ({len(result.errors)}件):")
                for error in result.errors:
                    print(f"  - {error}")
            
            if result.warnings:
                print(f"\n警告 ({len(result.warnings)}件):")
                for warning in result.warnings:
                    print(f"  - {warning}")
            
            if result.suggestions:
                print(f"\n提案 ({len(result.suggestions)}件):")
                for suggestion in result.suggestions:
                    print(f"  - {suggestion}")
        else:
            print("[ERROR] 結果を取得できませんでした")
    
    except Exception as e:
        print(f"[ERROR] {level.value}レベルテストエラー: {e}")

def test_individual_components():
    """個別コンポーネントテスト"""
    print("\n[TEST] 個別コンポーネントテスト")
    print("="*50)
    
    # テストデータ管理システム
    print("\n1. テストデータ管理システム")
    try:
        from src.testing.test_data_manager import TestDataManager
        manager = TestDataManager()
        
        # 各種データ取得テスト
        regression_data = manager.get_test_data("regression", "basic_regression")
        stress_data = manager.get_test_data("stress", "market_crash")
        
        print("[OK] テストデータ管理システム正常")
        print(f"   回帰テストデータ: {len(regression_data)} 項目")
        print(f"   ストレステストデータ: {len(stress_data)} 項目")
    except Exception as e:
        print(f"[ERROR] テストデータ管理システムエラー: {e}")
    
    # 検証レポーター
    print("\n2. 検証レポーター")
    try:
        from src.testing.validation_reporter import ValidationReporter
        from src.testing.dssms_validation_framework import ValidationConfig, ValidationResult
        
        config = ValidationConfig(
            validation_levels=[],
            parallel_execution=False,
            early_termination=False,
            auto_fix_attempts=3,
            high_level_criteria={},
            timeout_seconds=300,
            log_level="INFO"
        )
        
        reporter = ValidationReporter(config)
        print("[OK] 検証レポーター正常")
    except Exception as e:
        print(f"[ERROR] 検証レポーターエラー: {e}")
    
    # 自動修正提案システム
    print("\n3. 自動修正提案システム")
    try:
        from src.testing.automated_fix_suggestions import AutoFixSuggestions
        
        fixer = AutoFixSuggestions(config)
        print("[OK] 自動修正提案システム正常")
    except Exception as e:
        print(f"[ERROR] 自動修正提案システムエラー: {e}")

if __name__ == "__main__":
    # 引数による実行モード切替
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "basic":
            run_specific_level_demo(ValidationLevel.BASIC)
        elif mode == "unit":
            run_specific_level_demo(ValidationLevel.UNIT)
        elif mode == "integration":
            run_specific_level_demo(ValidationLevel.INTEGRATION)
        elif mode == "performance":
            run_specific_level_demo(ValidationLevel.PERFORMANCE)
        elif mode == "production":
            run_specific_level_demo(ValidationLevel.PRODUCTION)
        elif mode == "components":
            test_individual_components()
        else:
            print("使用法: python demo_validation_framework.py [basic|unit|integration|performance|production|components]")
            sys.exit(1)
    else:
        # フルデモ実行
        exit_code = main()
        sys.exit(exit_code)
