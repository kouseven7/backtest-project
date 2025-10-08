"""
DSSMS Task 2.3 実行スクリプト
============================

DSSMSシステムのTask 2.3「パフォーマンス最適化と検証」を実行するためのスクリプトです。
PowerShellからの実行に対応しています。

Author: DSSMS Development Team
Created: 2025-01-22
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# プロジェクトルートを設定
PROJECT_ROOT = Path(r"C:\Users\imega\Documents\my_backtest_project")
sys.path.append(str(PROJECT_ROOT))

def run_task_2_3():
    """Task 2.3の実行"""
    try:
        print("\n[ROCKET] DSSMS Task 2.3: パフォーマンス最適化と検証 開始")
        print("=" * 60)
        
        # メインコントローラーをインポート・実行
        try:
            from dssms_task_2_3_main_controller import DSSMSTask23Controller
            controller = DSSMSTask23Controller()
            results = controller.execute_task_2_3_complete()
            
            print("\n[OK] Task 2.3 実行完了")
            return results
            
        except ImportError as e:
            print(f"[ERROR] メインコントローラーのインポートに失敗: {e}")
            print("個別コンポーネントで実行を試みます...")
            return run_task_2_3_individual_components()
            
    except Exception as e:
        print(f"[ERROR] Task 2.3 実行エラー: {e}")
        return None

def run_task_2_3_individual_components():
    """個別コンポーネントでの実行"""
    results = {
        'performance_optimization': None,
        'integration_testing': None,
        'quality_assurance': None
    }
    
    # 1. パフォーマンス最適化
    try:
        print("\n[UP] パフォーマンス最適化実行中...")
        from dssms_task_2_3_performance_optimizer import DSSMSPerformanceOptimizer
        optimizer = DSSMSPerformanceOptimizer()
        perf_results = optimizer.run_performance_benchmark()
        results['performance_optimization'] = perf_results
        print("[OK] パフォーマンス最適化完了")
    except Exception as e:
        print(f"[ERROR] パフォーマンス最適化エラー: {e}")
    
    # 2. 統合テスト
    try:
        print("\n[TEST] 統合テスト実行中...")
        from dssms_task_2_3_integration_test_suite import DSSMSIntegrationTestSuite
        test_suite = DSSMSIntegrationTestSuite()
        test_results = test_suite.run_all_tests()
        results['integration_testing'] = test_results
        print(f"[OK] 統合テスト完了: {test_results['tests_passed']}/{test_results['tests_passed'] + test_results['tests_failed']} 成功")
    except Exception as e:
        print(f"[ERROR] 統合テストエラー: {e}")
    
    # 3. 品質保証
    try:
        print("\n[SEARCH] 品質保証実行中...")
        from dssms_task_2_3_quality_assurance import DSSMSQualityAssuranceSystem
        qa_system = DSSMSQualityAssuranceSystem()
        qa_report = qa_system.run_comprehensive_quality_assurance()
        results['quality_assurance'] = qa_report
        print(f"[OK] 品質保証完了: スコア {qa_report.overall_score:.1f}/100")
    except Exception as e:
        print(f"[ERROR] 品質保証エラー: {e}")
    
    return results

def main():
    """メイン実行関数"""
    print("[ROCKET] DSSMS Task 2.3: パフォーマンス最適化と検証")
    print("=" * 60)
    
    # Task 2.3の実行
    results = run_task_2_3()
    
    if results:
        print("\n[SUCCESS] Task 2.3 実行完了!")
        print("=" * 60)
        print("[LIST] 成果物:")
        print("  • 最適化済みシステム")
        print("  • 統合テストスイート") 
        print("  • パフォーマンスベンチマーク")
        print("  • 品質保証レポート")
        print("=" * 60)
    else:
        print("\n[ERROR] Task 2.3 実行に問題が発生しました")
        print("ログファイルを確認してください:")
        print(f"  {PROJECT_ROOT / 'logs' / 'dssms_task_2_3_main.log'}")

if __name__ == "__main__":
    main()
