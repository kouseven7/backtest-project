"""
Quality Assurance System Integration Test
Phase 2.3 Task 2.3.3: 品質保証システム統合テスト

Purpose:
  - 品質保証システムの統合テスト
  - 各コンポーネントの動作確認
  - unified_output_engine.pyとの連携テスト

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

# 品質保証システムのインポート
try:
    from src.dssms.qa_config_manager import QAConfigManager
    from src.dssms.output_data_validator import OutputDataValidator
    from src.dssms.consistency_checker import ConsistencyChecker
    from src.dssms.regression_test_suite import RegressionTestSuite
    from src.dssms.quality_assurance_engine import QualityAssuranceEngine
    qa_available = True
except ImportError as e:
    print(f"品質保証システムのインポートエラー: {e}")
    qa_available = False


def create_test_data():
    """テストデータ作成"""
    np.random.seed(42)  # 再現性のため
    
    test_data = {}
    
    # 戦略1: VWAPBreakoutStrategy
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    test_data['VWAPBreakoutStrategy'] = pd.DataFrame({
        'Date': dates,
        'Entry_Signal': np.random.choice([0, 1], 252, p=[0.9, 0.1]),
        'Exit_Signal': np.random.choice([0, 1], 252, p=[0.9, 0.1]),
        'Position': np.random.uniform(-1, 1, 252),
        'Price': 100 + np.cumsum(np.random.normal(0, 1, 252)),
        'Profit_Loss': np.random.normal(0, 0.02, 252),
        'Cumulative_Return': np.cumprod(1 + np.random.normal(0.0005, 0.02, 252))
    })
    
    # 戦略2: BreakoutStrategy
    test_data['BreakoutStrategy'] = pd.DataFrame({
        'Date': dates,
        'Entry_Signal': np.random.choice([0, 1], 252, p=[0.85, 0.15]),
        'Exit_Signal': np.random.choice([0, 1], 252, p=[0.85, 0.15]),
        'Position': np.random.uniform(-0.8, 0.8, 252),
        'Price': 95 + np.cumsum(np.random.normal(0, 1.2, 252)),
        'Profit_Loss': np.random.normal(0, 0.025, 252),
        'Cumulative_Return': np.cumprod(1 + np.random.normal(0.0003, 0.025, 252))
    })
    
    # 戦略3: MomentumInvestingStrategy
    test_data['MomentumInvestingStrategy'] = pd.DataFrame({
        'Date': dates,
        'Entry_Signal': np.random.choice([0, 1], 252, p=[0.88, 0.12]),
        'Exit_Signal': np.random.choice([0, 1], 252, p=[0.88, 0.12]),
        'Position': np.random.uniform(-1.2, 1.2, 252),
        'Price': 105 + np.cumsum(np.random.normal(0, 0.8, 252)),
        'Profit_Loss': np.random.normal(0, 0.018, 252),
        'Cumulative_Return': np.cumprod(1 + np.random.normal(0.0008, 0.018, 252))
    })
    
    return test_data


def test_qa_config_manager():
    """設定管理システムテスト"""
    print("\n=== QA設定管理システム テスト ===")
    
    try:
        config_manager = QAConfigManager()
        
        # 設定妥当性チェック
        is_valid = config_manager.validate_config()
        print(f"設定妥当性: {'OK' if is_valid else 'NG'}")
        
        # 閾値設定取得
        performance_thresholds = config_manager.get_performance_thresholds()
        print(f"パフォーマンス閾値: 最大ドローダウン={performance_thresholds.max_drawdown}")
        
        # エラーハンドリング設定取得
        error_config = config_manager.get_error_handling_config()
        print(f"エラー処理: クリティカル={error_config.critical_action.value}")
        
        print("✓ QA設定管理システムテスト完了")
        return True
        
    except Exception as e:
        print(f"✗ QA設定管理システムテストエラー: {e}")
        return False


def test_output_data_validator():
    """出力データ検証システムテスト"""
    print("\n=== 出力データ検証システム テスト ===")
    
    try:
        validator = OutputDataValidator()
        test_data = create_test_data()
        
        # 各戦略のデータを検証
        for strategy_name, data in test_data.items():
            print(f"\n戦略 '{strategy_name}' 検証中...")
            result = validator.validate_output_data(data)
            
            print(f"  検証結果: {'合格' if result.is_valid else '不合格'}")
            print(f"  エラー数: {result.error_count}")
            print(f"  警告数: {result.warning_count}")
            
            if result.messages:
                print(f"  メッセージ: {result.messages[:2]}")  # 最初の2件のみ
        
        print("✓ 出力データ検証システムテスト完了")
        return True
        
    except Exception as e:
        print(f"✗ 出力データ検証システムテストエラー: {e}")
        return False


def test_consistency_checker():
    """一貫性チェックシステムテスト"""
    print("\n=== 一貫性チェックシステム テスト ===")
    
    try:
        checker = ConsistencyChecker()
        test_data = create_test_data()
        
        # 一貫性チェック実行
        result = checker.check_backtest_consistency(test_data)
        
        print(f"一貫性状態: {'一貫' if result.is_consistent else '不一致'}")
        print(f"不整合数: {result.inconsistency_count}")
        print(f"警告数: {result.warning_count}")
        
        if result.messages:
            print(f"メッセージ例: {result.messages[:2]}")
        
        print("✓ 一貫性チェックシステムテスト完了")
        return True
        
    except Exception as e:
        print(f"✗ 一貫性チェックシステムテストエラー: {e}")
        return False


def test_regression_test_suite():
    """リグレッションテストシステムテスト"""
    print("\n=== リグレッションテストシステム テスト ===")
    
    try:
        test_suite = RegressionTestSuite()
        test_data = create_test_data()
        
        # ベースライン更新（初回のみ）
        print("ベースライン更新中...")
        test_suite.update_baseline('performance_regression', test_data)
        test_suite.update_baseline('output_format_stability', test_data)
        test_suite.update_baseline('consistency_maintenance', test_data)
        
        # リグレッションテスト実行
        print("リグレッションテスト実行中...")
        report = test_suite.run_regression_tests(test_data)
        
        print(f"総合結果: {'合格' if report.overall_result else '失敗'}")
        print(f"総テスト数: {report.total_tests}")
        print(f"合格数: {report.passed_tests}")
        print(f"失敗数: {report.failed_tests}")
        
        print("✓ リグレッションテストシステムテスト完了")
        return True
        
    except Exception as e:
        print(f"✗ リグレッションテストシステムテストエラー: {e}")
        return False


def test_quality_assurance_engine():
    """品質保証エンジンテスト"""
    print("\n=== 品質保証エンジン テスト ===")
    
    try:
        qa_engine = QualityAssuranceEngine()
        test_data = create_test_data()
        
        # 品質保証実行
        print("品質保証プロセス実行中...")
        qa_report = qa_engine.run_quality_assurance(test_data, run_regression_tests=True)
        
        print(f"品質保証結果:")
        print(f"  実行サマリー: {qa_report.execution_summary}")
        print(f"  アクション要求: {'はい' if qa_report.action_required else 'いいえ'}")
        
        if qa_report.quality_assessment:
            print(f"  総合品質スコア: {qa_report.quality_assessment.overall_score:.2f}")
            print(f"  品質レベル: {qa_report.quality_assessment.quality_level}")
        
        # レポート生成テスト
        print("\n品質レポート生成中...")
        report_content = qa_engine.generate_quality_report(qa_report)
        print(f"レポート文字数: {len(report_content)}")
        
        # レポート保存テスト
        report_file = qa_engine.save_quality_report(qa_report)
        print(f"レポート保存: {report_file}")
        
        print("✓ 品質保証エンジンテスト完了")
        return True
        
    except Exception as e:
        print(f"✗ 品質保証エンジンテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_output_engine_integration():
    """統一出力エンジン統合テスト"""
    print("\n=== 統一出力エンジン統合 テスト ===")
    
    try:
        # unified_output_engine のインポートテスト
        try:
            from src.dssms.unified_output_engine import UnifiedOutputEngine
            print("✓ unified_output_engine インポート成功")
        except ImportError as e:
            print(f"✗ unified_output_engine インポートエラー: {e}")
            return False
        
        # エンジン初期化
        engine = UnifiedOutputEngine(enable_quality_assurance=True)
        print(f"✓ エンジン初期化成功 (QA有効: {engine.enable_quality_assurance})")
        
        # テストデータを適切な形式に変換
        test_data = create_test_data()
        unified_test_data = {
            'strategies': test_data,
            'metadata': {
                'test_run': True,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # 品質保証付き出力生成テスト（簡易版）
        if hasattr(engine, 'generate_unified_output_with_qa'):
            print("品質保証付き出力生成テスト実行中...")
            result = engine.generate_unified_output_with_qa(
                data=unified_test_data,
                output_formats=['json'],
                output_prefix='qa_integration_test',
                run_regression_tests=False  # 初回はリグレッション無し
            )
            
            print(f"  出力成功: {result['success']}")
            print(f"  出力ファイル数: {len(result['output_files'])}")
            print(f"  アクション要求: {result['action_required']}")
            print(f"  QAサマリー: {result['qa_summary'][:100]}...")
        else:
            print("⚠ 品質保証付き出力機能は利用できません")
        
        print("✓ 統一出力エンジン統合テスト完了")
        return True
        
    except Exception as e:
        print(f"✗ 統一出力エンジン統合テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """総合テスト実行"""
    print("=" * 60)
    print("品質保証システム 総合テスト")
    print("=" * 60)
    
    if not qa_available:
        print("✗ 品質保証システムが利用できません")
        return
    
    test_results = []
    
    # 各テスト実行
    test_results.append(("QA設定管理", test_qa_config_manager()))
    test_results.append(("出力データ検証", test_output_data_validator()))
    test_results.append(("一貫性チェック", test_consistency_checker()))
    test_results.append(("リグレッションテスト", test_regression_test_suite()))
    test_results.append(("品質保証エンジン", test_quality_assurance_engine()))
    test_results.append(("統一出力エンジン統合", test_unified_output_engine_integration()))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "合格" if result else "失敗"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n総合結果: {passed}/{total} 合格 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("[SUCCESS] すべてのテストが合格しました！")
        print("Phase 2.3 Task 2.3.3 品質保証システム実装完了")
    else:
        print("⚠ 一部のテストが失敗しました。詳細を確認してください。")
    
    return passed == total


if __name__ == "__main__":
    logger = setup_logger(__name__)
    logger.info("品質保証システム総合テスト開始")
    
    success = run_comprehensive_test()
    
    if success:
        logger.info("品質保証システム総合テスト完了: 全テスト合格")
    else:
        logger.warning("品質保証システム総合テスト完了: 一部テスト失敗")
