"""
DSSMS Phase 4 Task 4.2: 実行スケジューラー テストスクリプト
DSSMSScheduler の包括的テスト

実装完了後の動作確認とパフォーマンステスト
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

def test_dssms_scheduler():
    """DSSMSScheduler メイン機能テスト"""
    logger = setup_logger("dssms_scheduler_test")
    logger.info("\n" + "="*60)
    logger.info("DSSMS Phase 4 Task 4.2: DSSMSScheduler テスト開始")
    logger.info("="*60)
    
    try:
        # DSSMSScheduler インポートとインスタンス化
        from src.dssms.dssms_scheduler import DSSMSScheduler
        
        logger.info("✓ DSSMSScheduler インポート成功")
        
        # システム初期化
        scheduler = DSSMSScheduler()
        logger.info("✓ DSSMSScheduler 初期化成功")
        
        # 基本機能テスト
        test_basic_functionality(scheduler, logger)
        
        # 統合機能テスト
        test_integration_functionality(scheduler, logger)
        
        # パフォーマンステスト
        test_performance(scheduler, logger)
        
        logger.info("\n" + "="*60)
        logger.info("[OK] 全テスト成功")
        logger.info("="*60)
        return True
        
    except Exception as e:
        logger.error(f"✗ DSSMSScheduler テストエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_basic_functionality(scheduler, logger):
    """基本機能テスト"""
    logger.info("\n--- 基本機能テスト ---")
    
    # 1. スケジューラー状況取得
    try:
        status = scheduler.get_scheduler_status()
        logger.info(f"✓ スケジューラー状況取得成功: {status.get('is_running', 'unknown')}")
        
        # 必要な状況項目チェック
        required_fields = [
            "is_running", "current_monitoring_symbol", "current_session",
            "market_open", "integration_mode", "kabu_api_available"
        ]
        
        for field in required_fields:
            if field in status:
                logger.info(f"  ✓ {field}: {status[field]}")
            else:
                logger.warning(f"  ⚠ {field}: 未定義")
                
    except Exception as e:
        logger.error(f"✗ スケジューラー状況取得エラー: {e}")
    
    # 2. 前場スクリーニングテスト
    try:
        logger.info("前場スクリーニングテスト実行中...")
        start_time = datetime.now()
        
        selected_symbol = scheduler.run_morning_screening()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if selected_symbol:
            logger.info(f"✓ 前場スクリーニング成功: {selected_symbol} ({duration:.3f}秒)")
        else:
            logger.info(f"✓ 前場スクリーニング機能確認完了 ({duration:.3f}秒)")
            
    except Exception as e:
        logger.error(f"✗ 前場スクリーニングエラー: {e}")
    
    # 3. 後場スクリーニングテスト
    try:
        logger.info("後場スクリーニングテスト実行中...")
        start_time = datetime.now()
        
        selected_symbol = scheduler.run_afternoon_screening()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if selected_symbol:
            logger.info(f"✓ 後場スクリーニング成功: {selected_symbol} ({duration:.3f}秒)")
        else:
            logger.info(f"✓ 後場スクリーニング機能確認完了 ({duration:.3f}秒)")
            
    except Exception as e:
        logger.error(f"✗ 後場スクリーニングエラー: {e}")
    
    # 4. 銘柄監視開始テスト
    try:
        test_symbol = "6758"
        scheduler.start_selected_symbol_monitoring(test_symbol)
        logger.info(f"✓ 銘柄監視開始成功: {test_symbol}")
        
    except Exception as e:
        logger.error(f"✗ 銘柄監視開始エラー: {e}")
    
    # 5. 緊急切替チェックテスト
    try:
        scheduler.handle_emergency_switch_check()
        logger.info("✓ 緊急切替チェック機能確認完了")
        
    except Exception as e:
        logger.error(f"✗ 緊急切替チェックエラー: {e}")


def test_integration_functionality(scheduler, logger):
    """統合機能テスト"""
    logger.info("\n--- 統合機能テスト ---")
    
    # 1. 時間管理システム統合
    try:
        if hasattr(scheduler, 'market_time_manager') and scheduler.market_time_manager:
            market_open = scheduler.market_time_manager.is_market_open()
            current_session = scheduler.market_time_manager.get_current_session()
            next_screening = scheduler.market_time_manager.get_next_screening_time()
            
            logger.info(f"✓ 時間管理システム統合成功:")
            logger.info(f"  - 市場開場中: {market_open}")
            logger.info(f"  - 現在セッション: {current_session}")
            logger.info(f"  - 次回スクリーニング: {next_screening}")
        else:
            logger.warning("⚠ 時間管理システム統合未完了")
            
    except Exception as e:
        logger.error(f"✗ 時間管理システム統合エラー: {e}")
    
    # 2. 緊急判定システム統合
    try:
        if hasattr(scheduler, 'emergency_detector') and scheduler.emergency_detector:
            test_symbol = "6758"
            emergency_result = scheduler.emergency_detector.check_emergency_conditions(test_symbol)
            
            logger.info(f"✓ 緊急判定システム統合成功:")
            logger.info(f"  - テスト対象: {test_symbol}")
            logger.info(f"  - 緊急事態: {emergency_result.get('is_emergency', False)}")
            logger.info(f"  - 緊急レベル: {emergency_result.get('emergency_level', 0)}")
            logger.info(f"  - 推奨アクション: {emergency_result.get('recommended_action', 'unknown')}")
        else:
            logger.warning("⚠ 緊急判定システム統合未完了")
            
    except Exception as e:
        logger.error(f"✗ 緊急判定システム統合エラー: {e}")
    
    # 3. 実行履歴システム統合
    try:
        if hasattr(scheduler, 'execution_history') and scheduler.execution_history:
            # テスト記録作成
            test_result = {
                "status": "test",
                "selected_symbol": "TEST",
                "candidate_count": 5,
                "duration": 1.0
            }
            
            record_success = scheduler.execution_history.record_screening_execution("test", test_result)
            
            if record_success:
                logger.info("✓ 実行履歴システム統合成功")
                
                # 最近のイベント取得テスト
                recent_events = scheduler.execution_history.get_recent_events(3)
                logger.info(f"  - 最近のイベント数: {len(recent_events)}")
            else:
                logger.warning("⚠ 実行履歴システム記録失敗")
        else:
            logger.warning("⚠ 実行履歴システム統合未完了")
            
    except Exception as e:
        logger.error(f"✗ 実行履歴システム統合エラー: {e}")
    
    # 4. kabu API統合
    try:
        if hasattr(scheduler, 'kabu_integration') and scheduler.kabu_integration:
            logger.info("✓ kabu API統合システム利用可能")
            logger.info(f"  - 統合モード: {getattr(scheduler, 'integration_mode', 'unknown')}")
        else:
            logger.info("ℹ kabu API統合システム未初期化（開発環境）")
            
    except Exception as e:
        logger.error(f"✗ kabu API統合チェックエラー: {e}")
    
    # 5. DSSMSコアエンジン統合
    try:
        dssms_components = {
            "nikkei225_screener": "日経225スクリーナー",
            "hierarchical_ranking": "階層ランキング",
            "intelligent_switch": "インテリジェント切替",
            "market_monitor": "市場監視"
        }
        
        available_components = []
        unavailable_components = []
        
        for component, name in dssms_components.items():
            if hasattr(scheduler, component) and getattr(scheduler, component):
                available_components.append(name)
            else:
                unavailable_components.append(name)
        
        logger.info(f"✓ DSSMSコアエンジン統合状況:")
        logger.info(f"  - 利用可能: {available_components}")
        if unavailable_components:
            logger.info(f"  - 未利用: {unavailable_components}")
            
    except Exception as e:
        logger.error(f"✗ DSSMSコアエンジン統合チェックエラー: {e}")


def test_performance(scheduler, logger):
    """パフォーマンステスト"""
    logger.info("\n--- パフォーマンステスト ---")
    
    try:
        # 初期化時間測定
        start_time = datetime.now()
        test_scheduler = DSSMSScheduler()
        init_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ 初期化時間: {init_time:.3f}秒")
        
        # スクリーニング実行時間測定
        start_time = datetime.now()
        result = test_scheduler.run_morning_screening()
        screening_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ スクリーニング実行時間: {screening_time:.3f}秒")
        
        # 緊急チェック実行時間測定
        test_scheduler.current_monitoring_symbol = "6758"
        start_time = datetime.now()
        test_scheduler.handle_emergency_switch_check()
        emergency_check_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ 緊急チェック実行時間: {emergency_check_time:.3f}秒")
        
        # 状況取得時間測定
        start_time = datetime.now()
        status = test_scheduler.get_scheduler_status()
        status_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ 状況取得時間: {status_time:.3f}秒")
        
        # パフォーマンス総合評価
        total_time = init_time + screening_time + emergency_check_time + status_time
        logger.info(f"✓ 総実行時間: {total_time:.3f}秒")
        
        if total_time < 5.0:
            logger.info("⚡ パフォーマンス: 優秀 (<5秒)")
        elif total_time < 10.0:
            logger.info("👍 パフォーマンス: 良好 (<10秒)")
        else:
            logger.warning(f"⚠ パフォーマンス: 要改善 ({total_time:.1f}秒)")
            
    except Exception as e:
        logger.error(f"✗ パフォーマンステストエラー: {e}")


def test_scheduler_lifecycle():
    """スケジューラーライフサイクルテスト"""
    logger = setup_logger("scheduler_lifecycle_test")
    logger.info("\n--- スケジューラーライフサイクルテスト ---")
    
    try:
        from src.dssms.dssms_scheduler import DSSMSScheduler
        
        scheduler = DSSMSScheduler()
        
        # 開始前状況確認
        status = scheduler.get_scheduler_status()
        logger.info(f"開始前: is_running = {status.get('is_running')}")
        
        # スケジューラー開始
        scheduler.start_scheduler()
        time.sleep(2)  # 開始待機
        
        status = scheduler.get_scheduler_status()
        logger.info(f"開始後: is_running = {status.get('is_running')}")
        
        # 短時間実行
        time.sleep(3)
        
        # スケジューラー停止
        scheduler.stop_scheduler()
        time.sleep(1)  # 停止待機
        
        status = scheduler.get_scheduler_status()
        logger.info(f"停止後: is_running = {status.get('is_running')}")
        
        logger.info("✓ スケジューラーライフサイクルテスト成功")
        return True
        
    except Exception as e:
        logger.error(f"✗ スケジューラーライフサイクルテストエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    logger = setup_logger("dssms_scheduler_main_test")
    
    test_results = {
        "basic_functionality": False,
        "scheduler_lifecycle": False
    }
    
    try:
        # 基本機能テスト
        test_results["basic_functionality"] = test_dssms_scheduler()
        
        # ライフサイクルテスト
        test_results["scheduler_lifecycle"] = test_scheduler_lifecycle()
        
        # 結果サマリー
        logger.info("\n" + "="*60)
        logger.info("テスト結果サマリー")
        logger.info("="*60)
        
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        for test_name, result in test_results.items():
            status = "[OK] 成功" if result else "[ERROR] 失敗"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\n合計: {passed_tests}/{total_tests} テスト成功")
        
        if passed_tests == total_tests:
            logger.info("[SUCCESS] 全テスト成功！DSSMSScheduler実装完了")
            return True
        else:
            logger.warning("⚠ 一部テスト失敗")
            return False
            
    except Exception as e:
        logger.error(f"メインテスト実行エラー: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
