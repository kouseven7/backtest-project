"""
Simple test for Score Update Trigger System
シンプルなスコアアップデートトリガーシステムのテスト
"""

import logging
import time
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trigger_system_imports():
    """トリガーシステムのインポートテスト"""
    print("[TEST] Testing Score Update Trigger System imports...")
    
    try:
        # 基本インポートテスト
        print("   Testing basic imports...")
        
        # モジュールインポート
        from config.score_update_trigger_system import (
            ScoreUpdateTriggerSystem, TriggerType, TriggerPriority, TriggerCondition
        )
        print("   ✓ Score Update Trigger System imported")
        
        from config.realtime_update_engine import (
            RealtimeUpdateEngine, UpdatePriority, UpdateRequest
        )
        print("   ✓ Realtime Update Engine imported")
        
        # 基本オブジェクト作成テスト
        print("   Testing object creation...")
        
        # トリガーシステム作成（依存関係なし）
        trigger_system = ScoreUpdateTriggerSystem()
        print("   ✓ Trigger system created")
        
        # リアルタイムエンジン作成（トリガーシステムと連携）
        realtime_engine = RealtimeUpdateEngine(trigger_system=trigger_system)
        print("   ✓ Realtime engine created")
        
        # トリガー条件作成テスト
        condition = TriggerCondition(
            condition_id="test_condition",
            trigger_type=TriggerType.MANUAL,
            priority=TriggerPriority.HIGH,
            description="Test trigger condition"
        )
        print("   ✓ Trigger condition created")
        
        # 更新リクエスト作成テスト
        request = UpdateRequest(
            request_id="test_request",
            strategy_name="test_strategy",
            ticker="TEST",
            trigger_type=TriggerType.MANUAL,
            metadata={"test": True}
        )
        print("   ✓ Update request created")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Import test failed: {e}")
        logger.error(f"Import test error: {e}")
        return False

def test_basic_trigger_operations():
    """基本トリガー操作テスト"""
    print("[TOOL] Testing basic trigger operations...")
    
    try:
        from config.score_update_trigger_system import (
            ScoreUpdateTriggerSystem, TriggerType, TriggerPriority
        )
        
        # トリガーシステム作成
        trigger_system = ScoreUpdateTriggerSystem()
        print("   ✓ Trigger system created")
        
        # トリガーシステム開始
        trigger_system.start()
        print("   ✓ Trigger system started")
        
        # 手動トリガーテスト
        event_id = trigger_system.manual_trigger(
            strategy_name="test_strategy",
            ticker="TEST",
            priority=TriggerPriority.HIGH,
            metadata={"test_mode": True}
        )
        print(f"   ✓ Manual trigger queued: {event_id}")
        
        # 統計情報確認
        time.sleep(1)  # 処理時間確保
        stats = trigger_system.get_trigger_statistics()
        print(f"   ✓ Trigger stats: {stats}")
        
        # 最近のイベント確認
        events = trigger_system.get_recent_events(limit=5)
        print(f"   ✓ Recent events: {len(events)} found")
        
        # トリガーシステム停止
        trigger_system.stop()
        print("   ✓ Trigger system stopped")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Basic operations test failed: {e}")
        logger.error(f"Basic operations test error: {e}")
        return False

def main():
    """メインテスト実行"""
    print("=" * 60)
    print("[ROCKET] Score Update Trigger System - Simple Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    results = {}
    
    # 1. インポートテスト
    results["imports"] = test_trigger_system_imports()
    print()
    
    # 2. 基本操作テスト（インポートが成功した場合のみ）
    if results["imports"]:
        results["basic_operations"] = test_basic_trigger_operations()
    else:
        results["basic_operations"] = False
        print("⏭️  Skipping basic operations test due to import failure")
    print()
    
    # 結果サマリー
    print("[LIST] Test Summary:")
    for test_name, success in results.items():
        status = "[OK] PASSED" if success else "[ERROR] FAILED"
        print(f"   {test_name}: {status}")
    
    overall_success = all(results.values())
    print(f"\n[TARGET] Overall result: {'[OK] ALL PASSED' if overall_success else '[ERROR] SOME FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
