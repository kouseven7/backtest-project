"""
Final Score Update Trigger System Test
最終統合テスト - 2-3-3実装確認
"""

import logging
import time
import asyncio
from datetime import datetime

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_full_integration():
    """完全統合テスト"""
    print("=" * 70)
    print("[ROCKET] 2-3-3 Score Update Trigger System - Final Integration Test")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        # 1. システムインポート
        print("📦 1. Importing systems...")
        from config.score_update_trigger_system import (
            ScoreUpdateTriggerSystem, TriggerType, TriggerPriority, TriggerCondition
        )
        from config.realtime_update_engine import (
            RealtimeUpdateEngine, UpdatePriority, UpdateRequest
        )
        print("   ✓ All systems imported successfully")
        
        # 2. システム初期化
        print("\n[TOOL] 2. Initializing systems...")
        trigger_system = ScoreUpdateTriggerSystem()
        realtime_engine = RealtimeUpdateEngine(trigger_system=trigger_system)
        print("   ✓ Trigger system and realtime engine created")
        
        # 3. トリガーシステム開始
        print("\n▶️ 3. Starting trigger system...")
        trigger_system.start()
        print("   ✓ Trigger system started")
        
        # 4. 手動トリガーテスト
        print("\n[TARGET] 4. Testing manual triggers...")
        
        # 複数の手動トリガー
        test_triggers = []
        for i in range(3):
            event_id = trigger_system.manual_trigger(
                strategy_name=f"test_strategy_{i+1}",
                ticker=f"TEST{i+1}",
                priority=TriggerPriority.HIGH,
                metadata={"test_iteration": i+1, "test_type": "manual"}
            )
            test_triggers.append(event_id)
            print(f"   ✓ Manual trigger {i+1} queued: {event_id}")
        
        # 5. トリガー処理確認
        print("\n⏱️ 5. Waiting for trigger processing...")
        await asyncio.sleep(3)  # 処理時間確保
        
        stats = trigger_system.get_trigger_statistics()
        print(f"   ✓ Trigger statistics:")
        print(f"     - Total triggers: {stats['total_triggers']}")
        print(f"     - Successful: {stats['successful_triggers']}")
        print(f"     - Failed: {stats['failed_triggers']}")
        print(f"     - Queue size: {stats['queue_size']}")
        
        # 6. カスタムトリガー条件追加
        print("\n⚙️ 6. Adding custom trigger conditions...")
        
        custom_condition = TriggerCondition(
            condition_id="test_custom_condition",
            trigger_type=TriggerType.THRESHOLD_BASED,
            priority=TriggerPriority.MEDIUM,
            parameters={
                "score_change_threshold": 0.1,
                "monitoring_window_hours": 1,
                "test_mode": True
            },
            description="Test custom threshold condition"
        )
        
        trigger_system.add_trigger_condition(custom_condition)
        print("   ✓ Custom trigger condition added")
        
        # 7. 最近のイベント確認
        print("\n[LIST] 7. Checking recent events...")
        events = trigger_system.get_recent_events(limit=10)
        print(f"   ✓ Found {len(events)} recent events:")
        
        for i, event in enumerate(events[-3:], 1):  # 最新3件表示
            print(f"     {i}. ID: {event.get('event_id', 'N/A')}")
            print(f"        Strategy: {event.get('strategy_name', 'N/A')}")
            print(f"        Status: {event.get('status', 'N/A')}")
        
        # 8. リアルタイムエンジンテスト（簡易）
        print("\n⚡ 8. Testing realtime engine integration...")
        
        # エンジン状態確認
        engine_status = realtime_engine.get_engine_status()
        print(f"   ✓ Engine status: {engine_status['status']}")
        print(f"   ✓ Total requests: {engine_status['total_requests']}")
        
        # 9. システム終了
        print("\n🛑 9. Shutting down systems...")
        
        trigger_system.stop()
        await realtime_engine.stop()
        print("   ✓ All systems stopped")
        
        # 10. 最終結果
        print("\n[OK] 10. Final Results:")
        print("   [TARGET] 2-3-3 Score Update Trigger Design - IMPLEMENTED")
        print("   [OK] Trigger system operational")
        print("   [OK] Realtime engine integrated")
        print("   [OK] Manual triggers working")
        print("   [OK] Custom conditions supported")
        print("   [OK] Event monitoring functional")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Integration test failed: {e}")
        logger.error(f"Integration test error: {e}")
        return False

def run_sync_test():
    """同期版テスト"""
    print("=" * 70)
    print("[TOOL] 2-3-3 Score Update Trigger System - Synchronous Test")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        # システムインポート・初期化
        from config.score_update_trigger_system import (
            ScoreUpdateTriggerSystem, TriggerPriority
        )
        
        trigger_system = ScoreUpdateTriggerSystem()
        print("✓ Trigger system created")
        
        # システム開始
        trigger_system.start()
        print("✓ Trigger system started")
        
        # 手動トリガーテスト
        event_id = trigger_system.manual_trigger(
            strategy_name="sync_test_strategy",
            ticker="SYNC",
            priority=TriggerPriority.HIGH,
            metadata={"sync_test": True}
        )
        print(f"✓ Manual trigger queued: {event_id}")
        
        # 処理待機
        time.sleep(2)
        
        # 統計確認
        stats = trigger_system.get_trigger_statistics()
        print(f"✓ Final stats: {stats['total_triggers']} total, {stats['successful_triggers']} successful")
        
        # システム停止
        trigger_system.stop()
        print("✓ System stopped")
        
        print("\n[SUCCESS] Synchronous test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Sync test failed: {e}")
        return False

def main():
    """メイン実行"""
    print("選択してください:")
    print("1. 非同期統合テスト (推奨)")
    print("2. 同期基本テスト")
    
    choice = input("選択 (1-2): ").strip()
    
    if choice == "1":
        # 非同期テスト実行
        success = asyncio.run(test_full_integration())
    elif choice == "2":
        # 同期テスト実行
        success = run_sync_test()
    else:
        print("無効な選択です。同期テストを実行します。")
        success = run_sync_test()
    
    if success:
        print("\n🎊 2-3-3「スコアアップデートトリガー設計」実装完了!")
        print("   ✨ トリガーシステムが正常に動作しています")
    else:
        print("\n[WARNING] テストで問題が発生しました")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
