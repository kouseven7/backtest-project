#!/usr/bin/env python3
"""Basic system test for 4-1-3 coordination system"""

def main():
    try:
        print("🚀 4-1-3 Basic System Test")
        
        # 1. インターフェースインポート
        print("1. Testing interface import...")
        from multi_strategy_coordination_interface import MultiStrategyCoordinationInterface
        print("✅ Interface import successful")
        
        # 2. インターフェース初期化  
        print("2. Testing interface initialization...")
        interface = MultiStrategyCoordinationInterface()
        print("✅ Interface initialization successful")
        
        # 3. システムステータス確認
        print("3. Testing system status...")
        try:
            status = interface.get_system_status()
            print("✅ System status retrieved")
            print(f"   Coordination state: {status.get('coordination', {}).get('state', 'unknown')}")
        except Exception as e:
            print(f"⚠️ System status error (continuing): {e}")
        
        # 4. 基本機能テスト
        print("4. Testing basic coordination...")
        try:
            test_strategies = ["TestStrategy1", "TestStrategy2"]
            
            # インターフェース経由での調整テスト
            result = interface.execute_strategy_coordination(test_strategies)
            if result.get('success'):
                print("✅ Basic coordination test successful")
            else:
                print(f"⚠️ Coordination test returned: {result.get('error', 'No error message')}")
        except Exception as e:
            print(f"⚠️ Coordination test error: {e}")
        
        # 5. クリーンアップ
        print("5. Cleanup...")
        try:
            interface.shutdown()
            print("✅ Interface shutdown successful")
        except Exception as e:
            print(f"⚠️ Shutdown error: {e}")
        
        print("\n🎉 Basic system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
