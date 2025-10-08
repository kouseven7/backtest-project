"""
Test: Main Integration System
File: test_main_integration.py
Description: 
  4-1-1「main.py への戦略セレクター統合」の包括テスト

Author: imega
Created: 2025-07-20
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# テスト対象のインポート
try:
    from config.multi_strategy_manager import MultiStrategyManager, ExecutionMode
    from config.strategy_execution_adapter import StrategyExecutionAdapter, StrategyExecutionConfig, ExecutionMethod
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Integration components not available: {e}")
    INTEGRATION_AVAILABLE = False

def test_multi_strategy_manager():
    """MultiStrategyManagerのテスト"""
    print("=== MultiStrategyManager Test ===")
    
    try:
        # マネージャー初期化
        manager = MultiStrategyManager()
        
        # 初期化テスト
        init_success = manager.initialize_systems()
        assert init_success, "System initialization failed"
        print("[OK] System initialization: PASSED")
        
        # 実行フローテスト
        test_strategies = ["VWAP_Bounce", "GC_Strategy", "Breakout"]
        test_market_data = {"trend": "uptrend", "volatility": "medium"}
        
        result = manager.execute_multi_strategy_flow(test_market_data, test_strategies)
        
        assert result is not None, "Execution result is None"
        assert hasattr(result, 'execution_mode'), "Result missing execution_mode"
        assert hasattr(result, 'status'), "Result missing status"
        print("[OK] Multi-strategy execution: PASSED")
        
        # サマリーテスト
        summary = manager.get_execution_summary()
        assert isinstance(summary, dict), "Summary should be dict"
        assert 'current_status' in summary, "Summary missing current_status"
        print("[OK] Summary generation: PASSED")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] MultiStrategyManager test failed: {e}")
        return False

def test_strategy_execution_adapter():
    """StrategyExecutionAdapterのテスト"""
    print("\n=== StrategyExecutionAdapter Test ===")
    
    try:
        # アダプター初期化
        adapter = StrategyExecutionAdapter()
        print("[OK] Adapter initialization: PASSED")
        
        # パラメータ取得テスト
        params = adapter.get_strategy_parameters("VWAPBounceStrategy")
        assert isinstance(params, dict), "Parameters should be dict"
        assert len(params) > 0, "Parameters should not be empty"
        print("[OK] Parameter retrieval: PASSED")
        
        # 単一戦略実行テスト
        test_market_data = {"price": [100, 101, 99, 102], "volume": [1000, 1100, 900, 1200]}
        result = adapter.execute_single_strategy("VWAPBounceStrategy", test_market_data)
        
        assert result is not None, "Execution result is None"
        assert hasattr(result, 'success'), "Result missing success field"
        assert hasattr(result, 'strategy_name'), "Result missing strategy_name"
        print("[OK] Single strategy execution: PASSED")
        
        # サマリーテスト
        summary = adapter.get_execution_summary()
        assert isinstance(summary, dict), "Summary should be dict"
        print("[OK] Execution summary: PASSED")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] StrategyExecutionAdapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_flow():
    """統合フローのテスト"""
    print("\n=== Integration Flow Test ===")
    
    try:
        # 設定ファイルの確認
        config_path = "config/main_integration_config.json"
        if not os.path.exists(config_path):
            print(f"ℹ️  Config file not found: {config_path}. Using defaults.")
        
        # マネージャー初期化
        manager = MultiStrategyManager()
        init_success = manager.initialize_systems()
        
        if not init_success:
            print("[WARNING]  System initialization failed, but continuing with fallback")
        
        # テストデータ準備
        test_strategies = ["VWAP_Bounce", "GC_Strategy", "Breakout", "Opening_Gap"]
        test_market_data = {
            "trend": "uptrend",
            "volatility": "medium"
        }
        
        # 実行モード別テスト
        execution_modes = [ExecutionMode.LEGACY_ONLY, ExecutionMode.HYBRID]
        
        for mode in execution_modes:
            print(f"\n  Testing {mode.value} mode...")
            manager.execution_mode = mode
            
            result = manager.execute_multi_strategy_flow(test_market_data, test_strategies)
            
            assert result is not None, f"Result is None for mode {mode.value}"
            assert result.execution_mode == mode, f"Execution mode mismatch"
            
            print(f"    [OK] {mode.value}: Strategy count = {len(result.selected_strategies)}")
            print(f"    [OK] {mode.value}: Execution time = {result.execution_time:.3f}s")
            print(f"    [OK] {mode.value}: Status = {result.status.value}")
        
        # 統合サマリーテスト
        final_summary = manager.get_execution_summary()
        print(f"\n  Final Summary:")
        print(f"    Total executions: {final_summary.get('total_executions', 0)}")
        print(f"    System health: {final_summary.get('system_health', {})}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling_and_fallback():
    """エラーハンドリングとフォールバックのテスト"""
    print("\n=== Error Handling & Fallback Test ===")
    
    try:
        # 意図的にエラーを発生させるテスト
        manager = MultiStrategyManager()
        
        # 無効なデータでのテスト
        invalid_data = None
        invalid_strategies = []
        
        result = manager.execute_multi_strategy_flow(invalid_data, invalid_strategies)
        
        # エラー時でも結果が返されることを確認
        assert result is not None, "Result should not be None even on error"
        print("[OK] Graceful error handling: PASSED")
        
        # フォールバック動作確認
        if hasattr(result, 'status'):
            print(f"    Status: {result.status.value}")
        if hasattr(result, 'errors') and result.errors:
            print(f"    Errors captured: {len(result.errors)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error handling test failed: {e}")
        return False

def performance_benchmark():
    """パフォーマンステスト"""
    print("\n=== Performance Benchmark ===")
    
    try:
        manager = MultiStrategyManager()
        manager.initialize_systems()
        
        # 複数回実行して平均時間を測定
        test_strategies = ["VWAP_Bounce", "GC_Strategy"]
        test_market_data = {"trend": "uptrend", "volatility": "medium"}
        
        execution_times = []
        
        for i in range(5):
            start_time = time.time()
            result = manager.execute_multi_strategy_flow(test_market_data, test_strategies)
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
        
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        
        print(f"  Execution times (5 runs):")
        print(f"    Average: {avg_time:.3f}s")
        print(f"    Min: {min_time:.3f}s") 
        print(f"    Max: {max_time:.3f}s")
        
        # パフォーマンス基準（1秒以内）
        assert avg_time < 1.0, f"Average execution time too slow: {avg_time:.3f}s"
        print("[OK] Performance benchmark: PASSED")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Performance benchmark failed: {e}")
        return False

def main():
    """メインテスト実行"""
    print("[ROCKET] 4-1-1「main.py への戦略セレクター統合」包括テスト開始")
    print("=" * 60)
    
    if not INTEGRATION_AVAILABLE:
        print("[ERROR] Integration components not available. Exiting.")
        return False
    
    test_results = []
    
    # 個別コンポーネントテスト
    test_results.append(test_multi_strategy_manager())
    test_results.append(test_strategy_execution_adapter())
    
    # 統合テスト
    test_results.append(test_integration_flow())
    test_results.append(test_error_handling_and_fallback())
    
    # パフォーマンステスト
    test_results.append(performance_benchmark())
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("[CHART] テスト結果サマリー")
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"成功: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("[SUCCESS] 全てのテストが成功しました！")
        print("[OK] 4-1-1「main.py への戦略セレクター統合」実装準備完了")
    else:
        print("[WARNING]  一部テストが失敗しました。実装を確認してください。")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
