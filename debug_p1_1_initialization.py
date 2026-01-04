"""
P3調査: DSSMSIntegratedBacktester初期化調査デバッグスクリプト

P1-1の詳細調査を実行し、初期化時の問題を特定する
"""

import sys
import os
sys.path.append('.')

def investigate_p1_1_initialization():
    """P1-1: DSSMSIntegratedBacktester初期化詳細調査"""
    
    print("=== P1-1: DSSMSIntegratedBacktester初期化詳細調査 ===")
    
    try:
        # Step 1: Import確認
        print("\n[STEP 1] Import確認開始...")
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        print("✅ DSSMSIntegratedBacktester import: 成功")
        
        # Step 2: 初期化実行
        print("\n[STEP 2] 初期化実行開始...")
        backtester = DSSMSIntegratedBacktester()
        print("✅ DSSMSIntegratedBacktester初期化: 成功")
        
        # Step 3: オブジェクト詳細確認
        print(f"\n[STEP 3] オブジェクト詳細:")
        print(f"  - Type: {type(backtester)}")
        print(f"  - __dict__ keys count: {len(backtester.__dict__)}")
        print(f"  - __dict__ keys: {list(backtester.__dict__.keys())}")
        
        # Step 4: 重要属性の確認
        print(f"\n[STEP 4] 重要属性確認:")
        important_attrs = ['logger', 'dss_core', 'advanced_ranking', 'market_analyzer', 'dynamic_strategy_selector']
        for attr in important_attrs:
            if hasattr(backtester, attr):
                value = getattr(backtester, attr)
                print(f"  - {attr}: {type(value)} = {value}")
            else:
                print(f"  - {attr}: 属性なし")
        
        # Step 5: main実行メソッドの確認
        print(f"\n[STEP 5] main実行メソッドの確認:")
        main_methods = ['run_integrated_backtest_with_strategy_selection', '_process_trading_day', '_get_optimal_symbol']
        for method in main_methods:
            if hasattr(backtester, method):
                print(f"  - {method}: メソッド存在")
            else:
                print(f"  - {method}: メソッドなし")
        
        return True, backtester
        
    except ImportError as e:
        print(f"❌ [P1-1] Import Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False, None
        
    except Exception as e:
        print(f"❌ [P1-1] Initialization Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False, None

if __name__ == "__main__":
    success, backtester = investigate_p1_1_initialization()
    
    if success:
        print(f"\n✅ P1-1調査完了: DSSMSIntegratedBacktester初期化成功")
        print(f"   Object ID: {id(backtester)}")
    else:
        print(f"\n❌ P1-1調査失敗: DSSMSIntegratedBacktester初期化失敗")