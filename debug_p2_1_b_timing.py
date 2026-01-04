"""
P3調査: P2-1-B 初期化タイミングによる動作差異の調査

統合実行（P1-4）vs 個別実行（P2-1-A）の初期化タイミングの差異を調査し、
なぜ同じメソッドが異なる結果を返すのかを解明する
"""

import sys
import os
from datetime import datetime
sys.path.append('.')

def investigate_p2_1_b_initialization_timing():
    """P2-1-B: 初期化タイミングによる動作差異の調査"""
    
    print("=== P2-1-B: 初期化タイミング差異調査 ===")
    
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        print("\n[TEST 1] 統合実行フローの完全シミュレーション")
        backtester = DSSMSIntegratedBacktester()
        
        print(f"初期化直後の状態:")
        print(f"  - dss_core: {backtester.dss_core}")
        print(f"  - _dss_initialized: {backtester._dss_initialized}")
        
        # 統合実行フロー: run_dynamic_backtest -> _process_daily_trading -> _get_optimal_symbol
        target_date = datetime(2025, 1, 15)
        
        print(f"\n[STEP 1] 統合実行フローでの_get_optimal_symbol()呼び出し")
        
        # _process_daily_trading() の一部を再現
        daily_result = {
            'date': target_date.strftime('%Y-%m-%d'),
            'symbol': backtester.current_symbol,
            'success': False,
            'execution_details': []
        }
        
        print(f"daily_result初期状態: {daily_result}")
        
        # 1. DSS Core V3による銘柄選択
        print(f"\n[KEY CALL] selected_symbol = self._get_optimal_symbol({target_date}, None)")
        selected_symbol = backtester._get_optimal_symbol(target_date, None)
        
        print(f"selected_symbol結果: {selected_symbol}")
        
        if not selected_symbol:
            daily_result['errors'] = ['銘柄選択失敗']
            print(f"❌ selected_symbolがNone - daily_resultにエラー追加")
            print(f"early return daily_result: {daily_result}")
            return False, daily_result
        else:
            print(f"✅ selected_symbolは正常: {selected_symbol}")
            return True, selected_symbol
        
    except Exception as e:
        print(f"❌ 調査実行エラー: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False, None

def investigate_run_dynamic_backtest_flow():
    """統合実行フローの完全再現"""
    
    print(f"\n=== [TEST 2] run_dynamic_backtest()完全フローのテスト ===")
    
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        backtester = DSSMSIntegratedBacktester()
        
        start_date = datetime(2025, 1, 15)
        end_date = datetime(2025, 1, 15)
        
        print(f"run_dynamic_backtest({start_date}, {end_date})実行...")
        
        result = backtester.run_dynamic_backtest(start_date, end_date)
        
        print(f"run_dynamic_backtest結果:")
        print(f"  - Result type: {type(result)}")
        print(f"  - daily_results count: {len(backtester.daily_results)}")
        
        if backtester.daily_results:
            latest_result = backtester.daily_results[-1]
            print(f"  - 最新日次結果:")
            print(f"    - date: {latest_result.get('date')}")
            print(f"    - symbol: {latest_result.get('symbol')}")
            print(f"    - success: {latest_result.get('success')}")
            print(f"    - execution_details count: {len(latest_result.get('execution_details', []))}")
            print(f"    - errors: {latest_result.get('errors', [])}")
            
            if latest_result.get('symbol') is None:
                print(f"❌ 統合実行でsymbol=None問題を確認")
                return False, latest_result
            else:
                print(f"✅ 統合実行でも正常にsymbol選択: {latest_result.get('symbol')}")
                return True, latest_result
                
        return False, None
        
    except Exception as e:
        print(f"❌ 統合実行フローエラー: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False, None

if __name__ == "__main__":
    
    # Test 1: 個別メソッド呼び出し
    success1, result1 = investigate_p2_1_b_initialization_timing()
    
    # Test 2: 統合フロー実行
    success2, result2 = investigate_run_dynamic_backtest_flow()
    
    print(f"\n=== P2-1-B調査結果まとめ ===")
    print(f"Test 1 (個別メソッド): {'成功' if success1 else '失敗'} - 結果: {result1}")
    print(f"Test 2 (統合フロー):   {'成功' if success2 else '失敗'} - 結果: {result2}")
    
    if success1 and not success2:
        print(f"\n🚨 矛盾確認: 個別は成功、統合は失敗")
    elif success1 and success2:
        print(f"\n✅ 両方成功: 問題解決の可能性")
    elif not success1 and not success2:
        print(f"\n❌ 両方失敗: 根本的な問題")
    else:
        print(f"\n❓ 予期しないパターン")