"""
P3調査: P1-4 実際の実行状態確認デバッグスクリプト
"""

import sys
import os
from datetime import datetime
sys.path.append('.')

def investigate_p1_4_actual_execution():
    """P1-4: 実際の実行状態の詳細確認"""
    
    print("=== P1-4: 実際の実行状態確認 ===")
    
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        backtester = DSSMSIntegratedBacktester()
        print("✅ DSSMSIntegratedBacktester初期化成功")
        
        # 短期テスト: 1日だけ実行
        start_date = datetime(2025, 1, 15)
        end_date = datetime(2025, 1, 15)
        
        print(f"[TEST] 短期実行テスト: {start_date} -> {end_date}")
        print(f"[TEST] daily_results初期状態: {len(backtester.daily_results)}件")
        
        result = backtester.run_dynamic_backtest(start_date, end_date)
        
        print(f"✅ run_dynamic_backtest実行完了")
        print(f"   Result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"   Result keys: {list(result.keys())}")
            print(f"   Success: {result.get('success', 'キーなし')}")
            print(f"   Total trading days: {result.get('total_trading_days', 'キーなし')}")
            print(f"   Successful days: {result.get('successful_days', 'キーなし')}")
        else:
            print("   Result: Not dict")
            
        print(f"[TEST] daily_results実行後状態: {len(backtester.daily_results)}件")
        
        if backtester.daily_results:
            latest_result = backtester.daily_results[-1]
            print(f"[TEST] 最新日次結果:")
            print(f"   - date: {latest_result.get('date')}")
            print(f"   - symbol: {latest_result.get('symbol')}")
            print(f"   - success: {latest_result.get('success')}")
            print(f"   - execution_details: {len(latest_result.get('execution_details', []))}")
            print(f"   - errors: {latest_result.get('errors', [])}")
        
        return True, result
        
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False, None

if __name__ == "__main__":
    success, result = investigate_p1_4_actual_execution()
    
    if success:
        print(f"\n✅ P1-4調査完了: 実行成功")
    else:
        print(f"\n❌ P1-4調査失敗: 実行失敗")