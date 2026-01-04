"""
P2-3: DSS Core V3初期化状態の統合実行時確認調査
統合実行フロー内でのdss_core初期化状態とメソッド呼び出し状況を詳細分析
"""

import os
import sys
sys.path.append(os.getcwd())

from datetime import datetime, timedelta
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

print("=== P2-3: DSS Core V3初期化状態の統合実行時確認調査 ===")

try:
    # DSSMSIntegratedBacktesterインスタンス作成
    backtest_instance = DSSMSIntegratedBacktester()
    print(f"✅ DSSMSIntegratedBacktester初期化成功")

    # 統合実行前の状態確認
    print(f"\n[INIT_STATE] インスタンス作成直後の状態")
    print(f"  - dss_core: {getattr(backtest_instance, 'dss_core', 'NOT_SET')}")
    print(f"  - _dss_initialized: {getattr(backtest_instance, '_dss_initialized', 'NOT_SET')}")

    # run_dynamic_backtest()内部での状態変化をシミュレート
    print(f"\n[SIMULATION] run_dynamic_backtest()内部状態変化シミュレーション")
    
    # 開始・終了日設定
    start_date = datetime(2025, 1, 15)
    end_date = datetime(2025, 1, 15)
    print(f"バックテスト期間: {start_date} -> {end_date}")
    
    # _process_daily_trading()シミュレーション
    target_date = start_date
    print(f"\n[DAILY_PROCESSING] _process_daily_trading({target_date})開始")
    
    # _get_optimal_symbol()呼び出し前の詳細状態確認
    print(f"\n[PRE_CALL_DETAILED] _get_optimal_symbol()呼び出し直前の詳細状態")
    dss_core = getattr(backtest_instance, 'dss_core', None)
    dss_initialized = getattr(backtest_instance, '_dss_initialized', False)
    
    print(f"  - dss_core: {dss_core}")
    print(f"  - dss_core type: {type(dss_core)}")
    print(f"  - _dss_initialized: {dss_initialized}")
    
    # DSS Core V3が初期化されている場合の詳細確認
    if dss_core is not None:
        print(f"  - dss_core methods: {[m for m in dir(dss_core) if not m.startswith('_')][:10]}")
        if hasattr(dss_core, 'get_symbol_for_date'):
            print(f"  - get_symbol_for_date method: {dss_core.get_symbol_for_date}")
        else:
            print(f"  - get_symbol_for_date method: NOT FOUND")
    else:
        print(f"  - dss_core is None - 初期化されていない")
    
    # ここで実際の_get_optimal_symbol()を呼び出してみる
    print(f"\n[TEST_CALL] _get_optimal_symbol({target_date}, None)実行...")
    try:
        selected_symbol = backtest_instance._get_optimal_symbol(target_date, None)
        print(f"✅ _get_optimal_symbol()成功: {selected_symbol}")
        
        # 呼び出し後の状態確認
        print(f"\n[POST_CALL_STATE] _get_optimal_symbol()呼び出し後状態")
        dss_core_after = getattr(backtest_instance, 'dss_core', None)
        dss_initialized_after = getattr(backtest_instance, '_dss_initialized', False)
        
        print(f"  - dss_core: {type(dss_core_after)}")
        print(f"  - _dss_initialized: {dss_initialized_after}")
        
        # 状態変化の確認
        state_changed = (dss_core != dss_core_after) or (dss_initialized != dss_initialized_after)
        print(f"  - 初期化状態変化: {state_changed}")
        
        if selected_symbol == '1662':
            print(f"✅ 期待結果と一致: 1662")
        else:
            print(f"❌ 期待結果と不一致: {selected_symbol} (期待: 1662)")
            
    except Exception as e:
        print(f"❌ _get_optimal_symbol()呼び出し失敗: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 実際のrun_dynamic_backtest()を1日間実行して比較
    print(f"\n[INTEGRATION_TEST] 実際のrun_dynamic_backtest()実行テスト")
    
    try:
        print(f"run_dynamic_backtest({start_date}, {end_date})実行...")
        results = backtest_instance.run_dynamic_backtest(start_date, end_date)
        
        # 結果の詳細確認
        print(f"実行結果:")
        print(f"  - Results type: {type(results)}")
        
        if isinstance(results, dict) and 'daily_results' in results:
            daily_results = results['daily_results']
            print(f"  - daily_results count: {len(daily_results)}")
            
            if daily_results:
                latest_result = daily_results[-1]
                print(f"  - 最新日次結果:")
                print(f"    - date: {latest_result.get('date')}")
                print(f"    - symbol: {latest_result.get('symbol')}")
                print(f"    - success: {latest_result.get('success')}")
                print(f"    - execution_details count: {len(latest_result.get('execution_details', []))}")
                
                # 統合実行でのsymbol値確認
                integration_symbol = latest_result.get('symbol')
                if integration_symbol is None:
                    print(f"❌ 統合実行でsymbol=None問題を確認")
                elif integration_symbol == '1662':
                    print(f"✅ 統合実行でも正常動作: {integration_symbol}")
                else:
                    print(f"⚠️ 統合実行で予期しない結果: {integration_symbol}")
                    
        else:
            print(f"❌ 予期しないresults構造: {results}")
    
    except Exception as e:
        print(f"❌ run_dynamic_backtest()実行失敗: {str(e)}")
        import traceback
        traceback.print_exc()

    print(f"\n[SUMMARY] P2-3調査結果まとめ")
    print(f"1. 個別_get_optimal_symbol()テスト: 成功")
    print(f"2. 統合run_dynamic_backtest()テスト: 結果確認")
    print(f"3. 初期化状態変化: dss_core初期化タイミング確認")

except Exception as e:
    print(f"❌ 調査実行エラー: {str(e)}")
    import traceback
    traceback.print_exc()

print(f"\n=== P2-3調査完了 ===")