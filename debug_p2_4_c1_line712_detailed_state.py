"""
P2-4-C1: Line 712付近の詳細状態確認デバッグスクリプト
_process_daily_trading()内でのselected_symbol代入プロセスを詳細追跡
"""

import os
import sys
sys.path.append(os.getcwd())

from datetime import datetime, timedelta
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
import logging

# ロギング設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s'
)

print("=== P2-4-C1: Line 712付近の詳細状態確認 ===")

try:
    # DSSMSIntegratedBacktesterインスタンス作成
    backtest_instance = DSSMSIntegratedBacktester()
    print(f"✅ DSSMSIntegratedBacktester初期化成功")
    
    # 対象日設定
    target_date = datetime(2025, 1, 15)
    print(f"\n[INVESTIGATION] Line 712付近の詳細状態確認: {target_date}")
    
    # Step 1: _get_optimal_symbol()の引数確認
    print(f"\n[STEP 1] _get_optimal_symbol()の引数確認")
    
    # _process_daily_trading()内での呼び出しをシミュレート
    print(f"Line 712呼び出しシミュレーション:")
    print(f"  target_date: {target_date} (type: {type(target_date)})")
    print(f"  target_symbols: None (default)")
    
    # Step 2: モジュールレベル変数確認
    print(f"\n[STEP 2] モジュールレベル変数確認")
    
    # dss_availableとfallback_policy_availableの状態確認
    from src.dssms import dssms_integrated_main
    
    print(f"モジュール変数状態:")
    print(f"  dss_available: {getattr(dssms_integrated_main, 'dss_available', 'UNDEFINED')}")
    print(f"  fallback_policy_available: {getattr(dssms_integrated_main, 'fallback_policy_available', 'UNDEFINED')}")
    
    # Step 3: self.dss_coreの状態確認
    print(f"\n[STEP 3] self.dss_coreの状態確認")
    
    print(f"初期化前のself.dss_core状態:")
    print(f"  self.dss_core: {backtest_instance.dss_core}")
    print(f"  self._dss_initialized: {getattr(backtest_instance, '_dss_initialized', 'UNDEFINED')}")
    
    # ensure_dss_core()実行
    dss_result = backtest_instance.ensure_dss_core()
    
    print(f"ensure_dss_core()実行後:")
    print(f"  result: {dss_result}")
    print(f"  self.dss_core: {backtest_instance.dss_core}")
    print(f"  self.dss_core is None: {backtest_instance.dss_core is None}")
    print(f"  self._dss_initialized: {getattr(backtest_instance, '_dss_initialized', 'UNDEFINED')}")
    
    # Step 4: _get_optimal_symbol()実行パス確認
    print(f"\n[STEP 4] _get_optimal_symbol()実行パス確認")
    
    # Line 1568の条件確認
    print(f"Line 1568条件: if self.dss_core and dss_available:")
    print(f"  self.dss_core: {backtest_instance.dss_core is not None}")
    print(f"  dss_available: {getattr(dssms_integrated_main, 'dss_available', 'UNDEFINED')}")
    print(f"  両方True: {backtest_instance.dss_core is not None and getattr(dssms_integrated_main, 'dss_available', False)}")
    
    if backtest_instance.dss_core is not None and getattr(dssms_integrated_main, 'dss_available', False):
        print(f"  → DSS Core V3による動的選択パス")
        
        # run_daily_selection()実行
        print(f"DSS Core V3.run_daily_selection()実行...")
        try:
            dss_result = backtest_instance.dss_core.run_daily_selection(target_date)
            print(f"  run_daily_selection()結果: {dss_result}")
            selected_symbol = dss_result.get('selected_symbol')
            print(f"  selected_symbol: {selected_symbol}")
            print(f"  selected_symbol type: {type(selected_symbol)}")
            print(f"  bool(selected_symbol): {bool(selected_symbol)}")
        except Exception as e:
            print(f"  ❌ run_daily_selection()エラー: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  → フォールバックパス（nikkei225_screener使用）")
        
        # nikkei225_screenerの確認
        print(f"self.nikkei225_screener: {backtest_instance.nikkei225_screener is not None}")
        
        if backtest_instance.nikkei225_screener:
            print(f"  → nikkei225_screenerパス実行")
            try:
                available_funds = backtest_instance.portfolio_value * 0.8
                print(f"  available_funds: {available_funds}")
                
                filtered_symbols = backtest_instance.nikkei225_screener.get_filtered_symbols(available_funds)
                print(f"  filtered_symbols length: {len(filtered_symbols)}")
                print(f"  filtered_symbols[:5]: {filtered_symbols[:5] if filtered_symbols else []}")
                
                if filtered_symbols:
                    # fallback_policy_availableの確認
                    fallback_available = getattr(dssms_integrated_main, 'fallback_policy_available', False)
                    print(f"  fallback_policy_available: {fallback_available}")
                    
                    if fallback_available:
                        print(f"    → fallback_policy使用パス")
                    else:
                        print(f"    → _advanced_ranking_selection()直接呼び出しパス")
                        
                        # _advanced_ranking_selection()の実行
                        try:
                            selected = backtest_instance._advanced_ranking_selection(filtered_symbols, target_date)
                            print(f"    _advanced_ranking_selection()結果: {selected}")
                            print(f"    selected type: {type(selected)}")
                            print(f"    bool(selected): {bool(selected)}")
                        except Exception as e:
                            print(f"    ❌ _advanced_ranking_selection()エラー: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    print(f"  ❌ filtered_symbols is empty")
            except Exception as e:
                print(f"  ❌ nikkei225_screenerエラー: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ❌ self.nikkei225_screener is None")

    # Step 5: 実際の_get_optimal_symbol()実行
    print(f"\n[STEP 5] 実際の_get_optimal_symbol()実行")
    
    try:
        print(f"_get_optimal_symbol({target_date}, None)実行...")
        selected_symbol = backtest_instance._get_optimal_symbol(target_date, None)
        
        print(f"戻り値詳細:")
        print(f"  selected_symbol: {selected_symbol}")
        print(f"  type: {type(selected_symbol)}")
        print(f"  repr: {repr(selected_symbol)}")
        print(f"  str: {str(selected_symbol)}")
        print(f"  bool: {bool(selected_symbol)}")
        print(f"  is None: {selected_symbol is None}")
        print(f"  == None: {selected_symbol == None}")
        print(f"  == '': {selected_symbol == ''}")
        print(f"  len (if str): {len(selected_symbol) if isinstance(selected_symbol, str) else 'N/A'}")
        
        # Line 713-714の判定シミュレート
        print(f"\nLine 713-714判定シミュレーション:")
        print(f"  if not selected_symbol:")
        if not selected_symbol:
            print(f"    → True (Falsyと判定)")
            print(f"    → daily_result['errors'].append('銘柄選択失敗')")
            print(f"    → return daily_result")
        else:
            print(f"    → False (Truthyと判定)")
            print(f"    → 処理継続")
            
    except Exception as e:
        print(f"❌ _get_optimal_symbol()実行エラー: {str(e)}")
        import traceback
        traceback.print_exc()

    # Step 6: 統合実行での実際の挙動確認
    print(f"\n[STEP 6] 統合実行での実際の挙動確認")
    
    try:
        print(f"_process_daily_trading({target_date})実行...")
        daily_result = backtest_instance._process_daily_trading(target_date)
        
        print(f"_process_daily_trading()結果:")
        print(f"  symbol: {daily_result.get('symbol')}")
        print(f"  success: {daily_result.get('success')}")
        print(f"  errors: {daily_result.get('errors', [])}")
        print(f"  execution_details: {len(daily_result.get('execution_details', []))}")
        
        # Line 712での実際の処理を推定
        if daily_result.get('symbol') is None and '銘柄選択失敗' in daily_result.get('errors', []):
            print(f"\n[ANALYSIS] Line 713-714でのFalsy判定確認:")
            print(f"  → selected_symbolがFalsyと判定され、早期returnされた")
            print(f"  → 'symbol': Noneが設定された")
        
    except Exception as e:
        print(f"❌ _process_daily_trading()実行エラー: {str(e)}")
        import traceback
        traceback.print_exc()

    print(f"\n[SUMMARY] P2-4-C1調査結果")
    print(f"調査完了: Line 712付近の詳細状態確認")

except Exception as e:
    print(f"❌ 調査実行エラー: {str(e)}")
    import traceback
    traceback.print_exc()

print(f"\n=== P2-4-C1調査完了 ===")