"""
P2-4-A1: execution_details生成箇所の詳細調査
統合実行時にexecution_detailsが0件になる原因を特定
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

print("=== P2-4-A1: execution_details生成箇所の詳細調査 ===")

try:
    # DSSMSIntegratedBacktesterインスタンス作成
    backtest_instance = DSSMSIntegratedBacktester()
    print(f"✅ DSSMSIntegratedBacktester初期化成功")
    
    # 対象日設定
    target_date = datetime(2025, 1, 15)
    print(f"\n[INVESTIGATION] execution_details生成プロセス調査: {target_date}")
    
    # Step 1: _process_daily_trading()内部でのexecution_details初期化確認
    print(f"\n[STEP 1] daily_result初期化でのexecution_details確認")
    
    # daily_resultの初期化をシミュレート（_process_daily_trading内部と同じ）
    daily_result = {
        'date': target_date.strftime('%Y-%m-%d'),
        'symbol': backtest_instance.current_symbol,
        'success': False,
        'portfolio_value_start': backtest_instance.portfolio_value,
        'daily_return': 0,
        'daily_return_rate': 0,
        'strategy_results': {},
        'switch_executed': False,
        'errors': [],
        'cash_balance': backtest_instance.portfolio_value,
        'position_value': 0,
        'peak_value': backtest_instance.peak_value,
        'drawdown_pct': 0,
        'cumulative_pnl': backtest_instance.cumulative_pnl,
        'total_trades': backtest_instance.total_trades_count,
        'active_positions': 0,
        'risk_status': 'Normal',
        'blocked_trades': 0,
        'risk_action': '',
        'execution_details': []  # ← ここで初期化される
    }
    
    print(f"daily_result初期化後:")
    print(f"  - execution_details初期値: {daily_result['execution_details']}")
    print(f"  - execution_details型: {type(daily_result['execution_details'])}")
    print(f"  - execution_details長さ: {len(daily_result['execution_details'])}")
    
    # Step 2: _get_optimal_symbol()実行
    print(f"\n[STEP 2] _get_optimal_symbol()実行")
    
    selected_symbol = backtest_instance._get_optimal_symbol(target_date, None)
    print(f"selected_symbol結果: {selected_symbol}")
    
    if not selected_symbol:
        print(f"❌ selected_symbol=None のため、ここで処理終了")
        print(f"  → execution_detailsは空のまま")
        print(f"  → daily_result['errors'].append('銘柄選択失敗')")
        print(f"  → return daily_result")
    else:
        print(f"✅ selected_symbol={selected_symbol} で処理継続")
        
        # Step 3: 戦略実行の確認
        print(f"\n[STEP 3] _execute_multi_strategies()内部での処理確認")
        
        # stock_dataとindex_dataの取得
        try:
            stock_data, index_data = backtest_instance._get_symbol_data(selected_symbol, target_date)
            print(f"stock_data取得結果:")
            print(f"  - stock_data is None: {stock_data is None}")
            if stock_data is not None:
                print(f"  - stock_data empty: {stock_data.empty}")
                print(f"  - stock_data length: {len(stock_data)}")
            
            if stock_data is None or stock_data.empty:
                print(f"❌ stock_dataが利用できないため、data_unavailable状態")
                print(f"  → 戦略実行せず、execution_detailsは空のまま")
                return_result = {
                    'status': 'data_unavailable',
                    'symbol': selected_symbol,
                    'date': target_date.strftime('%Y-%m-%d')
                }
                print(f"  → 戻り値: {return_result}")
                
                # このケースでのdaily_resultへの影響確認
                print(f"\n[IMPACT_CHECK] data_unavailableの場合のdaily_resultへの影響")
                print(f"  - strategy_result['execution_details']は存在しない")
                print(f"  - daily_result['execution_details']は空のまま")
                print(f"  - 結果: execution_details=0件")
                
            else:
                print(f"✅ stock_data利用可能 → 戦略実行へ進む")
                
                # MainSystemControllerの初期化確認
                print(f"\n[STEP 4] MainSystemController初期化・実行確認")
                
                if backtest_instance.main_controller is None:
                    print(f"main_controller未初期化 → 新規作成される")
                else:
                    print(f"main_controller既存: {type(backtest_instance.main_controller)}")
                
                # バックテスト期間設定の確認
                backtest_start_date = backtest_instance.dssms_backtest_start_date
                backtest_end_date = target_date + timedelta(days=7)
                warmup_days = backtest_instance.warmup_days
                
                print(f"バックテスト設定:")
                print(f"  - backtest_start_date: {backtest_start_date}")
                print(f"  - backtest_end_date: {backtest_end_date}")
                print(f"  - warmup_days: {warmup_days}")
                
                # 実際のexecute_comprehensive_backtest()は呼び出さず、
                # execution_details生成プロセスの理論的確認
                print(f"\n[THEORETICAL_CHECK] execution_details生成プロセス")
                print(f"1. MainSystemController.execute_comprehensive_backtest()実行")
                print(f"2. main_new_result['execution_results']['execution_details']生成")
                print(f"3. _convert_main_new_result()でexecution_details抽出")
                print(f"4. dssms_result['execution_details']に設定")
                print(f"5. strategy_result['execution_details']として戻り値")
                print(f"6. daily_result['execution_details'].extend()で追加")
                
                # MainSystemControllerの戻り値構造確認（既存の知識ベース）
                print(f"\n[EXPECTED_STRUCTURE] 期待される戻り値構造")
                print(f"main_new_result = {{")
                print(f"  'status': 'SUCCESS',")
                print(f"  'execution_results': {{")
                print(f"    'execution_details': [  # ← ここにトレード履歴")
                print(f"      {{'action': 'BUY', 'timestamp': '2025-01-15', ...}},")
                print(f"      {{'action': 'SELL', 'timestamp': '2025-01-16', ...}}")
                print(f"    ]")
                print(f"  }}")
                print(f"}}")
                
        except Exception as e:
            print(f"❌ _get_symbol_data()実行エラー: {str(e)}")
            import traceback
            traceback.print_exc()

    # Step 5: 実際の統合実行との比較
    print(f"\n[STEP 5] 実際の統合実行結果との比較")
    
    try:
        print(f"run_dynamic_backtest({target_date}, {target_date})実行...")
        results = backtest_instance.run_dynamic_backtest(target_date, target_date)
        
        if isinstance(results, dict) and 'daily_results' in results:
            daily_results = results['daily_results']
            if daily_results:
                latest_result = daily_results[-1]
                
                print(f"統合実行結果:")
                print(f"  - date: {latest_result.get('date')}")
                print(f"  - symbol: {latest_result.get('symbol')}")
                print(f"  - success: {latest_result.get('success')}")
                print(f"  - execution_details count: {len(latest_result.get('execution_details', []))}")
                print(f"  - errors: {latest_result.get('errors', [])}")
                
                # 詳細分析
                if latest_result.get('symbol') is None:
                    print(f"  → 統合実行でsymbol=None → 銘柄選択失敗")
                    print(f"  → この時点でexecution_detailsは空")
                elif len(latest_result.get('execution_details', [])) == 0:
                    print(f"  → 統合実行でsymbol選択成功だがexecution_details=0")
                    print(f"  → 戦略実行またはresult変換で問題発生")
                else:
                    print(f"  → 統合実行で正常にexecution_details生成")
        
    except Exception as e:
        print(f"❌ 統合実行エラー: {str(e)}")
        import traceback
        traceback.print_exc()

    # 調査結果まとめ
    print(f"\n[SUMMARY] P2-4-A1調査結果")
    print(f"1. execution_details初期化: daily_result['execution_details'] = []")
    print(f"2. _get_optimal_symbol()結果によって処理分岐:")
    print(f"   - symbol=None → 処理終了、execution_details=0")
    print(f"   - symbol有効 → _execute_multi_strategies()実行")
    print(f"3. 戦略実行でのexecution_details生成:")
    print(f"   - stock_data無効 → data_unavailable、execution_details=0")
    print(f"   - stock_data有効 → MainSystemController実行 → execution_details生成")
    print(f"4. _convert_main_new_result()でexecution_details変換")
    print(f"5. daily_result['execution_details'].extend()で追加")

except Exception as e:
    print(f"❌ 調査実行エラー: {str(e)}")
    import traceback
    traceback.print_exc()

print(f"\n=== P2-4-A1調査完了 ===")