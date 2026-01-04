"""
P2-4-A1: execution_details生成箇所の詳細調査（修正版）
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

print("=== P2-4-A1: execution_details生成箇所の詳細調査（修正版） ===")

try:
    # DSSMSIntegratedBacktesterインスタンス作成
    backtest_instance = DSSMSIntegratedBacktester()
    print(f"✅ DSSMSIntegratedBacktester初期化成功")
    
    # 対象日設定
    target_date = datetime(2025, 1, 15)
    print(f"\n[INVESTIGATION] execution_details生成プロセス調査: {target_date}")
    
    # Step 1: _process_daily_trading()内部でのexecution_details初期化確認
    print(f"\n[STEP 1] daily_result初期化でのexecution_details確認")
    
    # backtest_instanceの属性を安全に取得
    portfolio_value = getattr(backtest_instance, 'portfolio_value', 1000000)
    peak_value = getattr(backtest_instance, 'peak_value', portfolio_value)
    cumulative_pnl = getattr(backtest_instance, 'cumulative_pnl', 0.0)
    total_trades_count = getattr(backtest_instance, 'total_trades_count', 0)
    current_symbol = getattr(backtest_instance, 'current_symbol', 'N/A')
    
    print(f"backtest_instance現在状態:")
    print(f"  - portfolio_value: {portfolio_value}")
    print(f"  - peak_value: {peak_value}")
    print(f"  - cumulative_pnl: {cumulative_pnl}")
    print(f"  - total_trades_count: {total_trades_count}")
    print(f"  - current_symbol: {current_symbol}")
    
    # daily_resultの初期化をシミュレート（_process_daily_trading内部と同じ）
    daily_result = {
        'date': target_date.strftime('%Y-%m-%d'),
        'symbol': current_symbol,
        'success': False,
        'portfolio_value_start': portfolio_value,
        'daily_return': 0,
        'daily_return_rate': 0,
        'strategy_results': {},
        'switch_executed': False,
        'errors': [],
        'cash_balance': portfolio_value,
        'position_value': 0,
        'peak_value': peak_value,
        'drawdown_pct': 0,
        'cumulative_pnl': cumulative_pnl,
        'total_trades': total_trades_count,
        'active_positions': 0,
        'risk_status': 'Normal',
        'blocked_trades': 0,
        'risk_action': '',
        'execution_details': []  # ← ここで初期化される
    }
    
    print(f"\ndaily_result初期化後:")
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
        
        # 統合実行での実際の挙動を確認
        print(f"\n[VERIFICATION] 実際の統合実行での挙動確認")
        try:
            # 1日間だけのテスト実行
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
                    
                    # 銘柄選択失敗の場合のerrors内容確認
                    if latest_result.get('symbol') is None:
                        print(f"\n✅ 統合実行でもsymbol=None確認")
                        print(f"  → これが原因でexecution_details=0となる")
                        print(f"  → Line 715-716: daily_result['errors'].append('銘柄選択失敗')")
                        print(f"  → return daily_result (execution_details=[]のまま)")
        except Exception as e:
            print(f"❌ 統合実行検証エラー: {str(e)}")
            import traceback
            traceback.print_exc()
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
                
                # このケースでの統合実行検証
                print(f"\n[VERIFICATION] data_unavailableケースでの統合実行検証")
                try:
                    results = backtest_instance.run_dynamic_backtest(target_date, target_date)
                    
                    if isinstance(results, dict) and 'daily_results' in results:
                        daily_results = results['daily_results']
                        if daily_results:
                            latest_result = daily_results[-1]
                            print(f"統合実行結果（data_unavailableケース）:")
                            print(f"  - symbol: {latest_result.get('symbol')}")
                            print(f"  - success: {latest_result.get('success')}")
                            print(f"  - execution_details count: {len(latest_result.get('execution_details', []))}")
                            print(f"  - errors: {latest_result.get('errors', [])}")
                            
                            # strategy_resultsの確認
                            strategy_results = latest_result.get('strategy_results', {})
                            print(f"  - strategy_results keys: {list(strategy_results.keys())}")
                            
                            for strategy_name, strategy_result in strategy_results.items():
                                if isinstance(strategy_result, dict):
                                    exec_details = strategy_result.get('execution_details', [])
                                    print(f"    - {strategy_name}: {len(exec_details)} execution_details")
                except Exception as e:
                    print(f"❌ data_unavailable統合実行検証エラー: {str(e)}")
                    
            else:
                print(f"✅ stock_data利用可能 → 戦略実行へ進む")
                print(f"  → MainSystemController.execute_comprehensive_backtest()実行")
                print(f"  → main_new_result['execution_results']['execution_details']生成")
                print(f"  → _convert_main_new_result()でexecution_details抽出")
                print(f"  → dssms_result['execution_details']に設定")
                
                # 実際の統合実行での検証
                print(f"\n[VERIFICATION] 正常ケースでの統合実行検証")
                try:
                    results = backtest_instance.run_dynamic_backtest(target_date, target_date)
                    
                    if isinstance(results, dict) and 'daily_results' in results:
                        daily_results = results['daily_results']
                        if daily_results:
                            latest_result = daily_results[-1]
                            print(f"統合実行結果（正常ケース）:")
                            print(f"  - symbol: {latest_result.get('symbol')}")
                            print(f"  - success: {latest_result.get('success')}")
                            print(f"  - execution_details count: {len(latest_result.get('execution_details', []))}")
                            
                            # execution_detailsの内容確認
                            exec_details = latest_result.get('execution_details', [])
                            if exec_details:
                                print(f"  - execution_details[0]: {exec_details[0]}")
                            
                            # strategy_resultsの詳細確認
                            strategy_results = latest_result.get('strategy_results', {})
                            print(f"  - strategy_results keys: {list(strategy_results.keys())}")
                            
                            for strategy_name, strategy_result in strategy_results.items():
                                if isinstance(strategy_result, dict):
                                    exec_details_strategy = strategy_result.get('execution_details', [])
                                    print(f"    - {strategy_name}: {len(exec_details_strategy)} execution_details")
                except Exception as e:
                    print(f"❌ 正常ケース統合実行検証エラー: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            print(f"❌ _get_symbol_data()実行エラー: {str(e)}")
            import traceback
            traceback.print_exc()

    # Step 4: 根本原因の推論
    print(f"\n[STEP 4] 根本原因の推論")
    print(f"P2-1～P2-3調査結果の総合判断:")
    print(f"1. _get_optimal_symbol()個別実行：常に成功（'1662'選択）")
    print(f"2. _get_optimal_symbol()統合実行：失敗（None）")
    print(f"3. 推論：統合実行時に_get_optimal_symbol()の実行環境に問題")
    print(f"")
    print(f"possible原因候補:")
    print(f"A. 初期化タイミング問題（DSS Core V3未初期化）")
    print(f"B. 依存データ不整合（stock_dataまたはindex_data）")
    print(f"C. 内部状態競合（マルチスレッド・ロック問題）")
    print(f"D. メモリ・リソース不足（大量データ処理時）")
    print(f"E. エラーハンドリング不備（例外の隠蔽）")

    # 調査結果まとめ
    print(f"\n[SUMMARY] P2-4-A1調査結果")
    print(f"execution_details生成フロー:")
    print(f"1. daily_result['execution_details'] = [] で初期化")
    print(f"2. selected_symbol = _get_optimal_symbol()")
    print(f"3. selected_symbol=None → 処理終了、execution_details=0")
    print(f"4. selected_symbol有効 → _execute_multi_strategies()実行")
    print(f"5. MainSystemController → execution_details生成")
    print(f"6. _convert_main_new_result() → execution_details抽出")
    print(f"7. daily_result['execution_details'].extend()で追加")
    print(f"")
    print(f"根本原因: 統合実行時の_get_optimal_symbol()がNoneを返すため")
    print(f"  → execution_details生成処理に到達しない")
    print(f"  → 結果的にexecution_details=0件となる")

except Exception as e:
    print(f"❌ 調査実行エラー: {str(e)}")
    import traceback
    traceback.print_exc()

print(f"\n=== P2-4-A1調査完了 ===")