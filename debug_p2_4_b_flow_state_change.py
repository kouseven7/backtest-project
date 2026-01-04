"""
P2-4-B: 統合実行フロー内状態変化調査
run_dynamic_backtest() → _process_daily_trading()フロー内で
_get_optimal_symbol()の戻り値がいつ・なぜNoneに変化するか特定
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

print("=== P2-4-B: 統合実行フロー内状態変化調査 ===")

try:
    # DSSMSIntegratedBacktesterインスタンス作成
    backtest_instance = DSSMSIntegratedBacktester()
    print(f"✅ DSSMSIntegratedBacktester初期化成功")
    
    # 対象日設定
    target_date = datetime(2025, 1, 15)
    print(f"\n[INVESTIGATION] 統合実行フロー内状態変化調査: {target_date}")
    
    # Step 1: 独立した_get_optimal_symbol()実行
    print(f"\n[STEP 1] 独立_get_optimal_symbol()実行（ベースライン確認）")
    
    independent_symbol = backtest_instance._get_optimal_symbol(target_date, None)
    print(f"独立実行結果: {independent_symbol}")
    
    if independent_symbol is None:
        print(f"❌ 独立実行でもNoneが返される → 別の問題が存在")
        print(f"  → この場合は統合実行フロー以外の原因")
    else:
        print(f"✅ 独立実行成功: {independent_symbol} → 統合フロー内に問題")

    # Step 2: _process_daily_trading()の部分的シミュレーション
    print(f"\n[STEP 2] _process_daily_trading()の部分的シミュレーション")
    
    # daily_result初期化（実際のコードと同じ）
    portfolio_value = getattr(backtest_instance, 'portfolio_value', 1000000)
    peak_value = getattr(backtest_instance, 'peak_value', portfolio_value)
    cumulative_pnl = getattr(backtest_instance, 'cumulative_pnl', 0.0)
    total_trades_count = getattr(backtest_instance, 'total_trades_count', 0)
    current_symbol = getattr(backtest_instance, 'current_symbol', None)
    
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
        'execution_details': []
    }
    
    print(f"daily_result初期化完了:")
    print(f"  - initial symbol: {daily_result['symbol']}")
    print(f"  - execution_details: {len(daily_result['execution_details'])}")
    
    # Step 3: _get_optimal_symbol()実行（_process_daily_trading内と同じ環境）
    print(f"\n[STEP 3] daily_result初期化後の_get_optimal_symbol()実行")
    
    # 実際の_process_daily_trading内と同じ順序で実行
    print(f"[DEBUG] _get_optimal_symbol(target_date={target_date}, existing_position=None)実行...")
    selected_symbol = backtest_instance._get_optimal_symbol(target_date, None)
    print(f"selected_symbol結果: {selected_symbol}")
    
    if selected_symbol != independent_symbol:
        print(f"❌ 結果不一致検出！")
        print(f"  - 独立実行: {independent_symbol}")
        print(f"  - daily_result初期化後: {selected_symbol}")
        print(f"  → daily_result初期化が_get_optimal_symbol()に影響を与える")
    else:
        print(f"✅ 結果一致: {selected_symbol}")
        print(f"  → daily_result初期化は影響なし")
    
    # Step 4: 実際の統合実行との対比
    print(f"\n[STEP 4] 実際の統合実行との詳細対比")
    
    # 統合実行の実行（詳細ログ監視）
    print(f"run_dynamic_backtest(single_date={target_date})実行中...")
    
    # ログレベルをDEBUGに一時変更して詳細を確認
    original_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = backtest_instance.run_dynamic_backtest(target_date, target_date)
        
        if isinstance(results, dict) and 'daily_results' in results:
            daily_results = results['daily_results']
            if daily_results:
                actual_result = daily_results[-1]
                
                print(f"\n統合実行結果:")
                print(f"  - 実際のsymbol: {actual_result.get('symbol')}")
                print(f"  - 実際のsuccess: {actual_result.get('success')}")
                print(f"  - 実際のexecution_details: {len(actual_result.get('execution_details', []))}")
                print(f"  - 実際のerrors: {actual_result.get('errors', [])}")
                
                # 詳細比較
                if actual_result.get('symbol') != selected_symbol:
                    print(f"\n🚨 決定的相違確認:")
                    print(f"  - シミュレーション: {selected_symbol}")
                    print(f"  - 統合実行: {actual_result.get('symbol')}")
                    print(f"  → 統合実行フロー内で戻り値が改変される")
                    
                    # エラーメッセージの詳細確認
                    errors = actual_result.get('errors', [])
                    if errors:
                        print(f"\n[ERROR_ANALYSIS] エラー詳細:")
                        for i, error in enumerate(errors):
                            print(f"  {i+1}. {error}")
                else:
                    print(f"✅ 一致: {selected_symbol}")
                    print(f"  → 問題は別の箇所にある")
        
    finally:
        # ログレベルを元に戻す
        logging.getLogger().setLevel(original_level)

    # Step 5: _process_daily_trading()メソッドの直接呼び出し
    print(f"\n[STEP 5] _process_daily_trading()メソッドの直接呼び出し")
    
    try:
        # _process_daily_trading()を直接呼び出し
        print(f"_process_daily_trading({target_date})直接呼び出し中...")
        direct_result = backtest_instance._process_daily_trading(target_date)
        
        print(f"直接呼び出し結果:")
        print(f"  - symbol: {direct_result.get('symbol')}")
        print(f"  - success: {direct_result.get('success')}")
        print(f"  - execution_details count: {len(direct_result.get('execution_details', []))}")
        print(f"  - errors: {direct_result.get('errors', [])}")
        
        # 比較分析
        if direct_result.get('symbol') != selected_symbol:
            print(f"\n🎯 根本原因特定:")
            print(f"  - 個別_get_optimal_symbol(): {selected_symbol}")
            print(f"  - _process_daily_trading()内: {direct_result.get('symbol')}")
            print(f"  → _process_daily_trading()内部で戻り値が変化")
            
            # エラーの詳細分析
            errors = direct_result.get('errors', [])
            if '銘柄選択失敗' in errors:
                print(f"  → Line 715-716で'銘柄選択失敗'が追加される")
                print(f"  → 実際の_get_optimal_symbol()は成功していたが、何らかの理由で無効化")
        else:
            print(f"✅ 直接呼び出しでも一致: {selected_symbol}")
            
    except Exception as e:
        print(f"❌ _process_daily_trading()直接呼び出しエラー: {str(e)}")
        import traceback
        traceback.print_exc()

    # Step 6: ソースコード上のLine 712付近の動作確認
    print(f"\n[STEP 6] Line 712付近の動作パターン確認")
    
    print(f"想定されるコードパターン:")
    print(f"  Line 712: selected_symbol = self._get_optimal_symbol(current_date, existing_position)")
    print(f"  Line 713: if not selected_symbol:")
    print(f"  Line 714:     self.logger.error(f'銘柄選択失敗: {{current_date}}')")
    print(f"  Line 715:     daily_result['errors'].append('銘柄選択失敗')")
    print(f"  Line 716:     return daily_result")
    print(f"")
    print(f"疑問点:")
    print(f"1. selected_symbolに実際に何が代入されているか？")
    print(f"2. if not selected_symbol: の判定で何がFalsyと判定されているか？")
    print(f"3. _get_optimal_symbol()の戻り値が途中で変更されていないか？")

    # 調査結果まとめ
    print(f"\n[SUMMARY] P2-4-B調査結果")
    print(f"調査目的: 統合実行フロー内での_get_optimal_symbol()戻り値変化原因特定")
    print(f"")
    print(f"主要発見:")
    print(f"1. 独立実行: _get_optimal_symbol() → {independent_symbol}")
    print(f"2. daily_result初期化後: _get_optimal_symbol() → {selected_symbol}")
    print(f"3. 統合実行結果: symbol → 確認要（上記ログ参照）")
    print(f"")
    print(f"次段階調査の必要性:")
    print(f"- Line 712での実際の代入値確認")
    print(f"- if not selected_symbol判定での詳細状態確認") 
    print(f"- 例外処理・エラーハンドリングの詳細追跡")

except Exception as e:
    print(f"❌ 調査実行エラー: {str(e)}")
    import traceback
    traceback.print_exc()

print(f"\n=== P2-4-B調査完了 ===")