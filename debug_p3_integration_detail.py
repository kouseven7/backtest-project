"""
P3修正効果詳細検証スクリプト - 統合実行時の日次結果確認

統合実行時のdaily_resultの詳細な内容とP3修正の動作確認
"""
import logging
from datetime import datetime
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

# デバッグ用ログレベル設定
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

def main():
    print("=== P3修正効果詳細検証: 統合実行時の日次結果確認 ===")
    
    try:
        # DSSMSIntegratedBacktester初期化
        backtester = DSSMSIntegratedBacktester()
        print("✅ DSSMSIntegratedBacktester初期化成功")
        
        start_date = datetime(2025, 1, 15)
        end_date = datetime(2025, 1, 15)
        
        # バックテスト実行前の状態確認
        print(f"\n[BEFORE] バックテスト前 current_symbol: {repr(backtester.current_symbol)}")
        
        # 統合実行
        print(f"\n[EXECUTION] run_dynamic_backtest実行開始...")
        results = backtester.run_dynamic_backtest(start_date=start_date, end_date=end_date)
        
        # 結果詳細確認
        print(f"\n[RESULTS] 統合実行結果:")
        print(f"  結果数: {len(results) if hasattr(results, '__len__') and not isinstance(results, str) else 'N/A'}件")
        print(f"  backtester.daily_results数: {len(backtester.daily_results)}件")
        
        if backtester.daily_results:
            daily_result = backtester.daily_results[0]
            print(f"\n[DAILY_RESULT] 最初の日次結果詳細:")
            print(f"  date: {repr(daily_result.get('date'))}")
            print(f"  symbol: {repr(daily_result.get('symbol'))}")
            print(f"  success: {daily_result.get('success')}")
            print(f"  switch_executed: {daily_result.get('switch_executed')}")
            print(f"  current_symbol(実行後): {repr(backtester.current_symbol)}")
            
            # P3修正の効果確認
            print(f"\n[P3_VERIFICATION] P3修正効果確認:")
            print(f"  daily_result['symbol']の値: {repr(daily_result.get('symbol'))}")
            print(f"  backtester.current_symbolの値: {repr(backtester.current_symbol)}")
            print(f"  両者の一致: {daily_result.get('symbol') == backtester.current_symbol}")
            
            # switch履歴確認
            if hasattr(backtester, 'switch_history'):
                print(f"  switch_history数: {len(backtester.switch_history)}")
                if backtester.switch_history:
                    last_switch = backtester.switch_history[-1]
                    print(f"  最後のswitch: {last_switch}")
            else:
                print(f"  switch_history: 未初期化")
        else:
            print(f"  ❌ daily_results が空です")
            
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()