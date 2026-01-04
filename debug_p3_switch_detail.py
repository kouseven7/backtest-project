"""
P3修正効果確認スクリプト - Switch実行詳細ログ

switch処理の実行状況とdaily_result['symbol']更新の確認
"""
import logging
from datetime import datetime
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

# デバッグ用ログレベル設定
logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(levelname)s:%(message)s')

def main():
    print("=== P3修正効果確認: Switch処理詳細ログ ===")
    
    try:
        # DSSMSIntegratedBacktester初期化
        backtester = DSSMSIntegratedBacktester()
        print("✅ DSSMSIntegratedBacktester初期化成功")
        
        target_date = datetime(2025, 1, 15)
        
        # DEBUG: switch処理前の状態確認
        print(f"\n[DEBUG] switch処理前 current_symbol: {repr(backtester.current_symbol)}")
        
        # _get_optimal_symbol実行
        selected_symbol = backtester._get_optimal_symbol(target_date, None)
        print(f"[DEBUG] _get_optimal_symbol結果: {repr(selected_symbol)}")
        
        if selected_symbol:
            # _evaluate_and_execute_switch実行
            switch_result = backtester._evaluate_and_execute_switch(selected_symbol, target_date)
            print(f"[DEBUG] switch_result: {switch_result}")
            print(f"[DEBUG] switch実行後 current_symbol: {repr(backtester.current_symbol)}")
            
            # daily_result初期化テスト
            daily_result = {
                'date': target_date.strftime('%Y-%m-%d'),
                'symbol': backtester.current_symbol,  # 初期化時点
                'switch_executed': False,
                'success': False
            }
            print(f"[DEBUG] daily_result初期化時 symbol: {repr(daily_result['symbol'])}")
            
            # P3修正箇所のテスト
            if switch_result.get('switch_executed', False):
                daily_result['switch_executed'] = True
                daily_result['symbol'] = backtester.current_symbol  # P3修正: switch後の銘柄を反映
                print(f"[DEBUG] P3修正適用後 daily_result['symbol']: {repr(daily_result['symbol'])}")
            else:
                print(f"[DEBUG] switch未実行: switch_executed={switch_result.get('switch_executed', False)}")
        else:
            print(f"[DEBUG] selected_symbol=Noneのため、switch処理はスキップ")
            
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()