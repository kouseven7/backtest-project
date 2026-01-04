"""
DSSMS symbol=None問題の詳細調査

例外を隠すことなく実際のエラーを捕捉し原因を特定します。
"""

import sys
import os
import traceback
from datetime import datetime, timedelta
import logging

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath('.'))

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_symbol_selection():
    """DSSMSの銘柄選択プロセスを詳細デバッグ"""
    
    try:
        # DSSMSのバックテスターV3を直接インポート
        from src.dssms.dssms_backtester_v3 import DSSMSBacktesterV3
        
        # 日付設定
        target_date = datetime(2025, 1, 15)
        
        print(f"\n=== DSSMS銘柄選択デバッグ開始 ({target_date}) ===")
        
        # DSSMSバックテスター初期化
        backtester = DSSMSBacktesterV3()
        print("[DEBUG] DSSMSバックテスター初期化成功")
        
        # 日次銘柄選択を直接実行
        print("[DEBUG] run_daily_selection実行開始...")
        
        try:
            # run_daily_selectionメソッドを直接呼び出し
            result = backtester.run_daily_selection(target_date)
            
            print(f"[DEBUG] run_daily_selection結果: {result}")
            
            if result is None:
                print("[ERROR] run_daily_selectionがNoneを返しました")
            elif isinstance(result, dict):
                optimal_symbol = result.get('optimal_symbol', None)
                print(f"[DEBUG] optimal_symbol: {optimal_symbol}")
                
                if optimal_symbol is None:
                    print("[ERROR] optimal_symbolがNoneです")
                    print(f"[DEBUG] 完全な結果辞書: {result}")
            else:
                print(f"[ERROR] 予期しない結果タイプ: {type(result)}")
                
        except Exception as selection_error:
            print(f"[ERROR] run_daily_selection実行中にエラー:")
            print(f"Error: {selection_error}")
            print(f"Traceback:")
            traceback.print_exc()
            
    except ImportError as import_error:
        print(f"[ERROR] インポートエラー: {import_error}")
        traceback.print_exc()
        
    except Exception as e:
        print(f"[ERROR] 初期化エラー: {e}")
        traceback.print_exc()

def debug_dssms_integrated_main():
    """DSSMS統合メインの_get_optimal_symbol方法をデバッグ"""
    
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        target_date = datetime(2025, 1, 15)
        print(f"\n=== DSSMS統合メイン _get_optimal_symbol デバッグ ({target_date}) ===")
        
        # DSSMSIntegratedBacktester初期化
        integrated_backtester = DSSMSIntegratedBacktester()
        print("[DEBUG] DSSMS統合バックテスター初期化成功")
        
        # _get_optimal_symbolメソッドを直接実行
        print("[DEBUG] _get_optimal_symbol実行開始...")
        
        try:
            # ここで例外を隠さずに実行
            optimal_symbol = integrated_backtester._get_optimal_symbol(target_date)
            
            print(f"[DEBUG] _get_optimal_symbol結果: {optimal_symbol}")
            
            if optimal_symbol is None:
                print("[ERROR] _get_optimal_symbolがNoneを返しました")
            
        except Exception as symbol_error:
            print(f"[ERROR] _get_optimal_symbol実行中にエラー:")
            print(f"Error: {symbol_error}")
            print(f"Traceback:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"[ERROR] DSSMS統合メイン初期化エラー: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("DSSMS symbol=None問題の詳細調査開始")
    
    # Step 1: DSS Core V3の銘柄選択を直接テスト
    debug_symbol_selection()
    
    print("\n" + "="*60 + "\n")
    
    # Step 2: DSSMS統合メインの銘柄取得を直接テスト
    debug_dssms_integrated_main()
    
    print("\nデバッグ完了")