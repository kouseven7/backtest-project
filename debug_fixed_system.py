"""
修正後のDSSMS統合システムの日次処理を詳細デバッグ

DSS Core V3が利用可能になったが、まだsymbol=Noneが返される問題を調査
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

def debug_daily_processing():
    """修正後のDSSMS日次処理をデバッグ"""
    
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        target_date = datetime(2025, 1, 15)
        print(f"\n=== 修正後DSSMS日次処理デバッグ ({target_date}) ===")
        
        # DSSMSIntegratedBacktester初期化
        integrated_backtester = DSSMSIntegratedBacktester()
        print("[DEBUG] DSSMS統合バックテスター初期化成功")
        
        # _process_daily_tradingメソッドを直接実行してみる
        print("[DEBUG] _process_daily_trading実行開始...")
        
        try:
            # 日次処理を実際に実行
            result = integrated_backtester._process_daily_trading(target_date)
            
            print(f"[DEBUG] _process_daily_trading結果: {result}")
            
            if result:
                symbol = result.get('symbol', None)
                execution_details = result.get('execution_details', 0)
                success = result.get('success', False)
                
                print(f"[DEBUG] symbol: {symbol}")
                print(f"[DEBUG] execution_details: {execution_details}")
                print(f"[DEBUG] success: {success}")
                
                if symbol is None:
                    print("[ERROR] symbolがNoneです - 銘柄選択に失敗")
                else:
                    print(f"[SUCCESS] 銘柄選択成功: {symbol}")
            else:
                print("[ERROR] _process_daily_trading結果がNoneまたは空です")
                
        except Exception as daily_error:
            print(f"[ERROR] _process_daily_trading実行中にエラー:")
            print(f"Error: {daily_error}")
            print(f"Traceback:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"[ERROR] 初期化エラー: {e}")
        traceback.print_exc()

def debug_components_initialization():
    """コンポーネント初期化状態を確認"""
    
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        print(f"\n=== コンポーネント初期化状態確認 ===")
        
        integrated_backtester = DSSMSIntegratedBacktester()
        print("[DEBUG] インスタンス作成完了")
        
        # 重要コンポーネントの初期化状態を確認
        print(f"[DEBUG] dss_core: {integrated_backtester.dss_core}")
        print(f"[DEBUG] nikkei225_screener: {integrated_backtester.nikkei225_screener}")
        print(f"[DEBUG] advanced_ranking: {integrated_backtester.advanced_ranking}")
        
        # ensure_componentsを実行
        integrated_backtester.ensure_components()
        print("[DEBUG] ensure_components実行完了")
        
        print(f"[DEBUG] dss_core after ensure: {integrated_backtester.dss_core}")
        print(f"[DEBUG] nikkei225_screener after ensure: {integrated_backtester.nikkei225_screener}")
        
        # DSS Core V3の利用可能性を再確認
        try:
            from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
            print("[DEBUG] DSS Core V3 import成功")
            dss_available = True
        except ImportError as e:
            print(f"[ERROR] DSS Core V3 import失敗: {e}")
            dss_available = False
        
        print(f"[DEBUG] dss_available: {dss_available}")
        print(f"[DEBUG] dss_core is not None: {integrated_backtester.dss_core is not None}")
        
        if integrated_backtester.dss_core and dss_available:
            print("[SUCCESS] DSS Core V3が利用可能です")
        else:
            print("[WARNING] DSS Core V3が利用できません - フォールバックが動作するはずです")
            
    except Exception as e:
        print(f"[ERROR] コンポーネント確認エラー: {e}")
        traceback.print_exc()

def debug_get_optimal_symbol_detailed():
    """_get_optimal_symbolの詳細デバッグ"""
    
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        target_date = datetime(2025, 1, 15)
        print(f"\n=== _get_optimal_symbol詳細デバッグ ({target_date}) ===")
        
        integrated_backtester = DSSMSIntegratedBacktester()
        print("[DEBUG] インスタンス作成完了")
        
        # 例外隠しを除去したバージョンで実行
        print("[DEBUG] _get_optimal_symbol実行開始（例外隠しなし）...")
        
        try:
            # ensure_componentsを実行
            integrated_backtester.ensure_components()
            integrated_backtester.ensure_advanced_ranking()
            
            # DSS Core V3の利用可能性確認
            from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
            dss_available = True
            
            if integrated_backtester.dss_core and dss_available:
                print("[DEBUG] DSS Core V3を使用して銘柄選択開始")
                
                # DSS Core V3による動的選択（例外隠しなし）
                try:
                    dss_result = integrated_backtester.dss_core.run_daily_selection(target_date)
                    print(f"[DEBUG] DSS run_daily_selection結果: {dss_result}")
                    
                    selected_symbol = dss_result.get('selected_symbol') if dss_result else None
                    print(f"[DEBUG] selected_symbol from DSS: {selected_symbol}")
                    
                    if selected_symbol:
                        print(f"[SUCCESS] DSS Core V3から銘柄選択成功: {selected_symbol}")
                        return selected_symbol
                    else:
                        print("[WARNING] DSS Core V3からsymbol=None返却")
                        
                except Exception as dss_error:
                    print(f"[ERROR] DSS Core V3実行エラー: {dss_error}")
                    traceback.print_exc()
            
            print("[DEBUG] フォールバック処理に移行")
            # フォールバック処理をテスト
            
        except Exception as symbol_error:
            print(f"[ERROR] _get_optimal_symbol実行中にエラー:")
            print(f"Error: {symbol_error}")
            print(f"Traceback:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"[ERROR] 詳細デバッグエラー: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("修正後DSSMS統合システムの詳細調査開始")
    
    # Step 1: コンポーネント初期化確認
    debug_components_initialization()
    
    print("\n" + "="*60 + "\n")
    
    # Step 2: _get_optimal_symbol詳細デバッグ
    debug_get_optimal_symbol_detailed()
    
    print("\n" + "="*60 + "\n")
    
    # Step 3: 日次処理全体デバッグ
    debug_daily_processing()
    
    print("\nデバッグ完了")