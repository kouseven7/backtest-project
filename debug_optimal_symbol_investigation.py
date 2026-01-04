"""
Priority A調査スクリプト - _get_optimal_symbol()内部実行状況詳細調査

統合実行時の_get_optimal_symbol()がNoneを返す原因を特定するための
詳細ログ取得スクリプト

Author: Backtest Project Team
Created: 2026-01-04
"""
import logging
from datetime import datetime
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

# デバッグ用ログレベル設定
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

class DetailedInvestigationBacktester(DSSMSIntegratedBacktester):
    """
    詳細調査用のDSSMSIntegratedBacktester拡張クラス
    _get_optimal_symbol()の内部実行状況を詳細ログ出力
    """
    
    def _get_optimal_symbol(self, target_date: datetime, 
                          target_symbols = None):
        """
        詳細調査版_get_optimal_symbol()
        オリジナルメソッドの実行状況を詳細ログ出力
        """
        print(f"\n=== [DETAILED] _get_optimal_symbol() 実行開始 ===")
        print(f"[ARGS] target_date: {repr(target_date)}")
        print(f"[ARGS] target_symbols: {repr(target_symbols)}")
        
        # 初期化状態確認
        print(f"\n[INIT_STATE] 初期化状態確認:")
        print(f"  dss_available: {globals().get('dss_available', 'UNKNOWN')}")
        print(f"  self._dss_initialized: {getattr(self, '_dss_initialized', 'UNKNOWN')}")
        print(f"  hasattr(self, 'dss_core'): {hasattr(self, 'dss_core')}")
        
        if hasattr(self, 'dss_core'):
            print(f"  self.dss_core: {repr(self.dss_core)}")
        else:
            print(f"  self.dss_core: 属性なし")
        
        # self.dssms_v3の確認
        if hasattr(self, 'dssms_v3'):
            print(f"  self.dssms_v3: {repr(self.dssms_v3)}")
        else:
            print(f"  self.dssms_v3: 属性なし")
        
        try:
            # オリジナルメソッドの実行
            print(f"\n[EXECUTION] オリジナル_get_optimal_symbol()実行中...")
            result = super()._get_optimal_symbol(target_date, target_symbols)
            
            print(f"[RESULT] 戻り値: {repr(result)}")
            print(f"[RESULT] 戻り値の型: {type(result)}")
            print(f"[RESULT] bool(result): {bool(result)}")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 例外発生: {type(e).__name__}: {e}")
            import traceback
            print(f"[TRACEBACK] スタックトレース:")
            traceback.print_exc()
            return None
        finally:
            print(f"=== [DETAILED] _get_optimal_symbol() 実行終了 ===\n")

def main():
    print("=== Priority A調査：_get_optimal_symbol()詳細調査 ===")
    
    try:
        # 詳細調査用バックテスター初期化
        print(f"\n[STEP 1] DetailedInvestigationBacktester初期化中...")
        backtester = DetailedInvestigationBacktester()
        print("✅ DetailedInvestigationBacktester初期化成功")
        
        target_date = datetime(2025, 1, 15)
        
        # 初期化後の状態確認
        print(f"\n[STEP 2] 初期化後状態詳細確認:")
        print(f"  current_symbol: {repr(backtester.current_symbol)}")
        print(f"  _dss_initialized: {getattr(backtester, '_dss_initialized', 'UNKNOWN')}")
        print(f"  _components_initialized: {getattr(backtester, '_components_initialized', 'UNKNOWN')}")
        
        # 単独実行: _get_optimal_symbol()直接呼び出し
        print(f"\n[STEP 3] 単独実行: _get_optimal_symbol()直接呼び出し")
        selected_symbol_single = backtester._get_optimal_symbol(target_date, None)
        print(f"単独実行結果: {repr(selected_symbol_single)}")
        
        # 統合実行: run_dynamic_backtest()経由
        print(f"\n[STEP 4] 統合実行: run_dynamic_backtest()経由")
        start_date = datetime(2025, 1, 15)
        end_date = datetime(2025, 1, 15)
        
        # 統合実行前の状態確認
        print(f"統合実行前 current_symbol: {repr(backtester.current_symbol)}")
        
        # run_dynamic_backtest実行
        results = backtester.run_dynamic_backtest(start_date=start_date, end_date=end_date)
        
        # 統合実行後の状態確認
        print(f"統合実行後 current_symbol: {repr(backtester.current_symbol)}")
        print(f"daily_results数: {len(backtester.daily_results)}")
        
        if backtester.daily_results:
            daily_result = backtester.daily_results[0]
            print(f"daily_result['symbol']: {repr(daily_result.get('symbol'))}")
            print(f"daily_result['success']: {daily_result.get('success')}")
            
        # 結果比較
        print(f"\n[STEP 5] 結果比較:")
        print(f"  単独実行結果: {repr(selected_symbol_single)}")
        print(f"  統合実行結果: {repr(backtester.daily_results[0].get('symbol') if backtester.daily_results else None)}")
        print(f"  結果一致: {selected_symbol_single == (backtester.daily_results[0].get('symbol') if backtester.daily_results else None)}")
        
    except Exception as e:
        print(f"❌ メイン処理エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()