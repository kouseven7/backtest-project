"""
Priority C2-2調査スクリプト - _get_optimal_symbol()依存コンポーネント状態詳細調査

統合実行時に_get_optimal_symbol()内部でどのコンポーネント初期化が失敗し、
どの例外が隠蔽されているかを特定するための詳細調査スクリプト

主な調査対象:
- self.dss_core初期化状態
- self.nikkei225_screener初期化状態  
- dss_available フラグ状態
- 実際に発生している例外内容

統合コンポーネント:
- src.dssms.dssms_integrated_main: DSSMSIntegratedBacktester
- logging: 調査ログ出力

セーフティ機能/注意事項:
- 例外隠蔽を回避するため独自の詳細ログ出力を追加
- 統合実行環境と同一条件での実行を保証
- 実際の初期化処理とエラー内容の完全追跡

Author: Backtest Project Team
Created: 2026-01-04
Last Modified: 2026-01-04
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
from datetime import datetime, timedelta
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

# ログ設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetailedInvestigationBacktester(DSSMSIntegratedBacktester):
    """
    Priority C2-2調査用：依存コンポーネント状態詳細確認版
    """
    
    def _get_optimal_symbol_detailed_investigation(self, target_date: datetime, 
                                                 target_symbols=None):
        """
        _get_optimal_symbol()の詳細調査版 - 例外隠蔽を回避し全詳細ログ出力
        """
        print(f"\n=== [C2-2] _get_optimal_symbol()依存コンポーネント詳細調査 ===")
        print(f"[C2-2] 対象日付: {target_date}")
        print(f"[C2-2] 対象銘柄: {target_symbols}")
        
        # ステップ1: 基本状態確認
        print(f"\n[C2-2-STEP1] 基本オブジェクト状態確認")
        print(f"  - self: {type(self).__name__}")
        print(f"  - self.portfolio_value: {getattr(self, 'portfolio_value', 'UNDEFINED')}")
        print(f"  - hasattr(self, 'dss_core'): {hasattr(self, 'dss_core')}")
        print(f"  - hasattr(self, 'nikkei225_screener'): {hasattr(self, 'nikkei225_screener')}")
        
        try:
            # ステップ2: ensure_components()詳細実行
            print(f"\n[C2-2-STEP2] self.ensure_components()実行")
            try:
                self.ensure_components()
                print(f"  ✅ self.ensure_components(): 成功")
            except Exception as e:
                print(f"  ❌ self.ensure_components(): エラー - {e}")
                print(f"  ❌ エラー詳細: {type(e).__name__}: {str(e)}")
                return None
            
            # ステップ3: ensure_advanced_ranking()詳細実行
            print(f"\n[C2-2-STEP3] self.ensure_advanced_ranking()実行")
            try:
                self.ensure_advanced_ranking()
                print(f"  ✅ self.ensure_advanced_ranking(): 成功")
            except Exception as e:
                print(f"  ❌ self.ensure_advanced_ranking(): エラー - {e}")
                print(f"  ❌ エラー詳細: {type(e).__name__}: {str(e)}")
                # 続行可能な場合は続行
            
            # ステップ4: ensure_dss_core()詳細実行
            print(f"\n[C2-2-STEP4] self.ensure_dss_core()実行")
            try:
                self.ensure_dss_core()
                print(f"  ✅ self.ensure_dss_core(): 成功")
                print(f"  - self.dss_core: {type(getattr(self, 'dss_core', None))}")
                print(f"  - self.dss_core値: {getattr(self, 'dss_core', 'UNDEFINED')}")
            except Exception as e:
                print(f"  ❌ self.ensure_dss_core(): エラー - {e}")
                print(f"  ❌ エラー詳細: {type(e).__name__}: {str(e)}")
                # 続行してフォールバック確認
            
            # ステップ5: dss_available確認
            print(f"\n[C2-2-STEP5] dss_available状態確認")
            try:
                # グローバル変数の確認（推測）
                import importlib
                dssms_module = importlib.import_module('src.dssms.dssms_integrated_main')
                dss_available = getattr(dssms_module, 'dss_available', 'UNDEFINED')
                print(f"  - dss_available: {dss_available}")
            except Exception as e:
                print(f"  ❌ dss_available確認エラー: {e}")
            
            # ステップ6: DSS Core V3処理実行確認
            print(f"\n[C2-2-STEP6] DSS Core V3処理実行確認")
            dss_core = getattr(self, 'dss_core', None)
            if dss_core:
                try:
                    print(f"  - self.dss_core利用可能、run_daily_selection()実行...")
                    dss_result = dss_core.run_daily_selection(target_date)
                    selected_symbol = dss_result.get('selected_symbol')
                    print(f"  ✅ DSS Core V3処理成功")
                    print(f"  - dss_result: {dss_result}")
                    print(f"  - selected_symbol: {selected_symbol}")
                    if selected_symbol:
                        print(f"  🎯 DSS選択完了: {selected_symbol}")
                        return selected_symbol
                except Exception as e:
                    print(f"  ❌ DSS Core V3処理エラー: {e}")
                    print(f"  ❌ エラー詳細: {type(e).__name__}: {str(e)}")
            else:
                print(f"  ⚠️ self.dss_core利用不可（None or 未定義）")
            
            # ステップ7: nikkei225_screener確認・フォールバック処理
            print(f"\n[C2-2-STEP7] nikkei225_screenerフォールバック処理")
            nikkei225_screener = getattr(self, 'nikkei225_screener', None)
            if nikkei225_screener:
                try:
                    print(f"  - self.nikkei225_screener利用可能")
                    print(f"  - 銘柄選定処理実行...")
                    
                    available_funds = self.portfolio_value * 0.8
                    print(f"  - available_funds: {available_funds}")
                    
                    filtered_symbols = nikkei225_screener.get_filtered_symbols(available_funds)
                    print(f"  - filtered_symbols数: {len(filtered_symbols) if filtered_symbols else 0}")
                    print(f"  - filtered_symbols[:5]: {filtered_symbols[:5] if filtered_symbols else []}")
                    
                    if filtered_symbols:
                        try:
                            selected = self._advanced_ranking_selection(filtered_symbols, target_date)
                            print(f"  ✅ フォールバック選択成功: {selected}")
                            return selected
                        except Exception as e:
                            print(f"  ❌ _advanced_ranking_selection()エラー: {e}")
                            print(f"  ❌ エラー詳細: {type(e).__name__}: {str(e)}")
                    else:
                        print(f"  ⚠️ filtered_symbolsが空または取得失敗")
                        
                except Exception as e:
                    print(f"  ❌ nikkei225_screener処理エラー: {e}")
                    print(f"  ❌ エラー詳細: {type(e).__name__}: {str(e)}")
            else:
                print(f"  ❌ self.nikkei225_screener利用不可（None or 未定義）")
            
            # ステップ8: 最終状態確認
            print(f"\n[C2-2-STEP8] 最終状態確認")
            print(f"  - 全処理パス実行完了")
            print(f"  - 有効な銘柄選択なし")
            print(f"  🎯 戻り値: None（全パス失敗）")
            return None
            
        except Exception as e:
            print(f"\n[C2-2-ERROR] 最上位例外発生: {e}")
            print(f"[C2-2-ERROR] エラー詳細: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[C2-2-ERROR] トレースバック:")
            traceback.print_exc()
            return None

def run_c2_2_investigation():
    """Priority C2-2調査実行"""
    print("=== Priority C2-2調査：_get_optimal_symbol()依存コンポーネント状態詳細確認 ===")
    
    target_date = datetime(2025, 1, 15)
    
    try:
        # 詳細調査版バックテスターインスタンス作成
        print(f"\n[C2-2] DetailedInvestigationBacktester初期化...")
        backtester = DetailedInvestigationBacktester()
        print(f"✅ 初期化成功")
        
        # 依存コンポーネント詳細調査実行
        print(f"\n[C2-2] 依存コンポーネント詳細調査実行...")
        result = backtester._get_optimal_symbol_detailed_investigation(target_date, None)
        
        print(f"\n=== [C2-2] 調査結果サマリー ===")
        print(f"対象日付: {target_date}")
        print(f"調査結果: {result}")
        print(f"結論: {'成功' if result else '失敗（None返却）'}")
        
        return result
        
    except Exception as e:
        print(f"❌ Priority C2-2調査実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Priority C2-2調査スクリプト実行開始")
    result = run_c2_2_investigation()
    print(f"\n✅ Priority C2-2調査完了: 結果={result}")