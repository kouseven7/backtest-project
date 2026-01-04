"""
Priority B1詳細調査スクリプト - _process_daily_trading()内実行フロー追跡

統合実行時に_get_optimal_symbol()が呼び出される前の処理停止原因を特定するため、
_process_daily_trading()メソッド内の実行フローを詳細に追跡する。

主な機能:
- _process_daily_trading()内各ステップの実行状況ログ出力
- 単独実行と統合実行の詳細比較
- 例外・エラーの詳細キャッチ・報告
- 引数・状態変数の各時点での値確認
- Line-by-line実行追跡によるサイレント失敗検出

統合コンポーネント:
- src.dssms.dssms_integrated_main: DSSMSIntegratedBacktester継承
- Enhanced Logger Manager: 詳細ログ出力
- Exception Handling: 全例外キャッチ・詳細報告

セーフティ機能/注意事項:
- 修正は一切行わず、調査のみ実施
- 既存のバックテスト実行を妨げない設計
- 実データのみ使用、フォールバック機能は報告対象

Author: Backtest Project Team
Created: 2026-01-04
Last Modified: 2026-01-04
"""

import sys
import os
import traceback
from datetime import datetime, timedelta
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# DSSMS統合モジュールのインポート
try:
    from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
    print("[IMPORT_SUCCESS] DSSMSIntegratedBacktester import successful")
except ImportError as e:
    print(f"[IMPORT_ERROR] DSSMSIntegratedBacktester import failed: {e}")
    sys.exit(1)


class DetailedInvestigationBacktester(DSSMSIntegratedBacktester):
    """
    Priority B1調査用: _process_daily_trading()内実行フロー詳細追跡
    
    統合実行時の_get_optimal_symbol()未実行問題の原因特定のため、
    _process_daily_trading()内の各ステップを詳細にログ出力する。
    """
    
    def __init__(self):
        print("[B1_INIT] DetailedInvestigationBacktester初期化開始")
        try:
            super().__init__()
            print("[B1_INIT] 親クラス初期化完了")
        except Exception as e:
            print(f"[B1_INIT_ERROR] 親クラス初期化失敗: {e}")
            print(f"[B1_INIT_ERROR] Traceback: {traceback.format_exc()}")
            raise
    
    def _process_daily_trading(self, target_date, target_symbols=None):
        """
        詳細調査版: _process_daily_trading()内実行フロー追跡
        
        各ステップでの状態確認・ログ出力により、
        統合実行時の処理停止ポイントを特定する。
        """
        print(f"\n[B1_FLOW_START] _process_daily_trading()実行開始")
        print(f"[B1_FLOW_ARGS] 引数確認:")
        print(f"  target_date: {target_date} (type: {type(target_date)})")
        print(f"  target_symbols: {target_symbols} (type: {type(target_symbols)})")
        
        # ========== Step 1: 初期化状況確認 ==========
        print(f"[B1_STEP_1] 初期化状況確認開始")
        try:
            print(f"[B1_STEP_1] self.dssms_v3 存在確認: {hasattr(self, 'dssms_v3')}")
            if hasattr(self, 'dssms_v3'):
                print(f"[B1_STEP_1] self.dssms_v3 値: {self.dssms_v3}")
                print(f"[B1_STEP_1] self.dssms_v3 型: {type(self.dssms_v3)}")
            
            print(f"[B1_STEP_1] self.current_symbol 存在確認: {hasattr(self, 'current_symbol')}")
            if hasattr(self, 'current_symbol'):
                print(f"[B1_STEP_1] self.current_symbol 値: {self.current_symbol}")
            
            print(f"[B1_STEP_1] self.switch_history 存在確認: {hasattr(self, 'switch_history')}")
            if hasattr(self, 'switch_history'):
                print(f"[B1_STEP_1] self.switch_history 長さ: {len(self.switch_history)}")
        except Exception as e:
            print(f"[B1_STEP_1_ERROR] 初期化状況確認でエラー: {e}")
            print(f"[B1_STEP_1_ERROR] Traceback: {traceback.format_exc()}")
        
        # ========== Step 2: daily_result初期化確認 ==========
        print(f"[B1_STEP_2] daily_result初期化開始")
        try:
            # 親クラスのdaily_result初期化部分を追跡
            daily_result = {
                'date': target_date,
                'symbol': self.current_symbol,
                'switch_executed': False,
                'execution_details': 0,
                'total_return': 0.0,
                'errors': []
            }
            print(f"[B1_STEP_2] daily_result初期化完了:")
            for key, value in daily_result.items():
                print(f"  {key}: {value} (type: {type(value)})")
        except Exception as e:
            print(f"[B1_STEP_2_ERROR] daily_result初期化でエラー: {e}")
            print(f"[B1_STEP_2_ERROR] Traceback: {traceback.format_exc()}")
            return None
        
        # ========== Step 3: _get_optimal_symbol()呼び出し直前確認 ==========
        print(f"[B1_STEP_3] _get_optimal_symbol()呼び出し直前状況確認")
        try:
            print(f"[B1_STEP_3] target_date値確認: {target_date}")
            print(f"[B1_STEP_3] target_symbols値確認: {target_symbols}")
            print(f"[B1_STEP_3] _get_optimal_symbol()呼び出し準備完了")
            
            # ========== CRITICAL POINT: _get_optimal_symbol()呼び出し ==========
            print(f"[B1_CRITICAL] _get_optimal_symbol()実行開始 - これが統合実行時に未実行")
            selected_symbol = self._get_optimal_symbol(target_date, target_symbols)
            print(f"[B1_CRITICAL] _get_optimal_symbol()実行完了 - 結果: {selected_symbol}")
            
        except Exception as e:
            print(f"[B1_STEP_3_ERROR] _get_optimal_symbol()呼び出しでエラー: {e}")
            print(f"[B1_STEP_3_ERROR] Traceback: {traceback.format_exc()}")
            selected_symbol = None
        
        # ========== Step 4: 結果確認・条件分岐追跡 ==========
        print(f"[B1_STEP_4] 結果確認・条件分岐追跡")
        print(f"[B1_STEP_4] selected_symbol値: {selected_symbol}")
        print(f"[B1_STEP_4] selected_symbol boolean評価: {bool(selected_symbol)}")
        
        if not selected_symbol:
            print(f"[B1_STEP_4] 早期リターン条件成立 - これが統合実行時の問題")
            daily_result['errors'].append('銘柄選択失敗')
            print(f"[B1_STEP_4] エラー追加後 daily_result: {daily_result}")
            print(f"[B1_STEP_4] 早期リターン実行")
            return daily_result
        
        # ========== Step 5: switch処理（到達確認） ==========
        print(f"[B1_STEP_5] switch処理到達 - 統合実行時は到達しないはず")
        print(f"[B1_STEP_5] selected_symbol: {selected_symbol}")
        print(f"[B1_STEP_5] self.current_symbol: {self.current_symbol}")
        
        # この後の処理は元の_process_daily_trading()と同等
        # （調査目的のため、switch処理以降は簡略化）
        daily_result['symbol'] = selected_symbol
        print(f"[B1_STEP_5] daily_result['symbol']更新: {daily_result['symbol']}")
        
        return daily_result


def run_detailed_investigation():
    """
    Priority B1詳細調査実行:
    単独実行と統合実行の_process_daily_trading()内実行フロー比較
    """
    print("\n" + "="*80)
    print("Priority B1詳細調査: _process_daily_trading()内実行フロー追跡")
    print("調査目的: 統合実行時の_get_optimal_symbol()未実行原因特定")
    print("="*80)
    
    # 調査対象日付
    target_date = datetime(2025, 1, 15)
    
    # ========== 詳細調査実行 ==========
    try:
        print(f"\n[B1_MAIN] DetailedInvestigationBacktester初期化開始")
        backtester = DetailedInvestigationBacktester()
        print(f"[B1_MAIN] DetailedInvestigationBacktester初期化完了")
        
        print(f"\n[B1_MAIN] _process_daily_trading()詳細調査実行")
        result = backtester._process_daily_trading(target_date)
        
        print(f"\n[B1_RESULT] 詳細調査結果:")
        print(f"  返り値: {result}")
        if result:
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print("  結果: None")
            
    except Exception as e:
        print(f"\n[B1_MAIN_ERROR] 詳細調査実行中にエラー: {e}")
        print(f"[B1_MAIN_ERROR] Traceback: {traceback.format_exc()}")


def run_comparison_investigation():
    """
    比較調査: 通常の統合実行との実行フロー比較
    （参考情報として簡易実行）
    """
    print(f"\n" + "="*80)
    print("参考: 通常の統合実行実行フロー確認")
    print("="*80)
    
    try:
        # 通常のDSSMSIntegratedBacktester
        print(f"[B1_COMP] 通常のDSSMSIntegratedBacktester初期化")
        normal_backtester = DSSMSIntegratedBacktester()
        
        target_date = datetime(2025, 1, 15)
        print(f"[B1_COMP] 通常実行: _process_daily_trading()実行")
        normal_result = normal_backtester._process_daily_trading(target_date)
        
        print(f"[B1_COMP] 通常実行結果:")
        print(f"  返り値: {normal_result}")
        if normal_result:
            for key, value in normal_result.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"[B1_COMP_ERROR] 通常実行比較中にエラー: {e}")
        print(f"[B1_COMP_ERROR] Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    print("Priority B1詳細調査スクリプト実行開始")
    print(f"実行日時: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"作業ディレクトリ: {os.getcwd()}")
    
    # 詳細調査実行
    run_detailed_investigation()
    
    # 比較調査実行  
    run_comparison_investigation()
    
    print(f"\n" + "="*80)
    print("Priority B1詳細調査スクリプト実行完了")
    print("次段階: 調査結果分析・Priority_B1_investigation_results.md作成")
    print("="*80)