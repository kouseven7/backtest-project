"""
P2-2緊急調査: _get_optimal_symbol()メソッド詳細デバッグ

本番統合実行時の_get_optimal_symbol()内部動作を詳細に分析し、
symbol=None問題の真因を特定する。

P2-1でrun_daily_selection()が正常動作することは確認済み。
P2-2では統合実行時の例外処理・状態管理を重点調査する。

Author: Backtest Project Team
Created: 2026-01-03
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def setup_detailed_logging():
    """詳細ログ設定"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler for debug output
    debug_log_path = Path(__file__).parent / "debug_get_optimal_symbol_log.txt"
    file_handler = logging.FileHandler(debug_log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add to root logger
    logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)

def investigate_get_optimal_symbol():
    """_get_optimal_symbol()メソッド詳細調査"""
    logger = setup_detailed_logging()
    
    print("=" * 80)
    print("🔍 P2-2緊急調査: _get_optimal_symbol()メソッド詳細分析")
    print("=" * 80)
    
    try:
        # Step 1: DSSMSIntegratedBacktester初期化
        print("\n[STEP 1] DSSMSIntegratedBacktester初期化...")
        
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        backtest_system = DSSMSIntegratedBacktester()
        
        print(f"✅ DSSMSIntegratedBacktester初期化成功")
        print(f"   - dss_core: {backtest_system.dss_core}")
        print(f"   - _dss_initialized: {backtest_system._dss_initialized}")
        
        # Step 2: DSS Core V3初期化状態確認
        print("\n[STEP 2] DSS Core V3初期化状態確認...")
        
        dss_core = backtest_system.ensure_dss_core()
        print(f"✅ DSS Core V3状態確認完了")
        print(f"   - dss_core is None: {dss_core is None}")
        print(f"   - dss_core type: {type(dss_core)}")
        
        # Step 3: target_dateを設定
        target_dates = [
            datetime(2025, 1, 15),
            datetime(2025, 1, 16)
        ]
        
        for target_date in target_dates:
            print(f"\n[STEP 3] target_date: {target_date.strftime('%Y-%m-%d')} での_get_optimal_symbol()実行...")
            
            try:
                # _get_optimal_symbol()を直接実行（例外キャッチ）
                print(f"   - _get_optimal_symbol()実行開始...")
                
                selected_symbol = backtest_system._get_optimal_symbol(target_date)
                
                print(f"✅ _get_optimal_symbol()実行結果:")
                print(f"   - selected_symbol: {selected_symbol}")
                print(f"   - 型: {type(selected_symbol)}")
                
                # 結果分析
                if selected_symbol is None:
                    print("❌ CRITICAL: selected_symbol is None!")
                    print("   - これがsymbol=None問題の発生箇所！")
                elif selected_symbol:
                    print(f"✅ SUCCESS: selected_symbol = '{selected_symbol}'")
                else:
                    print(f"⚠️ UNEXPECTED: selected_symbol = {repr(selected_symbol)}")
                
                # Step 4: DSS Core V3の内部状態を直接確認
                print(f"\n[STEP 4] DSS Core V3内部動作確認...")
                
                if dss_core:
                    try:
                        # run_daily_selection()を直接実行
                        dss_result = dss_core.run_daily_selection(target_date)
                        print(f"✅ DSS Core V3 run_daily_selection()結果:")
                        print(f"   - dss_result type: {type(dss_result)}")
                        print(f"   - selected_symbol: {dss_result.get('selected_symbol') if isinstance(dss_result, dict) else 'N/A'}")
                        
                        if isinstance(dss_result, dict):
                            for key, value in dss_result.items():
                                if key == 'ranking':
                                    print(f"   - {key}: [{len(value)} items] (省略)")
                                elif key == 'execution_time_ms':
                                    print(f"   - {key}: {value:.1f}ms")
                                else:
                                    print(f"   - {key}: {value}")
                        
                    except Exception as e:
                        print(f"❌ DSS Core V3 run_daily_selection()でエラー:")
                        print(f"   - エラー: {e}")
                        print(f"   - traceback: {traceback.format_exc()}")
                else:
                    print("❌ dss_core is None - DSS Core V3初期化失敗")
                
            except Exception as e:
                print(f"❌ _get_optimal_symbol()でエラー発生:")
                print(f"   - エラー: {e}")
                print(f"   - 型: {type(e)}")
                print(f"   - traceback:")
                traceback.print_exc()
                
                # 例外の詳細分析
                print(f"\n[例外詳細分析]")
                print(f"   - Exception message: {str(e)}")
                print(f"   - Exception args: {e.args}")
                print(f"   - Exception __class__: {e.__class__}")
        
        # Step 5: 複数日実行による状態変化確認
        print(f"\n[STEP 5] 複数日連続実行による状態変化確認...")
        
        consecutive_dates = [
            datetime(2025, 1, 13),  # 月曜日
            datetime(2025, 1, 14),  # 火曜日  
            datetime(2025, 1, 15),  # 水曜日
            datetime(2025, 1, 16),  # 木曜日
            datetime(2025, 1, 17),  # 金曜日
        ]
        
        results = []
        for i, date in enumerate(consecutive_dates):
            print(f"   - Day {i+1}: {date.strftime('%Y-%m-%d')} ...")
            
            try:
                selected = backtest_system._get_optimal_symbol(date)
                results.append((date, selected, None))
                print(f"     Result: {selected}")
            except Exception as e:
                results.append((date, None, e))
                print(f"     Error: {e}")
        
        print(f"\n[連続実行結果サマリー]")
        for date, selected, error in results:
            status = "✅ SUCCESS" if selected else ("❌ ERROR" if error else "⚠️ None")
            print(f"   - {date.strftime('%Y-%m-%d')}: {selected} ({status})")
        
        # Step 6: Components状態確認
        print(f"\n[STEP 6] Components状態確認...")
        
        components_status = {
            "dss_core": backtest_system.dss_core,
            "nikkei225_screener": backtest_system.nikkei225_screener,
            "advanced_ranking_engine": backtest_system.advanced_ranking_engine,
            "_dss_initialized": backtest_system._dss_initialized,
            "_ranking_initialized": backtest_system._ranking_initialized,
            "_components_initialized": backtest_system._components_initialized,
        }
        
        for component, status in components_status.items():
            print(f"   - {component}: {status} (type: {type(status)})")
        
        print(f"\n✅ P2-2調査完了")
        return True
        
    except Exception as e:
        print(f"❌ P2-2調査でエラー発生:")
        print(f"   - エラー: {e}")
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    print(f"🔍 P2-2緊急調査開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = investigate_get_optimal_symbol()
    
    if success:
        print(f"\n✅ P2-2調査完了: 結果は debug_get_optimal_symbol_log.txt を参照")
    else:
        print(f"\n❌ P2-2調査失敗")
    
    print(f"🔍 P2-2調査終了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()