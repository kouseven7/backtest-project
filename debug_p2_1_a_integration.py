"""
P3調査: P2-1-A _get_optimal_symbol()統合実行時の詳細デバッグ

統合実行時の_get_optimal_symbol()の内部動作を詳細に追跡し、
P2-2個別実行時との差異を特定する
"""

import sys
import os
from datetime import datetime
sys.path.append('.')

def investigate_p2_1_a_get_optimal_symbol_integration():
    """P2-1-A: _get_optimal_symbol()統合実行時の詳細調査"""
    
    print("=== P2-1-A: _get_optimal_symbol()統合実行時詳細調査 ===")
    
    try:
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        backtester = DSSMSIntegratedBacktester()
        print("✅ DSSMSIntegratedBacktester初期化成功")
        
        # 統合実行時の環境確認
        print(f"\n[ENV_CHECK] 統合実行時の環境状態:")
        print(f"  - dss_core: {backtester.dss_core}")
        print(f"  - advanced_ranking_engine: {getattr(backtester, 'advanced_ranking_engine', 'なし')}")
        print(f"  - _dss_initialized: {backtester._dss_initialized}")
        print(f"  - _ranking_initialized: {backtester._ranking_initialized}")
        
        # テスト対象のパラメータ（P3統合実行と同じ）
        target_date = datetime(2025, 1, 15)
        target_symbols = None  # 統合実行時の実際の値
        
        print(f"\n[PARAMS] テストパラメータ:")
        print(f"  - target_date: {target_date}")
        print(f"  - target_symbols: {target_symbols}")
        
        # _get_optimal_symbol()の詳細実行（ステップバイステップ）
        print(f"\n[STEP 1] _get_optimal_symbol()実行開始...")
        
        # メソッドを直接呼び出してデバッグ
        try:
            selected_symbol = backtester._get_optimal_symbol(target_date, target_symbols)
            
            print(f"✅ _get_optimal_symbol()実行完了")
            print(f"   戻り値: {selected_symbol}")
            print(f"   戻り値の型: {type(selected_symbol)}")
            
            if selected_symbol is None:
                print(f"❌ 戻り値がNone - これが問題の核心")
            else:
                print(f"✅ 戻り値は正常 - 銘柄選択成功")
                
        except Exception as e:
            print(f"❌ _get_optimal_symbol()でエラー発生:")
            print(f"   エラー: {e}")
            import traceback
            print(f"   Full traceback: {traceback.format_exc()}")
            return False, None
        
        # 内部状態の詳細確認
        print(f"\n[STEP 2] 実行後の内部状態確認:")
        print(f"  - dss_core: {backtester.dss_core}")
        print(f"  - _dss_initialized: {backtester._dss_initialized}")
        print(f"  - _ranking_initialized: {backtester._ranking_initialized}")
        
        return True, selected_symbol
        
    except Exception as e:
        print(f"❌ 調査実行エラー: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False, None

def compare_with_p2_2_success():
    """P2-2成功例との比較"""
    
    print(f"\n=== P2-2成功例との比較 ===")
    
    # P2-2の成功例の情報
    p2_2_success_data = {
        'target_dates': [
            datetime(2025, 1, 15),
            datetime(2025, 1, 16),  
            datetime(2025, 1, 17),
            datetime(2025, 1, 20),
            datetime(2025, 1, 21)
        ],
        'selected_symbols': ['1662', '6954', '6954', '6954', '6954'],
        'execution_method': '個別実行（debug_get_optimal_symbol_investigation.py）'
    }
    
    print(f"[P2-2成功例]")
    for i, (date, symbol) in enumerate(zip(p2_2_success_data['target_dates'], p2_2_success_data['selected_symbols'])):
        print(f"  - Day {i+1}: {date.strftime('%Y-%m-%d')} -> {symbol}")
    
    print(f"[P3統合実行] （今回の調査対象）")
    print(f"  - target_date: 2025-01-15 -> None （問題）")
    
    print(f"\n[差異分析]")
    print(f"  - 実行方法: P2-2は個別実行、P3は統合実行")
    print(f"  - 初期化: P2-2は単体初期化、P3は統合初期化")
    print(f"  - パラメータ: 同一（2025-01-15, target_symbols=None）")

if __name__ == "__main__":
    success, selected_symbol = investigate_p2_1_a_get_optimal_symbol_integration()
    
    if success:
        print(f"\n✅ P2-1-A調査完了:")
        print(f"   統合実行時の戻り値: {selected_symbol}")
        
        # P2-2との比較
        compare_with_p2_2_success()
        
    else:
        print(f"\n❌ P2-1-A調査失敗")