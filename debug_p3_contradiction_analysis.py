"""
P3矛盾解析: P2-4-C調査結果とP3実行結果の矛盾を検証

P2-4-C2調査結果:
- result['symbol']: '1662' (最終的に正常値)
- result['success']: False
- execution_details: 0

P3実行結果:
- symbol=None, success=False

この矛盾の原因を特定する
"""
import sys
import os
from datetime import datetime

# プロジェクトルートをsys.pathに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

def debug_p3_contradiction_analysis():
    """
    P3矛盾解析: P2-4-C vs P3実行結果の差異確認
    """
    print("=== P3矛盾解析: P2-4-C vs P3実行結果の差異確認 ===")
    
    # DSSMSIntegratedBacktester初期化
    backtester = DSSMSIntegratedBacktester()
    print("✅ DSSMSIntegratedBacktester初期化成功")
    
    target_date = datetime(2025, 1, 15)
    
    # Step 1: P2-4-C2と同じ単体実行
    print(f"\n[STEP 1] P2-4-C2再現: _process_daily_trading()直接実行")
    try:
        result_single = backtester._process_daily_trading(target_date, None)
        print(f"  単体実行result['symbol']: {repr(result_single['symbol'])}")
        print(f"  単体実行result['success']: {result_single['success']}")
        print(f"  単体実行execution_details: {len(result_single.get('execution_details', []))}")
    except Exception as e:
        print(f"  ❌ 単体実行エラー: {e}")
    
    # Step 2: P3と同じ統合実行
    print(f"\n[STEP 2] P3再現: run_dynamic_backtest()統合実行")
    try:
        start_date = datetime(2025, 1, 15)
        end_date = datetime(2025, 1, 15)
        
        # 統合実行
        results = backtester.run_dynamic_backtest(
            start_date=start_date, 
            end_date=end_date
        )
        
        print(f"  統合実行results: {len(results)}件")
        if results:
            last_result = results[-1]
            print(f"  統合実行result['symbol']: {repr(last_result.get('symbol'))}")
            print(f"  統合実行result['success']: {last_result.get('success')}")
            print(f"  統合実行execution_details: {len(last_result.get('execution_details', []))}")
            
            # daily_resultの詳細比較
            print(f"\n  [詳細比較]")
            print(f"    単体 vs 統合 symbol: {repr(result_single.get('symbol'))} vs {repr(last_result.get('symbol'))}")
            print(f"    単体 vs 統合 success: {result_single.get('success')} vs {last_result.get('success')}")
            print(f"    一致判定: {result_single.get('symbol') == last_result.get('symbol')}")
        else:
            print(f"  ❌ 統合実行結果が空")
            
    except Exception as e:
        print(f"  ❌ 統合実行エラー: {e}")
    
    # Step 3: 実行コンテキスト差異確認
    print(f"\n[STEP 3] 実行コンテキスト差異確認")
    print(f"  単体実行時backtester.current_symbol: {repr(backtester.current_symbol)}")
    print(f"  統合実行後backtester.current_symbol: {repr(backtester.current_symbol)}")
    
    print("\n=== P3矛盾解析完了 ===")

if __name__ == "__main__":
    debug_p3_contradiction_analysis()