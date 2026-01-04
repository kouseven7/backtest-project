"""
DSS Core V3 run_daily_selection() 内部動作詳細分析テスト

Phase 1.5調査用: run_daily_selection()の各ステップで何が起きているかを詳細分析

Author: Backtest Project Team
Created: 2026-01-03
"""
import sys
import os
import traceback
from datetime import datetime
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dssms.dssms_backtester_v3 import DSSBacktesterV3

def test_run_daily_selection_detailed():
    """run_daily_selection()の内部動作を詳細分析"""
    print("=== DSS Core V3 run_daily_selection() 内部動作詳細分析 ===")
    
    # ログ設定を詳細レベルに
    logging.basicConfig(level=logging.DEBUG, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        print("Step 1: DSSBacktesterV3インスタンス化")
        backtester = DSSBacktesterV3()
        print(f"  - backtester作成成功: {type(backtester)}")
        
        print("\nStep 2: DSSBacktesterV3確認")
        print(f"  - コンストラクタ完了（__init__内で初期化実行）")
        print(f"  - type: {type(backtester)}")
        
        print("\nStep 3: 初期化状態確認")
        print(f"  - symbol_universe: {backtester.symbol_universe[:5] if backtester.symbol_universe else 'None'}...")
        print(f"  - components初期化済み: {hasattr(backtester, 'components')}")
        
        print("\nStep 4: run_daily_selection()実行")
        target_date = datetime(2025, 1, 15)
        print(f"  - target_date: {target_date}")
        
        # 詳細な実行
        result = backtester.run_daily_selection(target_date)
        
        print("\nStep 5: 実行結果確認")
        print(f"  - result type: {type(result)}")
        print(f"  - result keys: {result.keys() if isinstance(result, dict) else 'Not dict'}")
        
        if isinstance(result, dict):
            print(f"  - selected_symbol: {result.get('selected_symbol')}")
            print(f"  - date: {result.get('date')}")
            print(f"  - execution_time_ms: {result.get('execution_time_ms')}")
            print(f"  - phase: {result.get('phase')}")
            
            if result.get('ranking'):
                print(f"  - ranking count: {len(result['ranking'])}")
                print(f"  - top 3 ranking: {result['ranking'][:3]}")
        
        print("\n=== run_daily_selection() 実行成功 ===")
        return result
        
    except Exception as e:
        print(f"\nERROR: run_daily_selection() 実行失敗")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_run_daily_selection_detailed()
    
    print(f"\n=== 最終結果 ===")
    if result and isinstance(result, dict):
        selected_symbol = result.get('selected_symbol')
        print(f"Selected Symbol: {selected_symbol}")
        if selected_symbol:
            print("✅ SUCCESS: 銘柄選択に成功")
        else:
            print("❌ FAILURE: selected_symbol is None/empty")
    else:
        print("❌ FAILURE: result is None or not dict")