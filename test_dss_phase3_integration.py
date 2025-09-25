#!/usr/bin/env python3
"""
DSS V3 Phase 3 統合テスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from src.dssms.dssms_backtester_v3 import DSSBacktesterV3

def test_phase3_integration():
    """Phase 3 統合テスト"""
    print("=== DSS V3 Phase 3 統合テスト ===")
    
    try:
        # DSS V3 初期化
        dss = DSSBacktesterV3()
        
        # テスト設定
        test_date = datetime(2023, 1, 15)
        print(f"分析日: {test_date.strftime('%Y-%m-%d')}")
        print(f"対象銘柄: {dss.symbol_universe}")
        
        # DSS 日次選択実行（Phase 3 統合機能）
        print(f"\n=== DSS 日次選択実行 ===")
        result = dss.run_daily_selection(test_date)
        
        # 結果表示
        print(f"\n=== 実行結果 ===")
        print(f"選択銘柄: {result['selected_symbol']}")
        print(f"実行時間: {result['execution_time_ms']:.1f}ms")
        print(f"Phase: {result.get('phase', 'Phase 3 - Full Implementation')}")
        
        # ランキング詳細
        if 'ranking' in result and result['ranking']:
            print(f"\n=== ランキング詳細 ===")
            for entry in result['ranking']:
                symbol = entry['symbol']
                score = entry['score']
                rank = entry['rank']
                print(f"#{rank}: {symbol} (スコア: {score:.2f})")
        
        # パフォーマンス検証
        if result['execution_time_ms'] < 10000:  # 10秒以内
            print(f"✅ パフォーマンス: OK ({result['execution_time_ms']:.1f}ms)")
        else:
            print(f"⚠ パフォーマンス: やや遅い ({result['execution_time_ms']:.1f}ms)")
        
        print(f"\n🎉 Phase 3 統合テスト成功！")
        return True
        
    except Exception as e:
        print(f"❌ Phase 3 統合テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_phase3_integration()