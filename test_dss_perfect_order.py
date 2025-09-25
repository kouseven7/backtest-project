#!/usr/bin/env python3
"""
DSS V3 パーフェクトオーダー計算機能テスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from src.dssms.dssms_backtester_v3 import DSSBacktesterV3

def test_perfect_order_calculation():
    """パーフェクトオーダー計算機能テスト"""
    print("=== DSS V3 パーフェクトオーダー計算テスト ===")
    
    try:
        # DSS V3 初期化
        dss = DSSBacktesterV3()
        
        # テスト銘柄（2銘柄のみ）
        test_symbols = ['7203', '9984']
        test_date = datetime(2023, 1, 15)
        
        print(f"テスト対象: {test_symbols}")
        print(f"分析日: {test_date.strftime('%Y-%m-%d')}")
        
        # 1. データ取得
        print("\n=== 1. データ取得 ===")
        market_data = dss.fetch_market_data(test_symbols, test_date)
        
        # 2. パーフェクトオーダースコア計算
        print("\n=== 2. パーフェクトオーダースコア計算 ===")
        scores = dss.calculate_perfect_order_scores(market_data)
        
        # 結果確認
        print(f"\n=== 計算結果 ===")
        for symbol, score in scores.items():
            print(f"{symbol}: スコア = {score:.2f}")
            if score >= 0.75:
                print(f"  → 🎯 高スコア銘柄！")
            elif score >= 0.5:
                print(f"  → ✅ 中スコア")
            else:
                print(f"  → ⚠ 低スコア")
        
        best_symbol = max(scores.items(), key=lambda x: x[1])
        print(f"\n最高スコア: {best_symbol[0]} ({best_symbol[1]:.2f})")
        
        print(f"\n✅ パーフェクトオーダー計算テスト成功")
        return True
        
    except Exception as e:
        print(f"❌ パーフェクトオーダー計算テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_perfect_order_calculation()