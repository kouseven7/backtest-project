#!/usr/bin/env python3
"""
並列最適化の修正をテストするスクリプト
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

def test_parallel_optimization():
    """並列最適化のテスト"""
    print("=== 並列最適化修正テスト ===")
    
    try:
        # データ取得
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        print(f"データ取得完了: {len(stock_data)}行")
        
        # テストデータを準備（少量）
        test_data = stock_data.iloc[-200:].copy()
        test_index = index_data.iloc[-200:].copy() if index_data is not None else None
        print(f"テストデータ準備完了: {len(test_data)}行")
        
        # 並列最適化実行
        from optimization.optimize_vwap_breakout_strategy import optimize_vwap_breakout_strategy
        result = optimize_vwap_breakout_strategy(test_data, test_index, use_parallel=True)
        
        if result is not None and not result.empty:
            print(f"✅ 並列最適化テスト成功: {len(result)}件の結果")
            print(f"最良スコア: {result.iloc[0]['score']}")
            return True
        else:
            print("❌ 最適化結果が空です")
            return False
            
    except Exception as e:
        print(f"❌ 並列最適化テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parallel_optimization()
    if success:
        print("\n🎉 すべてのテストが成功しました！")
    else:
        print("\n💥 テストが失敗しました")
