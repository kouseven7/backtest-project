"""
Performance Metrics修正テスト & Main.py エラー修正テスト
Author: imega
Created: 2025-07-22
"""
import sys
import os
from datetime import datetime

# プロジェクトのルートディレクトリを sys.path に追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_performance_metrics_fix():
    """performance_metrics修正のテスト"""
    print("=== Performance Metrics 修正テスト ===")
    
    try:
        from metrics.performance_metrics import calculate_win_rate
        import pandas as pd
        import numpy as np
        
        # テストデータ作成（DataFrame - main.pyエラーの原因）
        test_df = pd.DataFrame({
            'PnL': [100, -50, 200, -30, 150, 0],
            'trade_pnl': [100, -50, 200, -30, 150, 0],  # 別列名もテスト
            'Date': pd.date_range('2023-01-01', periods=6),
            'Strategy': ['TestStrategy'] * 6
        })
        
        print(f"テストDataFrame作成: {len(test_df)}行")
        print(f"PnL列: {test_df['PnL'].tolist()}")
        
        # 修正された関数をテスト（DataFrame入力 - 以前はエラーだった）
        win_rate_df = calculate_win_rate(test_df)
        print(f"[OK] DataFrame入力での勝率計算: {win_rate_df:.2%}")
        
        # Seriesでのテスト
        test_series = pd.Series([100, -50, 200, -30, 150])
        win_rate_series = calculate_win_rate(test_series)
        print(f"[OK] Series入力での勝率計算: {win_rate_series:.2%}")
        
        # 空データのテスト（エラーハンドリング確認）
        empty_df = pd.DataFrame()
        win_rate_empty = calculate_win_rate(empty_df)
        print(f"[OK] 空DataFrame入力での勝率計算: {win_rate_empty:.2%}")
        
        # None入力のテスト（エラーハンドリング確認）
        win_rate_none = calculate_win_rate(None)
        print(f"[OK] None入力での勝率計算: {win_rate_none:.2%}")
        
        print("[OK] Performance Metrics修正テスト - 全て成功")
        return True
        
    except Exception as e:
        print(f"[ERROR] Performance Metrics テストエラー: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_main_execution():
    """main.py実行テスト"""
    print("\n=== Main.py 実行テスト ===")
    
    try:
        # データ取得テスト
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        print(f"[OK] データ取得成功: {ticker} ({len(stock_data)}日分)")
        
        # シミュレーション実行テスト（修正されたperformance_metricsを使用）
        from output.simulation_handler import simulate_and_save
        print("バックテスト実行中...")
        
        results = simulate_and_save(stock_data, ticker)
        
        print("[OK] バックテスト実行成功（エラーなし）")
        print(f"結果: {type(results)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Main.py実行エラー: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """メイン関数"""
    print("=" * 60)
    print("Performance Metrics修正 & Main.py エラー修正テスト")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # 1. Performance Metrics修正テスト
    print("\n[TOOL] Step 1: Performance Metrics修正テスト")
    if not test_performance_metrics_fix():
        print("[ERROR] Performance Metrics修正テスト失敗")
        return False
    
    # 2. Main.py実行テスト
    print("\n[ROCKET] Step 2: Main.py実行テスト")
    if not test_main_execution():
        print("[ERROR] Main.py実行テスト失敗")
        return False
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("[OK] 全テスト成功！")
    print(f"実行時間: {duration.total_seconds():.2f}秒")
    print("Main.pyのエラー修正完了 - 実行可能状態")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] 次は本格的にmain.pyを実行できます:")
        print("   python main.py")
    sys.exit(0 if success else 1)
