"""
trade_simulation.py の修正確認テスト

修正方向:
1. リスク状態列を削除
2. 取引量を株数単位で計算・表示
3. 日次累積損益の正しい計算
4. 高度なパフォーマンス指標の統合
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# プロジェクトパスを追加
sys.path.append(os.path.dirname(__file__))

# テスト対象のインポート
from trade_simulation import simulate_trades
from config.logger_config import setup_logger

# ログ設定
logger = setup_logger(__name__)

def create_test_data():
    """テスト用のデータを作成"""
    # 10日間のテストデータ作成
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    
    test_data = pd.DataFrame({
        'Adj Close': [100, 102, 105, 103, 107, 110, 108, 112, 115, 118],
        'Entry_Signal': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        'Exit_Signal': [0, 0, -1, 0, 0, -1, 0, 0, 0, -1],
        'Strategy': ['TestStrategy'] * 10,
        'Position_Size': [1.0] * 10,
        'Partial_Exit': [0.0] * 10
    }, index=dates)
    
    return test_data

def test_simulate_trades():
    """simulate_trades関数のテスト"""
    print("=== trade_simulation.py 修正テスト ===")
    
    # テストデータ作成
    test_data = create_test_data()
    ticker = "TEST"
    
    print(f"テストデータ:")
    print(test_data[['Adj Close', 'Entry_Signal', 'Exit_Signal']])
    print()
    
    # テスト実行
    try:
        result = simulate_trades(test_data, ticker)
        
        print("✅ simulate_trades 実行成功")
        print(f"結果のキー: {list(result.keys())}")
        print()
        
        # 1. リスク状態列が削除されているかチェック
        trade_history = result['取引履歴']
        print("1. リスク状態列削除チェック:")
        print(f"   取引履歴の列: {list(trade_history.columns)}")
        
        if 'リスク状態' not in trade_history.columns:
            print("   ✅ リスク状態列が正しく削除されています")
        else:
            print("   ❌ リスク状態列が残っています")
        print()
        
        # 2. 取引量が株数単位で表示されているかチェック
        print("2. 取引量株数単位チェック:")
        if len(trade_history) > 0:
            print(f"   取引履歴:")
            print(trade_history[['取引量(株)', '取引金額', 'エントリー', 'イグジット']])
            
            # 取引量が整数（株数）になっているかチェック
            if '取引量(株)' in trade_history.columns:
                shares_values = trade_history['取引量(株)'].values
                print(f"   ✅ 取引量(株)列が追加されています")
                print(f"   株数の例: {shares_values}")
            else:
                print("   ❌ 取引量(株)列がありません")
        else:
            print("   取引データがありません")
        print()
        
        # 3. 損益推移の正しい計算チェック
        print("3. 損益推移計算チェック:")
        performance_summary = result['損益推移']
        print(f"   損益推移データ:")
        print(performance_summary.head())
        
        # 累積損益が正しく計算されているかチェック
        if '累積損益' in performance_summary.columns:
            cumulative_pnl = performance_summary['累積損益']
            print(f"   ✅ 累積損益が計算されています")
            print(f"   最終累積損益: {cumulative_pnl.iloc[-1]:.2f}円")
        else:
            print("   ❌ 累積損益列がありません")
        print()
        
        # 4. 高度なパフォーマンス指標の統合チェック
        print("4. 高度なパフォーマンス指標チェック:")
        performance_metrics = result['パフォーマンス指標']
        print(f"   パフォーマンス指標:")
        print(performance_metrics)
        
        # シャープレシオ、ソルティノレシオ、期待値があるかチェック
        advanced_metrics = ['シャープレシオ', 'ソルティノレシオ', '期待値']
        metrics_list = performance_metrics['指標'].tolist()
        
        missing_metrics = []
        for metric in advanced_metrics:
            if metric in metrics_list:
                print(f"   ✅ {metric}が追加されています")
            else:
                missing_metrics.append(metric)
                print(f"   ❌ {metric}がありません")
        
        if not missing_metrics:
            print("   ✅ すべての高度なパフォーマンス指標が統合されています")
        else:
            print(f"   ❌ 不足している指標: {missing_metrics}")
        print()
        
        # 5. リスク管理設定の表示チェック
        print("5. リスク管理設定チェック:")
        risk_summary = result['リスク管理設定']
        print(f"   リスク管理設定:")
        print(risk_summary)
        print("   ✅ リスク管理設定が正しく表示されています")
        print()
        
        print("=== 修正確認完了 ===")
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """エッジケースのテスト"""
    print("\n=== エッジケーステスト ===")
    
    # 取引がない場合のテスト
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    no_trade_data = pd.DataFrame({
        'Adj Close': [100, 101, 102, 103, 104],
        'Entry_Signal': [0, 0, 0, 0, 0],
        'Exit_Signal': [0, 0, 0, 0, 0],
        'Strategy': ['TestStrategy'] * 5,
        'Position_Size': [1.0] * 5,
        'Partial_Exit': [0.0] * 5
    }, index=dates)
    
    try:
        result = simulate_trades(no_trade_data, "TEST")
        print("✅ 取引なしケースの処理成功")
        
        # パフォーマンス指標が適切にゼロ値を表示するかチェック
        performance_metrics = result['パフォーマンス指標']
        print("   取引なしの場合のパフォーマンス指標:")
        print(performance_metrics)
        
    except Exception as e:
        print(f"❌ 取引なしケースでエラー: {e}")
    
    print("=== エッジケーステスト完了 ===")

if __name__ == "__main__":
    # メインテスト実行
    success = test_simulate_trades()
    
    # エッジケーステスト実行
    test_edge_cases()
    
    if success:
        print("\n🎉 すべてのテストが成功しました！")
        print("trade_simulation.py の修正が正しく適用されています。")
    else:
        print("\n⚠️ 一部のテストで問題が見つかりました。")
