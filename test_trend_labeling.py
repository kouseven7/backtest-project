#!/usr/bin/env python3
"""
trend_labeling.py のテストスクリプト
"""

import sys
import os
sys.path.append(r'C:\Users\imega\Documents\my_backtest_project')

def test_trend_labeling():
    """trend_labeling.pyの基本機能をテスト"""
    print("=== trend_labeling.py テスト開始 ===")
    
    # インポートテスト
    try:
        from indicators.trend_labeling import TrendLabeler, label_trends_for_dataframe
        print("✓ trend_labeling.py のインポートに成功しました")
    except Exception as e:
        print(f"✗ trend_labeling.py のインポートに失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # データ取得テスト
    try:
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, _ = get_parameters_and_data()
        print(f"✓ データ取得に成功: {len(stock_data)}行のデータ")
        print(f"  データ期間: {stock_data.index[0]} から {stock_data.index[-1]}")
        print(f"  カラム: {list(stock_data.columns)}")
    except Exception as e:
        print(f"✗ データ取得に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ラベリングテスト
    try:
        # 少量のデータでテスト
        test_data = stock_data.iloc[-100:].copy()
        print(f"✓ テストデータ準備完了: {len(test_data)}行")
        
        labeler = TrendLabeler(test_data)
        print("✓ TrendLabeler初期化完了")
        
        labeled_data = labeler.label_trends(method='advanced', window_size=10)
        print("✓ ラベリング処理完了")
        
        # 結果の確認
        trend_counts = labeled_data['trend'].value_counts()
        print(f"✓ ラベリングに成功")
        print(f"  トレンド分布: {dict(trend_counts)}")
        print(f"  平均信頼度: {labeled_data['trend_confidence'].mean():.3f}")
        
        # トレンド期間抽出テスト
        uptrend_periods = labeler.extract_trend_periods("uptrend", min_period_length=3)
        print(f"✓ 上昇トレンド期間抽出: {len(uptrend_periods)}期間")
        
        # 保存テスト
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "test_labeled_data.csv")
        success = labeler.save_labeled_data(save_path)
        if success:
            print(f"✓ ラベリングデータの保存に成功: {save_path}")
        else:
            print("✗ ラベリングデータの保存に失敗")
            
    except Exception as e:
        print(f"✗ ラベリングに失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=== テスト完了 ===")
    return True

if __name__ == "__main__":
    success = test_trend_labeling()
    if success:
        print("すべてのテストが成功しました！")
    else:
        print("一部のテストが失敗しました。")
