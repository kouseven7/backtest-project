#!/usr/bin/env python3
"""
シンプルなトレンド判定テスト
"""

import sys
import os
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# デバッグ用：パスの確認
print(f"現在のパス: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    # データ取得
    from data_fetcher import get_parameters_and_data
    ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
    
    print("=== データ取得成功 ===")
    print(f"銘柄: {ticker}")
    print(f"データ数: {len(stock_data)}")
    
    # トレンド判定のテスト
    from indicators.trend_analysis import detect_trend_with_confidence
    
    trend, confidence = detect_trend_with_confidence(stock_data)
    print(f"\n=== 最新トレンド判定 ===")
    print(f"トレンド: {trend}")
    print(f"信頼度: {confidence:.1%}")
    
    # 精度測定のテスト
    from indicators.trend_accuracy_validator import TrendAccuracyValidator
    
    validator = TrendAccuracyValidator(stock_data, "Adj Close")
    
    # シンプルなSMAトレンド判定器
    def simple_sma_detector(data):
        if len(data) < 20:
            return "unknown"
        
        short_sma = data["Adj Close"].rolling(10).mean().iloc[-1]
        long_sma = data["Adj Close"].rolling(20).mean().iloc[-1]
        current_price = data["Adj Close"].iloc[-1]
        
        if current_price > short_sma > long_sma:
            return "uptrend"
        elif current_price < short_sma < long_sma:
            return "downtrend"
        else:
            return "range-bound"
    
    # 単一手法での精度検証
    print(f"\n=== 精度検証実行中 ===")
    
    validation_params = {"future_window": 10, "trend_threshold": 0.02}
    accuracy_results = validator.validate_trend_accuracy(simple_sma_detector, validation_params)
    
    if "error" in accuracy_results:
        print(f"エラー: {accuracy_results['error']}")
    else:
        print(f"全体精度: {accuracy_results['overall_accuracy']:.1%}")
        print(f"上昇トレンド精度: {accuracy_results['uptrend_accuracy']:.1%}")
        print(f"下降トレンド精度: {accuracy_results['downtrend_accuracy']:.1%}")
        print(f"レンジ相場精度: {accuracy_results['range-bound_accuracy']:.1%}")
        print(f"サンプル数: {accuracy_results['total_samples']}")
        
        # 9割精度チェック
        if accuracy_results['overall_accuracy'] >= 0.9:
            print("\n○○ 目標精度90%を達成！")
        else:
            print(f"\n△△ 目標精度90%に未達（現在: {accuracy_results['overall_accuracy']:.1%}）")
    
    print("\n=== テスト完了 ===")
    
except Exception as e:
    print(f"エラー: {e}")
    import traceback
    traceback.print_exc()
