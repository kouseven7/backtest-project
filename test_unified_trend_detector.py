"""
Module: Test Unified Trend Detector
File: test_unified_trend_detector.py
Description: 
  統一トレンド判定インターフェースのテストスクリプトです。
  様々な戦略に対してトレンド判定を行い、信頼度や互換性を検証します。

Author: imega
Created: 2025-07-03
Modified: 2025-07-03

Dependencies:
  - pandas
  - indicators.unified_trend_detector
  - data_fetcher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend
from data_fetcher import fetch_stock_data
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO, 
                   format='[%(levelname)s] %(asctime)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def test_unified_trend_detection():
    """基本的なトレンド判定テスト"""
    # サンプルデータの作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    
    # 上昇トレンドデータ
    up_df = pd.DataFrame({
        'High': [100 + i * 0.7 + np.random.random() * 3 for i in range(100)],
        'Low': [95 + i * 0.7 - np.random.random() * 2 for i in range(100)],
        'Adj Close': [98 + i * 0.7 for i in range(100)],
        'Volume': np.random.randint(100, 1000, 100),
        'VWAP': [97 + i * 0.65 for i in range(100)]
    }, index=dates)
    
    # 下降トレンドデータ
    down_df = pd.DataFrame({
        'High': [150 - i * 0.5 + np.random.random() * 3 for i in range(100)],
        'Low': [145 - i * 0.5 - np.random.random() * 2 for i in range(100)],
        'Adj Close': [148 - i * 0.5 for i in range(100)],
        'Volume': np.random.randint(100, 1000, 100),
        'VWAP': [147 - i * 0.45 for i in range(100)]
    }, index=dates)
    
    # レンジ相場データ
    range_df = pd.DataFrame({
        'High': [100 + np.sin(i/5) * 5 + np.random.random() * 2 for i in range(100)],
        'Low': [95 + np.sin(i/5) * 5 - np.random.random() * 2 for i in range(100)],
        'Adj Close': [98 + np.sin(i/5) * 5 for i in range(100)],
        'Volume': np.random.randint(100, 1000, 100),
        'VWAP': [97 + np.sin(i/5) * 4.5 for i in range(100)]
    }, index=dates)
    
    datasets = {
        "uptrend": up_df,
        "downtrend": down_df,
        "range-bound": range_df
    }
    
    methods = ["sma", "macd", "combined", "advanced"]
    strategies = ["default", "VWAP_Bounce", "VWAP_Breakout", "Golden_Cross"]
    
    results = []
    
    # 各データセット、メソッド、戦略の組み合わせをテスト
    for data_type, df in datasets.items():
        print(f"\n=== テスト: {data_type}データ ===")
        
        for method in methods:
            for strategy in strategies:
                # 統一トレンド判定
                detector = UnifiedTrendDetector(df, price_column="Adj Close", 
                                               strategy_name=strategy,
                                               method=method,
                                               vwap_column="VWAP")
                
                trend, confidence = detector.detect_trend_with_confidence()
                conf_level = detector.get_trend_confidence_level()
                is_compatible = detector.is_strategy_compatible()
                
                # 結果を保存
                result = {
                    "データ種類": data_type,
                    "メソッド": method,
                    "戦略": strategy,
                    "判定トレンド": trend,
                    "信頼度": confidence,
                    "信頼度レベル": conf_level,
                    "戦略適合": is_compatible,
                    "検出精度": 1.0 if trend == data_type else 0.0
                }
                results.append(result)
                
                # 結果を表示
                print(f"メソッド: {method}, 戦略: {strategy}")
                print(f"  結果: {trend}, 信頼度: {confidence:.2f} ({conf_level})")
                print(f"  戦略適合性: {'適合' if is_compatible else '不適合'}")
                print(f"  精度: {'正解' if trend == data_type else '不正解'}")
    
    # 結果をデータフレームに変換
    results_df = pd.DataFrame(results)
    
    # 精度の集計
    method_accuracy = results_df.groupby("メソッド")["検出精度"].mean()
    strategy_accuracy = results_df.groupby("戦略")["検出精度"].mean()
    
    print("\n=== メソッド別精度 ===")
    for method, acc in method_accuracy.items():
        print(f"{method}: {acc:.2%}")
    
    print("\n=== 戦略別精度 ===")
    for strategy, acc in strategy_accuracy.items():
        print(f"{strategy}: {acc:.2%}")
    
    return results_df

def test_simple_trend_detection():
    """単純なサンプルデータでトレンド判定をテスト"""
    # サンプルデータの作成（簡易版）
    dates = pd.date_range(start="2023-01-01", periods=50, freq='B')
    
    # 上昇トレンド
    uptrend_data = pd.DataFrame({
        'High': [100 + i for i in range(50)],
        'Low': [95 + i for i in range(50)],
        'Close': [98 + i for i in range(50)],
        'Adj Close': [98 + i for i in range(50)],
        'Volume': np.random.randint(1000, 5000, 50),
        'VWAP': [97 + i for i in range(50)]
    }, index=dates)
    
    # トレンド判定をテスト
    print("\n=== 簡易版トレンドテスト ===")
    
    # 様々な戦略でテスト
    strategies = ["default", "VWAP_Bounce", "VWAP_Breakout", "Golden_Cross"]
    methods = ["sma", "macd", "combined", "advanced"]
    
    for strategy in strategies:
        print(f"\n【戦略: {strategy}】")
        
        for method in methods:
            try:
                detector = UnifiedTrendDetector(
                    uptrend_data, 
                    price_column="Adj Close",
                    strategy_name=strategy, 
                    method=method,
                    vwap_column="VWAP"
                )
                
                trend, confidence = detector.detect_trend_with_confidence()
                
                print(f"メソッド: {method}")
                print(f"  判定: {trend}, 信頼度: {confidence:.2f}")
                print(f"  説明: {detector.get_trend_description()}")
                
            except Exception as e:
                print(f"  【エラー】メソッド {method}: {e}")

if __name__ == "__main__":
    print("=== 統一トレンド判定テスト ===")
    print("\n1. 基本トレンドテスト")
    try:
        results = test_unified_trend_detection()
        print("\n基本テスト成功")
    except Exception as e:
        print(f"基本テストでエラー: {e}")
        
    print("\n2. 単純トレンドテスト")
    try:
        test_simple_trend_detection()
        print("\n単純テスト成功")
    except Exception as e:
        print(f"単純トレンドテストでエラー: {e}")
