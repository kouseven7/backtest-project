"""
Simple test for TrendStrategyMatrix
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

try:
    print("1. インポートテスト...")
    from analysis.trend_strategy_matrix import TrendStrategyMatrix
    print("   ✓ TrendStrategyMatrix インポート成功")
    
    print("\n2. サンプルデータ作成...")
    # シンプルなサンプルデータ
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    np.random.seed(42)
    prices = 100 * (1 + np.random.normal(0, 0.01, len(dates))).cumprod()
    
    stock_data = pd.DataFrame({
        'Open': prices,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Adj Close': prices,
        'Volume': 1000000
    }, index=dates)
    
    labeled_data = pd.DataFrame({
        'trend': ['uptrend'] * 30 + ['downtrend'] * 30 + ['range-bound'] * (len(dates) - 60),
        'trend_confidence': 0.8,
        'trend_reliable': True
    }, index=dates)
    
    print(f"   ✓ サンプルデータ作成完了: {len(stock_data)}日間")
    
    print("\n3. TrendStrategyMatrix初期化...")
    matrix = TrendStrategyMatrix(
        stock_data=stock_data,
        labeled_data=labeled_data,
        price_column="Adj Close"
    )
    print("   ✓ 初期化成功")
    
    print("\n4. テスト完了！")
    print("TrendStrategyMatrixが正常に動作しています。")
    
except Exception as e:
    print(f"\nエラー発生: {e}")
    import traceback
    traceback.print_exc()
