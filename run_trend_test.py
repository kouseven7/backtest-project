"""
簡単なトレンド判定テスト
"""
import sys
import pandas as pd
import numpy as np

# プロジェクトパスを追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# 統一トレンドインターフェースをインポート
from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend

# サンプルデータの作成
def create_test_data():
    dates = pd.date_range(start="2023-01-01", periods=50, freq='B')
    
    # 上昇トレンド
    df = pd.DataFrame({
        'High': [100 + i for i in range(50)],
        'Low': [95 + i for i in range(50)],
        'Close': [98 + i for i in range(50)],
        'Adj Close': [98 + i for i in range(50)],
        'Volume': np.random.randint(1000, 5000, 50),
        'VWAP': [97 + i for i in range(50)]
    }, index=dates)
    
    return df

def main():
    # データ作成
    data = create_test_data()
    print("=== 統一トレンド判定テスト ===")
    
    # 戦略とメソッドの組み合わせをテスト
    strategies = ["default", "VWAP_Bounce", "VWAP_Breakout", "Golden_Cross"]
    methods = ["sma", "macd", "combined", "advanced"]
    
    for strategy in strategies:
        print(f"\n【戦略: {strategy}】")
        
        for method in methods:
            try:
                # 統一トレンド判定インターフェースをテスト
                print(f"メソッド: {method}")
                
                # クラスベースAPI
                detector = UnifiedTrendDetector(
                    data, 
                    price_column="Adj Close",
                    strategy_name=strategy, 
                    method=method,
                    vwap_column="VWAP"
                )
                
                trend, confidence = detector.detect_trend_with_confidence()
                print(f"  クラスAPI結果: {trend}, 信頼度: {confidence:.2f}")
                
                # 関数ベースAPI
                func_trend = detect_unified_trend(
                    data,
                    price_column="Adj Close",
                    strategy=strategy,
                    method=method,
                    vwap_column="VWAP"
                )
                
                print(f"  関数API結果: {func_trend}")
                
            except Exception as e:
                print(f"  【エラー】メソッド {method}: {e}")

if __name__ == "__main__":
    main()
