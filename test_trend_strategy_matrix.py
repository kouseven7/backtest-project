"""
Test script for TrendStrategyMatrix
テスト用スクリプト：トレンド×戦略パフォーマンスマトリクス
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスを追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from analysis.trend_strategy_matrix import TrendStrategyMatrix
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.contrarian_strategy import ContrarianStrategy

def create_sample_data() -> pd.DataFrame:
    """テスト用のサンプルデータを作成"""
    print("サンプルデータ作成中...")
    
    # 日付範囲
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 価格データ生成（ランダムウォーク with トレンド）
    np.random.seed(42)
    n_days = len(dates)
    
    # 基本価格（100からスタート）
    base_price = 100
    daily_returns = np.random.normal(0.001, 0.02, n_days)  # 平均0.1%、標準偏差2%
    
    # トレンドを追加
    trend_component = np.sin(np.linspace(0, 4*np.pi, n_days)) * 0.005
    daily_returns += trend_component
    
    prices = [base_price]
    for ret in daily_returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 高値・安値・出来高の生成
    highs = [p * np.random.uniform(1.01, 1.05) for p in prices]
    lows = [p * np.random.uniform(0.95, 0.99) for p in prices]
    volumes = np.random.randint(100000, 1000000, n_days)
    
    # DataFrameを作成
    stock_data = pd.DataFrame({
        'Open': prices,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Adj Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # 土日を除去（平日のみ）
    stock_data = stock_data[stock_data.index.weekday < 5]
    
    print(f"サンプルデータ作成完了: {len(stock_data)}日間")
    return stock_data

def create_sample_labeled_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """サンプルのラベリングデータを作成"""
    print("ラベリングデータ作成中...")
    
    n_days = len(stock_data)
    np.random.seed(123)
    
    # 価格の移動平均でトレンドを判定
    ma_short = stock_data['Adj Close'].rolling(window=20).mean()
    ma_long = stock_data['Adj Close'].rolling(window=50).mean()
    
    # トレンド分類
    trends = []
    confidences = []
    reliable_flags = []
    
    for i in range(len(stock_data)):
        if i < 50:  # 初期期間
            trends.append("range-bound")
            confidences.append(0.5)
            reliable_flags.append(False)
        else:
            ma_short_val = ma_short.iloc[i]
            ma_long_val = ma_long.iloc[i]
            price_change = (stock_data['Adj Close'].iloc[i] / stock_data['Adj Close'].iloc[i-20] - 1)
            
            if ma_short_val > ma_long_val and price_change > 0.05:
                trends.append("uptrend")
                confidences.append(min(0.9, 0.7 + abs(price_change)))
                reliable_flags.append(True)
            elif ma_short_val < ma_long_val and price_change < -0.05:
                trends.append("downtrend") 
                confidences.append(min(0.9, 0.7 + abs(price_change)))
                reliable_flags.append(True)
            else:
                trends.append("range-bound")
                confidences.append(0.6 + np.random.uniform(-0.1, 0.1))
                reliable_flags.append(np.random.choice([True, False], p=[0.7, 0.3]))
    
    labeled_data = pd.DataFrame({
        'trend': trends,
        'trend_confidence': confidences,
        'trend_reliable': reliable_flags
    }, index=stock_data.index)
    
    print(f"ラベリングデータ作成完了:")
    print(f"  - uptrend: {(labeled_data['trend'] == 'uptrend').sum()}日")
    print(f"  - downtrend: {(labeled_data['trend'] == 'downtrend').sum()}日")
    print(f"  - range-bound: {(labeled_data['trend'] == 'range-bound').sum()}日")
    
    return labeled_data

def test_trend_strategy_matrix():
    """TrendStrategyMatrixのテスト実行"""
    print("=" * 80)
    print("TREND STRATEGY MATRIX TEST")
    print("=" * 80)
    
    try:
        # 1. データ準備
        print("\n1. データ準備...")
        stock_data = create_sample_data()
        labeled_data = create_sample_labeled_data(stock_data)
        
        # 2. マトリクス初期化
        print("\n2. TrendStrategyMatrix初期化...")
        matrix = TrendStrategyMatrix(
            stock_data=stock_data,
            labeled_data=labeled_data,
            price_column="Adj Close"
        )
        
        # 3. 戦略定義
        print("\n3. 戦略定義...")
        strategies = [
            (MomentumInvestingStrategy, {
                "ma_short_window": 10,
                "ma_long_window": 20,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70
            }),
            (ContrarianStrategy, {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "gap_threshold": 0.05,
                "stop_loss": 0.04,
                "take_profit": 0.05
            })
        ]
        
        print(f"使用する戦略: {len(strategies)}個")
        for strategy_class, params in strategies:
            print(f"  - {strategy_class.__name__}: {len(params)}パラメータ")
        
        # 4. マトリクス生成
        print("\n4. パフォーマンスマトリクス生成...")
        results = matrix.generate_matrix(
            strategies=strategies,
            min_period_length=5,  # テスト用に短めに設定
            min_confidence=0.6
        )
        
        # 5. 結果確認
        print("\n5. 結果確認...")
        print(f"マトリクス生成完了: {len(results)}セクション")
        
        # 戦略数とトレンド数
        matrix_data = results.get("matrix_data", {})
        print(f"戦略数: {len(matrix_data)}")
        
        for strategy_name, trend_results in matrix_data.items():
            print(f"\n戦略: {strategy_name}")
            for trend_type, metrics in trend_results.items():
                if "error" not in metrics:
                    total_return = metrics.get("total_return", 0)
                    win_rate = metrics.get("win_rate", 0)
                    total_trades = metrics.get("total_trades", 0)
                    adaptation_score = metrics.get("trend_adaptation_score", 0)
                    
                    print(f"  {trend_type:12}: リターン{total_return:6.2%}, 勝率{win_rate:6.1%}, "
                          f"トレード{total_trades:3d}回, 適応度{adaptation_score:.3f}")
                else:
                    print(f"  {trend_type:12}: エラー - {metrics.get('error', 'Unknown')}")
        
        # 6. ランキング確認
        print("\n6. 戦略ランキング...")
        rankings = results.get("strategy_rankings", {})
        
        if "overall" in rankings:
            print("総合ランキング:")
            for i, strategy in enumerate(rankings["overall"][:5]):
                print(f"  {i+1}. {strategy['strategy']:<25} スコア: {strategy['overall_score']:.3f}")
        
        # 7. 結果保存
        print("\n7. 結果保存...")
        saved_files = matrix.save_results("test_trend_strategy_matrix")
        
        print("保存されたファイル:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        
        # 8. 推奨機能テスト
        print("\n8. 戦略推奨機能テスト...")
        recommendation = matrix.get_strategy_recommendation(
            current_trend="uptrend",
            risk_tolerance="medium"
        )
        
        if "recommended_strategy" in recommendation:
            rec = recommendation["recommended_strategy"]
            print(f"推奨戦略: {rec.get('strategy', 'N/A')}")
            print(f"理由: {rec.get('recommendation_reason', 'N/A')}")
        else:
            print(f"推奨エラー: {recommendation.get('error', 'Unknown')}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\nテスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trend_strategy_matrix()
    
    if success:
        print("\n✓ すべてのテストが正常に完了しました")
        print("✓ ファイルが正常に保存されました")
        print("✓ TrendStrategyMatrixは正常に動作しています")
    else:
        print("\n✗ テストが失敗しました")
        print("✗ ログを確認してエラーを修正してください")
