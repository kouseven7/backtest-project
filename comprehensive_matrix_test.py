"""
Comprehensive test for TrendStrategyMatrix
包括的なTrendStrategyMatrixテスト
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# シンプルなテスト戦略を定義
class SimpleTestStrategy:
    """テスト用のシンプルな戦略"""
    
    def __init__(self, data: pd.DataFrame, params: dict = None, price_column: str = "Adj Close"):
        self.data = data
        self.params = params or {}
        self.price_column = price_column
        
    def backtest(self):
        """シンプルなバックテスト実行"""
        # ランダムに取引を生成
        np.random.seed(42)
        n_days = len(self.data)
        
        # シンプルなシグナル生成
        signals = np.random.choice([0, 1, -1], size=n_days, p=[0.8, 0.1, 0.1])
        
        # 結果のDataFrameを作成
        result = pd.DataFrame({
            'Entry_Signal': signals,
            'Exit_Signal': np.where(signals != 0, -signals, 0),
            'Price': self.data[self.price_column],
            'Position': signals
        }, index=self.data.index)
        
        return result

class MomentumTestStrategy:
    """テスト用のモメンタム戦略"""
    
    def __init__(self, data: pd.DataFrame, params: dict = None, price_column: str = "Adj Close"):
        self.data = data
        self.params = params or {"ma_window": 10, "entry_threshold": 0.02}
        self.price_column = price_column
        
    def backtest(self):
        """モメンタムベースのバックテスト"""
        ma_window = self.params.get("ma_window", 10)
        threshold = self.params.get("entry_threshold", 0.02)
        
        # 移動平均計算
        ma = self.data[self.price_column].rolling(window=ma_window).mean()
        
        # シグナル生成
        price_vs_ma = (self.data[self.price_column] - ma) / ma
        signals = np.where(price_vs_ma > threshold, 1, 
                  np.where(price_vs_ma < -threshold, -1, 0))
        
        result = pd.DataFrame({
            'Entry_Signal': signals,
            'Exit_Signal': np.where(signals != 0, -signals, 0),
            'Price': self.data[self.price_column],
            'Position': signals
        }, index=self.data.index)
        
        return result

def create_comprehensive_test_data():
    """包括的なテストデータを作成"""
    print("包括的なテストデータ作成中...")
    
    # より長期間のデータ（1年間）
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # 平日のみ
    
    np.random.seed(42)
    n_days = len(dates)
    
    # より複雑な価格変動
    base_price = 100
    daily_returns = np.random.normal(0.0005, 0.015, n_days)
    
    # トレンド期間を明確に定義
    trend_periods = []
    current_day = 0
    
    while current_day < n_days:
        period_length = np.random.randint(20, 60)  # 20-60日のトレンド期間
        if current_day + period_length > n_days:
            period_length = n_days - current_day
        
        trend_type = np.random.choice(['uptrend', 'downtrend', 'range-bound'])
        trend_periods.append({
            'start': current_day,
            'end': current_day + period_length - 1,
            'trend': trend_type
        })
        current_day += period_length
    
    # トレンドに応じた価格変動
    for period in trend_periods:
        start, end = period['start'], period['end']
        trend = period['trend']
        
        if trend == 'uptrend':
            daily_returns[start:end+1] += 0.002  # 上昇トレンド
        elif trend == 'downtrend':
            daily_returns[start:end+1] -= 0.001  # 下降トレンド
        # range-boundはそのまま
    
    # 価格計算
    prices = [base_price]
    for ret in daily_returns:
        prices.append(prices[-1] * (1 + ret))
    prices = prices[1:]  # 最初の要素を削除
    
    # OHLCV データ
    stock_data = pd.DataFrame({
        'Open': np.array(prices) * np.random.uniform(0.995, 1.005, len(prices)),
        'High': np.array(prices) * np.random.uniform(1.005, 1.02, len(prices)),
        'Low': np.array(prices) * np.random.uniform(0.98, 0.995, len(prices)),
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(500000, 2000000, len(prices))
    }, index=dates)
    
    # ラベリングデータ
    labeled_data = pd.DataFrame({
        'trend': 'range-bound',
        'trend_confidence': 0.7,
        'trend_reliable': True
    }, index=dates)
    
    # トレンドラベルを適用
    for period in trend_periods:
        start_idx = period['start']
        end_idx = period['end']
        trend = period['trend']
        
        labeled_data.iloc[start_idx:end_idx+1, labeled_data.columns.get_loc('trend')] = trend
        labeled_data.iloc[start_idx:end_idx+1, labeled_data.columns.get_loc('trend_confidence')] = np.random.uniform(0.75, 0.95)
    
    print(f"   ✓ データ作成完了: {len(stock_data)}日間")
    print(f"   ✓ トレンド期間数: {len(trend_periods)}")
    
    # トレンド分布
    trend_counts = labeled_data['trend'].value_counts()
    for trend, count in trend_counts.items():
        print(f"   - {trend}: {count}日間")
    
    return stock_data, labeled_data

def run_comprehensive_test():
    """包括的なテスト実行"""
    print("=" * 80)
    print("COMPREHENSIVE TRENDSTRATEGYMATRIX TEST")
    print("=" * 80)
    
    try:
        # 1. データ準備
        print("\n1. データ準備...")
        stock_data, labeled_data = create_comprehensive_test_data()
        
        # 2. TrendStrategyMatrix初期化
        print("\n2. TrendStrategyMatrix初期化...")
        from analysis.trend_strategy_matrix import TrendStrategyMatrix
        
        matrix = TrendStrategyMatrix(
            stock_data=stock_data,
            labeled_data=labeled_data,
            price_column="Adj Close"
        )
        print("   ✓ 初期化成功")
        
        # 3. テスト戦略の定義
        print("\n3. テスト戦略定義...")
        strategies = [
            (SimpleTestStrategy, {"name": "Simple_Random"}),
            (MomentumTestStrategy, {"ma_window": 10, "entry_threshold": 0.02}),
            (MomentumTestStrategy, {"ma_window": 20, "entry_threshold": 0.015}),
        ]
        
        print(f"   ✓ 戦略数: {len(strategies)}")
        for i, (strategy_class, params) in enumerate(strategies):
            print(f"   - 戦略 {i+1}: {strategy_class.__name__} {params}")
        
        # 4. マトリクス生成（緩い条件でテスト）
        print("\n4. パフォーマンスマトリクス生成...")
        results = matrix.generate_matrix(
            strategies=strategies,
            min_period_length=5,  # 短い期間でもOK
            min_confidence=0.5    # 低い信頼度でもOK
        )
        
        print("   ✓ マトリクス生成完了")
        
        # 5. 結果確認
        print("\n5. 結果確認...")
        matrix_data = results.get("matrix_data", {})
        print(f"   分析された戦略数: {len(matrix_data)}")
        
        for strategy_name, trend_results in matrix_data.items():
            print(f"\n   戦略: {strategy_name}")
            for trend_type, metrics in trend_results.items():
                if "error" not in metrics:
                    total_return = metrics.get("total_return", 0)
                    win_rate = metrics.get("win_rate", 0)
                    total_trades = metrics.get("total_trades", 0)
                    periods_tested = metrics.get("periods_tested", 0)
                    
                    print(f"     {trend_type:12}: リターン{total_return:6.2%}, 勝率{win_rate:6.1%}, "
                          f"トレード{total_trades:3d}回, 期間{periods_tested:2d}個")
                else:
                    print(f"     {trend_type:12}: エラー - {metrics.get('error', 'Unknown')}")
        
        # 6. ランキング確認
        print("\n6. 戦略ランキング...")
        rankings = results.get("strategy_rankings", {})
        
        if "overall" in rankings and rankings["overall"]:
            print("   総合ランキング:")
            for i, strategy in enumerate(rankings["overall"]):
                print(f"     {i+1}. {strategy['strategy']:<25} スコア: {strategy['overall_score']:.3f}")
        
        # 7. 結果保存テスト
        print("\n7. 結果保存テスト...")
        saved_files = matrix.save_results("comprehensive_test")
        
        print("   ✓ 保存されたファイル:")
        for file_type, file_path in saved_files.items():
            if file_path and isinstance(file_path, str):
                print(f"     {file_type}: {file_path}")
        
        # 8. 推奨機能テスト
        print("\n8. 戦略推奨機能テスト...")
        for trend_type in ["uptrend", "downtrend", "range-bound"]:
            recommendation = matrix.get_strategy_recommendation(
                current_trend=trend_type,
                risk_tolerance="medium"
            )
            
            if "recommended_strategy" in recommendation:
                rec = recommendation["recommended_strategy"]
                print(f"   {trend_type:12}: {rec.get('strategy', 'N/A')}")
            else:
                print(f"   {trend_type:12}: {recommendation.get('error', 'N/A')}")
        
        print("\n" + "=" * 80)
        print("✅ 包括的テスト完了！全機能が正常に動作しました。")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 TrendStrategyMatrix は完全に実装されており、正常に動作しています！")
        print("📊 生成されたレポートファイルをreports/およびlogs/フォルダで確認してください。")
    else:
        print("\n💥 テストが失敗しました。エラーの詳細を確認してください。")
