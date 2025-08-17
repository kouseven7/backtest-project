"""
DSSMS Phase 1 統合テストスクリプト
Task 1.3: 日経225スクリーナー統合テスト

全コンポーネントの動作確認とエラーハンドリングテスト
"""

import sys
from pathlib import Path
import time
from datetime import datetime

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# DSSMSモジュールインポート
from src.dssms.nikkei225_screener import Nikkei225Screener
from src.dssms.dssms_data_manager import DSSMSDataManager
from src.dssms.fundamental_analyzer import FundamentalAnalyzer
from src.dssms.perfect_order_detector import PerfectOrderDetector

def test_nikkei225_screener():
    """日経225スクリーナーテスト"""
    print("=== Test 1: Nikkei225 Screener ===")
    
    try:
        screener = Nikkei225Screener()
        
        # 100万円の利用可能資金でテスト
        available_funds = 1_000_000
        
        # 統計取得
        print("1.1 Getting screening statistics...")
        stats = screener.get_screening_statistics(available_funds)
        print("Screening Statistics:")
        for stage, count in stats.items():
            print(f"  {stage}: {count} symbols")
        
        # 最終選定
        print("\n1.2 Running full screening...")
        filtered_symbols = screener.get_filtered_symbols(available_funds)
        print(f"Final selected symbols ({len(filtered_symbols)}):")
        for symbol in filtered_symbols:
            print(f"  {symbol}")
        
        print("✅ Nikkei225 Screener test passed")
        return filtered_symbols
        
    except Exception as e:
        print(f"❌ Nikkei225 Screener test failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_data_manager(test_symbols):
    """データマネージャーテスト"""
    print("\n=== Test 2: DSSMS Data Manager ===")
    
    try:
        data_manager = DSSMSDataManager()
        
        # 単一銘柄テスト
        print("2.1 Single symbol test:")
        if test_symbols:
            symbol = test_symbols[0]
            data = data_manager.get_multi_timeframe_data(symbol)
            
            for timeframe, df in data.items():
                print(f"  {symbol} {timeframe}: {len(df)} records")
                if not df.empty:
                    print(f"    Latest: {df.index[-1].strftime('%Y-%m-%d')} Close: {df['Close'].iloc[-1]:.2f}")
        
        # バッチ取得テスト
        print("\n2.2 Batch fetch test:")
        batch_symbols = test_symbols[:2] if len(test_symbols) >= 2 else test_symbols
        batch_data = data_manager.batch_get_multi_timeframe_data(batch_symbols, max_workers=2)
        
        for symbol, timeframe_data in batch_data.items():
            print(f"  {symbol}:")
            for timeframe, df in timeframe_data.items():
                print(f"    {timeframe}: {len(df)} records")
        
        # キャッシュ統計
        print("\n2.3 Cache statistics:")
        cache_stats = data_manager.get_cache_stats()
        for timeframe, stats in cache_stats.items():
            print(f"  {timeframe}: {stats['active_entries']}/{stats['total_entries']} active")
        
        print("✅ Data Manager test passed")
        return batch_data
        
    except Exception as e:
        print(f"❌ Data Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_fundamental_analyzer(test_symbols):
    """業績分析器テスト"""
    print("\n=== Test 3: Fundamental Analyzer ===")
    
    try:
        analyzer = FundamentalAnalyzer()
        
        # 個別分析テスト
        print("3.1 Individual analysis:")
        for symbol in test_symbols[:3]:  # 最大3銘柄
            print(f"\n{symbol}:")
            print(f"  Operating profit positive: {analyzer.check_operating_profit_positive(symbol)}")
            print(f"  Consecutive growth: {analyzer.check_consecutive_growth(symbol)}")
            print(f"  Consensus beat: {analyzer.check_consensus_beat(symbol)}")
            print(f"  Fundamental score: {analyzer.calculate_fundamental_score(symbol):.3f}")
        
        # バッチ分析テスト
        print("\n3.2 Batch analysis:")
        batch_results = analyzer.batch_analyze_fundamentals(test_symbols[:3])
        
        for symbol, result in batch_results.items():
            print(f"  {symbol}: Score {result['fundamental_score']:.3f}")
        
        # サマリー
        print("\n3.3 Analysis summary:")
        summary = analyzer.get_analysis_summary(test_symbols[:3])
        
        print(f"  Total symbols: {summary['total_symbols']}")
        print(f"  Operating profit positive: {summary['operating_profit_positive']}")
        print(f"  Consecutive growth: {summary['consecutive_growth']}")
        print(f"  Consensus beat: {summary['consensus_beat']}")
        print(f"  Average score: {summary['avg_fundamental_score']:.3f}")
        
        print("✅ Fundamental Analyzer test passed")
        return batch_results
        
    except Exception as e:
        print(f"❌ Fundamental Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_perfect_order_detector(test_symbols, batch_data):
    """パーフェクトオーダー検出器テスト"""
    print("\n=== Test 4: Perfect Order Detector ===")
    
    try:
        detector = PerfectOrderDetector()
        
        # 単一銘柄テスト
        print("4.1 Single symbol test:")
        if test_symbols and batch_data:
            symbol = test_symbols[0]
            if symbol in batch_data:
                data_dict = batch_data[symbol]
                
                result = detector.check_multi_timeframe_perfect_order(symbol, data_dict)
                
                print(f"\nMulti-timeframe Perfect Order Analysis for {symbol}:")
                print(f"  Priority Level: {result.priority_level}")
                print(f"  Composite Score: {result.composite_score:.3f}")
                print(f"  Daily Perfect Order: {result.daily_result.is_perfect_order}")
                print(f"  Weekly Perfect Order: {result.weekly_result.is_perfect_order}")
                print(f"  Monthly Perfect Order: {result.monthly_result.is_perfect_order}")
                
                print(f"\nDaily Analysis:")
                print(f"  Current Price: {result.daily_result.current_price:.2f}")
                print(f"  SMA(5): {result.daily_result.sma_short:.2f}")
                print(f"  SMA(25): {result.daily_result.sma_medium:.2f}")
                print(f"  SMA(75): {result.daily_result.sma_long:.2f}")
                print(f"  Strength Score: {result.daily_result.strength_score:.3f}")
                print(f"  Trend Duration: {result.daily_result.trend_duration_days} days")
        
        # バッチ分析テスト
        print("\n4.2 Batch analysis test:")
        def data_provider(symbol):
            return batch_data.get(symbol, {})
        
        batch_symbols = [s for s in test_symbols[:3] if s in batch_data]
        if batch_symbols:
            results = detector.batch_analyze_symbols(batch_symbols, data_provider)
            
            print(f"Batch analysis results ({len(results)} symbols):")
            for result in results:
                print(f"  {result.symbol}: Priority {result.priority_level}, Score {result.composite_score:.3f}")
        
        # キャッシュ統計
        cache_stats = detector.get_cache_stats()
        print(f"\n4.3 Cache Statistics:")
        print(f"  Active entries: {cache_stats['active_entries']}/{cache_stats['total_entries']}")
        
        print("✅ Perfect Order Detector test passed")
        return True
        
    except Exception as e:
        print(f"❌ Perfect Order Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """統合テスト"""
    print("\n=== Test 5: Integration Test ===")
    
    try:
        print("5.1 Initializing all components...")
        screener = Nikkei225Screener()
        data_manager = DSSMSDataManager()
        analyzer = FundamentalAnalyzer()
        detector = PerfectOrderDetector()
        
        print("5.2 Running integrated screening workflow...")
        
        # 1. 銘柄スクリーニング
        available_funds = 1_000_000
        filtered_symbols = screener.get_filtered_symbols(available_funds)
        print(f"  Screened symbols: {len(filtered_symbols)}")
        
        if not filtered_symbols:
            print("  No symbols passed screening filters")
            return False
        
        # 2. データ取得（最大3銘柄）
        test_symbols = filtered_symbols[:3]
        batch_data = data_manager.batch_get_multi_timeframe_data(test_symbols, max_workers=2)
        print(f"  Data fetched for: {len(batch_data)} symbols")
        
        # 3. 業績分析
        fundamental_results = analyzer.batch_analyze_fundamentals(test_symbols)
        print(f"  Fundamental analysis completed for: {len(fundamental_results)} symbols")
        
        # 4. パーフェクトオーダー検出
        def data_provider(symbol):
            return batch_data.get(symbol, {})
        
        available_symbols = [s for s in test_symbols if s in batch_data]
        perfect_order_results = detector.batch_analyze_symbols(available_symbols, data_provider)
        print(f"  Perfect order analysis completed for: {len(perfect_order_results)} symbols")
        
        # 5. 統合結果
        print("\n5.3 Integration results:")
        for symbol in available_symbols:
            fundamental = fundamental_results.get(symbol, {})
            perfect_order = next((r for r in perfect_order_results if r.symbol == symbol), None)
            
            print(f"  {symbol}:")
            print(f"    Fundamental Score: {fundamental.get('fundamental_score', 0.0):.3f}")
            if perfect_order:
                print(f"    Perfect Order Priority: {perfect_order.priority_level}")
                print(f"    Perfect Order Score: {perfect_order.composite_score:.3f}")
        
        print("✅ Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🚀 DSSMS Phase 1 Task 1.3 統合テスト開始")
    print(f"テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 各コンポーネントのテスト
    test_symbols = test_nikkei225_screener()
    
    if test_symbols:
        batch_data = test_data_manager(test_symbols)
        fundamental_results = test_fundamental_analyzer(test_symbols)
        perfect_order_results = test_perfect_order_detector(test_symbols, batch_data)
        integration_success = test_integration()
    else:
        print("⚠️ スクリーナーテストが失敗したため、後続テストをスキップします")
        integration_success = False
    
    # 結果サマリー
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 テスト結果サマリー")
    print(f"実行時間: {elapsed_time:.2f}秒")
    print(f"テスト終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if integration_success:
        print("🎉 DSSMS Phase 1 Task 1.3 実装テスト: 成功")
    else:
        print("❌ DSSMS Phase 1 Task 1.3 実装テスト: 一部失敗")
    
    return integration_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
