"""
Stage 3統合テスト: SmartCache・OptimizedAlgorithmEngine検証
独立テストでStage 3統合効果を直接測定
"""

import time
import logging
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.logger_config import setup_logger
from src.dssms.nikkei225_screener import Nikkei225Screener

def test_stage3_integration():
    """Stage 3統合効果テスト"""
    
    logger = setup_logger("stage3_test", "logs/stage3_integration_test.log")
    
    print("🚀 Stage 3統合テスト開始")
    print("=" * 60)
    
    try:
        # Nikkei225Screenerインスタンス作成
        print("📊 Nikkei225Screener初期化中...")
        screener = Nikkei225Screener()
        print("✅ 初期化完了")
        
        # テスト設定
        available_funds = 1000000  # 100万円
        
        # Stage 3-1: SmartCache統合テスト
        print("\n🔄 Stage 3-1: SmartCache統合テスト")
        print("-" * 40)
        
        start_time = time.time()
        
        # 1回目実行（キャッシュなし）
        print("💾 1回目実行 (キャッシュなし)...")
        symbols_1st = screener.screen_symbols(available_funds)
        first_run_time = time.time() - start_time
        
        print(f"⏱️  1回目実行時間: {first_run_time:.2f}秒")
        print(f"📈 選択銘柄数: {len(symbols_1st)}")
        
        # 2回目実行（キャッシュあり）
        print("\n💾 2回目実行 (キャッシュあり)...")
        start_time = time.time()
        symbols_2nd = screener.screen_symbols(available_funds)
        second_run_time = time.time() - start_time
        
        print(f"⏱️  2回目実行時間: {second_run_time:.2f}秒")
        print(f"📈 選択銘柄数: {len(symbols_2nd)}")
        
        # キャッシュ効果測定
        if first_run_time > 0:
            cache_speedup = ((first_run_time - second_run_time) / first_run_time) * 100
            print(f"🚀 キャッシュ高速化効果: {cache_speedup:.1f}%")
        
        # Stage 3-2: OptimizedAlgorithmEngine統合テスト
        print("\n⚡ Stage 3-2: OptimizedAlgorithmEngine統合テスト")
        print("-" * 40)
        
        # アルゴリズム最適化統計取得
        if hasattr(screener, 'algorithm_optimizer'):
            stats = screener.algorithm_optimizer.get_optimization_stats()
            print(f"🔢 NumPy操作回数: {stats['numpy_operations']}")
            print(f"📊 ベクトル化計算数: {stats['vectorized_calculations']}")
            print(f"⏩ 早期終了回数: {stats['early_terminations']}")
            print(f"⏱️  節約処理時間: {stats['processing_time_saved']:.2f}秒")
        
        # キャッシュ統計取得
        if hasattr(screener, 'cached_fetcher'):
            cache_stats = screener.cached_fetcher.cache.get_cache_stats()
            print(f"\n💾 キャッシュ統計:")
            print(f"   ヒット数: {cache_stats.get('hits', 0)}")
            print(f"   ミス数: {cache_stats.get('misses', 0)}")
            print(f"   ヒット率: {cache_stats.get('hit_rate', 0):.1f}%")
            print(f"   キャッシュサイズ: {cache_stats.get('cache_size', 0)}")
        
        # パフォーマンス評価
        print("\n📊 Stage 3統合パフォーマンス評価")
        print("-" * 40)
        
        total_time_saved = 0
        if hasattr(screener, 'algorithm_optimizer'):
            total_time_saved = screener.algorithm_optimizer.get_optimization_stats()['processing_time_saved']
        
        cache_time_saved = max(0, first_run_time - second_run_time)
        total_optimization = total_time_saved + cache_time_saved
        
        print(f"⚡ アルゴリズム最適化節約: {total_time_saved:.2f}秒")
        print(f"💾 キャッシュ最適化節約: {cache_time_saved:.2f}秒")
        print(f"🚀 総最適化効果: {total_optimization:.2f}秒")
        
        baseline_time = first_run_time
        if baseline_time > 0:
            total_reduction_percent = (total_optimization / baseline_time) * 100
            print(f"📈 総削減率: {total_reduction_percent:.1f}%")
            
            # Stage 3目標達成判定
            if total_reduction_percent >= 85:
                print("🎉 Stage 3目標達成! (85%以上削減)")
            elif total_reduction_percent >= 70:
                print("✅ Stage 3部分達成 (70%以上削減)")
            else:
                print("⚠️  Stage 3目標未達成 (85%削減未満)")
        
        print(f"\n🔍 選択銘柄サンプル:")
        for i, symbol in enumerate(symbols_1st[:5]):
            print(f"   {i+1}. {symbol}")
            
        return {
            'first_run_time': first_run_time,
            'second_run_time': second_run_time,
            'cache_speedup': cache_speedup if first_run_time > 0 else 0,
            'symbols_count': len(symbols_1st),
            'total_optimization': total_optimization,
            'total_reduction_percent': total_reduction_percent if baseline_time > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Stage 3統合テスト失敗: {e}")
        print(f"❌ テスト失敗: {e}")
        return None

def main():
    """メイン実行関数"""
    
    results = test_stage3_integration()
    
    if results:
        print("\n" + "=" * 60)
        print("📊 Stage 3統合テスト結果サマリー")
        print("=" * 60)
        print(f"1回目実行時間: {results['first_run_time']:.2f}秒")
        print(f"2回目実行時間: {results['second_run_time']:.2f}秒")
        print(f"キャッシュ高速化: {results['cache_speedup']:.1f}%")
        print(f"選択銘柄数: {results['symbols_count']}")
        print(f"総最適化効果: {results['total_optimization']:.2f}秒")
        print(f"総削減率: {results['total_reduction_percent']:.1f}%")
        
        if results['total_reduction_percent'] >= 85:
            print("🏆 Stage 3統合: 目標達成!")
        else:
            print("⚠️  Stage 3統合: さらなる最適化が必要")
    else:
        print("❌ Stage 3統合テスト失敗")

if __name__ == "__main__":
    main()