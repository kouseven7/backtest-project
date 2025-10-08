"""
Stage 3統合 基本動作テスト - 簡潔版
SmartCache・OptimizedAlgorithmEngine統合の基本動作確認
"""

import time
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_cache_integration():
    """キャッシュ統合テスト"""
    print("🔄 Stage 3-1: SmartCache統合テスト")
    print("-" * 40)
    
    try:
        from src.dssms.screener_cache_integration import create_screener_cache_integration
        
        # キャッシュ統合作成
        cached_fetcher = create_screener_cache_integration()
        print("[OK] SmartCache統合作成成功")
        
        # テストデータ取得
        test_symbol = "7203"  # トヨタ
        
        print(f"[CHART] テスト銘柄: {test_symbol}")
        
        # 1回目（キャッシュなし）
        start_time = time.time()
        market_cap_1 = cached_fetcher.get_market_cap_data_cached(test_symbol)
        time_1 = time.time() - start_time
        
        print(f"⏱️  1回目実行時間: {time_1:.2f}秒")
        print(f"[MONEY] 時価総額データ: {market_cap_1 is not None}")
        
        # 2回目（キャッシュあり）
        start_time = time.time()
        market_cap_2 = cached_fetcher.get_market_cap_data_cached(test_symbol)
        time_2 = time.time() - start_time
        
        print(f"⏱️  2回目実行時間: {time_2:.2f}秒")
        print(f"[MONEY] 時価総額データ: {market_cap_2 is not None}")
        
        # キャッシュ効果
        if time_1 > 0 and time_2 < time_1:
            speedup = ((time_1 - time_2) / time_1) * 100
            print(f"[ROCKET] キャッシュ高速化: {speedup:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] キャッシュ統合テスト失敗: {e}")
        return False

def test_algorithm_integration():
    """アルゴリズム最適化統合テスト"""
    print("\n⚡ Stage 3-2: OptimizedAlgorithmEngine統合テスト")
    print("-" * 40)
    
    try:
        from src.dssms.algorithm_optimization_integration import create_algorithm_optimization_integration
        
        # アルゴリズム最適化作成
        optimizer = create_algorithm_optimization_integration()
        print("[OK] OptimizedAlgorithmEngine統合作成成功")
        
        # 統計取得
        stats = optimizer.get_optimization_stats()
        print(f"[CHART] 初期統計:")
        print(f"   NumPy操作: {stats['numpy_operations']}")
        print(f"   ベクトル化計算: {stats['vectorized_calculations']}")
        print(f"   処理時間節約: {stats['processing_time_saved']:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] アルゴリズム統合テスト失敗: {e}")
        return False

def main():
    """メイン実行"""
    print("[ROCKET] Stage 3統合 基本動作テスト開始")
    print("=" * 60)
    
    # キャッシュ統合テスト
    cache_success = test_cache_integration()
    
    # アルゴリズム統合テスト
    algorithm_success = test_algorithm_integration()
    
    # 結果まとめ
    print("\n" + "=" * 60)
    print("[CHART] Stage 3統合テスト結果")
    print("=" * 60)
    print(f"SmartCache統合: {'[OK] 成功' if cache_success else '[ERROR] 失敗'}")
    print(f"AlgorithmEngine統合: {'[OK] 成功' if algorithm_success else '[ERROR] 失敗'}")
    
    overall_success = cache_success and algorithm_success
    print(f"\n🏆 総合結果: {'[OK] Stage 3統合成功' if overall_success else '[ERROR] Stage 3統合失敗'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)