#!/usr/bin/env python3
"""
ハイブリッドランキングシステム 基本機能確認テスト
"""

import sys
import os
import time
from pathlib import Path

# パス設定
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.dssms.hybrid_ranking_engine import HybridRankingEngine


def test_system_initialization():
    """システム初期化テスト"""
    print("=== システム初期化テスト ===")
    
    try:
        config_path = Path(__file__).parent / "config" / "dssms" / "hybrid_ranking_config.json"
        engine = HybridRankingEngine(str(config_path))
        print("✓ システム初期化成功")
        return True
    except Exception as e:
        print(f"✗ システム初期化失敗: {e}")
        return False


def test_configuration_loading():
    """設定ファイル読み込みテスト"""
    print("=== 設定ファイル読み込みテスト ===")
    
    try:
        config_path = Path(__file__).parent / "config" / "dssms" / "hybrid_ranking_config.json"
        
        if not config_path.exists():
            print(f"✗ 設定ファイルが存在しません: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = eval(f.read())  # json.loads代わり
            
        print(f"✓ 設定ファイル読み込み成功: {len(config)}セクション")
        
        # 主要セクション確認
        required_sections = ['ranking_engine', 'data_integration', 'adaptive_scoring', 'performance_optimization']
        for section in required_sections:
            if section in config:
                print(f"  ✓ {section}セクション確認")
            else:
                print(f"  ✗ {section}セクション不足")
        
        return True
        
    except Exception as e:
        print(f"✗ 設定ファイル読み込み失敗: {e}")
        return False


def test_component_availability():
    """コンポーネント可用性テスト"""
    print("=== コンポーネント可用性テスト ===")
    
    components = [
        ("HybridRankingEngine", "src.dssms.hybrid_ranking_engine"),
        ("RankingDataIntegrator", "src.dssms.ranking_data_integrator"),
        ("AdaptiveScoreCalculator", "src.dssms.adaptive_score_calculator"),
        ("RankingPerformanceOptimizer", "src.dssms.ranking_performance_optimizer")
    ]
    
    all_available = True
    
    for class_name, module_name in components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {class_name}: 利用可能")
        except ImportError as e:
            print(f"✗ {class_name}: インポートエラー - {e}")
            all_available = False
        except AttributeError as e:
            print(f"✗ {class_name}: クラス不足 - {e}")
            all_available = False
        except Exception as e:
            print(f"✗ {class_name}: その他のエラー - {e}")
            all_available = False
    
    return all_available


def test_basic_functionality():
    """基本機能テスト"""
    print("=== 基本機能テスト ===")
    
    try:
        config_path = Path(__file__).parent / "config" / "dssms" / "hybrid_ranking_config.json"
        engine = HybridRankingEngine(str(config_path))
        
        # システム状態確認
        status = engine.get_system_status()
        print(f"✓ システム状態取得成功: {len(status)}項目")
        
        # キャッシュクリア
        engine.clear_cache()
        print("✓ キャッシュクリア成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本機能テスト失敗: {e}")
        return False


def run_all_tests():
    """全テスト実行"""
    print("DSSMS Phase 2 Task 2.2: ハイブリッドランキングシステム 基本機能確認テスト")
    print("=" * 60)
    
    tests = [
        ("システム初期化", test_system_initialization),
        ("設定ファイル読み込み", test_configuration_loading),
        ("コンポーネント可用性", test_component_availability),
        ("基本機能", test_basic_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}テスト実行中...")
        start_time = time.time()
        
        try:
            result = test_func()
            execution_time = time.time() - start_time
            
            if result:
                print(f"✓ {test_name}テスト: 成功 ({execution_time:.3f}秒)")
                results.append((test_name, "成功", execution_time))
            else:
                print(f"✗ {test_name}テスト: 失敗 ({execution_time:.3f}秒)")
                results.append((test_name, "失敗", execution_time))
        
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"✗ {test_name}テスト: 例外発生 - {e} ({execution_time:.3f}秒)")
            results.append((test_name, f"例外: {e}", execution_time))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    success_count = 0
    total_time = 0
    
    for test_name, result, execution_time in results:
        status_icon = "✓" if result == "成功" else "✗"
        print(f"{status_icon} {test_name}: {result} ({execution_time:.3f}秒)")
        
        if result == "成功":
            success_count += 1
        total_time += execution_time
    
    print(f"\n成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"総実行時間: {total_time:.3f}秒")
    
    if success_count == len(results):
        print("\n🎉 全テスト成功！ハイブリッドランキングシステム実装完了")
    else:
        print(f"\n⚠️  {len(results) - success_count}個のテストが失敗しました")
    
    return success_count == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
