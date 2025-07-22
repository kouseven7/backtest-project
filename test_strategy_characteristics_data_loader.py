"""
Test Script: Strategy Characteristics Data Loader
File: test_strategy_characteristics_data_loader.py
Description: 
  strategy_characteristics_data_loader.pyの機能テスト
  ロード・キャッシュ・更新・検索機能の動作検証

Author: imega
Created: 2025-07-08
Modified: 2025-07-08
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import unittest

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config.strategy_characteristics_data_loader import (
        StrategyCharacteristicsDataLoader,
        LoadOptions,
        UpdateOptions,
        create_data_loader,
        create_load_options,
        create_update_options
    )
    from config.strategy_data_persistence import StrategyDataPersistence
    from config.strategy_characteristics_manager import StrategyCharacteristicsManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TestStrategyCharacteristicsDataLoader(unittest.TestCase):
    """データローダーのテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.test_base_path = os.path.join("logs", "test_strategy_data_loader")
        self.loader = StrategyCharacteristicsDataLoader(self.test_base_path, cache_size=50)
        
        # テスト用戦略データ作成
        self.test_strategies = ["test_vwap_bounce", "test_momentum", "test_breakout"]
        self._create_test_data()
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # テストディレクトリのクリーンアップは任意
        pass
    
    def _create_test_data(self):
        """テスト用データの作成"""
        try:
            for strategy in self.test_strategies:
                # 特性データを直接characteristics_managerに追加
                trend_data = {
                    "uptrend": {"suitability_score": 0.8, "confidence": 0.9},
                    "downtrend": {"suitability_score": 0.6, "confidence": 0.8},
                    "sideways": {"suitability_score": 0.7, "confidence": 0.85}
                }
                
                for trend, data in trend_data.items():
                    self.loader.characteristics_manager.save_trend_suitability(
                        strategy, trend, data
                    )
                
                # ボラティリティデータ追加
                vol_data = {
                    "high": {"suitability_score": 0.9, "confidence": 0.9},
                    "medium": {"suitability_score": 0.8, "confidence": 0.85},
                    "low": {"suitability_score": 0.5, "confidence": 0.7}
                }
                
                for vol_level, data in vol_data.items():
                    self.loader.characteristics_manager.save_volatility_suitability(
                        strategy, vol_level, data
                    )
                
                print(f"✓ テストデータ作成完了: {strategy}")
                
        except Exception as e:
            print(f"テストデータ作成エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def test_single_strategy_load(self):
        """単一戦略ロードテスト"""
        print("\n=== 単一戦略ロードテスト ===")
        
        strategy = self.test_strategies[0]
        options = LoadOptions(use_cache=True, include_history=True, include_parameters=True)
        
        # 初回ロード（キャッシュミス）
        data = self.loader.load_strategy_characteristics(strategy, options)
        
        self.assertIsNotNone(data, "戦略データが取得できませんでした")
        self.assertIn("metadata", data, "メタデータが含まれていません")
        self.assertIn("performance", data, "パフォーマンスデータが含まれていません")
        self.assertIn("parameters", data, "パラメータデータが含まれていません")
        
        print(f"✓ 戦略 '{strategy}' のデータロード成功")
        print(f"  - データセクション: {list(data.keys())}")
        
        # 2回目ロード（キャッシュヒット）
        data2 = self.loader.load_strategy_characteristics(strategy, options)
        self.assertIsNotNone(data2, "キャッシュからのデータ取得に失敗")
        
        print(f"✓ キャッシュからのロード成功")
    
    def test_batch_load(self):
        """バッチロードテスト"""
        print("\n=== バッチロードテスト ===")
        
        options = LoadOptions(use_cache=True, include_parameters=True)
        batch_data = self.loader.load_multiple_strategies(self.test_strategies, options)
        
        self.assertIsInstance(batch_data, dict, "バッチロード結果が辞書ではありません")
        
        loaded_count = len([v for v in batch_data.values() if v is not None])
        print(f"✓ バッチロード完了: {loaded_count}/{len(self.test_strategies)} 戦略")
        
        for strategy, data in batch_data.items():
            if data:
                print(f"  - {strategy}: OK")
            else:
                print(f"  - {strategy}: データなし")
    
    def test_cache_functionality(self):
        """キャッシュ機能テスト"""
        print("\n=== キャッシュ機能テスト ===")
        
        strategy = self.test_strategies[0]
        options = LoadOptions(use_cache=True, cache_ttl_seconds=60)
        
        # 初回ロード
        start_time = datetime.now()
        data1 = self.loader.load_strategy_characteristics(strategy, options)
        first_load_time = (datetime.now() - start_time).total_seconds()
        
        # キャッシュからロード
        start_time = datetime.now()
        data2 = self.loader.load_strategy_characteristics(strategy, options)
        cache_load_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ 初回ロード時間: {first_load_time:.4f}秒")
        print(f"✓ キャッシュロード時間: {cache_load_time:.4f}秒")
        print(f"✓ 速度向上: {first_load_time/cache_load_time:.1f}倍")
        
        # キャッシュ統計確認
        stats = self.loader.get_cache_stats()
        print(f"✓ キャッシュ統計: {stats}")
        
        self.assertGreater(stats['cache_hits'], 0, "キャッシュヒットがありません")
    
    def test_data_update(self):
        """データ更新テスト"""
        print("\n=== データ更新テスト ===")
        
        strategy = self.test_strategies[0]
        
        # 元データ取得
        original_data = self.loader.load_strategy_characteristics(
            strategy, LoadOptions(use_cache=False)
        )
        
        self.assertIsNotNone(original_data, "元データが取得できませんでした")
        
        # データ更新
        update_data = {
            "metadata": {
                **original_data["metadata"],
                "last_updated": datetime.now().isoformat(),
                "test_flag": True
            }
        }
        
        options = UpdateOptions(create_backup=True, validate_before_update=True)
        success = self.loader.update_strategy_characteristics(strategy, update_data, options)
        
        self.assertTrue(success, "データ更新に失敗しました")
        print(f"✓ 戦略 '{strategy}' のデータ更新成功")
        
        # 更新後データ確認
        updated_data = self.loader.load_strategy_characteristics(
            strategy, LoadOptions(use_cache=False)
        )
        
        self.assertIn("test_flag", updated_data["metadata"], "更新データが反映されていません")
        print(f"✓ 更新内容の確認完了")
    
    def test_search_functionality(self):
        """検索機能テスト"""
        print("\n=== 検索機能テスト ===")
        
        # 高パフォーマンス戦略検索
        high_performance_strategies = self.loader.search_strategies({
            "min_sharpe_ratio": 1.0,
            "min_win_rate": 0.5
        })
        
        print(f"✓ 高パフォーマンス戦略: {len(high_performance_strategies)} 個")
        for strategy in high_performance_strategies:
            print(f"  - {strategy}")
        
        # トレンド適応性検索
        trend_strategies = self.loader.search_strategies({
            "trend_environment": "uptrend",
            "min_adaptability": 0.7
        })
        
        print(f"✓ アップトレンド適応戦略: {len(trend_strategies)} 個")
        for strategy in trend_strategies:
            print(f"  - {strategy}")
    
    def test_cache_cleanup(self):
        """キャッシュクリーンアップテスト"""
        print("\n=== キャッシュクリーンアップテスト ===")
        
        # キャッシュデータ作成
        for strategy in self.test_strategies:
            self.loader.load_strategy_characteristics(
                strategy, LoadOptions(use_cache=True)
            )
        
        initial_stats = self.loader.get_cache_stats()
        print(f"✓ クリーンアップ前: {initial_stats['cache_entries']} エントリー")
        
        # クリーンアップ実行
        self.loader.cleanup_cache(force=True)
        
        final_stats = self.loader.get_cache_stats()
        print(f"✓ クリーンアップ後: {final_stats['cache_entries']} エントリー")
        
        self.assertEqual(final_stats['cache_entries'], 0, "キャッシュがクリアされていません")


def run_comprehensive_test():
    """包括的テストの実行"""
    print("=" * 60)
    print("戦略特性データローダー 包括的テスト")
    print("=" * 60)
    
    # テストスイート作成
    suite = unittest.TestSuite()
    
    # テストケース追加
    test_cases = [
        'test_single_strategy_load',
        'test_batch_load', 
        'test_cache_functionality',
        'test_data_update',
        'test_search_functionality',
        'test_cache_cleanup'
    ]
    
    for test_case in test_cases:
        suite.addTest(TestStrategyCharacteristicsDataLoader(test_case))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    
    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


def run_quick_demo():
    """簡単なデモ実行"""
    print("=" * 60)
    print("戦略特性データローダー 簡単デモ")
    print("=" * 60)
    
    try:
        # データローダー作成
        loader = create_data_loader()
        print("✓ データローダー初期化完了")
        
        # 利用可能な戦略を確認
        strategies = loader.persistence_manager.list_strategies()
        print(f"✓ 利用可能な戦略: {len(strategies)} 個")
        
        if strategies:
            # 最初の戦略をテスト
            test_strategy = strategies[0]
            options = create_load_options(include_history=True, include_parameters=True)
            
            data = loader.load_strategy_characteristics(test_strategy, options)
            
            if data:
                print(f"✓ 戦略 '{test_strategy}' ロード成功")
                print(f"  - データセクション: {list(data.keys())}")
                
                if 'metadata' in data:
                    metadata = data['metadata']
                    print(f"  - 戦略名: {metadata.get('strategy_name', 'N/A')}")
                    print(f"  - 最終更新: {metadata.get('last_updated', 'N/A')}")
                
                if 'performance' in data and 'summary' in data['performance']:
                    summary = data['performance']['summary']
                    print(f"  - 総リターン: {summary.get('total_return', 'N/A')}")
                    print(f"  - シャープレシオ: {summary.get('sharpe_ratio', 'N/A')}")
            else:
                print(f"✗ 戦略 '{test_strategy}' ロード失敗")
        
        # キャッシュ統計表示
        stats = loader.get_cache_stats()
        print(f"\n✓ キャッシュ統計:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        print("\n✓ デモ完了")
        return True
        
    except Exception as e:
        print(f"✗ デモ実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """メイン実行部"""
    import argparse
    
    parser = argparse.ArgumentParser(description="戦略特性データローダーテスト")
    parser.add_argument("--demo", action="store_true", help="簡単デモを実行")
    parser.add_argument("--test", action="store_true", help="包括的テストを実行")
    
    args = parser.parse_args()
    
    if args.demo:
        success = run_quick_demo()
    elif args.test:
        success = run_comprehensive_test()
    else:
        # デフォルトはデモ実行
        print("オプション未指定: 簡単デモを実行します")
        print("包括的テストを実行するには --test オプションを使用してください")
        success = run_quick_demo()
    
    sys.exit(0 if success else 1)
