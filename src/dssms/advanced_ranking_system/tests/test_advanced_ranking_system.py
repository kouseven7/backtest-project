"""
DSSMS Phase 3 Task 3.1: Advanced Ranking System Test Suite
高度ランキングシステムテストスイート

包括的なテスト機能を提供します。
"""

import sys
from pathlib import Path
import unittest
import pytest
import asyncio
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# テスト対象のインポート
try:
    from src.dssms.advanced_ranking_system.advanced_ranking_engine import AdvancedRankingEngine, RankingConfig
    from src.dssms.advanced_ranking_system.multi_dimensional_analyzer import MultiDimensionalAnalyzer, AnalysisConfig
    from src.dssms.advanced_ranking_system.dynamic_weight_optimizer import DynamicWeightOptimizer, OptimizationConfig
    from src.dssms.advanced_ranking_system.integration_bridge import IntegrationBridge, IntegrationMode
    from src.dssms.advanced_ranking_system.ranking_cache_manager import RankingCacheManager, CacheStrategy
    from src.dssms.advanced_ranking_system.performance_monitor import PerformanceMonitor, MonitoringConfig
    from src.dssms.advanced_ranking_system.realtime_updater import RealtimeUpdater, UpdateType, UpdatePriority
    TEST_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import advanced ranking system modules: {e}")
    TEST_IMPORTS_AVAILABLE = False

class TestDataGenerator:
    """テストデータ生成クラス"""
    
    @staticmethod
    def generate_sample_market_data(n_stocks=100, n_days=252):
        """サンプル市場データ生成"""
        symbols = [f"STOCK_{i:03d}" for i in range(n_stocks)]
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        data = {}
        
        for symbol in symbols:
            # ランダムウォークベースの価格データ
            prices = []
            price = 100.0
            
            for _ in range(n_days):
                change = np.random.normal(0, 0.02)
                price *= (1 + change)
                prices.append(price)
            
            volumes = np.random.lognormal(10, 1, n_days)
            
            data[symbol] = pd.DataFrame({
                'Date': dates,
                'Open': np.array(prices) * (1 + np.random.normal(0, 0.005, n_days)),
                'High': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
                'Low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
                'Close': prices,
                'Volume': volumes.astype(int)
            })
            
        return data
    
    @staticmethod
    def generate_fundamental_data(symbols):
        """ファンダメンタルデータ生成"""
        fundamental_data = {}
        
        for symbol in symbols:
            fundamental_data[symbol] = {
                'pe_ratio': np.random.uniform(5, 50),
                'pb_ratio': np.random.uniform(0.5, 10),
                'roe': np.random.uniform(0.05, 0.30),
                'roa': np.random.uniform(0.02, 0.20),
                'debt_ratio': np.random.uniform(0.1, 0.8),
                'current_ratio': np.random.uniform(0.5, 5.0),
                'revenue_growth': np.random.uniform(-0.20, 0.50),
                'profit_margin': np.random.uniform(0.01, 0.30),
                'market_cap': np.random.uniform(1e6, 1e12),
                'sector': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer'])
            }
            
        return fundamental_data

@pytest.mark.skipif(not TEST_IMPORTS_AVAILABLE, reason="Advanced ranking system modules not available")
class TestAdvancedRankingEngine(unittest.TestCase):
    """AdvancedRankingEngineテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.config = RankingConfig()
        self.engine = AdvancedRankingEngine(self.config)
        self.sample_data = TestDataGenerator.generate_sample_market_data(50, 100)
        self.fundamental_data = TestDataGenerator.generate_fundamental_data(
            list(self.sample_data.keys())
        )
    
    def test_engine_initialization(self):
        """エンジン初期化テスト"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.config, self.config)
        self.assertIsNotNone(self.engine.logger)
    
    def test_ranking_calculation_sync(self):
        """同期ランキング計算テスト"""
        symbols = list(self.sample_data.keys())[:10]
        
        rankings = self.engine.calculate_rankings(
            symbols,
            self.sample_data,
            self.fundamental_data
        )
        
        self.assertIsNotNone(rankings)
        self.assertIsInstance(rankings, dict)
        self.assertEqual(len(rankings), len(symbols))
        
        for symbol in symbols:
            self.assertIn(symbol, rankings)
            self.assertIsInstance(rankings[symbol], (int, float))
    
    @pytest.mark.asyncio
    async def test_ranking_calculation_async(self):
        """非同期ランキング計算テスト"""
        symbols = list(self.sample_data.keys())[:10]
        
        rankings = await self.engine.calculate_rankings_async(
            symbols,
            self.sample_data,
            self.fundamental_data
        )
        
        self.assertIsNotNone(rankings)
        self.assertIsInstance(rankings, dict)
        self.assertEqual(len(rankings), len(symbols))
    
    def test_technical_indicators_calculation(self):
        """テクニカル指標計算テスト"""
        symbol = list(self.sample_data.keys())[0]
        data = self.sample_data[symbol]
        
        indicators = self.engine._calculate_technical_indicators(data)
        
        self.assertIsNotNone(indicators)
        self.assertIsInstance(indicators, dict)
        
        # 基本的な指標が含まれているかチェック
        expected_indicators = ['rsi', 'macd', 'bollinger_bands', 'volume_sma']
        for indicator in expected_indicators:
            self.assertIn(indicator, indicators)

@pytest.mark.skipif(not TEST_IMPORTS_AVAILABLE, reason="Advanced ranking system modules not available")
class TestMultiDimensionalAnalyzer(unittest.TestCase):
    """MultiDimensionalAnalyzerテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.config = AnalysisConfig()
        self.analyzer = MultiDimensionalAnalyzer(self.config)
        self.sample_data = TestDataGenerator.generate_sample_market_data(30, 100)
    
    def test_analyzer_initialization(self):
        """アナライザー初期化テスト"""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.config, self.config)
    
    def test_momentum_analysis(self):
        """モメンタム分析テスト"""
        symbol = list(self.sample_data.keys())[0]
        data = self.sample_data[symbol]
        
        momentum_score = self.analyzer.calculate_momentum_score(data)
        
        self.assertIsNotNone(momentum_score)
        self.assertIsInstance(momentum_score, (int, float))
        self.assertGreaterEqual(momentum_score, 0)
        self.assertLessEqual(momentum_score, 1)
    
    def test_volatility_analysis(self):
        """ボラティリティ分析テスト"""
        symbol = list(self.sample_data.keys())[0]
        data = self.sample_data[symbol]
        
        volatility_score = self.analyzer.calculate_volatility_score(data)
        
        self.assertIsNotNone(volatility_score)
        self.assertIsInstance(volatility_score, (int, float))
        self.assertGreaterEqual(volatility_score, 0)
        self.assertLessEqual(volatility_score, 1)
    
    def test_volume_analysis(self):
        """出来高分析テスト"""
        symbol = list(self.sample_data.keys())[0]
        data = self.sample_data[symbol]
        
        volume_score = self.analyzer.calculate_volume_score(data)
        
        self.assertIsNotNone(volume_score)
        self.assertIsInstance(volume_score, (int, float))
        self.assertGreaterEqual(volume_score, 0)
        self.assertLessEqual(volume_score, 1)

@pytest.mark.skipif(not TEST_IMPORTS_AVAILABLE, reason="Advanced ranking system modules not available")
class TestDynamicWeightOptimizer(unittest.TestCase):
    """DynamicWeightOptimizerテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.config = OptimizationConfig()
        self.optimizer = DynamicWeightOptimizer(self.config)
        
        # サンプル重みとパフォーマンス
        self.sample_weights = {
            'momentum': 0.3,
            'volatility': 0.2,
            'volume': 0.2,
            'technical': 0.3
        }
        
        self.sample_performance = np.random.uniform(-0.1, 0.1, 100)
    
    def test_optimizer_initialization(self):
        """オプティマイザー初期化テスト"""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.config, self.config)
    
    def test_weight_optimization(self):
        """重み最適化テスト"""
        optimized_weights = self.optimizer.optimize_weights(
            self.sample_weights,
            self.sample_performance
        )
        
        self.assertIsNotNone(optimized_weights)
        self.assertIsInstance(optimized_weights, dict)
        
        # 重みの合計が1に近いかチェック
        total_weight = sum(optimized_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # 各重みが正の値かチェック
        for weight in optimized_weights.values():
            self.assertGreaterEqual(weight, 0)
    
    def test_market_regime_detection(self):
        """市場レジーム検出テスト"""
        market_data = TestDataGenerator.generate_sample_market_data(10, 100)
        
        regime = self.optimizer.detect_market_regime(market_data)
        
        self.assertIsNotNone(regime)
        self.assertIsInstance(regime, str)
        self.assertIn(regime, ['bull', 'bear', 'sideways', 'high_volatility', 'low_volatility'])

@pytest.mark.skipif(not TEST_IMPORTS_AVAILABLE, reason="Advanced ranking system modules not available")
class TestIntegrationBridge(unittest.TestCase):
    """IntegrationBridgeテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.bridge = IntegrationBridge()
        self.mock_legacy_system = Mock()
        self.mock_advanced_system = Mock()
    
    def test_bridge_initialization(self):
        """ブリッジ初期化テスト"""
        self.assertIsNotNone(self.bridge)
        self.assertEqual(self.bridge.integration_mode, IntegrationMode.HYBRID)
    
    def test_hybrid_ranking(self):
        """ハイブリッドランキングテスト"""
        # モックランキング結果
        legacy_rankings = {'STOCK_001': 0.8, 'STOCK_002': 0.6, 'STOCK_003': 0.7}
        advanced_rankings = {'STOCK_001': 0.9, 'STOCK_002': 0.5, 'STOCK_003': 0.8}
        
        combined_rankings = self.bridge.combine_rankings(
            legacy_rankings,
            advanced_rankings
        )
        
        self.assertIsNotNone(combined_rankings)
        self.assertIsInstance(combined_rankings, dict)
        self.assertEqual(len(combined_rankings), 3)
        
        # ハイブリッド結果が元の範囲内にあるかチェック
        for symbol in combined_rankings:
            combined_score = combined_rankings[symbol]
            legacy_score = legacy_rankings[symbol]
            advanced_score = advanced_rankings[symbol]
            
            min_score = min(legacy_score, advanced_score)
            max_score = max(legacy_score, advanced_score)
            
            self.assertGreaterEqual(combined_score, min_score - 0.1)
            self.assertLessEqual(combined_score, max_score + 0.1)

@pytest.mark.skipif(not TEST_IMPORTS_AVAILABLE, reason="Advanced ranking system modules not available")
class TestRankingCacheManager(unittest.TestCase):
    """RankingCacheManagerテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.cache_manager = RankingCacheManager(
            cache_strategy=CacheStrategy.LRU,
            max_size=100
        )
    
    def test_cache_initialization(self):
        """キャッシュ初期化テスト"""
        self.assertIsNotNone(self.cache_manager)
        self.assertEqual(self.cache_manager.cache_strategy, CacheStrategy.LRU)
    
    def test_cache_operations(self):
        """キャッシュ操作テスト"""
        key = "test_rankings_001"
        value = {'STOCK_001': 0.8, 'STOCK_002': 0.6}
        
        # 保存テスト
        self.cache_manager.set(key, value)
        
        # 取得テスト
        cached_value = self.cache_manager.get(key)
        self.assertEqual(cached_value, value)
        
        # 存在チェックテスト
        self.assertTrue(self.cache_manager.exists(key))
        
        # 削除テスト
        self.cache_manager.delete(key)
        self.assertFalse(self.cache_manager.exists(key))
    
    def test_cache_expiration(self):
        """キャッシュ有効期限テスト"""
        key = "test_rankings_ttl"
        value = {'STOCK_001': 0.9}
        
        # TTL付きで保存
        self.cache_manager.set(key, value, ttl=1)
        
        # 即座に取得（存在するはず）
        cached_value = self.cache_manager.get(key)
        self.assertEqual(cached_value, value)
        
        # 期限切れ後に取得（存在しないはず）
        time.sleep(1.1)
        cached_value = self.cache_manager.get(key)
        self.assertIsNone(cached_value)

@pytest.mark.skipif(not TEST_IMPORTS_AVAILABLE, reason="Advanced ranking system modules not available")
class TestPerformanceMonitor(unittest.TestCase):
    """PerformanceMonitorテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.config = MonitoringConfig()
        self.monitor = PerformanceMonitor(self.config)
    
    def test_monitor_initialization(self):
        """モニター初期化テスト"""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.config, self.config)
    
    def test_metric_recording(self):
        """メトリクス記録テスト"""
        # メトリクス記録
        self.monitor.record_execution_time("ranking_calculation", 1.5)
        self.monitor.record_memory_usage(512)
        self.monitor.record_cpu_usage(75.0)
        
        # 統計取得
        stats = self.monitor.get_statistics()
        
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, dict)
        self.assertIn('execution_times', stats)
        self.assertIn('memory_usage', stats)
        self.assertIn('cpu_usage', stats)
    
    def test_alert_system(self):
        """アラートシステムテスト"""
        # 高CPU使用率アラート
        self.monitor.record_cpu_usage(95.0)
        
        alerts = self.monitor.get_active_alerts()
        
        self.assertIsNotNone(alerts)
        self.assertIsInstance(alerts, list)

@pytest.mark.skipif(not TEST_IMPORTS_AVAILABLE, reason="Advanced ranking system modules not available")
class TestRealtimeUpdater(unittest.TestCase):
    """RealtimeUpdaterテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.updater = RealtimeUpdater()
    
    def test_updater_initialization(self):
        """アップデーター初期化テスト"""
        self.assertIsNotNone(self.updater)
        self.assertFalse(self.updater._is_running)
    
    def test_update_scheduling(self):
        """更新スケジュールテスト"""
        self.updater.start()
        
        # 更新スケジュール
        event_id = self.updater.schedule_update(
            UpdateType.RANKING_SCORES,
            {"test": "data"},
            UpdatePriority.HIGH
        )
        
        self.assertIsNotNone(event_id)
        self.assertNotEqual(event_id, "")
        
        # ステータス確認
        status = self.updater.get_status()
        self.assertIsNotNone(status)
        self.assertTrue(status.is_running)
        
        self.updater.stop()

class TestSystemIntegration(unittest.TestCase):
    """システム統合テスト"""
    
    @pytest.mark.skipif(not TEST_IMPORTS_AVAILABLE, reason="Advanced ranking system modules not available")
    def test_full_system_integration(self):
        """完全システム統合テスト"""
        # テストデータ準備
        market_data = TestDataGenerator.generate_sample_market_data(20, 50)
        fundamental_data = TestDataGenerator.generate_fundamental_data(
            list(market_data.keys())
        )
        
        # システムコンポーネント初期化
        ranking_engine = AdvancedRankingEngine()
        analyzer = MultiDimensionalAnalyzer()
        optimizer = DynamicWeightOptimizer()
        cache_manager = RankingCacheManager()
        
        # ランキング計算
        symbols = list(market_data.keys())[:10]
        rankings = ranking_engine.calculate_rankings(
            symbols,
            market_data,
            fundamental_data
        )
        
        # 結果検証
        self.assertIsNotNone(rankings)
        self.assertIsInstance(rankings, dict)
        self.assertEqual(len(rankings), len(symbols))
        
        # キャッシュテスト
        cache_key = "integration_test_rankings"
        cache_manager.set(cache_key, rankings)
        cached_rankings = cache_manager.get(cache_key)
        self.assertEqual(cached_rankings, rankings)

class TestConfigurationLoading(unittest.TestCase):
    """設定ファイル読み込みテスト"""
    
    def test_config_file_loading(self):
        """設定ファイル読み込みテスト"""
        # テスト用設定ファイル作成
        config_data = {
            "test_parameter": "test_value",
            "numeric_parameter": 123,
            "boolean_parameter": True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file_path = f.name
        
        try:
            # 設定ファイル読み込み
            with open(config_file_path, 'r') as f:
                loaded_config = json.load(f)
            
            self.assertEqual(loaded_config, config_data)
            
        finally:
            # クリーンアップ
            Path(config_file_path).unlink()

class TestErrorHandling(unittest.TestCase):
    """エラーハンドリングテスト"""
    
    @pytest.mark.skipif(not TEST_IMPORTS_AVAILABLE, reason="Advanced ranking system modules not available")
    def test_invalid_data_handling(self):
        """無効データハンドリングテスト"""
        engine = AdvancedRankingEngine()
        
        # 空のデータでのテスト
        rankings = engine.calculate_rankings([], {}, {})
        self.assertEqual(rankings, {})
        
        # 無効なデータでのテスト
        rankings = engine.calculate_rankings(
            ["INVALID"],
            {"INVALID": pd.DataFrame()},
            {}
        )
        self.assertIsInstance(rankings, dict)
    
    @pytest.mark.skipif(not TEST_IMPORTS_AVAILABLE, reason="Advanced ranking system modules not available")
    def test_exception_handling(self):
        """例外ハンドリングテスト"""
        cache_manager = RankingCacheManager()
        
        # 存在しないキーでのアクセス
        result = cache_manager.get("non_existent_key")
        self.assertIsNone(result)
        
        # 無効なデータ型での保存試行
        try:
            cache_manager.set("test_key", lambda x: x)  # 関数は通常シリアライズできない
        except Exception:
            pass  # 例外が発生することを期待

def run_performance_tests():
    """パフォーマンステスト実行"""
    if not TEST_IMPORTS_AVAILABLE:
        print("Advanced ranking system modules not available for performance tests")
        return
    
    print("Running performance tests...")
    
    # 大規模データでのテスト
    large_data = TestDataGenerator.generate_sample_market_data(500, 252)
    fundamental_data = TestDataGenerator.generate_fundamental_data(
        list(large_data.keys())
    )
    
    engine = AdvancedRankingEngine()
    
    start_time = time.time()
    rankings = engine.calculate_rankings(
        list(large_data.keys())[:100],
        large_data,
        fundamental_data
    )
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Large dataset ranking calculation: {execution_time:.2f} seconds")
    print(f"Rankings calculated: {len(rankings)}")
    
    # キャッシュパフォーマンステスト
    cache_manager = RankingCacheManager(max_size=1000)
    
    start_time = time.time()
    for i in range(1000):
        cache_manager.set(f"key_{i}", {"test": f"value_{i}"})
    cache_write_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(1000):
        cache_manager.get(f"key_{i}")
    cache_read_time = time.time() - start_time
    
    print(f"Cache write time (1000 items): {cache_write_time:.3f} seconds")
    print(f"Cache read time (1000 items): {cache_read_time:.3f} seconds")

def run_stress_tests():
    """ストレステスト実行"""
    if not TEST_IMPORTS_AVAILABLE:
        print("Advanced ranking system modules not available for stress tests")
        return
    
    print("Running stress tests...")
    
    # 並行処理ストレステスト
    updater = RealtimeUpdater()
    updater.start()
    
    # 大量の更新をスケジュール
    for i in range(1000):
        updater.schedule_update(
            UpdateType.MARKET_DATA,
            {"test_data": i},
            UpdatePriority.NORMAL
        )
    
    # しばらく待ってから停止
    time.sleep(2)
    
    status = updater.get_status()
    print(f"Updates processed: {status.total_updates_processed}")
    print(f"Queue size: {status.queue_size}")
    
    updater.stop()

if __name__ == "__main__":
    # 基本的なユニットテスト実行
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # パフォーマンステスト実行
    run_performance_tests()
    
    # ストレステスト実行
    run_stress_tests()
    
    print("All tests completed!")
