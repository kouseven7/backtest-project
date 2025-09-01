"""
DSSMS Phase 3 Task 3.1: Advanced Ranking System Demo
高度ランキングシステム動作確認スクリプト

システム全体の動作確認とデモンストレーションを実行します。
"""

import sys
from pathlib import Path
import asyncio
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# ロガー設定
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 高度ランキングシステムのインポート
try:
    from src.dssms.advanced_ranking_system.advanced_ranking_engine import AdvancedRankingEngine, RankingConfig
    from src.dssms.advanced_ranking_system.multi_dimensional_analyzer import MultiDimensionalAnalyzer, AnalysisConfig
    from src.dssms.advanced_ranking_system.dynamic_weight_optimizer import DynamicWeightOptimizer, OptimizationConfig
    from src.dssms.advanced_ranking_system.integration_bridge import IntegrationBridge, IntegrationMode
    from src.dssms.advanced_ranking_system.ranking_cache_manager import RankingCacheManager, CacheStrategy
    from src.dssms.advanced_ranking_system.performance_monitor import PerformanceMonitor, MonitoringConfig
    from src.dssms.advanced_ranking_system.realtime_updater import RealtimeUpdater, UpdateType, UpdatePriority
    SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.error(f"Advanced ranking system modules not available: {e}")
    SYSTEM_AVAILABLE = False

class DemoDataGenerator:
    """デモ用データ生成クラス"""
    
    @staticmethod
    def generate_market_data(n_stocks: int = 50, n_days: int = 252) -> Dict[str, pd.DataFrame]:
        """市場データ生成"""
        logger.info(f"Generating market data for {n_stocks} stocks, {n_days} days")
        
        symbols = [f"DEMO_{i:03d}" for i in range(n_stocks)]
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        data = {}
        
        for i, symbol in enumerate(symbols):
            # セクター別の特性を持たせる
            sector_factor = 1.0 + (i % 5) * 0.1
            
            # ランダムウォーク + トレンド
            prices = []
            price = 100.0 * sector_factor
            trend = np.random.uniform(-0.0005, 0.0005)
            
            for day in range(n_days):
                # トレンド + ランダムウォーク + 季節性
                seasonal = 0.001 * np.sin(2 * np.pi * day / 252)
                daily_return = trend + seasonal + np.random.normal(0, 0.02)
                price *= (1 + daily_return)
                prices.append(price)
            
            volumes = np.random.lognormal(10, 1, n_days).astype(int)
            
            # OHLC生成
            high_low_range = np.random.uniform(0.005, 0.02, n_days)
            open_close_range = np.random.uniform(-0.01, 0.01, n_days)
            
            data[symbol] = pd.DataFrame({
                'Date': dates,
                'Open': np.array(prices) * (1 + open_close_range),
                'High': np.array(prices) * (1 + high_low_range),
                'Low': np.array(prices) * (1 - high_low_range),
                'Close': prices,
                'Volume': volumes
            })
            
        logger.info(f"Market data generated successfully")
        return data
    
    @staticmethod
    def generate_fundamental_data(symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """ファンダメンタルデータ生成"""
        logger.info(f"Generating fundamental data for {len(symbols)} symbols")
        
        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Utilities']
        fundamental_data = {}
        
        for i, symbol in enumerate(symbols):
            sector = sectors[i % len(sectors)]
            
            # セクター別の特徴的な指標
            if sector == 'Technology':
                pe_base, pb_base = 25, 5
                roe_base, growth_base = 0.15, 0.20
            elif sector == 'Healthcare':
                pe_base, pb_base = 20, 3
                roe_base, growth_base = 0.12, 0.10
            elif sector == 'Finance':
                pe_base, pb_base = 12, 1.2
                roe_base, growth_base = 0.10, 0.05
            elif sector == 'Energy':
                pe_base, pb_base = 15, 2
                roe_base, growth_base = 0.08, -0.05
            elif sector == 'Consumer':
                pe_base, pb_base = 18, 2.5
                roe_base, growth_base = 0.14, 0.08
            else:  # Utilities
                pe_base, pb_base = 16, 1.5
                roe_base, growth_base = 0.09, 0.03
            
            fundamental_data[symbol] = {
                'pe_ratio': max(5, pe_base + np.random.normal(0, 5)),
                'pb_ratio': max(0.5, pb_base + np.random.normal(0, 1)),
                'roe': max(0.01, roe_base + np.random.normal(0, 0.05)),
                'roa': max(0.01, roe_base * 0.6 + np.random.normal(0, 0.03)),
                'debt_ratio': max(0.05, min(0.8, 0.3 + np.random.normal(0, 0.15))),
                'current_ratio': max(0.5, 1.5 + np.random.normal(0, 0.5)),
                'revenue_growth': growth_base + np.random.normal(0, 0.1),
                'profit_margin': max(0.01, 0.1 + np.random.normal(0, 0.05)),
                'market_cap': np.random.uniform(1e9, 1e12),
                'sector': sector
            }
            
        logger.info(f"Fundamental data generated successfully")
        return fundamental_data

class AdvancedRankingSystemDemo:
    """高度ランキングシステムデモクラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = logger
        self.demo_results = {}
        
        if SYSTEM_AVAILABLE:
            # システムコンポーネント初期化
            self.ranking_engine = None
            self.analyzer = None
            self.optimizer = None
            self.integration_bridge = None
            self.cache_manager = None
            self.performance_monitor = None
            self.realtime_updater = None
        
    def initialize_system(self):
        """システム初期化"""
        if not SYSTEM_AVAILABLE:
            self.logger.error("Advanced ranking system not available")
            return False
        
        try:
            self.logger.info("Initializing advanced ranking system components...")
            
            # 各コンポーネント初期化
            self.ranking_engine = AdvancedRankingEngine()
            self.analyzer = MultiDimensionalAnalyzer()
            self.optimizer = DynamicWeightOptimizer()
            self.integration_bridge = IntegrationBridge()
            self.cache_manager = RankingCacheManager()
            self.performance_monitor = PerformanceMonitor()
            self.realtime_updater = RealtimeUpdater()
            
            self.logger.info("System components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False
    
    def run_basic_ranking_demo(self, market_data: Dict, fundamental_data: Dict):
        """基本ランキングデモ"""
        if not SYSTEM_AVAILABLE:
            self.logger.warning("Skipping basic ranking demo - system not available")
            return {}
        
        try:
            self.logger.info("Running basic ranking demonstration...")
            
            symbols = list(market_data.keys())[:20]  # 20銘柄で実行
            start_time = time.time()
            
            # ランキング計算
            rankings = self.ranking_engine.calculate_rankings(
                symbols,
                market_data,
                fundamental_data
            )
            
            execution_time = time.time() - start_time
            
            # 結果分析
            sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            top_10 = sorted_rankings[:10]
            bottom_10 = sorted_rankings[-10:]
            
            results = {
                'total_symbols': len(symbols),
                'execution_time': execution_time,
                'top_10': top_10,
                'bottom_10': bottom_10,
                'average_score': np.mean(list(rankings.values())),
                'score_std': np.std(list(rankings.values()))
            }
            
            self.demo_results['basic_ranking'] = results
            
            self.logger.info(f"Basic ranking completed in {execution_time:.2f} seconds")
            self.logger.info(f"Top performer: {top_10[0][0]} (score: {top_10[0][1]:.4f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Basic ranking demo failed: {e}")
            return {}
    
    async def run_async_ranking_demo(self, market_data: Dict, fundamental_data: Dict):
        """非同期ランキングデモ"""
        if not SYSTEM_AVAILABLE:
            self.logger.warning("Skipping async ranking demo - system not available")
            return {}
        
        try:
            self.logger.info("Running async ranking demonstration...")
            
            symbols = list(market_data.keys())[:30]  # 30銘柄で実行
            start_time = time.time()
            
            # 非同期ランキング計算
            rankings = await self.ranking_engine.calculate_rankings_async(
                symbols,
                market_data,
                fundamental_data
            )
            
            execution_time = time.time() - start_time
            
            results = {
                'total_symbols': len(symbols),
                'execution_time': execution_time,
                'rankings_count': len(rankings)
            }
            
            self.demo_results['async_ranking'] = results
            
            self.logger.info(f"Async ranking completed in {execution_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Async ranking demo failed: {e}")
            return {}
    
    def run_multi_dimensional_analysis_demo(self, market_data: Dict):
        """多次元分析デモ"""
        if not SYSTEM_AVAILABLE:
            self.logger.warning("Skipping multi-dimensional analysis demo - system not available")
            return {}
        
        try:
            self.logger.info("Running multi-dimensional analysis demonstration...")
            
            symbol = list(market_data.keys())[0]
            data = market_data[symbol]
            
            start_time = time.time()
            
            # 各次元のスコア計算
            momentum_score = self.analyzer.calculate_momentum_score(data)
            volatility_score = self.analyzer.calculate_volatility_score(data)
            volume_score = self.analyzer.calculate_volume_score(data)
            technical_score = self.analyzer.calculate_technical_score(data)
            
            execution_time = time.time() - start_time
            
            results = {
                'symbol': symbol,
                'momentum_score': momentum_score,
                'volatility_score': volatility_score,
                'volume_score': volume_score,
                'technical_score': technical_score,
                'execution_time': execution_time
            }
            
            self.demo_results['multi_dimensional_analysis'] = results
            
            self.logger.info(f"Multi-dimensional analysis completed in {execution_time:.3f} seconds")
            self.logger.info(f"Sample scores - Momentum: {momentum_score:.3f}, Volatility: {volatility_score:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-dimensional analysis demo failed: {e}")
            return {}
    
    def run_weight_optimization_demo(self, market_data: Dict):
        """重み最適化デモ"""
        if not SYSTEM_AVAILABLE:
            self.logger.warning("Skipping weight optimization demo - system not available")
            return {}
        
        try:
            self.logger.info("Running weight optimization demonstration...")
            
            # 初期重み
            initial_weights = {
                'momentum': 0.25,
                'volatility': 0.20,
                'volume': 0.15,
                'technical': 0.20,
                'fundamental': 0.20
            }
            
            # サンプルパフォーマンスデータ
            performance_data = np.random.uniform(-0.1, 0.1, 100)
            
            start_time = time.time()
            
            # 重み最適化実行
            optimized_weights = self.optimizer.optimize_weights(
                initial_weights,
                performance_data
            )
            
            # 市場レジーム検出
            market_regime = self.optimizer.detect_market_regime(market_data)
            
            execution_time = time.time() - start_time
            
            results = {
                'initial_weights': initial_weights,
                'optimized_weights': optimized_weights,
                'market_regime': market_regime,
                'execution_time': execution_time,
                'improvement': self._calculate_weight_improvement(initial_weights, optimized_weights)
            }
            
            self.demo_results['weight_optimization'] = results
            
            self.logger.info(f"Weight optimization completed in {execution_time:.3f} seconds")
            self.logger.info(f"Market regime detected: {market_regime}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Weight optimization demo failed: {e}")
            return {}
    
    def run_integration_demo(self, market_data: Dict, fundamental_data: Dict):
        """統合機能デモ"""
        if not SYSTEM_AVAILABLE:
            self.logger.warning("Skipping integration demo - system not available")
            return {}
        
        try:
            self.logger.info("Running integration demonstration...")
            
            symbols = list(market_data.keys())[:15]
            
            # 模擬的なレガシーシステムランキング
            legacy_rankings = {
                symbol: np.random.uniform(0.3, 0.8) 
                for symbol in symbols
            }
            
            # 高度システムランキング
            advanced_rankings = self.ranking_engine.calculate_rankings(
                symbols,
                market_data,
                fundamental_data
            )
            
            start_time = time.time()
            
            # 統合ランキング計算
            integrated_rankings = self.integration_bridge.combine_rankings(
                legacy_rankings,
                advanced_rankings
            )
            
            execution_time = time.time() - start_time
            
            results = {
                'legacy_count': len(legacy_rankings),
                'advanced_count': len(advanced_rankings),
                'integrated_count': len(integrated_rankings),
                'execution_time': execution_time,
                'correlation': self._calculate_ranking_correlation(legacy_rankings, advanced_rankings)
            }
            
            self.demo_results['integration'] = results
            
            self.logger.info(f"Integration completed in {execution_time:.3f} seconds")
            self.logger.info(f"Ranking correlation: {results['correlation']:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Integration demo failed: {e}")
            return {}
    
    def run_cache_performance_demo(self):
        """キャッシュパフォーマンスデモ"""
        if not SYSTEM_AVAILABLE:
            self.logger.warning("Skipping cache performance demo - system not available")
            return {}
        
        try:
            self.logger.info("Running cache performance demonstration...")
            
            # 書き込みパフォーマンステスト
            write_start = time.time()
            for i in range(1000):
                self.cache_manager.set(f"demo_key_{i}", {"score": np.random.random()})
            write_time = time.time() - write_start
            
            # 読み込みパフォーマンステスト
            read_start = time.time()
            hit_count = 0
            for i in range(1000):
                result = self.cache_manager.get(f"demo_key_{i}")
                if result is not None:
                    hit_count += 1
            read_time = time.time() - read_start
            
            # キャッシュ統計
            stats = self.cache_manager.get_statistics()
            
            results = {
                'write_time': write_time,
                'read_time': read_time,
                'hit_rate': hit_count / 1000,
                'cache_size': stats.get('cache_size', 0),
                'total_requests': stats.get('total_requests', 0)
            }
            
            self.demo_results['cache_performance'] = results
            
            self.logger.info(f"Cache write time: {write_time:.3f}s, read time: {read_time:.3f}s")
            self.logger.info(f"Cache hit rate: {results['hit_rate']:.2%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Cache performance demo failed: {e}")
            return {}
    
    def run_performance_monitoring_demo(self):
        """パフォーマンス監視デモ"""
        if not SYSTEM_AVAILABLE:
            self.logger.warning("Skipping performance monitoring demo - system not available")
            return {}
        
        try:
            self.logger.info("Running performance monitoring demonstration...")
            
            # メトリクス記録
            for i in range(10):
                execution_time = np.random.uniform(0.5, 2.0)
                memory_usage = np.random.randint(100, 1000)
                cpu_usage = np.random.uniform(20, 90)
                
                self.performance_monitor.record_execution_time("demo_operation", execution_time)
                self.performance_monitor.record_memory_usage(memory_usage)
                self.performance_monitor.record_cpu_usage(cpu_usage)
            
            # 統計取得
            stats = self.performance_monitor.get_statistics()
            alerts = self.performance_monitor.get_active_alerts()
            health_score = self.performance_monitor.get_health_score()
            
            results = {
                'statistics': stats,
                'active_alerts': len(alerts),
                'health_score': health_score,
                'monitoring_enabled': True
            }
            
            self.demo_results['performance_monitoring'] = results
            
            self.logger.info(f"System health score: {health_score:.2f}")
            self.logger.info(f"Active alerts: {len(alerts)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Performance monitoring demo failed: {e}")
            return {}
    
    def run_realtime_update_demo(self):
        """リアルタイム更新デモ"""
        if not SYSTEM_AVAILABLE:
            self.logger.warning("Skipping realtime update demo - system not available")
            return {}
        
        try:
            self.logger.info("Running realtime update demonstration...")
            
            # リアルタイム更新開始
            self.realtime_updater.start()
            
            # 更新イベントスケジュール
            event_ids = []
            for i in range(50):
                event_id = self.realtime_updater.schedule_update(
                    UpdateType.RANKING_SCORES,
                    {"demo_update": i},
                    UpdatePriority.NORMAL
                )
                if event_id:
                    event_ids.append(event_id)
            
            # 少し待機
            time.sleep(2)
            
            # ステータス取得
            status = self.realtime_updater.get_status()
            queue_status = self.realtime_updater.get_queue_status()
            
            # 停止
            self.realtime_updater.stop()
            
            results = {
                'scheduled_events': len(event_ids),
                'processed_updates': status.total_updates_processed,
                'failed_updates': status.failed_updates,
                'queue_status': queue_status,
                'average_processing_time': status.average_processing_time
            }
            
            self.demo_results['realtime_update'] = results
            
            self.logger.info(f"Scheduled {len(event_ids)} events, processed {status.total_updates_processed}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Realtime update demo failed: {e}")
            return {}
    
    def _calculate_weight_improvement(self, initial: Dict, optimized: Dict) -> float:
        """重み改善度計算"""
        try:
            initial_variance = np.var(list(initial.values()))
            optimized_variance = np.var(list(optimized.values()))
            return (initial_variance - optimized_variance) / initial_variance
        except:
            return 0.0
    
    def _calculate_ranking_correlation(self, rankings1: Dict, rankings2: Dict) -> float:
        """ランキング相関計算"""
        try:
            common_symbols = set(rankings1.keys()) & set(rankings2.keys())
            if len(common_symbols) < 2:
                return 0.0
            
            values1 = [rankings1[symbol] for symbol in common_symbols]
            values2 = [rankings2[symbol] for symbol in common_symbols]
            
            return np.corrcoef(values1, values2)[0, 1]
        except:
            return 0.0
    
    def generate_demo_report(self):
        """デモレポート生成"""
        try:
            report = {
                'demo_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'system_available': SYSTEM_AVAILABLE,
                    'demos_completed': len(self.demo_results),
                    'total_execution_time': sum(
                        result.get('execution_time', 0) 
                        for result in self.demo_results.values()
                    )
                },
                'demo_results': self.demo_results
            }
            
            # レポートファイル保存
            report_file = f"advanced_ranking_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Demo report saved to {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate demo report: {e}")
            return {}

def main():
    """メイン実行関数"""
    logger.info("Starting Advanced Ranking System Demonstration")
    
    # デモシステム初期化
    demo = AdvancedRankingSystemDemo()
    
    if not demo.initialize_system():
        logger.error("Failed to initialize demo system")
        return
    
    # テストデータ生成
    logger.info("Generating demo data...")
    market_data = DemoDataGenerator.generate_market_data(100, 252)
    fundamental_data = DemoDataGenerator.generate_fundamental_data(
        list(market_data.keys())
    )
    
    # 各デモ実行
    demo_functions = [
        ("Basic Ranking", lambda: demo.run_basic_ranking_demo(market_data, fundamental_data)),
        ("Multi-dimensional Analysis", lambda: demo.run_multi_dimensional_analysis_demo(market_data)),
        ("Weight Optimization", lambda: demo.run_weight_optimization_demo(market_data)),
        ("Integration", lambda: demo.run_integration_demo(market_data, fundamental_data)),
        ("Cache Performance", lambda: demo.run_cache_performance_demo()),
        ("Performance Monitoring", lambda: demo.run_performance_monitoring_demo()),
        ("Realtime Update", lambda: demo.run_realtime_update_demo())
    ]
    
    for demo_name, demo_func in demo_functions:
        logger.info(f"Running {demo_name} demo...")
        try:
            result = demo_func()
            if result:
                logger.info(f"{demo_name} demo completed successfully")
            else:
                logger.warning(f"{demo_name} demo returned empty result")
        except Exception as e:
            logger.error(f"{demo_name} demo failed: {e}")
    
    # 非同期デモ実行
    logger.info("Running Async Ranking demo...")
    try:
        asyncio.run(demo.run_async_ranking_demo(market_data, fundamental_data))
        logger.info("Async Ranking demo completed successfully")
    except Exception as e:
        logger.error(f"Async Ranking demo failed: {e}")
    
    # レポート生成
    logger.info("Generating demo report...")
    report = demo.generate_demo_report()
    
    # サマリー出力
    logger.info("=== DEMO SUMMARY ===")
    logger.info(f"System Available: {SYSTEM_AVAILABLE}")
    logger.info(f"Demos Completed: {len(demo.demo_results)}")
    
    if demo.demo_results:
        total_time = sum(
            result.get('execution_time', 0) 
            for result in demo.demo_results.values()
        )
        logger.info(f"Total Execution Time: {total_time:.2f} seconds")
        
        # 各デモ結果サマリー
        for demo_name, result in demo.demo_results.items():
            if 'execution_time' in result:
                logger.info(f"{demo_name}: {result['execution_time']:.3f}s")
    
    logger.info("Advanced Ranking System Demonstration completed")

if __name__ == "__main__":
    main()
