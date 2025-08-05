"""
Demo: Score Update Trigger System Integration
File: demo_score_update_trigger_integration.py
Description:
  2-3-3「スコアアップデートトリガー設計」統合デモ
  ScoreUpdateTriggerSystemとRealtimeUpdateEngineの連携動作確認

Author: GitHub Copilot
Created: 2025-07-13
Modified: 2025-07-13
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

# プロジェクトモジュールのインポート
try:
    from config.score_update_trigger_system import (
        ScoreUpdateTriggerSystem, TriggerType, TriggerPriority, TriggerCondition
    )
    from config.realtime_update_engine import (
        RealtimeUpdateEngine, UpdatePriority, UpdateRequest
    )
    from config.enhanced_score_history_manager import EnhancedScoreHistoryManager
    from config.strategy_scoring_model import StrategyScoreCalculator
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from config.score_update_trigger_system import (
        ScoreUpdateTriggerSystem, TriggerType, TriggerPriority, TriggerCondition
    )
    from config.realtime_update_engine import (
        RealtimeUpdateEngine, UpdatePriority, UpdateRequest
    )
    from config.enhanced_score_history_manager import EnhancedScoreHistoryManager
    from config.strategy_scoring_model import StrategyScoreCalculator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('score_trigger_demo.log')
    ]
)

logger = logging.getLogger(__name__)

class TriggerSystemIntegrationDemo:
    """
    スコアアップデートトリガーシステム統合デモ
    
    2-3-3で設計したシステムの動作を実演：
    1. トリガー条件の設定と監視
    2. リアルタイムエンジンとの連携
    3. 各種トリガータイプの動作確認
    4. パフォーマンス測定
    """
    
    def __init__(self):
        """初期化"""
        # コアコンポーネント
        self.enhanced_manager = None
        self.score_calculator = None
        self.trigger_system = None
        self.realtime_engine = None
        
        # デモ状態
        self.demo_results = {}
        self.demo_start_time = None
        
        logger.info("Trigger System Integration Demo initialized")
    
    async def run_demo(self):
        """デモ実行"""
        self.demo_start_time = datetime.now()
        
        print("=" * 80)
        print("🚀 Score Update Trigger System Integration Demo")
        print("=" * 80)
        print(f"Started at: {self.demo_start_time}")
        print()
        
        try:
            # 1. システム初期化
            await self._setup_systems()
            
            # 2. 基本トリガー動作確認
            await self._demo_basic_triggers()
            
            # 3. 閾値ベーストリガー確認
            await self._demo_threshold_triggers()
            
            # 4. バッチ処理確認
            await self._demo_batch_processing()
            
            # 5. パフォーマンス測定
            await self._demo_performance_test()
            
            # 6. エラー耐性確認
            await self._demo_error_handling()
            
            # 7. 結果サマリー
            await self._display_demo_summary()
            
        except Exception as e:
            logger.error(f"Demo execution failed: {e}")
            print(f"❌ Demo failed: {e}")
        
        finally:
            await self._cleanup_systems()
        
        print("\n✅ Demo completed!")
    
    async def _setup_systems(self):
        """システム初期化"""
        print("📋 1. Setting up systems...")
        
        try:
            # 拡張スコア履歴管理システム
            self.enhanced_manager = EnhancedScoreHistoryManager()
            print("   ✓ Enhanced Score History Manager")
            
            # スコア計算器
            self.score_calculator = StrategyScoreCalculator()
            print("   ✓ Strategy Score Calculator")
            
            # トリガーシステム
            self.trigger_system = ScoreUpdateTriggerSystem(
                enhanced_manager=self.enhanced_manager,
                score_calculator=self.score_calculator
            )
            print("   ✓ Score Update Trigger System")
            
            # リアルタイムエンジン
            self.realtime_engine = RealtimeUpdateEngine(
                trigger_system=self.trigger_system,
                enhanced_manager=self.enhanced_manager,
                score_calculator=self.score_calculator
            )
            print("   ✓ Realtime Update Engine")
            
            # システム開始
            self.trigger_system.start()
            print("   ✓ Trigger system started")
            
            # リアルタイムエンジン開始（バックグラウンド）
            engine_task = asyncio.create_task(self.realtime_engine.start())
            await asyncio.sleep(1)  # 開始待機
            print("   ✓ Realtime engine started")
            
            # 結果記録
            self.demo_results["setup_success"] = True
            
        except Exception as e:
            self.demo_results["setup_success"] = False
            self.demo_results["setup_error"] = str(e)
            raise
        
        print("   ✅ All systems ready!")
        print()
    
    async def _demo_basic_triggers(self):
        """基本トリガー動作確認"""
        print("🎯 2. Basic trigger operations...")
        
        results = {}
        
        try:
            # 手動トリガーテスト
            print("   Testing manual triggers...")
            
            event_id = self.trigger_system.manual_trigger(
                strategy_name="demo_strategy_1",
                ticker="DEMO1",
                priority=TriggerPriority.HIGH,
                metadata={"test_type": "manual", "demo_step": "basic"}
            )
            
            results["manual_trigger_id"] = event_id
            print(f"   ✓ Manual trigger queued: {event_id}")
            
            # 少し待機してトリガー処理を確認
            await asyncio.sleep(2)
            
            # トリガー統計確認
            stats = self.trigger_system.get_trigger_statistics()
            results["initial_stats"] = stats
            print(f"   ✓ Trigger stats: {stats['total_triggers']} total, {stats['queue_size']} queued")
            
            # カスタムトリガー条件追加
            print("   Adding custom trigger condition...")
            
            custom_condition = TriggerCondition(
                condition_id="demo_custom_trigger",
                trigger_type=TriggerType.EVENT_BASED,
                priority=TriggerPriority.MEDIUM,
                parameters={
                    "demo_parameter": True,
                    "sensitivity": 0.1
                },
                description="Demo custom trigger condition"
            )
            
            self.trigger_system.add_trigger_condition(custom_condition)
            results["custom_condition_added"] = True
            print("   ✓ Custom trigger condition added")
            
            # リアルタイムエンジン状態確認
            engine_status = self.realtime_engine.get_engine_status()
            results["engine_status"] = engine_status
            print(f"   ✓ Engine status: {engine_status['status']}, {engine_status['queue_size']} queued")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Basic triggers demo failed: {e}")
        
        self.demo_results["basic_triggers"] = results
        print("   ✅ Basic trigger operations completed!")
        print()
    
    async def _demo_threshold_triggers(self):
        """閾値ベーストリガー確認"""
        print("📊 3. Threshold-based triggers...")
        
        results = {}
        
        try:
            # 閾値トリガー条件追加
            print("   Setting up threshold trigger...")
            
            threshold_condition = TriggerCondition(
                condition_id="demo_threshold_trigger",
                trigger_type=TriggerType.THRESHOLD_BASED,
                priority=TriggerPriority.HIGH,
                parameters={
                    "score_change_threshold": 0.05,  # 5%変化で発火
                    "monitoring_window_hours": 1,
                    "demo_mode": True
                },
                strategy_filter=["demo_strategy_2"],
                cooldown_seconds=30,  # 30秒クールダウン
                description="Demo threshold trigger (5% score change)"
            )
            
            self.trigger_system.add_trigger_condition(threshold_condition)
            results["threshold_condition_added"] = True
            print("   ✓ Threshold trigger condition added (5% threshold)")
            
            # 初期スコア生成（シミュレーション用）
            print("   Generating initial scores...")
            
            initial_requests = []
            for i in range(3):
                request = UpdateRequest(
                    request_id=f"threshold_demo_init_{i}",
                    strategy_name="demo_strategy_2",
                    ticker=f"THRESH{i}",
                    trigger_type=TriggerType.MANUAL,
                    priority=3,
                    metadata={"phase": "initial", "iteration": i}
                )
                
                task_id = await self.realtime_engine.submit_update_request(
                    request, UpdatePriority.NORMAL
                )
                initial_requests.append(task_id)
            
            # 初期処理完了待機
            await asyncio.sleep(3)
            results["initial_scores_generated"] = len(initial_requests)
            print(f"   ✓ Generated {len(initial_requests)} initial scores")
            
            # 閾値を超える変化をシミュレーション
            print("   Simulating significant score changes...")
            
            change_requests = []
            for i in range(2):
                request = UpdateRequest(
                    request_id=f"threshold_demo_change_{i}",
                    strategy_name="demo_strategy_2",
                    ticker=f"THRESH{i}",
                    trigger_type=TriggerType.MANUAL,
                    priority=1,  # 高優先度
                    metadata={
                        "phase": "significant_change",
                        "expected_threshold_trigger": True,
                        "iteration": i
                    }
                )
                
                task_id = await self.realtime_engine.submit_update_request(
                    request, UpdatePriority.REALTIME
                )
                change_requests.append(task_id)
            
            # 変化処理完了待機
            await asyncio.sleep(5)
            results["change_scores_generated"] = len(change_requests)
            print(f"   ✓ Simulated {len(change_requests)} significant changes")
            
            # 閾値トリガーの動作確認
            recent_events = self.trigger_system.get_recent_events(limit=10)
            threshold_events = [
                e for e in recent_events 
                if e.get("condition_id") == "demo_threshold_trigger"
            ]
            
            results["threshold_events_triggered"] = len(threshold_events)
            print(f"   ✓ Threshold trigger fired {len(threshold_events)} times")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Threshold triggers demo failed: {e}")
        
        self.demo_results["threshold_triggers"] = results
        print("   ✅ Threshold-based triggers completed!")
        print()
    
    async def _demo_batch_processing(self):
        """バッチ処理確認"""
        print("🔄 4. Batch processing...")
        
        results = {}
        
        try:
            # バッチジョブ開始前の状態
            print("   Checking batch job capabilities...")
            
            initial_engine_status = self.realtime_engine.get_engine_status()
            results["initial_engine_status"] = initial_engine_status
            print(f"   ✓ Initial queue size: {initial_engine_status['queue_size']}")
            
            # 大量の低優先度リクエストを生成
            print("   Generating batch update requests...")
            
            batch_requests = []
            strategies = ["batch_strategy_1", "batch_strategy_2", "batch_strategy_3"]
            tickers = ["BATCH1", "BATCH2", "BATCH3", "BATCH4"]
            
            for strategy in strategies:
                for ticker in tickers:
                    request = UpdateRequest(
                        request_id=f"batch_{strategy}_{ticker}_{int(time.time())}",
                        strategy_name=strategy,
                        ticker=ticker,
                        trigger_type=TriggerType.TIME_BASED,
                        priority=5,  # バッチ優先度
                        metadata={
                            "batch_demo": True,
                            "batch_size": len(strategies) * len(tickers)
                        }
                    )
                    
                    task_id = await self.realtime_engine.submit_update_request(
                        request, UpdatePriority.BATCH
                    )
                    batch_requests.append(task_id)
            
            results["batch_requests_submitted"] = len(batch_requests)
            print(f"   ✓ Submitted {len(batch_requests)} batch requests")
            
            # バッチ処理進捗監視
            print("   Monitoring batch processing...")
            
            monitoring_start = time.time()
            batch_completed = False
            max_monitoring_time = 30  # 最大30秒監視
            
            while time.time() - monitoring_start < max_monitoring_time and not batch_completed:
                await asyncio.sleep(2)
                
                current_status = self.realtime_engine.get_engine_status()
                
                print(f"   📊 Queue: {current_status['queue_size']}, "
                      f"Completed: {current_status['successful_updates']}, "
                      f"Failed: {current_status['failed_updates']}")
                
                # バッチ処理完了判定
                if current_status['queue_size'] < 5:  # キューがほぼ空になった
                    batch_completed = True
            
            final_status = self.realtime_engine.get_engine_status()
            results["final_engine_status"] = final_status
            results["batch_processing_time"] = time.time() - monitoring_start
            results["batch_completed"] = batch_completed
            
            print(f"   ✓ Batch processing {'completed' if batch_completed else 'monitored'}")
            print(f"   ✓ Final queue size: {final_status['queue_size']}")
            print(f"   ✓ Processing time: {results['batch_processing_time']:.2f}s")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Batch processing demo failed: {e}")
        
        self.demo_results["batch_processing"] = results
        print("   ✅ Batch processing completed!")
        print()
    
    async def _demo_performance_test(self):
        """パフォーマンス測定"""
        print("⚡ 5. Performance testing...")
        
        results = {}
        
        try:
            # パフォーマンステスト設定
            test_strategies = ["perf_strategy"]
            test_tickers = ["PERF1", "PERF2", "PERF3"]
            test_requests_per_ticker = 5
            
            total_requests = len(test_strategies) * len(test_tickers) * test_requests_per_ticker
            
            print(f"   Running performance test with {total_requests} requests...")
            
            # 開始時刻記録
            perf_start_time = time.time()
            initial_stats = self.realtime_engine.get_engine_status()
            
            # 高優先度リクエストを大量送信
            perf_requests = []
            
            for strategy in test_strategies:
                for ticker in test_tickers:
                    for i in range(test_requests_per_ticker):
                        request = UpdateRequest(
                            request_id=f"perf_{strategy}_{ticker}_{i}_{int(time.time())}",
                            strategy_name=strategy,
                            ticker=ticker,
                            trigger_type=TriggerType.MANUAL,
                            priority=2,  # 高優先度
                            metadata={
                                "performance_test": True,
                                "batch_number": i,
                                "expected_total": total_requests
                            }
                        )
                        
                        task_id = await self.realtime_engine.submit_update_request(
                            request, UpdatePriority.HIGH
                        )
                        perf_requests.append(task_id)
            
            print(f"   ✓ Submitted {len(perf_requests)} performance test requests")
            
            # パフォーマンス監視
            monitoring_start = time.time()
            performance_completed = False
            
            while time.time() - monitoring_start < 20 and not performance_completed:  # 20秒監視
                await asyncio.sleep(1)
                
                current_stats = self.realtime_engine.get_engine_status()
                
                # 処理完了判定
                completed_delta = (current_stats['successful_updates'] + current_stats['failed_updates']) - \
                                (initial_stats['successful_updates'] + initial_stats['failed_updates'])
                
                if completed_delta >= total_requests:
                    performance_completed = True
                
                print(f"   📈 Processed: {completed_delta}/{total_requests}, "
                      f"Queue: {current_stats['queue_size']}, "
                      f"Avg time: {current_stats['average_processing_time']:.4f}s")
            
            # 最終結果
            perf_end_time = time.time()
            final_stats = self.realtime_engine.get_engine_status()
            
            total_processing_time = perf_end_time - perf_start_time
            completed_requests = (final_stats['successful_updates'] + final_stats['failed_updates']) - \
                               (initial_stats['successful_updates'] + initial_stats['failed_updates'])
            
            throughput = completed_requests / total_processing_time if total_processing_time > 0 else 0
            
            results.update({
                "total_requests": total_requests,
                "completed_requests": completed_requests,
                "total_processing_time": total_processing_time,
                "throughput_per_second": throughput,
                "average_processing_time": final_stats['average_processing_time'],
                "performance_completed": performance_completed,
                "success_rate": (final_stats['successful_updates'] - initial_stats['successful_updates']) / completed_requests if completed_requests > 0 else 0
            })
            
            print(f"   ✓ Performance test completed:")
            print(f"     - Total time: {total_processing_time:.2f}s")
            print(f"     - Throughput: {throughput:.2f} requests/sec")
            print(f"     - Avg processing: {final_stats['average_processing_time']:.4f}s")
            print(f"     - Success rate: {results['success_rate']*100:.1f}%")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Performance test failed: {e}")
        
        self.demo_results["performance_test"] = results
        print("   ✅ Performance testing completed!")
        print()
    
    async def _demo_error_handling(self):
        """エラー耐性確認"""
        print("🛡️  6. Error handling and resilience...")
        
        results = {}
        
        try:
            # 無効なデータでリクエスト送信
            print("   Testing error resilience...")
            
            error_requests = []
            
            # 1. 無効な戦略名
            invalid_strategy_request = UpdateRequest(
                request_id="error_test_invalid_strategy",
                strategy_name="",  # 空の戦略名
                ticker="ERROR1",
                trigger_type=TriggerType.MANUAL,
                priority=2,
                metadata={"error_test": "invalid_strategy"}
            )
            
            task_id = await self.realtime_engine.submit_update_request(
                invalid_strategy_request, UpdatePriority.HIGH
            )
            error_requests.append(task_id)
            
            # 2. 無効なティッカー
            invalid_ticker_request = UpdateRequest(
                request_id="error_test_invalid_ticker",
                strategy_name="error_test_strategy",
                ticker=None,  # None ティッカー
                trigger_type=TriggerType.MANUAL,
                priority=2,
                metadata={"error_test": "invalid_ticker"}
            )
            
            # このリクエストはエラーになるはずなのでtry-catchで処理
            try:
                task_id = await self.realtime_engine.submit_update_request(
                    invalid_ticker_request, UpdatePriority.HIGH
                )
                error_requests.append(task_id)
            except Exception as e:
                print(f"   ✓ Caught expected error for invalid ticker: {type(e).__name__}")
                results["invalid_ticker_error_caught"] = True
            
            # 3. 大量の同時リクエスト（負荷テスト）
            print("   Testing high load resilience...")
            
            load_test_start = time.time()
            concurrent_requests = []
            
            for i in range(20):  # 20個の同時リクエスト
                request = UpdateRequest(
                    request_id=f"load_test_{i}",
                    strategy_name="load_test_strategy",
                    ticker=f"LOAD{i % 5}",  # 5種類のティッカーを循環
                    trigger_type=TriggerType.MANUAL,
                    priority=1,
                    metadata={"load_test": True, "batch": i}
                )
                
                # 非同期で同時送信
                task = asyncio.create_task(
                    self.realtime_engine.submit_update_request(request, UpdatePriority.REALTIME)
                )
                concurrent_requests.append(task)
            
            # 全ての同時リクエスト完了待機
            load_task_ids = await asyncio.gather(*concurrent_requests, return_exceptions=True)
            
            load_test_time = time.time() - load_test_start
            successful_load_requests = sum(1 for result in load_task_ids if isinstance(result, str))
            
            results.update({
                "load_test_requests": len(concurrent_requests),
                "successful_load_requests": successful_load_requests,
                "load_test_time": load_test_time,
                "load_test_success_rate": successful_load_requests / len(concurrent_requests)
            })
            
            print(f"   ✓ Load test: {successful_load_requests}/{len(concurrent_requests)} successful in {load_test_time:.2f}s")
            
            # 4. システム状態確認
            await asyncio.sleep(3)  # 処理完了待機
            
            final_system_status = {
                "trigger_stats": self.trigger_system.get_trigger_statistics(),
                "engine_status": self.realtime_engine.get_engine_status()
            }
            
            results["final_system_status"] = final_system_status
            results["system_still_responsive"] = True
            
            print("   ✓ System remains responsive after error tests")
            
        except Exception as e:
            results["error"] = str(e)
            results["system_still_responsive"] = False
            logger.error(f"Error handling demo failed: {e}")
        
        self.demo_results["error_handling"] = results
        print("   ✅ Error handling testing completed!")
        print()
    
    async def _display_demo_summary(self):
        """結果サマリー表示"""
        print("📋 7. Demo Summary")
        print("=" * 50)
        
        # 実行時間
        demo_duration = (datetime.now() - self.demo_start_time).total_seconds()
        print(f"📅 Total demo time: {demo_duration:.2f} seconds")
        print()
        
        # 各段階の結果
        for stage, results in self.demo_results.items():
            print(f"🔍 {stage.replace('_', ' ').title()}:")
            
            if isinstance(results, dict):
                if "error" in results:
                    print(f"   ❌ Failed: {results['error']}")
                else:
                    # 主要メトリクスを表示
                    key_metrics = self._extract_key_metrics(stage, results)
                    for metric, value in key_metrics.items():
                        print(f"   ✓ {metric}: {value}")
            
            print()
        
        # 最終システム状態
        if self.trigger_system and self.realtime_engine:
            print("🎯 Final System Status:")
            
            trigger_stats = self.trigger_system.get_trigger_statistics()
            engine_status = self.realtime_engine.get_engine_status()
            
            print(f"   📊 Total triggers fired: {trigger_stats['total_triggers']}")
            print(f"   ⚡ Total updates processed: {engine_status['total_requests']}")
            print(f"   ✅ Success rate: {(engine_status['successful_updates'] / max(engine_status['total_requests'], 1) * 100):.1f}%")
            print(f"   ⏱️  Average processing time: {engine_status['average_processing_time']:.4f}s")
            print(f"   🔄 Queue size: {engine_status['queue_size']}")
            print()
        
        # 結果保存
        demo_summary = {
            "demo_completed_at": datetime.now().isoformat(),
            "demo_duration_seconds": demo_duration,
            "results": self.demo_results
        }
        
        try:
            with open("score_trigger_demo_results.json", "w") as f:
                json.dump(demo_summary, f, indent=2, default=str)
            print("💾 Demo results saved to: score_trigger_demo_results.json")
        except Exception as e:
            logger.warning(f"Failed to save demo results: {e}")
        
        print("✅ Demo summary completed!")
    
    def _extract_key_metrics(self, stage: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """段階別主要メトリクス抽出"""
        key_metrics = {}
        
        if stage == "setup":
            key_metrics["Setup Success"] = results.get("setup_success", False)
        
        elif stage == "basic_triggers":
            key_metrics["Manual Trigger ID"] = results.get("manual_trigger_id", "N/A")
            if "initial_stats" in results:
                stats = results["initial_stats"]
                key_metrics["Initial Triggers"] = stats.get("total_triggers", 0)
        
        elif stage == "threshold_triggers":
            key_metrics["Threshold Events"] = results.get("threshold_events_triggered", 0)
            key_metrics["Initial Scores"] = results.get("initial_scores_generated", 0)
            key_metrics["Change Scores"] = results.get("change_scores_generated", 0)
        
        elif stage == "batch_processing":
            key_metrics["Batch Requests"] = results.get("batch_requests_submitted", 0)
            key_metrics["Processing Time"] = f"{results.get('batch_processing_time', 0):.2f}s"
            key_metrics["Completed"] = results.get("batch_completed", False)
        
        elif stage == "performance_test":
            key_metrics["Throughput"] = f"{results.get('throughput_per_second', 0):.2f} req/s"
            key_metrics["Success Rate"] = f"{results.get('success_rate', 0)*100:.1f}%"
            key_metrics["Avg Processing"] = f"{results.get('average_processing_time', 0):.4f}s"
        
        elif stage == "error_handling":
            key_metrics["Load Test Success"] = f"{results.get('load_test_success_rate', 0)*100:.1f}%"
            key_metrics["System Responsive"] = results.get("system_still_responsive", False)
        
        return key_metrics
    
    async def _cleanup_systems(self):
        """システムクリーンアップ"""
        print("🧹 Cleaning up systems...")
        
        try:
            # リアルタイムエンジン停止
            if self.realtime_engine:
                await self.realtime_engine.stop()
                print("   ✓ Realtime engine stopped")
            
            # トリガーシステム停止
            if self.trigger_system:
                self.trigger_system.stop()
                print("   ✓ Trigger system stopped")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        print("   ✅ Cleanup completed!")


# =============================================================================
# メイン実行
# =============================================================================

async def main():
    """メインデモ実行"""
    demo = TriggerSystemIntegrationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # デモ実行
    asyncio.run(main())
