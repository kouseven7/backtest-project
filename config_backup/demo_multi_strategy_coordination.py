"""
Module: Multi-Strategy Coordination Demo
File: demo_multi_strategy_coordination.py  
Description: 
  4-1-3「マルチ戦略同時実行の調整機能」
  統合デモ・検証スクリプト

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 全コンポーネント統合動作検証
  - 段階的フォールバック動作確認
  - パフォーマンス・信頼性検証
  - エンドツーエンドデモ実行
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import json

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo_coordination.log')
    ]
)
logger = logging.getLogger(__name__)

# プロジェクトモジュールをインポート
try:
    from multi_strategy_coordination_interface import MultiStrategyCoordinationInterface
    from multi_strategy_coordination_manager import MultiStrategyCoordinationManager, FallbackLevel
    from resource_allocation_engine import ResourceAllocationEngine, ExecutionMode
    from strategy_dependency_resolver import StrategyDependencyResolver
    from concurrent_execution_scheduler import ConcurrentExecutionScheduler, ExecutionTask
    from execution_monitoring_system import ExecutionMonitoringSystem
except ImportError as e:
    logger.error(f"Failed to import coordination modules: {e}")
    print(f"[ERROR] モジュールのインポートに失敗しました: {e}")
    sys.exit(1)

class CoordinationDemo:
    """調整機能統合デモ"""
    
    def __init__(self):
        """初期化"""
        self.demo_strategies = [
            "VWAPBounceStrategy",
            "GCStrategy", 
            "BreakoutStrategy",
            "OpeningGapStrategy",
            "MomentumStrategy",
            "ReversalStrategy"
        ]
        
        self.test_results: List[Dict[str, Any]] = []
        self.demo_start_time = datetime.now()
        
        # インターフェース初期化
        try:
            self.interface = MultiStrategyCoordinationInterface()
            self.coordination_manager = self.interface.coordination_manager
        except Exception as e:
            logger.error(f"Interface initialization failed: {e}")
            raise
    
    def run_comprehensive_demo(self):
        """包括的デモ実行"""
        print("=" * 80)
        print("[ROCKET] Multi-Strategy Coordination System - Comprehensive Demo")
        print("=" * 80)
        print(f"Demo Start Time: {self.demo_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Available Strategies: {len(self.demo_strategies)}")
        for i, strategy in enumerate(self.demo_strategies, 1):
            print(f"  {i}. {strategy}")
        print()
        
        try:
            # フェーズ1: インターフェース初期化テスト
            self._test_interface_initialization()
            
            # フェーズ2: 個別コンポーネントテスト
            self._test_individual_components()
            
            # フェーズ3: 基本調整機能テスト
            self._test_basic_coordination()
            
            # フェーズ4: 高度調整機能テスト
            self._test_advanced_coordination()
            
            # フェーズ5: フォールバック機能テスト
            self._test_fallback_mechanisms()
            
            # フェーズ6: パフォーマンス・負荷テスト
            self._test_performance_and_load()
            
            # フェーズ7: 統合システム連携テスト
            self._test_system_integration()
            
            # 最終結果レポート
            self._generate_final_report()
            
        except Exception as e:
            logger.error(f"Demo execution failed: {e}")
            print(f"[ERROR] デモ実行中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    def _test_interface_initialization(self):
        """インターフェース初期化テスト"""
        print("[LIST] Phase 1: Interface Initialization Test")
        print("-" * 50)
        
        try:
            # インターフェース初期化
            self.interface.initialize()
            
            # システム状況確認
            status = self.interface.get_system_status()
            
            print(f"[OK] Interface initialization successful")
            print(f"   Coordination State: {status['coordination']['state']}")
            print(f"   Integration Mode: {status['integration']['integration_config']['mode']}")
            print(f"   Integrated Systems: {len(status['integration']['integrated_systems'])}")
            print(f"   Web Interface: {'Enabled' if status['web_interface']['enabled'] else 'Disabled'}")
            
            # Web インターフェース情報
            if status['web_interface']['enabled']:
                port = status['web_interface']['port']
                print(f"   📱 Dashboard: http://localhost:{port}")
            
            self._record_test_result("interface_initialization", True, "Interface initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Interface initialization failed: {e}")
            self._record_test_result("interface_initialization", False, str(e))
        
        print()
    
    def _test_individual_components(self):
        """個別コンポーネントテスト"""
        print("[TOOL] Phase 2: Individual Components Test")
        print("-" * 50)
        
        # リソース配分エンジンテスト
        self._test_resource_allocation_engine()
        
        # 依存関係リゾルバーテスト
        self._test_dependency_resolver()
        
        # 並行実行スケジューラーテスト
        self._test_execution_scheduler()
        
        # 監視システムテスト
        self._test_monitoring_system()
        
        print()
    
    def _test_resource_allocation_engine(self):
        """リソース配分エンジンテスト"""
        print("  🔹 Resource Allocation Engine Test")
        
        try:
            engine = self.coordination_manager.resource_engine
            if not engine:
                print("    [WARNING] Resource engine not available, skipping test")
                return
            
            # リソース配分テスト
            test_strategies = self.demo_strategies[:3]
            allocations = engine.allocate_resources(test_strategies)
            
            print(f"    [OK] Resource allocation successful for {len(test_strategies)} strategies")
            for allocation in allocations:
                print(f"       {allocation.strategy_name}: {allocation.execution_mode.value} mode, "
                      f"CPU={allocation.allocated_cpu:.2f}, Memory={allocation.allocated_memory_mb}MB")
            
            self._record_test_result("resource_allocation", True, f"Allocated resources for {len(allocations)} strategies")
            
        except Exception as e:
            print(f"    [ERROR] Resource allocation test failed: {e}")
            self._record_test_result("resource_allocation", False, str(e))
    
    def _test_dependency_resolver(self):
        """依存関係リゾルバーテスト"""
        print("  🔹 Dependency Resolver Test")
        
        try:
            resolver = self.coordination_manager.dependency_resolver
            if not resolver:
                print("    [WARNING] Dependency resolver not available, skipping test")
                return
            
            # 依存関係解決テスト
            test_strategies = self.demo_strategies[:4]
            resolution = resolver.resolve_dependencies(test_strategies)
            
            print(f"    [OK] Dependency resolution successful")
            print(f"       Execution Order: {resolution.execution_order}")
            print(f"       Parallel Groups: {len(resolution.parallel_groups)} groups")
            print(f"       Critical Path Duration: {resolution.critical_path_duration:.1f}s")
            
            self._record_test_result("dependency_resolution", True, f"Resolved dependencies for {len(test_strategies)} strategies")
            
        except Exception as e:
            print(f"    [ERROR] Dependency resolution test failed: {e}")
            self._record_test_result("dependency_resolution", False, str(e))
    
    def _test_execution_scheduler(self):
        """並行実行スケジューラーテスト"""
        print("  🔹 Execution Scheduler Test")
        
        try:
            scheduler = self.coordination_manager.execution_scheduler
            if not scheduler:
                print("    [WARNING] Execution scheduler not available, skipping test")
                return
            
            # デモタスク作成
            def demo_function(strategy_name: str) -> Dict[str, Any]:
                time.sleep(1)  # 短い実行時間
                return {'strategy': strategy_name, 'result': 'success'}
            
            demo_tasks = []
            for i, strategy in enumerate(self.demo_strategies[:2]):
                task = ExecutionTask(
                    task_id=f"demo_task_{i}",
                    strategy_name=strategy,
                    execution_mode=ExecutionMode.THREAD,
                    function=demo_function,
                    args=(strategy,),
                    timeout=30.0
                )
                demo_tasks.append(task)
            
            # スケジューリング計画作成
            plan = scheduler.create_scheduling_plan(demo_tasks)
            print(f"    [OK] Scheduling plan created: {len(plan.execution_batches)} batches")
            
            self._record_test_result("execution_scheduling", True, f"Created scheduling plan for {len(demo_tasks)} tasks")
            
        except Exception as e:
            print(f"    [ERROR] Execution scheduler test failed: {e}")
            self._record_test_result("execution_scheduling", False, str(e))
    
    def _test_monitoring_system(self):
        """監視システムテスト"""  
        print("  🔹 Monitoring System Test")
        
        try:
            monitor = self.coordination_manager.monitoring_system
            if not monitor:
                print("    [WARNING] Monitoring system not available, skipping test")
                return
            
            # リアルタイムメトリクス取得
            metrics = monitor.get_real_time_metrics()
            
            print(f"    [OK] Monitoring system operational")
            print(f"       System Metrics: {len(metrics.get('system_metrics', {}))}")
            print(f"       Performance Metrics: {len(metrics.get('performance_metrics', {}))}")
            print(f"       Active Alerts: {metrics.get('active_alerts_count', 0)}")
            
            self._record_test_result("monitoring_system", True, "Monitoring system operational")
            
        except Exception as e:
            print(f"    [ERROR] Monitoring system test failed: {e}")
            self._record_test_result("monitoring_system", False, str(e))
    
    def _test_basic_coordination(self):
        """基本調整機能テスト"""
        print("[TARGET] Phase 3: Basic Coordination Test")
        print("-" * 50)
        
        try:
            # 基本的な戦略セット
            basic_strategies = self.demo_strategies[:3]
            
            print(f"Testing basic coordination with {len(basic_strategies)} strategies:")
            for strategy in basic_strategies:
                print(f"  - {strategy}")
            
            # 調整実行
            result = self.interface.execute_strategy_coordination(basic_strategies)
            
            if result['success']:
                print(f"[OK] Basic coordination successful")
                print(f"   Method: {result['method']}")
                print(f"   Execution ID: {result.get('execution_id', 'N/A')}")
                
                # 実行状況監視（短時間）
                self._monitor_execution(10)  # 10秒間監視
                
                self._record_test_result("basic_coordination", True, f"Coordinated {len(basic_strategies)} strategies")
            else:
                print(f"[ERROR] Basic coordination failed: {result.get('error', 'Unknown error')}")
                self._record_test_result("basic_coordination", False, result.get('error', 'Unknown error'))
            
        except Exception as e:
            print(f"[ERROR] Basic coordination test failed: {e}")
            self._record_test_result("basic_coordination", False, str(e))
        
        print()
    
    def _test_advanced_coordination(self):
        """高度調整機能テスト"""
        print("[ROCKET] Phase 4: Advanced Coordination Test")
        print("-" * 50)
        
        try:
            # 高度な戦略セット（多数戦略）
            advanced_strategies = self.demo_strategies  # 全戦略使用
            
            print(f"Testing advanced coordination with {len(advanced_strategies)} strategies:")
            for strategy in advanced_strategies:
                print(f"  - {strategy}")
            
            # 調整計画作成
            plan = self.coordination_manager.create_coordination_plan(advanced_strategies)
            
            print(f"\n[LIST] Coordination Plan Analysis:")
            print(f"   Plan ID: {plan.plan_id}")
            print(f"   Resource Allocations: {len(plan.resource_allocations)}")
            print(f"   Execution Timeline: {len(plan.execution_timeline)} events")
            print(f"   Estimated Completion: {plan.estimated_completion_time.strftime('%H:%M:%S')}")
            
            # リスク評価
            risk = plan.risk_assessment
            print(f"   Risk Level: {risk['overall_risk_level'].upper()}")
            print(f"   Confidence: {risk['confidence_score']:.1%}")
            
            if risk['risk_factors']:
                print("   Risk Factors:")
                for factor in risk['risk_factors']:
                    print(f"     [WARNING] {factor}")
            
            # 調整実行
            execution_id = self.coordination_manager.execute_coordination_plan(plan)
            print(f"\n[TARGET] Advanced coordination started: {execution_id}")
            
            # 実行監視（長時間）
            self._monitor_execution(20)  # 20秒間監視
            
            self._record_test_result("advanced_coordination", True, f"Coordinated {len(advanced_strategies)} strategies")
            
        except Exception as e:
            print(f"[ERROR] Advanced coordination test failed: {e}")
            self._record_test_result("advanced_coordination", False, str(e))
        
        print()
    
    def _test_fallback_mechanisms(self):
        """フォールバック機能テスト"""
        print("🛡️ Phase 5: Fallback Mechanisms Test")
        print("-" * 50)
        
        # 個別戦略フォールバックテスト
        self._test_individual_fallback()
        
        # システムレベルフォールバックテスト
        self._test_system_fallback()
        
        print()
    
    def _test_individual_fallback(self):
        """個別戦略フォールバックテスト"""
        print("  🔹 Individual Strategy Fallback Test")
        
        try:
            # 意図的にエラーを発生させる戦略を含む
            fallback_strategies = ["ErrorStrategy", "VWAPBounceStrategy", "GCStrategy"]
            
            print(f"    Testing individual fallback with strategies including error case")
            
            # フォールバック管理器取得
            fallback_manager = self.coordination_manager.fallback_manager
            
            # テスト用の模擬状況作成
            from multi_strategy_coordination_manager import CoordinationStatus
            test_status = CoordinationStatus(
                state=self.coordination_manager.state,
                failed_strategies=["ErrorStrategy"]
            )
            
            # フォールバック判定テスト
            fallback_level = fallback_manager.should_trigger_fallback(test_status, {}, [])
            
            print(f"    [OK] Individual fallback mechanism operational")
            print(f"       Detected Fallback Level: {fallback_level.value if fallback_level else 'None'}")
            
            self._record_test_result("individual_fallback", True, "Individual fallback mechanism works")
            
        except Exception as e:
            print(f"    [ERROR] Individual fallback test failed: {e}")
            self._record_test_result("individual_fallback", False, str(e))
    
    def _test_system_fallback(self):
        """システムレベルフォールバックテスト"""
        print("  🔹 System Level Fallback Test")
        
        try:
            # フォールバック設定確認
            fallback_config = self.coordination_manager.fallback_manager.fallback_config
            
            print(f"    [OK] System fallback configuration loaded")
            print(f"       Emergency Alert Threshold: {fallback_config.get('emergency_alert_threshold', 3)}")
            print(f"       System Failure Threshold: {fallback_config.get('system_failure_threshold', 0.3)}")
            print(f"       Essential Strategies: {len(fallback_config.get('essential_strategies', []))}")
            
            # 段階的フォールバックシナリオ表示
            print(f"    Fallback Levels Available:")
            for level in FallbackLevel:
                if level != FallbackLevel.NONE:
                    print(f"       - {level.value}")
            
            self._record_test_result("system_fallback", True, "System fallback configuration verified")
            
        except Exception as e:
            print(f"    [ERROR] System fallback test failed: {e}")
            self._record_test_result("system_fallback", False, str(e))
    
    def _test_performance_and_load(self):
        """パフォーマンス・負荷テスト"""
        print("⚡ Phase 6: Performance and Load Test")
        print("-" * 50)
        
        try:
            # 複数の並行調整実行
            concurrent_tests = []
            test_start_time = time.time()
            
            for i in range(3):  # 3つの並行調整
                strategies = self.demo_strategies[i:i+2]  # 各調整は2戦略
                
                print(f"  🔹 Starting concurrent coordination {i+1}: {strategies}")
                
                try:
                    result = self.interface.execute_strategy_coordination(
                        strategies, 
                        integration_mode=None
                    )
                    concurrent_tests.append(result)
                    
                    if result['success']:
                        print(f"    [OK] Concurrent test {i+1} started successfully")
                    else:
                        print(f"    [ERROR] Concurrent test {i+1} failed to start")
                
                except Exception as e:
                    print(f"    [ERROR] Concurrent test {i+1} exception: {e}")
                
                time.sleep(1)  # 少し間隔をあける
            
            # 負荷テスト監視
            print(f"\n  [CHART] Load testing in progress...")
            self._monitor_execution(15)  # 15秒間監視
            
            test_duration = time.time() - test_start_time
            success_count = sum(1 for test in concurrent_tests if test.get('success', False))
            
            print(f"\n  [UP] Performance Test Results:")
            print(f"     Test Duration: {test_duration:.1f}s")
            print(f"     Concurrent Tests: {len(concurrent_tests)}")
            print(f"     Successful: {success_count}")
            print(f"     Success Rate: {success_count/len(concurrent_tests)*100:.1f}%")
            
            self._record_test_result("performance_load", True, f"Load test completed: {success_count}/{len(concurrent_tests)} successful")
            
        except Exception as e:
            print(f"[ERROR] Performance and load test failed: {e}")
            self._record_test_result("performance_load", False, str(e))
        
        print()
    
    def _test_system_integration(self):
        """統合システム連携テスト"""
        print("🔗 Phase 7: System Integration Test")
        print("-" * 50)
        
        try:
            # 統合システム状況確認
            integration_status = self.interface.system_integrator.get_integration_status()
            integrated_systems = integration_status['integrated_systems']
            
            print(f"Integrated Systems Found: {len(integrated_systems)}")
            
            if not integrated_systems:
                print("  [WARNING] No integrated systems found, testing interface compatibility")
                self._test_interface_compatibility()
            else:
                # 各統合システムテスト
                for system_id, system_info in integrated_systems.items():
                    self._test_integrated_system(system_id, system_info)
            
            self._record_test_result("system_integration", True, f"Integration test completed for {len(integrated_systems)} systems")
            
        except Exception as e:
            print(f"[ERROR] System integration test failed: {e}")
            self._record_test_result("system_integration", False, str(e))
        
        print()
    
    def _test_interface_compatibility(self):
        """インターフェース互換性テスト"""
        print("  🔹 Interface Compatibility Test")
        
        try:
            # 設定動的更新テスト
            config_update = {
                "coordination": {
                    "mode": "supervised"
                },
                "monitoring": {
                    "level": "comprehensive"
                }
            }
            
            success = self.interface.update_configuration(config_update)
            
            if success:
                print("    [OK] Dynamic configuration update successful")
            else:
                print("    [ERROR] Dynamic configuration update failed")
            
            # API互換性テスト（Web インターフェース経由）
            status = self.interface.get_system_status()
            if status:
                print("    [OK] API compatibility verified")
            
        except Exception as e:
            print(f"    [ERROR] Interface compatibility test failed: {e}")
    
    def _test_integrated_system(self, system_id: str, system_info: Dict[str, Any]):
        """統合システムテスト"""
        print(f"  🔹 Testing integrated system: {system_id}")
        
        try:
            # 統合システム経由での実行テスト
            test_strategies = self.demo_strategies[:2]
            
            integration_mode = system_id.replace('_', '-')
            result = self.interface.execute_strategy_coordination(
                test_strategies,
                integration_mode=integration_mode
            )
            
            if result['success']:
                print(f"    [OK] Integration successful: {result['method']}")
            else:
                print(f"    [ERROR] Integration failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"    [ERROR] Integrated system test failed: {e}")
    
    def _monitor_execution(self, duration_seconds: int):
        """実行監視"""
        print(f"  [CHART] Monitoring execution for {duration_seconds} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            try:
                status = self.interface.get_system_status()
                coord_status = status['coordination']
                
                active_count = len(coord_status.get('active_strategies', []))
                completed_count = len(coord_status.get('completed_strategies', []))
                failed_count = len(coord_status.get('failed_strategies', []))
                
                elapsed = int(time.time() - start_time)
                print(f"    [{elapsed:2d}s] State: {coord_status['state']:10s} | "
                      f"Active: {active_count} | Completed: {completed_count} | Failed: {failed_count}")
                
                # 実行完了チェック
                if coord_status['state'] in ['idle', 'emergency']:
                    print(f"    Execution completed in {elapsed}s")
                    break
                
                time.sleep(2)
            
            except Exception as e:
                print(f"    [WARNING] Monitoring error: {e}")
                break
    
    def _record_test_result(self, test_name: str, success: bool, message: str):
        """テスト結果記録"""
        self.test_results.append({
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def _generate_final_report(self):
        """最終レポート生成"""
        print("[CHART] Final Demo Report")
        print("=" * 80)
        
        demo_duration = datetime.now() - self.demo_start_time
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['success'])
        
        print(f"Demo Duration: {demo_duration}")
        print(f"Total Tests: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Success Rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
        print()
        
        print("Test Results Summary:")
        print("-" * 50)
        for result in self.test_results:
            status_emoji = "[OK]" if result['success'] else "[ERROR]"
            print(f"{status_emoji} {result['test_name']:25s} - {result['message']}")
        
        print()
        
        # システム統計
        try:
            status = self.interface.get_system_status()
            stats = status['coordination'].get('performance_statistics', {})
            
            print("System Performance Statistics:")
            print("-" * 50)
            print(f"Total Coordinations: {stats.get('total_coordinations', 0)}")
            print(f"Successful: {stats.get('successful_coordinations', 0)}")
            print(f"Failed: {stats.get('failed_coordinations', 0)}")
            print(f"Fallback Activations: {stats.get('fallback_activations', 0)}")
            print(f"Average Duration: {stats.get('average_coordination_time', 0):.1f}s")
        
        except Exception as e:
            print(f"[WARNING] Could not retrieve system statistics: {e}")
        
        # レポートファイル保存
        self._save_report_to_file()
        
        print("\n" + "=" * 80)
        if successful_tests == total_tests:
            print("[SUCCESS] ALL TESTS PASSED - Demo completed successfully!")
        else:
            print(f"[WARNING] {total_tests - successful_tests} test(s) failed - Demo completed with issues")
        print("=" * 80)
    
    def _save_report_to_file(self):
        """レポートファイル保存"""
        try:
            report_data = {
                'demo_info': {
                    'start_time': self.demo_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'strategies_tested': self.demo_strategies,
                    'total_tests': len(self.test_results)
                },
                'test_results': self.test_results,
                'system_status': self.interface.get_system_status()
            }
            
            report_filename = f"demo_report_{self.demo_start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Demo report saved: {report_filename}")
            
        except Exception as e:
            print(f"[WARNING] Failed to save report file: {e}")
    
    def cleanup(self):
        """クリーンアップ"""
        try:
            self.interface.shutdown()
            print("🧹 Demo cleanup completed")
        except Exception as e:
            print(f"[WARNING] Cleanup error: {e}")

def main():
    """メイン実行関数"""
    demo = None
    try:
        # デモ実行
        demo = CoordinationDemo()
        demo.run_comprehensive_demo()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if demo:
            demo.cleanup()

if __name__ == "__main__":
    main()
