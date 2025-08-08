"""
フェーズ3B 監視ダッシュボード テストスクリプト

このスクリプトは、監視ダッシュボードの機能をテストし、
統合的な動作確認を行います。
"""

import asyncio
import time
import threading
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# プロジェクト内インポート
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.logger_config import setup_logger
from src.monitoring.dashboard import MonitoringDashboard, DashboardConfig, create_dashboard
from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.alert_manager import AlertManager, AlertRule, AlertLevel, AlertCategory, NotificationChannel
from src.data.data_feed_integration import IntegratedDataFeedSystem, DataQualityMetrics, DataQualityLevel


class DashboardTestSuite:
    """ダッシュボードテストスイート"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.test_results = []
        
        # テスト用コンポーネント
        self.data_feed_system = None
        self.dashboard = None
        self.metrics_collector = None
        self.alert_manager = None
        
    def run_all_tests(self):
        """全テスト実行"""
        self.logger.info("Starting dashboard test suite...")
        
        try:
            # 1. 基本コンポーネントテスト
            self.test_component_initialization()
            
            # 2. メトリクス収集テスト
            self.test_metrics_collection()
            
            # 3. アラート管理テスト
            self.test_alert_management()
            
            # 4. ダッシュボード統合テスト
            self.test_dashboard_integration()
            
            # 5. エラーハンドリングテスト
            self.test_error_handling()
            
            # 6. パフォーマンステスト
            self.test_performance()
            
            # テスト結果レポート
            self.generate_test_report()
            
        except Exception as e:
            self.logger.error(f"Test suite execution failed: {e}")
            self.test_results.append({
                'test_name': 'test_suite_execution',
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            # クリーンアップ
            self.cleanup_test_environment()
            
    def test_component_initialization(self):
        """コンポーネント初期化テスト"""
        self.logger.info("Testing component initialization...")
        
        try:
            # データフィードシステム初期化
            self.data_feed_system = IntegratedDataFeedSystem()
            self._assert_not_none(self.data_feed_system, "Data feed system initialization")
            
            # メトリクス収集器初期化
            self.metrics_collector = MetricsCollector()
            self._assert_not_none(self.metrics_collector, "Metrics collector initialization")
            
            # アラート管理器初期化
            self.alert_manager = AlertManager()
            self._assert_not_none(self.alert_manager, "Alert manager initialization")
            
            # ダッシュボード設定
            dashboard_config = DashboardConfig(
                host="localhost",
                port=8081,  # テスト用ポート
                auto_refresh_interval=2,
                enable_real_time_updates=True
            )
            
            # ダッシュボード初期化
            self.dashboard = create_dashboard(self.data_feed_system, dashboard_config)
            self._assert_not_none(self.dashboard, "Dashboard initialization")
            
            self._record_test_result("component_initialization", "PASSED")
            
        except Exception as e:
            self._record_test_result("component_initialization", "FAILED", str(e))
            raise
            
    def test_metrics_collection(self):
        """メトリクス収集テスト"""
        self.logger.info("Testing metrics collection...")
        
        try:
            # サンプルメトリクス記録
            self.metrics_collector.record_performance_metrics(
                "test_operation", 100.5, True
            )
            
            self.metrics_collector.record_network_metrics(
                "test_source", 10, 50.0, 0.95, 1, 0
            )
            
            self.metrics_collector.record_cache_metrics(
                "memory", 800, 200, 1024*1024*10, 1000, 5
            )
            
            # メトリクス集計確認
            summary = self.metrics_collector.get_all_metrics_summary(5)
            self._assert_not_none(summary, "Metrics summary generation")
            self._assert_true('performance' in summary, "Performance metrics in summary")
            self._assert_true('network' in summary, "Network metrics in summary")
            
            self._record_test_result("metrics_collection", "PASSED")
            
        except Exception as e:
            self._record_test_result("metrics_collection", "FAILED", str(e))
            
    def test_alert_management(self):
        """アラート管理テスト"""
        self.logger.info("Testing alert management...")
        
        try:
            # アラートルール追加
            test_rule = AlertRule(
                rule_id="test_rule",
                name="テストルール",
                category=AlertCategory.DATA_QUALITY,
                level=AlertLevel.WARNING,
                condition="test_value > 0.5",
                threshold=0.5,
                time_window_minutes=5,
                description="テスト用のアラートルール"
            )
            
            self.alert_manager.add_alert_rule(test_rule)
            self._assert_true("test_rule" in self.alert_manager.alert_rules, "Alert rule addition")
            
            # テスト用メトリクスでアラート評価
            test_metrics = {
                'quality': {
                    'test_value': 0.8,  # 閾値超過でアラートトリガー
                    'overall_score': 0.9
                }
            }
            
            self.alert_manager.evaluate_metrics(test_metrics)
            
            # アラート確認
            active_alerts = self.alert_manager.get_active_alerts()
            self.logger.info(f"Generated {len(active_alerts)} test alerts")
            
            # アラート統計確認
            stats = self.alert_manager.get_alert_stats()
            self._assert_not_none(stats, "Alert statistics generation")
            
            self._record_test_result("alert_management", "PASSED")
            
        except Exception as e:
            self._record_test_result("alert_management", "FAILED", str(e))
            
    def test_dashboard_integration(self):
        """ダッシュボード統合テスト"""
        self.logger.info("Testing dashboard integration...")
        
        try:
            # システムメトリクス更新テスト
            self.dashboard._update_system_metrics()
            self._assert_not_none(self.dashboard.current_metrics, "System metrics update")
            
            # 品質履歴更新テスト
            self.dashboard._update_quality_history()
            
            # アラート処理テスト
            self.dashboard._process_alerts()
            
            # API エンドポイントテスト（モック）
            self._test_api_endpoints()
            
            self._record_test_result("dashboard_integration", "PASSED")
            
        except Exception as e:
            self._record_test_result("dashboard_integration", "FAILED", str(e))
            
    def _test_api_endpoints(self):
        """APIエンドポイントテスト（モック）"""
        # メトリクス取得テスト
        if self.dashboard.current_metrics:
            metrics_dict = self.dashboard.current_metrics.to_dict()
            self._assert_not_none(metrics_dict, "Metrics API response")
            self._assert_true('timestamp' in metrics_dict, "Timestamp in metrics")
            
        # システム状態テスト
        uptime = self.dashboard._get_uptime()
        self._assert_not_none(uptime, "Uptime calculation")
        
        cache_status = self.dashboard._get_cache_status()
        self._assert_not_none(cache_status, "Cache status")
        
        error_summary = self.dashboard._get_error_summary()
        self._assert_not_none(error_summary, "Error summary")
        
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        self.logger.info("Testing error handling...")
        
        try:
            # 不正データでのメトリクス記録テスト
            try:
                self.metrics_collector.record_performance_metrics(
                    "", -1.0, True  # 不正なデータ
                )
                # エラーが発生しても継続することを確認
            except Exception:
                pass  # 期待されるエラー
                
            # 不正な条件でのアラート評価テスト
            try:
                invalid_metrics = {'invalid_key': 'invalid_value'}
                self.alert_manager.evaluate_metrics(invalid_metrics)
                # エラーが発生しても継続することを確認
            except Exception:
                pass  # 期待されるエラー
                
            # 存在しないアラートIDでの操作テスト
            result = self.alert_manager.acknowledge_alert("nonexistent_id", "test_user")
            self._assert_false(result, "Nonexistent alert acknowledgment")
            
            self._record_test_result("error_handling", "PASSED")
            
        except Exception as e:
            self._record_test_result("error_handling", "FAILED", str(e))
            
    def test_performance(self):
        """パフォーマンステスト"""
        self.logger.info("Testing performance...")
        
        try:
            # 大量メトリクス記録テスト
            start_time = time.time()
            
            for i in range(1000):
                self.metrics_collector.record_performance_metrics(
                    f"operation_{i}", float(i), i % 10 != 0  # 10%のエラー率
                )
                
            collection_time = time.time() - start_time
            self.logger.info(f"1000 metrics collection time: {collection_time:.2f}s")
            
            # 集計パフォーマンステスト
            start_time = time.time()
            summary = self.metrics_collector.get_all_metrics_summary(60)
            aggregation_time = time.time() - start_time
            self.logger.info(f"Metrics aggregation time: {aggregation_time:.2f}s")
            
            # パフォーマンス閾値チェック
            self._assert_true(collection_time < 5.0, "Metrics collection performance")
            self._assert_true(aggregation_time < 2.0, "Metrics aggregation performance")
            
            self._record_test_result("performance", "PASSED")
            
        except Exception as e:
            self._record_test_result("performance", "FAILED", str(e))
            
    def test_dashboard_startup(self):
        """ダッシュボード起動テスト（バックグラウンド）"""
        self.logger.info("Testing dashboard startup...")
        
        try:
            # バックグラウンドでダッシュボード開始
            dashboard_thread = threading.Thread(
                target=self._start_dashboard_background,
                daemon=True
            )
            dashboard_thread.start()
            
            # 少し待機してから状態確認
            time.sleep(3)
            
            # ダッシュボードエージェントが動作していることを確認
            self._assert_not_none(self.dashboard.agent, "Dashboard agent exists")
            
            # エージェント停止
            if self.dashboard.agent.is_running:
                self.dashboard.agent.stop()
                
            self._record_test_result("dashboard_startup", "PASSED")
            
        except Exception as e:
            self._record_test_result("dashboard_startup", "FAILED", str(e))
            
    def _start_dashboard_background(self):
        """バックグラウンドでダッシュボード開始"""
        try:
            # エージェントのみ開始（Webサーバーは開始しない）
            self.dashboard.agent.start()
            time.sleep(2)  # 短時間動作
            self.dashboard.agent.stop()
        except Exception as e:
            self.logger.error(f"Dashboard background start error: {e}")
            
    def generate_test_report(self):
        """テストレポート生成"""
        self.logger.info("Generating test report...")
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASSED'])
        failed_tests = total_tests - passed_tests
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%",
                'test_date': datetime.now().isoformat()
            },
            'test_results': self.test_results,
            'system_info': {
                'python_version': sys.version,
                'test_duration': self._calculate_test_duration()
            }
        }
        
        # レポートファイル保存
        report_path = Path(__file__).parent / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # コンソール出力
        self.logger.info("=" * 60)
        self.logger.info("DASHBOARD TEST REPORT")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {failed_tests}")
        self.logger.info(f"Success Rate: {report['test_summary']['success_rate']}")
        self.logger.info(f"Report saved to: {report_path}")
        self.logger.info("=" * 60)
        
        # 失敗したテストの詳細
        if failed_tests > 0:
            self.logger.error("FAILED TESTS:")
            for result in self.test_results:
                if result['status'] == 'FAILED':
                    self.logger.error(f"- {result['test_name']}: {result.get('error', 'Unknown error')}")
                    
    def _calculate_test_duration(self):
        """テスト実行時間計算"""
        if not self.test_results:
            return "0s"
            
        timestamps = [
            datetime.fromisoformat(r['timestamp']) 
            for r in self.test_results if 'timestamp' in r
        ]
        
        if len(timestamps) < 2:
            return "Unknown"
            
        duration = max(timestamps) - min(timestamps)
        return f"{duration.total_seconds():.1f}s"
        
    def cleanup_test_environment(self):
        """テスト環境クリーンアップ"""
        self.logger.info("Cleaning up test environment...")
        
        try:
            # エージェント停止
            if self.dashboard and self.dashboard.agent.is_running:
                self.dashboard.agent.stop()
                
            # アラート監視停止
            if self.alert_manager and self.alert_manager.is_running:
                self.alert_manager.stop_monitoring()
                
            self.logger.info("Test environment cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def _record_test_result(self, test_name: str, status: str, error: str = None):
        """テスト結果記録"""
        result = {
            'test_name': test_name,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        if error:
            result['error'] = error
            
        self.test_results.append(result)
        
        log_msg = f"Test '{test_name}': {status}"
        if error:
            log_msg += f" - {error}"
            
        if status == "PASSED":
            self.logger.info(log_msg)
        else:
            self.logger.error(log_msg)
            
    def _assert_true(self, condition: bool, message: str):
        """アサーション: True"""
        if not condition:
            raise AssertionError(f"Assertion failed: {message}")
            
    def _assert_false(self, condition: bool, message: str):
        """アサーション: False"""
        if condition:
            raise AssertionError(f"Assertion failed: {message}")
            
    def _assert_not_none(self, value, message: str):
        """アサーション: Not None"""
        if value is None:
            raise AssertionError(f"Assertion failed: {message} - Value is None")
            
    def _assert_equal(self, expected, actual, message: str):
        """アサーション: Equal"""
        if expected != actual:
            raise AssertionError(f"Assertion failed: {message} - Expected {expected}, got {actual}")


def main():
    """メイン実行関数"""
    print("フェーズ3B 監視ダッシュボード テストスイート")
    print("=" * 60)
    
    test_suite = DashboardTestSuite()
    
    try:
        test_suite.run_all_tests()
        print("\nテスト実行完了!")
        
    except KeyboardInterrupt:
        print("\nテストが中断されました")
        test_suite.cleanup_test_environment()
        
    except Exception as e:
        print(f"\nテスト実行中にエラーが発生しました: {e}")
        test_suite.cleanup_test_environment()


if __name__ == "__main__":
    main()
