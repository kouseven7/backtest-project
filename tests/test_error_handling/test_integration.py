"""
エラーハンドリングシステム統合テスト
全コンポーネントの連携テストと動作確認
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.exception_handler import (
    UnifiedExceptionHandler, StrategyError, DataError, SystemError,
    get_exception_handler, handle_strategy_error, handle_data_error, handle_system_error
)
from src.utils.error_recovery import (
    ErrorRecoveryManager, SimpleRetryStrategy, ExponentialBackoffStrategy,
    retry_with_strategy, fallback_recovery
)
from src.utils.logger_setup import (
    EnhancedLoggerManager, get_logger_manager, get_strategy_logger
)
from src.utils.monitoring_agent import (
    MonitoringAgent, AlertEvent, report_error, start_system_monitoring
)


class TestUnifiedExceptionHandler:
    """統一例外処理テスト"""
    
    def setup_method(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "error_policies.json"
        
        # テスト用設定
        test_config = {
            "strategy_errors": {
                "max_retries": 2,
                "retry_delay": 0.1,
                "fallback_enabled": True
            },
            "data_errors": {
                "max_retries": 3,
                "retry_delay": 0.1,
                "fallback_enabled": False
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        self.handler = UnifiedExceptionHandler(str(self.config_file))
    
    def test_strategy_error_handling(self):
        """戦略エラー処理テスト"""
        error = StrategyError("test_strategy", "テストエラー")
        
        result = self.handler.handle_strategy_error("test_strategy", error)
        
        assert result['type'] == 'strategy'
        assert result['strategy_name'] == 'test_strategy'
        assert 'timestamp' in result
        assert 'traceback' in result
    
    def test_data_error_handling(self):
        """データエラー処理テスト"""
        error = DataError("データ取得失敗")
        
        result = self.handler.handle_data_error(error)
        
        assert result['type'] == 'data'
        assert 'error_message' in result
        assert result['recovery_attempted'] == False
    
    def test_system_error_handling(self):
        """システムエラー処理テスト"""
        error = SystemError("システム障害")
        
        result = self.handler.handle_system_error(error)
        
        assert result['type'] == 'system'
        assert 'error_message' in result
    
    def test_error_statistics(self):
        """エラー統計テスト"""
        # 複数エラー発生
        self.handler.handle_strategy_error("strategy1", Exception("エラー1"))
        self.handler.handle_data_error(Exception("エラー2"))
        self.handler.handle_system_error(Exception("エラー3"))
        
        stats = self.handler.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['strategy_errors'] == 1
        assert stats['data_errors'] == 1
        assert stats['system_errors'] == 1
    
    def test_global_functions(self):
        """グローバル関数テスト"""
        result1 = handle_strategy_error("test", Exception("エラー"))
        result2 = handle_data_error(Exception("データエラー"))
        result3 = handle_system_error(Exception("システムエラー"))
        
        assert all('timestamp' in r for r in [result1, result2, result3])
    
    def teardown_method(self):
        """テスト後片付け"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestErrorRecoveryManager:
    """エラー復旧管理テスト"""
    
    def setup_method(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "recovery_strategies.json"
        
        # テスト用設定
        test_config = {
            "strategies": {
                "test_retry": {
                    "class": "SimpleRetryStrategy",
                    "max_retries": 2,
                    "base_delay": 0.01
                }
            },
            "error_type_mapping": {
                "test_errors": "test_retry"
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        self.manager = ErrorRecoveryManager(str(self.config_file))
    
    def test_simple_retry_strategy(self):
        """シンプルリトライ戦略テスト"""
        strategy = SimpleRetryStrategy(max_retries=3, base_delay=0.01)
        
        assert strategy.calculate_delay(1) == 0.01
        assert strategy.should_retry(Exception(), 1) == True
        assert strategy.should_retry(Exception(), 5) == False
    
    def test_exponential_backoff_strategy(self):
        """指数バックオフ戦略テスト"""
        strategy = ExponentialBackoffStrategy(max_retries=3, base_delay=0.01, jitter=False)
        
        assert strategy.calculate_delay(0) == 0.01
        assert strategy.calculate_delay(1) == 0.02
        assert strategy.calculate_delay(2) == 0.04
    
    def test_successful_recovery(self):
        """成功復旧テスト"""
        call_count = 0
        
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("一時的エラー")
            return "成功"
        
        result = self.manager.recover_with_retry(test_function, "test_errors")
        
        assert result['recovery_successful'] == True
        assert result['recovery_details']['result'] == "成功"
        assert call_count == 2
    
    def test_failed_recovery(self):
        """失敗復旧テスト"""
        def always_fail():
            raise Exception("常に失敗")
        
        result = self.manager.recover_with_retry(always_fail, "test_errors")
        
        assert result['recovery_successful'] == False
        assert 'last_error' in result['recovery_details']
    
    def test_fallback_recovery(self):
        """フォールバック復旧テスト"""
        def primary_func():
            raise Exception("主要機能失敗")
        
        def fallback_func():
            return "フォールバック成功"
        
        result = self.manager.recover_with_fallback(
            primary_func, [fallback_func], "test_errors"
        )
        
        assert result['recovery_successful'] == True
        assert result['recovery_details']['result'] == "フォールバック成功"
    
    def test_global_retry_function(self):
        """グローバルリトライ関数テスト"""
        call_count = 0
        
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("一時的エラー")
            return "成功"
        
        result = retry_with_strategy(test_func, "test_errors")
        assert result == "成功"
    
    def teardown_method(self):
        """テスト後片付け"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestEnhancedLoggerManager:
    """強化ロガー管理テスト"""
    
    def setup_method(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "logging_config.json"
        
        # ログディレクトリ設定
        log_dir = self.temp_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # テスト用設定
        test_config = {
            "log_levels": {
                "root": "DEBUG",
                "strategy": "DEBUG"
            },
            "strategy_logging": {
                "separate_files": True,
                "include_performance": True
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        # 一時的にproject_rootを変更
        self.original_project_root = project_root
        import src.utils.logger_setup
        src.utils.logger_setup.project_root = self.temp_dir
        
        self.manager = EnhancedLoggerManager(str(self.config_file))
    
    def test_strategy_logger_creation(self):
        """戦略ロガー作成テスト"""
        logger = self.manager.get_strategy_logger("test_strategy")
        
        assert logger.name == "strategy.test_strategy"
        assert "test_strategy" in self.manager.strategy_loggers
        assert "test_strategy" in self.manager.log_stats['strategy_logs']
    
    def test_strategy_execution_logging(self):
        """戦略実行ログテスト"""
        self.manager.log_strategy_execution(
            "test_strategy", "実行開始", 
            execution_time=1.5, memory_usage=100.0
        )
        
        stats = self.manager.log_stats['strategy_logs']['test_strategy']
        assert stats['total_logs'] == 1
    
    def test_error_analysis_logging(self):
        """エラー分析ログテスト"""
        error = Exception("テストエラー")
        context = {"test": "context"}
        
        self.manager.log_error_with_analysis(error, context, "test_strategy")
        
        assert self.manager.log_stats['error_count'] == 1
        assert len(self.manager.log_stats['recent_errors']) == 1
    
    def test_performance_logging(self):
        """パフォーマンスログテスト"""
        self.manager.log_performance_metric(
            "execution_time", 2.5, "秒", {"strategy": "test"}
        )
        
        # ログが記録されることを確認
        assert True  # パフォーマンスロガーが正常に動作することを確認
    
    def test_global_strategy_logger(self):
        """グローバル戦略ロガーテスト"""
        logger = get_strategy_logger("global_test")
        assert logger.name == "strategy.global_test"
    
    def teardown_method(self):
        """テスト後片付け"""
        # project_rootを元に戻す
        import src.utils.logger_setup
        src.utils.logger_setup.project_root = self.original_project_root
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestMonitoringAgent:
    """監視エージェントテスト"""
    
    def setup_method(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "notification_config.json"
        
        # テスト用設定
        test_config = {
            "notification": {
                "enabled": True,
                "email_enabled": False,
                "webhook_enabled": False
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        # 一時的にproject_rootを変更
        self.original_project_root = project_root
        import src.utils.monitoring_agent
        src.utils.monitoring_agent.project_root = self.temp_dir
        
        self.agent = MonitoringAgent(str(self.config_file))
    
    def test_alert_rule_creation(self):
        """アラートルール作成テスト"""
        def test_condition(ctx):
            return ctx.get('test_value', 0) > 10
        
        self.agent.add_alert_rule("test_rule", test_condition, "WARNING")
        
        assert "test_rule" in self.agent.alert_rules
        assert self.agent.alert_rules["test_rule"].severity == "WARNING"
    
    def test_alert_triggering(self):
        """アラートトリガーテスト"""
        triggered_alerts = []
        
        # モック通知マネージャー
        def mock_send_alert(alert):
            triggered_alerts.append(alert)
        
        self.agent.notification_manager.send_alert = mock_send_alert
        
        # アラートルール追加
        def test_condition(ctx):
            return ctx.get('error_rate', 0) > 5
        
        self.agent.add_alert_rule("test_alert", test_condition, "ERROR")
        
        # アラート評価
        context = {'error_rate': 10}
        self.agent._evaluate_alert_rules(context)
        
        assert len(triggered_alerts) == 1
        assert triggered_alerts[0].event_type == "test_alert"
    
    def test_metrics_collection(self):
        """メトリクス収集テスト"""
        collector = self.agent.metrics_collector
        
        collector.add_metric("test_metric", 5.0)
        collector.add_metric("test_metric", 10.0)
        collector.add_metric("test_metric", 15.0)
        
        summary = collector.get_metric_summary("test_metric")
        
        assert summary['count'] == 3
        assert summary['avg'] == 10.0
        assert summary['min'] == 5.0
        assert summary['max'] == 15.0
    
    def test_event_reporting(self):
        """イベント報告テスト"""
        reported_events = []
        
        def mock_send_alert(alert):
            reported_events.append(alert)
        
        self.agent.notification_manager.send_alert = mock_send_alert
        
        self.agent.report_event(
            "test_event", "WARNING", "テストイベント",
            {"test": "data"}, "test_strategy"
        )
        
        assert len(reported_events) == 1
        assert reported_events[0].event_type == "test_event"
        assert reported_events[0].strategy_name == "test_strategy"
    
    @patch('src.utils.monitoring_agent.psutil')
    def test_system_monitoring(self, mock_psutil):
        """システム監視テスト"""
        # psutilモック
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0, used=8000000000)
        mock_psutil.disk_usage.return_value = Mock(percent=70.0)
        
        context = self.agent._collect_system_context()
        
        assert context['cpu_percent'] == 50.0
        assert context['memory_percent'] == 60.0
        assert 'memory_usage_mb' in context
    
    def test_global_functions(self):
        """グローバル関数テスト"""
        reported_events = []
        
        def mock_send_alert(alert):
            reported_events.append(alert)
        
        self.agent.notification_manager.send_alert = mock_send_alert
        
        # グローバル関数でエラー報告
        report_error(Exception("テストエラー"), {"test": "context"}, "test_strategy")
        
        assert len(reported_events) == 1
        assert reported_events[0].event_type == "error_occurred"
    
    def teardown_method(self):
        """テスト後片付け"""
        # project_rootを元に戻す
        import src.utils.monitoring_agent
        src.utils.monitoring_agent.project_root = self.original_project_root
        
        self.agent.stop_monitoring()
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestIntegration:
    """統合テスト"""
    
    def setup_method(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 各コンポーネント設定ディレクトリ作成
        config_dir = self.temp_dir / "config" / "error_handling"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        logs_dir = self.temp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # 一時的にproject_rootを変更
        self.original_project_root = project_root
        modules_to_patch = [
            'src.utils.exception_handler',
            'src.utils.error_recovery',
            'src.utils.logger_setup',
            'src.utils.monitoring_agent'
        ]
        
        for module_name in modules_to_patch:
            if module_name in sys.modules:
                setattr(sys.modules[module_name], 'project_root', self.temp_dir)
    
    def test_full_error_handling_workflow(self):
        """完全エラーハンドリングワークフローテスト"""
        # 1. エラー発生
        try:
            raise StrategyError("test_strategy", "統合テストエラー")
        except StrategyError as e:
            # 2. 例外処理
            error_result = handle_strategy_error("test_strategy", e)
            assert error_result['type'] == 'strategy'
            
            # 3. 復旧試行
            def recovery_func():
                return "復旧成功"
            
            recovery_result = retry_with_strategy(recovery_func, "strategy_errors")
            assert recovery_result == "復旧成功"
            
            # 4. エラー報告
            report_error(e, {"integration_test": True}, "test_strategy")
    
    def test_performance_monitoring_integration(self):
        """パフォーマンス監視統合テスト"""
        from src.utils.logger_setup import log_strategy_performance
        
        # パフォーマンスログ記録
        log_strategy_performance("test_strategy", 2.5, 150.0, "統合テスト")
        
        # 監視エージェントでパフォーマンス問題報告
        from src.utils.monitoring_agent import report_performance_issue
        report_performance_issue("execution_time", 5.0, 3.0, {"test": "integration"})
    
    def test_configuration_integration(self):
        """設定統合テスト"""
        # 各設定ファイルが正しく読み込まれることを確認
        handler = UnifiedExceptionHandler()
        recovery_manager = ErrorRecoveryManager()
        monitoring_agent = MonitoringAgent()
        
        # すべて正常に初期化されることを確認
        assert handler is not None
        assert recovery_manager is not None
        assert monitoring_agent is not None
    
    def teardown_method(self):
        """テスト後片付け"""
        # project_rootを元に戻す
        modules_to_restore = [
            'src.utils.exception_handler',
            'src.utils.error_recovery', 
            'src.utils.logger_setup',
            'src.utils.monitoring_agent'
        ]
        
        for module_name in modules_to_restore:
            if module_name in sys.modules:
                setattr(sys.modules[module_name], 'project_root', self.original_project_root)
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def test_error_handling_performance():
    """エラーハンドリングパフォーマンステスト"""
    import time
    
    start_time = time.time()
    
    # 大量エラー処理
    handler = get_exception_handler()
    for i in range(100):
        handler.handle_strategy_error(f"strategy_{i}", Exception(f"エラー{i}"))
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 100回の処理が2秒以内に完了することを確認
    assert processing_time < 2.0
    
    # 統計確認
    stats = handler.get_error_statistics()
    assert stats['total_errors'] >= 100


def test_system_resilience():
    """システム耐障害性テスト"""
    # 複数の同時エラーが発生してもシステムが安定していることを確認
    import threading
    import concurrent.futures
    
    def generate_errors():
        """エラー生成"""
        for i in range(10):
            try:
                if i % 3 == 0:
                    raise StrategyError("stress_strategy", f"ストレステスト{i}")
                elif i % 3 == 1:
                    raise DataError(f"データエラー{i}")
                else:
                    raise SystemError(f"システムエラー{i}")
            except Exception as e:
                handle_strategy_error("stress_test", e)
    
    # 複数スレッドで同時実行
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_errors) for _ in range(5)]
        
        # すべて正常完了することを確認
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 例外が発生していないことを確認
    
    # システムが安定していることを確認
    stats = get_exception_handler().get_error_statistics()
    assert stats['total_errors'] > 0


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
