"""
強化エラーハンドリングシステム統合テスト
Phase 2: エラー処理強化実装の動作確認とSystemFallbackPolicy統合テスト

Author: imega
Created: 2025-10-07
Task: TODO(tag:phase2, rationale:強化エラーハンドリングシステム動作確認)
"""

import pytest
import logging
from datetime import datetime
from unittest.mock import Mock, patch
from typing import Dict, Any

# プロジェクト内インポート
try:
    from src.config.enhanced_error_handling import (
        ErrorSeverity,
        ErrorRecoveryManager,
        EnhancedErrorHandler,
        EnhancedErrorRecord,
        initialize_global_error_handler
    )
    from src.config.system_modes import SystemFallbackPolicy, ComponentType, SystemMode
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("テスト実行には適切なパス設定が必要です")


class TestEnhancedErrorHandling:
    """強化エラーハンドリングシステムのテストクラス"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.fallback_policy = SystemFallbackPolicy(SystemMode.DEVELOPMENT)
        self.recovery_manager = ErrorRecoveryManager()
        self.error_handler = EnhancedErrorHandler(
            fallback_policy=self.fallback_policy,
            recovery_manager=self.recovery_manager
        )
    
    def test_error_severity_classification(self):
        """エラー重要度分類テスト"""
        # CRITICAL エラーテスト
        test_error = ValueError("Critical system failure")
        
        with patch.object(self.error_handler.logger, 'critical') as mock_critical:
            result = self.error_handler.handle_error(
                severity=ErrorSeverity.CRITICAL,
                component_type=ComponentType.DSSMS_CORE,
                component_name="TestComponent",
                error=test_error
            )
            
            # CRITICAL ログが出力されたことを確認
            mock_critical.assert_called()
            assert len(self.error_handler.error_records) == 1
            
            record = self.error_handler.error_records[0]
            assert record.severity == ErrorSeverity.CRITICAL
            assert record.component_type == ComponentType.DSSMS_CORE
            assert record.error_type == "ValueError"
    
    def test_error_recovery_mechanism(self):
        """エラー回復メカニズムテスト"""
        # モック回復関数
        def mock_recovery_func(context: Dict[str, Any]) -> bool:
            return True  # 回復成功をシミュレート
        
        # 回復戦略登録
        self.recovery_manager.register_recovery_strategy(
            component_name="TestComponent",
            error_type="ConnectionError",
            recovery_func=mock_recovery_func,
            max_retries=3
        )
        
        # 回復処理テスト
        test_error = ConnectionError("Network connection failed")
        result = self.error_handler.handle_error(
            severity=ErrorSeverity.ERROR,
            component_type=ComponentType.DATA_FETCHER,
            component_name="TestComponent",
            error=test_error
        )
        
        # 回復成功を確認
        assert result == True
        assert len(self.error_handler.error_records) == 1
        
        record = self.error_handler.error_records[0]
        assert record.recovery_attempted == True
        assert record.recovery_successful == True
    
    def test_production_mode_critical_handling(self):
        """Production mode CRITICAL エラー処理テスト"""
        # Production modeに変更
        production_policy = SystemFallbackPolicy(SystemMode.PRODUCTION)
        production_handler = EnhancedErrorHandler(
            fallback_policy=production_policy,
            recovery_manager=self.recovery_manager
        )
        
        test_error = RuntimeError("Production critical error")
        
        # Production modeでCRITICALエラーは例外を再発生させる
        with pytest.raises(RuntimeError):
            production_handler.handle_error(
                severity=ErrorSeverity.CRITICAL,
                component_type=ComponentType.STRATEGY_ENGINE,
                component_name="ProductionComponent",
                error=test_error
            )
    
    def test_warning_level_handling(self):
        """WARNING レベル処理テスト"""
        test_error = Warning("Minor issue detected")
        
        with patch.object(self.error_handler.logger, 'warning') as mock_warning:
            result = self.error_handler.handle_error(
                severity=ErrorSeverity.WARNING,
                component_type=ComponentType.RISK_MANAGER,
                component_name="RiskComponent",
                error=test_error
            )
            
            # WARNING ログが出力されたことを確認
            mock_warning.assert_called()
            assert len(self.error_handler.error_records) == 1
    
    def test_error_statistics_generation(self):
        """エラー統計生成テスト"""
        # 複数のエラーを発生させる
        errors = [
            (ErrorSeverity.CRITICAL, ValueError("Critical 1")),
            (ErrorSeverity.ERROR, ConnectionError("Error 1")),
            (ErrorSeverity.WARNING, UserWarning("Warning 1")),
            (ErrorSeverity.INFO, Exception("Info 1"))
        ]
        
        for severity, error in errors:
            self.error_handler.handle_error(
                severity=severity,
                component_type=ComponentType.DSSMS_CORE,
                component_name="StatTestComponent",
                error=error
            )
        
        # 統計取得
        stats = self.error_handler.get_error_statistics()
        
        assert stats['total_errors'] == 4
        assert stats['by_severity']['critical'] == 1
        assert stats['by_severity']['error'] == 1
        assert stats['by_severity']['warning'] == 1
        assert stats['by_severity']['info'] == 1
        assert len(stats['recent_errors']) == 4
    
    def test_recovery_retry_limit(self):
        """回復処理再試行制限テスト"""
        retry_count = 0
        
        def failing_recovery_func(context: Dict[str, Any]) -> bool:
            nonlocal retry_count
            retry_count += 1
            return False  # 常に失敗
        
        # 回復戦略登録 (最大2回試行)
        self.recovery_manager.register_recovery_strategy(
            component_name="FailingComponent",
            error_type="TestError",
            recovery_func=failing_recovery_func,
            max_retries=2
        )
        
        # 複数回エラーを発生させて再試行制限をテスト
        test_error = Exception("Test error")
        
        for i in range(3):  # 3回実行
            self.error_handler.handle_error(
                severity=ErrorSeverity.ERROR,
                component_type=ComponentType.MULTI_STRATEGY,
                component_name="FailingComponent",
                error=test_error
            )
        
        # 最大再試行数 (2回) を超えないことを確認
        assert retry_count == 2
    
    def test_global_error_handler_initialization(self):
        """グローバルエラーハンドラー初期化テスト"""
        fallback_policy = SystemFallbackPolicy(SystemMode.TESTING)
        
        global_handler = initialize_global_error_handler(fallback_policy)
        
        assert global_handler is not None
        assert global_handler.fallback_policy.mode == SystemMode.TESTING
    
    def test_error_log_export(self, tmp_path):
        """エラーログ出力テスト"""
        # テスト用エラー生成
        test_error = ValueError("Export test error")
        self.error_handler.handle_error(
            severity=ErrorSeverity.ERROR,
            component_type=ComponentType.DATA_FETCHER,
            component_name="ExportTestComponent",
            error=test_error,
            context={"test_context": "export_test"}
        )
        
        # ログファイル出力
        log_file = tmp_path / "test_error_log.json"
        self.error_handler.export_error_log(str(log_file))
        
        # ファイルが作成されたことを確認
        assert log_file.exists()
        
        # ファイル内容確認
        import json
        with open(log_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        assert 'export_timestamp' in log_data
        assert 'system_mode' in log_data
        assert 'statistics' in log_data
        assert 'detailed_records' in log_data
        assert len(log_data['detailed_records']) == 1


def run_enhanced_error_handling_demo():
    """
    強化エラーハンドリングシステムデモ実行
    TODO(tag:phase2, rationale:実際の使用例・統合動作確認)
    """
    print("=== 強化エラーハンドリングシステム デモ実行 ===")
    
    # システム初期化
    fallback_policy = SystemFallbackPolicy(SystemMode.DEVELOPMENT)
    recovery_manager = ErrorRecoveryManager()
    error_handler = EnhancedErrorHandler(fallback_policy, recovery_manager)
    
    # 回復戦略登録例
    def network_recovery(context):
        print(f"  ネットワーク回復処理実行中...")
        return True  # 成功をシミュレート
    
    def data_recovery(context):
        print(f"  データ回復処理実行中...")
        return False  # 失敗をシミュレート
    
    recovery_manager.register_recovery_strategy(
        "NetworkComponent", "ConnectionError", network_recovery, max_retries=3
    )
    recovery_manager.register_recovery_strategy(
        "DataComponent", "DataError", data_recovery, max_retries=2
    )
    
    # 各レベルのエラーテスト
    print("\n1. WARNING レベルエラー (回復成功)")
    error_handler.handle_error(
        ErrorSeverity.WARNING,
        ComponentType.DATA_FETCHER,
        "NetworkComponent",
        ConnectionError("Network timeout"),
        context={"retry_count": 1}
    )
    
    print("\n2. ERROR レベルエラー (回復失敗)")
    error_handler.handle_error(
        ErrorSeverity.ERROR,
        ComponentType.DSSMS_CORE,
        "DataComponent",
        Exception("Data processing failed"),
        context={"data_size": 10000}
    )
    
    print("\n3. INFO レベル情報記録")
    error_handler.handle_error(
        ErrorSeverity.INFO,
        ComponentType.STRATEGY_ENGINE,
        "StrategyComponent",
        Exception("Strategy execution completed"),
        context={"strategy_id": "VWAP_001"}
    )
    
    # 統計表示
    print("\n=== エラー統計 ===")
    stats = error_handler.get_error_statistics()
    print(f"総エラー数: {stats['total_errors']}")
    print(f"重要度別: {stats['by_severity']}")
    print(f"回復率: {stats['recovery_rate']}%")
    print(f"システムモード: {stats['system_mode']}")
    
    # エラーログ出力
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"enhanced_error_demo_{timestamp}.json"
    error_handler.export_error_log(log_filename)
    print(f"\n詳細エラーログを出力しました: {log_filename}")
    
    print("\n=== デモ実行完了 ===")


if __name__ == "__main__":
    # デモ実行
    run_enhanced_error_handling_demo()
    
    # テスト実行 (pytest がインストールされている場合)
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest がインストールされていません。テストをスキップします。")