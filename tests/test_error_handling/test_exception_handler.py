"""
例外処理ハンドラー単体テスト
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.exception_handler import (
    UnifiedExceptionHandler, StrategyError, DataError, SystemError
)


class TestStrategyError:
    """戦略エラークラステスト"""
    
    def test_strategy_error_creation(self):
        """戦略エラー作成テスト"""
        error = StrategyError("test_strategy", "テストメッセージ")
        
        assert error.strategy_name == "test_strategy"
        assert str(error) == "test_strategy: テストメッセージ"
    
    def test_strategy_error_with_original(self):
        """原因エラー付き戦略エラーテスト"""
        original_error = ValueError("元のエラー")
        error = StrategyError("test_strategy", "テストメッセージ", original_error)
        
        assert error.original_error == original_error


class TestUnifiedExceptionHandler:
    """統一例外処理ハンドラーテスト"""
    
    def setup_method(self):
        """テスト前準備"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "error_policies.json"
        
        # テスト用設定
        self.test_config = {
            "strategy_errors": {
                "max_retries": 3,
                "retry_delay": 0.1,
                "fallback_enabled": True,
                "continue_on_fail": False
            },
            "data_errors": {
                "max_retries": 2,
                "retry_delay": 0.5,
                "fallback_enabled": False,
                "continue_on_fail": True
            },
            "system_errors": {
                "max_retries": 1,
                "retry_delay": 1.0,
                "fallback_enabled": True,
                "continue_on_fail": False
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        self.handler = UnifiedExceptionHandler(str(self.config_file))
    
    def test_initialization_with_config(self):
        """設定ファイル付き初期化テスト"""
        assert self.handler.error_policies == self.test_config
        assert self.handler.config_path == self.config_file
    
    def test_initialization_without_config(self):
        """設定ファイルなし初期化テスト"""
        handler = UnifiedExceptionHandler()
        
        # デフォルト設定が読み込まれることを確認
        assert handler.error_policies is not None
        assert 'strategy_errors' in handler.error_policies
    
    def test_load_error_policies_file_not_found(self):
        """設定ファイル未見つけテスト"""
        non_existent_file = self.temp_dir / "non_existent.json"
        handler = UnifiedExceptionHandler(str(non_existent_file))
        
        # デフォルト設定が使用されることを確認
        assert handler.error_policies is not None
    
    def test_load_error_policies_invalid_json(self):
        """不正JSON設定ファイルテスト"""
        invalid_json_file = self.temp_dir / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("{ invalid json }")
        
        handler = UnifiedExceptionHandler(str(invalid_json_file))
        
        # デフォルト設定が使用されることを確認
        assert handler.error_policies is not None
    
    def test_handle_strategy_error(self):
        """戦略エラー処理テスト"""
        error = Exception("テスト戦略エラー")
        context = {"test_key": "test_value"}
        
        result = self.handler.handle_strategy_error("test_strategy", error, context)
        
        # 結果構造確認
        assert result['type'] == 'strategy'
        assert result['strategy_name'] == 'test_strategy'
        assert result['error_message'] == str(error)
        assert result['context'] == context
        assert 'timestamp' in result
        assert 'traceback' in result
        assert 'recovery_attempted' in result
        assert 'recovery_successful' in result
    
    def test_handle_strategy_error_with_retry(self):
        """リトライ付き戦略エラー処理テスト"""
        call_count = 0
        
        def retry_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("一時的エラー")
            return "成功"
        
        error = Exception("テスト戦略エラー")
        result = self.handler.handle_strategy_error(
            "test_strategy", error, retry_func=retry_function
        )
        
        # リトライが実行されることを確認
        assert result['recovery_attempted'] == True
        assert call_count >= 1
    
    def test_handle_data_error(self):
        """データエラー処理テスト"""
        error = Exception("テストデータエラー")
        context = {"data_source": "test_api"}
        
        result = self.handler.handle_data_error(error, context)
        
        assert result['type'] == 'data'
        assert result['error_message'] == str(error)
        assert result['context'] == context
    
    def test_handle_system_error(self):
        """システムエラー処理テスト"""
        error = Exception("テストシステムエラー")
        context = {"component": "test_component"}
        
        result = self.handler.handle_system_error(error, context)
        
        assert result['type'] == 'system'
        assert result['error_message'] == str(error)
        assert result['context'] == context
    
    def test_error_statistics_tracking(self):
        """エラー統計追跡テスト"""
        # 初期状態確認
        initial_stats = self.handler.get_error_statistics()
        assert initial_stats['total_errors'] == 0
        
        # 各種エラー発生
        self.handler.handle_strategy_error("strategy1", Exception("エラー1"))
        self.handler.handle_strategy_error("strategy2", Exception("エラー2"))
        self.handler.handle_data_error(Exception("データエラー"))
        self.handler.handle_system_error(Exception("システムエラー"))
        
        # 統計確認
        stats = self.handler.get_error_statistics()
        assert stats['total_errors'] == 4
        assert stats['strategy_errors'] == 2
        assert stats['data_errors'] == 1
        assert stats['system_errors'] == 1
    
    def test_error_rate_calculation(self):
        """エラー率計算テスト"""
        # 複数エラー発生
        for i in range(5):
            self.handler.handle_strategy_error(f"strategy{i}", Exception(f"エラー{i}"))
        
        # 成功ケースの代替として処理回数を手動設定
        self.handler.error_stats['total_processed'] = 10
        
        stats = self.handler.get_error_statistics()
        expected_rate = (5 / 10) * 100  # 50%
        
        # エラー率が計算されることを確認
        assert 'error_rate' in stats
    
    def test_attempt_recovery_without_retry_func(self):
        """リトライ関数なし復旧試行テスト"""
        error_info = {
            'type': 'strategy',
            'strategy_name': 'test',
            'error_message': 'テスト'
        }
        
        result = self.handler._attempt_recovery('strategy_errors', None, error_info)
        
        assert result['recovery_attempted'] == False
        assert result['recovery_successful'] == False
    
    def test_attempt_recovery_with_successful_retry(self):
        """成功リトライ復旧試行テスト"""
        call_count = 0
        
        def successful_retry():
            nonlocal call_count
            call_count += 1
            return "成功"
        
        error_info = {
            'type': 'strategy',
            'strategy_name': 'test',
            'error_message': 'テスト'
        }
        
        result = self.handler._attempt_recovery('strategy_errors', successful_retry, error_info)
        
        assert result['recovery_attempted'] == True
        assert result['recovery_successful'] == True
        assert call_count == 1
    
    def test_attempt_recovery_with_failed_retry(self):
        """失敗リトライ復旧試行テスト"""
        def always_fail():
            raise Exception("常に失敗")
        
        error_info = {
            'type': 'strategy',
            'strategy_name': 'test',
            'error_message': 'テスト'
        }
        
        result = self.handler._attempt_recovery('strategy_errors', always_fail, error_info)
        
        assert result['recovery_attempted'] == True
        assert result['recovery_successful'] == False
        assert 'retry_errors' in result
    
    def test_log_error(self):
        """エラーログテスト"""
        with patch.object(self.handler.logger, 'error') as mock_logger:
            error_info = {
                'type': 'strategy',
                'strategy_name': 'test',
                'error_message': 'テストエラー'
            }
            
            self.handler._log_error(error_info)
            
            # ロガーが呼び出されることを確認
            mock_logger.assert_called_once()
    
    def test_create_error_report(self):
        """エラーレポート作成テスト"""
        # いくつかエラーを発生させる
        self.handler.handle_strategy_error("test", Exception("テスト"))
        
        report_path = self.handler.create_error_report()
        
        # レポートファイルが作成されることを確認
        assert report_path != ""
        if report_path:
            report_file = Path(report_path)
            # ファイルが実際に作成されている場合のみ確認
            if report_file.exists():
                assert report_file.suffix == '.json'
    
    def test_reset_statistics(self):
        """統計リセットテスト"""
        # エラー発生
        self.handler.handle_strategy_error("test", Exception("テスト"))
        
        # 統計確認
        stats_before = self.handler.get_error_statistics()
        assert stats_before['total_errors'] > 0
        
        # リセット
        self.handler.reset_statistics()
        
        # リセット後確認
        stats_after = self.handler.get_error_statistics()
        assert stats_after['total_errors'] == 0
        assert stats_after['strategy_errors'] == 0
        assert stats_after['data_errors'] == 0
        assert stats_after['system_errors'] == 0
    
    def teardown_method(self):
        """テスト後片付け"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestGlobalExceptionHandling:
    """グローバル例外処理テスト"""
    
    def test_get_exception_handler_singleton(self):
        """例外ハンドラーシングルトンテスト"""
        from src.utils.exception_handler import get_exception_handler
        
        handler1 = get_exception_handler()
        handler2 = get_exception_handler()
        
        # 同一インスタンスが返されることを確認
        assert handler1 is handler2
    
    def test_global_handle_strategy_error(self):
        """グローバル戦略エラー処理テスト"""
        from src.utils.exception_handler import handle_strategy_error
        
        result = handle_strategy_error("test_strategy", Exception("テスト"))
        
        assert result['type'] == 'strategy'
        assert result['strategy_name'] == 'test_strategy'
    
    def test_global_handle_data_error(self):
        """グローバルデータエラー処理テスト"""
        from src.utils.exception_handler import handle_data_error
        
        result = handle_data_error(Exception("データエラー"))
        
        assert result['type'] == 'data'
    
    def test_global_handle_system_error(self):
        """グローバルシステムエラー処理テスト"""
        from src.utils.exception_handler import handle_system_error
        
        result = handle_system_error(Exception("システムエラー"))
        
        assert result['type'] == 'system'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
