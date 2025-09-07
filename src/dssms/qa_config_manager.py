"""
Quality Assurance Configuration Manager
Phase 2.3 Task 2.3.3: 品質保証システム

Purpose:
  - 品質保証システム設定管理
  - 動的設定読み込み・更新
  - 検証閾値管理
  - エラーハンドリング設定

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Integration:
  - 既存config/validation/と統合
  - output/quality_assurance/data_validator.pyとの連携
  - unified_output_engine.pyとの統合
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger


class ErrorLevel(Enum):
    """エラーレベル定義"""
    CRITICAL = "ERROR"
    WARNING = "WARN"
    INFO = "INFO"


class ErrorAction(Enum):
    """エラー時のアクション定義"""
    STOP_PROCESSING = "stop_processing"
    CONTINUE_PROCESSING = "continue_processing"
    RETRY = "retry"


@dataclass
class PerformanceThresholds:
    """パフォーマンス閾値設定"""
    min_total_return: float = -0.50
    max_total_return: float = 10.0
    min_sharpe_ratio: float = -3.0
    max_sharpe_ratio: float = 5.0
    max_drawdown: float = 0.60
    min_win_rate: float = 0.05
    max_win_rate: float = 0.95
    tolerance_percentage: float = 0.05


@dataclass
class DataConsistencyThresholds:
    """データ一貫性閾値設定"""
    missing_data_threshold: float = 0.02
    duplicate_data_threshold: float = 0.01
    outlier_detection_std: float = 3.0
    timestamp_consistency: bool = True
    price_data_consistency: bool = True


@dataclass
class OutputFormatConfig:
    """出力フォーマット設定"""
    required_columns: List[str] = field(default_factory=lambda: [
        "Date", "Entry_Signal", "Exit_Signal", "Position", 
        "Price", "Profit_Loss", "Cumulative_Return"
    ])
    column_data_types: Dict[str, str] = field(default_factory=lambda: {
        "Date": "datetime", "Entry_Signal": "int", "Exit_Signal": "int",
        "Position": "float", "Price": "float", "Profit_Loss": "float",
        "Cumulative_Return": "float"
    })
    validate_column_ranges: bool = True


@dataclass
class ErrorHandlingConfig:
    """エラーハンドリング設定"""
    critical_action: ErrorAction = ErrorAction.STOP_PROCESSING
    warning_action: ErrorAction = ErrorAction.CONTINUE_PROCESSING
    info_action: ErrorAction = ErrorAction.CONTINUE_PROCESSING
    critical_log_level: ErrorLevel = ErrorLevel.CRITICAL
    warning_log_level: ErrorLevel = ErrorLevel.WARNING
    info_log_level: ErrorLevel = ErrorLevel.INFO
    notify_critical: bool = True
    notify_warning: bool = False
    notify_info: bool = False


@dataclass
class RegressionTestingConfig:
    """リグレッションテスト設定"""
    enabled: bool = True
    tolerance_threshold: float = 0.05
    baseline_comparison: bool = True
    performance_comparison: bool = True
    output_format_comparison: bool = True
    test_cases_directory: str = "tests/regression"
    baseline_data_directory: str = "tests/baseline_data"


@dataclass
class LoggingConfig:
    """ログ設定"""
    quality_log_file: str = "logs/quality_assurance.log"
    validation_log_file: str = "logs/validation_results.log"
    regression_log_file: str = "logs/regression_tests.log"
    log_level: str = "INFO"
    detailed_validation_logs: bool = True
    performance_logs: bool = True


@dataclass
class IntegrationConfig:
    """統合設定"""
    unified_output_engine_integration: bool = True
    existing_validator_integration: bool = True
    automatic_correction: bool = True
    pre_output_validation: bool = True
    post_output_validation: bool = False


class QAConfigManager:
    """品質保証設定管理クラス"""
    
    def __init__(self, config_file_path: Optional[Union[str, Path]] = None):
        """
        初期化
        
        Args:
            config_file_path: 設定ファイルパス（Noneの場合はデフォルト使用）
        """
        self.logger = setup_logger(__name__)
        
        # デフォルト設定ファイルパス
        if config_file_path is None:
            default_path = project_root / "config" / "quality_assurance" / "qa_config.json"
            self.config_file_path = Path(default_path)
        else:
            self.config_file_path = Path(config_file_path)
        
        # 設定オブジェクト初期化
        self.performance_thresholds = PerformanceThresholds()
        self.data_consistency_thresholds = DataConsistencyThresholds()
        self.output_format_config = OutputFormatConfig()
        self.error_handling_config = ErrorHandlingConfig()
        self.regression_testing_config = RegressionTestingConfig()
        self.logging_config = LoggingConfig()
        self.integration_config = IntegrationConfig()
        
        # 設定読み込み
        self.load_config()
        
        self.logger.info(f"QA Config Manager 初期化完了: {self.config_file_path}")
    
    def load_config(self) -> None:
        """設定ファイル読み込み"""
        try:
            if not self.config_file_path.exists():
                self.logger.warning(f"設定ファイルが見つかりません: {self.config_file_path}")
                self.logger.info("デフォルト設定を使用します")
                return
            
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            qa_config = config_data.get('quality_assurance', {})
            
            # パフォーマンス閾値設定
            perf_config = qa_config.get('validation_thresholds', {}).get('performance_metrics', {})
            if perf_config:
                self.performance_thresholds = PerformanceThresholds(**perf_config)
            
            # データ一貫性閾値設定
            data_config = qa_config.get('validation_thresholds', {}).get('data_consistency', {})
            if data_config:
                self.data_consistency_thresholds = DataConsistencyThresholds(**data_config)
            
            # 出力フォーマット設定
            format_config = qa_config.get('validation_thresholds', {}).get('output_format', {})
            if format_config:
                self.output_format_config = OutputFormatConfig(**format_config)
            
            # エラーハンドリング設定
            error_config = qa_config.get('error_handling', {})
            if error_config:
                self._load_error_handling_config(error_config)
            
            # リグレッションテスト設定
            regression_config = qa_config.get('regression_testing', {})
            if regression_config:
                self.regression_testing_config = RegressionTestingConfig(**regression_config)
            
            # ログ設定
            logging_config = qa_config.get('logging', {})
            if logging_config:
                self.logging_config = LoggingConfig(**logging_config)
            
            # 統合設定
            integration_config = qa_config.get('integration', {})
            if integration_config:
                self.integration_config = IntegrationConfig(**integration_config)
            
            self.logger.info("設定ファイル読み込み完了")
            
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            self.logger.info("デフォルト設定を使用します")
    
    def _load_error_handling_config(self, error_config: Dict[str, Any]) -> None:
        """エラーハンドリング設定読み込み"""
        critical_config = error_config.get('critical_errors', {})
        warning_config = error_config.get('warnings', {})
        info_config = error_config.get('info', {})
        
        self.error_handling_config = ErrorHandlingConfig(
            critical_action=ErrorAction(critical_config.get('action', 'stop_processing')),
            warning_action=ErrorAction(warning_config.get('action', 'continue_processing')),
            info_action=ErrorAction(info_config.get('action', 'continue_processing')),
            critical_log_level=ErrorLevel(critical_config.get('log_level', 'ERROR')),
            warning_log_level=ErrorLevel(warning_config.get('log_level', 'WARN')),
            info_log_level=ErrorLevel(info_config.get('log_level', 'INFO')),
            notify_critical=critical_config.get('notify', True),
            notify_warning=warning_config.get('notify', False),
            notify_info=info_config.get('notify', False)
        )
    
    def save_config(self) -> None:
        """設定ファイル保存"""
        try:
            # ディレクトリ作成
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data: Dict[str, Any] = {
                "quality_assurance": {
                    "validation_thresholds": {
                        "performance_metrics": self._performance_thresholds_to_dict(),
                        "data_consistency": self._data_consistency_thresholds_to_dict(),
                        "output_format": self._output_format_config_to_dict()
                    },
                    "error_handling": self._error_handling_config_to_dict(),
                    "regression_testing": self._regression_testing_config_to_dict(),
                    "logging": self._logging_config_to_dict(),
                    "integration": self._integration_config_to_dict()
                }
            }
            
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"設定ファイル保存完了: {self.config_file_path}")
            
        except Exception as e:
            self.logger.error(f"設定ファイル保存エラー: {e}")
    
    def _performance_thresholds_to_dict(self) -> Dict[str, Any]:
        """パフォーマンス閾値を辞書に変換"""
        return {
            "min_total_return": self.performance_thresholds.min_total_return,
            "max_total_return": self.performance_thresholds.max_total_return,
            "min_sharpe_ratio": self.performance_thresholds.min_sharpe_ratio,
            "max_sharpe_ratio": self.performance_thresholds.max_sharpe_ratio,
            "max_drawdown": self.performance_thresholds.max_drawdown,
            "min_win_rate": self.performance_thresholds.min_win_rate,
            "max_win_rate": self.performance_thresholds.max_win_rate,
            "tolerance_percentage": self.performance_thresholds.tolerance_percentage
        }
    
    def _data_consistency_thresholds_to_dict(self) -> Dict[str, Any]:
        """データ一貫性閾値を辞書に変換"""
        return {
            "missing_data_threshold": self.data_consistency_thresholds.missing_data_threshold,
            "duplicate_data_threshold": self.data_consistency_thresholds.duplicate_data_threshold,
            "outlier_detection_std": self.data_consistency_thresholds.outlier_detection_std,
            "timestamp_consistency": self.data_consistency_thresholds.timestamp_consistency,
            "price_data_consistency": self.data_consistency_thresholds.price_data_consistency
        }
    
    def _output_format_config_to_dict(self) -> Dict[str, Any]:
        """出力フォーマット設定を辞書に変換"""
        return {
            "required_columns": self.output_format_config.required_columns,
            "column_data_types": self.output_format_config.column_data_types,
            "validate_column_ranges": self.output_format_config.validate_column_ranges
        }
    
    def _error_handling_config_to_dict(self) -> Dict[str, Any]:
        """エラーハンドリング設定を辞書に変換"""
        return {
            "critical_errors": {
                "action": self.error_handling_config.critical_action.value,
                "log_level": self.error_handling_config.critical_log_level.value,
                "notify": self.error_handling_config.notify_critical
            },
            "warnings": {
                "action": self.error_handling_config.warning_action.value,
                "log_level": self.error_handling_config.warning_log_level.value,
                "notify": self.error_handling_config.notify_warning
            },
            "info": {
                "action": self.error_handling_config.info_action.value,
                "log_level": self.error_handling_config.info_log_level.value,
                "notify": self.error_handling_config.notify_info
            }
        }
    
    def _regression_testing_config_to_dict(self) -> Dict[str, Any]:
        """リグレッションテスト設定を辞書に変換"""
        return {
            "enabled": self.regression_testing_config.enabled,
            "tolerance_threshold": self.regression_testing_config.tolerance_threshold,
            "baseline_comparison": self.regression_testing_config.baseline_comparison,
            "performance_comparison": self.regression_testing_config.performance_comparison,
            "output_format_comparison": self.regression_testing_config.output_format_comparison,
            "test_cases_directory": self.regression_testing_config.test_cases_directory,
            "baseline_data_directory": self.regression_testing_config.baseline_data_directory
        }
    
    def _logging_config_to_dict(self) -> Dict[str, Any]:
        """ログ設定を辞書に変換"""
        return {
            "quality_log_file": self.logging_config.quality_log_file,
            "validation_log_file": self.logging_config.validation_log_file,
            "regression_log_file": self.logging_config.regression_log_file,
            "log_level": self.logging_config.log_level,
            "detailed_validation_logs": self.logging_config.detailed_validation_logs,
            "performance_logs": self.logging_config.performance_logs
        }
    
    def _integration_config_to_dict(self) -> Dict[str, Any]:
        """統合設定を辞書に変換"""
        return {
            "unified_output_engine_integration": self.integration_config.unified_output_engine_integration,
            "existing_validator_integration": self.integration_config.existing_validator_integration,
            "automatic_correction": self.integration_config.automatic_correction,
            "pre_output_validation": self.integration_config.pre_output_validation,
            "post_output_validation": self.integration_config.post_output_validation
        }
    
    def get_performance_thresholds(self) -> PerformanceThresholds:
        """パフォーマンス閾値取得"""
        return self.performance_thresholds
    
    def get_data_consistency_thresholds(self) -> DataConsistencyThresholds:
        """データ一貫性閾値取得"""
        return self.data_consistency_thresholds
    
    def get_output_format_config(self) -> OutputFormatConfig:
        """出力フォーマット設定取得"""
        return self.output_format_config
    
    def get_error_handling_config(self) -> ErrorHandlingConfig:
        """エラーハンドリング設定取得"""
        return self.error_handling_config
    
    def get_regression_testing_config(self) -> RegressionTestingConfig:
        """リグレッションテスト設定取得"""
        return self.regression_testing_config
    
    def get_logging_config(self) -> LoggingConfig:
        """ログ設定取得"""
        return self.logging_config
    
    def get_integration_config(self) -> IntegrationConfig:
        """統合設定取得"""
        return self.integration_config
    
    def update_performance_threshold(self, threshold_name: str, value: float) -> None:
        """パフォーマンス閾値更新"""
        if hasattr(self.performance_thresholds, threshold_name):
            setattr(self.performance_thresholds, threshold_name, value)
            self.logger.info(f"パフォーマンス閾値更新: {threshold_name} = {value}")
        else:
            self.logger.warning(f"無効な閾値名: {threshold_name}")
    
    def validate_config(self) -> bool:
        """設定妥当性検証"""
        try:
            # パフォーマンス閾値検証
            if (self.performance_thresholds.min_total_return >= 
                self.performance_thresholds.max_total_return):
                self.logger.error("パフォーマンス閾値エラー: min_total_return >= max_total_return")
                return False
            
            if (self.performance_thresholds.min_sharpe_ratio >= 
                self.performance_thresholds.max_sharpe_ratio):
                self.logger.error("パフォーマンス閾値エラー: min_sharpe_ratio >= max_sharpe_ratio")
                return False
            
            if not (0 <= self.performance_thresholds.max_drawdown <= 1):
                self.logger.error("パフォーマンス閾値エラー: max_drawdown は 0-1 の範囲である必要があります")
                return False
            
            # データ一貫性閾値検証
            if not (0 <= self.data_consistency_thresholds.missing_data_threshold <= 1):
                self.logger.error("データ一貫性閾値エラー: missing_data_threshold は 0-1 の範囲である必要があります")
                return False
            
            # 出力フォーマット設定検証
            if not self.output_format_config.required_columns:
                self.logger.error("出力フォーマット設定エラー: required_columns が空です")
                return False
            
            self.logger.info("設定妥当性検証完了")
            return True
            
        except Exception as e:
            self.logger.error(f"設定妥当性検証エラー: {e}")
            return False


if __name__ == "__main__":
    # テスト実行
    config_manager = QAConfigManager()
    
    print("=== 品質保証設定管理システム テスト ===")
    print(f"パフォーマンス閾値: {config_manager.get_performance_thresholds()}")
    print(f"データ一貫性閾値: {config_manager.get_data_consistency_thresholds()}")
    print(f"エラーハンドリング設定: {config_manager.get_error_handling_config()}")
    
    # 設定妥当性検証
    is_valid = config_manager.validate_config()
    print(f"設定妥当性: {'OK' if is_valid else 'NG'}")
    
    # 設定保存テスト
    config_manager.save_config()
    print("設定保存テスト完了")
