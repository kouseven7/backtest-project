"""
Output Data Validator
Phase 2.3 Task 2.3.3: 品質保証システム

Purpose:
  - 出力データの事前検証
  - パフォーマンスメトリクス検証
  - 出力フォーマット検証
  - データ一貫性チェック

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Integration:
  - qa_config_manager.pyとの連携
  - 既存output/quality_assurance/data_validator.pyとの統合
  - unified_output_engine.pyへの組み込み
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from src.dssms.qa_config_manager import (
    QAConfigManager, 
    OutputFormatConfig,
    ErrorLevel,
    ErrorAction
)


@dataclass
class ValidationResult:
    """検証結果"""
    is_valid: bool = True
    error_level: ErrorLevel = ErrorLevel.INFO
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    messages: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    validation_time: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_trades: int = 0
    loss_trades: int = 0
    average_profit: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0


class OutputDataValidator:
    """出力データ検証システム"""
    
    def __init__(self, config_manager: Optional[QAConfigManager] = None):
        """
        初期化
        
        Args:
            config_manager: 設定管理システム（Noneの場合は新規作成）
        """
        self.config_manager = config_manager or QAConfigManager()
        self.logger = setup_logger(__name__)
        
        # 設定取得
        self.performance_thresholds = self.config_manager.get_performance_thresholds()
        self.data_consistency_thresholds = self.config_manager.get_data_consistency_thresholds()
        self.output_format_config = self.config_manager.get_output_format_config()
        self.error_handling_config = self.config_manager.get_error_handling_config()
        
        self.logger.info("Output Data Validator 初期化完了")
    
    def validate_output_data(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        出力データ総合検証
        
        Args:
            data: 検証対象データ
            metadata: メタデータ（戦略名、期間等）
        
        Returns:
            ValidationResult: 検証結果
        """
        start_time = datetime.now()
        result = ValidationResult(validation_time=start_time)
        
        try:
            self.logger.info("出力データ検証開始")
            
            # 1. 出力フォーマット検証
            format_result = self._validate_output_format(data)
            self._merge_validation_results(result, format_result)
            
            # 2. データ一貫性検証
            consistency_result = self._validate_data_consistency(data)
            self._merge_validation_results(result, consistency_result)
            
            # 3. パフォーマンスメトリクス検証
            if 'Profit_Loss' in data.columns or 'Cumulative_Return' in data.columns:
                performance_result = self._validate_performance_metrics(data)
                self._merge_validation_results(result, performance_result)
            
            # 4. 統計的妥当性検証
            statistical_result = self._validate_statistical_validity(data)
            self._merge_validation_results(result, statistical_result)
            
            # 最終判定
            result.is_valid = (result.error_count == 0)
            
            # 検証時間記録
            result.validation_time = datetime.now()
            
            self.logger.info(f"出力データ検証完了: エラー={result.error_count}, 警告={result.warning_count}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"出力データ検証エラー: {e}")
            result.is_valid = False
            result.error_level = ErrorLevel.CRITICAL
            result.error_count += 1
            result.messages.append(f"検証処理エラー: {e}")
            return result
    
    def _validate_output_format(self, data: pd.DataFrame) -> ValidationResult:
        """出力フォーマット検証"""
        result = ValidationResult()
        
        try:
            # 必須カラム存在チェック
            missing_columns = []
            for required_col in self.output_format_config.required_columns:
                if required_col not in data.columns:
                    missing_columns.append(required_col)
            
            if missing_columns:
                result.error_count += 1
                result.error_level = ErrorLevel.CRITICAL
                result.messages.append(f"必須カラムが不足: {missing_columns}")
                result.is_valid = False
            
            # データ型チェック
            if self.output_format_config.validate_column_ranges:
                type_errors = self._validate_column_types(data)
                if type_errors:
                    result.warning_count += len(type_errors)
                    result.messages.extend(type_errors)
            
            # データ範囲チェック
            range_errors = self._validate_column_ranges(data)
            if range_errors:
                result.warning_count += len(range_errors)
                result.messages.extend(range_errors)
            
            self.logger.info("出力フォーマット検証完了")
            
        except Exception as e:
            result.error_count += 1
            result.error_level = ErrorLevel.CRITICAL
            result.messages.append(f"フォーマット検証エラー: {e}")
            result.is_valid = False
            
        return result
    
    def _validate_data_consistency(self, data: pd.DataFrame) -> ValidationResult:
        """データ一貫性検証"""
        result = ValidationResult()
        
        try:
            # 欠損データチェック
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > self.data_consistency_thresholds.missing_data_threshold:
                result.warning_count += 1
                result.messages.append(f"欠損データ比率が閾値超過: {missing_ratio:.2%}")
            
            # 重複データチェック
            duplicate_ratio = data.duplicated().sum() / len(data)
            if duplicate_ratio > self.data_consistency_thresholds.duplicate_data_threshold:
                result.warning_count += 1
                result.messages.append(f"重複データ比率が閾値超過: {duplicate_ratio:.2%}")
            
            # 外れ値チェック
            if self.data_consistency_thresholds.outlier_detection_std > 0:
                outlier_info = self._detect_outliers(data)
                if outlier_info:
                    result.warning_count += len(outlier_info)
                    result.messages.extend(outlier_info)
            
            # タイムスタンプ一貫性チェック
            if (self.data_consistency_thresholds.timestamp_consistency and 
                'Date' in data.columns):
                timestamp_errors = self._validate_timestamps(data)
                if timestamp_errors:
                    result.warning_count += len(timestamp_errors)
                    result.messages.extend(timestamp_errors)
            
            self.logger.info("データ一貫性検証完了")
            
        except Exception as e:
            result.error_count += 1
            result.error_level = ErrorLevel.CRITICAL
            result.messages.append(f"一貫性検証エラー: {e}")
            result.is_valid = False
            
        return result
    
    def _validate_performance_metrics(self, data: pd.DataFrame) -> ValidationResult:
        """パフォーマンスメトリクス検証"""
        result = ValidationResult()
        
        try:
            # パフォーマンスメトリクス計算
            metrics = self._calculate_performance_metrics(data)
            result.details['performance_metrics'] = metrics.__dict__
            
            # 閾値チェック
            if metrics.total_return < self.performance_thresholds.min_total_return:
                result.warning_count += 1
                result.messages.append(
                    f"総リターンが最小閾値未満: {metrics.total_return:.2%} < "
                    f"{self.performance_thresholds.min_total_return:.2%}"
                )
            
            if metrics.total_return > self.performance_thresholds.max_total_return:
                result.warning_count += 1
                result.messages.append(
                    f"総リターンが最大閾値超過: {metrics.total_return:.2%} > "
                    f"{self.performance_thresholds.max_total_return:.2%}"
                )
            
            if metrics.sharpe_ratio < self.performance_thresholds.min_sharpe_ratio:
                result.warning_count += 1
                result.messages.append(
                    f"シャープレシオが最小閾値未満: {metrics.sharpe_ratio:.2f} < "
                    f"{self.performance_thresholds.min_sharpe_ratio:.2f}"
                )
            
            if metrics.max_drawdown > self.performance_thresholds.max_drawdown:
                result.warning_count += 1
                result.messages.append(
                    f"最大ドローダウンが閾値超過: {metrics.max_drawdown:.2%} > "
                    f"{self.performance_thresholds.max_drawdown:.2%}"
                )
            
            if metrics.win_rate < self.performance_thresholds.min_win_rate:
                result.warning_count += 1
                result.messages.append(
                    f"勝率が最小閾値未満: {metrics.win_rate:.2%} < "
                    f"{self.performance_thresholds.min_win_rate:.2%}"
                )
            
            self.logger.info("パフォーマンスメトリクス検証完了")
            
        except Exception as e:
            result.error_count += 1
            result.error_level = ErrorLevel.CRITICAL
            result.messages.append(f"パフォーマンス検証エラー: {e}")
            result.is_valid = False
            
        return result
    
    def _validate_statistical_validity(self, data: pd.DataFrame) -> ValidationResult:
        """統計的妥当性検証"""
        result = ValidationResult()
        
        try:
            # データサイズチェック
            if len(data) < 10:
                result.warning_count += 1
                result.messages.append(f"データサイズが小さすぎます: {len(data)} < 10")
            
            # 数値データの統計的チェック
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in data.columns:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        # 無限値チェック
                        if np.isinf(col_data).any():
                            result.error_count += 1
                            result.messages.append(f"カラム '{col}' に無限値が含まれています")
                        
                        # 極値チェック
                        std_val = col_data.std()
                        mean_val = col_data.mean()
                        if std_val > 0:
                            extreme_values = np.abs((col_data - mean_val) / std_val) > 5
                            if extreme_values.any():
                                result.warning_count += 1
                                result.messages.append(f"カラム '{col}' に極値が含まれています")
            
            self.logger.info("統計的妥当性検証完了")
            
        except Exception as e:
            result.error_count += 1
            result.error_level = ErrorLevel.CRITICAL
            result.messages.append(f"統計的妥当性検証エラー: {e}")
            result.is_valid = False
            
        return result
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> PerformanceMetrics:
        """パフォーマンスメトリクス計算"""
        metrics = PerformanceMetrics()
        
        try:
            # 累積リターン計算
            if 'Cumulative_Return' in data.columns:
                cumulative_returns = data['Cumulative_Return'].dropna()
                if len(cumulative_returns) > 0:
                    metrics.total_return = cumulative_returns.iloc[-1] - 1.0 if cumulative_returns.iloc[-1] > 1 else cumulative_returns.iloc[-1]
            
            # 損益データから計算
            if 'Profit_Loss' in data.columns:
                pl_data = data['Profit_Loss'].dropna()
                if len(pl_data) > 0:
                    metrics.total_trades = len(pl_data[pl_data != 0])
                    metrics.profit_trades = len(pl_data[pl_data > 0])
                    metrics.loss_trades = len(pl_data[pl_data < 0])
                    
                    if metrics.total_trades > 0:
                        metrics.win_rate = metrics.profit_trades / metrics.total_trades
                        
                        if metrics.profit_trades > 0:
                            metrics.average_profit = pl_data[pl_data > 0].mean()
                        if metrics.loss_trades > 0:
                            metrics.average_loss = pl_data[pl_data < 0].mean()
                            
                        # プロフィットファクター計算
                        total_profit = pl_data[pl_data > 0].sum()
                        total_loss = abs(pl_data[pl_data < 0].sum())
                        if total_loss > 0:
                            metrics.profit_factor = total_profit / total_loss
                    
                    # シャープレシオ計算
                    if len(pl_data) > 1:
                        if pl_data.std() > 0:
                            metrics.sharpe_ratio = pl_data.mean() / pl_data.std() * np.sqrt(252)
                    
                    # 最大ドローダウン計算
                    cumulative_pl = pl_data.cumsum()
                    running_max = cumulative_pl.expanding().max()
                    drawdown = (cumulative_pl - running_max) / running_max
                    metrics.max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0
            
        except Exception as e:
            self.logger.error(f"パフォーマンスメトリクス計算エラー: {e}")
            
        return metrics
    
    def _validate_column_types(self, data: pd.DataFrame) -> List[str]:
        """カラムデータ型検証"""
        errors = []
        
        for col, expected_type in self.output_format_config.column_data_types.items():
            if col in data.columns:
                if expected_type == "datetime":
                    if not pd.api.types.is_datetime64_any_dtype(data[col]):
                        errors.append(f"カラム '{col}' のデータ型が不正: datetime型が期待されます")
                elif expected_type == "int":
                    if not pd.api.types.is_integer_dtype(data[col]):
                        errors.append(f"カラム '{col}' のデータ型が不正: int型が期待されます")
                elif expected_type == "float":
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        errors.append(f"カラム '{col}' のデータ型が不正: float型が期待されます")
        
        return errors
    
    def _validate_column_ranges(self, data: pd.DataFrame) -> List[str]:
        """カラム値範囲検証"""
        errors = []
        
        # シグナル列の範囲チェック
        signal_columns = ['Entry_Signal', 'Exit_Signal']
        for col in signal_columns:
            if col in data.columns:
                unique_values = data[col].dropna().unique()
                valid_values = [0, 1, -1]
                invalid_values = [v for v in unique_values if v not in valid_values]
                if invalid_values:
                    errors.append(f"カラム '{col}' に無効な値: {invalid_values}")
        
        # 価格・損益列の範囲チェック
        value_columns = ['Price', 'Profit_Loss', 'Position']
        for col in value_columns:
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    if np.any(np.isnan(col_data)) or np.any(np.isinf(col_data)):
                        errors.append(f"カラム '{col}' にNaNまたは無限値が含まれています")
        
        return errors
    
    def _detect_outliers(self, data: pd.DataFrame) -> List[str]:
        """外れ値検出"""
        outlier_info = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    
                    if std_val > 0:
                        outliers = np.abs((col_data - mean_val) / std_val) > self.data_consistency_thresholds.outlier_detection_std
                        outlier_count = outliers.sum()
                        
                        if outlier_count > 0:
                            outlier_ratio = outlier_count / len(col_data)
                            outlier_info.append(
                                f"カラム '{col}' で外れ値検出: {outlier_count}個 ({outlier_ratio:.2%})"
                            )
        
        return outlier_info
    
    def _validate_timestamps(self, data: pd.DataFrame) -> List[str]:
        """タイムスタンプ検証"""
        errors = []
        
        if 'Date' in data.columns:
            try:
                dates = pd.to_datetime(data['Date'], errors='coerce')
                
                # NaT（無効な日付）チェック
                nat_count = dates.isna().sum()
                if nat_count > 0:
                    errors.append(f"無効な日付が {nat_count} 個含まれています")
                
                # 日付順序チェック
                valid_dates = dates.dropna()
                if len(valid_dates) > 1:
                    if not valid_dates.is_monotonic_increasing:
                        errors.append("日付が昇順でありません")
                
                # 重複日付チェック
                duplicate_dates = valid_dates.duplicated().sum()
                if duplicate_dates > 0:
                    errors.append(f"重複する日付が {duplicate_dates} 個含まれています")
                    
            except Exception as e:
                errors.append(f"タイムスタンプ検証エラー: {e}")
        
        return errors
    
    def _merge_validation_results(self, main_result: ValidationResult, sub_result: ValidationResult) -> None:
        """検証結果マージ"""
        main_result.error_count += sub_result.error_count
        main_result.warning_count += sub_result.warning_count
        main_result.info_count += sub_result.info_count
        main_result.messages.extend(sub_result.messages)
        main_result.details.update(sub_result.details)
        
        # エラーレベル更新
        if sub_result.error_level == ErrorLevel.CRITICAL:
            main_result.error_level = ErrorLevel.CRITICAL
        elif (sub_result.error_level == ErrorLevel.WARNING and 
              main_result.error_level != ErrorLevel.CRITICAL):
            main_result.error_level = ErrorLevel.WARNING
        
        # 有効性判定
        if not sub_result.is_valid:
            main_result.is_valid = False
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """検証結果サマリー取得"""
        summary_lines = [
            "=== 出力データ検証結果 ===",
            f"検証状態: {'合格' if result.is_valid else '不合格'}",
            f"エラーレベル: {result.error_level.value}",
            f"エラー数: {result.error_count}",
            f"警告数: {result.warning_count}",
            f"情報数: {result.info_count}",
            f"検証時間: {result.validation_time}",
            ""
        ]
        
        if result.messages:
            summary_lines.append("=== 検証メッセージ ===")
            for i, message in enumerate(result.messages, 1):
                summary_lines.append(f"{i}. {message}")
            summary_lines.append("")
        
        if result.details:
            summary_lines.append("=== 検証詳細 ===")
            for key, value in result.details.items():
                summary_lines.append(f"{key}: {value}")
        
        return "\n".join(summary_lines)


if __name__ == "__main__":
    # テスト実行
    validator = OutputDataValidator()
    
    # テストデータ作成
    test_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100),
        'Entry_Signal': np.random.choice([0, 1], 100),
        'Exit_Signal': np.random.choice([0, 1], 100),
        'Position': np.random.uniform(-1, 1, 100),
        'Price': np.random.uniform(90, 110, 100),
        'Profit_Loss': np.random.normal(0, 0.01, 100),
        'Cumulative_Return': np.cumprod(1 + np.random.normal(0, 0.01, 100))
    })
    
    print("=== 出力データ検証システム テスト ===")
    result = validator.validate_output_data(test_data)
    print(validator.get_validation_summary(result))
