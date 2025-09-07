"""
Consistency Checker
Phase 2.3 Task 2.3.3: 品質保証システム

Purpose:
  - バックテスト結果の一貫性チェック
  - 戦略間の整合性検証
  - 時系列データの一貫性確認
  - パラメータ設定の妥当性検証

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Integration:
  - qa_config_manager.pyとの連携
  - output_data_validator.pyとの統合
  - 既存validation系システムとの互換性保持
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from src.dssms.qa_config_manager import QAConfigManager, ErrorLevel


@dataclass
class ConsistencyResult:
    """一貫性チェック結果"""
    is_consistent: bool = True
    error_level: ErrorLevel = ErrorLevel.INFO
    inconsistency_count: int = 0
    warning_count: int = 0
    messages: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    check_time: Optional[datetime] = None


@dataclass
class StrategyConsistency:
    """戦略一貫性情報"""
    strategy_name: str = ""
    signal_consistency: float = 0.0
    parameter_consistency: float = 0.0
    performance_consistency: float = 0.0
    overall_consistency: float = 0.0
    issues: List[str] = field(default_factory=list)


class ConsistencyChecker:
    """一貫性チェックシステム"""
    
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
        
        self.logger.info("Consistency Checker 初期化完了")
    
    def check_backtest_consistency(self, 
                                 backtest_results: Dict[str, pd.DataFrame],
                                 metadata: Optional[Dict[str, Any]] = None) -> ConsistencyResult:
        """
        バックテスト結果一貫性チェック
        
        Args:
            backtest_results: 戦略別バックテスト結果
            metadata: メタデータ（期間、パラメータ等）
        
        Returns:
            ConsistencyResult: 一貫性チェック結果
        """
        start_time = datetime.now()
        result = ConsistencyResult(check_time=start_time)
        
        try:
            self.logger.info("バックテスト一貫性チェック開始")
            
            # 1. 時系列一貫性チェック
            timeseries_result = self._check_timeseries_consistency(backtest_results)
            self._merge_consistency_results(result, timeseries_result)
            
            # 2. シグナル一貫性チェック
            signal_result = self._check_signal_consistency(backtest_results)
            self._merge_consistency_results(result, signal_result)
            
            # 3. パフォーマンス一貫性チェック
            performance_result = self._check_performance_consistency(backtest_results)
            self._merge_consistency_results(result, performance_result)
            
            # 4. 戦略間一貫性チェック
            strategy_result = self._check_strategy_consistency(backtest_results)
            self._merge_consistency_results(result, strategy_result)
            
            # 5. データ品質一貫性チェック
            quality_result = self._check_data_quality_consistency(backtest_results)
            self._merge_consistency_results(result, quality_result)
            
            # 最終判定
            result.is_consistent = (result.inconsistency_count == 0)
            result.check_time = datetime.now()
            
            self.logger.info(f"バックテスト一貫性チェック完了: 不整合={result.inconsistency_count}, 警告={result.warning_count}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"一貫性チェックエラー: {e}")
            result.is_consistent = False
            result.error_level = ErrorLevel.CRITICAL
            result.inconsistency_count += 1
            result.messages.append(f"一貫性チェック処理エラー: {e}")
            return result
    
    def _check_timeseries_consistency(self, backtest_results: Dict[str, pd.DataFrame]) -> ConsistencyResult:
        """時系列一貫性チェック"""
        result = ConsistencyResult()
        
        try:
            date_ranges = {}
            date_frequencies = {}
            
            # 各戦略の時系列情報収集
            for strategy_name, data in backtest_results.items():
                if 'Date' in data.columns:
                    dates = pd.to_datetime(data['Date'], errors='coerce').dropna()
                    if len(dates) > 0:
                        date_ranges[strategy_name] = {
                            'start': dates.min(),
                            'end': dates.max(),
                            'count': len(dates)
                        }
                        
                        # 日付頻度チェック
                        if len(dates) > 1:
                            date_diffs = dates.diff().dropna()
                            mode_diff = date_diffs.mode()
                            if len(mode_diff) > 0:
                                date_frequencies[strategy_name] = mode_diff[0]
            
            # 時系列範囲一貫性チェック
            if len(date_ranges) > 1:
                start_dates = [info['start'] for info in date_ranges.values()]
                end_dates = [info['end'] for info in date_ranges.values()]
                
                # 開始日の一貫性
                start_diff = max(start_dates) - min(start_dates)
                if start_diff.days > 7:  # 1週間以上の差
                    result.warning_count += 1
                    result.messages.append(
                        f"戦略間で開始日に大きな差: {start_diff.days}日"
                    )
                
                # 終了日の一貫性
                end_diff = max(end_dates) - min(end_dates)
                if end_diff.days > 7:
                    result.warning_count += 1
                    result.messages.append(
                        f"戦略間で終了日に大きな差: {end_diff.days}日"
                    )
            
            # 日付頻度一貫性チェック
            if len(date_frequencies) > 1:
                unique_frequencies = set(date_frequencies.values())
                if len(unique_frequencies) > 1:
                    result.warning_count += 1
                    result.messages.append(
                        f"戦略間で日付頻度が不一致: {list(unique_frequencies)}"
                    )
            
            result.details['date_ranges'] = date_ranges
            result.details['date_frequencies'] = date_frequencies
            
            self.logger.info("時系列一貫性チェック完了")
            
        except Exception as e:
            result.inconsistency_count += 1
            result.error_level = ErrorLevel.CRITICAL
            result.messages.append(f"時系列一貫性チェックエラー: {e}")
            result.is_consistent = False
            
        return result
    
    def _check_signal_consistency(self, backtest_results: Dict[str, pd.DataFrame]) -> ConsistencyResult:
        """シグナル一貫性チェック"""
        result = ConsistencyResult()
        
        try:
            signal_stats = {}
            
            # 各戦略のシグナル統計収集
            for strategy_name, data in backtest_results.items():
                stats = {}
                
                if 'Entry_Signal' in data.columns:
                    entry_signals = data['Entry_Signal'].dropna()
                    stats['entry_signal_count'] = len(entry_signals[entry_signals != 0])
                    stats['entry_signal_rate'] = stats['entry_signal_count'] / len(data) if len(data) > 0 else 0
                
                if 'Exit_Signal' in data.columns:
                    exit_signals = data['Exit_Signal'].dropna()
                    stats['exit_signal_count'] = len(exit_signals[exit_signals != 0])
                    stats['exit_signal_rate'] = stats['exit_signal_count'] / len(data) if len(data) > 0 else 0
                
                # シグナル対称性チェック
                if 'Entry_Signal' in data.columns and 'Exit_Signal' in data.columns:
                    entry_count = len(data[data['Entry_Signal'] != 0])
                    exit_count = len(data[data['Exit_Signal'] != 0])
                    signal_balance = abs(entry_count - exit_count) / max(entry_count, exit_count, 1)
                    stats['signal_balance'] = signal_balance
                    
                    if signal_balance > 0.2:  # 20%以上の不均衡
                        result.warning_count += 1
                        result.messages.append(
                            f"戦略 '{strategy_name}' でシグナル不均衡: エントリー{entry_count}, エグジット{exit_count}"
                        )
                
                signal_stats[strategy_name] = stats
            
            # 戦略間シグナル頻度比較
            if len(signal_stats) > 1:
                entry_rates = [stats.get('entry_signal_rate', 0) for stats in signal_stats.values()]
                exit_rates = [stats.get('exit_signal_rate', 0) for stats in signal_stats.values()]
                
                # エントリーシグナル頻度の一貫性
                if entry_rates:
                    entry_rate_std = np.std(entry_rates)
                    entry_rate_mean = np.mean(entry_rates)
                    if entry_rate_mean > 0 and entry_rate_std / entry_rate_mean > 0.5:  # CV > 0.5
                        result.warning_count += 1
                        result.messages.append(
                            f"戦略間でエントリーシグナル頻度のばらつきが大きい: CV={entry_rate_std/entry_rate_mean:.2f}"
                        )
                
                # エグジットシグナル頻度の一貫性
                if exit_rates:
                    exit_rate_std = np.std(exit_rates)
                    exit_rate_mean = np.mean(exit_rates)
                    if exit_rate_mean > 0 and exit_rate_std / exit_rate_mean > 0.5:
                        result.warning_count += 1
                        result.messages.append(
                            f"戦略間でエグジットシグナル頻度のばらつきが大きい: CV={exit_rate_std/exit_rate_mean:.2f}"
                        )
            
            result.details['signal_stats'] = signal_stats
            
            self.logger.info("シグナル一貫性チェック完了")
            
        except Exception as e:
            result.inconsistency_count += 1
            result.error_level = ErrorLevel.CRITICAL
            result.messages.append(f"シグナル一貫性チェックエラー: {e}")
            result.is_consistent = False
            
        return result
    
    def _check_performance_consistency(self, backtest_results: Dict[str, pd.DataFrame]) -> ConsistencyResult:
        """パフォーマンス一貫性チェック"""
        result = ConsistencyResult()
        
        try:
            performance_metrics = {}
            
            # 各戦略のパフォーマンス計算
            for strategy_name, data in backtest_results.items():
                metrics = {}
                
                if 'Profit_Loss' in data.columns:
                    pl_data = data['Profit_Loss'].dropna()
                    if len(pl_data) > 0:
                        metrics['total_return'] = pl_data.sum()
                        metrics['win_rate'] = len(pl_data[pl_data > 0]) / len(pl_data[pl_data != 0]) if len(pl_data[pl_data != 0]) > 0 else 0
                        metrics['avg_return'] = pl_data.mean()
                        metrics['return_std'] = pl_data.std()
                        metrics['sharpe_ratio'] = metrics['avg_return'] / metrics['return_std'] if metrics['return_std'] > 0 else 0
                
                if 'Cumulative_Return' in data.columns:
                    cum_returns = data['Cumulative_Return'].dropna()
                    if len(cum_returns) > 0:
                        metrics['final_return'] = cum_returns.iloc[-1] - 1.0 if cum_returns.iloc[-1] > 1 else cum_returns.iloc[-1]
                        
                        # ドローダウン計算
                        running_max = cum_returns.expanding().max()
                        drawdown = (cum_returns - running_max) / running_max
                        metrics['max_drawdown'] = abs(drawdown.min()) if not drawdown.empty else 0.0
                
                performance_metrics[strategy_name] = metrics
            
            # パフォーマンス異常値検出
            if len(performance_metrics) > 1:
                for metric_name in ['total_return', 'sharpe_ratio', 'max_drawdown']:
                    values = [metrics.get(metric_name, 0) for metrics in performance_metrics.values()]
                    values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
                    
                    if len(values) > 1:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        
                        if std_val > 0:
                            for strategy_name, metrics in performance_metrics.items():
                                value = metrics.get(metric_name, 0)
                                if not np.isnan(value) and not np.isinf(value):
                                    z_score = abs((value - mean_val) / std_val)
                                    if z_score > 3:  # 3σ外れ値
                                        result.warning_count += 1
                                        result.messages.append(
                                            f"戦略 '{strategy_name}' の {metric_name} が異常値: {value:.4f} (Z-score: {z_score:.2f})"
                                        )
            
            # 極端なパフォーマンス値チェック
            for strategy_name, metrics in performance_metrics.items():
                if metrics.get('sharpe_ratio', 0) > 5:
                    result.warning_count += 1
                    result.messages.append(
                        f"戦略 '{strategy_name}' のシャープレシオが異常に高い: {metrics['sharpe_ratio']:.2f}"
                    )
                
                if metrics.get('max_drawdown', 0) > 0.8:
                    result.warning_count += 1
                    result.messages.append(
                        f"戦略 '{strategy_name}' の最大ドローダウンが異常に高い: {metrics['max_drawdown']:.2%}"
                    )
                
                if metrics.get('win_rate', 0) > 0.95:
                    result.warning_count += 1
                    result.messages.append(
                        f"戦略 '{strategy_name}' の勝率が異常に高い: {metrics['win_rate']:.2%}"
                    )
            
            result.details['performance_metrics'] = performance_metrics
            
            self.logger.info("パフォーマンス一貫性チェック完了")
            
        except Exception as e:
            result.inconsistency_count += 1
            result.error_level = ErrorLevel.CRITICAL
            result.messages.append(f"パフォーマンス一貫性チェックエラー: {e}")
            result.is_consistent = False
            
        return result
    
    def _check_strategy_consistency(self, backtest_results: Dict[str, pd.DataFrame]) -> ConsistencyResult:
        """戦略間一貫性チェック"""
        result = ConsistencyResult()
        
        try:
            strategy_consistency = {}
            
            # 各戦略の一貫性評価
            for strategy_name, data in backtest_results.items():
                consistency = StrategyConsistency(strategy_name=strategy_name)
                
                # シグナル一貫性（エントリーとエグジットのバランス）
                if 'Entry_Signal' in data.columns and 'Exit_Signal' in data.columns:
                    entry_count = len(data[data['Entry_Signal'] != 0])
                    exit_count = len(data[data['Exit_Signal'] != 0])
                    if entry_count + exit_count > 0:
                        consistency.signal_consistency = 1.0 - abs(entry_count - exit_count) / (entry_count + exit_count)
                
                # パフォーマンス一貫性（リターンの安定性）
                if 'Profit_Loss' in data.columns:
                    pl_data = data['Profit_Loss'].dropna()
                    if len(pl_data) > 1 and pl_data.std() > 0:
                        # シャープレシオベースの一貫性スコア
                        sharpe = pl_data.mean() / pl_data.std()
                        consistency.performance_consistency = max(0, min(1, (sharpe + 2) / 4))  # -2～2を0～1にマップ
                
                # 総合一貫性スコア
                consistency.overall_consistency = (
                    consistency.signal_consistency * 0.4 +
                    consistency.parameter_consistency * 0.2 +
                    consistency.performance_consistency * 0.4
                )
                
                # 一貫性が低い場合の警告
                if consistency.overall_consistency < 0.6:
                    result.warning_count += 1
                    result.messages.append(
                        f"戦略 '{strategy_name}' の一貫性が低い: {consistency.overall_consistency:.2f}"
                    )
                    consistency.issues.append("一貫性スコアが低い")
                
                strategy_consistency[strategy_name] = consistency
            
            # 戦略間相関チェック（同じシグナルタイミングの検出）
            strategy_names = list(backtest_results.keys())
            if len(strategy_names) > 1:
                for i, strategy1 in enumerate(strategy_names):
                    for j, strategy2 in enumerate(strategy_names[i+1:], i+1):
                        correlation = self._calculate_signal_correlation(
                            backtest_results[strategy1], 
                            backtest_results[strategy2]
                        )
                        
                        if correlation > 0.8:  # 高い相関
                            result.warning_count += 1
                            result.messages.append(
                                f"戦略 '{strategy1}' と '{strategy2}' のシグナル相関が高い: {correlation:.2f}"
                            )
            
            result.details['strategy_consistency'] = {
                name: {
                    'signal_consistency': sc.signal_consistency,
                    'performance_consistency': sc.performance_consistency,
                    'overall_consistency': sc.overall_consistency,
                    'issues': sc.issues
                }
                for name, sc in strategy_consistency.items()
            }
            
            self.logger.info("戦略間一貫性チェック完了")
            
        except Exception as e:
            result.inconsistency_count += 1
            result.error_level = ErrorLevel.CRITICAL
            result.messages.append(f"戦略間一貫性チェックエラー: {e}")
            result.is_consistent = False
            
        return result
    
    def _check_data_quality_consistency(self, backtest_results: Dict[str, pd.DataFrame]) -> ConsistencyResult:
        """データ品質一貫性チェック"""
        result = ConsistencyResult()
        
        try:
            quality_metrics = {}
            
            # 各戦略のデータ品質評価
            for strategy_name, data in backtest_results.items():
                metrics = {}
                
                # 欠損値比率
                total_cells = len(data) * len(data.columns)
                missing_cells = data.isnull().sum().sum()
                metrics['missing_ratio'] = missing_cells / total_cells if total_cells > 0 else 0
                
                # 重複行比率
                metrics['duplicate_ratio'] = data.duplicated().sum() / len(data) if len(data) > 0 else 0
                
                # 異常値比率（数値列のみ）
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                outlier_count = 0
                total_numeric_values = 0
                
                for col in numeric_cols:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        total_numeric_values += len(col_data)
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outliers = (col_data < lower_bound) | (col_data > upper_bound)
                            outlier_count += outliers.sum()
                
                metrics['outlier_ratio'] = outlier_count / total_numeric_values if total_numeric_values > 0 else 0
                
                quality_metrics[strategy_name] = metrics
            
            # データ品質の戦略間比較
            if len(quality_metrics) > 1:
                for metric_name in ['missing_ratio', 'duplicate_ratio', 'outlier_ratio']:
                    values = [metrics[metric_name] for metrics in quality_metrics.values()]
                    
                    if len(values) > 1:
                        max_val = max(values)
                        min_val = min(values)
                        
                        # 品質のばらつきチェック
                        if max_val - min_val > 0.05:  # 5%以上の差
                            result.warning_count += 1
                            result.messages.append(
                                f"戦略間で{metric_name}のばらつきが大きい: 最大{max_val:.2%}, 最小{min_val:.2%}"
                            )
                
                # 異常に品質が悪い戦略の検出
                for strategy_name, metrics in quality_metrics.items():
                    if metrics['missing_ratio'] > self.data_consistency_thresholds.missing_data_threshold:
                        result.warning_count += 1
                        result.messages.append(
                            f"戦略 '{strategy_name}' の欠損値比率が高い: {metrics['missing_ratio']:.2%}"
                        )
                    
                    if metrics['duplicate_ratio'] > self.data_consistency_thresholds.duplicate_data_threshold:
                        result.warning_count += 1
                        result.messages.append(
                            f"戦略 '{strategy_name}' の重複データ比率が高い: {metrics['duplicate_ratio']:.2%}"
                        )
            
            result.details['quality_metrics'] = quality_metrics
            
            self.logger.info("データ品質一貫性チェック完了")
            
        except Exception as e:
            result.inconsistency_count += 1
            result.error_level = ErrorLevel.CRITICAL
            result.messages.append(f"データ品質一貫性チェックエラー: {e}")
            result.is_consistent = False
            
        return result
    
    def _calculate_signal_correlation(self, data1: pd.DataFrame, data2: pd.DataFrame) -> float:
        """シグナル相関計算"""
        try:
            if 'Date' in data1.columns and 'Date' in data2.columns:
                # 日付で結合
                merged = pd.merge(data1[['Date', 'Entry_Signal', 'Exit_Signal']], 
                                data2[['Date', 'Entry_Signal', 'Exit_Signal']], 
                                on='Date', suffixes=('_1', '_2'))
                
                if len(merged) > 0:
                    # エントリーシグナルの相関
                    entry_corr = merged['Entry_Signal_1'].corr(merged['Entry_Signal_2'])
                    # エグジットシグナルの相関
                    exit_corr = merged['Exit_Signal_1'].corr(merged['Exit_Signal_2'])
                    
                    # 平均相関
                    correlations = [corr for corr in [entry_corr, exit_corr] if not np.isnan(corr)]
                    return np.mean(correlations) if correlations else 0.0
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _merge_consistency_results(self, main_result: ConsistencyResult, sub_result: ConsistencyResult) -> None:
        """一貫性チェック結果マージ"""
        main_result.inconsistency_count += sub_result.inconsistency_count
        main_result.warning_count += sub_result.warning_count
        main_result.messages.extend(sub_result.messages)
        main_result.details.update(sub_result.details)
        
        # エラーレベル更新
        if sub_result.error_level == ErrorLevel.CRITICAL:
            main_result.error_level = ErrorLevel.CRITICAL
        elif (sub_result.error_level == ErrorLevel.WARNING and 
              main_result.error_level != ErrorLevel.CRITICAL):
            main_result.error_level = ErrorLevel.WARNING
        
        # 一貫性判定
        if not sub_result.is_consistent:
            main_result.is_consistent = False
    
    def get_consistency_summary(self, result: ConsistencyResult) -> str:
        """一貫性チェック結果サマリー取得"""
        summary_lines = [
            "=== 一貫性チェック結果 ===",
            f"一貫性状態: {'一貫' if result.is_consistent else '不一致'}",
            f"エラーレベル: {result.error_level.value}",
            f"不整合数: {result.inconsistency_count}",
            f"警告数: {result.warning_count}",
            f"チェック時間: {result.check_time}",
            ""
        ]
        
        if result.messages:
            summary_lines.append("=== チェックメッセージ ===")
            for i, message in enumerate(result.messages, 1):
                summary_lines.append(f"{i}. {message}")
            summary_lines.append("")
        
        if result.details:
            summary_lines.append("=== チェック詳細 ===")
            for key, value in result.details.items():
                summary_lines.append(f"{key}: {value}")
        
        return "\n".join(summary_lines)


if __name__ == "__main__":
    # テスト実行
    checker = ConsistencyChecker()
    
    # テストデータ作成
    test_results = {
        'Strategy1': pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'Entry_Signal': np.random.choice([0, 1], 100),
            'Exit_Signal': np.random.choice([0, 1], 100),
            'Profit_Loss': np.random.normal(0, 0.01, 100),
            'Cumulative_Return': np.cumprod(1 + np.random.normal(0, 0.01, 100))
        }),
        'Strategy2': pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'Entry_Signal': np.random.choice([0, 1], 100),
            'Exit_Signal': np.random.choice([0, 1], 100),
            'Profit_Loss': np.random.normal(0, 0.015, 100),
            'Cumulative_Return': np.cumprod(1 + np.random.normal(0, 0.015, 100))
        })
    }
    
    print("=== 一貫性チェックシステム テスト ===")
    result = checker.check_backtest_consistency(test_results)
    print(checker.get_consistency_summary(result))
