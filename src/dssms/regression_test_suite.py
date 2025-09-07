"""
Regression Test Suite
Phase 2.3 Task 2.3.3: 品質保証システム

Purpose:
  - バックテスト結果のリグレッションテスト
  - ベースライン比較による変更検証
  - パフォーマンス回帰検出
  - 自動テストケース管理

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Integration:
  - qa_config_manager.pyとの連携
  - ベースラインデータとの自動比較
  - CIパイプライン対応
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import pickle
import hashlib
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from src.dssms.qa_config_manager import QAConfigManager, ErrorLevel


@dataclass
class TestCase:
    """テストケース定義"""
    name: str = ""
    description: str = ""
    test_type: str = "performance"  # performance, output_format, consistency
    baseline_data: Optional[Dict[str, Any]] = None
    tolerance: float = 0.05
    enabled: bool = True
    created_at: Optional[datetime] = None
    last_run: Optional[datetime] = None


@dataclass
class RegressionResult:
    """リグレッションテスト結果"""
    test_name: str = ""
    passed: bool = True
    error_level: ErrorLevel = ErrorLevel.INFO
    deviation: float = 0.0
    tolerance: float = 0.05
    baseline_value: Optional[Any] = None
    current_value: Optional[Any] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionTestReport:
    """リグレッションテストレポート"""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    test_results: List[RegressionResult] = field(default_factory=list)
    overall_result: bool = True
    execution_time: Optional[datetime] = None
    summary: str = ""


class RegressionTestSuite:
    """リグレッションテストスイート"""
    
    def __init__(self, config_manager: Optional[QAConfigManager] = None):
        """
        初期化
        
        Args:
            config_manager: 設定管理システム（Noneの場合は新規作成）
        """
        self.config_manager = config_manager or QAConfigManager()
        self.logger = setup_logger(__name__)
        
        # 設定取得
        self.regression_config = self.config_manager.get_regression_testing_config()
        
        # ディレクトリ設定
        self.test_cases_dir = Path(project_root) / self.regression_config.test_cases_directory
        self.baseline_data_dir = Path(project_root) / self.regression_config.baseline_data_directory
        
        # ディレクトリ作成
        self.test_cases_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_data_dir.mkdir(parents=True, exist_ok=True)
        
        # テストケース読み込み
        self.test_cases: Dict[str, TestCase] = {}
        self.load_test_cases()
        
        self.logger.info("Regression Test Suite 初期化完了")
    
    def run_regression_tests(self, 
                           backtest_results: Dict[str, pd.DataFrame],
                           metadata: Optional[Dict[str, Any]] = None) -> RegressionTestReport:
        """
        リグレッションテスト実行
        
        Args:
            backtest_results: 現在のバックテスト結果
            metadata: メタデータ
        
        Returns:
            RegressionTestReport: テスト実行レポート
        """
        start_time = datetime.now()
        report = RegressionTestReport(execution_time=start_time)
        
        try:
            self.logger.info("リグレッションテスト開始")
            
            if not self.regression_config.enabled:
                self.logger.info("リグレッションテストが無効化されています")
                report.skipped_tests = len(self.test_cases)
                report.summary = "リグレッションテストが無効化されています"
                return report
            
            # 各テストケース実行
            for test_name, test_case in self.test_cases.items():
                if not test_case.enabled:
                    report.skipped_tests += 1
                    continue
                
                try:
                    result = self._run_single_test(test_case, backtest_results, metadata)
                    report.test_results.append(result)
                    
                    if result.passed:
                        report.passed_tests += 1
                    else:
                        report.failed_tests += 1
                        
                    # テストケース実行時刻更新
                    test_case.last_run = datetime.now()
                    
                except Exception as e:
                    self.logger.error(f"テストケース '{test_name}' 実行エラー: {e}")
                    failed_result = RegressionResult(
                        test_name=test_name,
                        passed=False,
                        error_level=ErrorLevel.CRITICAL,
                        message=f"テスト実行エラー: {e}"
                    )
                    report.test_results.append(failed_result)
                    report.failed_tests += 1
            
            # 総合判定
            report.total_tests = len(self.test_cases)
            report.overall_result = (report.failed_tests == 0)
            
            # サマリー作成
            report.summary = self._generate_test_summary(report)
            
            # テストケース保存
            self.save_test_cases()
            
            self.logger.info(f"リグレッションテスト完了: 合格={report.passed_tests}, 失敗={report.failed_tests}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"リグレッションテスト実行エラー: {e}")
            report.overall_result = False
            report.summary = f"テスト実行エラー: {e}"
            return report
    
    def _run_single_test(self, 
                        test_case: TestCase, 
                        backtest_results: Dict[str, pd.DataFrame],
                        metadata: Optional[Dict[str, Any]] = None) -> RegressionResult:
        """単一テストケース実行"""
        result = RegressionResult(
            test_name=test_case.name,
            tolerance=test_case.tolerance
        )
        
        try:
            if test_case.test_type == "performance":
                result = self._test_performance_regression(test_case, backtest_results, result)
            elif test_case.test_type == "output_format":
                result = self._test_output_format_regression(test_case, backtest_results, result)
            elif test_case.test_type == "consistency":
                result = self._test_consistency_regression(test_case, backtest_results, result)
            else:
                result.passed = False
                result.error_level = ErrorLevel.WARNING
                result.message = f"未知のテストタイプ: {test_case.test_type}"
            
        except Exception as e:
            result.passed = False
            result.error_level = ErrorLevel.CRITICAL
            result.message = f"テスト実行エラー: {e}"
        
        return result
    
    def _test_performance_regression(self, 
                                   test_case: TestCase, 
                                   backtest_results: Dict[str, pd.DataFrame],
                                   result: RegressionResult) -> RegressionResult:
        """パフォーマンスリグレッションテスト"""
        if not test_case.baseline_data:
            result.passed = False
            result.message = "ベースラインデータが設定されていません"
            return result
        
        # 現在のパフォーマンス計算
        current_performance = self._calculate_performance_metrics(backtest_results)
        
        # ベースラインと比較
        baseline_performance = test_case.baseline_data
        
        # 主要メトリクスの比較
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for metric in key_metrics:
            if metric in baseline_performance and metric in current_performance:
                baseline_val = baseline_performance[metric]
                current_val = current_performance[metric]
                
                if baseline_val != 0:
                    deviation = abs((current_val - baseline_val) / baseline_val)
                else:
                    deviation = abs(current_val)
                
                if deviation > test_case.tolerance:
                    result.passed = False
                    result.error_level = ErrorLevel.WARNING
                    result.deviation = max(result.deviation, deviation)
                    result.message += f"{metric}: {deviation:.2%} 差異 (閾値: {test_case.tolerance:.2%}); "
        
        result.baseline_value = baseline_performance
        result.current_value = current_performance
        result.details = {
            'metric_comparisons': {
                metric: {
                    'baseline': baseline_performance.get(metric),
                    'current': current_performance.get(metric),
                    'deviation': abs((current_performance.get(metric, 0) - baseline_performance.get(metric, 0)) / baseline_performance.get(metric, 1)) if baseline_performance.get(metric, 0) != 0 else 0
                }
                for metric in key_metrics
                if metric in baseline_performance and metric in current_performance
            }
        }
        
        if result.passed:
            result.message = "パフォーマンス回帰なし"
        
        return result
    
    def _test_output_format_regression(self, 
                                     test_case: TestCase, 
                                     backtest_results: Dict[str, pd.DataFrame],
                                     result: RegressionResult) -> RegressionResult:
        """出力フォーマットリグレッションテスト"""
        if not test_case.baseline_data:
            result.passed = False
            result.message = "ベースラインデータが設定されていません"
            return result
        
        # 現在の出力フォーマット情報取得
        current_format = self._extract_format_info(backtest_results)
        baseline_format = test_case.baseline_data
        
        # カラム構造比較
        format_issues = []
        
        for strategy_name, current_info in current_format.items():
            if strategy_name in baseline_format:
                baseline_info = baseline_format[strategy_name]
                
                # カラム数比較
                if current_info['column_count'] != baseline_info['column_count']:
                    format_issues.append(f"{strategy_name}: カラム数変更 {baseline_info['column_count']} -> {current_info['column_count']}")
                
                # カラム名比較
                current_columns = set(current_info['columns'])
                baseline_columns = set(baseline_info['columns'])
                
                missing_columns = baseline_columns - current_columns
                extra_columns = current_columns - baseline_columns
                
                if missing_columns:
                    format_issues.append(f"{strategy_name}: 削除されたカラム {list(missing_columns)}")
                
                if extra_columns:
                    format_issues.append(f"{strategy_name}: 追加されたカラム {list(extra_columns)}")
                
                # データ型比較
                for col in current_columns & baseline_columns:
                    if (current_info['data_types'].get(col) != 
                        baseline_info['data_types'].get(col)):
                        format_issues.append(
                            f"{strategy_name}.{col}: データ型変更 "
                            f"{baseline_info['data_types'].get(col)} -> {current_info['data_types'].get(col)}"
                        )
        
        if format_issues:
            result.passed = False
            result.error_level = ErrorLevel.WARNING
            result.message = "; ".join(format_issues)
        else:
            result.message = "出力フォーマット変更なし"
        
        result.baseline_value = baseline_format
        result.current_value = current_format
        result.details = {'format_issues': format_issues}
        
        return result
    
    def _test_consistency_regression(self, 
                                   test_case: TestCase, 
                                   backtest_results: Dict[str, pd.DataFrame],
                                   result: RegressionResult) -> RegressionResult:
        """一貫性リグレッションテスト"""
        # 現在の一貫性スコア計算
        current_consistency = self._calculate_consistency_scores(backtest_results)
        
        if test_case.baseline_data:
            baseline_consistency = test_case.baseline_data
            
            # 一貫性スコア比較
            consistency_degradation = []
            
            for strategy_name, current_score in current_consistency.items():
                if strategy_name in baseline_consistency:
                    baseline_score = baseline_consistency[strategy_name]
                    score_diff = baseline_score - current_score
                    
                    if score_diff > test_case.tolerance:
                        consistency_degradation.append(
                            f"{strategy_name}: 一貫性低下 {baseline_score:.2f} -> {current_score:.2f}"
                        )
            
            if consistency_degradation:
                result.passed = False
                result.error_level = ErrorLevel.WARNING
                result.message = "; ".join(consistency_degradation)
            else:
                result.message = "一貫性回帰なし"
        else:
            result.message = "一貫性ベースライン未設定"
        
        result.baseline_value = test_case.baseline_data
        result.current_value = current_consistency
        
        return result
    
    def _calculate_performance_metrics(self, backtest_results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """パフォーマンスメトリクス計算"""
        metrics = {}
        
        all_pl = []
        all_returns = []
        
        for strategy_name, data in backtest_results.items():
            if 'Profit_Loss' in data.columns:
                pl_data = data['Profit_Loss'].dropna()
                all_pl.extend(pl_data.tolist())
            
            if 'Cumulative_Return' in data.columns:
                returns = data['Cumulative_Return'].dropna()
                if len(returns) > 0:
                    all_returns.append(returns.iloc[-1])
        
        if all_pl:
            metrics['total_return'] = sum(all_pl)
            metrics['avg_return'] = np.mean(all_pl)
            metrics['return_std'] = np.std(all_pl)
            metrics['sharpe_ratio'] = metrics['avg_return'] / metrics['return_std'] if metrics['return_std'] > 0 else 0
            metrics['win_rate'] = len([pl for pl in all_pl if pl > 0]) / len([pl for pl in all_pl if pl != 0]) if len([pl for pl in all_pl if pl != 0]) > 0 else 0
            
            # ドローダウン計算
            cumulative_pl = np.cumsum(all_pl)
            running_max = np.maximum.accumulate(cumulative_pl)
            drawdown = (cumulative_pl - running_max) / np.maximum(running_max, 1)
            metrics['max_drawdown'] = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        return metrics
    
    def _extract_format_info(self, backtest_results: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """出力フォーマット情報抽出"""
        format_info = {}
        
        for strategy_name, data in backtest_results.items():
            info = {
                'column_count': len(data.columns),
                'columns': list(data.columns),
                'data_types': {col: str(data[col].dtype) for col in data.columns},
                'row_count': len(data)
            }
            format_info[strategy_name] = info
        
        return format_info
    
    def _calculate_consistency_scores(self, backtest_results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """一貫性スコア計算"""
        consistency_scores = {}
        
        for strategy_name, data in backtest_results.items():
            score = 1.0  # 基本スコア
            
            # シグナル一貫性
            if 'Entry_Signal' in data.columns and 'Exit_Signal' in data.columns:
                entry_count = len(data[data['Entry_Signal'] != 0])
                exit_count = len(data[data['Exit_Signal'] != 0])
                if entry_count + exit_count > 0:
                    signal_balance = 1.0 - abs(entry_count - exit_count) / (entry_count + exit_count)
                    score *= signal_balance
            
            # パフォーマンス一貫性
            if 'Profit_Loss' in data.columns:
                pl_data = data['Profit_Loss'].dropna()
                if len(pl_data) > 1:
                    # 収益の安定性（変動係数の逆数）
                    if pl_data.mean() != 0:
                        cv = abs(pl_data.std() / pl_data.mean())
                        stability = 1.0 / (1.0 + cv)
                        score *= stability
            
            consistency_scores[strategy_name] = score
        
        return consistency_scores
    
    def create_test_case(self, 
                        name: str, 
                        test_type: str,
                        description: str = "",
                        tolerance: float = 0.05) -> TestCase:
        """新しいテストケース作成"""
        test_case = TestCase(
            name=name,
            description=description,
            test_type=test_type,
            tolerance=tolerance,
            created_at=datetime.now()
        )
        
        self.test_cases[name] = test_case
        self.logger.info(f"テストケース作成: {name}")
        
        return test_case
    
    def update_baseline(self, 
                       test_name: str, 
                       backtest_results: Dict[str, pd.DataFrame]) -> None:
        """ベースラインデータ更新"""
        if test_name not in self.test_cases:
            self.logger.warning(f"テストケースが見つかりません: {test_name}")
            return
        
        test_case = self.test_cases[test_name]
        
        if test_case.test_type == "performance":
            baseline_data = self._calculate_performance_metrics(backtest_results)
        elif test_case.test_type == "output_format":
            baseline_data = self._extract_format_info(backtest_results)
        elif test_case.test_type == "consistency":
            baseline_data = self._calculate_consistency_scores(backtest_results)
        else:
            self.logger.warning(f"未対応のテストタイプ: {test_case.test_type}")
            return
        
        test_case.baseline_data = baseline_data
        
        # ベースラインデータをファイルに保存
        baseline_file = self.baseline_data_dir / f"{test_name}_baseline.pkl"
        with open(baseline_file, 'wb') as f:
            pickle.dump(baseline_data, f)
        
        self.logger.info(f"ベースライン更新: {test_name}")
    
    def load_test_cases(self) -> None:
        """テストケース読み込み"""
        test_cases_file = self.test_cases_dir / "test_cases.json"
        
        if not test_cases_file.exists():
            self.logger.info("テストケースファイルが見つかりません。新規作成します。")
            self._create_default_test_cases()
            return
        
        try:
            with open(test_cases_file, 'r', encoding='utf-8') as f:
                test_cases_data = json.load(f)
            
            for name, data in test_cases_data.items():
                test_case = TestCase(**data)
                
                # ベースラインデータ読み込み
                baseline_file = self.baseline_data_dir / f"{name}_baseline.pkl"
                if baseline_file.exists():
                    try:
                        with open(baseline_file, 'rb') as f:
                            test_case.baseline_data = pickle.load(f)
                    except Exception as e:
                        self.logger.warning(f"ベースラインデータ読み込みエラー ({name}): {e}")
                
                self.test_cases[name] = test_case
            
            self.logger.info(f"テストケース読み込み完了: {len(self.test_cases)}件")
            
        except Exception as e:
            self.logger.error(f"テストケース読み込みエラー: {e}")
            self._create_default_test_cases()
    
    def save_test_cases(self) -> None:
        """テストケース保存"""
        try:
            test_cases_data = {}
            
            for name, test_case in self.test_cases.items():
                data = {
                    'name': test_case.name,
                    'description': test_case.description,
                    'test_type': test_case.test_type,
                    'tolerance': test_case.tolerance,
                    'enabled': test_case.enabled,
                    'created_at': test_case.created_at.isoformat() if test_case.created_at else None,
                    'last_run': test_case.last_run.isoformat() if test_case.last_run else None
                }
                test_cases_data[name] = data
            
            test_cases_file = self.test_cases_dir / "test_cases.json"
            with open(test_cases_file, 'w', encoding='utf-8') as f:
                json.dump(test_cases_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("テストケース保存完了")
            
        except Exception as e:
            self.logger.error(f"テストケース保存エラー: {e}")
    
    def _create_default_test_cases(self) -> None:
        """デフォルトテストケース作成"""
        default_cases = [
            {
                'name': 'performance_regression',
                'description': '主要パフォーマンスメトリクスの回帰テスト',
                'test_type': 'performance',
                'tolerance': 0.05
            },
            {
                'name': 'output_format_stability',
                'description': '出力フォーマットの安定性テスト',
                'test_type': 'output_format',
                'tolerance': 0.0
            },
            {
                'name': 'consistency_maintenance',
                'description': '戦略一貫性の維持テスト',
                'test_type': 'consistency',
                'tolerance': 0.1
            }
        ]
        
        for case_data in default_cases:
            test_case = TestCase(**case_data, created_at=datetime.now())
            self.test_cases[case_data['name']] = test_case
        
        self.save_test_cases()
        self.logger.info("デフォルトテストケース作成完了")
    
    def _generate_test_summary(self, report: RegressionTestReport) -> str:
        """テストサマリー生成"""
        summary_lines = [
            f"総テスト数: {report.total_tests}",
            f"合格: {report.passed_tests}",
            f"失敗: {report.failed_tests}",
            f"スキップ: {report.skipped_tests}",
            f"成功率: {(report.passed_tests / max(report.total_tests, 1)) * 100:.1f}%"
        ]
        
        if report.failed_tests > 0:
            summary_lines.append("\n失敗したテスト:")
            for result in report.test_results:
                if not result.passed:
                    summary_lines.append(f"- {result.test_name}: {result.message}")
        
        return "\n".join(summary_lines)
    
    def get_test_report_summary(self, report: RegressionTestReport) -> str:
        """テストレポートサマリー取得"""
        summary_lines = [
            "=== リグレッションテストレポート ===",
            f"実行時間: {report.execution_time}",
            f"総合結果: {'合格' if report.overall_result else '失敗'}",
            "",
            report.summary,
            ""
        ]
        
        if report.test_results:
            summary_lines.append("=== 詳細結果 ===")
            for result in report.test_results:
                status = "合格" if result.passed else "失敗"
                summary_lines.append(f"{result.test_name}: {status}")
                if result.message:
                    summary_lines.append(f"  メッセージ: {result.message}")
                if result.deviation > 0:
                    summary_lines.append(f"  差異: {result.deviation:.2%}")
        
        return "\n".join(summary_lines)


if __name__ == "__main__":
    # テスト実行
    test_suite = RegressionTestSuite()
    
    # テストデータ作成
    test_results = {
        'Strategy1': pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'Entry_Signal': np.random.choice([0, 1], 100),
            'Exit_Signal': np.random.choice([0, 1], 100),
            'Profit_Loss': np.random.normal(0, 0.01, 100),
            'Cumulative_Return': np.cumprod(1 + np.random.normal(0, 0.01, 100))
        })
    }
    
    print("=== リグレッションテストシステム テスト ===")
    
    # ベースライン更新
    test_suite.update_baseline('performance_regression', test_results)
    
    # テスト実行
    report = test_suite.run_regression_tests(test_results)
    print(test_suite.get_test_report_summary(report))
