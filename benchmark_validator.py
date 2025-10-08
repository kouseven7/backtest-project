"""
パフォーマンスベンチマーク検証
5-3-3 戦略間相関を考慮した配分最適化システム 負荷テスト用

Author: imega
Created: 2025-07-24
Task: Load Testing for 5-3-3
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import logging
import os
from datetime import datetime

# 既存のパフォーマンス監視インポート
try:
    from performance_monitor import PerformanceMetrics
except ImportError:
    # フォールバック
    @dataclass
    class PerformanceMetrics:
        execution_time: float
        peak_memory_mb: float
        avg_cpu_percent: float
        memory_growth_mb: float
        gc_collections: int
        start_memory_mb: float
        end_memory_mb: float
        thread_count: int
        process_count: int

@dataclass 
class BenchmarkThresholds:
    """ベンチマーク閾値設定"""
    max_time: float
    max_memory: float
    max_cpu_percent: float = 80.0
    max_memory_growth: float = 100.0  # MB
    max_gc_collections: int = 10

@dataclass
class ValidationResult:
    """検証結果"""
    test_name: str
    passed: bool
    metrics: PerformanceMetrics
    thresholds: BenchmarkThresholds
    violations: List[str]
    score: float  # 0-100の総合スコア

class BenchmarkValidator:
    """パフォーマンスベンチマーク検証器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.benchmarks = self._load_benchmarks(config_file)
        self.validation_history: List[ValidationResult] = []
        
    def _load_benchmarks(self, config_file: Optional[str] = None) -> Dict[str, BenchmarkThresholds]:
        """ベンチマーク基準値読み込み"""
        
        default_benchmarks = {
            "small_scale": BenchmarkThresholds(
                max_time=10.0,
                max_memory=128.0,
                max_cpu_percent=70.0,
                max_memory_growth=50.0,
                max_gc_collections=5
            ),
            "medium_scale": BenchmarkThresholds(
                max_time=30.0,
                max_memory=256.0,
                max_cpu_percent=80.0,
                max_memory_growth=100.0,
                max_gc_collections=10
            ),
            "large_scale": BenchmarkThresholds(
                max_time=120.0,
                max_memory=512.0,
                max_cpu_percent=90.0,
                max_memory_growth=200.0,
                max_gc_collections=20
            ),
            "stress_test": BenchmarkThresholds(
                max_time=300.0,
                max_memory=1024.0,
                max_cpu_percent=95.0,
                max_memory_growth=500.0,
                max_gc_collections=50
            )
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 設定ファイルからベンチマークを更新
                for test_name, thresholds in config_data.get('benchmarks', {}).items():
                    default_benchmarks[test_name] = BenchmarkThresholds(**thresholds)
                    
                self.logger.info(f"Benchmarks loaded from: {config_file}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load config file: {e}, using defaults")
        
        return default_benchmarks
        
    def validate_performance(self, test_name: str, metrics: PerformanceMetrics) -> ValidationResult:
        """パフォーマンス検証"""
        
        # 対応するベンチマークを取得
        if test_name not in self.benchmarks:
            # デフォルトとして medium_scale を使用
            benchmark = self.benchmarks.get('medium_scale', self.benchmarks['small_scale'])
            self.logger.warning(f"No benchmark for {test_name}, using default")
        else:
            benchmark = self.benchmarks[test_name]
        
        violations = []
        
        # 各指標をチェック
        if metrics.execution_time > benchmark.max_time:
            violations.append(f"実行時間超過: {metrics.execution_time:.2f}s > {benchmark.max_time}s")
            
        if metrics.peak_memory_mb > benchmark.max_memory:
            violations.append(f"メモリ使用量超過: {metrics.peak_memory_mb:.1f}MB > {benchmark.max_memory}MB")
            
        if metrics.avg_cpu_percent > benchmark.max_cpu_percent:
            violations.append(f"CPU使用率超過: {metrics.avg_cpu_percent:.1f}% > {benchmark.max_cpu_percent}%")
            
        if metrics.memory_growth_mb > benchmark.max_memory_growth:
            violations.append(f"メモリ増加量超過: {metrics.memory_growth_mb:.1f}MB > {benchmark.max_memory_growth}MB")
            
        if metrics.gc_collections > benchmark.max_gc_collections:
            violations.append(f"GC回数超過: {metrics.gc_collections} > {benchmark.max_gc_collections}")
        
        # 総合スコア計算（0-100）
        score = self._calculate_score(metrics, benchmark)
        
        # 検証結果作成
        result = ValidationResult(
            test_name=test_name,
            passed=len(violations) == 0,
            metrics=metrics,
            thresholds=benchmark,
            violations=violations,
            score=score
        )
        
        self.validation_history.append(result)
        
        # ログ出力
        status = "PASS" if result.passed else "FAIL"
        self.logger.info(f"Validation {status}: {test_name} (Score: {score:.1f}/100)")
        
        if violations:
            for violation in violations:
                self.logger.warning(f"  - {violation}")
        
        return result
        
    def _calculate_score(self, metrics: PerformanceMetrics, benchmark: BenchmarkThresholds) -> float:
        """総合スコア計算（0-100）"""
        
        # 各指標のスコア（0-25ずつ、合計100）
        time_score = max(0, 25 * (1 - metrics.execution_time / benchmark.max_time))
        memory_score = max(0, 25 * (1 - metrics.peak_memory_mb / benchmark.max_memory))
        cpu_score = max(0, 25 * (1 - metrics.avg_cpu_percent / benchmark.max_cpu_percent))
        growth_score = max(0, 25 * (1 - metrics.memory_growth_mb / benchmark.max_memory_growth))
        
        total_score = time_score + memory_score + cpu_score + growth_score
        return min(100.0, max(0.0, total_score))
        
    def generate_performance_report(self, results: Optional[List[ValidationResult]] = None) -> str:
        """パフォーマンスレポート生成"""
        
        if results is None:
            results = self.validation_history
            
        if not results:
            return "検証結果がありません。"
        
        report_lines = [
            "=" * 80,
            "5-3-3 負荷テスト パフォーマンスレポート",
            "=" * 80,
            f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"テスト実行数: {len(results)}",
            ""
        ]
        
        # サマリー
        passed_count = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / len(results)
        
        report_lines.extend([
            "[CHART] 実行サマリー",
            "-" * 40,
            f"合格: {passed_count}/{len(results)} ({passed_count/len(results)*100:.1f}%)",
            f"平均スコア: {avg_score:.1f}/100",
            ""
        ])
        
        # 個別テスト結果
        report_lines.append("[LIST] 個別テスト結果")
        report_lines.append("-" * 40)
        
        for result in results:
            status_emoji = "[OK]" if result.passed else "[ERROR]"
            report_lines.extend([
                f"{status_emoji} {result.test_name} (スコア: {result.score:.1f}/100)",
                f"   実行時間: {result.metrics.execution_time:.2f}s",
                f"   ピークメモリ: {result.metrics.peak_memory_mb:.1f}MB", 
                f"   平均CPU: {result.metrics.avg_cpu_percent:.1f}%",
                f"   メモリ増加: {result.metrics.memory_growth_mb:.1f}MB",
                ""
            ])
            
            if result.violations:
                report_lines.append("   [WARNING] 違反項目:")
                for violation in result.violations:
                    report_lines.append(f"     - {violation}")
                report_lines.append("")
        
        # パフォーマンス傾向分析
        if len(results) > 1:
            report_lines.extend([
                "[UP] パフォーマンス傾向分析",
                "-" * 40
            ])
            
            # 実行時間分析
            exec_times = [r.metrics.execution_time for r in results]
            memory_peaks = [r.metrics.peak_memory_mb for r in results]
            
            report_lines.extend([
                f"実行時間: 最小 {min(exec_times):.2f}s, 最大 {max(exec_times):.2f}s, 平均 {sum(exec_times)/len(exec_times):.2f}s",
                f"メモリ使用: 最小 {min(memory_peaks):.1f}MB, 最大 {max(memory_peaks):.1f}MB, 平均 {sum(memory_peaks)/len(memory_peaks):.1f}MB",
                ""
            ])
        
        # 推奨事項
        report_lines.extend([
            "[IDEA] 推奨事項",
            "-" * 40
        ])
        
        failed_results = [r for r in results if not r.passed]
        if failed_results:
            report_lines.append("❗ 改善が必要な項目:")
            common_violations = {}
            for result in failed_results:
                for violation in result.violations:
                    violation_type = violation.split(':')[0]
                    common_violations[violation_type] = common_violations.get(violation_type, 0) + 1
                    
            for violation_type, count in sorted(common_violations.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  - {violation_type}: {count}件")
                
            report_lines.extend([
                "",
                "[TOOL] 対策案:",
                "  - アルゴリズムの最適化",
                "  - メモリ使用量の削減",
                "  - 並列処理の改善",
                "  - キャッシュ機構の導入",
            ])
        else:
            report_lines.append("✨ 全テストが基準を満たしています！")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
        
    def export_results(self, filepath: str, results: Optional[List[ValidationResult]] = None):
        """結果をJSONエクスポート"""
        
        if results is None:
            results = self.validation_history
            
        try:
            export_data = {
                "report_info": {
                    "generated_at": datetime.now().isoformat(),
                    "total_tests": len(results),
                    "passed_tests": sum(1 for r in results if r.passed),
                    "average_score": sum(r.score for r in results) / len(results) if results else 0
                },
                "test_results": [
                    {
                        "test_name": r.test_name,
                        "passed": r.passed,
                        "score": r.score,
                        "violations": r.violations,
                        "metrics": {
                            "execution_time": r.metrics.execution_time,
                            "peak_memory_mb": r.metrics.peak_memory_mb,
                            "avg_cpu_percent": r.metrics.avg_cpu_percent,
                            "memory_growth_mb": r.metrics.memory_growth_mb,
                            "gc_collections": r.metrics.gc_collections,
                            "start_memory_mb": r.metrics.start_memory_mb,
                            "end_memory_mb": r.metrics.end_memory_mb,
                            "thread_count": r.metrics.thread_count,
                            "process_count": r.metrics.process_count
                        },
                        "thresholds": {
                            "max_time": r.thresholds.max_time,
                            "max_memory": r.thresholds.max_memory,
                            "max_cpu_percent": r.thresholds.max_cpu_percent,
                            "max_memory_growth": r.thresholds.max_memory_growth,
                            "max_gc_collections": r.thresholds.max_gc_collections
                        }
                    }
                    for r in results
                ],
                "benchmarks": {
                    name: {
                        "max_time": bench.max_time,
                        "max_memory": bench.max_memory,
                        "max_cpu_percent": bench.max_cpu_percent,
                        "max_memory_growth": bench.max_memory_growth,
                        "max_gc_collections": bench.max_gc_collections
                    }
                    for name, bench in self.benchmarks.items()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Results exported to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            
    def clear_history(self):
        """検証履歴クリア"""
        self.validation_history.clear()
        self.logger.info("Validation history cleared")

# テスト用の実行例
if __name__ == "__main__":
    import time
    
    # ロギング設定
    logging.basicConfig(level=logging.INFO)
    
    validator = BenchmarkValidator()
    
    # 模擬メトリクス作成
    test_metrics = PerformanceMetrics(
        execution_time=5.2,
        peak_memory_mb=89.5,
        avg_cpu_percent=45.2,
        memory_growth_mb=25.3,
        gc_collections=3,
        start_memory_mb=64.2,
        end_memory_mb=89.5,
        thread_count=4,
        process_count=120
    )
    
    # 検証実行
    result = validator.validate_performance("small_scale", test_metrics)
    
    print(f"検証結果: {'合格' if result.passed else '不合格'}")
    print(f"スコア: {result.score:.1f}/100")
    
    if result.violations:
        print("違反項目:")
        for violation in result.violations:
            print(f"  - {violation}")
    
    # レポート生成
    print("\n" + "="*50)
    print(validator.generate_performance_report())
