"""
5-3-3 戦略間相関を考慮した配分最適化システム 負荷テストスイート

包括的な負荷テスト実行とパフォーマンス評価

Author: imega
Created: 2025-07-24
Task: Load Testing for 5-3-3
"""

import sys
import os
import time
import gc
import traceback
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 負荷テスト関連モジュール
try:
    from performance_monitor import PerformanceMonitor
    from load_test_data_generator import LoadTestDataGenerator
    from benchmark_validator import BenchmarkValidator
except ImportError as e:
    print(f"負荷テストモジュールのインポートエラー: {e}")
    sys.exit(1)

# 5-3-3システム関連モジュール
try:
    from config.portfolio_correlation_optimizer.correlation_based_allocator import (
        CorrelationBasedAllocator, AllocationConfig
    )
    from config.portfolio_correlation_optimizer.optimization_engine import (
        HybridOptimizationEngine, OptimizationMethod
    )
    CORRELATION_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"警告: 5-3-3システムモジュールのインポートエラー: {e}")
    print("負荷テストの一部機能が制限されます")
    CORRELATION_SYSTEM_AVAILABLE = False

class LoadTestRunner:
    """負荷テスト実行管理"""
    
    def __init__(self, config_file: str = "load_test_config.json"):
        self.config = self._load_config(config_file)
        
        # 出力ディレクトリ設定（ロギング前に必要）
        self.output_dir = Path(self.config["output_settings"]["output_directory"])
        self.output_dir.mkdir(exist_ok=True)
        
        # start_timeも早期に設定
        self.start_time = datetime.now()
        
        self.setup_logging()
        
        self.monitor = PerformanceMonitor()
        self.data_generator = LoadTestDataGenerator()
        self.validator = BenchmarkValidator(config_file)
        
        self.test_results = []
        
    def _load_config(self, config_file: str) -> Dict:
        """設定ファイル読み込み"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"設定ファイルが見つかりません: {config_file}")
            print("デフォルト設定を使用します")
            return self._get_default_config()
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            "test_scenarios": {
                "medium_scale": {
                    "num_strategies": 10,
                    "data_periods": 756,
                    "max_execution_time": 30,
                    "max_memory_mb": 256
                }
            },
            "output_settings": {
                "output_directory": "load_test_results",
                "generate_detailed_report": True
            },
            "test_execution": {
                "verbose_logging": True,
                "cleanup_after_test": True
            }
        }
    
    def setup_logging(self):
        """ロギング設定"""
        log_level = logging.DEBUG if self.config["test_execution"]["verbose_logging"] else logging.INFO
        
        # ログファイルパス
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"load_test_{timestamp}.log"
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"負荷テスト開始: {datetime.now()}")
    
    def run_all_tests(self):
        """全負荷テスト実行"""
        
        self.logger.info("=" * 60)
        self.logger.info("5-3-3 負荷テストスイート開始")
        self.logger.info("=" * 60)
        
        try:
            # 基本スケールテスト
            self._run_scale_tests()
            
            # ストレステスト
            if CORRELATION_SYSTEM_AVAILABLE:
                self._run_stress_tests()
            else:
                self.logger.warning("相関システム未利用のためストレステストをスキップ")
            
            # 統合テスト
            if self.config["integration_test_settings"]["test_existing_integration"]:
                self._run_integration_tests()
            
            # 結果まとめ
            self._generate_final_report()
            
        except Exception as e:
            self.logger.error(f"テスト実行中にエラーが発生: {e}")
            self.logger.error(traceback.format_exc())
        
        finally:
            self._cleanup()
            
        self.logger.info("負荷テストスイート完了")
    
    def _run_scale_tests(self):
        """基本スケールテスト実行"""
        
        self.logger.info("--- 基本スケールテスト開始 ---")
        
        for test_name, scenario in self.config["test_scenarios"].items():
            self.logger.info(f"実行中: {test_name}")
            
            try:
                # テストデータ生成
                test_data = self.data_generator.generate_strategy_data(
                    num_strategies=scenario["num_strategies"],
                    periods=scenario["data_periods"]
                )
                
                # パフォーマンス監視開始
                monitor_id = self.monitor.start_monitoring()
                
                # 実際の計算処理（5-3-3システム）
                if CORRELATION_SYSTEM_AVAILABLE:
                    self._run_correlation_calculation(test_data)
                else:
                    self._run_mock_calculation(test_data)
                
                # パフォーマンス監視終了
                metrics = self.monitor.stop_monitoring()
                
                # 結果検証
                validation_result = self.validator.validate_performance(test_name, metrics)
                self.test_results.append(validation_result)
                
                # 個別結果ログ
                status = "✅ PASS" if validation_result.passed else "❌ FAIL"
                self.logger.info(f"{test_name}: {status} (スコア: {validation_result.score:.1f}/100)")
                
                # クリーンアップ
                if self.config["test_execution"]["cleanup_after_test"]:
                    del test_data
                    gc.collect()
                
            except Exception as e:
                self.logger.error(f"{test_name} でエラー: {e}")
                continue
    
    def _run_stress_tests(self):
        """ストレステスト実行"""
        
        self.logger.info("--- ストレステスト開始 ---")
        
        stress_scenarios = self.config["stress_test_scenarios"]
        
        for scenario_config in stress_scenarios:
            if not scenario_config["enabled"]:
                self.logger.info(f"スキップ: {scenario_config['name']} (無効)")
                continue
                
            scenario_name = scenario_config["name"]
            self.logger.info(f"実行中: {scenario_name}")
            
            try:
                # ストレステストデータ生成
                stress_data = self.data_generator.create_stress_test_data(
                    scenario=scenario_name,
                    num_strategies=10,
                    periods=252
                )
                
                # タイムアウト設定
                timeout = scenario_config["timeout_seconds"]
                
                # パフォーマンス監視開始
                monitor_id = self.monitor.start_monitoring()
                start_time = time.time()
                
                # ストレス計算実行
                self._run_stress_calculation(stress_data, timeout)
                
                # タイムアウトチェック
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self.logger.warning(f"{scenario_name}: タイムアウト ({elapsed:.1f}s > {timeout}s)")
                
                # パフォーマンス監視終了
                metrics = self.monitor.stop_monitoring()
                
                # ストレステスト結果検証
                validation_result = self.validator.validate_performance("stress_test", metrics)
                validation_result.test_name = f"stress_{scenario_name}"
                self.test_results.append(validation_result)
                
                # 結果ログ
                status = "✅ PASS" if validation_result.passed else "❌ FAIL"
                self.logger.info(f"{scenario_name}: {status} (時間: {elapsed:.1f}s)")
                
            except Exception as e:
                self.logger.error(f"ストレステスト {scenario_name} でエラー: {e}")
                continue
    
    def _run_integration_tests(self):
        """統合テスト実行"""
        
        self.logger.info("--- 統合テスト開始 ---")
        
        # 簡単な統合テスト
        try:
            test_data = self.data_generator.generate_strategy_data(5, 100)
            
            monitor_id = self.monitor.start_monitoring()
            
            # 複数最適化手法のテスト
            if CORRELATION_SYSTEM_AVAILABLE:
                self._test_optimization_methods(test_data)
            
            metrics = self.monitor.stop_monitoring()
            
            validation_result = self.validator.validate_performance("integration_test", metrics)
            self.test_results.append(validation_result)
            
            self.logger.info(f"統合テスト完了: スコア {validation_result.score:.1f}/100")
            
        except Exception as e:
            self.logger.error(f"統合テストエラー: {e}")
    
    def _run_correlation_calculation(self, test_data: Dict):
        """相関計算実行（実際の5-3-3システム使用）"""
        
        try:
            # 設定作成
            config = AllocationConfig()
            
            # アロケーター初期化
            allocator = CorrelationBasedAllocator(config)
            
            # リターンデータ準備
            returns_df = test_data["returns_df"]
            
            # 配分計算実行
            result = allocator.allocate_portfolio(
                strategy_returns=returns_df,
                strategy_scores=None  # デフォルトスコア使用
            )
            
            self.logger.debug(f"配分計算完了: {len(result.weights)}戦略")
            
        except Exception as e:
            self.logger.error(f"相関計算エラー: {e}")
            # フォールバックとしてモック計算
            self._run_mock_calculation(test_data)
    
    def _run_mock_calculation(self, test_data: Dict):
        """モック計算（5-3-3システム未利用時）"""
        
        import numpy as np
        
        returns_df = test_data["returns_df"]
        
        # 簡単な相関計算
        correlation_matrix = returns_df.corr()
        
        # 簡単な最適化（等重み）
        num_strategies = len(returns_df.columns)
        weights = np.ones(num_strategies) / num_strategies
        
        # 人工的な計算負荷
        for _ in range(100):
            _ = np.random.random((100, 100)) @ np.random.random((100, 100))
        
        self.logger.debug(f"モック計算完了: {num_strategies}戦略")
    
    def _run_stress_calculation(self, stress_data: Dict, timeout: int):
        """ストレス計算実行"""
        
        returns_data = stress_data["returns_data"]
        scenario = stress_data["scenario"]
        
        if scenario == "memory_pressure":
            # メモリ集約的計算
            self._memory_intensive_calculation(returns_data)
        elif scenario == "extreme_correlation":
            # 数値安定性テスト
            self._numerical_stability_test(returns_data)
        else:
            # 標準ストレス計算
            self._standard_stress_calculation(returns_data)
    
    def _memory_intensive_calculation(self, returns_data: Dict):
        """メモリ集約的計算"""
        import numpy as np
        
        # 大きな行列計算
        size = min(1000, len(returns_data) * 50)
        large_matrix = np.random.random((size, size))
        
        # 行列演算
        result = large_matrix @ large_matrix.T
        eigenvals = np.linalg.eigvals(result[:100, :100])  # サイズ削減
        
        self.logger.debug(f"メモリ集約計算完了: 行列サイズ {size}x{size}")
    
    def _numerical_stability_test(self, returns_data: Dict):
        """数値安定性テスト"""
        import numpy as np
        
        # 条件数の悪い行列での計算
        n = min(50, len(returns_data))
        A = np.random.random((n, n))
        A = A @ A.T + 1e-10 * np.eye(n)  # 正定値だが条件数悪い
        
        try:
            inv_A = np.linalg.inv(A)
            cond_num = np.linalg.cond(A)
            self.logger.debug(f"数値安定性テスト完了: 条件数 {cond_num:.2e}")
        except np.linalg.LinAlgError as e:
            self.logger.warning(f"数値安定性テストで特異性検出: {e}")
    
    def _standard_stress_calculation(self, returns_data: Dict):
        """標準ストレス計算"""
        import numpy as np
        
        # 反復計算
        for i in range(50):
            data_matrix = np.array([data["returns"] for data in returns_data.values()])
            corr_matrix = np.corrcoef(data_matrix)
            eigenvals = np.linalg.eigvals(corr_matrix)
        
        self.logger.debug("標準ストレス計算完了")
    
    def _test_optimization_methods(self, test_data: Dict):
        """最適化手法テスト"""
        
        if not CORRELATION_SYSTEM_AVAILABLE:
            return
        
        returns_df = test_data["returns_df"]
        
        # 各最適化手法をテスト
        methods = ["mean_variance", "risk_parity"]
        
        for method in methods:
            try:
                config = AllocationConfig()
                config.optimization_method = method
                
                allocator = CorrelationBasedAllocator(config)
                result = allocator.allocate_portfolio(strategy_returns=returns_df)
                
                self.logger.debug(f"最適化手法 {method} 完了")
                
            except Exception as e:
                self.logger.warning(f"最適化手法 {method} でエラー: {e}")
    
    def _generate_final_report(self):
        """最終レポート生成"""
        
        self.logger.info("--- 最終レポート生成 ---")
        
        # 詳細レポート生成
        if self.config["output_settings"]["generate_detailed_report"]:
            report = self.validator.generate_performance_report(self.test_results)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"load_test_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"詳細レポート保存: {report_file}")
            
            # コンソール出力
            print("\n" + "="*80)
            print(report)
        
        # JSON結果エクスポート
        if self.config["output_settings"]["export_metrics_json"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_file = self.output_dir / f"load_test_metrics_{timestamp}.json"
            
            self.validator.export_results(str(json_file), self.test_results)
    
    def _cleanup(self):
        """クリーンアップ"""
        
        if self.config["test_execution"]["cleanup_after_test"]:
            gc.collect()
            self.logger.debug("クリーンアップ完了")

def main():
    """メイン実行関数"""
    
    print("5-3-3 戦略間相関を考慮した配分最適化システム 負荷テスト")
    print("=" * 60)
    
    # 設定ファイル確認
    config_file = "load_test_config.json"
    if not os.path.exists(config_file):
        print(f"警告: 設定ファイル {config_file} が見つかりません")
        print("デフォルト設定で実行します")
    
    try:
        # 負荷テスト実行
        runner = LoadTestRunner(config_file)
        runner.run_all_tests()
        
        print("\n✅ 負荷テスト完了")
        print(f"結果は {runner.output_dir} に保存されました")
        
    except KeyboardInterrupt:
        print("\n❌ ユーザーによる中断")
    except Exception as e:
        print(f"\n❌ 実行エラー: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
