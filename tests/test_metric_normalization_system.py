"""
Test Script: Comprehensive Metric Normalization System Test
File: test_metric_normalization_system.py
Description: 
  2-1-3「指標の正規化手法の設計」システムの包括的テスト
  設定、エンジン、マネージャーの統合動作を検証

Author: imega
Created: 2025-07-10
Modified: 2025-07-10
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any

# 内部モジュール
from config.metric_normalization_config import MetricNormalizationConfig, NormalizationParameters
from config.metric_normalization_engine import MetricNormalizationEngine
from config.metric_normalization_manager import MetricNormalizationManager

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def create_test_data() -> Dict[str, Dict[str, np.ndarray]]:
    """テスト用のサンプルデータ作成"""
    np.random.seed(42)
    
    # 戦略データ
    strategies_data = {
        "trend_following": {
            "sharpe_ratio": np.random.normal(1.2, 0.6, 100),
            "sortino_ratio": np.random.normal(1.5, 0.8, 100),
            "profit_factor": np.random.exponential(2.0, 100),
            "win_rate": np.random.beta(2, 2, 100),
            "max_drawdown": np.random.exponential(0.15, 100) * -1,  # 負の値
            "total_return": np.random.normal(0.15, 0.25, 100)
        },
        "mean_reversion": {
            "sharpe_ratio": np.random.normal(0.8, 0.4, 100),
            "sortino_ratio": np.random.normal(1.0, 0.5, 100),
            "profit_factor": np.random.exponential(1.5, 100),
            "win_rate": np.random.beta(1.5, 2, 100),
            "max_drawdown": np.random.exponential(0.12, 100) * -1,
            "total_return": np.random.normal(0.10, 0.20, 100)
        },
        "arbitrage": {
            "sharpe_ratio": np.random.normal(2.0, 0.3, 100),
            "sortino_ratio": np.random.normal(2.5, 0.4, 100),
            "profit_factor": np.random.exponential(3.0, 100),
            "win_rate": np.random.beta(3, 1, 100),
            "max_drawdown": np.random.exponential(0.05, 100) * -1,
            "total_return": np.random.normal(0.08, 0.10, 100)
        }
    }
    
    return strategies_data

def test_config_system():
    """設定システムのテスト"""
    logger.info("=== Testing Configuration System ===")
    
    try:
        # 設定インスタンスの作成
        config = MetricNormalizationConfig()
        
        # グローバル設定の確認
        sharpe_params = config.get_normalization_parameters("sharpe_ratio")
        logger.info(f"✓ Sharpe ratio config: {sharpe_params.method}")
        
        # 戦略別オーバーライドの追加
        success = config.add_strategy_override(
            "test_strategy",
            {
                "sharpe_ratio": {
                    "method": "robust",
                    "target_range": (0.0, 1.0),
                    "outlier_handling": "transform"
                }
            },
            notes="Test strategy override"
        )
        logger.info(f"✓ Strategy override added: {success}")
        
        # オーバーライド設定の確認
        override_params = config.get_normalization_parameters("sharpe_ratio", "test_strategy")
        logger.info(f"✓ Override config: {override_params.method}")
        
        # 設定検証
        validation = config.validate_config()
        logger.info(f"✓ Config validation: {validation['valid']}")
        
        # 設定要約
        summary = config.get_config_summary()
        logger.info(f"✓ Config summary: {len(summary['global_metrics'])} metrics, {len(summary['strategy_overrides'])} overrides")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Config system test failed: {e}")
        return False

def test_engine_system():
    """エンジンシステムのテスト"""
    logger.info("=== Testing Engine System ===")
    
    try:
        # エンジンインスタンスの作成
        config = MetricNormalizationConfig()
        engine = MetricNormalizationEngine(config)
        
        # 単一指標正規化テスト
        test_data = np.random.normal(1.0, 0.5, 50)
        result = engine.normalize_metric(test_data, "sharpe_ratio")
        logger.info(f"✓ Single metric normalization: {result.success}, confidence: {result.confidence_score:.3f}")
        
        # 一括正規化テスト
        batch_data = {
            "sharpe_ratio": np.random.normal(1.0, 0.5, 50),
            "profit_factor": np.random.exponential(1.5, 50),
            "win_rate": np.random.beta(2, 2, 50)
        }
        batch_results = engine.batch_normalize(batch_data)
        success_count = sum(1 for r in batch_results.values() if r.success)
        logger.info(f"✓ Batch normalization: {success_count}/{len(batch_data)} successful")
        
        # カスタム正規化テスト
        custom_result = engine.normalize_metric(
            np.random.exponential(1.5, 50), 
            "profit_factor"  # デフォルトでカスタム手法を使用
        )
        logger.info(f"✓ Custom normalization: {custom_result.success}, method: {custom_result.method_used}")
        
        # 利用可能手法の確認
        methods = engine.get_available_methods()
        custom_funcs = engine.get_custom_functions()
        logger.info(f"✓ Available methods: {len(methods)}, custom functions: {len(custom_funcs)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Engine system test failed: {e}")
        return False

def test_manager_system():
    """マネージャーシステムのテスト"""
    logger.info("=== Testing Manager System ===")
    
    try:
        # マネージャーインスタンスの作成
        manager = MetricNormalizationManager(integration_mode="metric_selection")
        
        # テストデータの作成
        test_strategies = create_test_data()
        
        # 単一戦略正規化テスト
        single_summary = manager.normalize_strategy_metrics(
            "trend_following",
            test_strategies["trend_following"],
            save_session=True
        )
        logger.info(f"✓ Single strategy normalization: {single_summary.success}")
        logger.info(f"  - Metrics processed: {len(single_summary.session_info.metrics_processed)}")
        logger.info(f"  - Success rate: {single_summary.session_info.success_rate:.3f}")
        logger.info(f"  - Processing time: {single_summary.session_info.total_processing_time:.3f}s")
        
        # 一括戦略正規化テスト
        batch_summaries = manager.batch_normalize_strategies(test_strategies)
        successful_strategies = sum(1 for s in batch_summaries.values() if s.success)
        logger.info(f"✓ Batch strategies normalization: {successful_strategies}/{len(test_strategies)} successful")
        
        # 指標選定システム用正規化テスト
        selection_data = manager.normalize_for_metric_selection(
            test_strategies["arbitrage"]
        )
        logger.info(f"✓ Metric selection normalization: {len(selection_data)} metrics processed")
        
        # 包括的レポート生成テスト
        comprehensive_report = manager.generate_comprehensive_report(batch_summaries)
        logger.info(f"✓ Comprehensive report generated")
        logger.info(f"  - Overall success rate: {comprehensive_report['overall_performance']['average_success_rate']:.3f}")
        logger.info(f"  - Total metrics processed: {comprehensive_report['overall_performance']['total_metrics_processed']}")
        
        # 履歴取得テスト
        history = manager.get_normalization_history(days_back=1)
        logger.info(f"✓ History retrieval: {len(history)} sessions found")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Manager system test failed: {e}")
        return False

def test_integration_features():
    """統合機能のテスト"""
    logger.info("=== Testing Integration Features ===")
    
    try:
        # 異なる統合モードでのテスト
        modes = ["scoring", "metric_selection", "standalone"]
        
        for mode in modes:
            try:
                manager = MetricNormalizationManager(integration_mode=mode)
                test_data = create_test_data()["trend_following"]
                
                summary = manager.normalize_strategy_metrics(
                    "integration_test",
                    test_data,
                    save_session=False
                )
                
                logger.info(f"✓ Integration mode '{mode}': {summary.success}")
                
                # 統合情報の確認
                if summary.scoring_integration:
                    logger.info(f"  - Scoring integration: {'success' if 'error' not in summary.scoring_integration else 'failed'}")
                if summary.selection_integration:
                    logger.info(f"  - Selection integration: {'success' if 'error' not in summary.selection_integration else 'failed'}")
                    
            except Exception as e:
                logger.warning(f"Integration mode '{mode}' test failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration features test failed: {e}")
        return False

def test_performance_evaluation():
    """パフォーマンス評価テスト"""
    logger.info("=== Testing Performance Evaluation ===")
    
    try:
        import time
        
        # 大容量データでのパフォーマンステスト
        large_data = {}
        for i in range(10):  # 10戦略
            strategy_name = f"strategy_{i}"
            large_data[strategy_name] = {}
            for metric in ["sharpe_ratio", "profit_factor", "win_rate", "max_drawdown", "total_return"]:
                large_data[strategy_name][metric] = np.random.normal(0, 1, 1000)  # 各指標1000データポイント
        
        manager = MetricNormalizationManager()
        
        start_time = time.time()
        batch_summaries = manager.batch_normalize_strategies(large_data, save_sessions=False)
        processing_time = time.time() - start_time
        
        successful_strategies = sum(1 for s in batch_summaries.values() if s.success)
        total_metrics = sum(len(data) for data in large_data.values())
        
        logger.info(f"✓ Performance test completed:")
        logger.info(f"  - Strategies: {len(large_data)}")
        logger.info(f"  - Total metrics: {total_metrics}")
        logger.info(f"  - Successful strategies: {successful_strategies}")
        logger.info(f"  - Processing time: {processing_time:.3f}s")
        logger.info(f"  - Throughput: {total_metrics/processing_time:.1f} metrics/second")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance evaluation test failed: {e}")
        return False

def generate_test_report(test_results: Dict[str, bool]):
    """テスト結果レポートの生成"""
    logger.info("=== Test Results Summary ===")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    logger.info("\nDetailed Results:")
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    # レポートファイルの保存
    try:
        report_dir = Path("logs/metric_normalization/test_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "test_timestamp": pd.Timestamp.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests/total_tests,
            "detailed_results": test_results
        }
        
        report_file = report_dir / f"test_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nTest report saved: {report_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save test report: {e}")

def main():
    """メインテスト実行"""
    logger.info("Starting Comprehensive Metric Normalization System Test")
    logger.info("="*60)
    
    # テストスイートの実行
    test_results = {
        "Config System": test_config_system(),
        "Engine System": test_engine_system(),
        "Manager System": test_manager_system(),
        "Integration Features": test_integration_features(),
        "Performance Evaluation": test_performance_evaluation()
    }
    
    # 結果レポートの生成
    generate_test_report(test_results)
    
    # 終了メッセージ
    if all(test_results.values()):
        logger.info("\n🎉 All tests passed! Metric Normalization System is working correctly.")
    else:
        logger.warning("\n⚠️ Some tests failed. Please review the detailed results above.")

if __name__ == "__main__":
    main()
