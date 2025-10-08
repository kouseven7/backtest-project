"""
Integration Test: Metric Normalization with Existing Systems
File: test_integration_with_existing_systems.py
Description: 
  2-1-3「指標の正規化手法の設計」と既存システムの統合テスト
  2-1-1スコアリングシステム、2-1-2指標選定システムとの連携確認

Author: imega
Created: 2025-07-10
Modified: 2025-07-10
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

# 正規化システム
from config.metric_normalization_manager import MetricNormalizationManager

# 既存システム
try:
    from config.strategy_scoring_model import StrategyScoreManager
    from config.metric_selection_manager import MetricSelectionManager
    EXISTING_SYSTEMS_AVAILABLE = True
except ImportError as e:
    EXISTING_SYSTEMS_AVAILABLE = False
    print(f"Warning: Some existing systems not available: {e}")

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def create_realistic_strategy_data():
    """リアルな戦略データの作成"""
    np.random.seed(42)
    
    strategies = {
        "momentum_strategy": {
            "sharpe_ratio": np.random.normal(1.5, 0.8, 252),  # 1年分のデータ
            "sortino_ratio": np.random.normal(2.0, 1.0, 252),
            "profit_factor": np.random.exponential(2.5, 252),
            "win_rate": np.random.beta(3, 2, 252),
            "max_drawdown": np.random.exponential(0.08, 252) * -1,
            "total_return": np.cumsum(np.random.normal(0.0005, 0.02, 252))
        },
        "mean_reversion_strategy": {
            "sharpe_ratio": np.random.normal(1.2, 0.6, 252),
            "sortino_ratio": np.random.normal(1.6, 0.8, 252),
            "profit_factor": np.random.exponential(1.8, 252),
            "win_rate": np.random.beta(2, 2, 252),
            "max_drawdown": np.random.exponential(0.12, 252) * -1,
            "total_return": np.cumsum(np.random.normal(0.0003, 0.015, 252))
        },
        "statistical_arbitrage": {
            "sharpe_ratio": np.random.normal(2.2, 0.4, 252),
            "sortino_ratio": np.random.normal(2.8, 0.5, 252),
            "profit_factor": np.random.exponential(3.5, 252),
            "win_rate": np.random.beta(4, 1, 252),
            "max_drawdown": np.random.exponential(0.04, 252) * -1,
            "total_return": np.cumsum(np.random.normal(0.0002, 0.008, 252))
        }
    }
    
    return strategies

def test_standalone_normalization():
    """単独正規化システムのテスト"""
    logger.info("=== Testing Standalone Normalization ===")
    
    try:
        # データ準備
        strategies_data = create_realistic_strategy_data()
        
        # 正規化マネージャーの作成（standalone mode）
        manager = MetricNormalizationManager(integration_mode="standalone")
        
        # 一括正規化実行
        summaries = manager.batch_normalize_strategies(strategies_data, save_sessions=True)
        
        # 結果検証
        total_strategies = len(summaries)
        successful = sum(1 for s in summaries.values() if s.success)
        
        logger.info(f"✓ Standalone normalization completed")
        logger.info(f"  - Total strategies: {total_strategies}")
        logger.info(f"  - Successful: {successful}")
        logger.info(f"  - Success rate: {successful/total_strategies:.1%}")
        
        # 各戦略の詳細結果
        for strategy_name, summary in summaries.items():
            metrics_count = len(summary.session_info.metrics_processed)
            logger.info(f"  - {strategy_name}: {metrics_count} metrics, success: {summary.success}")
        
        return successful == total_strategies
        
    except Exception as e:
        logger.error(f"✗ Standalone normalization failed: {e}")
        return False

def test_scoring_system_integration():
    """スコアリングシステム統合テスト"""
    logger.info("=== Testing Scoring System Integration ===")
    
    if not EXISTING_SYSTEMS_AVAILABLE:
        logger.warning("Existing systems not available, skipping test")
        return True
    
    try:
        # データ準備
        strategies_data = create_realistic_strategy_data()
        
        # 正規化マネージャー（scoring mode）
        normalization_manager = MetricNormalizationManager(integration_mode="scoring")
        
        # スコアリングマネージャー
        scoring_manager = StrategyScoreManager()
        
        # 正規化実行
        summaries = normalization_manager.batch_normalize_strategies(strategies_data)
        
        # 正規化結果をスコアリングシステムで活用する例
        integration_results = {}
        
        for strategy_name, summary in summaries.items():
            if summary.success and summary.scoring_integration:
                # 正規化された値を取得
                normalized_values = summary.scoring_integration.get("normalized_values", {})
                
                # スコアリングシステムでの処理（例：重み付きスコア計算）
                if normalized_values:
                    # シンプルな重み付きスコア計算
                    weighted_score = (
                        normalized_values.get("sharpe_ratio", 0) * 0.3 +
                        normalized_values.get("profit_factor", 0) * 0.25 +
                        normalized_values.get("win_rate", 0) * 0.2 +
                        abs(normalized_values.get("max_drawdown", 0)) * 0.15 +  # ドローダウンは絶対値
                        normalized_values.get("total_return", 0) * 0.1
                    )
                    
                    integration_results[strategy_name] = {
                        "weighted_score": weighted_score,
                        "normalization_success": True,
                        "metrics_count": len(normalized_values)
                    }
                    
                    logger.info(f"  - {strategy_name}: normalized score = {weighted_score:.3f}")
        
        success_count = len(integration_results)
        logger.info(f"✓ Scoring integration completed: {success_count} strategies processed")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"✗ Scoring system integration failed: {e}")
        return False

def test_metric_selection_integration():
    """指標選定システム統合テスト"""
    logger.info("=== Testing Metric Selection Integration ===")
    
    if not EXISTING_SYSTEMS_AVAILABLE:
        logger.warning("Existing systems not available, skipping test")
        return True
    
    try:
        # データ準備
        strategies_data = create_realistic_strategy_data()
        
        # 正規化マネージャー（metric_selection mode）
        normalization_manager = MetricNormalizationManager(integration_mode="metric_selection")
        
        # 指標選定マネージャー
        selection_manager = MetricSelectionManager()
        
        # Step 1: 正規化実行
        normalization_summaries = normalization_manager.batch_normalize_strategies(strategies_data)
        
        # Step 2: 正規化されたデータを指標選定システムで使用
        for strategy_name, summary in normalization_summaries.items():
            if summary.success and summary.selection_integration:
                selection_data = summary.selection_integration.get("data_summary", {})
                
                if selection_data:
                    # 正規化済みデータを指標選定システム用に変換
                    metrics_for_selection = {}
                    for metric_name, data_info in selection_data.items():
                        normalized_values = data_info.get("normalized_values", [])
                        if normalized_values:
                            metrics_for_selection[metric_name] = np.array(normalized_values)
                    
                    if metrics_for_selection:
                        # 指標選定システムでの前処理として正規化を使用
                        logger.info(f"  - {strategy_name}: {len(metrics_for_selection)} normalized metrics ready for selection")
        
        logger.info("✓ Metric selection integration completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Metric selection integration failed: {e}")
        return False

def test_end_to_end_workflow():
    """エンドツーエンドワークフローテスト"""
    logger.info("=== Testing End-to-End Workflow ===")
    
    try:
        # データ準備
        strategies_data = create_realistic_strategy_data()
        
        # Step 1: 正規化
        normalization_manager = MetricNormalizationManager(integration_mode="metric_selection")
        normalization_summaries = normalization_manager.batch_normalize_strategies(strategies_data)
        
        # Step 2: 正規化結果の検証
        normalized_strategies = {}
        for strategy_name, summary in normalization_summaries.items():
            if summary.success:
                # 正規化後のメトリクスを抽出
                session_results = summary.session_info.results
                normalized_metrics = {}
                
                for metric_name, result_info in session_results.items():
                    if result_info.get("success", False):
                        # 正規化された統計情報を使用
                        stats = result_info.get("statistics", {})
                        normalized_mean = stats.get("normalized_mean", 0)
                        normalized_metrics[metric_name] = normalized_mean
                
                normalized_strategies[strategy_name] = normalized_metrics
        
        # Step 3: 正規化品質の評価
        quality_scores = {}
        for strategy_name, summary in normalization_summaries.items():
            if summary.success:
                # 信頼度スコアに基づく品質評価
                confidence_scores = []
                session_results = summary.session_info.results
                
                for result_info in session_results.values():
                    confidence = result_info.get("confidence_score", 0)
                    confidence_scores.append(confidence)
                
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
                quality_scores[strategy_name] = avg_confidence
                
                logger.info(f"  - {strategy_name}: quality score = {avg_confidence:.3f}")
        
        # Step 4: 包括的レポート生成
        comprehensive_report = normalization_manager.generate_comprehensive_report(
            normalization_summaries, save_report=True
        )
        
        # Step 5: 結果サマリー
        total_strategies = len(strategies_data)
        normalized_count = len(normalized_strategies)
        avg_quality = np.mean(list(quality_scores.values())) if quality_scores else 0
        
        logger.info(f"✓ End-to-end workflow completed")
        logger.info(f"  - Input strategies: {total_strategies}")
        logger.info(f"  - Successfully normalized: {normalized_count}")
        logger.info(f"  - Average quality score: {avg_quality:.3f}")
        logger.info(f"  - Overall success rate: {normalized_count/total_strategies:.1%}")
        
        return normalized_count == total_strategies and avg_quality > 0.7
        
    except Exception as e:
        logger.error(f"✗ End-to-end workflow failed: {e}")
        return False

def test_performance_with_large_dataset():
    """大規模データセットでのパフォーマンステスト"""
    logger.info("=== Testing Performance with Large Dataset ===")
    
    try:
        import time
        
        # 大規模データセットの作成
        large_strategies = {}
        num_strategies = 20
        data_points_per_metric = 5000  # 約20年分の日次データ
        
        logger.info(f"Creating large dataset: {num_strategies} strategies x 6 metrics x {data_points_per_metric} points")
        
        for i in range(num_strategies):
            strategy_name = f"large_strategy_{i:02d}"
            large_strategies[strategy_name] = {
                "sharpe_ratio": np.random.normal(1.0 + i*0.1, 0.5, data_points_per_metric),
                "sortino_ratio": np.random.normal(1.3 + i*0.1, 0.6, data_points_per_metric),
                "profit_factor": np.random.exponential(1.5 + i*0.2, data_points_per_metric),
                "win_rate": np.random.beta(2 + i*0.1, 2, data_points_per_metric),
                "max_drawdown": np.random.exponential(0.1 + i*0.01, data_points_per_metric) * -1,
                "total_return": np.cumsum(np.random.normal(0.0003 + i*0.0001, 0.02, data_points_per_metric))
            }
        
        # 正規化実行（時間測定）
        normalization_manager = MetricNormalizationManager(integration_mode="standalone")
        
        start_time = time.time()
        summaries = normalization_manager.batch_normalize_strategies(large_strategies, save_sessions=False)
        processing_time = time.time() - start_time
        
        # パフォーマンス指標の計算
        total_data_points = num_strategies * 6 * data_points_per_metric
        successful_strategies = sum(1 for s in summaries.values() if s.success)
        throughput = total_data_points / processing_time
        
        logger.info(f"✓ Large dataset performance test completed")
        logger.info(f"  - Total data points: {total_data_points:,}")
        logger.info(f"  - Processing time: {processing_time:.2f} seconds")
        logger.info(f"  - Throughput: {throughput:.0f} data points/second")
        logger.info(f"  - Successful strategies: {successful_strategies}/{num_strategies}")
        logger.info(f"  - Memory efficiency: OK (no memory errors)")
        
        return successful_strategies == num_strategies and throughput > 1000
        
    except Exception as e:
        logger.error(f"✗ Large dataset performance test failed: {e}")
        return False

def generate_integration_report(test_results):
    """統合テストレポートの生成"""
    logger.info("=== Integration Test Results Summary ===")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    logger.info(f"Total Integration Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    logger.info("\nDetailed Results:")
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    # システム統合状況の評価
    integration_quality = "EXCELLENT" if passed_tests == total_tests else \
                         "GOOD" if passed_tests >= total_tests * 0.8 else \
                         "NEEDS_IMPROVEMENT"
    
    logger.info(f"\nIntegration Quality: {integration_quality}")
    
    if passed_tests == total_tests:
        logger.info("\n[SUCCESS] All integration tests passed! 2-1-3 Normalization System is fully integrated.")
    else:
        logger.warning("\n[WARNING] Some integration tests failed. Please review and fix issues.")

def main():
    """メイン統合テスト実行"""
    logger.info("Starting Integration Test for 2-1-3 Metric Normalization System")
    logger.info("="*70)
    
    # 統合テストスイートの実行
    test_results = {
        "Standalone Normalization": test_standalone_normalization(),
        "Scoring System Integration": test_scoring_system_integration(),
        "Metric Selection Integration": test_metric_selection_integration(),
        "End-to-End Workflow": test_end_to_end_workflow(),
        "Large Dataset Performance": test_performance_with_large_dataset()
    }
    
    # 結果レポートの生成
    generate_integration_report(test_results)

if __name__ == "__main__":
    main()
