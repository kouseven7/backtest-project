"""
フェーズ2統合テストスクリプト（簡易版）
統合ウォークフォワードテスト・パフォーマンス集約・レポート生成の統合テスト
既存システムへの依存を最小化したバージョン
"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.logger_config import setup_logger

def run_phase2_simple_test():
    """フェーズ2簡易テスト実行（既存システムへの依存最小化）"""
    logger = setup_logger(__name__)
    logger.info("=== フェーズ2簡易統合テスト開始 ===")
    
    try:
        # 1. 統合ウォークフォワードテストのインポートと基本テスト
        logger.info("1. 統合ウォークフォワードテストシステムの確認...")
        
        try:
            from src.analysis.simple_walkforward import (
                SimpleWalkforwardTester, 
                create_test_configuration,
                TestConfiguration,
                ProcessingMode
            )
            
            # テスト設定の作成
            config = create_test_configuration(
                symbols=["AAPL"],
                strategies=["VWAPBreakoutStrategy"],
                start_date="2023-06-01",
                end_date="2023-06-30",
                processing_mode="sequential"
            )
            
            logger.info(f"ウォークフォワード設定作成成功: {config.to_dict()}")
            
            # テスターインスタンスの作成テスト
            tester = SimpleWalkforwardTester(config)
            logger.info("ウォークフォワードテスター初期化成功")
            
            walkforward_success = True
            
        except Exception as e:
            logger.error(f"ウォークフォワードテスト失敗: {e}")
            walkforward_success = False
        
        # 2. パフォーマンス集約システムのテスト
        logger.info("2. パフォーマンス集約システムの確認...")
        
        try:
            from src.analysis.performance_aggregator import (
                PerformanceAggregator,
                create_aggregation_config,
                AggregationConfig,
                PerformanceMetrics
            )
            
            # 集約設定の作成
            agg_config = create_aggregation_config(
                time_granularity="monthly"
            )
            
            logger.info(f"集約設定作成成功: {agg_config}")
            
            # 集約システムの初期化テスト
            aggregator = PerformanceAggregator(agg_config)
            logger.info("パフォーマンス集約システム初期化成功")
            
            # ダミーデータでのテスト
            dummy_results = [
                {
                    'combination': {'strategy': 'TestStrategy', 'symbol': 'AAPL', 'start_date': '2023-01-01', 'end_date': '2023-06-30'},
                    'summary_metrics': {'avg_return': 0.05, 'avg_sharpe_ratio': 1.2, 'avg_win_rate': 0.6},
                    'market_classification': {'market_state': 'bull_market'},
                    'time_period': {'start_date': '2023-01-01', 'end_date': '2023-06-30'},
                    'data_quality': {'completeness': 0.95}
                }
            ]
            
            aggregated = aggregator.aggregate_walkforward_results(dummy_results)
            logger.info(f"パフォーマンス集約テスト成功: {list(aggregated.keys())}")
            
            aggregation_success = True
            
        except Exception as e:
            logger.error(f"パフォーマンス集約テスト失敗: {e}")
            aggregated = {
                'summary': {'total_results': 1, 'strategies_analyzed': 1, 'symbols_analyzed': 1},
                'strategy_market_performance': {'TestStrategy': {'bull_market': {'total_return': {'mean': 0.05}}}},
                'performance_rankings': {'overall': {'TestStrategy': 0.8}}
            }
            aggregation_success = False
        
        # 3. レポート生成システムのテスト
        logger.info("3. レポート生成システムの確認...")
        
        try:
            from src.reports.strategy_comparison import (
                StrategyComparisonReporter,
                create_report_config,
                ReportConfig
            )
            
            # レポート設定の作成
            report_config = create_report_config(
                output_formats=["json"],  # 依存関係の少ないJSONのみ
                include_charts=False,     # チャート無効化
                include_interactive_dashboard=False,  # ダッシュボード無効化
                output_directory="output/test_reports"
            )
            
            logger.info(f"レポート設定作成成功: {report_config}")
            
            # レポーター初期化テスト
            reporter = StrategyComparisonReporter(report_config)
            logger.info("レポート生成システム初期化成功")
            
            # レポート生成テスト
            generated_files = reporter.generate_comprehensive_report(aggregated)
            logger.info(f"レポート生成テスト成功: {generated_files}")
            
            # ファイル存在確認
            for format_type, file_path in generated_files.items():
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    logger.info(f"{format_type}レポート生成確認: {file_path} ({file_size:,} bytes)")
                else:
                    logger.warning(f"{format_type}レポートファイルが見つかりません: {file_path}")
            
            report_success = True
            
        except Exception as e:
            logger.error(f"レポート生成テスト失敗: {e}")
            report_success = False
        
        # 4. 結果サマリー
        logger.info("=== テスト結果サマリー ===")
        logger.info(f"ウォークフォワードテスト: {'成功' if walkforward_success else '失敗'}")
        logger.info(f"パフォーマンス集約: {'成功' if aggregation_success else '失敗'}")
        logger.info(f"レポート生成: {'成功' if report_success else '失敗'}")
        
        overall_success = walkforward_success and aggregation_success and report_success
        
        if overall_success:
            logger.info("[SUCCESS] フェーズ2統合テスト: 全体的に成功")
        else:
            logger.warning("[WARNING] フェーズ2統合テスト: 一部失敗があります")
        
        logger.info("=== フェーズ2簡易統合テスト完了 ===")
        return overall_success
        
    except Exception as e:
        logger.error(f"統合テスト全体失敗: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_individual_modules():
    """個別モジュールの単体テスト"""
    logger = setup_logger(__name__)
    logger.info("=== 個別モジュール単体テスト開始 ===")
    
    # テスト結果
    test_results = {}
    
    # 1. 統合ウォークフォワードモジュール
    try:
        from src.analysis.simple_walkforward import ProcessingMode, TestConfiguration
        test_results['simple_walkforward'] = '成功'
        logger.info("[OK] simple_walkforward モジュールインポート成功")
    except Exception as e:
        test_results['simple_walkforward'] = f'失敗: {e}'
        logger.error(f"[ERROR] simple_walkforward モジュールインポート失敗: {e}")
    
    # 2. パフォーマンス集約モジュール
    try:
        from src.analysis.performance_aggregator import AggregationConfig, PerformanceMetrics
        test_results['performance_aggregator'] = '成功'
        logger.info("[OK] performance_aggregator モジュールインポート成功")
    except Exception as e:
        test_results['performance_aggregator'] = f'失敗: {e}'
        logger.error(f"[ERROR] performance_aggregator モジュールインポート失敗: {e}")
    
    # 3. レポート生成モジュール
    try:
        from src.reports.strategy_comparison import ReportConfig, ReportSection
        test_results['strategy_comparison'] = '成功'
        logger.info("[OK] strategy_comparison モジュールインポート成功")
    except Exception as e:
        test_results['strategy_comparison'] = f'失敗: {e}'
        logger.error(f"[ERROR] strategy_comparison モジュールインポート失敗: {e}")
    
    # 結果サマリー
    logger.info("=== 個別テスト結果 ===")
    for module, result in test_results.items():
        logger.info(f"{module}: {result}")
    
    success_count = sum(1 for r in test_results.values() if r == '成功')
    total_count = len(test_results)
    
    logger.info(f"成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    return success_count == total_count

def main():
    """メイン実行"""
    
    # 個別モジュールテスト
    individual_success = test_individual_modules()
    
    if individual_success:
        # 統合テスト実行
        integration_success = run_phase2_simple_test()
        return 0 if integration_success else 1
    else:
        print("個別モジュールテストでエラーがあるため、統合テストをスキップします。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
