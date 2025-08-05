"""
フェーズ2統合テストスクリプト
統合ウォークフォワードテスト・パフォーマンス集約・レポート生成の統合テスト
"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.logger_config import setup_logger
from src.analysis.comprehensive_walkforward import ComprehensiveWalkforwardTester, create_test_configuration
from src.analysis.performance_aggregator import PerformanceAggregator, create_aggregation_config
from src.reports.strategy_comparison import StrategyComparisonReporter, create_report_config

def run_phase2_integration_test():
    """フェーズ2統合テスト実行"""
    logger = setup_logger(__name__)
    logger.info("=== フェーズ2統合テスト開始 ===")
    
    try:
        # テスト設定
        test_symbols = ["AAPL", "MSFT"]
        test_strategies = ["VWAPBreakoutStrategy", "BreakoutStrategy"]
        
        # 1. 統合ウォークフォワードテスト
        logger.info("1. 統合ウォークフォワードテスト実行中...")
        
        walkforward_config = create_test_configuration(
            symbols=test_symbols,
            strategies=test_strategies,
            start_date="2023-01-01",
            end_date="2023-06-30",
            processing_mode="sequential",  # テスト用に逐次実行
            window_size_days=60,
            step_size_days=10,
            output_directory="output/test_comprehensive_walkforward"
        )
        
        walkforward_tester = ComprehensiveWalkforwardTester(walkforward_config)
        walkforward_results = walkforward_tester.execute_comprehensive_test()
        
        logger.info(f"ウォークフォワードテスト完了: {walkforward_tester.progress.total_tests}テスト実行")
        
        # 2. パフォーマンス集約
        logger.info("2. パフォーマンス集約実行中...")
        
        aggregation_config = create_aggregation_config(
            time_granularity="monthly",
            correlation_threshold=0.6
        )
        
        aggregator = PerformanceAggregator(aggregation_config)
        
        # ウォークフォワード結果から個別結果を抽出
        individual_results = []
        if 'strategy_breakdown' in walkforward_results:
            for strategy, strategy_data in walkforward_results['strategy_breakdown'].items():
                # ダミーの個別結果を生成（実際の実装では適切に抽出）
                for i in range(strategy_data.get('test_count', 1)):
                    result = {
                        'combination': {
                            'strategy': strategy,
                            'symbol': 'AAPL',  # テスト用
                            'start_date': '2023-01-01',
                            'end_date': '2023-06-30'
                        },
                        'summary_metrics': strategy_data.get('avg_metrics', {}),
                        'market_classification': {'market_state': 'bull_market'},
                        'time_period': {'start_date': '2023-01-01', 'end_date': '2023-06-30'},
                        'data_quality': {'completeness': 0.95}
                    }
                    individual_results.append(result)
        
        if individual_results:
            aggregated_results = aggregator.aggregate_walkforward_results(individual_results)
            logger.info(f"パフォーマンス集約完了: {len(individual_results)}結果を集約")
        else:
            # フォールバック：空の集約結果を生成
            aggregated_results = {
                'summary': {'total_results': 0, 'strategies_analyzed': 0, 'symbols_analyzed': 0},
                'strategy_market_performance': {},
                'performance_rankings': {'overall': {}}
            }
            logger.warning("個別結果が取得できませんでした。空の集約結果を使用します。")
        
        # 3. レポート生成
        logger.info("3. レポート生成実行中...")
        
        report_config = create_report_config(
            output_formats=["excel", "html", "json"],
            include_charts=True,
            include_interactive_dashboard=True,
            output_directory="output/test_reports"
        )
        
        reporter = StrategyComparisonReporter(report_config)
        generated_reports = reporter.generate_comprehensive_report(aggregated_results)
        
        logger.info(f"レポート生成完了: {len(generated_reports)}形式で生成")
        
        # 4. 結果サマリー
        logger.info("=== テスト結果サマリー ===")
        logger.info(f"ウォークフォワードテスト: {walkforward_tester.progress.completion_rate*100:.1f}% 完了")
        logger.info(f"集約結果: {aggregated_results['summary']['total_results']}件処理")
        logger.info(f"生成レポート: {list(generated_reports.keys())}")
        
        # ファイル確認
        logger.info("=== 生成ファイル確認 ===")
        for report_type, file_path in generated_reports.items():
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                logger.info(f"{report_type}: {file_path} ({file_size:,} bytes)")
            else:
                logger.warning(f"{report_type}: ファイルが見つかりません - {file_path}")
        
        logger.info("=== フェーズ2統合テスト完了 ===")
        return True
        
    except Exception as e:
        logger.error(f"統合テスト失敗: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """メイン実行"""
    success = run_phase2_integration_test()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
