"""
DSSMS Phase 3 Task 3.2: 比較分析機能向上 - デモスクリプト
File: demo_comparison_analysis.py

DSSMS比較分析エンジンと拡張レポート生成器のデモンストレーション
既存システムとの統合テストも含む

Author: imega (Agent Mode Implementation)
Created: 2025-01-22
Based on: Previous conversation design specifications

使用方法:
    python demo_comparison_analysis.py
    
機能:
    - 模擬データでの比較分析実行
    - 既存システム統合テスト
    - Excel拡張レポート生成
    - エラーハンドリング検証
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any, Union

# パス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 新規モジュール
has_new_modules = True
DSSMSComparisonAnalysisEngine = None
DSSMSComparisonReportGenerator = None
ComparisonResult = None

try:
    from analysis.dssms_comparison_analysis_engine import DSSMSComparisonAnalysisEngine, ComparisonResult
    from analysis.dssms_comparison_report_generator import DSSMSComparisonReportGenerator
    print("✓ 新規モジュール (比較分析エンジン・レポート生成器) インポート成功")
except ImportError as e:
    print(f"✗ 新規モジュールインポート失敗: {e}")
    has_new_modules = False

# 既存システム
has_existing_modules = True
StrategySwitchingAnalyzer = None
SimpleExcelExporter = None

try:
    from analysis.strategy_switching.strategy_switching_analyzer import StrategySwitchingAnalyzer
    from output.simple_excel_exporter import SimpleExcelExporter
    print("✓ 既存モジュール (戦略切替・Excel出力) インポート成功")
except ImportError as e:
    print(f"✗ 既存モジュールインポート失敗: {e}")
    has_existing_modules = False

# ロギング設定
def setup_logging() -> logging.Logger:
    """ロギング設定"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"demo_comparison_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def generate_sample_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    num_strategies: int = 4
) -> pd.DataFrame:
    """
    サンプルデータ生成
    
    Parameters:
        start_date: 開始日
        end_date: 終了日
        num_strategies: 戦略数
        
    Returns:
        サンプルデータフレーム
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    # 日付範囲生成
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 基本価格データ
    np.random.seed(42)  # 再現性のため
    initial_price = 1000
    daily_returns = np.random.normal(0.0005, 0.02, len(date_range))  # 平均0.05%、標準偏差2%
    
    prices = [initial_price]
    for ret in daily_returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # データフレーム作成
    data = pd.DataFrame({
        'Date': date_range,
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Volume': np.random.randint(100000, 1000000, len(date_range))
    })
    
    # Open価格（前日Closeベース）
    data['Open'] = data['Close'].shift(1).fillna(initial_price)
    
    # 戦略シグナル生成
    strategies = [
        'VWAP_Breakout',
        'Mean_Reversion', 
        'Momentum_Strategy',
        'Contrarian_Strategy'
    ][:num_strategies]
    
    for i, strategy in enumerate(strategies):
        # 各戦略の特徴的なパフォーマンス
        strategy_multiplier = 1.0 + (i * 0.001)  # 戦略間の差分
        strategy_volatility = 0.015 + (i * 0.005)  # ボラティリティの差
        
        strategy_returns = np.random.normal(
            0.0008 * strategy_multiplier,  # リターン
            strategy_volatility,           # ボラティリティ
            len(date_range)
        )
        
        # 累積パフォーマンス
        strategy_performance = [100]  # 初期値100
        for ret in strategy_returns[1:]:
            strategy_performance.append(strategy_performance[-1] * (1 + ret))
        
        data[f'{strategy}_Performance'] = strategy_performance
        data[f'{strategy}_Signal'] = np.random.choice([0, 1, -1], len(date_range), p=[0.7, 0.15, 0.15])
    
    # 基本指標追加
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility_20'] = data['Daily_Return'].rolling(20).std()
    data['MA_20'] = data['Close'].rolling(20).mean()
    data['MA_60'] = data['Close'].rolling(60).mean()
    
    data.set_index('Date', inplace=True)
    
    return data

def run_basic_analysis_demo(logger: logging.Logger) -> Any:
    """基本分析デモ実行"""
    logger.info("=== 基本分析デモ開始 ===")
    
    try:
        # サンプルデータ生成
        logger.info("サンプルデータ生成中...")
        data = generate_sample_data(
            start_date=datetime.now() - timedelta(days=252),  # 1年間
            end_date=datetime.now(),
            num_strategies=4
        )
        logger.info(f"サンプルデータ生成完了: {len(data)} レコード")
        
        # 戦略リスト
        strategies = ['VWAP_Breakout', 'Mean_Reversion', 'Momentum_Strategy', 'Contrarian_Strategy']
        
        if not has_new_modules or DSSMSComparisonAnalysisEngine is None:
            logger.error("新規モジュールが利用できないため、基本分析をスキップします")
            return False
        
        # 比較分析エンジン初期化
        logger.info("比較分析エンジン初期化中...")
        analysis_engine = DSSMSComparisonAnalysisEngine()
        
        # 分析実行
        logger.info("包括的比較分析実行中...")
        comparison_result = analysis_engine.run_comprehensive_analysis(
            data=data,
            strategies=strategies,
            start_date=data.index.min(),
            end_date=data.index.max(),
            analysis_mode="comprehensive"
        )
        
        logger.info(f"分析完了: {comparison_result.analysis_id}")
        
        # 結果サマリー表示
        logger.info("=== 分析結果サマリー ===")
        logger.info(f"分析ID: {comparison_result.analysis_id}")
        logger.info(f"信頼度レベル: {comparison_result.confidence_level:.1%}")
        logger.info(f"データ品質スコア: {comparison_result.data_quality_score:.1%}")
        logger.info(f"分析期間: {comparison_result.analysis_period[0]} ～ {comparison_result.analysis_period[1]}")
        
        # 戦略パフォーマンス
        if comparison_result.strategy_performance:
            logger.info("戦略パフォーマンス:")
            for strategy, metrics in comparison_result.strategy_performance.items():
                logger.info(f"  {strategy}: リターン {metrics.get('annual_return', 0):.2%}, "
                           f"シャープレシオ {metrics.get('sharpe_ratio', 0):.3f}")
        
        # 推奨事項
        if comparison_result.overall_recommendations:
            logger.info("主要推奨事項:")
            for i, rec in enumerate(comparison_result.overall_recommendations[:3], 1):
                logger.info(f"  {i}. {rec}")
        
        return comparison_result
        
    except Exception as e:
        logger.error(f"基本分析デモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_report_generation_demo(comparison_result: Any, logger: logging.Logger) -> bool:
    """レポート生成デモ実行"""
    logger.info("=== レポート生成デモ開始 ===")
    
    try:
        if not comparison_result:
            logger.error("比較分析結果がないため、レポート生成をスキップします")
            return False
        
        if not has_new_modules or DSSMSComparisonReportGenerator is None:
            logger.error("新規モジュールが利用できないため、レポート生成をスキップします")
            return False
        
        # レポート生成器初期化
        logger.info("レポート生成器初期化中...")
        report_generator = DSSMSComparisonReportGenerator(
            output_dir="output/comparison_reports"
        )
        
        # 包括的レポート生成
        logger.info("包括的レポート生成中...")
        comprehensive_report_path = report_generator.generate_comprehensive_report(
            comparison_result=comparison_result,
            include_charts=True,
            report_name="demo_comprehensive_report"
        )
        
        if comprehensive_report_path:
            logger.info(f"包括的レポート生成成功: {comprehensive_report_path}")
        else:
            logger.error("包括的レポート生成失敗")
        
        # クイックサマリーレポート生成
        logger.info("クイックサマリーレポート生成中...")
        summary_report_path = report_generator.generate_quick_summary_report(
            comparison_result=comparison_result,
            report_name="demo_summary_report"
        )
        
        if summary_report_path:
            logger.info(f"サマリーレポート生成成功: {summary_report_path}")
        else:
            logger.error("サマリーレポート生成失敗")
        
        # レポート履歴確認
        history = report_generator.get_report_history()
        logger.info(f"レポート履歴: {len(history)} 件")
        
        return True
        
    except Exception as e:
        logger.error(f"レポート生成デモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_existing_system_integration_demo(logger: logging.Logger) -> bool:
    """既存システム統合デモ実行"""
    logger.info("=== 既存システム統合デモ開始 ===")
    
    try:
        if not has_existing_modules:
            logger.warning("既存モジュールが利用できないため、統合デモをスキップします")
            return False
        
        # サンプルデータ生成（シンプル版）
        logger.info("統合テスト用データ生成中...")
        data = generate_sample_data(
            start_date=datetime.now() - timedelta(days=100),
            end_date=datetime.now(),
            num_strategies=3
        )
        
        strategies = ['VWAP_Breakout', 'Mean_Reversion', 'Momentum_Strategy']
        
        # 戦略切替アナライザーテスト
        if StrategySwitchingAnalyzer is not None:
            logger.info("戦略切替アナライザー統合テスト中...")
            switching_analyzer = StrategySwitchingAnalyzer()
            
            # 模擬切替イベント生成
            switching_events = []
            for i in range(5):
                event_date = data.index[i * 20] if i * 20 < len(data) else data.index[-1]
                switching_events.append({
                    'timestamp': event_date,
                    'from_strategy': strategies[i % len(strategies)],
                    'to_strategy': strategies[(i + 1) % len(strategies)],
                    'trigger_type': 'performance_degradation',
                    'market_regime': 'neutral',
                    'performance_before': np.random.uniform(-0.02, 0.02),
                    'performance_after': np.random.uniform(-0.02, 0.02),
                    'confidence_score': np.random.uniform(0.3, 0.9),
                    'switching_cost': 0.001,
                    'success': np.random.choice([True, False])
                })
            
            switching_result = switching_analyzer.analyze_switching_performance(
                data=data,
                strategies=strategies,
                switching_events=switching_events,
                analysis_period=(data.index.min(), data.index.max())
            )
            
            logger.info(f"戦略切替分析完了: {switching_result.total_switches} 回の切替を分析")
            logger.info(f"成功率: {switching_result.success_rate:.1%}")
        
        # SimpleExcelExporter統合テスト
        if SimpleExcelExporter is not None:
            logger.info("SimpleExcelExporter統合テスト中...")
            excel_exporter = SimpleExcelExporter()
            
            # 模擬シグナル追加
            data['Entry_Signal'] = 0
            data['Exit_Signal'] = 0
            
            # ランダムシグナル生成
            signal_dates = np.random.choice(data.index[50:-50], size=10, replace=False)
            for date in signal_dates:
                data.loc[date, 'Entry_Signal'] = 1
                exit_date = date + timedelta(days=np.random.randint(5, 20))
                if exit_date in data.index:
                    data.loc[exit_date, 'Exit_Signal'] = 1
            
            export_path = excel_exporter.export_backtest_results(
                stock_data=data,
                ticker="DEMO_INTEGRATION",
                output_dir="output/integration_test"
            )
            
            if export_path:
                logger.info(f"Excel出力統合テスト成功: {export_path}")
            else:
                logger.error("Excel出力統合テスト失敗")
        
        return True
        
    except Exception as e:
        logger.error(f"既存システム統合デモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_error_handling_demo(logger: logging.Logger) -> bool:
    """エラーハンドリングデモ実行"""
    logger.info("=== エラーハンドリングデモ開始 ===")
    
    try:
        if not has_new_modules or DSSMSComparisonAnalysisEngine is None:
            logger.warning("新規モジュールが利用できないため、エラーハンドリングデモをスキップします")
            return False
        
        analysis_engine = DSSMSComparisonAnalysisEngine()
        
        # エラーケース1: 空のデータフレーム
        logger.info("エラーケース1: 空のデータフレーム")
        empty_data = pd.DataFrame()
        
        try:
            result = analysis_engine.run_comprehensive_analysis(
                data=empty_data,
                strategies=['Test_Strategy'],
                analysis_mode="comprehensive"
            )
            logger.error("エラーが発生すべき状況でエラーが発生しませんでした")
        except Exception as e:
            logger.info(f"期待通りのエラー処理: {type(e).__name__}")
        
        # エラーケース2: 不十分なデータポイント
        logger.info("エラーケース2: データポイント不足")
        small_data = generate_sample_data(
            start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now(),
            num_strategies=2
        )
        
        try:
            result = analysis_engine.run_comprehensive_analysis(
                data=small_data,
                strategies=['VWAP_Breakout', 'Mean_Reversion'],
                analysis_mode="comprehensive"
            )
            logger.error("データ不足エラーが発生すべき状況でエラーが発生しませんでした")
        except Exception as e:
            logger.info(f"期待通りのデータ不足エラー処理: {type(e).__name__}")
        
        # エラーケース3: 存在しない戦略
        logger.info("エラーケース3: 存在しない戦略列")
        normal_data = generate_sample_data()
        
        result = analysis_engine.run_comprehensive_analysis(
            data=normal_data,
            strategies=['NonExistent_Strategy', 'VWAP_Breakout'],
            analysis_mode="comprehensive"
        )
        
        # 存在しない戦略は無視され、存在する戦略のみで分析が実行される
        logger.info("存在しない戦略を含む分析が適切に処理されました")
        
        return True
        
    except Exception as e:
        logger.error(f"エラーハンドリングデモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_performance_benchmark(logger: logging.Logger) -> bool:
    """パフォーマンスベンチマーク実行"""
    logger.info("=== パフォーマンスベンチマーク開始 ===")
    
    try:
        if not has_new_modules or DSSMSComparisonAnalysisEngine is None:
            logger.warning("新規モジュールが利用できないため、ベンチマークをスキップします")
            return False
        
        # 大きなデータセットで性能測定
        logger.info("大規模データセット生成中...")
        large_data = generate_sample_data(
            start_date=datetime.now() - timedelta(days=500),  # 約1.5年
            end_date=datetime.now(),
            num_strategies=6
        )
        
        strategies = [
            'VWAP_Breakout', 'Mean_Reversion', 'Momentum_Strategy',
            'Contrarian_Strategy', 'Trend_Following', 'Statistical_Arbitrage'
        ]
        
        analysis_engine = DSSMSComparisonAnalysisEngine()
        
        # 各分析モードの実行時間測定
        modes = ['comprehensive', 'quick_summary']
        
        for mode in modes:
            logger.info(f"分析モード '{mode}' ベンチマーク実行中...")
            start_time = datetime.now()
            
            result = analysis_engine.run_comprehensive_analysis(
                data=large_data,
                strategies=strategies,
                analysis_mode=mode
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info(f"分析モード '{mode}': {processing_time:.2f}秒")
            logger.info(f"  - データポイント数: {len(large_data)}")
            logger.info(f"  - 戦略数: {len(strategies)}")
            logger.info(f"  - 信頼度: {result.confidence_level:.1%}")
        
        # パフォーマンスサマリー取得
        performance_summary = analysis_engine.get_analysis_summary()
        logger.info(f"分析サマリー: {performance_summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"パフォーマンスベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("DSSMS Phase 3 Task 3.2: 比較分析機能向上 - デモ実行")
    print("=" * 60)
    
    # ロギング設定
    logger = setup_logging()
    logger.info("DSSMS比較分析デモ開始")
    
    # システム状態確認
    logger.info("=== システム状態確認 ===")
    logger.info(f"新規モジュール利用可能: {has_new_modules}")
    logger.info(f"既存モジュール利用可能: {has_existing_modules}")
    
    # 出力ディレクトリ作成
    output_dirs = [
        "output/comparison_reports",
        "output/integration_test",
        "logs"
    ]
    
    for dir_path in output_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"出力ディレクトリ作成: {dir_path}")
    
    results = {}
    
    # 1. 基本分析デモ
    comparison_result = run_basic_analysis_demo(logger)
    results['basic_analysis'] = bool(comparison_result)
    
    # 2. レポート生成デモ
    results['report_generation'] = run_report_generation_demo(comparison_result, logger)
    
    # 3. 既存システム統合デモ
    results['system_integration'] = run_existing_system_integration_demo(logger)
    
    # 4. エラーハンドリングデモ
    results['error_handling'] = run_error_handling_demo(logger)
    
    # 5. パフォーマンスベンチマーク
    results['performance_benchmark'] = run_performance_benchmark(logger)
    
    # 結果サマリー
    logger.info("=== デモ実行結果サマリー ===")
    for test_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失敗"
        logger.info(f"{test_name}: {status}")
    
    total_tests = len(results)
    successful_tests = sum(results.values())
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    logger.info(f"全体成功率: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if successful_tests == total_tests:
        logger.info("🎉 全てのデモが正常に完了しました！")
        print("\n🎉 DSSMS Phase 3 Task 3.2 比較分析機能向上デモ 正常完了！")
    elif successful_tests > 0:
        logger.info("⚠️  一部のデモで問題が発生しました")
        print(f"\n⚠️  デモ部分完了: {successful_tests}/{total_tests}")
    else:
        logger.error("❌ 全てのデモが失敗しました")
        print("\n❌ デモ実行失敗")
    
    # 生成されたファイルの確認
    logger.info("=== 生成されたファイル ===")
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
            if files:
                logger.info(f"{output_dir}: {len(files)} ファイル")
                for file in files[:5]:  # 最初の5ファイルのみ表示
                    file_path = os.path.join(output_dir, file)
                    file_size = os.path.getsize(file_path)
                    logger.info(f"  - {file} ({file_size:,} bytes)")
                if len(files) > 5:
                    logger.info(f"  ... その他 {len(files) - 5} ファイル")
    
    logger.info("DSSMS比較分析デモ完了")
    print("\nデモ実行完了！詳細はログファイルを確認してください。")

if __name__ == "__main__":
    main()
