#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4-B-3-2: Real Market Data統合テスト

実際の市場データ（5803.T, 7203.T, 6758.T）を使用した
統合システムの互換性・安定性検証

バックテスト基本理念遵守・Phase 4-B系列成果維持
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
# データ取得・処理関数は関数内でインポート

# ロガー設定
logger = setup_logger(__name__)


def phase4b3_real_market_data_integration_test() -> Tuple[bool, Dict[str, Any]]:
    """
    Phase 4-B-3-2: Real Market Data統合テスト
    実際の市場データでの統合システム互換性検証
    
    Returns:
        Tuple[bool, Dict]: (成功フラグ, テスト結果詳細)
    """
    try:
        logger.info("Phase 4-B-3-2: Starting real market data integration test")
        
        # テスト対象銘柄（実際の市場データ）
        test_symbols = ['5803.T', '7203.T', '6758.T']
        test_results = {}
        overall_success = True
        
        for symbol in test_symbols:
            logger.info(f"Testing real market data integration for {symbol}")
            
            # 個別銘柄テスト実行
            symbol_result = test_single_symbol_integration(symbol)
            test_results[symbol] = symbol_result
            
            # 総合成功判定
            if not symbol_result.get('integration_success', False):
                overall_success = False
                logger.warning(f"Real market data test failed for {symbol}")
        
        # [OK] Real market data統合品質評価
        integration_quality_score = calculate_real_data_integration_quality(test_results)
        
        # [OK] バックテスト基本理念遵守確認（Real data環境）
        backtest_principle_compliance = verify_backtest_principle_in_real_data(test_results)
        
        final_result = {
            'overall_success': overall_success,
            'tested_symbols': test_symbols,
            'individual_results': test_results,
            'integration_quality_score': integration_quality_score,
            'backtest_principle_compliance': backtest_principle_compliance,
            'test_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'test_summary': {
                'total_symbols': len(test_symbols),
                'successful_symbols': sum(1 for r in test_results.values() if r.get('integration_success', False)),
                'total_trades_generated': sum(r.get('trades_count', 0) for r in test_results.values()),
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: 'excel_outputs_created': sum(1 for r in test_results.values() if r.get('excel_output_success', False))
            }
        }
        
        logger.info(f"Phase 4-B-3-2 Real Market Data Integration Test: Success={overall_success}")
        logger.info(f"  - Symbols tested: {len(test_symbols)}")
        logger.info(f"  - Successful integrations: {final_result['test_summary']['successful_symbols']}")
        logger.info(f"  - Total trades generated: {final_result['test_summary']['total_trades_generated']}")
        logger.info(f"  - Integration quality score: {integration_quality_score}")
        
        return overall_success, final_result
        
    except Exception as e:
        logger.error(f"Phase 4-B-3-2 real market data integration test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False, {'error': str(e)}


def test_single_symbol_integration(symbol: str) -> Dict[str, Any]:
    """
    単一銘柄での統合システムテスト
    
    Args:
        symbol (str): テスト対象銘柄コード
    
    Returns:
        Dict[str, Any]: 個別銘柄テスト結果
    """
    try:
        logger.info(f"Starting integration test for symbol: {symbol}")
        
        # [OK] Real market dataの取得
        data_fetcher_result = fetch_real_market_data(symbol)
        if not data_fetcher_result.get('data_fetch_success', False):
            logger.warning(f"Failed to fetch real market data for {symbol}")
            return {
                'integration_success': False,
                'error': 'Data fetch failed',
                'symbol': symbol
            }
        
        # [OK] 統合システムでのバックテスト実行
        backtest_result = execute_integrated_backtest_with_real_data(
            symbol, 
            data_fetcher_result['data']
        )
        
        # [OK] 結果品質検証
        quality_metrics = validate_real_data_backtest_quality(backtest_result, symbol)
        
        symbol_result = {
            'symbol': symbol,
            'integration_success': backtest_result.get('backtest_execution_success', False),
            'data_fetch_success': data_fetcher_result.get('data_fetch_success', False),
            'data_rows': data_fetcher_result.get('data_rows', 0),
            'trades_count': backtest_result.get('trades_count', 0),
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Entry_Signal/Exit_Signal output affected
# ORIGINAL: 'excel_output_success': backtest_result.get('excel_output_success', False),
            'quality_score': quality_metrics.get('overall_quality_score', 0),
            'backtest_principle_compliant': quality_metrics.get('backtest_principle_compliant', False),
            'execution_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_period': data_fetcher_result.get('data_period', 'Unknown'),
            'performance_metrics': backtest_result.get('performance_metrics', {}),
            'error_details': backtest_result.get('error', None) if not backtest_result.get('backtest_execution_success', False) else None
        }
        
        logger.info(f"Symbol {symbol} integration test completed: Success={symbol_result['integration_success']}")
        return symbol_result
        
    except Exception as e:
        logger.error(f"Integration test failed for symbol {symbol}: {e}")
        return {
            'symbol': symbol,
            'integration_success': False,
            'error': str(e),
            'execution_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def fetch_real_market_data(symbol: str) -> Dict[str, Any]:
    """
    実際の市場データを取得
    
    Args:
        symbol (str): 銘柄コード
    
    Returns:
        Dict[str, Any]: データ取得結果
    """
    try:
        logger.info(f"Fetching real market data for {symbol}")
        
        # 既存のfetch_stock_data関数を使用してリアルデータ取得
        from data_fetcher import fetch_stock_data
        
        # 過去1年間のデータを取得
        end_date = datetime.now()
        start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
        
        raw_data = fetch_stock_data(
            symbol, 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d")
        )
        
        if raw_data is None or raw_data.empty:
            logger.warning(f"No data fetched for {symbol}")
            return {
                'data_fetch_success': False,
                'error': 'No data available',
                'symbol': symbol
            }
        
        # データ前処理（既存のpreprocess_data関数を使用）
        from data_processor import preprocess_data
        processed_data = preprocess_data(raw_data.copy())
        
        # データ品質確認
        data_quality = validate_market_data_quality(processed_data, symbol)
        
        return {
            'data_fetch_success': True,
            'data': processed_data,
            'data_rows': len(processed_data),
            'data_period': f"{processed_data.index[0].strftime('%Y-%m-%d')} to {processed_data.index[-1].strftime('%Y-%m-%d')}",
            'data_quality': data_quality,
            'fetch_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': symbol
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch real market data for {symbol}: {e}")
        return {
            'data_fetch_success': False,
            'error': str(e),
            'symbol': symbol
        }


def validate_market_data_quality(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    市場データの品質検証
    
    Args:
        data (pd.DataFrame): 市場データ
        symbol (str): 銘柄コード
    
    Returns:
        Dict[str, Any]: データ品質評価
    """
    try:
        quality_metrics = {
            'data_completeness': len(data) > 200,  # 最低200営業日
            'required_columns_present': all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']),
            'no_missing_critical_values': not data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any(),
            'price_data_validity': (data['High'] >= data['Low']).all() and (data['High'] >= data['Close']).all(),
            'volume_data_validity': (data['Volume'] >= 0).all(),
            'technical_indicators_present': any(col in data.columns for col in ['RSI_14', 'SMA_5', 'SMA_25'])
        }
        
        overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
        
        return {
            'overall_quality': overall_quality,
            'quality_checks': quality_metrics,
            'data_rows': len(data),
            'date_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
            'symbol': symbol
        }
        
    except Exception as e:
        logger.error(f"Data quality validation failed for {symbol}: {e}")
        return {
            'overall_quality': 0,
            'quality_checks': {},
            'error': str(e)
        }


def execute_integrated_backtest_with_real_data(symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Real market dataを使用した統合バックテスト実行
    
    Args:
        symbol (str): 銘柄コード
        data (pd.DataFrame): 市場データ
    
    Returns:
        Dict[str, Any]: バックテスト実行結果
    """
    try:
        logger.info(f"Executing integrated backtest for {symbol} with real market data")
        
        # [OK] バックテスト基本理念遵守: 実際のbacktest実行必須
        # 統合システムの戦略を順次実行
        strategies_to_test = [
            'VWAPBreakoutStrategy',
            'MomentumInvestingStrategy',
            'BreakoutStrategy'
        ]
        
        integrated_results = pd.DataFrame()
        total_trades = 0
        strategy_performances = {}
        
        for strategy_name in strategies_to_test:
            try:
                # 個別戦略実行
                strategy_result = execute_single_strategy_with_real_data(strategy_name, data, symbol)
                
                if strategy_result.get('execution_success', False):
                    # 取引統計更新
                    trades_in_strategy = strategy_result.get('trades_count', 0)
                    total_trades += trades_in_strategy
                    strategy_performances[strategy_name] = {
                        'trades': trades_in_strategy,
                        'performance': strategy_result.get('performance_metrics', {})
                    }
                    
                    # 結果統合
                    if strategy_result.get('result_data') is not None:
                        integrated_results = strategy_result['result_data']
                        
                else:
                    logger.warning(f"Strategy {strategy_name} failed for {symbol}")
                    
            except Exception as strategy_error:
                logger.error(f"Strategy {strategy_name} execution failed for {symbol}: {strategy_error}")
        
        # [OK] Excel出力実行
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: excel_output_success = False
        excel_file_path = None
        
        if total_trades > 0 and not integrated_results.empty:
            try:
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Entry_Signal/Exit_Signal output affected
# ORIGINAL: excel_file_path = create_real_data_excel_output(symbol, integrated_results, strategy_performances)
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: excel_output_success = excel_file_path is not None
            except Exception as excel_error:
                logger.error(f"Excel output failed for {symbol}: {excel_error}")
        
        # バックテスト実行結果
        backtest_result = {
            'backtest_execution_success': total_trades > 0,
            'trades_count': total_trades,
            'strategies_tested': strategies_to_test,
            'successful_strategies': len(strategy_performances),
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: 'excel_output_success': excel_output_success,
            'excel_file_path': excel_file_path,
            'performance_metrics': calculate_integrated_performance_metrics(strategy_performances),
            'execution_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': symbol,
            'data_rows_processed': len(data)
        }
        
        logger.info(f"Integrated backtest completed for {symbol}: {total_trades} trades generated")
        return backtest_result
        
    except Exception as e:
        logger.error(f"Integrated backtest execution failed for {symbol}: {e}")
        return {
            'backtest_execution_success': False,
            'error': str(e),
            'symbol': symbol
        }


def execute_single_strategy_with_real_data(strategy_name: str, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    単一戦略をreal market dataで実行
    
    Args:
        strategy_name (str): 戦略名
        data (pd.DataFrame): 市場データ
        symbol (str): 銘柄コード
    
    Returns:
        Dict[str, Any]: 戦略実行結果
    """
    try:
        # 戦略パラメータ取得
        from config.optimized_parameters import get_optimized_parameters, get_default_parameters
        
        try:
            params = get_optimized_parameters(strategy_name)
        except:
            params = get_default_parameters(strategy_name)
        
        # 戦略クラス取得・実行
        strategy_class = get_strategy_class(strategy_name)
        if strategy_class is None:
            return {
                'execution_success': False,
                'error': f'Strategy class not found: {strategy_name}'
            }
        
        # [OK] バックテスト基本理念遵守: 実際のbacktest()実行
        strategy_instance = strategy_class(**params)
        result_data = strategy_instance.backtest(data.copy())
        
        # シグナル・取引検証
        trades_count = 0
        if 'Entry_Signal' in result_data.columns and 'Exit_Signal' in result_data.columns:
            trades_count = (result_data['Entry_Signal'] == 1).sum() + (result_data['Exit_Signal'] == 1).sum()
        
        return {
            'execution_success': trades_count > 0,
            'trades_count': trades_count,
            'result_data': result_data,
            'performance_metrics': calculate_strategy_performance_metrics(result_data),
            'strategy_name': strategy_name,
            'symbol': symbol
        }
        
    except Exception as e:
        logger.error(f"Single strategy execution failed for {strategy_name} on {symbol}: {e}")
        return {
            'execution_success': False,
            'error': str(e),
            'strategy_name': strategy_name,
            'symbol': symbol
        }


def get_strategy_class(strategy_name: str):
    """戦略クラスを取得"""
    try:
        if strategy_name == 'VWAPBreakoutStrategy':
            from strategies.vwap_breakout_strategy import VWAPBreakoutStrategy
            return VWAPBreakoutStrategy
        elif strategy_name == 'MomentumInvestingStrategy':
            from strategies.momentum_investing_strategy import MomentumInvestingStrategy
            return MomentumInvestingStrategy
        elif strategy_name == 'BreakoutStrategy':
            from strategies.breakout_strategy import BreakoutStrategy
            return BreakoutStrategy
        else:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return None
    except ImportError as e:
        logger.error(f"Failed to import strategy {strategy_name}: {e}")
        return None


# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Entry_Signal/Exit_Signal output affected
# ORIGINAL: def create_real_data_excel_output(symbol: str, results: pd.DataFrame, strategy_performances: Dict) -> Optional[str]:
    """
    Real market dataテスト結果のExcel出力作成
    
    Args:
        symbol (str): 銘柄コード
        results (pd.DataFrame): バックテスト結果
        strategy_performances (Dict): 戦略パフォーマンス
    
    Returns:
        Optional[str]: 出力ファイルパス
    """
    try:
        from output.simple_excel_exporter import SimpleExcelExporter
        
        exporter = SimpleExcelExporter()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Phase 4-B-3-2専用出力ディレクトリ
        output_dir = "backtest_results/phase4b3_real_data_tests"
        os.makedirs(output_dir, exist_ok=True)
        
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: output_file = f"{output_dir}/real_data_test_{symbol}_{timestamp}.xlsx"
        
        # Excel出力実行
        exporter.export_to_excel(results, symbol, output_file)
        
        logger.info(f"Real data test Excel output created: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to create Excel output for {symbol}: {e}")
        return None


def calculate_strategy_performance_metrics(result_data: pd.DataFrame) -> Dict[str, Any]:
    """戦略パフォーマンス指標計算"""
    try:
        if 'Portfolio_Value' in result_data.columns:
            initial_value = result_data['Portfolio_Value'].iloc[0]
            final_value = result_data['Portfolio_Value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value
        else:
            total_return = 0
        
        return {
            'total_return': total_return,
            'final_portfolio_value': result_data.get('Portfolio_Value', pd.Series([0])).iloc[-1],
            'data_points': len(result_data)
        }
    except:
        return {'total_return': 0, 'final_portfolio_value': 0, 'data_points': 0}


def calculate_integrated_performance_metrics(strategy_performances: Dict) -> Dict[str, Any]:
    """統合パフォーマンス指標計算"""
    try:
        total_trades = sum(perf['trades'] for perf in strategy_performances.values())
        avg_return = sum(perf['performance'].get('total_return', 0) for perf in strategy_performances.values()) / len(strategy_performances) if strategy_performances else 0
        
        return {
            'total_integrated_trades': total_trades,
            'average_strategy_return': avg_return,
            'successful_strategies': len(strategy_performances)
        }
    except:
        return {'total_integrated_trades': 0, 'average_strategy_return': 0, 'successful_strategies': 0}


def validate_real_data_backtest_quality(backtest_result: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """Real data backtestの品質検証"""
    try:
        # バックテスト基本理念遵守確認
        backtest_principle_compliant = (
            backtest_result.get('backtest_execution_success', False) and
            backtest_result.get('trades_count', 0) > 0 and
            backtest_result.get('successful_strategies', 0) > 0
        )
        
        # 品質スコア計算
        quality_factors = {
            'execution_success': backtest_result.get('backtest_execution_success', False),
            'adequate_trades': backtest_result.get('trades_count', 0) >= 10,
            'multiple_strategies': backtest_result.get('successful_strategies', 0) >= 2,
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Entry_Signal/Exit_Signal output affected
# ORIGINAL: 'excel_output': backtest_result.get('excel_output_success', False)
        }
        
        overall_quality_score = sum(quality_factors.values()) / len(quality_factors)
        
        return {
            'overall_quality_score': overall_quality_score,
            'backtest_principle_compliant': backtest_principle_compliant,
            'quality_factors': quality_factors,
            'symbol': symbol
        }
        
    except Exception as e:
        logger.error(f"Quality validation failed for {symbol}: {e}")
        return {
            'overall_quality_score': 0,
            'backtest_principle_compliant': False,
            'error': str(e)
        }


def calculate_real_data_integration_quality(test_results: Dict[str, Dict]) -> float:
    """Real data統合品質スコア計算"""
    try:
        quality_scores = [result.get('quality_score', 0) for result in test_results.values()]
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0
    except:
        return 0


def verify_backtest_principle_in_real_data(test_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Real data環境でのバックテスト基本理念遵守確認"""
    try:
        compliance_checks = {}
        total_compliant = 0
        
        for symbol, result in test_results.items():
            is_compliant = result.get('backtest_principle_compliant', False)
            compliance_checks[symbol] = is_compliant
            if is_compliant:
                total_compliant += 1
        
        overall_compliance = total_compliant / len(test_results) if test_results else 0
        
        return {
            'overall_compliance_rate': overall_compliance,
            'compliant_symbols': total_compliant,
            'total_symbols': len(test_results),
            'individual_compliance': compliance_checks,
            'compliance_threshold_met': overall_compliance >= 0.8  # 80%以上で合格
        }
        
    except Exception as e:
        logger.error(f"Backtest principle verification failed: {e}")
        return {
            'overall_compliance_rate': 0,
            'compliance_threshold_met': False,
            'error': str(e)
        }


def phase4b3_real_data_test_report(test_results: Tuple[bool, Dict[str, Any]]) -> str:
    """
    Phase 4-B-3-2 Real market data統合テスト結果レポート生成
    
    Args:
        test_results: Real market dataテスト結果
    
    Returns:
        str: レポート内容
    """
    success, results = test_results
    
    report = f"""
# Phase 4-B-3-2: Real Market Data統合テスト結果レポート

## 実行サマリー
- **実行日時**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Real Market Data統合テスト**: {'[OK] 成功' if success else '[ERROR] 失敗'}
- **テスト対象銘柄**: {', '.join(results.get('tested_symbols', []))}

## テスト結果サマリー
- **総銘柄数**: {results.get('test_summary', {}).get('total_symbols', 0)}
- **統合成功銘柄数**: {results.get('test_summary', {}).get('successful_symbols', 0)}
- **総取引数**: {results.get('test_summary', {}).get('total_trades_generated', 0)}
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: - **Excel出力作成数**: {results.get('test_summary', {}).get('excel_outputs_created', 0)}
- **統合品質スコア**: {results.get('integration_quality_score', 0):.2f}

## 個別銘柄結果
"""
    
    individual_results = results.get('individual_results', {})
    for symbol, result in individual_results.items():
        status = '[OK]' if result.get('integration_success', False) else '[ERROR]'
        trades = result.get('trades_count', 0)
        quality = result.get('quality_score', 0)
        
        report += f"""
### {symbol}
- **統合テスト**: {status} {'成功' if result.get('integration_success', False) else '失敗'}
- **取引数**: {trades}
- **品質スコア**: {quality:.2f}
- **データ行数**: {result.get('data_rows', 0)}
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: - **Excel出力**: {'[OK]' if result.get('excel_output_success', False) else '[ERROR]'}
"""
    
    # バックテスト基本理念遵守確認
    backtest_compliance = results.get('backtest_principle_compliance', {})
    report += f"""
## バックテスト基本理念遵守確認
- **全体遵守率**: {backtest_compliance.get('overall_compliance_rate', 0):.1%}
- **遵守銘柄数**: {backtest_compliance.get('compliant_symbols', 0)}/{backtest_compliance.get('total_symbols', 0)}
- **遵守基準達成**: {'[OK]' if backtest_compliance.get('compliance_threshold_met', False) else '[ERROR]'}
"""
    
    return report


if __name__ == "__main__":
    """Phase 4-B-3-2実行: Real Market Data統合テスト"""
    
    logger.info("Starting Phase 4-B-3-2: Real Market Data Integration Test")
    
    try:
        # Phase 4-B-3-2: Real Market Data統合テスト実行
        test_success, test_results = phase4b3_real_market_data_integration_test()
        
        # 結果レポート生成
        report = phase4b3_real_data_test_report((test_success, test_results))
        
        # レポート出力
        report_file = f"Phase4B3_2_Real_Data_Integration_Test_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 結果表示
        print("\n" + "="*80)
        print("[ROCKET] Phase 4-B-3-2: Real Market Data統合テスト 実行完了")
        print("="*80)
        print(f"[CHART] Real Data統合テスト結果: {'[OK] 成功' if test_success else '[ERROR] 失敗'}")
        
        if test_success:
            test_summary = test_results.get('test_summary', {})
            print(f"[TARGET] テスト対象銘柄: {test_summary.get('total_symbols', 0)}")
            print(f"[OK] 統合成功銘柄: {test_summary.get('successful_symbols', 0)}")
            print(f"[UP] 総取引数: {test_summary.get('total_trades_generated', 0)}")
            print(f"[CHART] 統合品質スコア: {test_results.get('integration_quality_score', 0):.2f}")
            
            backtest_compliance = test_results.get('backtest_principle_compliance', {})
            print(f"🔒 バックテスト基本理念遵守: {backtest_compliance.get('overall_compliance_rate', 0):.1%}")
        else:
            print(f"[ERROR] 問題発見: {test_results.get('error', 'Unknown error')}")
        
        print(f"📄 詳細レポート: {report_file}")
        print("="*80)
        
        # 次工程への移行判定
        if test_success:
            print("[OK] Phase 4-B-3-3 (Production mode準備完了検証) への移行準備完了")
        else:
            print("[WARNING]  Real Market Data統合問題解決後にPhase 4-B-3-3へ移行")
            
    except Exception as e:
        logger.error(f"Phase 4-B-3-2 execution failed: {e}")
        print(f"[ERROR] Phase 4-B-3-2実行エラー: {e}")
        # TODO(tag:phase4b3, rationale:Phase 4-B-3-2 real market data integration success required)