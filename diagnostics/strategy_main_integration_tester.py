"""
同日Entry/Exit問題診断 - 戦略単体テスター（修正版）
戦略の実際のコンストラクタに合わせて診断を実行
"""
import sys
import os
import pandas as pd
import json
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from data_fetcher import get_parameters_and_data
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators

logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\diagnostics\strategy_diagnosis.log")

def test_strategy_with_main_integration(stock_data, index_data):
    """main.pyの戦略実行ロジックを使った診断"""
    
    # main.pyから必要な関数をインポートして使用
    sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")
    from main import _execute_individual_strategy, get_default_parameters
    
    # 戦略インポート
    from strategies.VWAP_Breakout import VWAPBreakoutStrategy
    from strategies.Momentum_Investing import MomentumInvestingStrategy
    from strategies.Breakout import BreakoutStrategy
    from strategies.VWAP_Bounce import VWAPBounceStrategy
    from strategies.Opening_Gap import OpeningGapStrategy
    from strategies.contrarian_strategy import ContrarianStrategy
    from strategies.gc_strategy_signal import GCStrategy
    
    strategies_to_test = [
        (VWAPBreakoutStrategy, "VWAPBreakoutStrategy"),
        (MomentumInvestingStrategy, "MomentumInvestingStrategy"),
        (BreakoutStrategy, "BreakoutStrategy"),
        (VWAPBounceStrategy, "VWAPBounceStrategy"),
        (OpeningGapStrategy, "OpeningGapStrategy"),
        (ContrarianStrategy, "ContrarianStrategy"),
        (GCStrategy, "GCStrategy")
    ]
    
    diagnosis_results = []
    problem_count = 0
    
    for strategy_class, strategy_name in strategies_to_test:
        logger.info(f"=== {strategy_name} main.py統合診断開始 ===")
        
        try:
            # デフォルトパラメータ取得
            params = get_default_parameters(strategy_name)
            
            # main.pyの戦略実行関数を使用
            result = _execute_individual_strategy(stock_data, index_data, strategy_name, strategy_class, params)
            
            if result is None:
                logger.error(f"{strategy_name}: 戦略実行結果がNone")
                continue
                
            # シグナル存在確認
            if 'Entry_Signal' not in result.columns or 'Exit_Signal' not in result.columns:
                logger.error(f"{strategy_name}: Entry_Signal/Exit_Signal列が存在しません")
                continue
            
            # 同日Entry/Exit検出
            same_day_entries_exits = []
            for date in result.index:
                entry_val = result.loc[date, 'Entry_Signal']
                exit_val = result.loc[date, 'Exit_Signal']
                
                if entry_val == 1 and exit_val == 1:
                    same_day_info = {
                        'date': date.strftime('%Y-%m-%d'),
                        'entry_signal': int(entry_val),
                        'exit_signal': int(exit_val)
                    }
                    same_day_entries_exits.append(same_day_info)
                    
                    # 要求仕様のログ出力
                    logger.warning(f"{strategy_name} 同日Entry/Exit検出: {date.strftime('%Y-%m-%d')} Entry={entry_val} Exit={exit_val}")
            
            # 統計情報
            entry_signals = result[result['Entry_Signal'] == 1]
            exit_signals = result[result['Exit_Signal'] == 1]
            
            # 正常パターンのログ
            if not same_day_entries_exits:
                if len(entry_signals) > 0 and len(exit_signals) > 0:
                    first_entry = entry_signals.index[0].strftime('%Y-%m-%d')
                    first_exit = exit_signals.index[0].strftime('%Y-%m-%d')
                    logger.info(f"{strategy_name} エントリーシグナル {first_entry} エグジットシグナル {first_exit} 正常")
                elif len(entry_signals) > 0:
                    first_entry = entry_signals.index[0].strftime('%Y-%m-%d')
                    logger.info(f"{strategy_name} エントリーシグナル {first_entry} エグジットシグナルなし 正常")
                elif len(exit_signals) > 0:
                    first_exit = exit_signals.index[0].strftime('%Y-%m-%d')
                    logger.info(f"{strategy_name} エントリーシグナルなし エグジットシグナル {first_exit} 正常")
                else:
                    logger.info(f"{strategy_name} エントリー・エグジットシグナルなし 正常")
            
            diagnosis_result = {
                'strategy_name': strategy_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'has_same_day_problem': len(same_day_entries_exits) > 0,
                'same_day_entries_exits': same_day_entries_exits,
                'signal_summary': {
                    'total_entries': len(entry_signals),
                    'total_exits': len(exit_signals),
                    'entry_dates': entry_signals.index.strftime('%Y-%m-%d').tolist(),
                    'exit_dates': exit_signals.index.strftime('%Y-%m-%d').tolist()
                }
            }
            
            if diagnosis_result['has_same_day_problem']:
                problem_count += 1
                logger.warning(f"{strategy_name}: 同日Entry/Exit問題検出 ({len(same_day_entries_exits)}件)")
            else:
                logger.info(f"{strategy_name}: 同日Entry/Exit問題なし")
                
            diagnosis_results.append(diagnosis_result)
            
        except Exception as e:
            logger.error(f"{strategy_name} 診断エラー: {e}")
            diagnosis_results.append({
                'strategy_name': strategy_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            })
    
    return diagnosis_results, problem_count

def run_strategy_diagnostics_main_integration():
    """main.py統合を使った戦略診断を実行"""
    
    # データ取得
    logger.info("診断用データ取得開始")
    ticker = "^N225"
    ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data(ticker)
    
    if stock_data is None or len(stock_data) == 0:
        logger.error("診断用データの取得に失敗")
        return
    
    # データ前処理
    stock_data = preprocess_data(stock_data)
    stock_data = compute_indicators(stock_data)
    
    logger.info(f"診断用データ準備完了: {len(stock_data)}行, 期間: {stock_data.index[0]} - {stock_data.index[-1]}")
    
    # 戦略診断実行
    diagnosis_results, problem_count = test_strategy_with_main_integration(stock_data, index_data)
    
    # 結果保存
    diagnosis_summary = {
        'diagnosis_timestamp': datetime.now().isoformat(),
        'method': 'main_integration',
        'total_strategies_tested': len(diagnosis_results),
        'strategies_with_same_day_problem': problem_count,
        'data_info': {
            'ticker': ticker,
            'total_rows': len(stock_data),
            'period': f"{stock_data.index[0]} to {stock_data.index[-1]}"
        },
        'strategy_results': diagnosis_results
    }
    
    # JSON保存
    output_file = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\strategy_main_integration_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diagnosis_summary, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"戦略main.py統合診断完了: {problem_count}/{len(diagnosis_results)}戦略で同日Entry/Exit問題検出")
    logger.info(f"診断結果保存: {output_file}")
    
    return diagnosis_summary

if __name__ == "__main__":
    print("=== 同日Entry/Exit問題診断 - main.py統合版 ===")
    results = run_strategy_diagnostics_main_integration()
    
    if results:
        print(f"\n診断完了:")
        print(f"- 検証戦略数: {results['total_strategies_tested']}")
        print(f"- 問題検出戦略数: {results['strategies_with_same_day_problem']}")
        print(f"- 結果ファイル: diagnostics/strategy_main_integration_results.json")