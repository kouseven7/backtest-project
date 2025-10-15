"""
同日Entry/Exit問題診断 - 戦略単体テスター
各戦略を個別実行してシグナル発生タイミングを検証
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

# 戦略インポート
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.Breakout import BreakoutStrategy
from strategies.VWAP_Bounce import VWAPBounceStrategy
from strategies.Opening_Gap import OpeningGapStrategy
from strategies.Opening_Gap_Fixed import OpeningGapFixedStrategy
from strategies.contrarian_strategy import ContrarianStrategy
from strategies.gc_strategy_signal import GCStrategy

logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\diagnostics\strategy_diagnosis.log")

def test_single_strategy(strategy_class, strategy_name, stock_data, index_data, params):
    """単一戦略のシグナル発生パターンを診断"""
    
    diagnosis_result = {
        'strategy_name': strategy_name,
        'timestamp': datetime.now().isoformat(),
        'data_period': f"{stock_data.index[0]} to {stock_data.index[-1]}",
        'total_rows': len(stock_data),
        'signals': {},
        'same_day_entries_exits': [],
        'signal_summary': {}
    }
    
    try:
        # 戦略実行
        logger.info(f"=== {strategy_name} 単体診断開始 ===")
        strategy = strategy_class(**params)
        result = strategy.backtest(stock_data, index_data)
        
        # シグナル列の存在確認
        if 'Entry_Signal' not in result.columns or 'Exit_Signal' not in result.columns:
            diagnosis_result['error'] = f"Required signal columns missing: Entry_Signal or Exit_Signal"
            return diagnosis_result
            
        # シグナル統計
        entry_signals = result[result['Entry_Signal'] == 1]
        exit_signals = result[result['Exit_Signal'] == 1]
        
        diagnosis_result['signal_summary'] = {
            'total_entries': len(entry_signals),
            'total_exits': len(exit_signals),
            'entry_dates': entry_signals.index.strftime('%Y-%m-%d').tolist(),
            'exit_dates': exit_signals.index.strftime('%Y-%m-%d').tolist()
        }
        
        # 同日Entry/Exit検出
        for date in result.index:
            entry_val = result.loc[date, 'Entry_Signal']
            exit_val = result.loc[date, 'Exit_Signal']
            
            if entry_val == 1 and exit_val == 1:
                same_day_info = {
                    'date': date.strftime('%Y-%m-%d'),
                    'entry_signal': int(entry_val),
                    'exit_signal': int(exit_val),
                    'close_price': float(result.loc[date, 'Close']) if 'Close' in result.columns else None
                }
                diagnosis_result['same_day_entries_exits'].append(same_day_info)
                
                # ログ出力（要求仕様）
                logger.warning(f"{strategy_name} 同日Entry/Exit検出: {date.strftime('%Y-%m-%d')} Entry={entry_val} Exit={exit_val}")
        
        # 正常パターンのログ出力
        if not diagnosis_result['same_day_entries_exits']:
            if len(entry_signals) > 0 and len(exit_signals) > 0:
                first_entry = entry_signals.index[0].strftime('%Y-%m-%d')
                first_exit = exit_signals.index[0].strftime('%Y-%m-%d')
                logger.info(f"{strategy_name} エントリーシグナル {first_entry} エグジットシグナル {first_exit} 正常")
        
        diagnosis_result['status'] = 'completed'
        diagnosis_result['has_same_day_problem'] = len(diagnosis_result['same_day_entries_exits']) > 0
        
    except Exception as e:
        diagnosis_result['error'] = str(e)
        diagnosis_result['status'] = 'failed'
        logger.error(f"{strategy_name} 診断エラー: {e}")
    
    return diagnosis_result

def run_all_strategy_diagnostics():
    """全戦略の診断を実行"""
    
    # データ取得
    logger.info("診断用データ取得開始")
    ticker = "^N225"  # 日経平均
    ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data(ticker)
    
    if stock_data is None or len(stock_data) == 0:
        logger.error("診断用データの取得に失敗")
        return
    
    # データ前処理
    stock_data = preprocess_data(stock_data)
    stock_data = compute_indicators(stock_data)
    
    logger.info(f"診断用データ準備完了: {len(stock_data)}行, 期間: {stock_data.index[0]} - {stock_data.index[-1]}")
    
    # 戦略リスト（デフォルトパラメータ付き）
    strategies_to_test = [
        (VWAPBreakoutStrategy, "VWAPBreakoutStrategy", {
            'vwap_period': 20, 'volume_threshold_multiplier': 1.5, 
            'breakout_threshold': 0.02, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.10
        }),
        (MomentumInvestingStrategy, "MomentumInvestingStrategy", {
            'momentum_period': 14, 'rsi_period': 14, 'rsi_overbought': 70, 
            'rsi_oversold': 30, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.10
        }),
        (BreakoutStrategy, "BreakoutStrategy", {
            'lookback_period': 20, 'volume_threshold': 1.5, 'breakout_threshold': 0.02,
            'stop_loss_pct': 0.05, 'take_profit_pct': 0.10
        }),
        (VWAPBounceStrategy, "VWAPBounceStrategy", {
            'vwap_period': 20, 'deviation_threshold': 0.02, 'volume_threshold': 1.2,
            'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
        }),
        (OpeningGapStrategy, "OpeningGapStrategy", {
            'gap_threshold': 0.02, 'volume_threshold': 1.5, 'confirmation_period': 3,
            'stop_loss_pct': 0.05, 'take_profit_pct': 0.10
        }),
        (ContrarianStrategy, "ContrarianStrategy", {
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'stop_loss_pct': 0.05, 'take_profit_pct': 0.08
        }),
        (GCStrategy, "GCStrategy", {
            'short_window': 5, 'long_window': 25, 'stop_loss': 0.05, 'take_profit': 0.10
        })
    ]
    
    all_results = []
    problem_count = 0
    
    # 各戦略を診断
    for strategy_class, strategy_name, params in strategies_to_test:
        logger.info(f"戦略診断開始: {strategy_name}")
        result = test_single_strategy(strategy_class, strategy_name, stock_data, index_data, params)
        all_results.append(result)
        
        if result.get('has_same_day_problem', False):
            problem_count += 1
            logger.warning(f"{strategy_name}: 同日Entry/Exit問題検出 ({len(result['same_day_entries_exits'])}件)")
        else:
            logger.info(f"{strategy_name}: 同日Entry/Exit問題なし")
    
    # 結果保存
    diagnosis_summary = {
        'diagnosis_timestamp': datetime.now().isoformat(),
        'total_strategies_tested': len(strategies_to_test),
        'strategies_with_same_day_problem': problem_count,
        'data_info': {
            'ticker': ticker,
            'total_rows': len(stock_data),
            'period': f"{stock_data.index[0]} to {stock_data.index[-1]}"
        },
        'strategy_results': all_results
    }
    
    # JSON保存
    output_file = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\strategy_diagnosis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diagnosis_summary, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"戦略単体診断完了: {problem_count}/{len(strategies_to_test)}戦略で同日Entry/Exit問題検出")
    logger.info(f"診断結果保存: {output_file}")
    
    return diagnosis_summary

if __name__ == "__main__":
    print("=== 同日Entry/Exit問題診断 - 戦略単体テスト ===")
    results = run_all_strategy_diagnostics()
    
    if results:
        print(f"\n診断完了:")
        print(f"- 検証戦略数: {results['total_strategies_tested']}")
        print(f"- 問題検出戦略数: {results['strategies_with_same_day_problem']}")
        print(f"- 結果ファイル: diagnostics/strategy_diagnosis_results.json")