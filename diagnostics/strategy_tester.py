"""
Module: strategy_tester.py
Description: 
  戦略単体テスト用の診断ツール。
  各戦略を個別に実行して同日Entry/Exit発生有無を検証します。

Author: diagnostic-tool
Created: 2025-10-15
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from data_fetcher import get_parameters_and_data
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from signal_processing import detect_exit_anomalies, check_same_day_entry_exit

# インポートする戦略クラス
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.Breakout import BreakoutStrategy
from strategies.VWAP_Bounce import VWAPBounceStrategy
from strategies.Opening_Gap import OpeningGapStrategy
from strategies.Opening_Gap_Fixed import OpeningGapFixedStrategy
from strategies.Opening_Gap_Enhanced import OpeningGapEnhancedStrategy
from strategies.contrarian_strategy import ContrarianStrategy
from strategies.gc_strategy_signal import GCStrategy

# 診断用ロガーの設定
log_file = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\logs\strategy_test.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logger = setup_logger("strategy_tester", log_file=log_file)

# 結果保存ディレクトリ
RESULTS_DIR = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_default_parameters(strategy_name: str) -> Dict[str, Any]:
    """戦略のデフォルトパラメータを取得"""
    defaults = {
        'VWAPBreakoutStrategy': {
            'vwap_period': 20,
            'volume_threshold_multiplier': 1.5,
            'breakout_threshold': 0.02,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        },
        'MomentumInvestingStrategy': {
            'momentum_period': 14,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        },
        'BreakoutStrategy': {
            'lookback_period': 20,
            'volume_threshold': 1.5,
            'breakout_threshold': 0.02,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        },
        'VWAPBounceStrategy': {
            'vwap_period': 20,
            'deviation_threshold': 0.02,
            'volume_threshold': 1.2,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        },
        'OpeningGapStrategy': {
            'gap_threshold': 0.02,
            'volume_threshold': 1.5,
            'confirmation_period': 3,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        },
        'OpeningGapFixedStrategy': {
            'gap_threshold': 0.01,
            'stop_loss': 0.02,
            'take_profit': 0.05
        },
        'OpeningGapEnhancedStrategy': {
            'gap_threshold': 0.01,
            'stop_loss': 0.02,
            'take_profit': 0.05
        },
        'ContrarianStrategy': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.08
        },
        'GCStrategy': {
            'short_window': 5,
            'long_window': 25,
            'stop_loss': 0.05,
            'take_profit': 0.10
        }
    }
    
    return defaults.get(strategy_name, {})


def analyze_same_day_signals(result_df: pd.DataFrame) -> Dict[str, Any]:
    """
    同日Entry/Exitの分析と発生原因の詳細調査
    特に各種シグナル、指標、価格条件などを詳細に調査して
    同日Entry/Exitが発生する根本原因を特定する
    """
    analysis = {
        'has_same_day_signals': False,
        'same_day_count': 0,
        'dates': [],
        'details': [],
        'root_causes': []  # 根本原因分析の結果を格納
    }
    
    if 'Entry_Signal' in result_df.columns and 'Exit_Signal' in result_df.columns:
        # 同じ日にEntry=1とExit=1の両方がある日を検出
        same_day_mask = (result_df['Entry_Signal'] == 1) & (result_df['Exit_Signal'] == 1)
        same_day_count = same_day_mask.sum()
        
        if same_day_count > 0:
            analysis['has_same_day_signals'] = True
            analysis['same_day_count'] = int(same_day_count)
            
            # 詳細情報を収集
            same_day_rows = result_df[same_day_mask]
            for idx, row in same_day_rows.iterrows():
                date_str = idx.strftime('%Y-%m-%d')
                analysis['dates'].append(date_str)
                
                detail = {
                    'date': date_str,
                    'open': float(row.get('Open', 0)),
                    'high': float(row.get('High', 0)),
                    'low': float(row.get('Low', 0)), 
                    'close': float(row.get('Close', 0)),
                    'entry_signal': int(row.get('Entry_Signal', 0)),
                    'exit_signal': int(row.get('Exit_Signal', 0)),
                    'price_action': {
                        'intraday_range_pct': (float(row.get('High', 0)) - float(row.get('Low', 0))) / float(row.get('Open', 0)) * 100 if float(row.get('Open', 0)) > 0 else 0,
                        'open_to_close_pct': (float(row.get('Close', 0)) - float(row.get('Open', 0))) / float(row.get('Open', 0)) * 100 if float(row.get('Open', 0)) > 0 else 0,
                    }
                }
                
                # 追加のシグナル情報があれば収集
                # 全てのシグナルとインジケータを収集
                for col in row.index:
                    # シグナル収集
                    if col.endswith('_Signal') and col not in ['Entry_Signal', 'Exit_Signal']:
                        detail[col] = float(row.get(col, 0))
                    # インジケータ収集
                    elif col in ['RSI', 'MACD', 'MACDSignal', 'MACDHist', 'SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'ATR', 'Bollinger_Upper', 'Bollinger_Lower', 'VWAP']:
                        detail['indicators'] = detail.get('indicators', {})
                        detail['indicators'][col] = float(row.get(col, 0))
                
                # 根本原因の推定
                root_cause = estimate_root_cause(row, detail)
                if root_cause:
                    analysis['root_causes'].append(root_cause)
                
                analysis['details'].append(detail)
    
    return analysis


def test_single_strategy(ticker: str, start_date: str, end_date: str, 
                        strategy_name: str, strategy_class, params: Dict[str, Any]) -> Dict[str, Any]:
    """単一戦略の実行とテスト"""
    logger.info(f"===== 戦略テスト開始: {strategy_name} =====")
    
    try:
        # データ取得
        stock_data, index_data = get_parameters_and_data(ticker, start_date, end_date)
        
        # データ前処理
        stock_data = preprocess_data(stock_data)
        
        # インジケータ計算
        compute_indicators(stock_data)
        
        # 戦略インスタンス作成
        strategy = strategy_class(stock_data, params)
        
        # バックテスト実行
        logger.info(f"バックテスト実行中: {strategy_name}")
        result_df = strategy.backtest()
        
        # 結果の検証
        signal_analysis = analyze_same_day_signals(result_df)
        anomaly_info = detect_exit_anomalies(result_df, strategy_name)
        
        # 統計情報の収集
        entry_count = int((result_df.get('Entry_Signal', 0) == 1).sum())
        exit_count = int((result_df.get('Exit_Signal', 0) == 1).sum())
        
        test_results = {
            'strategy_name': strategy_name,
            'ticker': ticker,
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'param_count': len(params),
            'params': params,
            'data_points': len(result_df),
            'date_range': {
                'start': result_df.index[0].strftime('%Y-%m-%d'),
                'end': result_df.index[-1].strftime('%Y-%m-%d')
            },
            'signal_counts': {
                'entry': entry_count,
                'exit': exit_count,
                'diff': entry_count - exit_count
            },
            'same_day_analysis': signal_analysis,
            'exit_anomalies': anomaly_info,
            'status': 'success'
        }
        
        # ログ出力
        if signal_analysis['has_same_day_signals']:
            logger.warning(f"同日Entry/Exit検出: {signal_analysis['same_day_count']}件")
            for date in signal_analysis['dates'][:5]:
                logger.warning(f"  - 日付: {date}")
            if len(signal_analysis['dates']) > 5:
                logger.warning(f"  - ...他 {len(signal_analysis['dates']) - 5} 件")
        
        # 結果をCSVファイルに保存
        csv_path = os.path.join(RESULTS_DIR, f"{ticker}_{strategy_name}_results.csv")
        result_df.to_csv(csv_path)
        logger.info(f"結果CSVを保存: {csv_path}")
        
        return test_results
    
    except Exception as e:
        logger.error(f"戦略テスト失敗: {strategy_name}, エラー: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'strategy_name': strategy_name,
            'ticker': ticker,
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'error',
            'error': str(e)
        }


def test_all_strategies(ticker: str = '7203.T', 
                       start_date: str = '2024-01-01', 
                       end_date: str = '2024-12-31') -> Dict[str, Any]:
    """全戦略のテスト実行"""
    
    # テスト対象の戦略
    strategies = [
        ('VWAPBreakoutStrategy', VWAPBreakoutStrategy),
        ('MomentumInvestingStrategy', MomentumInvestingStrategy),
        ('BreakoutStrategy', BreakoutStrategy),
        ('VWAPBounceStrategy', VWAPBounceStrategy),
        ('OpeningGapStrategy', OpeningGapStrategy),
        ('OpeningGapFixedStrategy', OpeningGapFixedStrategy),
        ('OpeningGapEnhancedStrategy', OpeningGapEnhancedStrategy),
        ('ContrarianStrategy', ContrarianStrategy),
        ('GCStrategy', GCStrategy)
    ]
    
    all_results = {
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker,
        'date_range': {'start': start_date, 'end': end_date},
        'strategy_count': len(strategies),
        'strategy_results': {},
        'summary': {
            'success_count': 0,
            'error_count': 0,
            'strategies_with_same_day_signals': [],
            'total_same_day_signals': 0
        }
    }
    
    # 各戦略をテスト
    for strategy_name, strategy_class in strategies:
        params = get_default_parameters(strategy_name)
        result = test_single_strategy(ticker, start_date, end_date, strategy_name, strategy_class, params)
        all_results['strategy_results'][strategy_name] = result
        
        # サマリー情報の更新
        if result['status'] == 'success':
            all_results['summary']['success_count'] += 1
            
            # 同日シグナルのカウント
            if result.get('same_day_analysis', {}).get('has_same_day_signals', False):
                all_results['summary']['strategies_with_same_day_signals'].append(strategy_name)
                all_results['summary']['total_same_day_signals'] += result['same_day_analysis']['same_day_count']
        else:
            all_results['summary']['error_count'] += 1
    
    # サマリーレポート
    logger.info("========== テスト完了 ==========")
    logger.info(f"成功: {all_results['summary']['success_count']}/{len(strategies)}")
    logger.info(f"同日Entry/Exit発生戦略: {len(all_results['summary']['strategies_with_same_day_signals'])}/{len(strategies)}")
    
    if all_results['summary']['strategies_with_same_day_signals']:
        logger.info("同日Entry/Exit発生戦略リスト:")
        for strategy in all_results['summary']['strategies_with_same_day_signals']:
            count = all_results['strategy_results'][strategy]['same_day_analysis']['same_day_count']
            logger.info(f"  - {strategy}: {count}件")
    
    # 結果をJSONファイルに保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(RESULTS_DIR, f"all_strategies_test_{timestamp}.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"詳細結果を保存: {json_path}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='バックテスト戦略の同日Entry/Exit問題診断')
    parser.add_argument('--ticker', type=str, default='7203.T', help='対象の銘柄コード')
    parser.add_argument('--start', type=str, default='2024-01-01', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--strategy', type=str, help='特定の戦略のみテスト (省略時は全戦略テスト)')
    
    args = parser.parse_args()
    
    print(f"===== バックテスト戦略の同日Entry/Exit問題診断 =====")
    print(f"対象銘柄: {args.ticker}")
    print(f"期間: {args.start} から {args.end}")
    
    if args.strategy:
        # 特定の戦略のみテスト
        print(f"対象戦略: {args.strategy}")
        strategy_map = {
            'VWAPBreakout': ('VWAPBreakoutStrategy', VWAPBreakoutStrategy),
            'MomentumInvesting': ('MomentumInvestingStrategy', MomentumInvestingStrategy),
            'Breakout': ('BreakoutStrategy', BreakoutStrategy),
            'VWAPBounce': ('VWAPBounceStrategy', VWAPBounceStrategy),
            'OpeningGap': ('OpeningGapStrategy', OpeningGapStrategy),
            'OpeningGapFixed': ('OpeningGapFixedStrategy', OpeningGapFixedStrategy),
            'OpeningGapEnhanced': ('OpeningGapEnhancedStrategy', OpeningGapEnhancedStrategy),
            'Contrarian': ('ContrarianStrategy', ContrarianStrategy),
            'GC': ('GCStrategy', GCStrategy)
        }
        
        if args.strategy in strategy_map:
            strategy_name, strategy_class = strategy_map[args.strategy]
            params = get_default_parameters(strategy_name)
            result = test_single_strategy(args.ticker, args.start, args.end, strategy_name, strategy_class, params)
            
            print(f"\n結果:")
            print(f"  状態: {'成功' if result['status'] == 'success' else '失敗'}")
            
            if result['status'] == 'success':
                print(f"  エントリーシグナル数: {result['signal_counts']['entry']}")
                print(f"  エグジットシグナル数: {result['signal_counts']['exit']}")
                print(f"  同日Entry/Exit: {result['same_day_analysis']['same_day_count']}件")
                
                if result['same_day_analysis']['has_same_day_signals']:
                    print("\n同日Entry/Exit発生日:")
                    for i, date in enumerate(result['same_day_analysis']['dates'][:10]):
                        print(f"  {i+1}. {date}")
                    
                    if len(result['same_day_analysis']['dates']) > 10:
                        print(f"  ...他 {len(result['same_day_analysis']['dates']) - 10} 件")
        else:
            print(f"エラー: 指定された戦略 '{args.strategy}' は存在しません")
    else:
        # 全戦略テスト
        print("全戦略テストを実行します...")
        results = test_all_strategies(args.ticker, args.start, args.end)
        
        print("\n===== テスト結果サマリー =====")
        print(f"成功戦略数: {results['summary']['success_count']}/{results['strategy_count']}")
        print(f"同日Entry/Exit発生戦略数: {len(results['summary']['strategies_with_same_day_signals'])}/{results['strategy_count']}")
        print(f"総同日Entry/Exit発生数: {results['summary']['total_same_day_signals']}件")
        
        if results['summary']['strategies_with_same_day_signals']:
            print("\n同日Entry/Exit発生戦略:")
            for strategy in results['summary']['strategies_with_same_day_signals']:
                count = results['strategy_results'][strategy]['same_day_analysis']['same_day_count']
                print(f"  - {strategy}: {count}件")
        
        print(f"\n詳細結果は {RESULTS_DIR} ディレクトリに保存されています")