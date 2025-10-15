"""
Module: signal_flow_tracer.py
Description: 
  シグナルフロー追跡用の診断ツール。
  エントリー→統合処理→エグジット→統合処理の各段階でログを取得し、
  シグナルがどの段階で変化するかを追跡します。

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
import copy
import traceback

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
from strategies.contrarian_strategy import ContrarianStrategy
from strategies.gc_strategy_signal import GCStrategy

# 診断用ロガーの設定
log_file = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\logs\signal_flow_tracer.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logger = setup_logger("signal_flow_tracer", log_file=log_file)

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


def trace_signal_flow(ticker: str, start_date: str, end_date: str, focus_strategy: str = None) -> Dict[str, Any]:
    """シグナルフローを追跡する関数"""
    logger.info(f"===== シグナルフロー追跡開始 =====")
    logger.info(f"銘柄: {ticker}, 期間: {start_date} から {end_date}")
    
    try:
        # データ取得
        stock_data, index_data = get_parameters_and_data(ticker, start_date, end_date)
        
        # データ前処理
        stock_data = preprocess_data(stock_data)
        
        # インジケータ計算
        compute_indicators(stock_data)
        
        # 追跡結果の初期化
        tracing_results = {
            'ticker': ticker,
            'date_range': {'start': start_date, 'end': end_date},
            'trace_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'focus_strategy': focus_strategy,
            'phase_results': {},
            'signal_flow': {},
            'same_day_signals': {},
            'signal_changes': {}
        }
        
        # 実行する戦略の設定
        strategies = []
        if focus_strategy:
            # 特定の戦略に焦点を当てる場合
            strategy_map = {
                'VWAPBreakout': ('VWAPBreakoutStrategy', VWAPBreakoutStrategy),
                'MomentumInvesting': ('MomentumInvestingStrategy', MomentumInvestingStrategy),
                'Breakout': ('BreakoutStrategy', BreakoutStrategy),
                'VWAPBounce': ('VWAPBounceStrategy', VWAPBounceStrategy),
                'OpeningGap': ('OpeningGapStrategy', OpeningGapStrategy),
                'OpeningGapFixed': ('OpeningGapFixedStrategy', OpeningGapFixedStrategy),
                'Contrarian': ('ContrarianStrategy', ContrarianStrategy),
                'GC': ('GCStrategy', GCStrategy)
            }
            if focus_strategy in strategy_map:
                strategies = [strategy_map[focus_strategy]]
            else:
                logger.error(f"指定された戦略 '{focus_strategy}' は存在しません")
                return {'error': f"指定された戦略 '{focus_strategy}' は存在しません"}
        else:
            # 全戦略を実行
            strategies = [
                ('VWAPBreakoutStrategy', VWAPBreakoutStrategy),
                ('MomentumInvestingStrategy', MomentumInvestingStrategy),
                ('BreakoutStrategy', BreakoutStrategy),
                ('VWAPBounceStrategy', VWAPBounceStrategy),
                ('OpeningGapStrategy', OpeningGapStrategy),
                ('OpeningGapFixedStrategy', OpeningGapFixedStrategy),
                ('ContrarianStrategy', ContrarianStrategy),
                ('GCStrategy', GCStrategy)
            ]
        
        # フェーズ1: 各戦略を個別に実行
        phase1_results = {}
        logger.info("フェーズ1: 各戦略の個別実行")
        
        for strategy_name, strategy_class in strategies:
            logger.info(f"  戦略 {strategy_name} を実行中...")
            params = get_default_parameters(strategy_name)
            
            try:
                # 戦略インスタンス作成
                strategy = strategy_class(stock_data, params)
                
                # バックテスト実行
                result_df = strategy.backtest()
                
                # 結果の分析
                same_day_analysis = check_same_day_entry_exit(result_df)
                
                # 結果を保存
                phase1_results[strategy_name] = {
                    'entry_count': int((result_df['Entry_Signal'] == 1).sum()),
                    'exit_count': int((result_df['Exit_Signal'] == 1).sum()),
                    'same_day_signals': same_day_analysis,
                    'dataframe': result_df
                }
                
                # 同日シグナルの詳細を記録
                if same_day_analysis['has_same_day_signals']:
                    logger.warning(f"  戦略 {strategy_name} で同日Entry/Exitを {same_day_analysis['same_day_count']} 件検出")
                    
                    # 同日シグナルの詳細を記録
                    same_day_dates = same_day_analysis['dates']
                    for date in same_day_dates:
                        date_obj = pd.to_datetime(date)
                        if date in tracing_results['same_day_signals']:
                            tracing_results['same_day_signals'][date]['strategies'].append(strategy_name)
                        else:
                            row_data = result_df.loc[date_obj].to_dict()
                            # 数値型に変換（シリアライズ可能にする）
                            for k, v in row_data.items():
                                if isinstance(v, (pd.Series, pd.DataFrame)):
                                    row_data[k] = v.values.tolist()
                                elif isinstance(v, (float, int, bool)) or v is None:
                                    row_data[k] = v
                                else:
                                    row_data[k] = str(v)
                            
                            tracing_results['same_day_signals'][date] = {
                                'strategies': [strategy_name],
                                'data': row_data
                            }
                
                # CSV形式で保存
                csv_path = os.path.join(RESULTS_DIR, f"signal_flow_{ticker}_{strategy_name}_phase1.csv")
                result_df.to_csv(csv_path)
                logger.info(f"  結果CSVを保存: {csv_path}")
                
            except Exception as e:
                logger.error(f"  戦略 {strategy_name} の実行に失敗: {str(e)}")
                logger.error(traceback.format_exc())
                phase1_results[strategy_name] = {
                    'error': str(e)
                }
        
        # フェーズ2: 統合処理のシミュレーション
        logger.info("フェーズ2: 統合処理のシミュレーション")
        
        # 統合用データフレームの初期化
        integrated_data = stock_data.copy()
        integrated_data['Entry_Signal'] = 0
        integrated_data['Exit_Signal'] = 0
        integrated_data['Active_Strategy'] = ''
        integrated_data['Strategy_Confidence'] = 0.0
        integrated_data['Position'] = 0
        
        # 各戦略の結果を統合（main.pyの統合ロジックをシミュレーション）
        phase2_results = {
            'integration_steps': [],
            'final_state': {}
        }
        
        # 戦略の優先順位に基づいて統合
        integration_priority = [
            'VWAPBreakoutStrategy',
            'MomentumInvestingStrategy', 
            'BreakoutStrategy',
            'VWAPBounceStrategy',
            'OpeningGapFixedStrategy',
            'OpeningGapStrategy',
            'ContrarianStrategy',
            'GCStrategy'
        ]
        
        # 各戦略の結果を優先順位に従って統合
        for strategy_name in integration_priority:
            if strategy_name not in phase1_results or 'error' in phase1_results[strategy_name]:
                continue
                
            strategy_result = phase1_results[strategy_name]['dataframe']
            
            # 統合前の状態を保存
            before_integration = {
                'entry_count': int((integrated_data['Entry_Signal'] == 1).sum()),
                'exit_count': int((integrated_data['Exit_Signal'] == 1).sum()),
            }
            
            # エントリーシグナル統合
            entry_count = 0
            entry_mask = (strategy_result['Entry_Signal'] == 1) & (integrated_data['Entry_Signal'] == 0)
            if entry_mask.any():
                integrated_data.loc[entry_mask, 'Entry_Signal'] = 1
                integrated_data.loc[entry_mask, 'Active_Strategy'] = strategy_name
                entry_count = entry_mask.sum()
            
            # エグジットシグナル統合
            exit_count = 0
            exit_indices = strategy_result[strategy_result['Exit_Signal'] == 1].index
            
            for exit_idx in exit_indices:
                # ポジションがあるかチェック（Active_Strategyが空でない）
                if integrated_data.loc[exit_idx, 'Active_Strategy'] != '':
                    integrated_data.loc[exit_idx, 'Exit_Signal'] = 1
                    integrated_data.loc[exit_idx, 'Active_Strategy'] = ''
                    exit_count += 1
            
            # 統合後の状態を保存
            after_integration = {
                'entry_count': int((integrated_data['Entry_Signal'] == 1).sum()),
                'exit_count': int((integrated_data['Exit_Signal'] == 1).sum()),
            }
            
            # 統合ステップを記録
            step_info = {
                'strategy': strategy_name,
                'before': before_integration,
                'after': after_integration,
                'changes': {
                    'new_entries': entry_count,
                    'new_exits': exit_count
                }
            }
            phase2_results['integration_steps'].append(step_info)
            
            logger.info(f"  戦略 {strategy_name} の統合: "
                      f"エントリー +{entry_count}, エグジット +{exit_count}")
        
        # 同日Entry/Exitの検出
        same_day_after_integration = check_same_day_entry_exit(integrated_data)
        phase2_results['final_state'] = {
            'entry_count': int((integrated_data['Entry_Signal'] == 1).sum()),
            'exit_count': int((integrated_data['Exit_Signal'] == 1).sum()),
            'same_day_signals': same_day_after_integration
        }
        
        if same_day_after_integration['has_same_day_signals']:
            logger.warning(f"統合後も同日Entry/Exitが {same_day_after_integration['same_day_count']} 件残っています")
            for date in same_day_after_integration['dates'][:5]:
                logger.warning(f"  - 日付: {date}")
            if len(same_day_after_integration['dates']) > 5:
                logger.warning(f"  - ...他 {len(same_day_after_integration['dates']) - 5} 件")
        
        # 統合結果のCSVを保存
        integrated_csv_path = os.path.join(RESULTS_DIR, f"signal_flow_{ticker}_integrated_results.csv")
        integrated_data.to_csv(integrated_csv_path)
        logger.info(f"統合結果CSVを保存: {integrated_csv_path}")
        
        # 最終結果をまとめる
        tracing_results['phase_results'] = {
            'phase1': phase1_results,
            'phase2': phase2_results
        }
        
        # シグナルフローの分析
        signal_flow_analysis = {}
        for date_str in tracing_results['same_day_signals']:
            strategies = tracing_results['same_day_signals'][date_str]['strategies']
            date_obj = pd.to_datetime(date_str)
            
            # 各戦略でのシグナル状態を確認
            strategies_signals = {}
            for strategy_name in strategies:
                if strategy_name in phase1_results and 'dataframe' in phase1_results[strategy_name]:
                    df = phase1_results[strategy_name]['dataframe']
                    if date_obj in df.index:
                        row = df.loc[date_obj]
                        strategies_signals[strategy_name] = {
                            'Entry_Signal': int(row.get('Entry_Signal', 0)),
                            'Exit_Signal': int(row.get('Exit_Signal', 0))
                        }
            
            # 統合後の状態を確認
            integrated_state = None
            if date_obj in integrated_data.index:
                row = integrated_data.loc[date_obj]
                integrated_state = {
                    'Entry_Signal': int(row.get('Entry_Signal', 0)),
                    'Exit_Signal': int(row.get('Exit_Signal', 0)),
                    'Active_Strategy': row.get('Active_Strategy', '')
                }
            
            signal_flow_analysis[date_str] = {
                'individual_strategies': strategies_signals,
                'integrated_state': integrated_state
            }
        
        tracing_results['signal_flow'] = signal_flow_analysis
        
        # 結果をJSONファイルに保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(RESULTS_DIR, f"signal_flow_trace_{ticker}_{timestamp}.json")
        
        # JSONに変換できるデータだけを保存
        simplified_results = copy.deepcopy(tracing_results)
        for strategy_name in phase1_results:
            if 'dataframe' in simplified_results['phase_results']['phase1'][strategy_name]:
                del simplified_results['phase_results']['phase1'][strategy_name]['dataframe']
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"詳細結果を保存: {json_path}")
        
        return tracing_results
        
    except Exception as e:
        logger.error(f"シグナルフロー追跡失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='バックテストのシグナルフロー追跡')
    parser.add_argument('--ticker', type=str, default='7203.T', help='対象の銘柄コード')
    parser.add_argument('--start', type=str, default='2024-01-01', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--strategy', type=str, help='特定の戦略のみ追跡 (省略時は全戦略)')
    
    args = parser.parse_args()
    
    print(f"===== バックテストのシグナルフロー追跡 =====")
    print(f"対象銘柄: {args.ticker}")
    print(f"期間: {args.start} から {args.end}")
    
    if args.strategy:
        print(f"対象戦略: {args.strategy}")
    else:
        print("全戦略を対象に追跡します")
    
    print("追跡を開始します...")
    results = trace_signal_flow(args.ticker, args.start, args.end, args.strategy)
    
    if 'error' in results:
        print(f"エラーが発生しました: {results['error']}")
    else:
        print("\n===== 追跡結果サマリー =====")
        
        # 同日シグナルの総数
        same_day_count = len(results['same_day_signals'])
        print(f"同日Entry/Exit発生日数: {same_day_count}日")
        
        # 戦略別の同日シグナル集計
        strategy_counts = {}
        for date_str, info in results['same_day_signals'].items():
            for strategy in info['strategies']:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        if strategy_counts:
            print("\n戦略別の同日Entry/Exit発生数:")
            for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {strategy}: {count}件")
        
        # 統合後の同日シグナル
        if 'phase2' in results['phase_results']:
            phase2 = results['phase_results']['phase2']
            if 'final_state' in phase2 and 'same_day_signals' in phase2['final_state']:
                same_day_after = phase2['final_state']['same_day_signals']
                if same_day_after['has_same_day_signals']:
                    print(f"\n統合後の同日Entry/Exit: {same_day_after['same_day_count']}件")
                else:
                    print("\n統合後の同日Entry/Exitはありません")
        
        print(f"\n詳細結果は {RESULTS_DIR} ディレクトリに保存されています")