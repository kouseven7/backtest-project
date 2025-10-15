"""
Module: integration_validator.py
Description: 
  統合処理検証用の診断ツール。
  複数戦略の統合前後でシグナルがどのように変化するかを検証し、
  問題が発生するポイントを特定します。

Author: diagnostic-tool
Created: 2025-10-15
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import traceback
import copy
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from data_fetcher import get_parameters_and_data
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from signal_processing import detect_exit_anomalies, check_same_day_entry_exit, filter_same_day_exit_signals

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
log_file = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\logs\integration_validator.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logger = setup_logger("integration_validator", log_file=log_file)

# 結果保存ディレクトリ
RESULTS_DIR = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\results"
CHARTS_DIR = r"C:\Users\imega\Documents\my_backtest_project\diagnostics\charts"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


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


def execute_strategy(strategy_name: str, strategy_class, stock_data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """単一戦略を実行する"""
    try:
        logger.info(f"戦略 {strategy_name} を実行中...")
        
        # 戦略インスタンス作成
        strategy = strategy_class(stock_data, params)
        
        # バックテスト実行
        result_df = strategy.backtest()
        
        # 結果の分析
        entry_count = int((result_df['Entry_Signal'] == 1).sum())
        exit_count = int((result_df['Exit_Signal'] == 1).sum())
        same_day_analysis = check_same_day_entry_exit(result_df)
        
        logger.info(f"戦略 {strategy_name} の実行完了: エントリー {entry_count}回, エグジット {exit_count}回")
        
        if same_day_analysis['has_same_day_signals']:
            logger.warning(f"戦略 {strategy_name} で同日Entry/Exitを {same_day_analysis['same_day_count']} 件検出")
        
        return {
            'status': 'success',
            'entry_count': entry_count,
            'exit_count': exit_count,
            'same_day_signals': same_day_analysis,
            'dataframe': result_df
        }
    
    except Exception as e:
        logger.error(f"戦略 {strategy_name} の実行に失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'error': str(e)
        }


def integrate_signals_standard(strategy_results: Dict[str, Dict[str, Any]], 
                             stock_data: pd.DataFrame, 
                             priority_list: List[str]) -> Dict[str, Any]:
    """標準の方法で複数戦略の結果を統合する"""
    logger.info("標準方式で戦略結果を統合中...")
    
    try:
        # 統合用データフレームの初期化
        integrated_data = stock_data.copy()
        integrated_data['Entry_Signal'] = 0
        integrated_data['Exit_Signal'] = 0
        integrated_data['Active_Strategy'] = ''
        integrated_data['Strategy_Confidence'] = 0.0
        integrated_data['Position'] = 0
        
        # 各戦略の結果を優先順位に従って統合
        integration_steps = []
        
        for strategy_name in priority_list:
            if (strategy_name not in strategy_results or 
                strategy_results[strategy_name].get('status') != 'success'):
                continue
            
            strategy_result = strategy_results[strategy_name]['dataframe']
            
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
            integration_steps.append(step_info)
            
            logger.info(f"戦略 {strategy_name} の統合: "
                       f"エントリー +{entry_count}, エグジット +{exit_count}")
        
        # 同日Entry/Exitの検出
        same_day_analysis = check_same_day_entry_exit(integrated_data)
        
        # 結果をまとめる
        integration_result = {
            'status': 'success',
            'method': 'standard',
            'steps': integration_steps,
            'entry_count': int((integrated_data['Entry_Signal'] == 1).sum()),
            'exit_count': int((integrated_data['Exit_Signal'] == 1).sum()),
            'same_day_signals': same_day_analysis,
            'dataframe': integrated_data
        }
        
        if same_day_analysis['has_same_day_signals']:
            logger.warning(f"統合後も同日Entry/Exitが {same_day_analysis['same_day_count']} 件残っています")
            
        return integration_result
        
    except Exception as e:
        logger.error(f"標準統合処理に失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'method': 'standard',
            'error': str(e)
        }


def integrate_signals_with_filter(strategy_results: Dict[str, Dict[str, Any]], 
                                stock_data: pd.DataFrame, 
                                priority_list: List[str]) -> Dict[str, Any]:
    """filter_same_day_exit_signalsを使用して同日シグナルをフィルタリングする統合方法"""
    logger.info("フィルタリング方式で戦略結果を統合中...")
    
    try:
        # 標準統合と同様のプロセスで実行
        integrated_data = stock_data.copy()
        integrated_data['Entry_Signal'] = 0
        integrated_data['Exit_Signal'] = 0
        integrated_data['Active_Strategy'] = ''
        integrated_data['Strategy_Confidence'] = 0.0
        integrated_data['Position'] = 0
        
        integration_steps = []
        
        for strategy_name in priority_list:
            if (strategy_name not in strategy_results or 
                strategy_results[strategy_name].get('status') != 'success'):
                continue
            
            strategy_result = strategy_results[strategy_name]['dataframe']
            
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
            
            step_info = {
                'strategy': strategy_name,
                'before': before_integration,
                'after': after_integration,
                'changes': {
                    'new_entries': entry_count,
                    'new_exits': exit_count
                }
            }
            integration_steps.append(step_info)
        
        # 同日Entry/Exit問題の検出と修正前の状態を保存
        same_day_before = check_same_day_entry_exit(integrated_data)
        before_filter = {
            'entry_count': int((integrated_data['Entry_Signal'] == 1).sum()),
            'exit_count': int((integrated_data['Exit_Signal'] == 1).sum()),
            'same_day_count': same_day_before['same_day_count']
        }
        
        # filter_same_day_exit_signals関数を適用
        if same_day_before['has_same_day_signals']:
            logger.info(f"フィルタリング前の同日Entry/Exit: {same_day_before['same_day_count']}件")
            filtered_data = filter_same_day_exit_signals(integrated_data)
            
            # フィルタリング後の状態を検証
            same_day_after = check_same_day_entry_exit(filtered_data)
            after_filter = {
                'entry_count': int((filtered_data['Entry_Signal'] == 1).sum()),
                'exit_count': int((filtered_data['Exit_Signal'] == 1).sum()),
                'same_day_count': same_day_after['same_day_count']
            }
            
            # 結果を記録
            filter_result = {
                'before': before_filter,
                'after': after_filter,
                'changes': {
                    'entry_diff': after_filter['entry_count'] - before_filter['entry_count'],
                    'exit_diff': after_filter['exit_count'] - before_filter['exit_count'],
                    'same_day_diff': after_filter['same_day_count'] - before_filter['same_day_count']
                }
            }
            
            logger.info(f"フィルタリング結果: 同日Entry/Exit {filter_result['changes']['same_day_diff']}件の変化")
            
            if same_day_after['has_same_day_signals']:
                logger.warning(f"フィルタリング後も同日Entry/Exitが {same_day_after['same_day_count']} 件残っています")
        else:
            logger.info("同日Entry/Exitはないため、フィルタリングは不要です")
            filtered_data = integrated_data.copy()
            same_day_after = same_day_before
            filter_result = {
                'before': before_filter,
                'after': before_filter,
                'changes': {'entry_diff': 0, 'exit_diff': 0, 'same_day_diff': 0}
            }
        
        # 結果をまとめる
        integration_result = {
            'status': 'success',
            'method': 'filtered',
            'steps': integration_steps,
            'filter_result': filter_result,
            'entry_count': int((filtered_data['Entry_Signal'] == 1).sum()),
            'exit_count': int((filtered_data['Exit_Signal'] == 1).sum()),
            'same_day_signals': same_day_after,
            'dataframe': filtered_data
        }
        
        return integration_result
        
    except Exception as e:
        logger.error(f"フィルタリング統合処理に失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'method': 'filtered',
            'error': str(e)
        }


def create_signal_comparison_chart(original_signals: Dict[str, Dict[str, Any]],
                                 standard_integrated: pd.DataFrame,
                                 filtered_integrated: Optional[pd.DataFrame],
                                 ticker: str,
                                 same_day_dates: List[str]) -> str:
    """同日Entry/Exitの問題を可視化するチャートを作成"""
    try:
        if not same_day_dates:
            logger.info("同日Entry/Exit問題がないため、チャートは作成しません")
            return ""
        
        # 先頭の5つの問題日だけを可視化
        dates_to_visualize = same_day_dates[:5]
        
        # 戦略名のリスト
        strategy_names = list(original_signals.keys())
        
        # 日付ごとにチャートを作成
        chart_files = []
        
        for date_str in dates_to_visualize:
            date_obj = pd.to_datetime(date_str)
            
            # チャートのサイズと設定
            plt.figure(figsize=(15, 10))
            sns.set_style("whitegrid")
            
            # タイトル設定
            plt.suptitle(f"{ticker} - 同日Entry/Exit問題分析: {date_str}", fontsize=16)
            
            # 戦略ごとのシグナルを可視化
            ax1 = plt.subplot(2, 1, 1)
            
            # 各戦略のシグナル状態を収集
            entries = []
            exits = []
            labels = []
            
            for i, strategy_name in enumerate(strategy_names):
                if (strategy_name in original_signals and 
                    original_signals[strategy_name].get('status') == 'success'):
                    df = original_signals[strategy_name]['dataframe']
                    
                    if date_obj in df.index:
                        row = df.loc[date_obj]
                        entry_signal = int(row.get('Entry_Signal', 0))
                        exit_signal = int(row.get('Exit_Signal', 0))
                        
                        entries.append(entry_signal)
                        exits.append(exit_signal)
                        labels.append(strategy_name)
            
            # 戦略ごとのシグナル状態をプロット
            x = np.arange(len(labels))
            width = 0.35
            
            ax1.bar(x - width/2, entries, width, label='Entry_Signal')
            ax1.bar(x + width/2, exits, width, label='Exit_Signal')
            
            ax1.set_ylabel('シグナル値')
            ax1.set_title('戦略別シグナル状態')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.legend()
            ax1.set_ylim(0, 1.5)  # シグナルは0または1
            
            # 統合結果を可視化
            ax2 = plt.subplot(2, 1, 2)
            
            # 統合データを準備
            integration_methods = ['標準統合']
            integration_entry = [int(standard_integrated.loc[date_obj, 'Entry_Signal'])]
            integration_exit = [int(standard_integrated.loc[date_obj, 'Exit_Signal'])]
            
            if filtered_integrated is not None:
                integration_methods.append('フィルタリング統合')
                if date_obj in filtered_integrated.index:
                    integration_entry.append(int(filtered_integrated.loc[date_obj, 'Entry_Signal']))
                    integration_exit.append(int(filtered_integrated.loc[date_obj, 'Exit_Signal']))
                else:
                    integration_entry.append(0)
                    integration_exit.append(0)
            
            # 統合結果をプロット
            x = np.arange(len(integration_methods))
            width = 0.35
            
            ax2.bar(x - width/2, integration_entry, width, label='Entry_Signal')
            ax2.bar(x + width/2, integration_exit, width, label='Exit_Signal')
            
            ax2.set_ylabel('シグナル値')
            ax2.set_title('統合後のシグナル状態')
            ax2.set_xticks(x)
            ax2.set_xticklabels(integration_methods)
            ax2.legend()
            ax2.set_ylim(0, 1.5)  # シグナルは0または1
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # チャートを保存
            chart_path = os.path.join(CHARTS_DIR, f"{ticker}_same_day_signal_{date_str.replace('-', '')}.png")
            plt.savefig(chart_path)
            plt.close()
            
            chart_files.append(chart_path)
            logger.info(f"チャート作成: {chart_path}")
        
        # サマリーチャート（すべての同日シグナルの総数）を作成
        plt.figure(figsize=(12, 8))
        
        # 戦略ごとの同日シグナル数
        strategy_same_day_counts = {}
        for strategy_name in strategy_names:
            if (strategy_name in original_signals and 
                original_signals[strategy_name].get('status') == 'success'):
                same_day_info = original_signals[strategy_name].get('same_day_signals', {})
                count = same_day_info.get('same_day_count', 0)
                strategy_same_day_counts[strategy_name] = count
        
        # 上位の戦略をプロット
        strategies = list(strategy_same_day_counts.keys())
        counts = list(strategy_same_day_counts.values())
        
        # 降順にソート
        sorted_indices = np.argsort(counts)[::-1]
        strategies = [strategies[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        plt.bar(strategies, counts)
        plt.xlabel('戦略')
        plt.ylabel('同日Entry/Exit件数')
        plt.title(f'{ticker} - 戦略別の同日Entry/Exit発生件数')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        summary_chart_path = os.path.join(CHARTS_DIR, f"{ticker}_same_day_signal_summary.png")
        plt.savefig(summary_chart_path)
        plt.close()
        
        chart_files.append(summary_chart_path)
        logger.info(f"サマリーチャート作成: {summary_chart_path}")
        
        return summary_chart_path
        
    except Exception as e:
        logger.error(f"チャート作成に失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return ""


def validate_integration(ticker: str, 
                        start_date: str, 
                        end_date: str,
                        focus_strategies: List[str] = None) -> Dict[str, Any]:
    """統合処理の検証を実行する"""
    logger.info(f"===== 統合処理検証開始 =====")
    logger.info(f"銘柄: {ticker}, 期間: {start_date} から {end_date}")
    
    try:
        # データ取得
        stock_data, index_data = get_parameters_and_data(ticker, start_date, end_date)
        
        # データ前処理
        stock_data = preprocess_data(stock_data)
        
        # インジケータ計算
        compute_indicators(stock_data)
        
        # 検証結果の初期化
        validation_results = {
            'ticker': ticker,
            'date_range': {'start': start_date, 'end': end_date},
            'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'focus_strategies': focus_strategies,
            'original_signals': {},
            'integration_results': {},
            'same_day_signals': {},
            'charts': {}
        }
        
        # 実行する戦略の設定
        strategies = []
        if focus_strategies:
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
            for strategy_short_name in focus_strategies:
                if strategy_short_name in strategy_map:
                    strategies.append(strategy_map[strategy_short_name])
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
        
        # フェーズ1: 各戦略の個別実行
        logger.info("フェーズ1: 各戦略の個別実行")
        original_signals = {}
        
        for strategy_name, strategy_class in strategies:
            params = get_default_parameters(strategy_name)
            result = execute_strategy(strategy_name, strategy_class, stock_data, params)
            original_signals[strategy_name] = result
            
            # 結果CSVを保存
            if result.get('status') == 'success' and 'dataframe' in result:
                csv_path = os.path.join(RESULTS_DIR, f"integration_validator_{ticker}_{strategy_name}.csv")
                result['dataframe'].to_csv(csv_path)
        
        # 戦略の優先順位設定
        priority_list = [
            'VWAPBreakoutStrategy',
            'MomentumInvestingStrategy', 
            'BreakoutStrategy',
            'VWAPBounceStrategy',
            'OpeningGapFixedStrategy',
            'OpeningGapStrategy',
            'ContrarianStrategy',
            'GCStrategy'
        ]
        
        # フェーズ2: 統合処理の検証
        logger.info("フェーズ2: 統合処理の検証")
        
        # 標準統合の実行
        standard_integration = integrate_signals_standard(original_signals, stock_data, priority_list)
        
        # フィルタリング統合の実行
        filtered_integration = integrate_signals_with_filter(original_signals, stock_data, priority_list)
        
        # 統合結果の保存
        if standard_integration.get('status') == 'success' and 'dataframe' in standard_integration:
            standard_csv_path = os.path.join(RESULTS_DIR, f"integration_validator_{ticker}_standard_integrated.csv")
            standard_integration['dataframe'].to_csv(standard_csv_path)
        
        if filtered_integration.get('status') == 'success' and 'dataframe' in filtered_integration:
            filtered_csv_path = os.path.join(RESULTS_DIR, f"integration_validator_{ticker}_filtered_integrated.csv")
            filtered_integration['dataframe'].to_csv(filtered_csv_path)
        
        # 同日シグナルの詳細分析
        same_day_signals = {}
        same_day_dates = set()
        
        # 各戦略の同日シグナルを収集
        for strategy_name, result in original_signals.items():
            if result.get('status') == 'success' and 'same_day_signals' in result:
                same_day_info = result['same_day_signals']
                if same_day_info.get('has_same_day_signals', False):
                    dates = same_day_info.get('dates', [])
                    same_day_signals[strategy_name] = dates
                    same_day_dates.update(dates)
        
        # 標準統合後の同日シグナルを収集
        if (standard_integration.get('status') == 'success' and 
            'same_day_signals' in standard_integration):
            same_day_info = standard_integration['same_day_signals']
            if same_day_info.get('has_same_day_signals', False):
                dates = same_day_info.get('dates', [])
                same_day_signals['標準統合後'] = dates
                same_day_dates.update(dates)
        
        # フィルタリング統合後の同日シグナルを収集
        if (filtered_integration.get('status') == 'success' and 
            'same_day_signals' in filtered_integration):
            same_day_info = filtered_integration['same_day_signals']
            if same_day_info.get('has_same_day_signals', False):
                dates = same_day_info.get('dates', [])
                same_day_signals['フィルタリング統合後'] = dates
                same_day_dates.update(dates)
        
        # 同日シグナルの可視化
        same_day_list = list(same_day_dates)
        same_day_list.sort()
        
        standard_df = standard_integration.get('dataframe') if standard_integration.get('status') == 'success' else None
        filtered_df = filtered_integration.get('dataframe') if filtered_integration.get('status') == 'success' else None
        
        chart_path = ""
        if standard_df is not None:
            chart_path = create_signal_comparison_chart(
                original_signals, 
                standard_df, 
                filtered_df, 
                ticker, 
                same_day_list
            )
        
        # 結果の保存
        validation_results.update({
            'original_signals': {k: v for k, v in original_signals.items() if k != 'dataframe'},
            'integration_results': {
                'standard': {k: v for k, v in standard_integration.items() if k != 'dataframe'},
                'filtered': {k: v for k, v in filtered_integration.items() if k != 'dataframe'}
            },
            'same_day_signals': {
                'by_strategy': same_day_signals,
                'all_dates': same_day_list
            },
            'charts': {
                'summary_chart': chart_path
            }
        })
        
        # JSONに変換できるデータに簡略化
        simplified_results = copy.deepcopy(validation_results)
        for strategy_name in simplified_results['original_signals']:
            if 'dataframe' in simplified_results['original_signals'][strategy_name]:
                del simplified_results['original_signals'][strategy_name]['dataframe']
        
        # 結果をJSONファイルに保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(RESULTS_DIR, f"integration_validation_{ticker}_{timestamp}.json")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"詳細結果を保存: {json_path}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"統合処理検証に失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='バックテストの統合処理検証')
    parser.add_argument('--ticker', type=str, default='7203.T', help='対象の銘柄コード')
    parser.add_argument('--start', type=str, default='2024-01-01', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--strategies', type=str, help='対象戦略（カンマ区切り）。省略時は全戦略対象')
    
    args = parser.parse_args()
    
    print(f"===== バックテストの統合処理検証 =====")
    print(f"対象銘柄: {args.ticker}")
    print(f"期間: {args.start} から {args.end}")
    
    focus_strategies = None
    if args.strategies:
        focus_strategies = [s.strip() for s in args.strategies.split(',')]
        print(f"対象戦略: {', '.join(focus_strategies)}")
    else:
        print("全戦略を対象に検証します")
    
    print("検証を開始します...")
    results = validate_integration(args.ticker, args.start, args.end, focus_strategies)
    
    if 'error' in results:
        print(f"エラーが発生しました: {results['error']}")
    else:
        print("\n===== 検証結果サマリー =====")
        
        # 同日シグナルの総数
        same_day_count = len(results['same_day_signals']['all_dates']) if 'same_day_signals' in results else 0
        print(f"同日Entry/Exit発生日数: {same_day_count}日")
        
        # 戦略別の同日シグナル
        if 'same_day_signals' in results and 'by_strategy' in results['same_day_signals']:
            strategy_signals = results['same_day_signals']['by_strategy']
            print("\n戦略別の同日Entry/Exit発生数:")
            for strategy, dates in strategy_signals.items():
                print(f"  - {strategy}: {len(dates)}件")
        
        # 統合結果の比較
        if ('integration_results' in results and 
            'standard' in results['integration_results'] and 
            'filtered' in results['integration_results']):
            
            std_result = results['integration_results']['standard']
            flt_result = results['integration_results']['filtered']
            
            print("\n統合方式による比較:")
            print(f"  標準統合: エントリー {std_result.get('entry_count', 'N/A')}件, "
                 f"エグジット {std_result.get('exit_count', 'N/A')}件, "
                 f"同日シグナル {std_result.get('same_day_signals', {}).get('same_day_count', 0)}件")
            
            print(f"  フィルタ統合: エントリー {flt_result.get('entry_count', 'N/A')}件, "
                 f"エグジット {flt_result.get('exit_count', 'N/A')}件, "
                 f"同日シグナル {flt_result.get('same_day_signals', {}).get('same_day_count', 0)}件")
            
            # フィルタリング効果
            if 'filter_result' in flt_result:
                filter_changes = flt_result['filter_result'].get('changes', {})
                print(f"\nフィルタリング効果:")
                print(f"  エントリー変化: {filter_changes.get('entry_diff', 0)}件")
                print(f"  エグジット変化: {filter_changes.get('exit_diff', 0)}件")
                print(f"  同日シグナル削減: {-filter_changes.get('same_day_diff', 0)}件")
        
        print(f"\n詳細結果は {RESULTS_DIR} ディレクトリに保存されています")
        
        if 'charts' in results and 'summary_chart' in results['charts']:
            chart_path = results['charts']['summary_chart']
            if chart_path:
                print(f"チャートは {CHARTS_DIR} ディレクトリに保存されています")