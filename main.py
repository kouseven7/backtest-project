"""
Module: Main
File: main.py
Description: 
  マルチ戦略バックテストシステムのメインエントリーポイント。
  承認済みの最適化パラメータを使用して複数の戦略を実行し、
  統合されたバックテスト結果を生成します。

Author: imega
Created: 2023-04-01
Modified: 2025-12-30

Features:
  - 承認済み最適化パラメータの自動読み込み
  - マルチ戦略シミュレーション（優先度順）
  - 統合されたExcel結果出力
  - 戦略別エントリー/エグジット統計
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from config.risk_management import RiskManagement
from config.optimized_parameters import OptimizedParameterManager
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.Breakout import BreakoutStrategy
from strategies.VWAP_Bounce import VWAPBounceStrategy
from strategies.Opening_Gap import OpeningGapStrategy
from strategies.contrarian_strategy import ContrarianStrategy
from strategies.gc_strategy_signal import GCStrategy
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from data_fetcher import get_parameters_and_data
from output.simulation_handler import simulate_and_save

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\backtest.log")

# リスク管理の初期化
risk_manager = RiskManagement(total_assets=1000000)  # 総資産100万円

# パラメータマネージャーの初期化
param_manager = OptimizedParameterManager()


def load_optimized_parameters(ticker: str) -> Dict[str, Dict[str, Any]]:
    """
    各戦略の承認済み最適化パラメータを読み込みます。
    
    Parameters:
        ticker (str): 銘柄シンボル
        
    Returns:
        Dict[str, Dict[str, Any]]: 戦略名をキーとするパラメータ辞書
    """
    strategies = [
        'VWAPBreakoutStrategy',
        'MomentumInvestingStrategy', 
        'BreakoutStrategy',
        'VWAPBounceStrategy',
        'OpeningGapStrategy',
        'ContrarianStrategy',
        'GCStrategy'
    ]
    
    optimized_params = {}
    
    for strategy_name in strategies:
        try:
            params = param_manager.load_approved_params(strategy_name, ticker)
            if params:
                optimized_params[strategy_name] = params
                logger.info(f"承認済みパラメータを読み込み - {strategy_name}: {params}")
            else:
                logger.warning(f"承認済みパラメータが見つかりません - {strategy_name}")
                # デフォルトパラメータを使用
                optimized_params[strategy_name] = get_default_parameters(strategy_name)
        except Exception as e:
            logger.error(f"パラメータ読み込みエラー - {strategy_name}: {e}")
            # デフォルトパラメータを使用
            optimized_params[strategy_name] = get_default_parameters(strategy_name)
    
    return optimized_params


def get_default_parameters(strategy_name: str) -> Dict[str, Any]:
    """
    戦略のデフォルトパラメータを取得します。
    
    Parameters:
        strategy_name (str): 戦略名
        
    Returns:
        Dict[str, Any]: デフォルトパラメータ
    """
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


def apply_strategies_with_optimized_params(stock_data: pd.DataFrame, index_data: pd.DataFrame, 
                                         optimized_params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    最適化されたパラメータを使用して戦略を適用します。
    
    Parameters:
        stock_data (pd.DataFrame): 株価データ
        index_data (pd.DataFrame): 市場インデックスデータ
        optimized_params (Dict): 戦略別最適化パラメータ
        
    Returns:
        pd.DataFrame: シグナルを追加した株価データ
    """
    logger.info("最適化パラメータを使用した戦略適用を開始")
    
    # 戦略の優先順位（高優先度から）
    strategy_priority = [
        ('VWAPBreakoutStrategy', VWAPBreakoutStrategy),
        ('MomentumInvestingStrategy', MomentumInvestingStrategy),
        ('BreakoutStrategy', BreakoutStrategy),
        ('VWAPBounceStrategy', VWAPBounceStrategy),
        ('OpeningGapStrategy', OpeningGapStrategy),
        ('ContrarianStrategy', ContrarianStrategy),
        ('GCStrategy', GCStrategy)
    ]
    
    # エントリー/エグジット統計
    strategy_stats = {}
    
    # 統合されたシグナル列を初期化
    stock_data['Entry_Signal'] = 0
    stock_data['Exit_Signal'] = 0
    stock_data['Strategy'] = ''
    stock_data['Position_Size'] = 1.0
    
    # 各日付でどの戦略がアクティブかを追跡
    active_positions = {}  # {日付: 戦略名}
    
    for strategy_name, strategy_class in strategy_priority:
        try:
            params = optimized_params.get(strategy_name, {})
            logger.info(f"戦略適用開始: {strategy_name} with params: {params}")
            
            # 戦略インスタンスを作成
            # 全ての戦略は data, params, price_column を受け取る形式に統一
            strategy = strategy_class(
                data=stock_data.copy(),  # コピーを使用して相互影響を避ける
                params=params,
                price_column="Adj Close"
            )
            
            # 戦略を実行してバックテスト結果を取得
            result = strategy.backtest()
            
            # エントリー/エグジット数を統計
            entry_signal_col = 'Entry_Signal'
            exit_signal_col = 'Exit_Signal'
            
            entry_count = 0
            exit_count = 0
            
            if entry_signal_col in result.columns:
                entry_count = (result[entry_signal_col] == 1).sum()
            if exit_signal_col in result.columns:
                exit_count = (result[exit_signal_col] == -1).sum()
            
            # 優先度順にシグナルを統合（既存シグナルがない場合のみ追加）
            for idx in result.index:
                # エントリーシグナルの統合
                if (result.loc[idx, entry_signal_col] == 1 and 
                    stock_data.loc[idx, 'Entry_Signal'] == 0 and
                    idx not in active_positions):
                    
                    stock_data.loc[idx, 'Entry_Signal'] = 1
                    stock_data.loc[idx, 'Strategy'] = strategy_name
                    active_positions[idx] = strategy_name
                
                # エグジットシグナルの統合（同じ戦略からのもののみ）
                if (result.loc[idx, exit_signal_col] == -1 and
                    idx in active_positions and
                    active_positions[idx] == strategy_name):
                    
                    stock_data.loc[idx, 'Exit_Signal'] = -1
                    # 戦略名は既に設定されている
                    # ポジションを削除
                    del active_positions[idx]
            
            strategy_stats[strategy_name] = {
                'entries': int(entry_count),
                'exits': int(exit_count),
                'integrated_entries': int((stock_data['Strategy'] == strategy_name).sum()),
                'integrated_exits': int((stock_data['Exit_Signal'] == -1) & (stock_data['Strategy'] == strategy_name)).sum()
            }
            
            logger.info(f"戦略完了: {strategy_name} - エントリー: {entry_count}, エグジット: {exit_count}")
            logger.info(f"  統合後: エントリー: {strategy_stats[strategy_name]['integrated_entries']}, エグジット: {strategy_stats[strategy_name]['integrated_exits']}")
            
        except Exception as e:
            logger.error(f"戦略適用エラー - {strategy_name}: {e}")
            strategy_stats[strategy_name] = {'entries': 0, 'exits': 0, 'error': str(e)}
    
    # 統計をログ出力
    logger.info("=== 戦略別エントリー/エグジット統計 ===")
    total_entries = 0
    total_exits = 0
    
    for strategy_name, stats in strategy_stats.items():
        if 'error' not in stats:
            logger.info(f"{strategy_name}: エントリー {stats['entries']}, エグジット {stats['exits']}")
            logger.info(f"  統合後: エントリー {stats.get('integrated_entries', 0)}, エグジット {stats.get('integrated_exits', 0)}")
            total_entries += stats.get('integrated_entries', 0)
            total_exits += stats.get('integrated_exits', 0)
        else:
            logger.error(f"{strategy_name}: エラー - {stats['error']}")
    
    logger.info(f"統合後合計: エントリー {total_entries}, エグジット {total_exits}")
    
    return stock_data


def main():
    try:
        logger.info("マルチ戦略バックテストシステムを開始")
        
        # データ取得と前処理
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        
        logger.info(f"データ期間: {start_date} から {end_date}")
        logger.info(f"データ行数: {len(stock_data)}")
        
        # 承認済み最適化パラメータを読み込み
        optimized_params = load_optimized_parameters(ticker)
        logger.info(f"読み込み完了: {len(optimized_params)} 戦略のパラメータ")
        
        # 最適化パラメータを使用して戦略を適用
        if index_data is None:
            # ダミーのindex_dataを作成
            index_data = stock_data[['Close']].copy()
            index_data.columns = ['Close']
        
        stock_data = apply_strategies_with_optimized_params(stock_data, index_data, optimized_params)
        
        # バックテスト結果をExcelに出力（splitsパラメータは除去）
        backtest_results = simulate_and_save(stock_data, ticker)
        
        logger.info(f"バックテスト結果をExcelに出力: {backtest_results}")
        logger.info("マルチ戦略バックテストシステムが正常に完了しました")
        
    except Exception as e:
        logger.exception(f"バックテスト処理中にエラーが発生: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()