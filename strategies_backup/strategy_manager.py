"""
Module: Strategy Manager
File: strategy_manager.py
Description: 
  複数の取引戦略を管理し、適用するためのモジュールです。
  各戦略からのシグナルを集約し、優先順位に基づいて取引シグナルを生成します。

Author: imega
Created: 2025-05-06
"""

import os
import pandas as pd
import logging

# プロジェクトのルートディレクトリを sys.path に追加するため
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from config.risk_management import RiskManagement
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.Breakout import BreakoutStrategy
from strategies.VWAP_Bounce import VWAPBounceStrategy
from strategies.Opening_Gap import OpeningGapStrategy
from strategies.contrarian_strategy import ContrarianStrategy
from strategies.gc_strategy_signal import GCStrategy
from config.error_handling import fetch_stock_data
from config.cache_manager import get_cache_filepath, save_cache

# ロガーの設定
logger = logging.getLogger(__name__)

# リスク管理の初期化（ここでも初期化するが、main.pyからも参照されるかもしれない）
risk_manager = RiskManagement(total_assets=1000000)

def apply_strategies(stock_data: pd.DataFrame, index_data: pd.DataFrame = None):
    """
    複数の戦略を適用し、シグナルを生成します。
    
    ⚠️ 警告: この関数はレガシーです。
    新しいmain.pyでは承認済み最適化パラメータを使用した
    apply_strategies_with_optimized_params() を使用してください。
    
    Parameters:
        stock_data (pd.DataFrame): 株価データ
        index_data (pd.DataFrame, optional): 市場インデックスデータ
        
    Returns:
        pd.DataFrame: シグナルを追加した株価データ
    """
    logger.warning("⚠️ apply_strategies() はレガシー関数です。新しいmain.pyを使用してください。")
    
    # VWAPBreakoutStrategy は index_data が必要なので、なければダミーデータを作成
    if index_data is None:
        logger.warning("インデックスデータがないため、VWAPBreakoutStrategyにはダミーデータを使用します。")
        index_data = stock_data.copy()  # 最低限のデータとして、同じデータを使用
    
    # DOWデータの取得（Opening Gap戦略用）
    try:
        start_date = stock_data.index[0].strftime('%Y-%m-%d')
        end_date = stock_data.index[-1].strftime('%Y-%m-%d')
        dow_cache_filepath = get_cache_filepath("^DJI", start_date, end_date)
        
        if os.path.exists(dow_cache_filepath):
            logger.info("キャッシュからDOWデータを読み込みます")
            dow_data = pd.read_pickle(dow_cache_filepath)
        else:
            logger.info("DOWデータを取得します")
            dow_data = fetch_stock_data("^DJI", start_date, end_date)
            save_cache(dow_data, "^DJI", start_date, end_date)
    except Exception as e:
        logger.error(f"DOWデータの取得に失敗しました: {str(e)}")
        dow_data = pd.DataFrame(index=stock_data.index)  # 空のデータで初期化
    
    # Entry_Signal と Exit_Signal カラムを追加（初期値は0）
    if 'Entry_Signal' not in stock_data.columns:
        stock_data['Entry_Signal'] = 0
    if 'Exit_Signal' not in stock_data.columns:
        stock_data['Exit_Signal'] = 0
    if 'Strategy' not in stock_data.columns:
        stock_data['Strategy'] = ""
    if 'Position_Size' not in stock_data.columns:
        stock_data['Position_Size'] = 0.0
    
    # 全ての戦略を適用
    strategies = {
        "VWAP Breakout": VWAPBreakoutStrategy(stock_data, index_data),
        "Momentum Investing": MomentumInvestingStrategy(stock_data),
        "Breakout": BreakoutStrategy(stock_data),
        "VWAP Bounce": VWAPBounceStrategy(stock_data),
        "Opening Gap": OpeningGapStrategy(stock_data, dow_data),
        "Contrarian": ContrarianStrategy(stock_data),
        "GC Strategy": GCStrategy(stock_data)
    }

    # シグナル統計用の辞書
    signal_stats = {strat_name: 0 for strat_name in strategies.keys()}
    
    # バックテスト用のリスク管理状態を初期化
    risk_mgr_state = {}  # {日付: {戦略名: ポジションサイズ}} の形式で保存
    current_positions = {}  # 現在保有中のポジション {戦略名: [エントリー日, エントリー価格]}

    # 各日を順にシミュレーション
    for idx in range(len(stock_data)):
        current_date = stock_data.index[idx]
        
        # まず全ての戦略のイグジットシグナルを確認
        for strategy_name, strategy in strategies.items():
            # 現在このストラテジーでポジションを持っている場合のみイグジットを確認
            if strategy_name in current_positions:
                exit_signal = strategy.generate_exit_signal(idx)
                if exit_signal == -1:
                    # イグジットが出たら記録
                    stock_data.at[current_date, 'Exit_Signal'] = -1
                    stock_data.at[current_date, 'Strategy'] = strategy_name
                    signal_stats[strategy_name] += 1
                    # ポジション解除
                    del current_positions[strategy_name]
                    logger.debug(f"{current_date}: {strategy_name} のイグジットシグナル")
        
        # 次に、ポジションがない場合のみエントリーシグナルを確認
        if not current_positions:  # 全戦略でポジションを持っていない場合のみ
            for strategy_name, strategy in strategies.items():
                entry_signal = strategy.generate_entry_signal(idx)
                if entry_signal == 1:
                    # エントリーシグナルが出たら記録
                    stock_data.at[current_date, 'Entry_Signal'] = 1
                    stock_data.at[current_date, 'Strategy'] = strategy_name
                    stock_data.at[current_date, 'Position_Size'] = 1.0
                    signal_stats[strategy_name] += 1
                    # ポジション保有
                    current_positions[strategy_name] = [current_date, stock_data[strategy.price_column].iloc[idx]]
                    logger.debug(f"{current_date}: {strategy_name} のエントリーシグナル")
                    break  # 一つの戦略でエントリーしたら他はチェックしない
    
    # ログ出力
    entry_count = stock_data['Entry_Signal'].sum()
    exit_count = (stock_data['Exit_Signal'] == -1).sum()
    logger.info(f"バックテスト期間中のエントリー回数: {entry_count}回")
    logger.info(f"バックテスト期間中のイグジット回数: {exit_count}回")
    
    return stock_data