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
    
    Parameters:
        stock_data (pd.DataFrame): 株価データ
        index_data (pd.DataFrame, optional): 市場インデックスデータ
        
    Returns:
        pd.DataFrame: シグナルを追加した株価データ
    """
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
            logger.info("DOWデータのキャッシュを使用します")
            dow_data = pd.read_csv(dow_cache_filepath, index_col=0, parse_dates=True)
        else:
            logger.info("DOWデータを取得します")
            dow_data = fetch_stock_data("^DJI", start_date, end_date)
            save_cache(dow_data, dow_cache_filepath)
    except Exception as e:
        logger.error(f"DOWデータの取得に失敗しました: {str(e)}")
        dow_data = stock_data.copy()  # 代替として株価データを使用
    
    # 全ての戦略を適用
    strategies = {
        "VWAP Breakout.py": VWAPBreakoutStrategy(stock_data, index_data),
        "Momentum Investing.py": MomentumInvestingStrategy(stock_data),
        "Breakout.py": BreakoutStrategy(stock_data),
        # 追加の戦略
        "VWAP Bounce.py": VWAPBounceStrategy(stock_data),
        "Opening Gap.py": OpeningGapStrategy(stock_data, dow_data),
        "Contrarian Strategy.py": ContrarianStrategy(stock_data),
        # GCStrategyを追加
        "GC Strategy.py": GCStrategy(stock_data)
    }

    # Entry_Signal と Exit_Signal カラムを追加（初期値は0）
    if 'Entry_Signal' not in stock_data.columns:
        stock_data['Entry_Signal'] = 0
    if 'Exit_Signal' not in stock_data.columns:
        stock_data['Exit_Signal'] = 0
    if 'Strategy' not in stock_data.columns:
        stock_data['Strategy'] = ""
    if 'Position_Size' not in stock_data.columns:
        stock_data['Position_Size'] = 0
    
    # 戦略シグナル分析用の列を追加
    for strat_name in strategies.keys():
        col_name = f"Signal_{strat_name.replace('.py', '').replace(' ', '_')}"
        stock_data[col_name] = 0
    
    # シグナル統計用の辞書
    signal_stats = {strat_name: 0 for strat_name in strategies.keys()}
    
    # バックテストのために、すべての日付に対してシグナルを生成
    logger.info("バックテスト用にすべての日付に対してシグナルを生成します")
    
    # ウォームアップ期間（最初の30日間はインジケーターが安定しないので除外）
    warmup_period = 30
    
    # バックテスト用のリスク管理状態を初期化
    risk_mgr_state = {}  # {日付: {戦略名: ポジションサイズ}} の形式で保存
    current_positions = {}  # 現在保有中のポジション {戦略名: [エントリー日, エントリー価格]}
    
    # シグナル分析用ログファイル
    signal_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'signal_analysis.log')
    signal_logger = setup_logger('signal_analysis', log_file=signal_log_path, level=logging.INFO)
    signal_logger.info(f"日付,{'、'.join(strategies.keys())},選択された戦略")
    
    for i in range(len(stock_data)):
        current_date = pd.to_datetime(stock_data.index[i])
        
        # このデータポイントまでのエントリーシグナルを確認
        entries_to_current = stock_data.iloc[:i+1]
        entry_indices = entries_to_current[entries_to_current['Entry_Signal'] > 0].index
        
        if len(entry_indices) > 0:
            # 最新のエントリー日を取得
            latest_entry = entry_indices[-1]
            entry_idx = stock_data.index.get_loc(latest_entry)
            entry_date = pd.to_datetime(latest_entry)
            
            # 保有日数計算
            holding_days = (current_date - entry_date).days
            
            # 以降の処理...
        else:
            # エントリーがない場合
            holding_days = 0
            # 以降の処理...
        
        # 前日のリスク管理状態をコピー
        if i > warmup_period:
            prev_date = stock_data.index[i-1]
            if prev_date in risk_mgr_state:
                # 前日の状態を今日の初期状態として設定
                risk_mgr_state[current_date] = risk_mgr_state[prev_date].copy()
            else:
                risk_mgr_state[current_date] = {}
        else:
            risk_mgr_state[current_date] = {}
        
        # 現在日のリスク管理状態を取得
        current_risk_state = risk_mgr_state[current_date]
        
        # 現在のポジション数をリスク管理システムに反映
        risk_manager.active_trades = current_risk_state
        
        # その日付までのデータで各戦略のシグナルを生成
        signals = {}
        
        for strategy_name, strategy in strategies.items():
            # 適切なインデックスで各戦略のシグナルを生成
            entry_signal = strategy.generate_entry_signal(idx=i)
            signals[strategy_name] = entry_signal
            
            # シグナルをデータフレームに記録
            col_name = f"Signal_{strategy_name.replace('.py', '').replace(' ', '_')}"
            stock_data.at[current_date, col_name] = entry_signal
            
            # シグナルが1の場合、統計を更新
            if entry_signal == 1:
                signal_stats[strategy_name] += 1
        
        # 戦略の優先順位設定を削除したため、シグナルの強さに基づいてソート（1のシグナルを優先）
        # 同じシグナル強度の場合は、ランダム性を与えるためにlistを使ってシャッフル
        prioritized_strategies = sorted(signals.keys(), key=lambda x: (-signals[x], hash(x) % 100))
        
        # 信号分析用ログ出力
        signal_values = [str(signals[s]) for s in strategies.keys()]
        selected_strategy = "なし"
        
        # 各戦略のシグナルに基づいてポジション管理
        for strategy_name in prioritized_strategies:
            entry_signal = signals[strategy_name]
            
            # エントリーシグナルが発生した場合
            if entry_signal == 1:
                # リスク管理: ポジションサイズを確認
                if risk_manager.check_position_size(strategy_name):
                    stock_data.at[current_date, 'Entry_Signal'] = 1
                    stock_data.at[current_date, 'Strategy'] = strategy_name
                    selected_strategy = strategy_name
                    
                    # ポジションサイズを1に設定（1単元の取引）
                    position_size = 1
                    stock_data.at[current_date, 'Position_Size'] = position_size
                    
                    # リスク管理システムの状態を更新
                    risk_manager.update_position(strategy_name, position_size)
                    current_risk_state[strategy_name] = current_risk_state.get(strategy_name, 0) + position_size
                    
                    # 現在のポジション情報を記録
                    entry_price = stock_data.at[current_date, 'Close'] if 'Close' in stock_data.columns else stock_data.at[current_date, 'Adj Close']
                    current_positions[strategy_name] = [current_date, entry_price]
                    
                    logger.debug(f"{current_date}: {strategy_name} からのエントリーシグナル - ポジションサイズ: {position_size}")
                    
                    # 選択理由のログ
                    competing_signals = [s for s, v in signals.items() if v == 1]
                    if len(competing_signals) > 1:
                        logger.info(f"{current_date}: 複数の戦略が同時にシグナルを出しました: {competing_signals}, 選択された戦略: {strategy_name}")
                    break  # 1つの戦略が選ばれたら終了
                else:
                    logger.debug(f"{current_date}: {strategy_name} のシグナルは検出されましたが、ポジションサイズの制限によりスキップされました")
        
        # シグナル分析ログに記録
        signal_logger.info(f"{current_date},{','.join(signal_values)},{selected_strategy}")
        
        # Exit シグナルの生成と処理
        for strategy_name, position_info in list(current_positions.items()):
            entry_date, entry_price = position_info
            
            # エントリーインデックスを取得する部分を確認
            entry_indices = stock_data.index[stock_data['Entry_Signal'] > 0].tolist()
            if entry_indices:
                entry_idx = stock_data.index.get_loc(entry_indices[-1])  # 最新のエントリーインデックス
                entry_date = pd.to_datetime(stock_data.index[entry_idx])
                
                # 保有日数計算
                current_date = pd.to_datetime(stock_data.index[i])
                holding_days = (current_date - entry_date).days
            else:
                # エントリーがない場合の処理
                holding_days = 0
            
            # 1. 保有期間による決済（3日間保持後）
            if holding_days >= 3:
                stock_data.at[current_date, 'Exit_Signal'] = -1
                # ポジションクローズ
                if strategy_name in current_risk_state:
                    del current_risk_state[strategy_name]
                del current_positions[strategy_name]
                logger.debug(f"{current_date}: {strategy_name} の保有期間（3日）経過によるイグジットシグナル")
                continue
            
            # 2. 損切りロジック - 2%以上の下落
            current_price = stock_data.at[current_date, 'Close'] if 'Close' in stock_data.columns else stock_data.at[current_date, 'Adj Close']
            price_change = (current_price - entry_price) / entry_price
            
            if price_change < -0.02:
                stock_data.at[current_date, 'Exit_Signal'] = -1
                # ポジションクローズ
                if strategy_name in current_risk_state:
                    del current_risk_state[strategy_name]
                del current_positions[strategy_name]
                logger.debug(f"{current_date}: {strategy_name} の2%以上の下落によるイグジットシグナル（損切り）")
                continue
    
    # ログ出力
    entry_count = stock_data['Entry_Signal'].sum()
    exit_count = (stock_data['Exit_Signal'] == -1).sum()
    logger.info(f"バックテスト期間中の総エントリー回数: {entry_count}回")
    logger.info(f"バックテスト期間中の総イグジット回数: {exit_count}回")
    
    # 戦略ごとのシグナル統計
    logger.info("各戦略のシグナル生成回数:")
    for strategy_name, count in signal_stats.items():
        logger.info(f"  {strategy_name}: {count}回")
    
    # 実際に選択された戦略の分布
    strategy_dist = stock_data[stock_data['Entry_Signal'] == 1]['Strategy'].value_counts()
    logger.info("実際に選択された戦略の分布:")
    for strategy_name, count in strategy_dist.items():
        logger.info(f"  {strategy_name}: {count}回 ({count/entry_count*100:.1f}%)")
    
    # 戦略間の競合状況分析
    logger.info("戦略間の競合状況分析:")
    conflict_days = 0
    for i in range(warmup_period, len(stock_data)):
        current_date = stock_data.index[i]
        competing_signals = sum(1 for col in stock_data.columns if col.startswith('Signal_') and stock_data.at[current_date, col] == 1)
        if competing_signals > 1:
            conflict_days += 1
    
    if entry_count > 0:
        logger.info(f"  複数戦略が同時にシグナルを出した日数: {conflict_days}日")
        logger.info(f"  選択された戦略が他の戦略より優先された比率: {conflict_days/entry_count*100:.1f}%")
    
    # リスク管理情報をメタデータとして追加
    stock_data.attrs['risk_management'] = risk_mgr_state
    stock_data.attrs['signal_stats'] = signal_stats
    
    return stock_data