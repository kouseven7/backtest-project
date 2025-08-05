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
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config.logger_config import setup_logger
from src.config.risk_management import RiskManagement
from src.config.optimized_parameters import OptimizedParameterManager

# ロガーの設定
logger = setup_logger(__name__, log_file=os.path.join(project_root, "logs", "backtest.log"))

# 新統合システムのインポート
try:
    from config.multi_strategy_manager import MultiStrategyManager, ExecutionMode
    from config.strategy_execution_adapter import StrategyExecutionAdapter
    integrated_system_available = True
    logger.info("統合マルチ戦略システムが利用可能です")
except ImportError as e:
    integrated_system_available = False
    logger.warning(f"統合システムが利用できません: {e}。従来システムを使用します。")
from src.indicators.unified_trend_detector import detect_unified_trend, detect_unified_trend_with_confidence
from src.strategies.VWAP_Breakout import VWAPBreakoutStrategy
from src.strategies.Momentum_Investing import MomentumInvestingStrategy
from src.strategies.Breakout import BreakoutStrategy
from src.strategies.VWAP_Bounce import VWAPBounceStrategy
from src.strategies.Opening_Gap import OpeningGapStrategy
from src.strategies.contrarian_strategy import ContrarianStrategy
from src.strategies.gc_strategy_signal import GCStrategy
from data_processor import preprocess_data
from src.indicators.indicator_calculator import compute_indicators
from data_fetcher import get_parameters_and_data
from output.simple_simulation_handler import simulate_and_save

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


def check_for_series_signal(signal) -> int:
    """シグナルがSeriesの場合の処理を行うヘルパー関数"""
    if isinstance(signal, pd.Series):
        # Seriesの場合、最初の値を取り出す
        if signal.empty:
            return 0  # 空のSeriesの場合は0を返す
        first_val = signal.iloc[0]
        return 1 if first_val == 1 else (-1 if first_val == -1 else 0)
    # NaNの場合は0を返す
    try:
        if pd.isna(signal):
            return 0
    except:
        pass
    # 通常の場合はint型に変換
    try:
        return int(signal)
    except:
        return 0


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
    stock_data['Position'] = 0  # ポジション状態を追跡する列を追加
    
    # 各日付でどの戦略がアクティブかを追跡
    active_positions = {}  # {日付: 戦略名}
    
    for strategy_name, strategy_class in strategy_priority:
        try:
            params = optimized_params.get(strategy_name, {})
            logger.info(f"戦略適用開始: {strategy_name} with params: {params}")
            
            # 戦略ごとに必要なパラメータを渡す
            if strategy_name == 'VWAPBreakoutStrategy':
                try:
                    strategy = strategy_class(
                        data=stock_data.copy(),  # コピーを使用して相互影響を避ける
                        index_data=index_data,  # index_dataを最初の引数に移動
                        params=params,
                        price_column="Adj Close"
                    )
                except Exception as e:
                    logger.error(f"VWAPBreakoutStrategy初期化エラー: {e}")
                    continue
            elif strategy_name == 'OpeningGapStrategy':
                strategy = strategy_class(
                    data=stock_data.copy(),
                    params=params,
                    price_column="Adj Close",
                    dow_data=index_data  # OpeningGapStrategyはdow_dataが必要
                )
            else:
                # その他の戦略は共通パラメータで初期化
                strategy = strategy_class(
                    data=stock_data.copy(),
                    params=params,
                    price_column="Adj Close"
                )
            
            # 戦略を実行してバックテスト結果を取得
            try:
                result = strategy.backtest()
                
                # インデックスをstock_dataと同じ型に変換して整合性を確保
                if not isinstance(result.index, type(stock_data.index)):
                    logger.warning(f"インデックスタイプ不一致: result={type(result.index)}, stock_data={type(stock_data.index)}. 変換を試みます。")
                    try:
                        result.index = pd.DatetimeIndex(result.index)
                    except Exception as e:
                        logger.error(f"インデックス変換エラー: {e}")
            except Exception as e:
                logger.error(f"バックテストエラー: {e}")
                result = pd.DataFrame(index=stock_data.index)  # 空のデータフレームを返す
            
            # エントリー/エグジット数を統計
            entry_signal_col = 'Entry_Signal'
            exit_signal_col = 'Exit_Signal'
            
            entry_count = 0
            exit_count = 0
            
            if entry_signal_col in result.columns:
                # Series型を安全に扱うための変換
                try:
                    if not pd.api.types.is_integer_dtype(result[entry_signal_col]):
                        # 0以外の値を持つエントリーをカウント
                        entry_signals = result[entry_signal_col].fillna(0)
                        entry_count = (entry_signals == 1).sum()
                    else:
                        entry_count = (result[entry_signal_col] == 1).sum()
                except Exception as e:
                    logger.error(f"エントリーカウントエラー: {e}")
                    entry_count = 0
                    
            if exit_signal_col in result.columns:
                # Series型を安全に扱うための変換
                try:
                    if not pd.api.types.is_integer_dtype(result[exit_signal_col]):
                        # -1の値を持つエグジットをカウント
                        exit_signals = result[exit_signal_col].fillna(0)
                        exit_count = (exit_signals == -1).sum()
                    else:
                        exit_count = (result[exit_signal_col] == -1).sum()
                except Exception as e:
                    logger.error(f"エグジットカウントエラー: {e}")
                    exit_count = 0
            
            # 優先度順にシグナルを統合（既存シグナルがない場合のみ追加）
            # エントリーとエグジットのシグナル処理を単純化
            # データフレームのチェック
            if not isinstance(result, pd.DataFrame):
                logger.warning(f"{strategy_name}: 結果がDataFrameではありません")
                continue
                
            if result.empty:
                logger.warning(f"{strategy_name}: 結果データフレームが空です")
                continue
                
            # 安全でシンプルな方法でシグナルを取得
            entry_dates = []
            exit_dates = []
            
            # columnsチェック
            if not isinstance(result, pd.DataFrame) or not hasattr(result, 'columns'):
                logger.error(f"{strategy_name}: データフレームが不正です")
                continue
            
            # エントリーシグナルを収集
            try:
                if entry_signal_col in result.columns:
                    # エントリー日を取得
                    entry_mask = result[entry_signal_col] == 1
                    entry_dates = result[entry_mask].index.tolist()
            except Exception as e:
                logger.error(f"{strategy_name} エントリーシグナル抽出エラー: {e}")
                entry_dates = []
                
            # エグジットシグナルを収集
            try:
                if exit_signal_col in result.columns:
                    # エグジット日を取得
                    exit_mask = result[exit_signal_col] == -1
                    exit_dates = result[exit_mask].index.tolist()
            except Exception as e:
                logger.error(f"{strategy_name} エグジットシグナル抽出エラー: {e}")
                exit_dates = []
            
            # シグナル統合
            for date in entry_dates:
                try:
                    if date in stock_data.index:
                        # リスク管理モジュールを使用してポジション制限をチェック
                        # 短期取引を促進するため、リスク管理を緩和
                        if risk_manager.check_position_size(strategy_name) and date not in active_positions:
                            # エントリーシグナルを追加
                            stock_data.loc[date, 'Entry_Signal'] = 1
                            stock_data.loc[date, 'Strategy'] = strategy_name
                            stock_data.loc[date, 'Position'] = 1  # ポジション状態を更新
                            active_positions[date] = strategy_name
                            # リスク管理モジュールのポジション情報も更新
                            risk_manager.update_position(strategy_name, 1)
                            logger.info(f"戦略統合: {strategy_name} エントリー: 日付={date}, 全ポジション数={risk_manager.get_total_positions()}")
                except Exception as e:
                    logger.error(f"エントリー統合エラー ({strategy_name}, 日付={date}): {e}")
                    
            for date in exit_dates:
                try:
                    if date in stock_data.index:
                        # このシグナルに対応するエントリー日を探す（同じ戦略からのもの）
                        matching_entries = [d for d, s in active_positions.items() 
                                          if s == strategy_name and d < date]
                        if matching_entries:
                            # 最も古いエントリーに対してイグジット（FIFO方式）
                            oldest_entry = sorted(matching_entries)[0]
                            stock_data.loc[date, 'Exit_Signal'] = -1
                            stock_data.loc[date, 'Position'] = 0  # ポジション状態を更新
                            logger.info(f"戦略統合: {strategy_name} イグジット: 日付={date}, エントリー日={oldest_entry}")
                            
                            # ポジションを削除
                            del active_positions[oldest_entry]
                            
                            # リスク管理モジュールのポジション情報も更新
                            risk_manager.update_position(strategy_name, -1)
                except Exception as e:
                    logger.error(f"エグジット統合エラー ({strategy_name}, 日付={date}): {e}")
                
# このブロック全体を削除（重複しているため）
# 代わりに、前のブロックに統合された修正が使用されます
            
            # 正確なカウントのために数値を変換
            # より安全な方法でカウント
            integrated_entries = sum(1 for _ in stock_data[stock_data['Strategy'] == strategy_name].index)
            integrated_exits = sum(1 for _ in stock_data[(stock_data['Exit_Signal'] == -1) & (stock_data['Strategy'] == strategy_name)].index)
            
            strategy_stats[strategy_name] = {
                'entries': int(entry_count),
                'exits': int(exit_count),
                'integrated_entries': integrated_entries,
                'integrated_exits': integrated_exits
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
    
    # 未処理のオープンポジションを最終日でクローズ
    # これにより、エントリーとエグジット回数の不均衡を修正
    if active_positions:
        logger.warning(f"バックテスト終了時に {len(active_positions)} 件の未決済ポジションがあります。最終日に強制決済します。")
        last_date = stock_data.index[-1]
        
        # 未決済ポジションを戦略ごとに集計
        positions_by_strategy = {}
        for entry_date, strategy_name in active_positions.items():
            if strategy_name not in positions_by_strategy:
                positions_by_strategy[strategy_name] = []
            positions_by_strategy[strategy_name].append(entry_date)
        
        # 戦略ごとの未決済ポジション数をログ出力
        for strategy_name, entry_dates in positions_by_strategy.items():
            logger.warning(f"戦略 {strategy_name} に {len(entry_dates)} 件の未決済ポジションがあります")
        
        # 未決済ポジションをすべて最終日で決済
        for entry_idx, strategy_name in active_positions.items():
            # 最終日に強制決済
            stock_data.loc[last_date, 'Exit_Signal'] = -1
            stock_data.loc[last_date, 'Position'] = 0  # ポジション状態を更新
            logger.info(f"強制決済: 戦略={strategy_name}, エントリー日={entry_idx}, 決済日={last_date}")
    
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
        
        # 統合システム利用可能性をローカル変数として設定
        use_integrated_system = integrated_system_available
        
        # 統合システム利用可能性をチェック
        if use_integrated_system:
            try:
                logger.info("統合マルチ戦略システムを使用してバックテストを実行します")
                
                # MultiStrategyManager を初期化
                manager = MultiStrategyManager()
                
                # 戦略実行アダプターを設定（必要に応じて使用）
                # adapter = StrategyExecutionAdapter()
                
                # システム初期化
                if manager.initialize_system():
                    logger.info("統合システムの初期化に成功しました")
                    
                    # マルチ戦略実行
                    available_strategies = list(optimized_params.keys())
                    results = manager.execute_multi_strategy_flow(stock_data, available_strategies)
                    
                    if results:
                        logger.info("統合システムでのバックテスト実行が完了しました")
                        # MultiStrategyResultから結果データを取得
                        result_data = results.combined_signals if hasattr(results, 'combined_signals') else stock_data
                        backtest_results = simulate_and_save(result_data, ticker)
                    else:
                        logger.warning("統合システムの実行結果が空でした。従来システムにフォールバックします。")
                        raise Exception("統合システムの実行に失敗")
                else:
                    logger.warning("統合システムの初期化に失敗しました。従来システムにフォールバックします。")
                    raise Exception("統合システムの初期化失敗")
                    
            except Exception as e:
                logger.error(f"統合システム実行中にエラー: {e}")
                logger.info("従来のマルチ戦略システムにフォールバックします")
                use_integrated_system = False
        
        if not use_integrated_system:
            logger.info("従来のマルチ戦略システムを使用してバックテストを実行します")
            
            # index_dataがNoneの場合は、同じ期間の日経平均などのインデックスを取得するか、
            # ダミーのindex_dataを作成する
            if index_data is None or index_data.empty:
                logger.warning("市場インデックスデータが取得できませんでした。ダミーデータを作成します。")
                # ダミーのindex_dataを作成（stock_dataと同じインデックスを使用）
                index_data = pd.DataFrame(index=stock_data.index)
                
                # 必要な列を追加
                for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                    if col in stock_data.columns:
                        index_data[col] = stock_data[col] * 0.9  # 適当な値を設定
                
                # データの完全性を確保
                index_data = index_data.fillna(method='ffill').fillna(method='bfill')
            
            # 最適化パラメータを使用して戦略を適用
            stock_data = apply_strategies_with_optimized_params(stock_data, index_data, optimized_params)
            
        # バックテスト結果をExcelに出力（新Excel出力モジュール使用）
        backtest_results = simulate_and_save(stock_data, ticker)
        logger.info(f"改良版Excel出力: {backtest_results}")
        
        logger.info("マルチ戦略バックテストシステムが正常に完了しました")
        
    except Exception as e:
        logger.exception(f"バックテスト処理中にエラーが発生: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()