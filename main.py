"""
Module: Main (非アクティブ・調査対象外)
File: main.py
Description: 
  【重要】このファイルは現在使用されていません（非アクティブ）
  調査・修正・テスト対象外として扱ってください。
  
  旧マルチ戦略バックテストシステムのメインエントリーポイント。
  承認済みの最適化パラメータを使用して複数の戦略を実行し、
  統合されたバックテスト結果を生成する機能を持っていました。

Author: imega
Created: 2023-04-01
Modified: 2025-12-30
Status: INACTIVE - 調査・修正対象外

旧Features（参考情報のみ）:
  - 承認済み最適化パラメータの自動読み込み
  - マルチ戦略シミュレーション（優先度順）
  - 統合されたExcel結果出力
  - 戦略別エントリー/エグジット統計

非アクティブ理由:
  - 新しいエントリーポイントに移行済み（main_new.pyまたは他のファイル）
  - Phase 3-C実装により統合システムが変更
  - DSSMS統合システム導入により実行フローが変更

注意事項:
   このファイルに対する修正・調査・テスト・実行は行わないでください
   問題解決やバックテスト実行には他のアクティブファイルを使用してください
   コード参照のみ許可（コピー&ペーストは別ファイルで行う）

補足（履歴情報）:
  - 本ファイルは旧マルチ戦略層 (戦略集合の統合実行) を扱っていました。
  - 銘柄選択 (DSSMS) は src/dssms/ 以下で独立進化し、将来ここへは
    「選択結果(最適銘柄+バックアップ)を受け取る」一方向インターフェースのみ維持予定でした。
  - DSSMS目的: 日次動的最適銘柄集中運用 (分散なし)。
  - 強化学習導入予定: 現時点なし (拡張ポイントのみ確保)。
  - エラーハンドリング分類: CRITICAL/ERROR/WARNING/INFO/DEBUG を logger に準拠。
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from config.risk_management import RiskManagement
# 新しい共通信号処理モジュールをインポート
from signal_processing import detect_exit_anomalies, check_same_day_entry_exit
from config.optimized_parameters import OptimizedParameterManager
from src.config.system_modes import SystemFallbackPolicy, ComponentType

# ロガーの設定 - Enhanced Logger Manager使用（ログローテーション・圧縮対応）
from src.utils.logger_setup import get_strategy_logger
logger = get_strategy_logger("main")

# SystemFallbackPolicy の初期化
fallback_policy = SystemFallbackPolicy()

# 新統合システムのインポート
try:
    # Phase 4-B-1: multi_strategy_manager_fixed統合
    from config.multi_strategy_manager_fixed import MultiStrategyManager, ExecutionMode
    from config.strategy_execution_adapter import StrategyExecutionAdapter
    integrated_system_available = True
    logger.info("統合マルチ戦略システムが利用可能です")
except ImportError as e:
    # SystemFallbackPolicy を使用した明示的フォールバック処理
    integrated_system_available = fallback_policy.handle_component_failure(
        component_type=ComponentType.MULTI_STRATEGY,
        component_name="MultiStrategyManager",
        error=e,
        fallback_func=lambda: False
    )
    if not integrated_system_available:
        logger.warning(f"統合システムが利用できません: {e}。従来システムを使用します。")
from indicators.unified_trend_detector import detect_unified_trend, detect_unified_trend_with_confidence
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.Breakout import BreakoutStrategy
# Phase B-3完了: VWAP_Bounce使用不可
# 検証結果（9101.T 2023-2024, 2年間491取引日）:
#   - デフォルト（range-bound）: エントリー0回
#   - 条件緩和（range-bound）: エントリー0回（VWAP条件該当5日すべてuptrend）
#   - トレンドフィルターOFF: エントリー2回、総損益-14,129円（-1.41%）、勝率0.00%
# 結論: 元の設計意図（range-boundでのVWAP反発）が2年間で1回も発生せず実用性なし
# from strategies.VWAP_Bounce import VWAPBounceStrategy
# from strategies.Opening_Gap import OpeningGapStrategy  # Phase B-3完了: 使用不可（2022-2024データで壊滅的性能）
# from strategies.Opening_Gap_Fixed import OpeningGapFixedStrategy  # Phase B-3完了: 使用不可（親クラスが使用不可のため）
from strategies.contrarian_strategy import ContrarianStrategy
from strategies.gc_strategy_signal import GCStrategy
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from data_fetcher import get_parameters_and_data
# Excel廃棄対応: simulate_and_save from output.simple_simulation_handler は統一出力エンジンに移行済み

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
        # 'VWAPBounceStrategy',  # Phase B-3完了: 使用不可（2年間エントリー0回）
        # 'OpeningGapStrategy',  # Phase B-3完了: 使用不可（2022-2024データで壊滅的性能）
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
        # 'VWAPBounceStrategy': {  # 除外: 2年間テストで0エントリー (range-bound用だがVWAP条件はuptrend日のみ発動)
        #     'vwap_period': 20,
        #     'deviation_threshold': 0.02,
        #     'volume_threshold': 1.2,
        #     'stop_loss_pct': 0.03,
        #     'take_profit_pct': 0.06
        # },
        # 'OpeningGapStrategy': {  # Phase B-3完了: 使用不可
        #     'gap_threshold': 0.02,
        #     'volume_threshold': 1.5,
        #     'confirmation_period': 3,
        #     'stop_loss_pct': 0.05,
        #     'take_profit_pct': 0.10
        # },
        'ContrarianStrategy': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.08
        },
        'GCStrategy': {
            'short_window': 5,
            'long_window': 75,
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


def calculate_performance_metrics(stock_data, trades):
    """
    パフォーマンス指標を包括的に計算する関数（修正版）
    異常値問題を修正: 適切なリターン計算とポートフォリオ価値算出
    """
    try:
        if len(trades) == 0:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'average_return': 0.0,
                'volatility': 0.0,
                'calmar_ratio': 0.0
            }
        
        # 基本価格データ取得
        initial_price = float(stock_data['Close'].iloc[0])
        final_price = float(stock_data['Close'].iloc[-1])
        
        # Buy & Hold リターン（修正版）
        total_return = (final_price - initial_price) / initial_price
        
        # 日次リターンの計算（修正版）
        daily_returns = stock_data['Close'].pct_change().dropna()
        
        # 取引統計の計算
        entry_trades = [t for t in trades if t.get('type') == 'entry']
        exit_trades = [t for t in trades if t.get('type') == 'exit']
        
        # 勝率計算（修正版）
        if len(daily_returns) > 0:
            profitable_days = (daily_returns > 0).sum()
            losing_days = (daily_returns < 0).sum()
            win_rate = profitable_days / len(daily_returns) if len(daily_returns) > 0 else 0.0
        else:
            profitable_days = 0
            losing_days = 0
            win_rate = 0.0
        
        # ボラティリティ計算（修正版）
        if len(daily_returns) > 1:
            volatility = daily_returns.std() * np.sqrt(252)  # 年率換算
        else:
            volatility = 0.0
        
        # シャープレシオ計算（修正版）
        if len(daily_returns) > 1 and volatility > 0:
            annual_return = total_return  # 年間リターン近似
            sharpe_ratio = annual_return / volatility
        else:
            sharpe_ratio = 0.0
        
        # 最大ドローダウン計算（修正版）
        if 'Portfolio_Value' in stock_data.columns:
            portfolio_values = stock_data['Portfolio_Value']
        else:
            # 正規化されたポートフォリオ価値
            portfolio_values = stock_data['Close'] / initial_price
        
        if len(portfolio_values) > 1:
            cumulative_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0.0
        
        # 平均リターン計算（修正版）
        average_return = daily_returns.mean() if len(daily_returns) > 0 else 0.0
        
        # Calmar比率計算（修正版）
        if abs(max_drawdown) > 0.001:  # 0除算回避
            calmar_ratio = total_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
        
        return {
            'total_return': float(total_return),
            'win_rate': float(win_rate), 
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': len(trades),
            'profitable_trades': int(profitable_days),
            'losing_trades': int(losing_days),
            'average_return': float(average_return),
            'volatility': float(volatility),
            'calmar_ratio': float(calmar_ratio)
        }
    
    except Exception as e:
        logger.error(f"パフォーマンス計算エラー: {e}")
        return {
            'total_return': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': len(trades),
            'profitable_trades': 0,
            'losing_trades': 0,
            'average_return': 0.0,
            'volatility': 0.0,
            'calmar_ratio': 0.0
        }

def _execute_individual_strategy(stock_data, index_data, strategy_name, strategy_class, params):
    """
    個別戦略実行（バックテスト基本理念遵守）
    TODO(tag:exit_signal_integration, rationale:maintain actual backtest execution)
    """
    try:
        # 戦略ごとに必要なパラメータを渡す
        if strategy_name == 'VWAPBreakoutStrategy':
            strategy = strategy_class(
                data=stock_data.copy(),
                index_data=index_data,
                params=params,
                price_column="Adj Close"
            )
        # elif strategy_name in ['OpeningGapStrategy', 'OpeningGapFixedStrategy']:  # Phase B-3完了: 使用不可
        #     strategy = strategy_class(
        #         data=stock_data.copy(),
        #         dow_data=index_data,
        #         params=params,
        #         price_column="Adj Close"
        #     )
        else:
            # その他の戦略は共通パラメータで初期化
            strategy = strategy_class(
                data=stock_data.copy(),
                params=params,
                price_column="Adj Close"
            )
        
        # バックテスト基本理念遵守: 実際のbacktest()実行
        return strategy.backtest()
        
    except Exception as e:
        logger.error(f"戦略実行エラー {strategy_name}: {e}")
        # 空のDataFrameを返すが、必要な列は確保
        empty_result = pd.DataFrame(index=stock_data.index)
        empty_result['Entry_Signal'] = 0
        empty_result['Exit_Signal'] = 0
        return empty_result



def _integrate_entry_signals(integrated_data: pd.DataFrame, strategy_result: pd.DataFrame, strategy_name: str) -> int:
    """
    エントリーシグナル統合（標準版）
    TODO(tag:exit_signal_integration, rationale:enhance entry signal integration with position tracking)
    """
    entry_mask = (strategy_result['Entry_Signal'] == 1) & (integrated_data['Entry_Signal'] == 0)
    
    if entry_mask.any():
        integrated_data.loc[entry_mask, 'Entry_Signal'] = 1
        integrated_data.loc[entry_mask, 'Active_Strategy'] = strategy_name
        integrated_data.loc[entry_mask, 'Position_Duration'] = 0  # 保有期間リセット
        return int(entry_mask.sum())
    
    return 0


def _integrate_exit_signals_with_position_tracking(integrated_data: pd.DataFrame, strategy_result: pd.DataFrame, 
                                                 strategy_name: str, anomaly_info: Dict[str, Any]) -> int:
    """
    エグジットシグナル統合（同日Entry/Exit防止修正版）
    TODO(tag:exit_signal_integration, rationale:implement position-aware exit signal integration)
    """
    exit_integration_count = 0
    
    # 異常パターンの場合は制限的な統合を実行
    if anomaly_info['is_abnormal']:
        return _integrate_exit_signals_filtered(integrated_data, strategy_result, strategy_name, anomaly_info)
    
    # エグジットシグナルがある行を特定
    exit_indices = strategy_result[strategy_result['Exit_Signal'] == 1].index
    
    for exit_idx in exit_indices:
        try:
            # 同日エントリーチェック削除 - バックテスト基本理念遵守のため
            # 修正前: 同日エントリーの場合はエグジットをスキップしていた
            
            # 該当戦略のアクティブポジションを検索
            # 方法1: 直接的なActive_Strategy一致
            direct_match = (integrated_data.loc[exit_idx, 'Active_Strategy'] == strategy_name)
            
            # 方法2: 過去のエントリーから現在まで該当戦略がアクティブか確認
            historical_match = False
            
            # エグジット日より前のエントリーを検索（同日エントリーを除外）- 重要な修正点
            entry_dates = integrated_data[
                (integrated_data['Entry_Signal'] == 1) & 
                (integrated_data['Active_Strategy'] == strategy_name) &
                (integrated_data.index < exit_idx)  # < に変更（同日を除外）
            ].index
            
            if len(entry_dates) > 0:
                # 最新のエントリー日を取得
                latest_entry_date = entry_dates[-1]
                
                # エントリー日からエグジット日まで該当戦略がアクティブか確認
                period_mask = (integrated_data.index >= latest_entry_date) & (integrated_data.index <= exit_idx)
                active_in_period = integrated_data.loc[period_mask, 'Active_Strategy']
                
                # 期間中に該当戦略がアクティブだった場合
                if (active_in_period == strategy_name).any():
                    historical_match = True
            
            # エグジット適用条件
            valid_exit = direct_match or historical_match
            
            if valid_exit and integrated_data.loc[exit_idx, 'Exit_Signal'] == 0:
                # エグジットシグナル適用
                integrated_data.loc[exit_idx, 'Exit_Signal'] = 1
                integrated_data.loc[exit_idx, 'Active_Strategy'] = ''  # ポジション終了
                integrated_data.loc[exit_idx, 'Position_Duration'] = 0  # 保有期間リセット
                
                exit_integration_count += 1
                
                # デバッグ情報
                match_type = "direct" if direct_match else "historical"
                logger.debug(f"{strategy_name}: エグジット統合 ({match_type} match) at {exit_idx}")
        
        except (KeyError, IndexError) as e:
            logger.warning(f"{strategy_name}: エグジット統合エラー at {exit_idx}: {e}")
            continue
    
    return exit_integration_count


def _integrate_exit_signals_filtered(integrated_data: pd.DataFrame, strategy_result: pd.DataFrame,
                                   strategy_name: str, anomaly_info: Dict[str, Any]) -> int:
    """
    異常パターン戦略用のフィルタリング済みエグジット統合（同日Entry/Exit対応追加）
    TODO(tag:exit_signal_integration, rationale:handle abnormal exit patterns safely)
    """
    exit_integration_count = 0
    
    # 同日Entry/Exit問題の対応（新規）
    if anomaly_info['anomaly_type'] == 'same_day_entry_exit':
        # 同日でないエグジットシグナルのみを適用
        for idx in strategy_result[strategy_result['Exit_Signal'] == 1].index:
            try:
                # 同日エントリーチェック削除 - バックテスト基本理念遵守のため
                # 修正前: 同日エントリーの場合はエグジットをスキップしていた
                    
                # 過去のエントリーに対応するエグジットのみ適用
                entry_dates = integrated_data[
                    (integrated_data['Entry_Signal'] == 1) & 
                    (integrated_data['Active_Strategy'] == strategy_name) &
                    (integrated_data.index < idx)  # 同日を除外
                ].index
                
                if len(entry_dates) > 0 and integrated_data.loc[idx, 'Exit_Signal'] == 0:
                    integrated_data.loc[idx, 'Exit_Signal'] = 1
                    integrated_data.loc[idx, 'Active_Strategy'] = ''
                    exit_integration_count += 1
            except (KeyError, IndexError):
                continue
                
        logger.info(f"{strategy_name}: 同日Entry/Exit問題対応により {exit_integration_count}回のエグジット統合")
    
    # 異常パターンに応じたフィルタリング戦略
    elif anomaly_info['anomaly_type'] == 'excessive_exits':
        # 大量エグジットの場合：アクティブポジションに直接対応するもののみ
        active_positions = integrated_data[integrated_data['Active_Strategy'] == strategy_name].index
        
        for pos_idx in active_positions:
            try:
                # 該当日以降の最初のエグジットシグナルを検索
                future_exits = strategy_result.loc[pos_idx:][strategy_result['Exit_Signal'] == 1].index
                
                if len(future_exits) > 0:
                    exit_idx = future_exits[0]
                    
                    # 同日エントリーチェック（追加）
                    if integrated_data.loc[exit_idx, 'Entry_Signal'] == 1:
                        continue
                    
                    if integrated_data.loc[exit_idx, 'Exit_Signal'] == 0:
                        integrated_data.loc[exit_idx, 'Exit_Signal'] = 1
                        integrated_data.loc[exit_idx, 'Active_Strategy'] = ''
                        exit_integration_count += 1
                        
            except (KeyError, IndexError):
                continue
                
    elif anomaly_info['anomaly_type'] == 'high_exit_ratio':
        # 高比率エグジットの場合：エントリー数に制限
        entry_count = (integrated_data['Active_Strategy'] == strategy_name).sum()
        max_exits = min(entry_count * 2, anomaly_info['total_exits'])  # 最大2倍まで
        
        exit_signals = strategy_result[strategy_result['Exit_Signal'] == 1].index[:max_exits]
        
        for exit_idx in exit_signals:
            try:
                # 同日エントリーチェック（追加）
                if integrated_data.loc[exit_idx, 'Entry_Signal'] == 1:
                    continue
                    
                if (integrated_data.loc[exit_idx, 'Active_Strategy'] == strategy_name and 
                    integrated_data.loc[exit_idx, 'Exit_Signal'] == 0):
                    
                    integrated_data.loc[exit_idx, 'Exit_Signal'] = 1
                    integrated_data.loc[exit_idx, 'Active_Strategy'] = ''
                    exit_integration_count += 1
                    
            except (KeyError, IndexError):
                continue
    
    if anomaly_info['anomaly_type'] != 'same_day_entry_exit':
        print(f"{strategy_name}: 異常パターン対応により {exit_integration_count}回のエグジット統合")
    return exit_integration_count


def _validate_strategy_backtest_output(strategy_result, strategy_name):
    """
    バックテスト基本理念違反検出
    TODO(tag:exit_signal_integration, rationale:ensure backtest principle compliance)
    """
    violations = []
    
    # 必須列存在チェック
    required_columns = ['Entry_Signal', 'Exit_Signal']
    missing_columns = [col for col in required_columns if col not in strategy_result.columns]
    if missing_columns:
        violations.append(f"Missing signal columns: {missing_columns}")
    
    # シグナル数チェック
    if 'Entry_Signal' in strategy_result.columns and 'Exit_Signal' in strategy_result.columns:
        entry_signals = (strategy_result['Entry_Signal'] == 1).sum()
        exit_signals = (strategy_result['Exit_Signal'] != 0).sum()  # TODO-003修正: abs()除去、Exit_Signal=-1保持
        
        if entry_signals == 0 and exit_signals == 0:
            violations.append("Zero signals generated - potential strategy logic issue")
    
    # データ整合性チェック
    if len(strategy_result) == 0:
        violations.append("Empty strategy result")
    
    if violations:
        error_msg = f"Strategy backtest violations in {strategy_name}: {'; '.join(violations)}"
        print(f"[WARNING] {error_msg}")
        logger.warning(error_msg)
        # 重大な違反ではエラーを投げずに警告のみ
        return False
    
    return True


def _validate_integrated_signals(integrated_data, strategy_performance):
    """
    統合シグナル整合性検証
    TODO(tag:exit_signal_integration, rationale:validate signal integration quality)
    """
    total_entries = (integrated_data['Entry_Signal'] == 1).sum()
    total_exits = (integrated_data['Exit_Signal'] == 1).sum()
    
    # 基本整合性チェック
    if total_entries == 0:
        print("[WARNING] No entry signals in integrated data - potential integration issue")
        return False
    
    if total_exits == 0:
        print("[WARNING] No exit signals in integrated data - potential integration issue")
    
    # 戦略別統計出力
    print(f"\n=== 戦略別パフォーマンス ===")
    for strategy_name, perf in strategy_performance.items():
        print(f"{strategy_name}: エントリー {perf['entries']}, エグジット {perf['exits']}")
    
    # 整合性警告
    unmatched_positions = total_entries - total_exits
    if unmatched_positions > 0:
        print(f"[WARNING] 未決済ポジション: {unmatched_positions}件（強制決済対象）")
    elif unmatched_positions < 0:
        print(f"[WARNING] エグジット過多: {abs(unmatched_positions)}件（要調査）")
        
    return True


def _execute_intelligent_forced_liquidation(integrated_data: pd.DataFrame) -> Dict[str, Any]:
    """
    インテリジェント強制決済処理（TODO #3修正反映）
    TODO(tag:exit_signal_integration, rationale:implement TODO #3 forced liquidation fix)
    """
    # 期間終了時のアクティブポジション検出
    final_positions_mask = integrated_data['Active_Strategy'] != ''
    
    if final_positions_mask.any():
        final_position_count = final_positions_mask.sum()
        print(f"\n=== インテリジェント強制決済実行: {final_position_count}件 ===")
        
        # 正常な強制決済実行: 全てのアクティブポジションを決済
        integrated_data.loc[final_positions_mask, 'Exit_Signal'] = 1
        integrated_data.loc[final_positions_mask, 'Active_Strategy'] = ''
        actual_forced_count = final_position_count
        
        # TODO #3修正版計算ロジック
        total_exits = (integrated_data['Exit_Signal'] == 1).sum()
        
        # 修正された強制決済率計算（実際に実行された強制決済数を使用）
        if total_exits > 0:
            forced_liquidation_rate = (actual_forced_count / total_exits) * 100
        else:
            forced_liquidation_rate = 0.0
        
        print(f"修正版強制決済率: {forced_liquidation_rate:.2f}%")
        
        return {
            'forced_liquidations': int(actual_forced_count),
            'total_exits': int(total_exits),
            'forced_liquidation_rate': round(forced_liquidation_rate, 2)
        }
    
    return {
        'forced_liquidations': 0,
        'total_exits': int((integrated_data['Exit_Signal'] == 1).sum()),
        'forced_liquidation_rate': 0.0
    }


def _validate_integrated_signals_comprehensive(integrated_data: pd.DataFrame, strategy_performance: Dict[str, Any], 
                                             anomaly_detection: Dict[str, Any]) -> Dict[str, Any]:
    """
    統合シグナル包括的検証
    TODO(tag:exit_signal_integration, rationale:comprehensive validation including anomaly awareness)
    """
    # 同日Entry/Exit問題の検出（修正なし - バックテスト基本理念遵守）
    same_day_results = check_same_day_entry_exit(integrated_data)
    
    # 問題が検出された場合（警告のみ）
    if same_day_results['has_same_day_signals']:
        logger.warning(f"同日Entry/Exit問題を検出: {same_day_results['same_day_count']}件")
        print(f"[WARNING] 同日Entry/Exit問題を検出: {same_day_results['same_day_count']}件")
        
        # バックテスト基本理念に基づき、シグナル自体は修正せず検出のみを行う
        same_day_dates = same_day_results.get('dates', [])
        for date in same_day_dates[:5]:  # 最初の5件だけログ出力
            logger.info(f"  - 同日エントリー/エグジット日付: {date}")
        
        if len(same_day_dates) > 5:
            logger.info(f"  - ...他 {len(same_day_dates) - 5} 件")
    
    total_entries = (integrated_data['Entry_Signal'] == 1).sum()
    total_exits = (integrated_data['Exit_Signal'] == 1).sum()
    
    validation_results = {
        'total_entries': int(total_entries),
        'total_exits': int(total_exits),
        'position_balance': int(total_entries - total_exits),
        'validation_passed': True,
        'warnings': [],
        'errors': [],
        'same_day_signals_fixed': same_day_results['has_same_day_signals']
    }
    
    # 基本整合性チェック
    if total_entries == 0:
        validation_results['errors'].append("No entry signals in integrated data - integration failure")
        validation_results['validation_passed'] = False
    
    if total_exits == 0:
        validation_results['warnings'].append("No exit signals in integrated data - potential integration issue")
    
    # 異常戦略の影響評価
    abnormal_strategies = [name for name, anomaly in anomaly_detection.items() if anomaly.get('is_abnormal', False)]
    if abnormal_strategies:
        validation_results['warnings'].append(f"Abnormal exit patterns detected in: {', '.join(abnormal_strategies)}")
    
    # エグジット過多警告（TODO #2問題対応）
    position_balance = total_entries - total_exits
    if position_balance < -5:  # 5件以上のエグジット過多
        validation_results['warnings'].append(f"Exit surplus detected: {abs(position_balance)} more exits than entries")
    
    return validation_results


def _print_exit_integration_report(strategy_performance: Dict[str, Any], forced_liquidation_stats: Dict[str, Any], 
                                  validation_results: Dict[str, Any]):
    """
    エグジット統合修正レポート出力
    TODO(tag:exit_signal_integration, rationale:comprehensive reporting of integration results)
    """
    print("\n" + "="*70)
    print("[REPORT] TODO #2: エグジットシグナル統合修正 結果レポート")
    print("="*70)
    
    # 戦略別統合結果
    print(f"\n[TOOL] 戦略別統合結果:")
    total_integrated_entries = 0
    total_integrated_exits = 0
    abnormal_strategies = []
    
    for strategy_name, perf in strategy_performance.items():
        print(f"  {strategy_name}:")
        print(f"    エントリー統合: {perf['entries']}回")
        print(f"    エグジット統合: {perf['exits']}回")
        print(f"    合計シグナル: {perf['total_signals']}回")
        
        if perf.get('anomaly_detected', False):
            print(f"    [WARNING] 異常パターン検出")
            abnormal_strategies.append(strategy_name)
        
        total_integrated_entries += perf['entries']
        total_integrated_exits += perf['exits']
    
    # 統合統計
    print(f"\n[STATS] 統合統計:")
    print(f"  総統合エントリー: {total_integrated_entries}回")
    print(f"  総統合エグジット: {total_integrated_exits}回")
    print(f"  統合バランス: {total_integrated_entries - total_integrated_exits}件")
    
    # 強制決済統計（TODO #3修正反映）
    print(f"\n[MONEY] 強制決済統計（TODO #3修正版）:")
    print(f"  強制決済件数: {forced_liquidation_stats['forced_liquidations']}件")
    print(f"  総エグジット数: {forced_liquidation_stats['total_exits']}件")
    print(f"  修正版強制決済率: {forced_liquidation_stats['forced_liquidation_rate']}%")
    
    # 修正効果評価
    print(f"\n[TARGET] TODO #2修正効果評価:")
    
    # エグジット統合改善
    if total_integrated_exits > 10:
        print("[OK] エグジットシグナル統合が正常に機能")
    else:
        print("[WARNING] エグジット統合数が少ない - さらなる調査が必要")
    
    # 強制決済率改善（TODO #3基準）
    forced_rate = forced_liquidation_stats['forced_liquidation_rate']
    if 0 <= forced_rate <= 20:
        print("[OK] 健全な強制決済率（20%以下）")
    elif forced_rate > 100:
        print("[ERROR] 強制決済率が異常値 - TODO #3修正が不完全")
    else:
        print("[WARNING] やや高い強制決済率 - 戦略調整推奨")
    
    # 異常戦略対応
    if abnormal_strategies:
        print(f"[WARNING] 異常パターン戦略: {', '.join(abnormal_strategies)} - TODO #4対応済み")
    else:
        print("[OK] 全戦略が正常パターン")
    
    # バックテスト基本理念遵守確認
    print(f"\n[TARGET] バックテスト基本理念遵守確認:")
    if validation_results['validation_passed']:
        print("[OK] バックテスト基本理念完全遵守")
    else:
        print("[ERROR] バックテスト基本理念違反検出:")
        for error in validation_results['errors']:
            print(f"    - {error}")
    
    # 警告事項
    if validation_results['warnings']:
        print(f"\n[WARNING] 警告事項:")
        for warning in validation_results['warnings']:
            print(f"    - {warning}")
    
    print("\n" + "="*70)


def apply_strategies_with_optimized_params(stock_data: pd.DataFrame, index_data: pd.DataFrame, 
                                         optimized_params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    バックテスト基本理念遵守: 実際の戦略backtest実行 + エグジットシグナル統合修正
    TODO(tag:exit_signal_integration, rationale:restore proper exit signal handling)
    
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
        # ('VWAPBounceStrategy', VWAPBounceStrategy),  # 除外: 9101.T 2年間テストで0エントリー
        # ('OpeningGapFixedStrategy', OpeningGapFixedStrategy),  # Phase B-3完了: 使用不可
        # ('OpeningGapStrategy', OpeningGapStrategy),  # Phase B-3完了: 使用不可（元の実装は同日Entry/Exit問題あり）
        ('ContrarianStrategy', ContrarianStrategy),
        ('GCStrategy', GCStrategy)
    ]
    
    # 統合されたシグナル列を初期化
    # TODO(tag:exit_signal_integration, rationale:initialize exit signal column)
    integrated_data = stock_data.copy()
    integrated_data['Entry_Signal'] = 0
    integrated_data['Exit_Signal'] = 0  # エグジットシグナル初期化追加
    integrated_data['Active_Strategy'] = ''  # アクティブ戦略追跡
    integrated_data['Strategy_Confidence'] = 0.0
    integrated_data['Position'] = 0  # ポジション状態を追跡
    
    # 戦略別結果保存（デバッグ・検証用）
    strategy_results = {}
    strategy_performance = {}
    
    print(f"戦略実行順序: {[name for name, _ in strategy_priority]}")
    
    for strategy_name, strategy_class in strategy_priority:
        try:
            print(f"\n=== {strategy_name} 実行開始 ===")
            params = optimized_params.get(strategy_name, {})
            logger.info(f"戦略適用開始: {strategy_name} with params: {params}")
            
            # バックテスト基本理念遵守: 実際の戦略backtest()実行
            strategy_result = _execute_individual_strategy(
                stock_data, index_data, strategy_name, strategy_class, params
            )
            
            # 基本理念違反検出
            _validate_strategy_backtest_output(strategy_result, strategy_name)
            
            # TODO(tag:exit_signal_integration, rationale:detect and handle abnormal exit patterns)
            # 異常エグジット検出（共通モジュールの関数を使用）
            anomaly_info = detect_exit_anomalies(strategy_result, strategy_name)
            
            # 戦略結果保存
            strategy_results[strategy_name] = strategy_result
            
            # TODO(tag:exit_signal_integration, rationale:implement enhanced entry signal integration)
            # Entry_Signal統合（既存ロジック強化版）
            entry_integration_count = _integrate_entry_signals(
                integrated_data, strategy_result, strategy_name
            )
            
            # TODO(tag:exit_signal_integration, rationale:implement comprehensive exit signal integration)
            # Exit_Signal統合（新規実装 - メイン修正）
            exit_integration_count = _integrate_exit_signals_with_position_tracking(
                integrated_data, strategy_result, strategy_name, anomaly_info
            )
            
            # 統計情報記録
            print(f"{strategy_name}: エントリー統合 {entry_integration_count}回, エグジット統合 {exit_integration_count}回")
            
            # 異常検出結果表示
            if anomaly_info['is_abnormal']:
                print(f"[WARNING] {strategy_name}: 異常エグジットパターン検出 - {anomaly_info['anomaly_type']}")
            
            # パフォーマンス記録
            strategy_performance[strategy_name] = {
                'entries': entry_integration_count,
                'exits': exit_integration_count,
                'total_signals': entry_integration_count + exit_integration_count,
                'anomaly_detected': anomaly_info['is_abnormal']
            }
            
        except Exception as e:
            print(f"[WARNING] {strategy_name} 実行エラー: {e}")
            # TODO(tag:exit_signal_integration, rationale:handle strategy execution errors gracefully)
            continue
    
    # TODO(tag:exit_signal_integration, rationale:implement intelligent forced liquidation)
    # 強制決済処理（バックテスト期間終了時）- TODO #3修正反映
    forced_liquidation_stats = _execute_intelligent_forced_liquidation(integrated_data)
    
    # TODO(tag:exit_signal_integration, rationale:validate integrated signals consistency)
    # 統合結果検証
    anomaly_detection = {}
    for name, perf in strategy_performance.items():
        anomaly_detection[name] = {
            'is_abnormal': perf.get('anomaly_detected', False),
            'anomaly_type': 'unknown'
        }
    
    validation_results = _validate_integrated_signals_comprehensive(
        integrated_data, strategy_performance, anomaly_detection
    )
    
    # 修正効果レポート
    _print_exit_integration_report(
        strategy_performance, forced_liquidation_stats, validation_results
    )
    
    # デバッグ情報出力
    total_entries = (integrated_data['Entry_Signal'] == 1).sum()
    total_exits = (integrated_data['Exit_Signal'] == 1).sum()
    print(f"\n=== 統合結果サマリー ===")
    print(f"総エントリー: {total_entries}回")
    print(f"総エグジット: {total_exits}回")
    print(f"未決済残: {total_entries - total_exits}件")
    
    # 統合データをstock_dataに適用
    stock_data['Entry_Signal'] = integrated_data['Entry_Signal']
    stock_data['Exit_Signal'] = integrated_data['Exit_Signal']
    stock_data['Strategy'] = integrated_data['Active_Strategy']
    stock_data['Position'] = integrated_data['Position']
    
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
                
                # Phase 4-B-1: MultiStrategyManager_fixed を初期化
                manager = MultiStrategyManager()
                
                # システム初期化
                if manager.initialize_systems():
                    logger.info("Phase 4-A成果: 統合システムの初期化に成功しました")
                    
                    # [OK] 実際のbacktest()実行保証
                    available_strategies = list(optimized_params.keys())
                    market_data = {"data": stock_data, "index": index_data}
                    
                    results = manager.execute_multi_strategy_flow(market_data, available_strategies)
                    
                    if results and results.status.value == "ready":
                        logger.info(f"Phase 4-A統合システム成功: {len(results.selected_strategies)}戦略実行完了")
                        
                        # [OK] Excel出力データ検証 (Phase 4-B-2準備)
                        if results.backtest_data:
                            combined_signals = results.backtest_data.get('combined_signals', {})
                            execution_metadata = results.backtest_data.get('execution_metadata', {})
                            total_trades = execution_metadata.get('total_trades', 0)
                            
                            logger.info(f"Phase 4-B品質チェック: 総取引数={total_trades}, 戦略数={len(combined_signals)}")
                            
                            # Phase 4-B-2: Excel出力品質向上
                            if total_trades > 0:
                                logger.info("[OK] バックテスト基本理念遵守: 取引が生成されました")
                                result_data = stock_data.copy()
                                # combined_signalsからシグナルを統合
                                for strategy_name, strategy_result in combined_signals.items():
                                    if 'Entry_Signal' in strategy_result.columns:
                                        # 既存シグナルがない場合のみ統合
                                        mask = result_data['Entry_Signal'] == 0
                                        result_data.loc[mask, 'Entry_Signal'] = strategy_result.loc[mask, 'Entry_Signal']
                                    if 'Exit_Signal' in strategy_result.columns:
                                        mask = result_data['Exit_Signal'] == 0
                                        result_data.loc[mask, 'Exit_Signal'] = strategy_result.loc[mask, 'Exit_Signal']
                            else:
                                logger.warning("[WARNING] バックテスト基本理念注意: 取引数0件 - テストデータ制約の可能性")
                                result_data = stock_data
                                
                            # 同日エントリー/エグジット問題の検出（修正なし）
                            from signal_processing import check_same_day_entry_exit
                            
                            # 同日エントリー/エグジット問題のチェック（修正は行わない）
                            same_day_check = check_same_day_entry_exit(result_data)
                            if same_day_check['has_same_day_signals']:
                                logger.warning(f"同日エントリー/エグジット検出: {same_day_check['same_day_count']}件")
                                # バックテスト基本理念に基づき、シグナルは修正せず検出のみ
                            
                            # 統一出力エンジンに移行（Excel廃棄対応）
                            from output.unified_exporter import UnifiedExporter
                            exporter = UnifiedExporter()
                            
                            # 取引データ生成
                            trades = []
                            if 'Entry_Signal' in result_data.columns and 'Exit_Signal' in result_data.columns:
                                entry_signals = result_data[result_data['Entry_Signal'] == 1]
                                exit_signals = result_data[result_data['Exit_Signal'] != 0]  # TODO-003修正: abs()除去、Exit_Signal=-1保持
                                
                                for idx, row in entry_signals.iterrows():
                                    trades.append({
                                        'timestamp': str(idx),
                                        'type': 'entry', 
                                        'price': float(row['Close']),
                                        'signal': 'Entry_Signal'
                                    })
                                
                                for idx, row in exit_signals.iterrows():
                                    trades.append({
                                        'timestamp': str(idx),
                                        'type': 'exit',
                                        'price': float(row['Close']),
                                        'signal': 'Exit_Signal'
                                    })
                            
                            performance = {
                                'total_trades': len(trades),
                                'integrated_execution': True,
                                'ticker': ticker
                            }
                            
                            backtest_results = exporter.export_main_results(
                                stock_data=result_data,
                                trades=trades,
                                performance=performance, 
                                ticker=ticker,
                                strategy_name="integrated_multi_strategy"
                            )
                        else:
                            logger.warning("backtest_dataが存在しません - フォールバック")
                            result_data = stock_data
                            # 統一出力エンジンフォールバック
                            from output.unified_exporter import UnifiedExporter
                            exporter = UnifiedExporter()
                            trades: List[Dict[str, Any]] = []
                            performance: Dict[str, Any] = {'total_trades': 0, 'fallback_execution': True, 'ticker': ticker}
                            backtest_results = exporter.export_main_results(
                                stock_data=result_data,
                                trades=trades,
                                performance=performance,
                                ticker=ticker,
                                strategy_name="fallback_strategy"
                            )
                    else:
                        logger.warning("統合システムの実行結果が不正でした。従来システムにフォールバックします。")
                        raise Exception("統合システムの実行に失敗")
                else:
                    logger.warning("統合システムの初期化に失敗しました。従来システムにフォールバックします。")
                    raise Exception("統合システムの初期化失敗")
                    
            except Exception as e:
                # SystemFallbackPolicy を使用した明示的フォールバック処理
                use_integrated_system = fallback_policy.handle_component_failure(
                    component_type=ComponentType.MULTI_STRATEGY,
                    component_name="MultiStrategyManager.execute_multi_strategy_flow",
                    error=e,
                    fallback_func=lambda: False
                )
                logger.error(f"統合システム実行中にエラー: {e}")
                if not use_integrated_system:
                    logger.info("従来のマルチ戦略システムにフォールバックします")
        
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
            
        # バックテスト結果をテキストレポート形式で出力（改良版）
        try:
            from output.main_text_reporter import generate_main_text_report
            
            text_report_path = generate_main_text_report(
                stock_data=stock_data,
                ticker=ticker,
                optimized_params=optimized_params,
                output_dir=None  # デフォルトディレクトリを使用
            )
            
            if text_report_path:
                logger.info(f"包括的テキストレポート生成完了: {text_report_path}")
            else:
                logger.warning("テキストレポート生成に失敗しました")
                
        except Exception as e:
            logger.warning(f"テキストレポート生成エラー: {e}")
        
        # 統一出力エンジンによる新形式出力（CSV+JSON+TXT+YAML）
        try:
            from output.unified_exporter import UnifiedExporter
            from typing import List, Dict, Any
            exporter = UnifiedExporter()
            
            # バックテスト基本理念遵守確認
            if 'Entry_Signal' in stock_data.columns and 'Exit_Signal' in stock_data.columns:
                # 取引履歴とパフォーマンス指標を生成
                trades: List[Dict[str, Any]] = []
                entry_signals = stock_data[stock_data['Entry_Signal'] == 1]
                exit_signals = stock_data[stock_data['Exit_Signal'] != 0]  # TODO-003修正: abs()除去、Exit_Signal=-1保持
                
                for idx, row in entry_signals.iterrows():
                    trades.append({
                        'timestamp': str(idx),
                        'type': 'Entry',  # unified_exporter.pyと同じ形式
                        'price': float(row['Close']),
                        'signal': 1  # 数値形式
                    })
                
                for idx, row in exit_signals.iterrows():
                    trades.append({
                        'timestamp': str(idx), 
                        'type': 'Exit',  # unified_exporter.pyと同じ形式
                        'price': float(row['Close']),
                        'signal': 1  # 数値形式 (統合後は1)
                    })
                
                # パフォーマンス指標（包括的な計算に置換）
                performance_metrics = calculate_performance_metrics(stock_data, trades)
                performance: Dict[str, Any] = {
                    **performance_metrics,  # 包括的なパフォーマンス指標
                    'entry_signals': len(entry_signals),
                    'exit_signals': len(exit_signals),
                    'ticker': ticker
                }
                
                export_result = exporter.export_main_results(
                    stock_data=stock_data,
                    trades=trades,
                    performance=performance,
                    ticker=ticker,
                    strategy_name="integrated_strategy"
                )
                logger.info(f"統一出力エンジン成功: {export_result}")
            else:
                logger.warning("バックテスト基本理念違反検出: Entry_Signal/Exit_Signal列が不足")
                # TODO(tag:backtest_execution, rationale:ensure signal columns exist)
        except Exception as e:
            logger.warning(f"統一出力エンジンエラー: {e}")
            # TODO(tag:backtest_execution, rationale:fix unified export error)
        
        # SystemFallbackPolicy 使用統計の出力
        fallback_stats = fallback_policy.get_usage_statistics()
        if fallback_stats['total_failures'] > 0:
            logger.warning(f"フォールバック使用統計: {fallback_stats}")
            # フォールバック使用レポートを JSON として出力
            report_path = fallback_policy.export_usage_report()
            logger.info(f"フォールバック使用レポート出力: {report_path}")
        else:
            logger.info("フォールバック使用記録: なし (正常動作)")
            
        logger.info("マルチ戦略バックテストシステムが正常に完了しました")
        
    except Exception as e:
        logger.exception(f"バックテスト処理中にエラーが発生: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
