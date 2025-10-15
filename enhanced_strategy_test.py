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

補足:
  - 本ファイルは既存マルチ戦略層 (戦略集合の統合実行) を扱う。
  - 銘柄選択 (DSSMS) は src/dssms/ 以下で独立進化し、将来ここへは
    「選択結果(最適銘柄+バックアップ)を受け取る」一方向インターフェースのみ維持。
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

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\backtest.log")

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
from strategies.VWAP_Bounce import VWAPBounceStrategy
from strategies.Opening_Gap import OpeningGapStrategy
from strategies.Opening_Gap_Fixed import OpeningGapFixedStrategy  # 修正版戦略追加
from strategies.Opening_Gap_Enhanced import OpeningGapEnhancedStrategy  # 強化版戦略追加
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
        'VWAPBounceStrategy',
        'OpeningGapStrategy',
        'OpeningGapEnhancedStrategy',  # 強化版戦略追加
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
        'OpeningGapEnhancedStrategy': {
            'gap_threshold': 0.02,
            'profit_target': 0.05,
            'stop_loss': 0.03,
            'trailing_threshold': 0.015,
            'max_hold_days': 5
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
        elif strategy_name in ['OpeningGapStrategy', 'OpeningGapFixedStrategy', 'OpeningGapEnhancedStrategy']:
            strategy = strategy_class(
                data=stock_data.copy(),
                dow_data=index_data,
                params=params,
                price_column="Adj Close"
            )
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
        ('VWAPBounceStrategy', VWAPBounceStrategy),
        ('OpeningGapEnhancedStrategy', OpeningGapEnhancedStrategy),  # 強化版戦略を優先
        # ('OpeningGapFixedStrategy', OpeningGapFixedStrategy),  # 修正版を使用
        # ('OpeningGapStrategy', OpeningGapStrategy),  # 元の実装は同日Entry/Exit問題あり
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
    
    # main.pyの残りの処理（既存のコード）
    # ...

# main.pyの残りの部分はそのまま利用します


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
        stock_data = apply_strategies_with_optimized_params(stock_data, index_data, optimized_params)
        
        # テスト用: OpeningGapEnhancedStrategyのみ実行
        try:
            print("\n=== OpeningGapEnhancedStrategy のみ実行テスト ===")
            enhanced_strategy = OpeningGapEnhancedStrategy(
                data=stock_data.copy(),
                dow_data=index_data,
                params=get_default_parameters('OpeningGapEnhancedStrategy'),
                price_column="Adj Close"
            )
            
            # 実際のバックテスト実行
            result = enhanced_strategy.backtest()
            
            # 結果検証
            entry_count = (result['Entry_Signal'] == 1).sum()
            exit_count = (result['Exit_Signal'] != 0).sum()
            
            print(f"EnhancedStrategy 結果:")
            print(f"エントリーシグナル: {entry_count}回")
            print(f"エグジットシグナル: {exit_count}回")
            print(f"未決済残: {entry_count - exit_count}件")
        except Exception as e:
            print(f"強化戦略実行エラー: {e}")
        
        # 統一出力エンジンによるバックテスト結果の出力
        try:
            from output.unified_exporter import UnifiedExporter
            exporter = UnifiedExporter()
            
            # 取引履歴とパフォーマンス指標を生成
            trades = []
            if 'Entry_Signal' in stock_data.columns and 'Exit_Signal' in stock_data.columns:
                entry_signals = stock_data[stock_data['Entry_Signal'] == 1]
                exit_signals = stock_data[stock_data['Exit_Signal'] != 0]
                
                for idx, row in entry_signals.iterrows():
                    trades.append({
                        'timestamp': str(idx),
                        'type': 'Entry', 
                        'price': float(row['Close']),
                        'signal': 1
                    })
                
                for idx, row in exit_signals.iterrows():
                    trades.append({
                        'timestamp': str(idx), 
                        'type': 'Exit',
                        'price': float(row['Close']),
                        'signal': 1
                    })
            
            performance = {
                'total_trades': len(trades),
                'entry_signals': len(entry_signals) if 'entry_signals' in locals() else 0,
                'exit_signals': len(exit_signals) if 'exit_signals' in locals() else 0,
                'ticker': ticker
            }
            
            export_result = exporter.export_main_results(
                stock_data=stock_data,
                trades=trades,
                performance=performance,
                ticker=ticker,
                strategy_name="enhanced_strategy_test"
            )
            print(f"統一出力エンジン成功: {export_result}")
        except Exception as e:
            print(f"統一出力エンジンエラー: {e}")
        
        print("拡張戦略テスト完了")
        
    except Exception as e:
        logger.exception(f"バックテスト処理中にエラーが発生: {e}")
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()