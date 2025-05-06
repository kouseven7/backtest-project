"""
Module: Main
File: main.py
Description: 
  バックテストプロジェクトのエントリーポイントとなるスクリプトです。
  設定ファイルの読み込み、データの前処理、インジケーター計算、戦略適用、
  そしてバックテスト結果の保存までの一連の処理を実行します。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - config.logger_config
  - config.risk_management
  - config.error_handling
  - config.cache_manager
  - indicators.basic_indicators
  - indicators.bollinger_atr
  - indicators.volume_indicators
  - preprocessing.returns
  - preprocessing.volatility
  - strategies.VWAP_Breakout
  - strategies.Momentum_Investing
  - strategies.Breakout
  - output.excel_result_exporter
  - trade_simulation
"""

#main.py

import sys
import os
import logging
import pandas as pd
from datetime import datetime
from metrics.performance_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_expectancy,
    calculate_max_consecutive_losses,
    calculate_max_consecutive_wins,
    calculate_avg_consecutive_losses,
    calculate_avg_consecutive_wins,
    calculate_max_drawdown_during_losses,
    calculate_total_trades,
    calculate_win_rate,
    calculate_total_profit,
    calculate_average_profit,
    calculate_max_profit,
    calculate_max_loss,
    calculate_max_drawdown,
    calculate_max_drawdown_amount,
    calculate_risk_return_ratio
)

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from config.risk_management import RiskManagement
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.Breakout import BreakoutStrategy
# 追加の戦略をインポート
from strategies.VWAP_Bounce import VWAPBounceStrategy
from strategies.Opening_Gap import OpeningGapStrategy
from strategies.contrarian_strategy import ContrarianStrategy
# GCStrategyを追加
from strategies.gc_strategy_signal import GCStrategy
# ウォークフォワード分割用の関数をインポート
from walk_forward.train_test_split import split_data_for_walk_forward
from output.excel_result_exporter import save_splits_to_excel
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from data_fetcher import get_parameters_and_data
from output.simulation_handler import simulate_and_save
from strategies.strategy_manager import apply_strategies

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\backtest.log")

# リスク管理の初期化
risk_manager = RiskManagement(total_assets=1000000)  # 総資産100万円


def main():
    try:
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        stock_data = apply_strategies(stock_data, index_data)

        # ウォークフォワード用の分割を先に実施
        train_size = 252  # 例: 1年
        test_size = 63    # 例: 3ヶ月
        splits = split_data_for_walk_forward(stock_data, train_size, test_size)

        # バックテスト結果をExcelに出力（splitsを追加で受け渡し）
        backtest_results = simulate_and_save(stock_data, ticker, splits=splits)

        # ここでログ等を出力
        logger.info(f"バックテスト結果をExcelに出力しました: {backtest_results}")
        logger.info("全体のバックテスト処理が正常に完了しました。")

    except Exception as e:
        logger.exception("バックテスト実行中にエラーが発生しました。")
        sys.exit(1)

if __name__ == "__main__":
    main()