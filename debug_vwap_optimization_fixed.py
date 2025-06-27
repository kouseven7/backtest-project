"""
VWAPブレイクアウト戦略のデバッグ用シンプルスクリプト（修正版）
一部の機能を確認することで最適化プロセスのデバッグが行えます
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import traceback

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# より詳細なロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("vwap_debug_fixed.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("vwap_debug")

def run_debug():
    """
    VWAPブレイクアウト戦略の動作を確認する（エラー解決版）
    """
    try:
        # ステップ1: データ取得
        logger.info("■ ステップ1: データ取得")
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        if stock_data is None:
            logger.error("データ取得に失敗しました")
            return
            
        logger.info(f"取得したデータ: {ticker}, {start_date}～{end_date}")
        logger.info(f"データ形状: {stock_data.shape}")
        logger.info(f"最初の5日分のデータ: \n{stock_data.head()}")
        
        # ステップ2: データ前処理とインジケータ計算
        logger.info("\n■ ステップ2: データ前処理とインジケータ計算")
        from data_processor import preprocess_data
        from indicators.indicator_calculator import compute_indicators
        from indicators.basic_indicators import calculate_vwap, calculate_sma, calculate_rsi
        
        # 前処理
        stock_data = preprocess_data(stock_data)
        # 基本インジケータの計算
        stock_data = compute_indicators(stock_data)
        
        # VWAP計算が必要な場合
        if 'VWAP' not in stock_data.columns:
            logger.info("VWAP計算中...")
            stock_data['VWAP'] = calculate_vwap(stock_data, price_column="Adj Close", volume_column="Volume")
        
        # 追加のインジケータ計算
        sma_short = 10
        sma_long = 30
        if f'SMA_{sma_short}' not in stock_data.columns:
            stock_data[f'SMA_{sma_short}'] = calculate_sma(stock_data, "Adj Close", sma_short)
        if f'SMA_{sma_long}' not in stock_data.columns:
            stock_data[f'SMA_{sma_long}'] = calculate_sma(stock_data, "Adj Close", sma_long)
        if 'RSI' not in stock_data.columns:
            stock_data['RSI'] = calculate_rsi(stock_data["Adj Close"], 14)
        
        logger.info(f"処理後のカラム: {stock_data.columns.tolist()}")
        
        # ステップ3: パラメータ確認
        logger.info("\n■ ステップ3: パラメータ確認")
        
        # テスト用のパラメータ
        params = {
            "stop_loss": 0.03,
            "take_profit": 0.1,
            "sma_short": 10,
            "sma_long": 30,
            "volume_threshold": 1.2,
            "confirmation_bars": 1,
            "breakout_min_percent": 0.003,
            "trailing_stop": 0.05,
            "trailing_start_threshold": 0.03,
            "max_holding_period": 10,
            "market_filter_method": "none",
            "rsi_filter_enabled": False,
            "atr_filter_enabled": True,  # ATRフィルターを有効化
            "partial_exit_enabled": False,
            "rsi_period": 14,
            "rsi_lower": 30,
            "rsi_upper": 70,
            "volume_increase_mode": "average"
        }
        
        # ステップ4: 戦略バックテスト
        logger.info("\n■ ステップ4: 戦略バックテスト")
        from strategies.VWAP_Breakout import VWAPBreakoutStrategy
        
        # エラーハンドリング付きで戦略インスタンスを作成
        try:
            # index_dataがNoneでないことを確認
            if index_data is None:
                logger.warning("index_dataがNoneです。ダミーデータを作成します。")
                index_data = pd.DataFrame({
                    'Adj Close': stock_data['Adj Close'].values,
                    'Open': stock_data['Open'].values,
                    'High': stock_data['High'].values,
                    'Low': stock_data['Low'].values,
                    'Close': stock_data['Close'].values,
                    'Volume': stock_data['Volume'].values
                }, index=stock_data.index)
            
            # 戦略インスタンス作成
            strategy = VWAPBreakoutStrategy(stock_data, index_data, params=params)
            strategy.initialize_strategy()
            
            # バックテスト実行
            logger.info("バックテスト実行中...")
            result_data = strategy.backtest()
            
            # 結果の確認
            trade_count = (result_data['Entry_Signal'] == 1).sum()
            exit_count = (result_data['Exit_Signal'] == -1).sum()
            logger.info(f"バックテスト結果: エントリー数={trade_count}, イグジット数={exit_count}")
            
            # 詳細結果
            columns_to_check = ['Entry_Signal', 'Exit_Signal', 'Position', 'Entry_Price']
            for col in columns_to_check:
                if col in result_data.columns:
                    logger.info(f"{col} の統計: {result_data[col].value_counts()}")
            
            # ステップ5: トレードシミュレーション
            logger.info("\n■ ステップ5: トレードシミュレーション")
            from trade_simulation import simulate_trades
            
            # simulate_tradesの存在確認
            if 'simulate_trades' not in dir():
                logger.error("simulate_trades関数が見つかりません")
                return
            
            trade_results = simulate_trades(result_data, ticker)
            logger.info(f"トレードシミュレーション結果のキー: {trade_results.keys() if isinstance(trade_results, dict) else '辞書ではありません'}")
            
            # ステップ6: 目的関数テスト
            logger.info("\n■ ステップ6: 目的関数テスト")
            from optimization.objective_functions import (
                sharpe_ratio_objective, 
                sortino_ratio_objective, 
                expectancy_objective,
                win_rate_expectancy_objective
            )
            
            objectives = {
                "sharpe_ratio": sharpe_ratio_objective,
                "sortino_ratio": sortino_ratio_objective,
                "expectancy": expectancy_objective,
                "win_rate_expectancy": win_rate_expectancy_objective
            }
            
            for name, func in objectives.items():
                try:
                    value = func(trade_results)
                    logger.info(f"{name}: {value}")
                except Exception as e:
                    logger.error(f"{name}計算でエラー: {e}")
                    logger.error(traceback.format_exc())
            
            # ステップ7: ミニ最適化テスト
            logger.info("\n■ ステップ7: ミニ最適化テスト")
            try:
                from optimization.optimize_vwap_breakout_strategy import optimize_vwap_breakout_strategy
                # データの一部だけを使用してテスト
                test_data = stock_data.iloc[-200:].copy()
                test_index = index_data.iloc[-200:].copy() if index_data is not None else None
                
                mini_result = optimize_vwap_breakout_strategy(test_data, test_index, use_parallel=False)
                if mini_result is not None and not mini_result.empty:
                    logger.info(f"ミニ最適化テスト成功: {len(mini_result)}件の結果")
                    logger.info(f"最適パラメータ: \n{mini_result.iloc[0].to_dict()}")
                else:
                    logger.error("最適化結果が空または無効です")
                    
            except Exception as e:
                logger.error(f"最適化テスト中にエラーが発生: {e}")
                logger.error(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"戦略実行エラー: {e}")
            logger.error(traceback.format_exc())
        
        logger.info("\n■ デバッグ完了")
        
    except Exception as e:
        logger.error(f"全体エラーが発生しました: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_debug()
