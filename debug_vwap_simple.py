"""
VWAPブレイクアウト戦略の最適化デバッグスクリプト（簡略版）
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# より詳細なロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("vwap_debug_simple.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("vwap_debug")

def run_debug():
    """
    VWAPブレイクアウト戦略のデバッグ（特に目的関数部分に焦点）
    """
    try:
        # ステップ1: データ取得
        logger.info("■ ステップ1: データ取得")
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # テストデータのサイズを縮小（200日だけ）
        test_data = stock_data.iloc[-200:].copy()
        test_index = index_data.iloc[-200:].copy() if index_data is not None else None
        
        # ステップ2: バックテスト実行
        logger.info("■ ステップ2: バックテスト実行")
        from strategies.VWAP_Breakout import VWAPBreakoutStrategy
        
        # テスト用のパラメータ
        params = {
            "stop_loss": 0.05,
            "take_profit": 0.1,
            "sma_short": 10,
            "sma_long": 20,
            "volume_threshold": 1.5,
            "confirmation_bars": 1,
            "breakout_min_percent": 0.005,
            "trailing_stop": 0.05,
            "trailing_start_threshold": 0.03,
            "max_holding_period": 10,
            "market_filter_method": "none",
            "rsi_filter_enabled": False,
            "atr_filter_enabled": False,
            "partial_exit_enabled": True,
            "partial_exit_threshold": 0.07,
            "partial_exit_portion": 0.5,
            "rsi_period": 14,
            "rsi_lower": 30,
            "rsi_upper": 70,
            "volume_increase_mode": "average"
        }
        
        # 戦略を実行
        strategy = VWAPBreakoutStrategy(test_data, test_index, params=params)
        strategy.initialize_strategy()
        result_data = strategy.backtest()
        
        # 取引状況確認
        trade_count = (result_data['Entry_Signal'] == 1).sum()
        exit_count = (result_data['Exit_Signal'] == -1).sum()
        logger.info(f"バックテスト結果: エントリー数={trade_count}, イグジット数={exit_count}")
        
        # ステップ3: トレードシミュレーション
        logger.info("■ ステップ3: トレードシミュレーション")
        from trade_simulation import simulate_trades
        trade_results = simulate_trades(result_data, ticker)
        
        # 取引履歴を確認
        trades_df = trade_results.get('取引履歴', pd.DataFrame())
        if not trades_df.empty:
            logger.info(f"取引数: {len(trades_df)}")
            logger.info(f"取引履歴カラム: {trades_df.columns.tolist()}")
            logger.info(f"取引結果サマリ: {trades_df['取引結果'].describe()}")
        else:
            logger.warning("取引がありません！")
        
        # ステップ4: 目的関数テスト
        logger.info("■ ステップ4: 目的関数テスト")
        from optimization.objective_functions import (
            sharpe_ratio_objective, 
            sortino_ratio_objective, 
            expectancy_objective,
            win_rate_expectancy_objective,
            create_custom_objective
        )
        
        # 個別の目的関数をテスト
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
        
        # 複合目的関数をテスト
        objectives_config = [
            {"name": "sharpe_ratio", "weight": 1.0},
            {"name": "win_rate", "weight": 0.5},
            {"name": "expectancy", "weight": 0.5}
        ]
        
        composite_objective = create_custom_objective(objectives_config)
        try:
            composite_score = composite_objective(trade_results)
            logger.info(f"複合目的関数スコア: {composite_score}")
        except Exception as e:
            logger.error(f"複合目的関数でエラー: {e}")
            
        logger.info("■ デバッグ完了")
        
    except Exception as e:
        logger.error(f"全体エラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_debug()
