"""
VWAP_Breakout戦略でのpartial_exit_thresholdエラー修正のテスト
"""
import pandas as pd
import numpy as np
import sys
import logging
from datetime import datetime, timedelta

from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from data_fetcher import get_parameters_and_data
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators

# ロギング設定
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def test_partial_exit_fix():
    """
    部分利確機能のエラー修正テスト
    """
    logger.info("テストデータの準備...")
    
    # テストデータの取得
    ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
    
    # データの前処理
    stock_data = preprocess_data(stock_data)
    stock_data = compute_indicators(stock_data)
    
    # 致命的なエラーのために別の設定でテスト
    # partial_exit_enabledがTrueだが、partial_exit_thresholdが未定義のケース
    test_params = {
        "sma_short": 10,
        "sma_long": 30,
        "stop_loss": 0.03,
        "take_profit": 0.1,
        "volume_threshold": 1.2,
        "confirmation_bars": 1,
        "breakout_min_percent": 0.003,
        "trailing_stop": 0.05,
        "trailing_start_threshold": 0.03,
        "max_holding_period": 10,
        "market_filter_method": "sma",
        "rsi_filter_enabled": False,
        "atr_filter_enabled": False,
        "partial_exit_enabled": True  # ここでpartial_exit_enabledのみをTrueに
        # partial_exit_thresholdとpartial_exit_portionは意図的に省略
    }
    
    logger.info("エラーが発生していたパターンでテスト: partial_exit_enabledのみTrue...")
    try:
        strategy = VWAPBreakoutStrategy(stock_data, index_data, test_params)
        result = strategy.backtest()
        logger.info("✅ テスト成功: 修正により例外がスローされなくなりました")
    except KeyError as e:
        logger.error(f"❌ テスト失敗: KeyErrorが発生しました: {e}")
        raise
    
    # 正常系テスト - 部分利確機能が有効で、必要なパラメータが揃っている場合
    test_params_ok = {
        "sma_short": 10,
        "sma_long": 30,
        "stop_loss": 0.03,
        "take_profit": 0.1,
        "volume_threshold": 1.2,
        "confirmation_bars": 1,
        "breakout_min_percent": 0.003,
        "trailing_stop": 0.05,
        "trailing_start_threshold": 0.03,
        "max_holding_period": 10,
        "market_filter_method": "sma",
        "rsi_filter_enabled": False,
        "atr_filter_enabled": False,
        "partial_exit_enabled": True,
        "partial_exit_threshold": 0.07,
        "partial_exit_portion": 0.5
    }
    
    logger.info("正常系テスト: パラメータが全て揃った状態...")
    strategy = VWAPBreakoutStrategy(stock_data, index_data, test_params_ok)
    result = strategy.backtest()
    logger.info("✅ 正常系テスト成功")
    
    logger.info("全てのテストが完了しました")
    return result

if __name__ == "__main__":
    logger.info("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    logger.info("partial_exit_threshold KeyErrorエラー修正のテスト開始")
    logger.info("='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    
    try:
        result = test_partial_exit_fix()
        logger.info("テスト完了: 修正が機能しています。")
    except Exception as e:
        logger.error(f"テスト失敗: {e}")
        raise
