"""
VWAP_Breakout戦略のEntry_Idx NaNエラー修正のテスト
"""
import pandas as pd
import numpy as np
import sys
import logging
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# ロギング設定
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from data_fetcher import get_parameters_and_data
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators

def test_entry_idx_fix():
    """
    Entry_Idx NaNエラー修正のテスト
    """
    logger.info("テストデータの準備...")
    
    # テストデータの取得
    ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
    
    # データ前処理
    stock_data = preprocess_data(stock_data)
    stock_data = compute_indicators(stock_data)
    
    # エッジケースのテスト: NaNを含むデータ
    params = {
        # 少ないエントリーを生成する極端なパラメータ
        "volume_threshold": 10.0,  # 非常に高いボリューム閾値
        "confirmation_bars": 5,    # 多数の確認バー
        "partial_exit_enabled": True,
        # partial_exit_threshold と partial_exit_portion は意図的に省略
    }
    
    # 戦略インスタンスの作成
    strategy = VWAPBreakoutStrategy(stock_data, index_data, params)
    
    try:
        # バックテスト実行
        logger.info("バックテスト実行中...")
        result = strategy.backtest()
        logger.info("バックテストが正常に完了しました。NaNエラー修正が機能しています。")
        
        # エントリーシグナル数の確認
        entry_count = result['Entry_Signal'].sum()
        logger.info(f"生成されたエントリーシグナル数: {entry_count}")
        
        # NaN値を含むEntry_Idxの確認
        nan_entries = result['Entry_Idx'].isna().sum()
        logger.info(f"Entry_Idxにあるnan値の数: {nan_entries}")
        
        return True
        
    except Exception as e:
        logger.error(f"テスト失敗: {e}")
        return False

if __name__ == "__main__":
    success = test_entry_idx_fix()
    if success:
        print("テスト成功：VWAP_BreakoutのEntry_Idx NaNエラー修正が正常に機能しています。")
    else:
        print("テスト失敗：修正が機能していないか、別の問題が発生しています。")
