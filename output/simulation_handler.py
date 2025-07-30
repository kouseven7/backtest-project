"""
Module: Simulation Handler
File: simulation_handler.py
Description: 
  バックテストシミュレーションの実行と結果の保存を行うモジュールです。
  トレードシミュレーション結果をExcelに出力し、ウォークフォワード検証の結果も処理します。

Author: imega
Created: 2025-06-05
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from config.logger_config import setup_logger
from trade_simulation import simulate_trades
from output.excel_result_exporter import save_backtest_results, save_splits_to_excel
# 新しいExcel出力モジュールもインポート
from output.simple_excel_exporter import save_backtest_results_simple

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\output.log")

def simulate_and_save(data: pd.DataFrame, ticker: str, splits: Optional[List[Tuple]]=None) -> str:
    """
    バックテストシミュレーションを実行し、結果をExcelファイルに保存します。
    
    Parameters:
        data (pd.DataFrame): バックテスト用の株価データ
        ticker (str): 銘柄コード
        splits (Optional[List[Tuple]]): ウォークフォワード分割のデータ (訓練データとテストデータのタプルのリスト)
        
    Returns:
        str: 保存したファイルのパス
    """
    logger.info(f"{ticker}のトレードシミュレーションを実行します")
    
    # トレードシミュレーションを実行
    trade_results = simulate_trades(data, ticker)
    
    # 出力ディレクトリの設定
    output_dir = os.path.join("backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル名の設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_results_{timestamp}"
    
    # 結果の保存（従来モジュール）
    filepath = save_backtest_results(trade_results, output_dir, filename)
    logger.info(f"従来Excel出力: {filepath}")
    
    # 新しいExcel出力モジュールでも保存
    try:
        new_filepath = save_backtest_results_simple(
            stock_data=data,
            ticker=ticker,
            output_dir=None,  # デフォルトディレクトリを使用
            filename=f"improved_{filename}.xlsx"
        )
        if new_filepath:
            logger.info(f"新Excel出力: {new_filepath}")
        else:
            logger.warning("新Excel出力に失敗しました")
    except Exception as e:
        logger.warning(f"新Excel出力エラー: {e}")
    
    # ウォークフォワード分割結果がある場合は保存
    if splits:
        split_output_path = os.path.join(output_dir, f"splits_{timestamp}.xlsx")
        save_splits_to_excel(splits, split_output_path)
        logger.info(f"ウォークフォワード分割結果を保存しました: {split_output_path}")
    
    logger.info(f"バックテスト結果を保存しました: {filepath}")
    return filepath
