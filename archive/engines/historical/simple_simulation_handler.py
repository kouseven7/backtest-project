"""
Module: Simple Simulation Handler
File: simple_simulation_handler.py
Description: 
  新しいExcel出力モジュールを使用したバックテストシミュレーションハンドラー。
  古いexcel_result_exporterモジュールへの依存を除去し、
  simple_excel_exporterのみを使用します。

Author: imega
Created: 2025-07-30
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from config.logger_config import setup_logger
from trade_simulation import simulate_trades
# 新しいExcel出力モジュールのみを使用
from output.simple_excel_exporter import save_backtest_results_simple

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\output.log")

def simulate_and_save_improved(data: pd.DataFrame, ticker: str, splits: Optional[List[Tuple]]=None) -> str:
    """
    バックテストシミュレーションを実行し、新しいExcel出力モジュールで結果を保存します。
    
    Parameters:
        data (pd.DataFrame): バックテスト用の株価データ
        ticker (str): 銘柄コード
        splits (Optional[List[Tuple]]): ウォークフォワード分割のデータ
        
    Returns:
        str: 保存したファイルのパス
    """
    logger.info(f"{ticker}のトレードシミュレーション（改良版）を実行します")
    
    try:
        # 新しいExcel出力モジュールで直接保存
        # trade_simulation.pyを経由せずに、データを直接処理
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"improved_backtest_{ticker}_{timestamp}.xlsx"
        
        # 新しいモジュールで出力
        filepath = save_backtest_results_simple(
            stock_data=data,
            ticker=ticker,
            output_dir=None,  # デフォルトディレクトリを使用
            filename=filename
        )
        
        if filepath:
            logger.info(f"改良版バックテスト結果を保存しました: {filepath}")
            
            # ウォークフォワード分割結果がある場合の処理（将来拡張）
            if splits:
                logger.info(f"ウォークフォワード分割データ: {len(splits)} 件（保存機能は今後実装予定）")
            
            return filepath
        else:
            logger.error("改良版Excel出力に失敗しました")
            return ""
            
    except Exception as e:
        logger.error(f"改良版シミュレーション実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return ""

def simulate_and_save_fallback(data: pd.DataFrame, ticker: str, splits: Optional[List[Tuple]]=None) -> str:
    """
    フォールバック用の簡易シミュレーション関数。
    古いモジュールが利用できない場合に使用。
    
    Parameters:
        data (pd.DataFrame): バックテスト用の株価データ
        ticker (str): 銘柄コード
        splits (Optional[List[Tuple]]): ウォークフォワード分割のデータ
        
    Returns:
        str: 保存したファイルのパス
    """
    logger.info(f"{ticker}のフォールバック版シミュレーションを実行します")
    
    try:
        # 古いtrade_simulation.pyの使用を避け、新しいモジュールで直接処理
        return simulate_and_save_improved(data, ticker, splits)
        
    except Exception as e:
        logger.error(f"フォールバック版シミュレーション実行エラー: {e}")
        return ""

# 古いsimulate_and_save関数の互換性のための別名
def simulate_and_save(data: pd.DataFrame, ticker: str, splits: Optional[List[Tuple]]=None) -> str:
    """
    互換性のためのラッパー関数。
    新しいExcel出力モジュールを使用したシミュレーションを実行します。
    
    Parameters:
        data (pd.DataFrame): バックテスト用の株価データ
        ticker (str): 銘柄コード
        splits (Optional[List[Tuple]]): ウォークフォワード分割のデータ
        
    Returns:
        str: 保存したファイルのパス
    """
    logger.info("互換性ラッパー経由で改良版シミュレーションを実行します")
    return simulate_and_save_improved(data, ticker, splits)
