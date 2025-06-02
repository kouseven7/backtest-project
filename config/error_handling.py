"""
Module: Error Handling
File: error_handling.py
Description: 
  エラーハンドリングとデータ取得に関連するユーティリティ関数を提供するモジュールです。
  Excelファイルの読み込みや株価データの取得を含みます。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - pandas
  - yfinance
  - config.logger_config
  - config.file_utils
"""

#error_handling.py
#エラーハンドリングモジュール

import logging
import pandas as pd
import yfinance as yf
import time
from config.logger_config import setup_logger
from config.file_utils import resolve_excel_file

# Excel設定ファイルのパスを、解決関数で取得
config_file = resolve_excel_file(r"C:\Users\imega\Documents\my_backtest_project\config\backtest_config.xlsx")


logger = setup_logger(__name__)

def read_excel_parameters(excel_file: str, sheet_name: str) -> pd.DataFrame:
    """
    Excelファイルから指定されたシートのパラメータを読み込む。
    エラー発生時にはログ出力を行い、例外を再スローする。
    """
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        logger.info(f"Excelシート '{sheet_name}' の読み込みに成功しました。")
        return df
    except FileNotFoundError as e:
        logger.exception(f"Excelファイルが見つかりませんでした: {excel_file}")
        raise
    except Exception as e:
        logger.exception(f"Excelシート '{sheet_name}' の読み込み中にエラーが発生しました。")
        raise

def fetch_stock_data(ticker: str, start_date: str, end_date: str, max_retries=3, wait_sec=10) -> pd.DataFrame:
    """
    yfinanceを用いて株価データを取得する（リトライ付き）。
    データが空の場合やその他のエラー発生時にはログ出力を行い、例外を再スローする。
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"{ticker} のデータ取得を開始します: {start_date} から {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"{ticker} の取得データが空です。")
            logger.info(f"{ticker} のデータ取得に成功しました。")
            return data
        except Exception as e:
            logger.warning(f"{ticker} のデータ取得失敗（{attempt+1}/{max_retries}回目）: {e}")
            if attempt < max_retries - 1:
                logger.info(f"{wait_sec}秒待機してリトライします...")
                time.sleep(wait_sec)
            else:
                logger.error(f"{ticker} のデータ取得に失敗しました（リトライ上限到達）")
                raise

def calculate_moving_average(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    """
    指定されたカラムの単純移動平均 (SMA) を計算する関数。
    カラムが存在しない場合などのエラー発生時にはログ出力を行い、例外を再スローする。
    """
    try:
        sma = data[column].rolling(window=window).mean()
        logger.info(f"{column} の {window}日移動平均の計算に成功しました。")
        return sma
    except KeyError as e:
        logger.exception(f"カラム '{column}' がデータに存在しません。")
        raise
    except Exception as e:
        logger.exception("移動平均の計算中にエラーが発生しました。")
        raise

# サンプル実行コード（直接実行時のみ）
if __name__ == "__main__":
    # グローバルの config_file を利用
    try:
        params_df = read_excel_parameters(config_file, "銘柄設定")
        ticker = params_df["銘柄"].iloc[0]
        start_date = params_df["開始日"].iloc[0]
        end_date = params_df["終了日"].iloc[0]
        logger.info(f"Excelから取得したパラメータ: {ticker}, {start_date}, {end_date}")
    except Exception as e:
        logger.error("Excelパラメータの読み込みに失敗しました。")
        raise

    try:
        stock_data = fetch_stock_data(ticker, start_date, end_date)
    except Exception as e:
        logger.error("株価データの取得に失敗しました。")
        raise

    try:
        sma_5 = calculate_moving_average(stock_data, 'Close', 5)
        stock_data['SMA_5'] = sma_5
    except Exception as e:
        logger.error("移動平均の計算に失敗しました。")
        raise