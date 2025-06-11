"""
Module: Data Fetcher
File: data_fetcher.py
Description: 
  Yahoo Finance から株価データを取得し、必要に応じて整形するためのモジュールです。
  データのフラット化やタイムゾーンの変換も含まれます。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - pandas
  - yfinance
  - config.logger_config
"""

# data_fetcher.py
import pandas as pd
import yfinance as yf
from config.logger_config import setup_logger
import pandas as pd
import os
import logging
from config.error_handling import read_excel_parameters, fetch_stock_data
from config.cache_manager import get_cache_filepath, save_cache
import datetime
from typing import Optional

logger = logging.getLogger(__name__)

def get_parameters_and_data(ticker: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Excel設定ファイルからパラメータ取得と市場データ取得（キャッシュ利用）を行います。
    引数が指定されていればそれを優先し、なければ設定ファイルから取得します。
    Returns:
        ticker (str), start_date (str), end_date (str), stock_data (pd.DataFrame), index_data (pd.DataFrame)
    """
    config_base_path = r"C:\Users\imega\Documents\my_backtest_project\config"
    config_file_xlsx = os.path.join(config_base_path, "backtest_config.xlsx")
    config_file_xlsm = os.path.join(config_base_path, "backtest_config.xlsm")
    config_csv = os.path.join(config_base_path, "config.csv")

    # 設定ファイル・デフォルト値による補完
    if ticker is not None and start_date is not None and end_date is not None:
        logger.info(f"引数指定: {ticker}, {start_date}, {end_date}")
    else:
        try:
            if os.path.exists(config_file_xlsx):
                config_df = read_excel_parameters(config_file_xlsx, "銘柄設定")
                logger.info(f"設定ファイル読み込み: {config_file_xlsx}")
            elif os.path.exists(config_file_xlsm):
                config_df = read_excel_parameters(config_file_xlsm, "銘柄設定")
                logger.info(f"設定ファイル読み込み: {config_file_xlsm}")
            else:
                logger.warning(f"Excel設定ファイルが見つからないため、CSVファイルを使用します: {config_csv}")
                config_df = pd.read_csv(config_csv)

            # 銘柄情報の取得
            if ticker is None:
                if "銘柄" in config_df.columns:
                    ticker = str(config_df["銘柄"].iloc[0])
                elif "ticker" in config_df.columns:
                    ticker = str(config_df["ticker"].iloc[0])
                else:
                    ticker = "9101.T"
                    logger.warning(f"銘柄情報が見つからないため、デフォルト値を使用します: {ticker}")
            # 日付情報の取得
            if start_date is None or end_date is None:
                if "開始日" in config_df.columns and "終了日" in config_df.columns:
                    s = config_df["開始日"].iloc[0]
                    e = config_df["終了日"].iloc[0]
                    if isinstance(s, (pd.Timestamp, datetime.datetime)):
                        start_date = s.strftime('%Y-%m-%d')
                    else:
                        start_date = str(s)
                    if isinstance(e, (pd.Timestamp, datetime.datetime)):
                        end_date = e.strftime('%Y-%m-%d')
                    else:
                        end_date = str(e)
                elif "start_date" in config_df.columns and "end_date" in config_df.columns:
                    start_date = str(config_df["start_date"].iloc[0])
                    end_date = str(config_df["end_date"].iloc[0])
                else:
                    start_date = "2023-01-01"
                    end_date = "2023-12-31"
                    logger.warning(f"日付情報が見つからないため、デフォルト値を使用します: {start_date} ~ {end_date}")
            logger.info(f"パラメータ取得: {ticker}, {start_date}, {end_date}")
        except Exception as e:
            logger.error(f"設定ファイルの読み込みに失敗しました: {str(e)}")
            if ticker is None:
                ticker = "9101.T"
            if start_date is None:
                start_date = "2023-01-01"
            if end_date is None:
                end_date = "2023-12-31"
            logger.warning(f"デフォルト値を使用します: {ticker}, {start_date}, {end_date}")

    # ここで必ずstr型にキャスト
    ticker = str(ticker)
    start_date = str(start_date)
    end_date = str(end_date)

    try:
        cache_filepath = get_cache_filepath(ticker, start_date, end_date)
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        save_cache(stock_data, cache_filepath)

        if 'Adj Close' not in stock_data.columns:
            logger.warning(f"'{ticker}' のデータに 'Adj Close' が存在しないため、'Close' 列を代用します。")
            stock_data['Adj Close'] = stock_data['Close']

        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
    except Exception as e:
        logger.error(f"データ取得に失敗しました: {str(e)}")
        cache_dir = r"C:\Users\imega\Documents\my_backtest_project\data_cache"
        cache_files = os.listdir(cache_dir)
        matching_files = [f for f in cache_files if f.startswith(ticker)]
        if matching_files:
            latest_file = sorted(matching_files)[-1]
            cache_path = os.path.join(cache_dir, latest_file)
            logger.info(f"キャッシュファイルを使用します: {cache_path}")
            stock_data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"銘柄 {ticker} のデータが見つかりませんでした。")

    try:
        index_ticker = "^N225" if ticker and ticker.endswith(".T") else "^GSPC"
        index_cache_filepath = get_cache_filepath(index_ticker, start_date, end_date)

        if (os.path.exists(index_cache_filepath)):
            logger.info(f"インデックス {index_ticker} のキャッシュを使用します")
            index_data = pd.read_csv(index_cache_filepath, index_col=0, parse_dates=True)
        else:
            logger.info(f"インデックス {index_ticker} のデータを取得します")
            index_data = fetch_stock_data(index_ticker, start_date, end_date)
            save_cache(index_data, index_cache_filepath)

        if 'Adj Close' not in index_data.columns:
            logger.warning(f"'{index_ticker}' のデータに 'Adj Close' が存在しないため、'Close' 列を 'Adj Close' として利用します。")
            index_data['Adj Close'] = index_data['Close']

        if isinstance(index_data.columns, pd.MultiIndex):
            index_data.columns = index_data.columns.get_level_values(0)
    except Exception as e:
        logger.error(f"インデックスデータの取得に失敗しました: {str(e)}")
        index_data = None
        logger.warning("インデックスデータなしで処理を続行します。")

    return ticker, start_date, end_date, stock_data, index_data

logger = setup_logger(__name__)

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame のカラムが MultiIndex になっている場合、最初のレベルのみを取得してフラット化します。
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def fetch_yahoo_data(ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if data.empty:
        logger.error(f"{ticker} のデータが空です。")
        raise ValueError(f"{ticker} のデータが空です。")
    data = flatten_columns(data)
    data.columns = data.columns.str.strip().str.title()
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('Asia/Tokyo')
    
    # "Adj Close" 列がなければ "Close" 列を代用
    if 'Adj Close' not in data.columns:
        logger.warning(f"{ticker} のデータに 'Adj Close' 列が存在しないため、'Close' 列を 'Adj Close' として利用します。")
        data['Adj Close'] = data['Close']
        
    return data

def fetch_yahoo_index_data(ticker: str, start_date: str = None, end_date: str = None, period: str = "max", interval: str = '1d') -> pd.DataFrame:
    """
    Yahoo Financeから指定したインデックスのデータを取得します。
    start_dateとend_dateが指定されている場合はその期間のデータを取得し、
    指定されていない場合はperiodパラメータを使用します。
    
    Parameters:
        ticker (str): ティッカーシンボル（例: ^N225）
        start_date (str, optional): 開始日（例: 2022-01-01）
        end_date (str, optional): 終了日（例: 2023-01-01）
        period (str, optional): 期間（start_dateとend_dateが指定されていない場合に使用）
        interval (str, optional): データの間隔（例: 1d, 1wk, 1mo）
        
    Returns:
        pd.DataFrame: 取得したデータ
    """
    if start_date and end_date:
        logger.info(f"Fetching index data for {ticker} from {start_date} to {end_date}")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        logger.info(f"Fetching index data for {ticker} with period={period}")
        data = yf.download(ticker, period=period, interval=interval)
    
    if data.empty:
        logger.error(f"{ticker} のデータが空です。")
        raise ValueError(f"{ticker} のデータが空です。ダミーデータは利用しません。")
    
    data = flatten_columns(data)
    data.columns = data.columns.str.strip().str.title()
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('Asia/Tokyo')
    
    # 必要なカラムセット
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.intersection(set(data.columns)):
        # 代替候補をチェック
        alt = None
        for candidate in ["Adj Close", "Adjclose", "Price"]:
            if candidate in data.columns:
                alt = candidate
                break
        if alt is not None:
            logger.warning(f"{ticker} のデータに必要なカラムが存在しないため、'{alt}' 列から補完します。")
            data["Open"] = data[alt]
            data["High"] = data[alt]
            data["Low"] = data[alt]
            data["Close"] = data[alt]
            data["Volume"] = 0
        else:
            logger.error(f"{ticker} のデータに必要なカラムが存在しません。")
            raise KeyError(f"{ticker} のデータに必要なカラムも 'Adj Close' も存在しません。")
    
    # "Adj Close" 列がなければ "Close" 列を代用
    if 'Adj Close' not in data.columns:
        logger.warning(f"{ticker} のデータに 'Adj Close' 列が存在しないため、'Close' 列を 'Adj Close' として利用します。")
        data['Adj Close'] = data['Close']
    
    return data