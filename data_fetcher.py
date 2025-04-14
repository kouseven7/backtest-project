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