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

def fetch_yahoo_index_data(ticker: str, period: str = "max", interval: str = '1d') -> pd.DataFrame:
    logger.info(f"Fetching index data for {ticker} with period={period}")
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        logger.error(f"{ticker} のデータが空です。")
        raise ValueError(f"{ticker} のデータが空です。")
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
            raise KeyError(f"{ticker} のデータに必要なカラムも 'Adj Close' も存在しません。")
    return data
