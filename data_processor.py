import pandas as pd
import preprocessing.returns as returns
import preprocessing.volatility as volatility
import logging

logger = logging.getLogger(__name__)

def preprocess_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    前処理として、日次リターンと累積リターン、ボラティリティを計算します。
    """
    stock_data = returns.add_returns(stock_data, price_column="Adj Close")
    stock_data = volatility.add_volatility(stock_data)
    logger.info("前処理（リターン、ボラティリティ計算）完了")
    return stock_data