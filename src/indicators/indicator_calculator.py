import pandas as pd
import logging
from indicators.basic_indicators import add_basic_indicators
from indicators.bollinger_atr import bollinger_atr
from indicators.volume_indicators import add_volume_indicators

logger = logging.getLogger(__name__)

def compute_indicators(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    基本インジケーター、ボリンジャーバンド、出来高関連指標を計算して追加します。
    """
    stock_data = add_basic_indicators(stock_data, price_column="Adj Close")
    stock_data = bollinger_atr(stock_data, price_column="Adj Close")
    stock_data = add_volume_indicators(stock_data, price_column="Adj Close")
    logger.info("インジケーター計算完了")
    return stock_data