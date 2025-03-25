# cache_manager.py
import os
import pandas as pd
from datetime import datetime, timedelta

#データ取得結果のキャッシュ管理を行うモジュール
#キャッシュファイルのパス生成、キャッシュの有効性チェック、読み込み、保存の関数を実装

def get_cache_filepath(ticker: str, start_date: str, end_date: str, interval: str = '1d', cache_dir: str = 'data_cache') -> str:
    """
    ティッカー、開始日、終了日、間隔からキャッシュ用のファイルパスを生成します。
    """
    filename = f"{ticker}_{start_date}_{end_date}_{interval}.csv"
    return os.path.join(cache_dir, filename)

def is_cache_valid(filepath: str, valid_days: int = 7) -> bool:
    """
    キャッシュファイルが存在し、かつ最終更新日が指定日数以内であれば True を返します。
    """
    if not os.path.exists(filepath):
        return False
    modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    return (datetime.now() - modified_time) < timedelta(days=valid_days)

def load_cache(filepath: str) -> pd.DataFrame:
    """
    指定のキャッシュファイルを読み込んで DataFrame を返します。
    """
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def save_cache(data: pd.DataFrame, filepath: str):
    """
    DataFrame を指定のキャッシュファイルパスに保存します。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath)
