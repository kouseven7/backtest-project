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
    読み込んだデータに必要なカラムがない場合は、可能な限り補完します。
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # カラム名を標準化
    if df.columns.str.contains('(?i)adj.?close').any():
        # 大文字小文字を区別せず "adj close" に近いカラム名を探す
        adj_close_cols = df.columns[df.columns.str.contains('(?i)adj.?close')]
        if len(adj_close_cols) > 0:
            df.rename(columns={adj_close_cols[0]: 'Adj Close'}, inplace=True)
    
    # 必須カラムのチェックと補完
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    for col in required_columns:
        if col not in df.columns:
            # 'Adj Close' がなければ 'Close' を使用
            if col == 'Adj Close' and 'Close' in df.columns:
                df['Adj Close'] = df['Close']
            # 'Close' がなく 'Adj Close' があれば逆も補完
            elif col == 'Close' and 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            # その他の必須カラムがなければ、エラーではなく警告を出してダミーデータを作成
            elif col not in ['Adj Close', 'Close']:
                print(f"警告: キャッシュデータに '{col}' カラムがありません。ダミーデータを生成します。")
                if col == 'Volume':
                    df[col] = 0
                else:
                    # 'Open', 'High', 'Low' のどれかが足りない場合は 'Close' か 'Adj Close' を使用
                    if 'Close' in df.columns:
                        df[col] = df['Close']
                    elif 'Adj Close' in df.columns:
                        df[col] = df['Adj Close']
    
    return df

def save_cache(data: pd.DataFrame, filepath: str):
    """
    DataFrame を指定のキャッシュファイルパスに保存します。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath)
