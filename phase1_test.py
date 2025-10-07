import sys
sys.path.append('.')
from main import *
import logging
logging.basicConfig(level=logging.DEBUG)

print("=== Phase 1: 基本動作確認 ===")

# データ取得部分のみテスト
ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
print(f'データ取得結果: {ticker}, {len(stock_data)}行')
print(f'価格範囲: {stock_data["Close"].min():.2f} - {stock_data["Close"].max():.2f}')
print(f'開始日: {start_date}, 終了日: {end_date}')
print(f'株価データ列: {list(stock_data.columns)}')