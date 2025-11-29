"""
yfinance実データ取得テスト
copilot-instructions.md準拠: 実際のデータ件数確認
"""
import yfinance as yf
from datetime import datetime, timedelta

print("=" * 60)
print("yfinance実データ取得テスト")
print("=" * 60)

# テストパラメータ
symbol = "9101.T"  # NYK Line (日本郵船)
end_date = datetime(2024, 1, 10)
start_date = end_date - timedelta(days=60)

print(f"\nテスト対象:")
print(f"  Symbol: {symbol}")
print(f"  Period: {start_date.date()} to {end_date.date()}")

# データ取得
try:
    ticker = yf.Ticker(symbol)
    stock_data = ticker.history(start=start_date, end=end_date + timedelta(days=1))
    
    if len(stock_data) > 0:
        print(f"\n結果: SUCCESS")
        print(f"  Data rows: {len(stock_data)}")
        print(f"  Columns: {list(stock_data.columns)}")
        print(f"  Date range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
        print(f"  Latest close: {stock_data['Close'].iloc[-1]:.2f} JPY")
        print(f"\ncopilot-instructions.md準拠:")
        print(f"  実データ取得: OK (件数={len(stock_data)} > 0)")
    else:
        print(f"\n結果: FAILED")
        print(f"  Error: No data retrieved")
        
except Exception as e:
    print(f"\n結果: ERROR")
    print(f"  Exception: {e}")

print("=" * 60)
