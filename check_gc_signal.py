import yfinance as yf
import pandas as pd

ticker = '9101.T'
data = yf.download(ticker, start='2024-07-01', end='2025-01-20', auto_adjust=False)

# 移動平均計算（Phase 1c対応: shift(1)適用）
data['SMA_5'] = data['Adj Close'].rolling(window=5).mean().shift(1)
data['SMA_25'] = data['Adj Close'].rolling(window=25).mean().shift(1)

# GCシグナル計算
data['GC_Signal'] = (
    (data['SMA_5'] > data['SMA_25']) & 
    (data['SMA_5'].shift(1) <= data['SMA_25'].shift(1))
).astype(int)

# 2025-01-10以降のデータを表示
recent = data[data.index >= '2025-01-10'][['Adj Close', 'SMA_5', 'SMA_25', 'GC_Signal']]
print("=== 2025-01-10以降のデータ ===")
print(recent)

# 2025-01-17のシグナルを確認
if '2025-01-17' in recent.index:
    row = recent.loc['2025-01-17']
    print(f"\n=== 2025-01-17のシグナル ===")
    print(f"Adj Close: {row['Adj Close']:.2f}")
    print(f"SMA_5: {row['SMA_5']:.2f}")
    print(f"SMA_25: {row['SMA_25']:.2f}")
    print(f"GC_Signal: {int(row['GC_Signal'])}")
    print(f"GC発生: {'はい' if row['GC_Signal'] == 1 else 'いいえ'}")
