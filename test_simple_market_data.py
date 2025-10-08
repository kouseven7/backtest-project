"""
TODO #14 Phase 2: RealMarketDataFetcher 簡易テスト
エラー箇所特定のための簡易版テスト

Author: AI Assistant
Created: 2025-10-08
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

print("=== RealMarketDataFetcher 簡易テスト ===")

# Step 1: yfinance データ取得テスト
print("\n1. yfinance データ取得テスト")
try:
    symbol = '^N225'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    
    print(f"データ取得中: {symbol} from {start_date} to {end_date}")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    print(f"取得データタイプ: {type(data)}")
    print(f"データ形状: {data.shape}")
    print(f"データ列: {list(data.columns)}")
    print(f"データ最初の5行:")
    print(data.head())
    
    # Step 2: 空チェックテスト
    print("\n2. 空チェックテスト")
    if isinstance(data, pd.DataFrame):
        print(f"DataFrame: len={len(data)}, empty={len(data) == 0}")
    else:
        print(f"非DataFrame: {type(data)}")
    
    # Step 3: Closeカラムテスト
    print("\n3. Closeカラムテスト")
    if 'Close' in data.columns:
        close_col = data['Close']
        print(f"Close列タイプ: {type(close_col)}")
        print(f"Close列形状: {close_col.shape}")
        print(f"Close NaN数: {close_col.isna().sum()}")
        print(f"Close全てNaN?: {close_col.isna().all()}")
        
        # Step 4: 問題箇所の特定
        print("\n4. 問題箇所特定テスト")
        try:
            # この行でエラーが発生している可能性
            if close_col.isna().all():
                print("Close列は全てNaN")
            else:
                print("Close列にデータあり")
                
                # dropna テスト
                close_clean = close_col.dropna()
                print(f"dropna後: {len(close_clean)} rows")
                
                if len(close_clean) > 0:
                    min_val = close_clean.min()
                    print(f"最小値: {min_val}")
                    
                    if min_val <= 0:
                        print("非正値検出")
                    else:
                        print("価格データOK")
                        
                    # 変動率テスト
                    if len(close_clean) > 1:
                        returns = close_clean.pct_change().dropna()
                        print(f"変動率数: {len(returns)}")
                        
                        # この条件式でエラーが発生している可能性
                        extreme_condition = (returns > 4) | (returns < -0.8)
                        print(f"極端変動条件タイプ: {type(extreme_condition)}")
                        
                        # ここでエラー？
                        extreme_moves = returns[extreme_condition]
                        print(f"極端変動数: {len(extreme_moves)}")
                
        except Exception as e:
            print(f"問題箇所特定エラー: {e}")
            print(f"エラータイプ: {type(e)}")
    
    print("\n[OK] 簡易テスト完了")
    
except Exception as e:
    print(f"[ERROR] テストエラー: {e}")
    print(f"エラータイプ: {type(e)}")
    import traceback
    traceback.print_exc()