"""
データ構造詳細分析: DSSMS取引0件問題のデータ不整合調査

DSSMSから戦略に渡されるデータのインデックス形式を確認
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def analyze_data_structure():
    print("=== DSSMS取引0件問題: データ構造分析 ===")
    
    # 8233データを直接取得してインデックス確認
    ticker = "8233.T"
    start_date = "2024-09-02"  # warmup期間込み
    end_date = "2025-01-31"
    
    try:
        # yfinance直接取得
        print(f"\n[1] yfinance直接取得: {ticker}")
        yf_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        print(f"   - Shape: {yf_data.shape}")
        print(f"   - Index type: {type(yf_data.index[0])}")
        print(f"   - Index sample: {yf_data.index[-5:].tolist()}")
        
        # 2025-01-30が存在するか確認
        target_date = pd.Timestamp('2025-01-30')
        print(f"\n[2] 2025-01-30存在チェック:")
        print(f"   - target_date: {target_date} ({type(target_date)})")
        print(f"   - in index: {target_date in yf_data.index}")
        
        # 2025-01-30前後の日付を確認
        print(f"\n[3] 2025-01-30前後の営業日:")
        for i, date in enumerate(yf_data.index):
            if '2025-01-2' in str(date) or '2025-01-3' in str(date):
                print(f"   - {i}: {date} ({type(date)})")
        
        # DSSMSが実際に送信している日付と比較
        print(f"\n[4] 日付フォーマット比較:")
        sample_date = yf_data.index[-1]
        variations = [
            sample_date,
            sample_date.normalize(),
            sample_date.strftime('%Y-%m-%d'),
            pd.Timestamp(sample_date.strftime('%Y-%m-%d')),
            pd.Timestamp('2025-01-30'),
            pd.Timestamp('2025-01-30 00:00:00'),
        ]
        
        for i, var in enumerate(variations):
            in_index = var in yf_data.index
            print(f"   - {i}: {var} ({type(var)}) -> in_index: {in_index}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    analyze_data_structure()