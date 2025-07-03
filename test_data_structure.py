#!/usr/bin/env python3
"""
データ構造確認とVWAPBounceStrategy簡易テスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def check_data_structure():
    """データ構造の確認"""
    print("=== データ構造確認 ===")
    
    # テスト用データ取得
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 30日間のみ
    
    try:
        data = yf.download("SPY", start=start_date, end=end_date)
        print(f"データ取得完了: {len(data)}日分")
        
        print("\nデータ構造:")
        print("カラム名:")
        print(data.columns.tolist())
        
        print("\nカラム型:")
        print(type(data.columns))
        
        print("\nデータサンプル（最初の5行）:")
        print(data.head())
        
        # カラムの平坦化（MultiIndexの場合）
        if isinstance(data.columns, pd.MultiIndex):
            print("\nMultiIndexカラムを平坦化...")
            data.columns = [col[0] if col[1] == 'SPY' else col[0] for col in data.columns]
            print("平坦化後のカラム:")
            print(data.columns.tolist())
        
        # 必要なカラムの確認
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"\n不足カラム: {missing_columns}")
            # 代替案を試す
            if 'Close' in data.columns and 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close']
                print("'Close'を'Adj Close'として使用")
        
        print(f"\n最終的なカラム: {data.columns.tolist()}")
        print(f"最終的なデータ形状: {data.shape}")
        
        return data
        
    except Exception as e:
        print(f"データ取得エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    check_data_structure()
