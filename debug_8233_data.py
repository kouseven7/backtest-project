import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_fetcher import get_parameters_and_data
from indicators.basic_indicators import calculate_sma

print("=== 8233 2025-01-30データ分析 ===")

# データ取得
ticker, start_date, end_date, data, _ = get_parameters_and_data(
    ticker="8233", 
    start_date="2024-09-01", 
    end_date="2025-01-31",
    warmup_days=150
)

print(f"データ期間: {data.index[0]} - {data.index[-1]}")
print(f"データ形状: {data.shape}")

# SMA計算
data['SMA_5'] = calculate_sma(data, 'Adj Close', 5)
data['SMA_25'] = calculate_sma(data, 'Adj Close', 25)

# 2025-01-30の前後データを確認
target_date = "2025-01-30"
try:
    target_loc = data.index.get_loc(target_date)
    print(f"target_date {target_date} のインデックス: {target_loc}")
    
    # 前後5日のデータ表示
    start_idx = max(0, target_loc - 5)
    end_idx = min(len(data) - 1, target_loc + 5)
    
    print("\n=== 前後データ (2025-01-24 - 2025-01-31) ===")
    subset = data.iloc[start_idx:end_idx+1][['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA_5', 'SMA_25']]
    print(subset)
    
    # 2025-01-30の詳細分析
    print(f"\n=== 2025-01-30 詳細判定 (idx={target_loc}) ===")
    
    # 前日データ（ルックアヘッドバイアス回避）
    if target_loc >= 1:
        prev_date = data.index[target_loc - 1]
        prev_price = data['Adj Close'].iloc[target_loc - 1]
        prev_volume = data['Volume'].iloc[target_loc - 1]
        prev_sma_5 = data['SMA_5'].iloc[target_loc - 1]
        prev_sma_25 = data['SMA_25'].iloc[target_loc - 1]
        
        # 前々日ボリューム
        if target_loc >= 2:
            prev_prev_volume = data['Volume'].iloc[target_loc - 2]
            volume_ratio = prev_volume / prev_prev_volume if prev_prev_volume > 0 else 0
        else:
            prev_prev_volume = prev_volume
            volume_ratio = 1.0
        
        print(f"前日 ({prev_date}): price={prev_price:.2f}, sma5={prev_sma_5:.2f}, sma25={prev_sma_25:.2f}")
        print(f"前日ボリューム: {prev_volume:,}")
        print(f"前々日ボリューム: {prev_prev_volume:,}")
        print(f"ボリューム比率: {volume_ratio:.3f}")
        
        # 条件判定
        print("\n--- 条件判定 ---")
        
        # 価格ブレイクアウト（緩和版）
        price_breakout = prev_price > prev_sma_25
        print(f"価格ブレイクアウト（緩和）: {prev_price:.2f} > {prev_sma_25:.2f} = {price_breakout}")
        
        # ボリューム条件（0.8閾値）
        volume_threshold = 0.8
        volume_condition = volume_ratio >= volume_threshold
        print(f"ボリューム条件: {volume_ratio:.3f} >= {volume_threshold} = {volume_condition}")
        
        # 最終判定
        entry_signal = price_breakout and volume_condition
        print(f"\n>>> 最終エントリーシグナル: {entry_signal}")
        
        # 翌日始値確認（エントリー価格）
        if target_loc < len(data) - 1:
            next_day_open = data['Open'].iloc[target_loc + 1]
            print(f"翌日始値 (エントリー価格): {next_day_open:.2f}")
        else:
            print("翌日データなし（最終日のため取引不可）")
    
except Exception as e:
    print(f"エラー: {e}")
    
    # データの最後の日を確認
    print(f"\n最後のデータ: {data.index[-1]}")
    
    # 2025年1月のデータがあるか確認
    jan_2025_data = data[data.index >= '2025-01-01']
    if len(jan_2025_data) > 0:
        print(f"2025年1月データ: {jan_2025_data.index[0]} - {jan_2025_data.index[-1]} ({len(jan_2025_data)}件)")
        print(jan_2025_data.tail())
    else:
        print("2025年1月のデータが見つかりません")