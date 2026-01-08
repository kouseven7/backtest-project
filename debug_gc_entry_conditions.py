"""
GCStrategy エントリー条件デバッグ - ゴールデンクロス判定確認

DSSMS実行時にGCStrategyがaction=holdを返す理由を特定するため、
2025-01-15〜2025-01-17期間でのSMAデータとエントリー条件を詳細確認する。

Author: AI Assistant
Created: 2026-01-08
"""

import sys
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_fetcher import get_parameters_and_data
import pandas as pd

def debug_gc_entry_conditions():
    """GCStrategyのエントリー条件を詳細デバッグ"""
    
    print("GCStrategy エントリー条件デバッグ開始")
    print("=" * 60)
    
    # 6954.Tのデータを取得（DSSMS実行時と同じ銘柄）
    ticker = "6954.T"
    
    try:
        # データ取得（2025-01-15前後のデータが必要）
        ticker_data, start_date, end_date, stock_data, index_data = get_parameters_and_data(
            ticker=ticker,
            start_date="2024-12-01",  # 十分な期間を取得してSMA計算可能にする
            end_date="2025-02-01",
            warmup_days=150
        )
        
        print(f"データ取得完了:")
        print(f"  銘柄: {ticker}")
        print(f"  期間: {start_date} ~ {end_date}")
        print(f"  データサイズ: {len(stock_data)} rows")
        print(f"  データ範囲: {stock_data.index[0]} ~ {stock_data.index[-1]}")
        
        # SMA計算（GCStrategyと同じパラメータ）
        short_window = 5
        long_window = 25
        
        stock_data[f'SMA_{short_window}'] = stock_data['Adj Close'].rolling(window=short_window).mean()
        stock_data[f'SMA_{long_window}'] = stock_data['Adj Close'].rolling(window=long_window).mean()
        
        # DSSMS実行期間（2025-01-15〜2025-01-17）のデータを抽出
        target_dates = ['2025-01-15', '2025-01-16', '2025-01-17']
        
        print(f"\nSMAデータ詳細分析:")
        print(f"  短期SMA: {short_window}日")
        print(f"  長期SMA: {long_window}日")
        print("-" * 40)
        
        # 対象期間とその前後のデータを確認
        for i, target_date in enumerate(target_dates):
            try:
                # 日付でフィルタリング
                date_str = target_date
                
                # データから該当日を検索
                matching_rows = stock_data[stock_data.index.strftime('%Y-%m-%d') == date_str]
                
                if len(matching_rows) == 0:
                    print(f"  {target_date}: データなし（営業日外の可能性）")
                    continue
                
                # 該当日のインデックスを取得
                idx = stock_data.index.get_loc(matching_rows.index[0])
                
                if idx < long_window:
                    print(f"  {target_date}: SMA計算不可（データ不足: idx={idx} < {long_window}）")
                    continue
                
                # SMA値を取得
                current_short = stock_data[f'SMA_{short_window}'].iloc[idx]
                current_long = stock_data[f'SMA_{long_window}'].iloc[idx]
                prev_short = stock_data[f'SMA_{short_window}'].iloc[idx-1]
                prev_long = stock_data[f'SMA_{long_window}'].iloc[idx-1]
                
                # ゴールデンクロス判定
                golden_cross = current_short > current_long and prev_short <= prev_long
                
                print(f"  {target_date} (idx={idx}):")
                print(f"    当日: SMA{short_window}={current_short:.2f}, SMA{long_window}={current_long:.2f}")
                print(f"    前日: SMA{short_window}={prev_short:.2f}, SMA{long_window}={prev_long:.2f}")
                print(f"    短期>長期: {current_short > current_long}")
                print(f"    前日短期<=長期: {prev_short <= prev_long}")
                print(f"    ゴールデンクロス: {golden_cross}")
                print(f"    エントリー判定: {'YES' if golden_cross else 'NO'}")
                print("")
                
            except Exception as e:
                print(f"  {target_date}: エラー - {e}")
        
        # 追加：2025-01-17前後の広範囲データをチェック
        print("\n広範囲SMAトレンド分析（2025-01-10〜2025-01-20）:")
        print("-" * 50)
        
        extended_range = pd.date_range(start='2025-01-10', end='2025-01-20', freq='D')
        
        for date in extended_range:
            date_str = date.strftime('%Y-%m-%d')
            matching_rows = stock_data[stock_data.index.strftime('%Y-%m-%d') == date_str]
            
            if len(matching_rows) == 0:
                continue
            
            idx = stock_data.index.get_loc(matching_rows.index[0])
            
            if idx < long_window:
                continue
            
            current_short = stock_data[f'SMA_{short_window}'].iloc[idx]
            current_long = stock_data[f'SMA_{long_window}'].iloc[idx]
            adj_close = stock_data['Adj Close'].iloc[idx]
            
            sma_relation = "SHORT>LONG" if current_short > current_long else "SHORT<=LONG"
            
            print(f"  {date_str}: 終値={adj_close:.2f}, SMA{short_window}={current_short:.2f}, SMA{long_window}={current_long:.2f} ({sma_relation})")
        
        print("\nデバッグ完了")
        
    except Exception as e:
        print(f"エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_gc_entry_conditions()