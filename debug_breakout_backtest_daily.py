"""
BreakoutStrategy backtest_daily() 詳細調査

BreakoutStrategyのbacktest_daily()メソッドでエントリーが0件になる原因を特定します。

主な調査ポイント:
- エントリー条件の詳細検証
- インジケーター値の確認
- ボリューム条件の確認
- SMA条件の確認
- デバッグログを詳細化

Author: Backtest Project Team
Created: 2026-01-10
Last Modified: 2026-01-10
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
sys.path.append(r'C:\Users\imega\Documents\my_backtest_project')

from data_fetcher import get_parameters_and_data
from strategies.Breakout import BreakoutStrategy
from indicators.basic_indicators import calculate_sma

def debug_breakout_backtest_daily():
    """BreakoutStrategy.backtest_daily() 詳細調査"""
    
    try:
        # DSSMS対象データ取得
        ticker_symbol = "8233.T"  # 前回のDSSMS選択銘柄
        start_date = "2024-09-03"  # 150日warmup考慮
        end_date = "2025-02-01"
        
        print(f"\n=== BreakoutStrategy backtest_daily() 詳細調査 ===")
        print(f"対象銘柄: {ticker_symbol}")
        print(f"調査期間: 2025-01-15 → 2025-01-31 (DSSMS期間)")
        print(f"データ期間: {start_date} → {end_date}")
        
        ticker, start, end, stock_data, index_data = get_parameters_and_data(
            ticker=ticker_symbol,
            start_date=start_date,
            end_date=end_date,
            warmup_days=0  # 既にwarmup考慮済み
        )
        
        if stock_data is None or stock_data.empty:
            print(f"[ERROR] データ取得失敗: {ticker}")
            return
        
        print(f"\nデータ取得成功: {stock_data.shape}")
        print(f"利用可能カラム: {list(stock_data.columns)}")
        print(f"日付範囲: {stock_data.index[0]} → {stock_data.index[-1]}")
        
        # BreakoutStrategy初期化
        strategy = BreakoutStrategy(stock_data)
        
        # DSSMS対象日付でテスト
        test_dates = [
            "2025-01-30",  # 1日目 (8233のまま)
            "2025-01-31"   # 2日目 (6723に切替後)
        ]
        
        for test_date in test_dates:
            try:
                target_date = pd.Timestamp(test_date)
                print(f"\n" + "="*60)
                print(f"日次テスト実行: {test_date}")
                print(f"="*60)
                
                # backtest_daily()実行
                result = strategy.backtest_daily(target_date, stock_data)
                
                print(f"[RESULT] action={result.get('action', 'N/A')}, signal={result.get('signal', 0)}")
                print(f"[RESULT] price={result.get('price', 0.0)}, shares={result.get('shares', 0)}")
                print(f"[RESULT] pnl={result.get('pnl', 0.0)}")
                
                # インジケーター値の詳細確認
                if target_date in stock_data.index:
                    idx = stock_data.index.get_loc(target_date)
                    current_price = stock_data['Adj Close'].iloc[idx]
                    volume = stock_data['Volume'].iloc[idx]
                    
                    print(f"\n[DATA_CHECK] {test_date} データ確認:")
                    print(f"  現在価格: {current_price:.2f}")
                    print(f"  ボリューム: {volume:,}")
                    
                    # SMA計算
                    sma_5 = calculate_sma(stock_data, 'Adj Close', 5)
                    sma_25 = calculate_sma(stock_data, 'Adj Close', 25)
                    
                    if idx < len(sma_5) and idx < len(sma_25):
                        sma5_val = sma_5.iloc[idx] if idx < len(sma_5) else "N/A"
                        sma25_val = sma_25.iloc[idx] if idx < len(sma_25) else "N/A"
                        print(f"  SMA5: {sma5_val:.2f}")
                        print(f"  SMA25: {sma25_val:.2f}")
                        
                        # ブレイクアウト条件確認
                        if sma5_val != "N/A" and sma25_val != "N/A":
                            breakout_condition = current_price > sma5_val and sma5_val > sma25_val
                            print(f"  ブレイクアウト条件: {breakout_condition}")
                            print(f"    - 価格 > SMA5: {current_price:.2f} > {sma5_val:.2f} = {current_price > sma5_val}")
                            print(f"    - SMA5 > SMA25: {sma5_val:.2f} > {sma25_val:.2f} = {sma5_val > sma25_val}")
                    
                    # 前日との比較
                    if idx > 0:
                        prev_volume = stock_data['Volume'].iloc[idx-1]
                        volume_ratio = volume / prev_volume if prev_volume > 0 else 0
                        print(f"  前日ボリューム: {prev_volume:,}")
                        print(f"  ボリューム倍率: {volume_ratio:.2f}")
                        print(f"  ボリューム条件(1.2倍以上): {volume_ratio >= 1.2}")
                    
                else:
                    print(f"[WARNING] {test_date} はデータに存在しません")
                    available_dates = stock_data.index[-5:]
                    print(f"利用可能な最新5日: {list(available_dates.strftime('%Y-%m-%d'))}")
                
            except Exception as e:
                print(f"[ERROR] {test_date} テスト実行エラー: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n=== BreakoutStrategy調査完了 ===")
        
    except Exception as e:
        print(f"[ERROR] 調査実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_breakout_backtest_daily()