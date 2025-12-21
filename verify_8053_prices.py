"""
8053銘柄の実データ検証スクリプト

GCStrategy調査: エントリー価格3362.07円（2025-01-30）が当日終値か翌日始値かを検証

Author: GitHub Copilot
Created: 2025-12-21
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def verify_8053_entry_price():
    """
    8053銘柄の2025-01-30エントリー価格を検証
    """
    print("=" * 80)
    print("8053銘柄 実データ検証")
    print("=" * 80)
    
    # データ取得期間（エントリー日の前後3日間）
    start_date = "2025-01-27"
    end_date = "2025-02-03"
    ticker = "8053.T"
    
    print(f"\n[1] データ取得")
    print(f"   銘柄: {ticker}")
    print(f"   期間: {start_date} ~ {end_date}")
    print(f"   yfinance auto_adjust=False（Adj Close取得）")
    
    try:
        # yfinance でデータ取得（auto_adjust=False必須）
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        
        if data.empty:
            print(f"   エラー: データが取得できませんでした")
            return
        
        # MultiIndexの場合はシングルインデックスに変換
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        print(f"   取得成功: {len(data)}行")
        print(f"   カラム: {list(data.columns)}")
        
        # データ表示
        print(f"\n[2] 価格データ（全期間）")
        print("-" * 80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.precision', 2)
        print(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
        print("-" * 80)
        
        # エントリー日（2025-01-30）の分析
        entry_date = "2025-01-30"
        next_date = "2025-01-31"
        
        print(f"\n[3] エントリー日の詳細分析")
        print("-" * 80)
        
        if entry_date in data.index.astype(str):
            entry_day = data.loc[entry_date]
            print(f"エントリー日: {entry_date}")
            print(f"  Open (始値):      {entry_day['Open']:.2f}円")
            print(f"  High (高値):      {entry_day['High']:.2f}円")
            print(f"  Low (安値):       {entry_day['Low']:.2f}円")
            print(f"  Close (終値):     {entry_day['Close']:.2f}円")
            print(f"  Adj Close (調整後終値): {entry_day['Adj Close']:.2f}円")
            print(f"  Volume (出来高):  {entry_day['Volume']:,.0f}")
        else:
            print(f"エラー: {entry_date}のデータが見つかりません")
        
        if next_date in data.index.astype(str):
            next_day = data.loc[next_date]
            print(f"\n翌日: {next_date}")
            print(f"  Open (始値):      {next_day['Open']:.2f}円")
            print(f"  High (高値):      {next_day['High']:.2f}円")
            print(f"  Low (安値):       {next_day['Low']:.2f}円")
            print(f"  Close (終値):     {next_day['Close']:.2f}円")
            print(f"  Adj Close (調整後終値): {next_day['Adj Close']:.2f}円")
            print(f"  Volume (出来高):  {next_day['Volume']:,.0f}")
        else:
            print(f"エラー: {next_date}のデータが見つかりません")
        
        print("-" * 80)
        
        # エントリー価格との比較
        entry_price_from_backtest = 3362.07
        print(f"\n[4] エントリー価格の検証")
        print("-" * 80)
        print(f"バックテスト結果のエントリー価格: {entry_price_from_backtest:.2f}円")
        print()
        
        if entry_date in data.index.astype(str):
            entry_day = data.loc[entry_date]
            diff_open = abs(entry_day['Open'] - entry_price_from_backtest)
            diff_close = abs(entry_day['Close'] - entry_price_from_backtest)
            diff_adj_close = abs(entry_day['Adj Close'] - entry_price_from_backtest)
            
            print(f"【2025-01-30（エントリー日）との比較】")
            print(f"  vs Open:      差額 {diff_open:.2f}円 ({diff_open/entry_day['Open']*100:.3f}%)")
            print(f"  vs Close:     差額 {diff_close:.2f}円 ({diff_close/entry_day['Close']*100:.3f}%)")
            print(f"  vs Adj Close: 差額 {diff_adj_close:.2f}円 ({diff_adj_close/entry_day['Adj Close']*100:.3f}%)")
        
        if next_date in data.index.astype(str):
            next_day = data.loc[next_date]
            diff_next_open = abs(next_day['Open'] - entry_price_from_backtest)
            diff_next_close = abs(next_day['Close'] - entry_price_from_backtest)
            diff_next_adj_close = abs(next_day['Adj Close'] - entry_price_from_backtest)
            
            print(f"\n【2025-01-31（翌日）との比較】")
            print(f"  vs Open:      差額 {diff_next_open:.2f}円 ({diff_next_open/next_day['Open']*100:.3f}%)")
            print(f"  vs Close:     差額 {diff_next_close:.2f}円 ({diff_next_close/next_day['Close']*100:.3f}%)")
            print(f"  vs Adj Close: 差額 {diff_next_adj_close:.2f}円 ({diff_next_adj_close/next_day['Adj Close']*100:.3f}%)")
        
        print("-" * 80)
        
        # 結論
        print(f"\n[5] 結論")
        print("-" * 80)
        
        if entry_date in data.index.astype(str) and next_date in data.index.astype(str):
            entry_day = data.loc[entry_date]
            next_day = data.loc[next_date]
            
            # 最も近い価格を特定
            prices = {
                f"{entry_date} Open": (entry_day['Open'], abs(entry_day['Open'] - entry_price_from_backtest)),
                f"{entry_date} Close": (entry_day['Close'], abs(entry_day['Close'] - entry_price_from_backtest)),
                f"{entry_date} Adj Close": (entry_day['Adj Close'], abs(entry_day['Adj Close'] - entry_price_from_backtest)),
                f"{next_date} Open": (next_day['Open'], abs(next_day['Open'] - entry_price_from_backtest)),
                f"{next_date} Close": (next_day['Close'], abs(next_day['Close'] - entry_price_from_backtest)),
                f"{next_date} Adj Close": (next_day['Adj Close'], abs(next_day['Adj Close'] - entry_price_from_backtest)),
            }
            
            closest = min(prices.items(), key=lambda x: x[1][1])
            
            print(f"エントリー価格 {entry_price_from_backtest:.2f}円 に最も近い価格:")
            print(f"  {closest[0]}: {closest[1][0]:.2f}円（差額 {closest[1][1]:.2f}円）")
            print()
            
            if "2025-01-30" in closest[0] and ("Close" in closest[0] or "Adj Close" in closest[0]):
                print("判定: 当日終値を使用（ルックアヘッドバイアスあり）")
            elif "2025-01-31 Open" in closest[0]:
                print("判定: 翌日始値を使用（ルックアヘッドバイアスなし）")
            else:
                print("判定: 不明（追加調査が必要）")
        
        print("-" * 80)
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_8053_entry_price()
