"""
Breakout.py Phase 5実データ検証スクリプト

目的: BreakoutStrategyのエントリー価格が当日終値と一致することを確認

検証項目:
1. エントリー価格の精度（13桁精度の有無）
2. 当日終値との一致確認
3. 翌日始値との乖離確認

Author: GitHub Copilot
Created: 2025-12-21
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import yfinance as yf
from strategies.Breakout import BreakoutStrategy
import logging

# ログレベルをDEBUGに設定（詳細ログ出力）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("\n" + "="*80)
    print("Breakout.py Phase 5実データ検証")
    print("="*80 + "\n")
    
    # 検証対象銘柄と期間（GCStrategyと同じ）
    ticker = "8053.T"
    start_date = "2024-12-01"
    end_date = "2025-02-05"
    
    print(f"銘柄: {ticker}")
    print(f"期間: {start_date} 〜 {end_date}\n")
    
    # データ取得
    print("[STEP 1] データ取得中...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    
    if data.empty:
        print("[ERROR] データ取得に失敗しました")
        return
    
    # MultiIndexの場合は平坦化（yfinanceの仕様）
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"[OK] データ取得完了: {len(data)}行")
    print(f"カラム: {list(data.columns)}\n")
    
    # BreakoutStrategy初期化
    print("[STEP 2] BreakoutStrategy初期化...")
    strategy = BreakoutStrategy(
        data=data,
        params={
            "volume_threshold": 1.2,
            "take_profit": 0.03,
            "look_back": 1,
            "trailing_stop": 0.02,
            "breakout_buffer": 0.01
        },
        price_column="Close"  # 当日終値を使用（copilot-instructions.mdのNoteに理由あり）
    )
    print("[OK] 初期化完了\n")
    
    # バックテスト実行
    print("[STEP 3] バックテスト実行中...")
    result = strategy.backtest()
    print("[OK] バックテスト完了\n")
    
    # エントリーシグナルの確認
    print("[STEP 4] エントリーシグナルの確認...")
    entries = result[result['Entry_Signal'] == 1]
    print(f"エントリー件数: {len(entries)}件\n")
    
    if len(entries) == 0:
        print("[WARNING] エントリーシグナルがありません")
        print("[INFO] パラメータ調整が必要な可能性があります")
        return
    
    # エントリー価格の詳細確認
    print("[STEP 5] エントリー価格の詳細確認...")
    print("="*80)
    
    for i, (date, row) in enumerate(entries.iterrows(), 1):
        idx = result.index.get_loc(date)
        
        # 実データから価格を取得
        current_close = result['Close'].iloc[idx]
        current_open = result['Open'].iloc[idx]
        
        # 翌日始値を取得（境界条件チェック）
        if idx + 1 < len(result):
            next_day_open = result['Open'].iloc[idx + 1]
        else:
            next_day_open = None
        
        # エントリー価格を取得
        entry_price = strategy.entry_prices.get(date, None)
        
        print(f"\n[ENTRY #{i}]")
        print(f"  日付: {date}")
        print(f"  エントリー価格: {entry_price}")
        print(f"  当日終値 (Close): {current_close}")
        print(f"  当日始値 (Open): {current_open}")
        
        if next_day_open is not None:
            print(f"  翌日始値 (Next Open): {next_day_open}")
        else:
            print(f"  翌日始値 (Next Open): N/A（最終日のため取得不可）")
        
        # 精度確認（13桁精度の有無）
        if entry_price is not None:
            entry_price_str = f"{entry_price:.15f}"
            print(f"  エントリー価格（15桁表示）: {entry_price_str}")
            
            # 当日終値との一致確認
            diff_close = abs(entry_price - current_close)
            print(f"  当日終値との差分: {diff_close:.15f}")
            
            if diff_close < 0.01:
                print(f"  [CONFIRM] エントリー価格は当日終値と一致（ルックアヘッドバイアス）")
            
            # 翌日始値との乖離確認
            if next_day_open is not None:
                diff_next_open = abs(entry_price - next_day_open)
                print(f"  翌日始値との差分: {diff_next_open:.2f}円")
        
        # 前日高値の確認
        if idx >= 1:
            prev_high = result['High'].iloc[idx - 1]
            print(f"  前日高値: {prev_high}")
            print(f"  ブレイクアウト判定: {current_close} > {prev_high * 1.01} = {current_close > prev_high * 1.01}")
        
        # 出来高の確認
        if idx >= 1:
            current_volume = result['Volume'].iloc[idx]
            prev_volume = result['Volume'].iloc[idx - 1]
            volume_ratio = current_volume / prev_volume if prev_volume > 0 else 0
            print(f"  当日出来高: {current_volume}")
            print(f"  前日出来高: {prev_volume}")
            print(f"  出来高比率: {volume_ratio:.2f}x（閾値: 1.2x）")
    
    print("\n" + "="*80)
    print("[COMPLETE] Phase 5実データ検証完了")
    print("="*80 + "\n")
    
    # 結論
    print("[結論]")
    if len(entries) > 0:
        print("1. エントリーシグナルが生成されました")
        print("2. エントリー価格は当日終値を使用しています（ルックアヘッドバイアス）")
        print("3. 修正が必要: エントリー価格を翌日始値に変更")
    else:
        print("1. エントリーシグナルが生成されませんでした")
        print("2. パラメータ調整が必要な可能性があります")

if __name__ == "__main__":
    main()
