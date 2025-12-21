"""
GCStrategy エントリー価格調査用スクリプト

8053銘柄で小規模バックテストを実行し、base_strategy.pyのデバッグログを確認

Author: GitHub Copilot
Created: 2025-12-21
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import yfinance as yf
from strategies.gc_strategy_signal import GCStrategy
import logging

# ログレベルをDEBUGに設定（デバッグログ表示）
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_gc_entry_price():
    """
    8053銘柄でGCStrategyのバックテストを実行し、エントリー価格を検証
    """
    print("=" * 80)
    print("GCStrategy エントリー価格調査")
    print("=" * 80)
    
    # データ取得
    ticker = "8053.T"
    start_date = "2024-12-01"  # 長期移動平均（25日）計算のため期間を大幅拡大
    end_date = "2025-02-05"
    
    print(f"\n[1] データ取得")
    print(f"   銘柄: {ticker}")
    print(f"   期間: {start_date} ~ {end_date}")
    
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    
    if data.empty:
        print("   エラー: データ取得失敗")
        return
    
    # MultiIndexの場合はシングルインデックスに変換
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    print(f"   取得成功: {len(data)}行")
    print(f"   カラム: {list(data.columns)}")
    
    # データ確認
    print(f"\n[2] 取得データ（全期間）")
    print("-" * 80)
    print(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
    print("-" * 80)
    
    # GCStrategyでバックテスト実行
    print(f"\n[3] GCStrategyバックテスト実行")
    print("-" * 80)
    
    params = {
        "short_window": 5,
        "long_window": 25,
        "take_profit": 0.05,
        "stop_loss": 0.03,
        "trailing_stop_pct": 0.03,
        "max_hold_days": 20,
        "exit_on_death_cross": True
    }
    
    strategy = GCStrategy(data, params=params, price_column="Adj Close", ticker="8053")
    result = strategy.backtest()
    
    print("-" * 80)
    
    # エントリーシグナルの確認
    entries = result[result['Entry_Signal'] == 1]
    
    print(f"\n[4] エントリーシグナル検出結果")
    print("-" * 80)
    print(f"検出件数: {len(entries)}件")
    
    if len(entries) > 0:
        print("\nエントリー詳細:")
        for idx, row in entries.iterrows():
            print(f"  日付: {idx}")
            print(f"  Open:      {row.get('Open', 'N/A')}")
            print(f"  Close:     {row.get('Close', 'N/A')}")
            print(f"  Adj Close: {row.get('Adj Close', 'N/A')}")
            
            # entry_prices辞書から記録されたエントリー価格を取得
            entry_idx_in_result = result.index.get_loc(idx)
            if entry_idx_in_result in strategy.entry_prices:
                recorded_price = strategy.entry_prices[entry_idx_in_result]
                print(f"  Recorded Entry Price: {recorded_price:.2f}")
            print()
    else:
        print("エントリーシグナルが検出されませんでした")
    
    print("-" * 80)
    
    # 結果サマリー
    print(f"\n[5] 結果サマリー")
    print("-" * 80)
    print(f"全行数: {len(result)}")
    print(f"エントリー: {(result['Entry_Signal'] == 1).sum()}件")
    print(f"イグジット: {(result['Exit_Signal'] == -1).sum()}件")
    print("-" * 80)

if __name__ == "__main__":
    test_gc_entry_price()
