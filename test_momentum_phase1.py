"""
Momentum_Investing.py Phase 1修正後の簡易検証

目的: エントリー価格が翌日始値に変更されたことを確認
期間: 2025-01-06〜2025-02-10（短期間）
銘柄: 8053.T
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import yfinance as yf
from strategies.Momentum_Investing import MomentumInvestingStrategy

# データ取得
print("データ取得中...")
symbol = "8053.T"
start_date = "2024-12-01"  # 期間を長くする
end_date = "2025-02-10"
data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)

if data.empty:
    print(f"エラー: {symbol}のデータを取得できませんでした")
    sys.exit(1)

# MultiIndex対策（yfinanceがMultiIndexで返す場合）
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print(f"データ取得完了: {len(data)}行")
print(f"カラム: {list(data.columns)}")
print(f"期間: {data.index[0]} 〜 {data.index[-1]}")

# 戦略初期化
params = {
    "sma_short": 20,
    "sma_long": 50,
    "rsi_period": 14,
    "rsi_lower": 30,  # 緩和: 50→30
    "rsi_upper": 80,  # 緩和: 68→80
    "volume_threshold": 1.0,  # 緩和: 1.18→1.0
    "take_profit": 0.12,
    "stop_loss": 0.06,
    "trailing_stop": 0.04,
}

strategy = MomentumInvestingStrategy(data, params)

# バックテスト実行
print("\nバックテスト実行中...")
try:
    result = strategy.backtest()
    print("バックテスト完了")
except Exception as e:
    print(f"エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# エントリー確認
entries = result[result['Entry_Signal'] == 1]
print(f"\nエントリー件数: {len(entries)}")

if len(entries) > 0:
    print("\n=== エントリー詳細 ===")
    for i, (date, row) in enumerate(entries.iterrows(), 1):
        entry_idx = result.index.get_loc(date)
        
        # 当日データ
        day_close = result['Adj Close'].iloc[entry_idx]
        day_open = result['Open'].iloc[entry_idx] if 'Open' in result.columns else None
        
        # 翌日データ
        if entry_idx + 1 < len(result):
            next_day_open = result['Open'].iloc[entry_idx + 1]
            next_date = result.index[entry_idx + 1]
        else:
            next_day_open = None
            next_date = None
        
        # エントリー価格
        entry_price = strategy.entry_prices.get(entry_idx, None)
        
        print(f"\nENTRY #{i} ({date.date()}):")
        print(f"  当日終値（Adj Close）: {day_close:.2f}円")
        if day_open:
            print(f"  当日始値（Open）     : {day_open:.2f}円")
        if next_day_open:
            print(f"  翌日始値（Open）     : {next_day_open:.2f}円 ({next_date.date()})")
        if entry_price:
            print(f"  エントリー価格        : {entry_price:.2f}円")
            
            # 比較
            if next_day_open:
                diff = entry_price - next_day_open
                print(f"  翌日始値との差分      : {diff:.2f}円 ({abs(diff):.10f}円)")
                
                if abs(diff) < 0.01:
                    print(f"  検証結果: ✅ エントリー価格は翌日始値と一致（差分0.01円未満）")
                else:
                    print(f"  検証結果: ❌ エントリー価格が翌日始値と不一致（差分{abs(diff):.2f}円）")
        else:
            print(f"  エントリー価格        : 記録なし（エラー）")
else:
    print("エントリーなし")

print("\n検証完了")
