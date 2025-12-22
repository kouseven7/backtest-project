"""
Momentum_Investing.py Phase 1修正後の構文チェック

目的: Phase 1修正が構文エラーなく動作することを確認
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import numpy as np
from strategies.Momentum_Investing import MomentumInvestingStrategy

# ダミーデータ作成（60日分）
dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
dummy_data = pd.DataFrame({
    'Open': np.random.uniform(3000, 3500, 60),
    'High': np.random.uniform(3100, 3600, 60),
    'Low': np.random.uniform(2900, 3400, 60),
    'Close': np.random.uniform(3000, 3500, 60),
    'Adj Close': np.random.uniform(3000, 3500, 60),
    'Volume': np.random.uniform(1000000, 5000000, 60)
}, index=dates)

print("ダミーデータ作成完了")
print(f"期間: {dummy_data.index[0]} 〜 {dummy_data.index[-1]}")
print(f"行数: {len(dummy_data)}")

# 戦略初期化
params = {
    "sma_short": 5,  # 短縮して計算可能に
    "sma_long": 10,  # 短縮して計算可能に
    "rsi_period": 5,  # 短縮して計算可能に
    "rsi_lower": 30,
    "rsi_upper": 80,
    "volume_threshold": 1.0,
    "take_profit": 0.12,
    "stop_loss": 0.06,
    "trailing_stop": 0.04,
}

print("\n戦略初期化中...")
try:
    strategy = MomentumInvestingStrategy(dummy_data, params)
    print("戦略初期化成功")
except Exception as e:
    print(f"戦略初期化エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# バックテスト実行
print("\nバックテスト実行中...")
try:
    result = strategy.backtest()
    print("バックテスト実行成功")
except Exception as e:
    print(f"バックテストエラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# エントリー確認
entries = result[result['Entry_Signal'] == 1]
print(f"\nエントリー件数: {len(entries)}")

# ループ範囲確認
print(f"データ総行数: {len(result)}")
print(f"最終日: {result.index[-1]}")

# 最終日にエントリーシグナルがあるか確認
last_day_entry = result.iloc[-1]['Entry_Signal']
print(f"最終日のエントリーシグナル: {last_day_entry}")

if last_day_entry == 1:
    print("[WARNING] 最終日にエントリーシグナルが発生しています（ループ範囲修正が未適用の可能性）")
else:
    print("[OK] 正常: 最終日にエントリーシグナルは発生していません（ループ範囲修正が適用済み）")

# entry_pricesの確認
print(f"\nentry_pricesの件数: {len(strategy.entry_prices)}")
if len(strategy.entry_prices) > 0:
    print("entry_pricesの内容:")
    for idx, price in list(strategy.entry_prices.items())[:5]:  # 最初の5件のみ表示
        date = result.index[idx]
        day_open = result['Open'].iloc[idx]
        next_idx = idx + 1
        if next_idx < len(result):
            next_day_open = result['Open'].iloc[next_idx]
            diff = price - next_day_open
            print(f"  idx={idx}, date={date.date()}, entry_price={price:.2f}, next_day_open={next_day_open:.2f}, diff={diff:.2f}")
        else:
            print(f"  idx={idx}, date={date.date()}, entry_price={price:.2f} (最終日)")

print("\n[OK] Phase 1修正の構文チェック完了（エラーなし）")
