"""
ContrararianStrategy Phase 1修正後の検証スクリプト

検証内容:
- エントリー価格が翌日始値に変更されているか
- 最終日のエントリーシグナルが0か
- ループ範囲が最終日を除外しているか
"""

import pandas as pd
import numpy as np
from strategies.contrarian_strategy import ContrarianStrategy

# ダミーデータの作成（60日分）
np.random.seed(42)
dates = pd.date_range(start="2025-01-01", periods=60, freq='B')

# 価格データの生成（トレンド + ノイズ）
base_price = 3000
prices = []
for i in range(60):
    # トレンド（上昇）+ ランダムノイズ
    trend = base_price + i * 5
    noise = np.random.randn() * 50
    prices.append(max(trend + noise, 100))  # 最低100円

# OHLC生成
data = []
for i, close in enumerate(prices):
    open_price = close * (1 + np.random.uniform(-0.02, 0.02))
    high = max(open_price, close) * (1 + abs(np.random.uniform(0, 0.02)))
    low = min(open_price, close) * (1 - abs(np.random.uniform(0, 0.02)))
    data.append({
        'Open': round(open_price, 2),
        'High': round(high, 2),
        'Low': round(low, 2),
        'Close': round(close, 2),
        'Adj Close': round(close, 2),
        'Volume': int(np.random.uniform(1000000, 5000000))
    })

df = pd.DataFrame(data, index=dates)

print("=" * 80)
print("ContrararianStrategy Phase 1修正検証")
print("=" * 80)
print(f"\nデータ期間: {df.index[0]} 〜 {df.index[-1]}")
print(f"データ件数: {len(df)}件")
print(f"\n価格データサンプル（最初の5件）:")
print(df[['Open', 'High', 'Low', 'Close', 'Adj Close']].head())

# パラメータ緩和（エントリーしやすく）
params = {
    "rsi_period": 5,        # RSI期間短縮
    "rsi_oversold": 50,     # RSI閾値大幅緩和（30→50）
    "gap_threshold": 0.005,  # ギャップ閾値大幅緩和（2%→0.5%）
    "stop_loss": 0.05,
    "take_profit": 0.08,
    "pin_bar_ratio": 1.5,   # ピンバー判定緩和（2.0→1.5）
    "max_hold_days": 10,
    "rsi_exit_level": 60,   # RSIエグジット緩和（50→60）
    "trailing_stop_pct": 0.03,
    "trend_filter_enabled": False  # トレンドフィルター無効化
}

print(f"\n使用パラメータ: rsi_period={params['rsi_period']}, rsi_oversold={params['rsi_oversold']}")
print(f"gap_threshold={params['gap_threshold']}, trend_filter_enabled={params['trend_filter_enabled']}")

# 戦略実行
strategy = ContrarianStrategy(df, params=params, price_column='Adj Close')
result = strategy.backtest()

# エントリーシグナル確認
entry_signals = result[result['Entry_Signal'] == 1]
print(f"\n--- エントリーシグナル確認 ---")
print(f"エントリー発生件数: {len(entry_signals)}件")
if len(entry_signals) > 0:
    print(f"エントリー日付: {entry_signals.index.tolist()}")
    print(f"最初のエントリー: {entry_signals.index[0]}")
    print(f"最後のエントリー: {entry_signals.index[-1]}")

# 最終日のエントリーシグナル確認
final_day_signal = result['Entry_Signal'].iloc[-1]
print(f"\n最終日のエントリーシグナル: {final_day_signal} (期待値: 0)")
if final_day_signal == 0:
    print("OK: 最終日のエントリーシグナルは0（ループ範囲修正適用済み）")
else:
    print("NG: 最終日のエントリーシグナルが1（ループ範囲修正が未適用）")

# entry_pricesの確認
print(f"\n--- entry_prices辞書確認 ---")
print(f"entry_pricesの件数: {len(strategy.entry_prices)}件")

# エントリー価格と翌日始値の比較（最初の5件）
print(f"\n--- エントリー価格と翌日始値の比較（最初の5件） ---")
comparison_count = 0
for idx, signal in enumerate(result['Entry_Signal']):
    if signal == 1 and comparison_count < 5:
        entry_price = strategy.entry_prices.get(idx)
        if entry_price is not None and idx + 1 < len(result):
            next_day_open = result['Open'].iloc[idx + 1]
            diff = entry_price - next_day_open
            print(f"idx={idx}, entry_price={entry_price:.2f}, next_day_open={next_day_open:.2f}, diff={diff:.2f}")
            comparison_count += 1

print("\n" + "=" * 80)
print("検証完了")
print("=" * 80)
