"""
検証スクリプト: MeanReversionStrategy Phase 1修正後の検証
- エントリー価格が翌日始値と一致するか確認
- 最終日のエントリーシグナルが0か確認（ループ範囲修正の確認）
- ダミーデータで平均回帰パターンを生成
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from strategies.mean_reversion_strategy import MeanReversionStrategy
from datetime import datetime, timedelta

# ダミーデータ生成（平均回帰パターン）
def generate_mean_reversion_data(n_days=60):
    """平均回帰パターンのダミーデータ生成"""
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    np.random.seed(42)
    base_price = 100
    mean_price = 100
    
    prices = []
    for i in range(n_days):
        if i == 0:
            price = base_price
        else:
            prev_price = prices[-1]
            # 平均回帰力（価格が100から離れるほど強く戻る力）
            revert_force = (mean_price - prev_price) * 0.08
            
            # ランダムショック
            shock = np.random.normal(0, 2.0)
            
            # 時々大きな逸脱（平均回帰の機会）
            if i % 15 == 10:
                shock += np.random.choice([-10, 10])  # 大きな逸脱
                
            price = prev_price + revert_force + shock
            price = max(80, min(120, price))  # 範囲制限
            
        prices.append(price)
    
    # OHLCV データ作成
    data = pd.DataFrame({
        'Open': [p * np.random.uniform(0.998, 1.002) for p in prices],
        'High': [p * np.random.uniform(1.005, 1.025) for p in prices],
        'Low': [p * np.random.uniform(0.975, 0.995) for p in prices],
        'Adj Close': prices,
        'Close': prices,
        'Volume': np.random.randint(100000, 500000, n_days)
    })
    data.index = pd.DatetimeIndex(dates)
    
    return data

# テスト実行
def main():
    print("=" * 60)
    print("MeanReversionStrategy Phase 1 Verification")
    print("=" * 60)
    
    # ダミーデータ生成
    data = generate_mean_reversion_data(n_days=60)
    print(f"Generated dummy data: {len(data)} days")
    print(f"Price range: {data['Adj Close'].min():.2f} - {data['Adj Close'].max():.2f}")
    print()
    
    # パラメータ緩和（エントリー発生しやすくする）
    params = {
        "sma_period": 15,
        "bb_period": 15,
        "bb_std_dev": 1.5,  # 緩和（デフォルト2.0）
        "zscore_period": 10,
        "zscore_entry_threshold": -1.2,  # 緩和（デフォルト-1.8）
        "zscore_exit_threshold": -0.2,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.08,
        "volume_confirmation": False,  # 無効化
        "rsi_filter": False,  # 無効化
        "atr_filter": False,  # 無効化
        "max_hold_days": 20,
    }
    
    # 戦略初期化・バックテスト実行
    strategy = MeanReversionStrategy(data, params)
    result = strategy.backtest()
    
    print("Backtest completed")
    print()
    
    # エントリーシグナル確認
    entry_signals = result[result['Entry_Signal'] == 1]
    num_entries = len(entry_signals)
    
    print(f"Entry signals: {num_entries}")
    print()
    
    if num_entries == 0:
        print("WARNING: No entry signals generated. Consider relaxing parameters.")
        print("- Current zscore_entry_threshold: -1.2")
        print("- Current bb_std_dev: 1.5")
        print()
        
        # Z-score範囲確認
        print("Z-Score statistics:")
        print(f"- Min: {result['Z_Score'].min():.2f}")
        print(f"- Max: {result['Z_Score'].max():.2f}")
        print(f"- Mean: {result['Z_Score'].mean():.2f}")
        print()
        return
    
    # Phase 1修正の検証
    print("-" * 60)
    print("Phase 1 Verification: Entry Price Check")
    print("-" * 60)
    
    all_match = True
    for idx in entry_signals.index:
        # idxの位置を取得
        idx_pos = result.index.get_loc(idx)
        
        # strategy.entry_pricesから取得
        if idx_pos in strategy.entry_prices:
            entry_price = strategy.entry_prices[idx_pos]
            
            # 翌日始値を取得（idx_pos + 1）
            if idx_pos + 1 < len(result):
                next_day_open = result['Open'].iloc[idx_pos + 1]
                
                # 差分計算
                diff = abs(entry_price - next_day_open)
                diff_pct = (diff / next_day_open) * 100 if next_day_open != 0 else 0
                
                match_status = "OK: Match" if diff < 0.01 else "ERROR: Mismatch"
                if diff >= 0.01:
                    all_match = False
                
                print(f"Date: {idx.strftime('%Y-%m-%d')}")
                print(f"  Entry Price: {entry_price:.2f}")
                print(f"  Next Day Open: {next_day_open:.2f}")
                print(f"  Diff: {diff:.2f} ({diff_pct:.4f}%)")
                print(f"  Status: {match_status}")
                print()
    
    print("-" * 60)
    print("Final Date Entry Signal Check (Loop Range Verification)")
    print("-" * 60)
    
    # 最終日のエントリーシグナル確認
    final_date = result.index[-1]
    final_entry_signal = result['Entry_Signal'].iloc[-1]
    
    print(f"Final date: {final_date.strftime('%Y-%m-%d')}")
    print(f"Final entry signal: {final_entry_signal}")
    
    if final_entry_signal == 0:
        print("Status: OK: No entry on final day (loop range correctly modified)")
    else:
        print("Status: ERROR: Entry signal on final day (loop range issue)")
        all_match = False
    
    print()
    print("=" * 60)
    print("Verification Result")
    print("=" * 60)
    
    if all_match:
        print("SUCCESS: All entry prices match next day open prices")
        print("SUCCESS: No entry signal on final day")
        print("Phase 1 modification is correct!")
    else:
        print("FAILURE: Some verification checks failed")
        print("Please review the modification")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
