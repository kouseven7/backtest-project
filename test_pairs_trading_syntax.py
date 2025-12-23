"""
PairsTradingStrategy Phase 1修正後の検証スクリプト

検証項目:
1. 構文エラーがないことを確認
2. バックテスト実行が成功することを確認
3. エントリー価格が翌日始値と一致することを確認
4. 最終日のエントリーシグナルが0であることを確認

Author: GitHub Copilot
Created: 2025-12-22
"""

import pandas as pd
import numpy as np
from strategies.pairs_trading_strategy import PairsTradingStrategy

def create_dummy_data(n_days=60):
    """ダミーデータ生成（エントリーシグナル発生しやすいパラメータ）"""
    dates = pd.date_range(start='2025-01-01', end='2025-03-01', freq='D')[:n_days]
    
    np.random.seed(456)
    base_price = 100
    
    # 移動平均間の乖離を意図的に作るデータパターン
    prices = []
    for i in range(n_days):
        if i == 0:
            price = base_price
        else:
            prev_price = prices[-1]
            
            # 基本トレンド
            base_change = np.random.normal(0, 0.01)
            
            # 周期的な乖離パターン（ペアトレーディングの機会）
            if i % 15 in [8, 9, 10]:  # 15日ごとに3日間の大きな乖離
                divergence = np.random.choice([-0.03, 0.03])  # ±3%の乖離
            else:
                divergence = 0
                
            # 価格更新
            total_change = base_change + divergence
            price = prev_price * (1 + total_change)
            price = max(85, min(115, price))  # 範囲制限
            
        prices.append(price)
    
    # OHLCV データ作成
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
        'High': [p * np.random.uniform(1.002, 1.015) for p in prices],
        'Low': [p * np.random.uniform(0.985, 0.998) for p in prices],
        'Adj Close': prices,
        'Volume': np.random.randint(100000, 300000, len(dates))
    })
    
    test_data['Close'] = test_data['Adj Close']
    test_data.set_index('Date', inplace=True)
    
    return test_data

def verify_entry_prices(strategy, result):
    """エントリー価格検証"""
    entry_indices = result[result['Entry_Signal'] == 1].index
    
    if len(entry_indices) == 0:
        print("No entry signals generated.")
        return True
    
    print(f"\nEntry price verification (Total entries: {len(entry_indices)}):")
    print("-" * 80)
    
    all_match = True
    for idx in entry_indices:
        # エントリーインデックスを取得
        entry_idx = result.index.get_loc(idx)
        
        # エントリー価格を取得
        if entry_idx not in strategy.entry_prices:
            print(f"ERROR: Entry price not found for index {entry_idx} ({idx})")
            all_match = False
            continue
            
        entry_price = strategy.entry_prices[entry_idx]
        
        # 翌日始値を取得
        if entry_idx + 1 >= len(result):
            print(f"ERROR: Next day data not available for index {entry_idx} ({idx})")
            all_match = False
            continue
            
        next_day_open = result['Open'].iloc[entry_idx + 1]
        
        # 差分計算
        diff = entry_price - next_day_open
        diff_pct = (diff / next_day_open) * 100
        
        print(f"Date: {idx.strftime('%Y-%m-%d')}, Entry={entry_price:.2f}, NextOpen={next_day_open:.2f}, "
              f"Diff={diff:.2f} ({diff_pct:.4f}%)")
        
        if abs(diff) > 0.01:  # 0.01円以上の差があればNG
            all_match = False
    
    print("-" * 80)
    if all_match:
        print("SUCCESS: All entry prices match next day open prices")
    else:
        print("FAILURE: Some entry prices do not match next day open prices")
    
    return all_match

def main():
    print("=" * 80)
    print("PairsTradingStrategy Phase 1 Verification Test")
    print("=" * 80)
    
    # ダミーデータ生成
    print("\n[Step 1] Creating dummy data...")
    data = create_dummy_data(n_days=60)
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # 戦略パラメータ（エントリーシグナル発生しやすい設定）
    params = {
        "short_ma_period": 5,
        "long_ma_period": 15,
        "spread_period": 10,
        "entry_threshold": 1.5,  # 閾値を下げてエントリーしやすく
        "exit_threshold": 0.3,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.08,
        "volume_filter": False,  # ボリュームフィルター無効
        "volatility_filter": False,  # ボラティリティフィルター無効
        "correlation_min": 0.5,  # 相関閾値を下げる
        "max_hold_days": 20
    }
    
    # 戦略インスタンス作成
    print("\n[Step 2] Initializing strategy...")
    strategy = PairsTradingStrategy(data, params)
    
    # バックテスト実行
    print("\n[Step 3] Running backtest...")
    try:
        result = strategy.backtest()
        print(f"Backtest completed successfully")
        print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"ERROR: Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # エントリーシグナル確認
    print("\n[Step 4] Checking entry signals...")
    entry_count = (result['Entry_Signal'] == 1).sum()
    exit_count = (result['Exit_Signal'] == 1).sum()
    print(f"Entry signals: {entry_count}")
    print(f"Exit signals: {exit_count}")
    
    if entry_count == 0:
        print("WARNING: No entry signals generated. Consider adjusting parameters.")
    
    # 最終日のエントリーシグナル確認
    print("\n[Step 5] Checking last day entry signal...")
    last_day_entry = result['Entry_Signal'].iloc[-1]
    print(f"Last day entry signal: {last_day_entry}")
    
    if last_day_entry != 0:
        print("ERROR: Last day should not have entry signal (IndexError risk)")
        return False
    else:
        print("SUCCESS: Last day has no entry signal")
    
    # エントリー価格検証
    print("\n[Step 6] Verifying entry prices...")
    price_match = verify_entry_prices(strategy, result)
    
    # スプレッドZスコア確認
    if 'Spread_ZScore' in result.columns:
        print("\n[Step 7] Checking Spread Z-Score range...")
        zscore_min = result['Spread_ZScore'].min()
        zscore_max = result['Spread_ZScore'].max()
        print(f"Spread Z-Score range: {zscore_min:.2f} to {zscore_max:.2f}")
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Backtest execution: SUCCESS")
    print(f"Entry signals: {entry_count}")
    print(f"Exit signals: {exit_count}")
    print(f"Last day entry signal: {last_day_entry} (Expected: 0)")
    print(f"Entry price verification: {'PASS' if price_match else 'FAIL'}")
    print("=" * 80)
    
    if price_match and last_day_entry == 0:
        print("\nOVERALL: PASS - Phase 1 modification is working correctly")
        return True
    else:
        print("\nOVERALL: FAIL - Phase 1 modification has issues")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
