"""
GC戦略のエグジット条件分析スクリプト

DSSMSで8053がエントリー後にエグジットしない原因を調査。

Analysis Points:
1. 2024-12-30エントリー後の各日でのエグジット条件評価
2. デッドクロス条件の確認
3. トレーリングストップ条件の確認
4. 損切り条件の確認
5. 最大保有期間条件の確認

Author: Backtest Project Team
Created: 2026-01-29
"""
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import numpy as np
from datetime import datetime
from strategies.gc_strategy_signal import GCStrategy
from data_fetcher import get_parameters_and_data

def analyze_exit_conditions():
    """エグジット条件分析"""
    
    print("=" * 80)
    print("GC戦略エグジット条件分析")
    print("=" * 80)
    
    # 8053のデータ取得
    ticker = "8053.T"
    start_date = "2024-01-01"
    end_date = "2025-12-30"
    
    print(f"\n[INFO] データ取得: {ticker}, {start_date} ~ {end_date}")
    ticker, start, end, stock_data, index_data = get_parameters_and_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        warmup_days=150
    )
    
    print(f"[INFO] データ取得完了: {len(stock_data)}行")
    print(f"[INFO] データ範囲: {stock_data.index[0]} ~ {stock_data.index[-1]}")
    
    # GC戦略初期化
    params = {
        "short_window": 5,
        "long_window": 25,
        "take_profit": None,  # 利益確定なし
        "stop_loss": 0.03,    # 3%損切り
        "trailing_stop_pct": 0.10,  # 10%トレーリングストップ
        "max_hold_days": 300,  # 最大保有期間300日
        "exit_on_death_cross": True,
        
        # フィルター無効化（エントリー条件の影響除外）
        "use_entry_filter": False,
        "trend_filter_enabled": False
    }
    
    strategy = GCStrategy(stock_data, params=params)
    
    print(f"\n[INFO] GC戦略初期化完了")
    print(f"[INFO] パラメータ: {params}")
    
    # エントリー日: 2024-12-30
    entry_date_str = "2024-12-30"
    entry_date = pd.Timestamp(entry_date_str)
    
    if entry_date not in stock_data.index:
        print(f"\n[ERROR] エントリー日 {entry_date_str} がデータに存在しません")
        return
    
    entry_idx = stock_data.index.get_loc(entry_date)
    
    # エントリー価格取得（翌日始値）
    if entry_idx + 1 < len(stock_data):
        entry_price = stock_data['Open'].iloc[entry_idx + 1]
    else:
        entry_price = stock_data['Close'].iloc[entry_idx]
    
    print(f"\n[INFO] エントリー情報:")
    print(f"  日付: {entry_date_str}")
    print(f"  インデックス: {entry_idx}")
    print(f"  エントリー価格: {entry_price:.2f}円")
    
    # entry_pricesを設定（generate_exit_signalが依存）
    strategy.entry_prices[entry_idx] = entry_price
    
    # エントリー後の全日付でエグジット条件を評価
    print(f"\n{'='*80}")
    print("エグジット条件評価（エントリー後の各営業日）")
    print(f"{'='*80}")
    print(f"{'日付':<12} {'idx':>4} {'終値':>8} {'SMA5':>8} {'SMA25':>8} {'デッドクロス':>10} {'トレーリング':>10} {'損切り':>10} {'保有期間':>10} {'結果':<20}")
    print("-" * 140)
    
    exit_found = False
    exit_date = None
    exit_reason = None
    
    for i in range(entry_idx + 1, len(stock_data)):
        date = stock_data.index[i]
        
        # エグジット価格（翌日始値）
        if i + 1 < len(stock_data):
            current_price = stock_data['Open'].iloc[i + 1]
        else:
            current_price = stock_data['Close'].iloc[i]
        
        sma5 = stock_data[f'SMA_5'].iloc[i]
        sma25 = stock_data[f'SMA_25'].iloc[i]
        
        # 各条件のチェック
        # 1. デッドクロス
        prev_sma5 = stock_data[f'SMA_5'].iloc[i-1]
        prev_sma25 = stock_data[f'SMA_25'].iloc[i-1]
        death_cross = (prev_sma5 >= prev_sma25 and sma5 < sma25)
        
        # 2. トレーリングストップ
        if entry_idx not in strategy.high_prices:
            strategy.high_prices[entry_idx] = entry_price
        else:
            strategy.high_prices[entry_idx] = max(strategy.high_prices[entry_idx], current_price)
        
        trailing_stop = strategy.high_prices[entry_idx] * (1 - params['trailing_stop_pct'])
        trailing_hit = current_price < trailing_stop
        
        # 3. 損切り
        stop_loss_price = entry_price * (1 - params['stop_loss'])
        stop_loss_hit = current_price <= stop_loss_price
        
        # 4. 最大保有期間
        days_held = i - entry_idx
        max_hold_hit = days_held >= params['max_hold_days']
        
        # generate_exit_signal呼び出し
        exit_signal, reason = strategy.generate_exit_signal(i, entry_idx)
        
        result_str = "ホールド"
        if exit_signal == -1:
            result_str = f"エグジット ({reason})"
            exit_found = True
            exit_date = date
            exit_reason = reason
        
        print(f"{date.strftime('%Y-%m-%d'):<12} {i:>4} {current_price:>8.2f} {sma5:>8.2f} {sma25:>8.2f} "
              f"{'YES' if death_cross else 'NO':>10} "
              f"{'YES' if trailing_hit else 'NO':>10} "
              f"{'YES' if stop_loss_hit else 'NO':>10} "
              f"{days_held:>10}日 {result_str:<20}")
        
        if exit_found:
            break
    
    # 結果サマリー
    print(f"\n{'='*80}")
    print("分析結果サマリー")
    print(f"{'='*80}")
    
    if exit_found:
        print(f"[結果] エグジットシグナル発生: {exit_date.strftime('%Y-%m-%d')}")
        print(f"[理由] {exit_reason}")
    else:
        print(f"[結果] エグジットシグナル未発生")
        print(f"[分析] エントリー後の全営業日（{len(stock_data) - entry_idx - 1}日）でエグジット条件が満たされませんでした")
        
        # 最終日の状態
        final_idx = len(stock_data) - 1
        final_date = stock_data.index[final_idx]
        final_price = stock_data['Close'].iloc[final_idx]
        final_sma5 = stock_data[f'SMA_5'].iloc[final_idx]
        final_sma25 = stock_data[f'SMA_25'].iloc[final_idx]
        
        print(f"\n[最終日状態] {final_date.strftime('%Y-%m-%d')}")
        print(f"  終値: {final_price:.2f}円")
        print(f"  SMA5: {final_sma5:.2f}円")
        print(f"  SMA25: {final_sma25:.2f}円")
        print(f"  エントリー価格: {entry_price:.2f}円")
        print(f"  含み益: {((final_price - entry_price) / entry_price * 100):.2f}%")
        print(f"  保有期間: {final_idx - entry_idx}日")
        
        # なぜエグジットしなかったかの分析
        print(f"\n[分析] エグジットしない理由:")
        
        if final_sma5 >= final_sma25:
            print(f"  - デッドクロス未発生: SMA5 ({final_sma5:.2f}) >= SMA25 ({final_sma25:.2f})")
        
        high_price = strategy.high_prices.get(entry_idx, entry_price)
        trailing_threshold = high_price * (1 - params['trailing_stop_pct'])
        if final_price >= trailing_threshold:
            print(f"  - トレーリングストップ未達: 現在価格 ({final_price:.2f}) >= トレーリング閾値 ({trailing_threshold:.2f})")
            print(f"    最高値: {high_price:.2f}円, トレーリング: {params['trailing_stop_pct']*100:.1f}%")
        
        stop_loss_threshold = entry_price * (1 - params['stop_loss'])
        if final_price > stop_loss_threshold:
            print(f"  - 損切り未達: 現在価格 ({final_price:.2f}) > 損切り閾値 ({stop_loss_threshold:.2f})")
        
        if (final_idx - entry_idx) < params['max_hold_days']:
            print(f"  - 最大保有期間未達: 保有期間 ({final_idx - entry_idx}日) < 最大保有期間 ({params['max_hold_days']}日)")

if __name__ == "__main__":
    analyze_exit_conditions()
