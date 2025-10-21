"""
Phase 5-A-12-4: VWAPBounceStrategy単体テスト
シグナル生成ロジックの詳細確認
"""
import sys
import os
sys.path.insert(0, r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import numpy as np
from data_fetcher import get_parameters_and_data
from strategies.VWAP_Bounce import VWAPBounceStrategy

def test_vwap_bounce_with_real_data():
    """実データでVWAPBounceStrategy単体テスト"""
    print("=" * 80)
    print("VWAPBounceStrategy 単体テスト - 実データ版")
    print("=" * 80)
    
    # 実データ取得
    print("\n[1/5] データ取得中...")
    ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
    print(f"  ティッカー: {ticker}")
    print(f"  データ行数: {len(stock_data)}")
    print(f"  カラム: {stock_data.columns.tolist()}")
    print(f"  インデックス型: {type(stock_data.index)}")
    print(f"  先頭日付: {stock_data.index[0]}")
    print(f"  最終日付: {stock_data.index[-1]}")
    
    # 戦略インスタンス化
    print("\n[2/5] 戦略インスタンス化...")
    try:
        strategy = VWAPBounceStrategy(data=stock_data)
        print("  ✓ インスタンス化成功")
    except Exception as e:
        print(f"  ✗ インスタンス化失敗: {e}")
        return
    
    # 初期化
    print("\n[3/5] 戦略初期化...")
    try:
        strategy.initialize_strategy()
        print("  ✓ 初期化成功")
        print(f"  VWAP計算済み: {'VWAP' in strategy.data.columns}")
        print(f"  ATR計算済み: {'ATR' in strategy.data.columns}")
    except Exception as e:
        print(f"  ✗ 初期化失敗: {e}")
        return
    
    # トレンド判定テスト
    print("\n[4/5] トレンド判定テスト...")
    from indicators.unified_trend_detector import detect_unified_trend
    
    for idx in [50, 100, 150, -1]:
        if idx == -1:
            idx = len(stock_data) - 1
        
        try:
            trend = detect_unified_trend(
                strategy.data.iloc[:idx + 1], 
                price_column=strategy.price_column,
                strategy="VWAP_Bounce"
            )
            print(f"  idx={idx:3d} ({strategy.data.index[idx].date()}): trend={trend}")
        except Exception as e:
            print(f"  idx={idx:3d}: エラー - {e}")
    
    # エントリー条件詳細チェック（最新50日分）
    print("\n[5/5] エントリー条件詳細チェック（最新50日）...")
    check_count = 0
    for idx in range(max(1, len(strategy.data) - 50), len(strategy.data)):
        try:
            # トレンド判定
            trend = detect_unified_trend(
                strategy.data.iloc[:idx + 1],
                price_column=strategy.price_column,
                strategy="VWAP_Bounce"
            )
            
            # 価格・VWAP取得
            current_price = strategy.data[strategy.price_column].iloc[idx]
            vwap = strategy.data['VWAP'].iloc[idx]
            previous_close = strategy.data[strategy.price_column].iloc[idx - 1]
            current_volume = strategy.data[strategy.volume_column].iloc[idx]
            previous_volume = strategy.data[strategy.volume_column].iloc[idx - 1]
            
            # 各条件チェック
            vwap_lower = vwap * strategy.params["vwap_lower_threshold"]
            price_near_vwap = (vwap_lower <= current_price <= vwap)
            price_change_pct = (current_price - previous_close) / previous_close
            bullish_candle = price_change_pct > strategy.params["bullish_candle_min_pct"]
            volume_ratio = current_volume / previous_volume
            volume_increase = volume_ratio > strategy.params["volume_increase_threshold"]
            
            # 条件を満たす場合のみ表示
            if price_near_vwap or (trend == "range-bound"):
                check_count += 1
                print(f"\n  idx={idx} ({strategy.data.index[idx].date()}):")
                print(f"    トレンド: {trend} {'✓' if trend == 'range-bound' else '✗'}")
                print(f"    VWAP近辺: {price_near_vwap} {'✓' if price_near_vwap else '✗'} (price={current_price:.2f}, vwap={vwap:.2f}, lower={vwap_lower:.2f})")
                print(f"    陽線形成: {bullish_candle} {'✓' if bullish_candle else '✗'} (change={price_change_pct*100:.2f}%)")
                print(f"    出来高増加: {volume_increase} {'✓' if volume_increase else '✗'} (ratio={volume_ratio:.2f})")
                
                all_conditions = (trend == "range-bound") and price_near_vwap and bullish_candle and volume_increase
                print(f"    → 全条件: {'✓ エントリー可能' if all_conditions else '✗ 条件不足'}")
        
        except Exception as e:
            print(f"  idx={idx}: エラー - {e}")
    
    if check_count == 0:
        print("  ※ 条件に近いケースが1件も見つかりませんでした")
    
    # バックテスト実行
    print("\n" + "=" * 80)
    print("バックテスト実行")
    print("=" * 80)
    try:
        result = strategy.backtest()
        
        entry_count = (result['Entry_Signal'] == 1).sum()
        exit_count = (result['Exit_Signal'] == -1).sum()
        
        print(f"\n結果:")
        print(f"  エントリーシグナル: {entry_count}件")
        print(f"  イグジットシグナル: {exit_count}件")
        
        if entry_count > 0:
            print(f"\nエントリー発生日:")
            entry_dates = result[result['Entry_Signal'] == 1].index
            for date in entry_dates:
                print(f"  - {date.date()}")
        else:
            print(f"\n⚠️ エントリーシグナルが0件です")
            print(f"  最も可能性の高い原因:")
            print(f"    1. トレンドが 'range-bound' ではない（実データはstrong_uptrend）")
            print(f"    2. VWAP近辺の条件を満たすケースがない")
            print(f"    3. 出来高増加条件が厳しすぎる（threshold={strategy.params['volume_increase_threshold']}）")
        
    except Exception as e:
        print(f"✗ バックテスト失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vwap_bounce_with_real_data()
