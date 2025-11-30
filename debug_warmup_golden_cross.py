"""
ウォームアップ期間機能の詳細調査 - 意図的にシグナルを発生させるケース

ゴールデンクロスを意図的に発生させてウォームアップ期間フィルタリングを検証
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from strategies.gc_strategy_signal import GCStrategy

def create_golden_cross_data():
    """ゴールデンクロスが確実に発生するデータを生成"""
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    
    # 最初は下降トレンド（SMA_5 < SMA_25）
    # day 35でゴールデンクロス発生
    # day 28でウォームアップ期間内のゴールデンクロスを発生させる
    
    prices = []
    for i in range(60):
        if i < 25:
            # 下降トレンド
            price = 100 - i * 0.5
        elif i < 28:
            # 横ばい
            price = 87 + (i - 25) * 0.1
        elif i < 32:
            # ウォームアップ期間内で急上昇（day 28-31）
            price = 87 + (i - 27) * 3
        elif i < 35:
            # 横ばい
            price = 99 + (i - 32) * 0.2
        else:
            # 取引期間で急上昇（day 35でゴールデンクロス）
            price = 99 + (i - 34) * 3
        
        prices.append(price)
    
    stock_data = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Close': prices,
        'Adj Close': prices,
        'Volume': [1000000] * 60
    }, index=dates)
    
    return stock_data, dates

def investigate_with_golden_cross():
    """ゴールデンクロスありのデータで調査"""
    
    print("=" * 80)
    print("ウォームアップ期間機能 詳細調査 - ゴールデンクロス発生ケース")
    print("=" * 80)
    
    stock_data, dates = create_golden_cross_data()
    
    print("\n1. テストデータの概要")
    print("-" * 80)
    print(f"データ期間: {dates[0]} - {dates[59]}")
    print(f"ウォームアップ期間: day 1-30 (インデックス0-29)")
    print(f"取引期間: day 31-60 (インデックス30-59)")
    print(f"意図的なゴールデンクロス発生タイミング:")
    print(f"  - day 28-31: ウォームアップ期間内の上昇（シグナル出ないはず）")
    print(f"  - day 35以降: 取引期間内の上昇（シグナル出る可能性）")
    
    # GCStrategyでテスト
    print("\n2. GCStrategy バックテスト実行")
    print("-" * 80)
    
    strategy = GCStrategy(stock_data.copy())
    
    trading_start_date = dates[30]  # day 31
    trading_end_date = dates[59]
    
    print(f"trading_start_date: {trading_start_date}")
    print(f"trading_end_date: {trading_end_date}")
    
    result = strategy.backtest(
        trading_start_date=trading_start_date,
        trading_end_date=trading_end_date
    )
    
    # SMAの値を確認
    print("\n3. SMA値の推移確認")
    print("-" * 80)
    
    key_days = [25, 27, 28, 29, 30, 31, 32, 35, 40]
    print(f"{'Day':>5} {'Date':>12} {'Close':>8} {'SMA_5':>8} {'SMA_25':>8} {'Entry':>6} {'Period':>10}")
    print("-" * 80)
    
    for day in key_days:
        if day - 1 < len(result):
            idx = day - 1
            date = result.index[idx]
            close = result['Close'].iloc[idx]
            sma5 = result['SMA_5'].iloc[idx] if 'SMA_5' in result.columns else np.nan
            sma25 = result['SMA_25'].iloc[idx] if 'SMA_25' in result.columns else np.nan
            entry = result['Entry_Signal'].iloc[idx]
            period = "Warmup" if idx < 30 else "Trading"
            
            print(f"{day:5} {str(date)[:10]:>12} {close:8.2f} {sma5:8.2f} {sma25:8.2f} {entry:6} {period:>10}")
    
    # ウォームアップ期間の確認
    print("\n4. ウォームアップ期間（day 1-30）のシグナル")
    print("-" * 80)
    
    warmup_data = result.iloc[:30]
    warmup_entry = warmup_data['Entry_Signal'].sum()
    warmup_exit = warmup_data['Exit_Signal'].sum()
    
    print(f"Entry_Signal: {warmup_entry}件")
    print(f"Exit_Signal: {warmup_exit}件")
    
    if warmup_entry > 0:
        print("\n[警告] ウォームアップ期間にEntry_Signalが検出されました:")
        for idx in range(30):
            if result['Entry_Signal'].iloc[idx] == 1:
                date = result.index[idx]
                print(f"  day {idx+1} ({date}): Close={result['Close'].iloc[idx]:.2f}, "
                      f"SMA_5={result['SMA_5'].iloc[idx]:.2f}, SMA_25={result['SMA_25'].iloc[idx]:.2f}")
    else:
        print("[OK] ウォームアップ期間にEntry_Signalなし")
    
    # 取引期間の確認
    print("\n5. 取引期間（day 31-60）のシグナル")
    print("-" * 80)
    
    trading_data = result.iloc[30:]
    trading_entry = trading_data['Entry_Signal'].sum()
    trading_exit = trading_data['Exit_Signal'].sum()
    
    print(f"Entry_Signal: {trading_entry}件")
    print(f"Exit_Signal: {trading_exit}件")
    
    if trading_entry > 0:
        print("\n取引期間のEntry_Signal詳細:")
        for idx in range(30, len(result)):
            if result['Entry_Signal'].iloc[idx] == 1:
                date = result.index[idx]
                print(f"  day {idx+1} ({date}): Close={result['Close'].iloc[idx]:.2f}, "
                      f"SMA_5={result['SMA_5'].iloc[idx]:.2f}, SMA_25={result['SMA_25'].iloc[idx]:.2f}")
    else:
        print("[注意] 取引期間にEntry_Signalなし")
    
    # 境界の確認
    print("\n6. 境界条件の詳細確認")
    print("-" * 80)
    
    for day in [30, 31]:
        idx = day - 1
        date = result.index[idx]
        entry = result['Entry_Signal'].iloc[idx]
        close = result['Close'].iloc[idx]
        sma5 = result['SMA_5'].iloc[idx]
        sma25 = result['SMA_25'].iloc[idx]
        
        print(f"day {day} ({date}):")
        print(f"  Entry_Signal: {entry}")
        print(f"  Close: {close:.2f}")
        print(f"  SMA_5: {sma5:.2f}")
        print(f"  SMA_25: {sma25:.2f}")
        print(f"  期間: {'ウォームアップ' if idx < 30 else '取引期間'}")
        
        if idx > 0:
            prev_sma5 = result['SMA_5'].iloc[idx-1]
            prev_sma25 = result['SMA_25'].iloc[idx-1]
            gc_condition = (sma5 > sma25 and prev_sma5 <= prev_sma25)
            print(f"  ゴールデンクロス条件: {gc_condition}")
            print(f"    現在: SMA_5 ({sma5:.2f}) > SMA_25 ({sma25:.2f}) = {sma5 > sma25}")
            print(f"    前日: SMA_5 ({prev_sma5:.2f}) <= SMA_25 ({prev_sma25:.2f}) = {prev_sma5 <= prev_sma25}")
        print()
    
    # 結果のまとめ
    print("\n7. 検証結果のまとめ")
    print("=" * 80)
    
    checks = {
        "ウォームアップ期間にEntry_Signalなし": warmup_entry == 0,
        "ウォームアップ期間にExit_Signalなし": warmup_exit == 0,
    }
    
    all_passed = True
    for check_name, check_result in checks.items():
        status = "[OK]" if check_result else "[NG]"
        print(f"{status} {check_name}")
        if not check_result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("結論: ウォームアップ期間機能は正常に動作しています")
        print("  - ウォームアップ期間内でシグナルが抑制されている")
        print("  - 取引期間でのみシグナル生成が許可されている")
    else:
        print("結論: ウォームアップ期間機能に問題が検出されました")
    print("=" * 80)
    
    # CSV出力
    output_path = "warmup_golden_cross_result.csv"
    result.to_csv(output_path)
    print(f"\n詳細結果を {output_path} に保存しました")
    
    return result


if __name__ == "__main__":
    result_df = investigate_with_golden_cross()
