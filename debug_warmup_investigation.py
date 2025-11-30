"""
ウォームアップ期間機能の詳細調査スクリプト

このスクリプトは以下を確認します:
1. ウォームアップ期間でシグナルが0件であること
2. 取引期間でのみシグナルが生成されること
3. 実際の取引タイミングと価格データ
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトルート追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from strategies.gc_strategy_signal import GCStrategy
from strategies.Breakout import BreakoutStrategy

def investigate_warmup_period():
    """ウォームアップ期間機能の詳細調査"""
    
    print("=" * 80)
    print("ウォームアップ期間機能 詳細調査")
    print("=" * 80)
    
    # 60日分のテストデータを生成
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    np.random.seed(42)  # 再現性のため
    
    stock_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(60) * 2),
        'High': 102 + np.cumsum(np.random.randn(60) * 2),
        'Low': 98 + np.cumsum(np.random.randn(60) * 2),
        'Close': 100 + np.cumsum(np.random.randn(60) * 2),
        'Volume': np.random.randint(1000000, 10000000, 60)
    }, index=dates)
    
    stock_data['Adj Close'] = stock_data['Close']
    
    print("\n1. データ概要")
    print("-" * 80)
    print(f"データ期間: {dates[0]} - {dates[59]}")
    print(f"データ件数: {len(stock_data)}日分")
    print(f"ウォームアップ期間設定: day 1-30 (インデックス0-29)")
    print(f"取引期間設定: day 31-60 (インデックス30-59)")
    
    # GCStrategyでテスト
    print("\n2. GCStrategy テスト")
    print("-" * 80)
    
    strategy = GCStrategy(stock_data.copy())
    
    # trading_start_date = day 31（インデックス30）
    trading_start_date = dates[30]
    trading_end_date = dates[59]
    
    print(f"trading_start_date: {trading_start_date}")
    print(f"trading_end_date: {trading_end_date}")
    
    # バックテスト実行
    result = strategy.backtest(
        trading_start_date=trading_start_date,
        trading_end_date=trading_end_date
    )
    
    # ウォームアップ期間の詳細確認
    print("\n3. ウォームアップ期間（day 1-30）のシグナル確認")
    print("-" * 80)
    warmup_data = result.iloc[:30]
    warmup_entry_signals = warmup_data['Entry_Signal'].sum()
    warmup_exit_signals = warmup_data['Exit_Signal'].sum()
    
    print(f"Entry_Signal合計: {warmup_entry_signals}件")
    print(f"Exit_Signal合計: {warmup_exit_signals}件")
    
    # シグナルがある場合は詳細を表示
    if warmup_entry_signals > 0:
        print("\n[警告] ウォームアップ期間にEntry_Signalが生成されています:")
        entry_dates = warmup_data[warmup_data['Entry_Signal'] == 1].index
        for date in entry_dates:
            idx = result.index.get_loc(date)
            print(f"  - {date} (day {idx+1}, インデックス{idx})")
            print(f"    Close: {result.loc[date, 'Close']:.2f}")
            print(f"    SMA_5: {result.loc[date, 'SMA_5']:.2f}")
            print(f"    SMA_25: {result.loc[date, 'SMA_25']:.2f}")
    else:
        print("[OK] ウォームアップ期間にEntry_Signalなし")
    
    if warmup_exit_signals > 0:
        print("\n[警告] ウォームアップ期間にExit_Signalが生成されています:")
        exit_dates = warmup_data[warmup_data['Exit_Signal'] == -1].index
        for date in exit_dates:
            idx = result.index.get_loc(date)
            print(f"  - {date} (day {idx+1}, インデックス{idx})")
    else:
        print("[OK] ウォームアップ期間にExit_Signalなし")
    
    # 取引期間の詳細確認
    print("\n4. 取引期間（day 31-60）のシグナル確認")
    print("-" * 80)
    trading_data = result.iloc[30:]
    trading_entry_signals = trading_data['Entry_Signal'].sum()
    trading_exit_signals = trading_data['Exit_Signal'].sum()
    
    print(f"Entry_Signal合計: {trading_entry_signals}件")
    print(f"Exit_Signal合計: {trading_exit_signals}件")
    
    # シグナルがある場合は詳細を表示
    if trading_entry_signals > 0:
        print("\n取引期間のEntry_Signal詳細:")
        entry_dates = trading_data[trading_data['Entry_Signal'] == 1].index
        for date in entry_dates:
            idx = result.index.get_loc(date)
            print(f"  - {date} (day {idx+1}, インデックス{idx})")
            print(f"    Close: {result.loc[date, 'Close']:.2f}")
            if 'SMA_5' in result.columns:
                print(f"    SMA_5: {result.loc[date, 'SMA_5']:.2f}")
            if 'SMA_25' in result.columns:
                print(f"    SMA_25: {result.loc[date, 'SMA_25']:.2f}")
    else:
        print("[注意] 取引期間にEntry_Signalが生成されていません")
        print("（戦略条件を満たさなかった可能性があります）")
    
    if trading_exit_signals > 0:
        print("\n取引期間のExit_Signal詳細:")
        exit_dates = trading_data[trading_data['Exit_Signal'] == -1].index
        for date in exit_dates:
            idx = result.index.get_loc(date)
            print(f"  - {date} (day {idx+1}, インデックス{idx})")
    
    # 境界条件の確認
    print("\n5. 境界条件の確認")
    print("-" * 80)
    print(f"day 30 (インデックス29): {dates[29]}")
    print(f"  Entry_Signal: {result.iloc[29]['Entry_Signal']}")
    print(f"  Exit_Signal: {result.iloc[29]['Exit_Signal']}")
    print(f"  期待値: 両方とも0（ウォームアップ期間）")
    
    print(f"\nday 31 (インデックス30): {dates[30]}")
    print(f"  Entry_Signal: {result.iloc[30]['Entry_Signal']}")
    print(f"  Exit_Signal: {result.iloc[30]['Exit_Signal']}")
    print(f"  期待値: シグナル生成可能（取引期間開始）")
    
    # データフレーム全体のサマリー
    print("\n6. 結果データフレームのサマリー")
    print("-" * 80)
    print(f"結果データフレーム行数: {len(result)}")
    print(f"Entry_Signal列の値の種類: {result['Entry_Signal'].unique()}")
    print(f"Exit_Signal列の値の種類: {result['Exit_Signal'].unique()}")
    
    # 検証結果のまとめ
    print("\n7. 検証結果のまとめ")
    print("=" * 80)
    
    checks = {
        "ウォームアップ期間にEntry_Signalなし": warmup_entry_signals == 0,
        "ウォームアップ期間にExit_Signalなし": warmup_exit_signals == 0,
        "取引期間のEntry_Signalは0または1": trading_data['Entry_Signal'].isin([0, 1]).all(),
        "取引期間のExit_Signalは0または-1": trading_data['Exit_Signal'].isin([0, -1]).all(),
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
    else:
        print("結論: ウォームアップ期間機能に問題が検出されました")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    result_df = investigate_warmup_period()
    
    # 詳細CSVを出力
    output_path = "warmup_investigation_result.csv"
    result_df.to_csv(output_path)
    print(f"\n詳細結果を {output_path} に保存しました")
