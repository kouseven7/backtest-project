#!/usr/bin/env python3
"""
同一日エントリー/エグジット現象の詳細分析
目的: 全ての取引が同一日で行われる理由を解明
"""

import pandas as pd
import sys
import os
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

def main():
    print("🔍 同一日エントリー/エグジット現象の詳細分析")
    print("="*60)
    
    # データ読み込み
    data_csv = r"output\main_outputs\csv\9101.T_integrated_strategy_20251012_071815_data.csv"
    trades_csv = r"output\main_outputs\csv\9101.T_integrated_strategy_20251012_071815_trades.csv"
    
    try:
        # データCSV（メインデータ）
        data_df = pd.read_csv(data_csv, index_col=0)
        data_df.index = pd.to_datetime(data_df.index)
        
        # トレードCSV（取引記録）
        trades_df = pd.read_csv(trades_csv)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        print("✅ データ読み込み完了")
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return
    
    # 1. 同一日エントリー/エグジット現象の確認
    print(f"\n📊 同一日取引現象の確認:")
    
    entry_trades = trades_df[trades_df['type'] == 'Entry'].copy()
    exit_trades = trades_df[trades_df['type'] == 'Exit'].copy()
    
    print(f"  エントリー数: {len(entry_trades)}")
    print(f"  エグジット数: {len(exit_trades)}")
    
    # 各エントリーに対応するエグジットの存在確認
    same_day_count = 0
    different_day_count = 0
    no_exit_count = 0
    
    for _, entry in entry_trades.iterrows():
        entry_date = entry['timestamp']
        
        # 同じ日のエグジットを探す
        same_day_exit = exit_trades[exit_trades['timestamp'] == entry_date]
        
        if len(same_day_exit) > 0:
            same_day_count += 1
        else:
            # 異なる日のエグジットがあるか
            later_exits = exit_trades[exit_trades['timestamp'] > entry_date]
            if len(later_exits) > 0:
                different_day_count += 1
            else:
                no_exit_count += 1
    
    print(f"\n📈 取引パターン分析:")
    print(f"  同一日エントリー/エグジット: {same_day_count}件")
    print(f"  異なる日エグジット: {different_day_count}件")
    print(f"  エグジットなし: {no_exit_count}件")
    
    # 2. この現象が「強制決済」かどうかの検証
    print(f"\n🎯 強制決済検証:")
    
    # データCSVでエントリーとエグジットが同じ行にあるかチェック
    entry_signals = data_df[data_df['Entry_Signal'] == 1]
    exit_signals = data_df[data_df['Exit_Signal'] == 1]
    
    same_row_signals = data_df[(data_df['Entry_Signal'] == 1) & (data_df['Exit_Signal'] == 1)]
    
    print(f"  データCSVでのエントリーシグナル数: {len(entry_signals)}")
    print(f"  データCSVでのエグジットシグナル数: {len(exit_signals)}")
    print(f"  同一行でエントリー&エグジット: {len(same_row_signals)}件")
    
    if len(same_row_signals) > 0:
        print(f"  同一行シグナルの日付（最初の10件）:")
        for idx in same_row_signals.index[:10]:
            close_price = data_df.loc[idx, 'Close']
            print(f"    {idx}: Close価格 {close_price}")
    
    # 3. 価格分析 - エントリー価格とエグジット価格の関係
    print(f"\n💰 価格分析:")
    
    # エントリー価格とエグジット価格の完全一致確認
    exact_match_count = 0
    price_differences = []
    
    for _, entry in entry_trades.iterrows():
        entry_date = entry['timestamp']
        entry_price = entry['price']
        
        # 同じ日のエグジットを探す
        same_day_exit = exit_trades[exit_trades['timestamp'] == entry_date]
        
        if len(same_day_exit) > 0:
            exit_price = same_day_exit.iloc[0]['price']
            price_diff = abs(entry_price - exit_price)
            price_differences.append(price_diff)
            
            if price_diff < 0.01:  # ほぼ同じ価格
                exact_match_count += 1
    
    print(f"  価格完全一致件数: {exact_match_count} / {len(entry_trades)}")
    if price_differences:
        avg_diff = sum(price_differences) / len(price_differences)
        max_diff = max(price_differences)
        print(f"  平均価格差: {avg_diff:.6f}")
        print(f"  最大価格差: {max_diff:.6f}")
    
    # 4. データCSVのClose価格との一致確認
    print(f"\n📅 Close価格との一致確認:")
    
    close_match_count = 0
    close_differences = []
    
    for _, entry in entry_trades.iterrows():
        entry_date = entry['timestamp']
        entry_price = entry['price']
        
        # データCSVの同じ日のClose価格
        if entry_date in data_df.index:
            close_price = data_df.loc[entry_date, 'Close']
            close_diff = abs(entry_price - close_price)
            close_differences.append(close_diff)
            
            if close_diff < 0.01:
                close_match_count += 1
    
    print(f"  Close価格完全一致件数: {close_match_count} / {len(entry_trades)}")
    if close_differences:
        avg_close_diff = sum(close_differences) / len(close_differences)
        max_close_diff = max(close_differences)
        print(f"  Close価格との平均差: {avg_close_diff:.6f}")
        print(f"  Close価格との最大差: {max_close_diff:.6f}")
    
    # 5. 最終日の特別処理確認
    print(f"\n🔚 最終日処理確認:")
    
    last_data_date = data_df.index.max()
    last_trade_date = trades_df['timestamp'].max()
    
    print(f"  データCSV最終日: {last_data_date}")
    print(f"  トレードCSV最終日: {last_trade_date}")
    
    # 最終日の取引
    final_day_trades = trades_df[trades_df['timestamp'] == last_trade_date]
    print(f"  最終日の取引数: {len(final_day_trades)}")
    
    if len(final_day_trades) > 0:
        final_entries = final_day_trades[final_day_trades['type'] == 'Entry']
        final_exits = final_day_trades[final_day_trades['type'] == 'Exit']
        print(f"    最終日エントリー: {len(final_entries)}件")
        print(f"    最終日エグジット: {len(final_exits)}件")
        
        if len(final_exits) > 0:
            final_exit_prices = final_exits['price'].unique()
            print(f"    最終日エグジット価格: {final_exit_prices}")
            
            # この価格が全体のエグジットで使われているか
            all_exit_prices = exit_trades['price'].values
            final_price_usage = sum(1 for p in all_exit_prices if abs(p - final_exit_prices[0]) < 0.01)
            print(f"    この価格での総エグジット数: {final_price_usage} / {len(exit_trades)}")
    
    # 6. 結論
    print(f"\n📋 分析結論:")
    print(f"現象の特徴:")
    print(f"  1. 全取引が同一日エントリー/エグジット: {same_day_count == len(entry_trades)}")
    print(f"  2. エントリー価格とエグジット価格が完全一致: {exact_match_count == len(entry_trades)}")
    print(f"  3. 取引価格がClose価格と一致: {close_match_count == len(entry_trades)}")
    
    if same_day_count == len(entry_trades) and exact_match_count == len(entry_trades):
        print(f"\n🎯 推定原因:")
        print(f"  これは「デイトレード強制決済」パターンです:")
        print(f"  - エントリーシグナルが発生した日に")
        print(f"  - 即座にエグジットシグナルも発生")
        print(f"  - その日のClose価格で両方の取引が記録")
        print(f"  - 実質的に利益/損失は0になる")
        print(f"  ")
        print(f"  この現象は以下を示唆:")
        print(f"  1. ポジション保持ロジックが機能していない")
        print(f"  2. エグジット条件が即座に満たされる")
        print(f"  3. または強制的に当日決済される仕組み")
    else:
        print(f"  より複雑な状況が発生しています")
    
    print("\n" + "="*60)
    print("🎯 分析完了")

if __name__ == "__main__":
    main()