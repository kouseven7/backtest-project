#!/usr/bin/env python3
"""
エグジットシグナル処理の詳細調査
目的: データCSVとトレードCSVの矛盾を解明
"""

import pandas as pd
import sys
import os
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

def main():
    print("🔍 エグジットシグナル処理の詳細調査開始")
    print("="*60)
    
    # 1. データCSVファイルの分析
    data_csv = r"output\main_outputs\csv\9101.T_integrated_strategy_20251012_071815_data.csv"
    trades_csv = r"output\main_outputs\csv\9101.T_integrated_strategy_20251012_071815_trades.csv"
    
    print("📄 ファイル分析:")
    print(f"  データCSV: {data_csv}")
    print(f"  トレードCSV: {trades_csv}")
    
    # データCSV読み込み
    try:
        data_df = pd.read_csv(data_csv, index_col=0)
        data_df.index = pd.to_datetime(data_df.index)
        print(f"\n📊 データCSV分析:")
        print(f"  行数: {len(data_df)}")
        
        # エグジットシグナルの確認
        exit_signals = data_df[data_df['Exit_Signal'] == 1]
        print(f"  エグジットシグナル数: {len(exit_signals)}")
        
        if len(exit_signals) > 0:
            print(f"  エグジット発生日:")
            for idx in exit_signals.index[:10]:  # 最初の10個
                close_price = data_df.loc[idx, 'Close']
                print(f"    {idx}: Close価格 {close_price}")
        else:
            print("  ⚠️ エグジットシグナルが見つかりません")
        
    except Exception as e:
        print(f"❌ データCSV読み込みエラー: {e}")
        return
    
    # トレードCSV読み込み
    try:
        trades_df = pd.read_csv(trades_csv)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        print(f"\n📈 トレードCSV分析:")
        print(f"  行数: {len(trades_df)}")
        
        # エグジットのみ抽出
        exit_trades = trades_df[trades_df['type'] == 'Exit']
        print(f"  エグジット数: {len(exit_trades)}")
        
        if len(exit_trades) > 0:
            print(f"  エグジット価格の分析:")
            unique_prices = exit_trades['price'].unique()
            print(f"    ユニーク価格数: {len(unique_prices)}")
            print(f"    価格範囲: {exit_trades['price'].min():.2f} - {exit_trades['price'].max():.2f}")
            
            print(f"  エグジット日時と価格（最初の10件）:")
            for _, row in exit_trades.head(10).iterrows():
                print(f"    {row['timestamp']}: 価格 {row['price']}")
                
            # 同一日にエントリーとエグジットがあるかチェック
            entry_trades = trades_df[trades_df['type'] == 'Entry']
            same_day_trades = []
            
            for _, exit_row in exit_trades.iterrows():
                exit_date = exit_row['timestamp']
                matching_entry = entry_trades[entry_trades['timestamp'] == exit_date]
                if not matching_entry.empty:
                    same_day_trades.append({
                        'date': exit_date,
                        'entry_price': matching_entry.iloc[0]['price'],
                        'exit_price': exit_row['price']
                    })
            
            print(f"\n🔍 同一日エントリー/エグジット分析:")
            print(f"  同一日取引数: {len(same_day_trades)}")
            if same_day_trades:
                print(f"  同一日取引詳細（最初の5件）:")
                for trade in same_day_trades[:5]:
                    print(f"    {trade['date']}: Entry {trade['entry_price']:.2f} → Exit {trade['exit_price']:.2f}")
        
    except Exception as e:
        print(f"❌ トレードCSV読み込みエラー: {e}")
        return
    
    # 3. データの整合性チェック
    print(f"\n🔧 データ整合性チェック:")
    
    # データCSVのエグジット日とトレードCSVのエグジット日を比較
    if len(exit_signals) > 0 and len(exit_trades) > 0:
        data_exit_dates = set(exit_signals.index.date)
        trade_exit_dates = set(exit_trades['timestamp'].dt.date)
        
        print(f"  データCSVエグジット日数: {len(data_exit_dates)}")
        print(f"  トレードCSVエグジット日数: {len(trade_exit_dates)}")
        
        common_dates = data_exit_dates.intersection(trade_exit_dates)
        print(f"  共通エグジット日数: {len(common_dates)}")
        
        if len(common_dates) > 0:
            print(f"  共通日の価格比較（最初の5日）:")
            for date in list(common_dates)[:5]:
                data_close = data_df.loc[data_df.index.date == date, 'Close'].iloc[0]
                trade_price = exit_trades[exit_trades['timestamp'].dt.date == date]['price'].iloc[0]
                print(f"    {date}: データCSV Close {data_close:.2f} vs トレードCSV {trade_price:.2f}")
        
        # 差異のあるケースの検出
        only_in_data = data_exit_dates - trade_exit_dates
        only_in_trades = trade_exit_dates - data_exit_dates
        
        if only_in_data:
            print(f"  ⚠️ データCSVのみにあるエグジット日: {len(only_in_data)}日")
        if only_in_trades:
            print(f"  ⚠️ トレードCSVのみにあるエグジット日: {len(only_in_trades)}日")
    
    # 4. 仮説の検証
    print(f"\n🎯 仮説検証:")
    print(f"ユーザー仮説: 「エグジットシグナルは生成されているが、")
    print(f"             その価格が使われずに最終日強制決済で計算される」")
    
    # データCSVでエグジットシグナルがあるか
    has_data_exits = len(exit_signals) > 0
    # トレードCSVでエグジット価格が多様か
    has_varied_exit_prices = len(exit_trades['price'].unique()) > 1 if len(exit_trades) > 0 else False
    # エグジット価格がエントリー価格と同じか（同一日取引）
    same_price_trades = len([t for t in same_day_trades if abs(t['entry_price'] - t['exit_price']) < 0.01]) if same_day_trades else 0
    
    print(f"\n検証結果:")
    print(f"  1. データCSVにエグジットシグナル存在: {has_data_exits}")
    print(f"  2. トレードCSVに多様なエグジット価格: {has_varied_exit_prices}")
    print(f"  3. 同一日エントリー/エグジット取引数: {len(same_day_trades) if same_day_trades else 0}")
    print(f"  4. 同一価格での取引数: {same_price_trades}")
    
    # 結論
    print(f"\n📋 結論:")
    if not has_data_exits and has_varied_exit_prices:
        print("  ✅ 仮説は部分的に正しい:")
        print("     - データCSVにエグジットシグナルがない")
        print("     - しかしトレードCSVには多様なエグジット価格が記録されている")
        print("     - これは中間処理でエグジットが発生していることを示唆")
    elif has_data_exits and has_varied_exit_prices:
        print("  ✅ 仮説は正しい:")
        print("     - エグジットシグナルは生成されている")
        print("     - 多様なエグジット価格も記録されている")
        print("     - 最終的な損益計算で強制決済価格が使われている可能性")
    else:
        print("  ❓ 状況はより複雑:")
        print(f"     - データCSVエグジット: {has_data_exits}")
        print(f"     - 多様なエグジット価格: {has_varied_exit_prices}")
    
    print("\n" + "="*60)
    print("🎯 調査完了")

if __name__ == "__main__":
    main()