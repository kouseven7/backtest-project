#!/usr/bin/env python3
"""
取引ファイル詳細分析ツール
目的: 124取引の内訳とentry/exit比率を詳細分析
"""
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
from pathlib import Path

def analyze_trades_detail():
    """
    取引ファイルの詳細分析
    """
    print("=" * 60)
    print("🔍 取引ファイル詳細分析")
    print("=" * 60)
    
    # 最新の取引ファイル
    trades_path = Path(r"C:\Users\imega\Documents\my_backtest_project\output\main_outputs\csv\7203.T_integrated_strategy_20251008_232614_trades.csv")
    
    try:
        trades_df = pd.read_csv(trades_path)
        
        print(f"📊 基本統計:")
        print(f"  - 総行数: {len(trades_df)}")
        print(f"  - 列: {list(trades_df.columns)}")
        
        # type列の分析
        if 'type' in trades_df.columns:
            type_counts = trades_df['type'].value_counts()
            print(f"\n📈 取引タイプ別統計:")
            for trade_type, count in type_counts.items():
                print(f"  - {trade_type}: {count}件")
        
        # 価格分析
        if 'price' in trades_df.columns:
            print(f"\n💰 価格統計:")
            print(f"  - 最高価格: ¥{trades_df['price'].max():.2f}")
            print(f"  - 最低価格: ¥{trades_df['price'].min():.2f}")
            print(f"  - 平均価格: ¥{trades_df['price'].mean():.2f}")
        
        # 全データ表示（最初の10行と最後の10行）
        print(f"\n📝 取引データ詳細（最初の10行）:")
        print(trades_df.head(10).to_string())
        
        print(f"\n📝 取引データ詳細（最後の10行）:")
        print(trades_df.tail(10).to_string())
        
        # Entry/Exit価格の比較
        if 'type' in trades_df.columns:
            entry_df = trades_df[trades_df['type'] == 'Entry']
            exit_df = trades_df[trades_df['type'] == 'Exit']
            
            if len(entry_df) > 0 and len(exit_df) > 0:
                print(f"\n🎯 Entry/Exit価格比較:")
                print(f"  - Entry平均価格: ¥{entry_df['price'].mean():.2f}")
                print(f"  - Exit平均価格: ¥{exit_df['price'].mean():.2f}")
                print(f"  - 価格差: ¥{exit_df['price'].mean() - entry_df['price'].mean():.2f}")
        
        return trades_df
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def analyze_data_csv_detail():
    """
    データCSVファイルの詳細分析
    """
    print(f"\n" + "=" * 60)
    print("🔍 データCSVファイル詳細分析")
    print("=" * 60)
    
    data_path = Path(r"C:\Users\imega\Documents\my_backtest_project\output\main_outputs\csv\7203.T_integrated_strategy_20251008_232614_data.csv")
    
    try:
        data_df = pd.read_csv(data_path)
        
        print(f"📊 基本統計:")
        print(f"  - 総行数: {len(data_df)}")
        print(f"  - 列数: {len(data_df.columns)}")
        
        # Entry_Signal/Exit_Signalの詳細分析
        entry_dates = data_df[data_df['Entry_Signal'] == 1]['Date'] if 'Date' in data_df.columns else []
        exit_dates = data_df[data_df['Exit_Signal'] == 1]['Date'] if 'Date' in data_df.columns else []
        
        print(f"\n📈 シグナル詳細:")
        print(f"  - Entry_Signal=1: {(data_df['Entry_Signal'] == 1).sum()}件")
        print(f"  - Exit_Signal=1: {(data_df['Exit_Signal'] == 1).sum()}件")
        
        if len(entry_dates) > 0:
            print(f"\n📅 エントリー日付サンプル（最初の5件）:")
            for date in entry_dates.head():
                print(f"  - {date}")
        
        if len(exit_dates) > 0:
            print(f"\n📅 エグジット日付サンプル（最初の5件）:")
            for date in exit_dates.head():
                print(f"  - {date}")
        
        # Strategyカラム分析
        if 'Strategy' in data_df.columns:
            strategy_entries = data_df[data_df['Entry_Signal'] == 1]['Strategy'].value_counts()
            strategy_exits = data_df[data_df['Exit_Signal'] == 1]['Strategy'].value_counts()
            
            print(f"\n📋 戦略別エントリー:")
            for strategy, count in strategy_entries.items():
                print(f"  - {strategy}: {count}件")
            
            print(f"\n📋 戦略別エグジット:")
            for strategy, count in strategy_exits.items():
                print(f"  - {strategy}: {count}件")
        
        return data_df
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def main():
    print("🔍 詳細取引分析開始")
    
    # 取引ファイル分析
    trades_df = analyze_trades_detail()
    
    # データファイル分析  
    data_df = analyze_data_csv_detail()
    
    # 矛盾点の最終確認
    print(f"\n" + "=" * 60)
    print("🎯 矛盾点最終確認")
    print("=" * 60)
    
    if trades_df is not None and data_df is not None:
        print(f"📊 最終統計:")
        print(f"  - 取引ファイル総行数: {len(trades_df)}")
        print(f"  - データファイルEntry_Signal=1: {(data_df['Entry_Signal'] == 1).sum()}")
        print(f"  - データファイルExit_Signal=1: {(data_df['Exit_Signal'] == 1).sum()}")
        
        # 取引タイプ別
        if 'type' in trades_df.columns:
            entry_trades = (trades_df['type'] == 'Entry').sum()
            exit_trades = (trades_df['type'] == 'Exit').sum()
            
            print(f"  - 取引ファイルEntry: {entry_trades}件")
            print(f"  - 取引ファイルExit: {exit_trades}件")
            
            print(f"\n❓ 重要な質問:")
            print(f"  1. 取引ファイル124行 = Entry{entry_trades} + Exit{exit_trades}?")
            print(f"  2. Entry数が一致するか: データファイル{(data_df['Entry_Signal'] == 1).sum()} vs 取引ファイル{entry_trades}")
            print(f"  3. Exit数が一致するか: データファイル{(data_df['Exit_Signal'] == 1).sum()} vs 取引ファイル{exit_trades}")

if __name__ == "__main__":
    main()