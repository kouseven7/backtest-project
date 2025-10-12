#!/usr/bin/env python3
"""
レポートデータ流れのデバッグスクリプト（修正版）
目的: 戦略名が空白でエグジット価格が固定される原因を特定
"""

import pandas as pd
import sys
import os
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger
from output.data_extraction_enhancer import MainDataExtractor, extract_and_analyze_main_data

logger = setup_logger(__name__)

def main():
    print("🚀 レポートデータ流れデバッグ開始（修正版）")
    print("="*50)
    
    # 最新の9101.TのCSVファイルを直接指定
    latest_csv = r"output\main_outputs\csv\9101.T_integrated_strategy_20251012_071815_data.csv"
    
    if not os.path.exists(latest_csv):
        print(f"❌ ファイルが見つかりません: {latest_csv}")
        return
    
    print(f"📄 読み込みファイル: {latest_csv}")
    
    try:
        stock_data = pd.read_csv(latest_csv, index_col=0)
        stock_data.index = pd.to_datetime(stock_data.index)
        
        print(f"📊 データ基本情報:")
        print(f"  行数: {len(stock_data)}")
        print(f"  列数: {len(stock_data.columns)}")
        print(f"  期間: {stock_data.index[0]} - {stock_data.index[-1]}")
        
        print(f"\n📋 列名:")
        for i, col in enumerate(stock_data.columns):
            print(f"  {i+1:2d}. {col}")
        
        # シグナル列の詳細調査
        print(f"\n🎯 シグナル列調査:")
        signal_columns = [col for col in stock_data.columns if 'Signal' in col or 'Strategy' in col]
        
        for col in signal_columns:
            if col in stock_data.columns:
                unique_values = stock_data[col].unique()
                non_zero_count = (stock_data[col] != 0).sum()
                print(f"  {col}:")
                print(f"    ユニーク値: {unique_values[:10]}")  # 最初の10個
                print(f"    非ゼロ数: {non_zero_count}")
                if col == 'Strategy' or 'Strategy' in col:
                    print(f"    戦略名: {[v for v in unique_values if v and str(v) != 'nan']}")
        
        # 取引抽出の詳細調査
        print(f"\n⚙️ 取引抽出プロセス調査:")
        extractor = MainDataExtractor()
        
        print(f"1. エントリーシグナル確認:")
        if 'Entry_Signal' in stock_data.columns:
            entry_signals = stock_data[stock_data['Entry_Signal'] == 1]
            print(f"   エントリー数: {len(entry_signals)}")
            if len(entry_signals) > 0:
                print(f"   最初のエントリー: {entry_signals.index[0]}")
                print(f"   最後のエントリー: {entry_signals.index[-1]}")
                
                # 戦略情報確認
                if 'Strategy' in stock_data.columns:
                    strategies = entry_signals['Strategy'].value_counts()
                    print(f"   戦略別エントリー数:")
                    for strategy, count in strategies.items():
                        print(f"     {strategy}: {count}回")
        
        print(f"\n2. エグジットシグナル確認:")
        if 'Exit_Signal' in stock_data.columns:
            exit_signals = stock_data[stock_data['Exit_Signal'] == 1]
            print(f"   エグジット数: {len(exit_signals)}")
            if len(exit_signals) > 0:
                print(f"   最初のエグジット: {exit_signals.index[0]}")
                print(f"   最後のエグジット: {exit_signals.index[-1]}")
        
        print(f"\n3. 実際の取引抽出:")
        trades = extractor.extract_accurate_trades(stock_data)
        print(f"   抽出された取引数: {len(trades)}")
        
        if trades:
            print(f"   最初の取引:")
            first_trade = trades[0]
            for key, value in first_trade.items():
                print(f"     {key}: {value}")
            
            # エグジット価格の分析
            exit_prices = [trade['exit_price'] for trade in trades]
            unique_exit_prices = set(exit_prices)
            print(f"\n   エグジット価格分析:")
            print(f"     ユニーク価格数: {len(unique_exit_prices)}")
            print(f"     価格範囲: {min(exit_prices):.2f} - {max(exit_prices):.2f}")
            if len(unique_exit_prices) == 1:
                print(f"     ⚠️ 全て同じ価格: {list(unique_exit_prices)[0]}")
            elif len(unique_exit_prices) <= 5:
                print(f"     ⚠️ 異なる価格が少ない: {sorted(unique_exit_prices)}")
            
            # 戦略名の分析
            strategies = [trade.get('strategy', 'Unknown') for trade in trades]
            unique_strategies = set(strategies)
            print(f"\n   戦略名分析:")
            print(f"     ユニーク戦略数: {len(unique_strategies)}")
            print(f"     戦略名: {list(unique_strategies)}")
            if '' in unique_strategies or 'Unknown' in unique_strategies:
                print(f"     ⚠️ 空の戦略名が検出されました")
        
        # 価格データの分析
        print(f"\n🔍 価格データ分析:")
        price_columns = ['Close', 'Adj Close', 'High', 'Low', 'Open']
        available_price_cols = [col for col in price_columns if col in stock_data.columns]
        
        print(f"利用可能な価格列: {available_price_cols}")
        
        if available_price_cols:
            final_date = stock_data.index[-1]
            print(f"最終日: {final_date}")
            
            for col in available_price_cols:
                final_price = stock_data.loc[final_date, col]
                print(f"  {col}: {final_price}")
            
            # 価格の変動を確認
            if 'Close' in stock_data.columns:
                close_prices = stock_data['Close']
                print(f"\n価格変動分析:")
                print(f"  最小値: {close_prices.min():.2f}")
                print(f"  最大値: {close_prices.max():.2f}")
                print(f"  最終値: {close_prices.iloc[-1]:.2f}")
                print(f"  変動幅: {close_prices.max() - close_prices.min():.2f}")
        
        # レポート生成テスト
        print(f"\n📝 レポート生成テスト:")
        try:
            analysis_result = extract_and_analyze_main_data(stock_data, "9101.T")
            print(f"  分析結果取得: ✅")
            print(f"  取引数: {len(analysis_result.get('trades', []))}")
            
            # 取引詳細の確認
            trades_data = analysis_result.get('trades', [])
            if trades_data:
                print(f"  最初の取引詳細:")
                first_trade = trades_data[0]
                for key, value in first_trade.items():
                    print(f"    {key}: {value}")
        
        except Exception as e:
            print(f"  エラー: {e}")
        
        # 強制決済の確認
        print(f"\n🔧 強制決済処理の確認:")
        if trades:
            forced_exits = [trade for trade in trades if trade.get('is_forced_exit', False)]
            print(f"  強制決済数: {len(forced_exits)}")
            print(f"  通常決済数: {len(trades) - len(forced_exits)}")
            
            if forced_exits:
                print(f"  強制決済の価格:")
                for trade in forced_exits[:5]:  # 最初の5件
                    print(f"    日付: {trade['exit_date']}, 価格: {trade['exit_price']}")
        
    except Exception as e:
        print(f"❌ ファイル読み込みエラー: {e}")
        return
    
    print("\n" + "="*50)
    print("🎯 調査完了")

if __name__ == "__main__":
    main()