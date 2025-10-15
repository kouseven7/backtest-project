"""
強化ポジション管理機能テスト用のシンプルなテストプログラム
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# 必要なモジュールのインポート
from strategies.enhanced_base_strategy import EnhancedBaseStrategy
from strategies.Opening_Gap_Enhanced import OpeningGapEnhancedStrategy
from config.error_handling import fetch_stock_data
import src.utils.yfinance_lazy_wrapper as yf

def test_enhanced_position_tracking():
    """
    EnhancedBaseStrategyのポジション管理機能をテストする
    """
    print("強化ポジション管理機能テスト開始")
    
    # テスト用のデータ取得
    start_date = '2024-01-01'
    end_date = '2024-02-29'  # 期間を短縮してテスト
    ticker = '7203.T'  # トヨタ
    index_ticker = '^N225'  # 日経平均
    
    try:
        # 株価データ取得
        print(f"{ticker}の株価データを取得中...")
        stock_data = yf.download(ticker, start_date, end_date)
        if stock_data is None or stock_data.empty:
            print("株価データの取得に失敗しました")
            return
        
        # インデックスデータ取得
        print(f"{index_ticker}のインデックスデータを取得中...")
        index_data = yf.download(index_ticker, start_date, end_date)
        if index_data is None or index_data.empty:
            print("インデックスデータの取得に失敗しました")
            return
        
        # OpeningGapEnhancedStrategyを初期化
        params = {
            'gap_threshold': 0.01,  # より小さな閾値でシグナルが出やすくする
            'profit_target': 0.03,
            'stop_loss': 0.02,
            'trailing_threshold': 0.015,
            'max_hold_days': 5
        }
        
        print("OpeningGapEnhancedStrategy初期化中...")
        price_column = "Adj Close" if "Adj Close" in stock_data.columns else "Close"
        
        strategy = OpeningGapEnhancedStrategy(
            data=stock_data,
            dow_data=index_data,
            params=params,
            price_column=price_column
        )
        
        # バックテスト実行
        print("バックテスト実行中...")
        result = strategy.backtest_with_position_tracking()  # 拡張バックテストメソッドを使用
        
        # 結果確認
        entry_signals = result[result['Entry_Signal'] == 1]
        exit_signals = result[result['Exit_Signal'] != 0]
        
        print(f"\n=== バックテスト結果 ===")
        print(f"テスト期間: {start_date} から {end_date}")
        print(f"データ行数: {len(result)}")
        print(f"エントリー回数: {len(entry_signals)}")
        print(f"エグジット回数: {len(exit_signals)}")
        print(f"未決済残: {len(entry_signals) - len(exit_signals)}")
        
        # ポジションサイズ列の確認
        if 'Position_Size' in result.columns:
            position_size_changes = result[result['Position_Size'].diff() != 0]
            print(f"\nポジションサイズの変更回数: {len(position_size_changes)}")
            
            # ポジションサイズの変更履歴を表示
            print("\nポジションサイズの変更履歴:")
            for idx, row in position_size_changes.iterrows():
                try:
                    position = float(row['Position_Size'])
                    entry = int(row['Entry_Signal']) == 1
                    exit_val = int(row['Exit_Signal']) != 0
                    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                    print(f"日付: {date_str}, 値: {position:.1f}, " + 
                          f"変化: {'エントリー' if entry else ('エグジット' if exit_val else '部分決済')}")
                except Exception as e:
                    print(f"表示エラー: {e}")
        
        # CSVに結果を保存
        result.to_csv('position_tracking_test_results.csv')
        print(f"\nテスト結果をCSVとして保存しました: position_tracking_test_results.csv")
        
        # 簡易ログ出力
        with open('position_tracking_test_log.txt', 'w') as f:
            f.write(f"=== バックテスト結果 ===\n")
            f.write(f"テスト期間: {start_date} から {end_date}\n")
            f.write(f"データ行数: {len(result)}\n")
            f.write(f"エントリー回数: {len(entry_signals)}\n")
            f.write(f"エグジット回数: {len(exit_signals)}\n")
            f.write(f"未決済残: {len(entry_signals) - len(exit_signals)}\n")
            
            # ポジションサイズの変更履歴を記録
            if 'Position_Size' in result.columns:
                position_size_changes_list = []
                for idx, row in position_size_changes.iterrows():
                    try:
                        date_str = str(idx)
                        position = float(row['Position_Size'])
                        position_size_changes_list.append(f"{date_str}: {position:.1f}")
                    except Exception as e:
                        position_size_changes_list.append(f"エラー: {e}")
                
                f.write("\nポジションサイズ変更履歴:\n")
                f.write("\n".join(position_size_changes_list[:10]))  # 最初の10件のみ
                if len(position_size_changes_list) > 10:
                    f.write("\n...など他にも変更あり")
        
        print("テスト結果のログを'position_tracking_test_log.txt'に保存しました")
        
        print("テスト完了")
        return True
        
    except Exception as e:
        print(f"テスト実行中にエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_position_tracking()