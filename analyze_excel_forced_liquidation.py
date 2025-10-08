"""
Excel出力データの直接調査スクリプト
強制決済価格問題の分析
"""

import pandas as pd
import json
from pathlib import Path

def analyze_excel_file():
    """生成されたExcelファイルを直接調査"""
    
    excel_file = r"C:\Users\imega\Documents\my_backtest_project\backtest_results\improved_results\improved_backtest_5803.T_20251008_121813.xlsx"
    
    try:
        # Excelファイル読み込み
        print(f"=== Excel分析: {Path(excel_file).name} ===")
        
        # 取引履歴シートを読み込み
        trades_df = pd.read_excel(excel_file, sheet_name='取引履歴')
        
        print(f"取引数: {len(trades_df)}件")
        print()
        
        # エグジット価格の分析
        if 'エグジット価格' in trades_df.columns:
            exit_prices = trades_df['エグジット価格']
            unique_exit_prices = exit_prices.unique()
            
            print("=== エグジット価格分析 ===")
            print(f"ユニークなエグジット価格数: {len(unique_exit_prices)}")
            print(f"全エグジット価格が同一: {len(unique_exit_prices) == 1}")
            
            if len(unique_exit_prices) <= 5:
                print("エグジット価格一覧:")
                for price in unique_exit_prices:
                    count = (exit_prices == price).sum()
                    print(f"  {price}: {count}件")
            
            print(f"6438.89での決済数: {(exit_prices == 6438.89).sum()}件")
            
        # エントリー価格の分析
        if 'エントリー価格' in trades_df.columns:
            entry_prices = trades_df['エントリー価格']
            unique_entry_prices = entry_prices.unique()
            
            print()
            print("=== エントリー価格分析 ===")
            print(f"ユニークなエントリー価格数: {len(unique_entry_prices)}")
            print(f"価格範囲: {entry_prices.min():.2f} - {entry_prices.max():.2f}")
            
        # 保有日数の分析
        if '保有日数' in trades_df.columns:
            holding_days = trades_df['保有日数']
            
            print()
            print("=== 保有日数分析 ===")
            print(f"平均保有日数: {holding_days.mean():.2f}日")
            print(f"保有日数範囲: {holding_days.min()} - {holding_days.max()}日")
            print(f"保有日数0日の取引数: {(holding_days == 0).sum()}件")
            
        # 先頭10件表示
        print()
        print("=== 取引履歴サンプル（先頭10件）===")
        display_columns = ['取引ID', 'エントリー日', 'エグジット日', 'エントリー価格', 'エグジット価格', '保有日数']
        available_columns = [col for col in display_columns if col in trades_df.columns]
        print(trades_df[available_columns].head(10).to_string(index=False))
        
        # 後半10件表示
        print()
        print("=== 取引履歴サンプル（後半10件）===")
        print(trades_df[available_columns].tail(10).to_string(index=False))
        
        return {
            'total_trades': len(trades_df),
            'unique_exit_prices': len(unique_exit_prices) if 'エグジット価格' in trades_df.columns else 0,
            'all_same_exit_price': len(unique_exit_prices) == 1 if 'エグジット価格' in trades_df.columns else False,
            'target_price_count': (trades_df['エグジット価格'] == 6438.89).sum() if 'エグジット価格' in trades_df.columns else 0
        }
        
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {excel_file}")
        return None
    except Exception as e:
        print(f"Excel読み込みエラー: {e}")
        return None

if __name__ == "__main__":
    result = analyze_excel_file()
    if result:
        print()
        print("=== 分析結果サマリー ===")
        for key, value in result.items():
            print(f"{key}: {value}")