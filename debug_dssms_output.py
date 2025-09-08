#!/usr/bin/env python3
"""
DSSMS出力問題のデバッグスクリプト
問題の根本原因を特定するための調査
"""

import pandas as pd
import sys
import os

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from output.data_extraction_enhancer import extract_and_analyze_main_data

def debug_dssms_output():
    """DSSMS出力問題のデバッグ"""
    print("=== DSSMS出力問題デバッグ開始 ===")
    
    try:
        from data_fetcher import get_parameters_and_data
        from data_processor import preprocess_data
        from indicators.indicator_calculator import compute_indicators
        
        print("\n1. データ取得テスト:")
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        
        print(f"ティッカー: {ticker}")
        print(f"期間: {start_date} ～ {end_date}")
        print(f"データ形状: {stock_data.shape}")
        print(f"実際のデータ期間: {stock_data.index[0]} ～ {stock_data.index[-1]}")
        
        # シグナル確認
        entry_count = len(stock_data[stock_data['Entry_Signal'] == 1])
        exit_count = len(stock_data[stock_data['Exit_Signal'] == -1])
        print(f"エントリーシグナル数: {entry_count}")
        print(f"エグジットシグナル数: {exit_count}")
        
        print("\n2. データ抽出エンハンサーテスト:")
        analyzed_data = extract_and_analyze_main_data(stock_data, ticker)
        
        print("解析結果:")
        print(f"  ティッカー: {analyzed_data['ticker']}")
        print(f"  取引数: {len(analyzed_data['trades'])}")
        print(f"  データ品質: {analyzed_data['data_quality']}")
        
        perf = analyzed_data['performance']
        print(f"パフォーマンス:")
        print(f"  最終価値: {perf['final_portfolio_value']:,.0f}円")
        print(f"  総リターン: {perf['total_return']:.2%}")
        print(f"  取引数: {perf['num_trades']}")
        print(f"  勝率: {perf['win_rate']:.2%}")
        
        # 取引詳細
        if analyzed_data['trades']:
            print(f"\n3. 取引詳細（最初の3件）:")
            for i, trade in enumerate(analyzed_data['trades'][:3]):
                print(f"  取引{i+1}: エントリー={trade['entry_date']} "
                      f"エグジット={trade['exit_date']} "
                      f"PnL={trade['pnl']:,.0f}円")
        
        print("\n4. Excel出力テスト:")
        # Excel出力処理をテスト
        from output.simple_excel_exporter import ExcelDataProcessor
        processor = ExcelDataProcessor()
        excel_data = processor.process_main_data(stock_data, ticker)
        
        print("Excel用データ構造:")
        print(f"  メタデータ: {excel_data['metadata']}")
        print(f"  サマリー: {excel_data['summary']}")
        print(f"  取引数: {len(excel_data['trades'])}")
        
        return True
        
    except Exception as e:
        print(f"デバッグエラー: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    debug_dssms_output()
