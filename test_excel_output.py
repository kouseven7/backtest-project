#!/usr/bin/env python3
"""
Excel出力処理のテストスクリプト
修正版の動作確認
"""

import pandas as pd
import sys
import os
from datetime import datetime

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: def test_excel_output():
    """Excel出力処理のテスト"""
    print("=== Excel出力処理テスト開始 ===")
    
    try:
        from output.simple_excel_exporter import save_backtest_results_simple
        
        # 1. main.pyから戦略適用済みのデータを取得
        print("\n1. 戦略適用済みデータ取得:")
        from data_fetcher import get_parameters_and_data
        from data_processor import preprocess_data
        from indicators.indicator_calculator import compute_indicators
        
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        stock_data = preprocess_data(stock_data)
        stock_data = compute_indicators(stock_data)
        
        print(f"基本データ形状: {stock_data.shape}")
        print(f"基本データ列: {stock_data.columns.tolist()}")
        
        # 2. 戦略シグナル列を模擬的に追加（main.pyの処理を簡略化）
        print("\n2. 戦略シグナル列追加（模擬）:")
        stock_data['Entry_Signal'] = 0
        stock_data['Exit_Signal'] = 0  
        stock_data['Strategy'] = ''
        
        # 簡単なテスト用シグナルを追加
        stock_data.loc[stock_data.index[10], 'Entry_Signal'] = 1
        stock_data.loc[stock_data.index[10], 'Strategy'] = 'TestStrategy'
        stock_data.loc[stock_data.index[20], 'Exit_Signal'] = -1
        
        print(f"シグナル追加後データ形状: {stock_data.shape}")
        print(f"エントリーシグナル数: {len(stock_data[stock_data['Entry_Signal'] == 1])}")
        print(f"エグジットシグナル数: {len(stock_data[stock_data['Exit_Signal'] == -1])}")
        
        # 3. Excel出力テスト
        print("\n3. Excel出力実行:")
        filepath = save_backtest_results_simple(
            stock_data=stock_data,
            ticker=ticker,
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: filename=f"test_excel_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        
        if filepath:
            print(f"✅ Excel出力成功: {filepath}")
            return True
        else:
            print("❌ Excel出力失敗")
            return False
            
    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: test_excel_output()
