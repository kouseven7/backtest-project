"""
Excel設定読み込みのデバッグツール

VSCodeボタンとコマンドラインの両方でExcel設定が正しく読み込まれているかをテスト
"""
import pandas as pd
import sys
import os

def test_excel_config():
    excel_file = 'config/backtest_config.xlsm'
    
    print(f"ワーキングディレクトリ: {os.getcwd()}")
    print(f"Excel ファイル存在確認: {os.path.exists(excel_file)}")
    
    try:
        # まずシート名一覧を確認
        xl_file = pd.ExcelFile(excel_file, engine='openpyxl')
        print(f"\n利用可能なシート名: {xl_file.sheet_names}")
        
        # 最初のシートで読み込みテスト
        sheet_name = xl_file.sheet_names[0]
        print(f"シート '{sheet_name}' で読み込み試行...")
        df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
        print(f"Excel読み込み成功!")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data:\n{df}")
        
        # 実際の構造に合わせて設定値抽出
        ticker = df['銘柄'].iloc[0]
        start_date = df['開始日'].iloc[0] 
        end_date = df['終了日'].iloc[0]
        
        print(f"\n=== 設定値確認 ===")
        print(f"ticker: {ticker} (型: {type(ticker)})")
        print(f"start_date: {start_date} (型: {type(start_date)})")
        print(f"end_date: {end_date} (型: {type(end_date)})")
        
        # data_fetcher.py の get_parameters_and_data 関数のような読み込みテスト
        print(f"\n=== data_fetcher.pyスタイル確認 ===")
        from data_fetcher import get_parameters_and_data
        ticker_from_func, start_from_func, end_from_func, _, _ = get_parameters_and_data()
        print(f"data_fetcher.py経由: ticker={ticker_from_func}, start={start_from_func}, end={end_from_func}")
        
    except Exception as e:
        print(f"エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_excel_config()