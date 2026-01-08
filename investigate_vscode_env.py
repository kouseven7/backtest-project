"""
VSCodeボタン実行環境調査ツール

VSCodeの「専用ターミナルでPythonファイルを実行する」ボタンと
コマンドライン実行の環境差異を調査する
"""
import os
import sys
import pandas as pd
from datetime import datetime

def investigate_execution_environment():
    """実行環境の詳細調査"""
    print("=" * 80)
    print("VSCode実行環境調査 - " + str(datetime.now()))
    print("=" * 80)
    
    # 1. 基本環境情報
    print("\n[1. 基本環境情報]")
    print(f"Python実行パス: {sys.executable}")
    print(f"作業ディレクトリ: {os.getcwd()}")
    print(f"スクリプトパス: {__file__}")
    print(f"Pythonバージョン: {sys.version}")
    
    # 2. 環境変数
    print("\n[2. 重要な環境変数]")
    important_vars = ['PATH', 'PYTHONPATH', 'VIRTUAL_ENV']
    for var in important_vars:
        value = os.environ.get(var, 'NOT_SET')
        if len(str(value)) > 100:
            print(f"{var}: {str(value)[:100]}...")
        else:
            print(f"{var}: {value}")
    
    # 3. Excel設定読み込みテスト
    print("\n[3. Excel設定読み込みテスト]")
    excel_file = 'config/backtest_config.xlsm'
    print(f"Excel ファイル存在: {os.path.exists(excel_file)}")
    
    try:
        xl_file = pd.ExcelFile(excel_file, engine='openpyxl')
        print(f"シート名: {xl_file.sheet_names}")
        
        sheet_name = xl_file.sheet_names[0]
        df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
        print(f"データ: {df.to_dict('records')[0] if not df.empty else 'EMPTY'}")
        
        # data_fetcher.pyスタイルテスト
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, _, _ = get_parameters_and_data()
        print(f"data_fetcher.py結果: ticker={ticker}, start={start_date}, end={end_date}")
        
    except Exception as e:
        print(f"Excel読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. main_new.py内のデフォルト値確認
    print("\n[4. main_new.py内のデフォルト設定確認]")
    try:
        with open('main_new.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # デフォルト値を検索
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '5803.T' in line or 'AAPL' in line:
                print(f"Line {i+1}: {line.strip()}")
                
    except Exception as e:
        print(f"main_new.py読み込みエラー: {e}")
    
    print("\n" + "=" * 80)
    print("調査完了")
    print("=" * 80)

if __name__ == "__main__":
    investigate_execution_environment()