"""
VSCode実行問題診断スクリプト

VSCode「専用ターミナルでPythonファイルを実行する」ボタンとコマンドライン実行の
結果差異を診断するためのテストスクリプト。

診断項目:
- Excel設定ファイル読み込み成功/失敗
- 実際に読み込まれた設定値の詳細出力
- エラーハンドリングの動作確認
- デフォルト値適用タイミングの検証

Author: Backtest Project Team
Created: 2026-01-08
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# プロジェクトパス設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.error_handling import read_excel_parameters

def diagnose_excel_reading():
    """Excel読み込み診断を実行"""
    
    print("=" * 80)
    print("VSCode実行問題診断開始")
    print(f"実行時刻: {datetime.now()}")
    print(f"作業ディレクトリ: {os.getcwd()}")
    print("=" * 80)
    
    # 設定ファイルパス
    config_base_path = r"C:\Users\imega\Documents\my_backtest_project\config"
    config_file_xlsm = os.path.join(config_base_path, "backtest_config.xlsm")
    config_file_xlsx = os.path.join(config_base_path, "backtest_config.xlsx")
    config_csv = os.path.join(config_base_path, "config.csv")
    
    print("1. ファイル存在確認")
    print(f"   config/backtest_config.xlsm: {os.path.exists(config_file_xlsm)}")
    print(f"   config/backtest_config.xlsx: {os.path.exists(config_file_xlsx)}")
    print(f"   config/config.csv: {os.path.exists(config_csv)}")
    
    # 優先順位に従って読み込みテスト
    print("\n2. Excel読み込みテスト")
    
    try:
        if os.path.exists(config_file_xlsx):
            print(f"   📊 XLSXファイル読み込み試行: {config_file_xlsx}")
            config_df = read_excel_parameters(config_file_xlsx, "銘柄設定")
            print(f"   ✅ XLSX読み込み成功")
        elif os.path.exists(config_file_xlsm):
            print(f"   📊 XLSMファイル読み込み試行: {config_file_xlsm}")
            config_df = read_excel_parameters(config_file_xlsm, "銘柄設定")
            print(f"   ✅ XLSM読み込み成功")
        else:
            print(f"   📄 CSVファイル読み込み試行: {config_csv}")
            config_df = pd.read_csv(config_csv)
            print(f"   ✅ CSV読み込み成功")
            
        # 読み込んだデータの詳細表示
        print("\n3. 読み込まれた設定値")
        print(f"   📋 データ形状: {config_df.shape}")
        print(f"   📋 列名: {list(config_df.columns)}")
        print(f"   📋 データ詳細:")
        for col in config_df.columns:
            value = config_df[col].iloc[0]
            print(f"      {col}: {value} (型: {type(value)})")
            
        # 実際の値抽出テスト
        print("\n4. 値抽出テスト")
        
        # 銘柄
        if "銘柄" in config_df.columns:
            ticker = str(config_df["銘柄"].iloc[0])
            print(f"   🎯 銘柄: {ticker}")
        elif "ticker" in config_df.columns:
            ticker = str(config_df["ticker"].iloc[0])
            print(f"   🎯 銘柄: {ticker}")
        else:
            ticker = "デフォルト値適用"
            print(f"   ⚠️ 銘柄: {ticker}")
            
        # 開始日
        if "開始日" in config_df.columns:
            s = config_df["開始日"].iloc[0]
            if isinstance(s, (pd.Timestamp, datetime)):
                start_date = s.strftime('%Y-%m-%d')
            else:
                start_date = str(s)
            print(f"   📅 開始日: {start_date} (元の型: {type(s)})")
        elif "start_date" in config_df.columns:
            start_date = str(config_df["start_date"].iloc[0])
            print(f"   📅 開始日: {start_date}")
        else:
            start_date = "デフォルト値適用"
            print(f"   ⚠️ 開始日: {start_date}")
            
        # 終了日
        if "終了日" in config_df.columns:
            e = config_df["終了日"].iloc[0]
            if isinstance(e, (pd.Timestamp, datetime)):
                end_date = e.strftime('%Y-%m-%d')
            else:
                end_date = str(e)
            print(f"   📅 終了日: {end_date} (元の型: {type(e)})")
        elif "end_date" in config_df.columns:
            end_date = str(config_df["end_date"].iloc[0])
            print(f"   📅 終了日: {end_date}")
        else:
            end_date = "デフォルト値適用"
            print(f"   ⚠️ 終了日: {end_date}")
            
        print("\n✅ Excel読み込み診断完了 - 成功")
        
    except Exception as e:
        print(f"\n❌ Excel読み込みエラー発生: {str(e)}")
        print(f"   エラータイプ: {type(e)}")
        print(f"   この場合、デフォルト値が適用されます:")
        print(f"      銘柄: 5803.T")
        print(f"      開始日: 2023-01-01")
        print(f"      終了日: 2023-12-31")
        print("   ⚠️ これが2022-2023年期間の原因です！")
    
    print("=" * 80)

if __name__ == "__main__":
    diagnose_excel_reading()