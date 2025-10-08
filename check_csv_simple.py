#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV出力の実際の品質確認スクリプト
"""

import pandas as pd
import glob

def check_csv_output():
    """最新のCSVファイルの品質確認"""
    
    # 最新のCSVファイルを取得
    csv_files = glob.glob('backtest_results/improved_results/*_fallback.csv')
    if not csv_files:
        print("CSVファイルが見つかりません")
        return
    
    latest_file = max(csv_files)
    print(f"[SEARCH] 検査対象ファイル: {latest_file}")
    
    # ファイル読み込み
    df = pd.read_csv(latest_file)
    print(f"[CHART] ファイル形状: {df.shape}")
    
    # N/A値チェック
    na_count = df.isna().sum().sum()
    print(f"[ERROR] N/A値の総数: {na_count}")
    
    # 内容確認
    print("\n[LIST] CSV内容:")
    for i, row in df.iterrows():
        item = row['項目'] if '項目' in df.columns else str(row.iloc[0])
        value = row['値'] if '値' in df.columns else str(row.iloc[1])
        
        print(f"  {i+1}. {item}: {value[:100]}{'...' if len(str(value)) > 100 else ''}")
    
    # 実際のバックテストデータが含まれているかチェック
    data_content = None
    for i, row in df.iterrows():
        if 'データ内容' in str(row.iloc[0]):
            data_content = str(row.iloc[1])
            break
    
    if data_content:
        print(f"\n[TARGET] バックテストデータの確認:")
        print(f"  - データ長: {len(data_content)} 文字")
        print(f"  - 価格データ含有: {'Close' in data_content and 'Open' in data_content}")
        print(f"  - シグナル含有: {'Entry_Signal' in data_content or 'Exit_Signal' in data_content}")
        print(f"  - 日付範囲: {'2024' in data_content}")
        
        # 数値データの存在確認
        import re
        numeric_values = re.findall(r'\d+\.\d+', data_content)
        print(f"  - 数値データ数: {len(numeric_values)} 個")
        if numeric_values:
            print(f"  - サンプル価格: {numeric_values[:5]}")
    
    print(f"\n[OK] 結論: CSVフォールバック出力は{'成功' if na_count == 0 and data_content else '問題あり'}")
    return na_count == 0 and data_content is not None

if __name__ == "__main__":
    check_csv_output()