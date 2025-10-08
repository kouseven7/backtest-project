"""
ExcelファイルのシートとSample内容を確認するクイック検査スクリプト
"""
import pandas as pd
from pathlib import Path

def check_excel_structure():
    """Excelファイルの構造を確認"""
    results_dir = Path("backtest_results/improved_results")
    excel_files = list(results_dir.glob("improved_backtest_7203.T_*.xlsx"))
    
    if not excel_files:
        print("[ERROR] Excelファイルが見つかりません")
        return
    
    latest_file = max(excel_files, key=lambda x: x.stat().st_mtime)
    print(f"[CHART] ファイル: {latest_file.name}")
    
    try:
        # ExcelファイルのSheet一覧を取得
        excel_file = pd.ExcelFile(latest_file)
        print(f"\n[LIST] 利用可能なシート:")
        for i, sheet_name in enumerate(excel_file.sheet_names):
            print(f"   {i+1}. {sheet_name}")
        
        # 各シートの内容をサンプル表示
        for sheet_name in excel_file.sheet_names[:3]:  # 最初の3シートのみ
            print(f"\n[UP] Sheet '{sheet_name}' Sample:")
            df = pd.read_excel(latest_file, sheet_name=sheet_name)
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   First 3 rows:")
            print(df.head(3).to_string(index=False))
            print("-" * 50)
        
    except Exception as e:
        print(f"[ERROR] エラー: {e}")

if __name__ == "__main__":
    check_excel_structure()
