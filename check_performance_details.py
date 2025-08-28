"""
パフォーマンス指標シートの詳細確認
"""
import pandas as pd
from pathlib import Path

def check_performance_sheet():
    """パフォーマンス指標シートの詳細確認"""
    results_dir = Path("backtest_results/improved_results")
    excel_files = list(results_dir.glob("improved_backtest_7203.T_*.xlsx"))
    
    if not excel_files:
        print("❌ Excelファイルが見つかりません")
        return
    
    latest_file = max(excel_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 ファイル: {latest_file.name}")
    
    try:
        # パフォーマンス指標の詳細読み取り
        performance_df = pd.read_excel(latest_file, sheet_name='パフォーマンス指標')
        
        print("\n📈 パフォーマンス指標 (全体):")
        print(performance_df.to_string(index=False))
        
        print("\n📊 データ型:")
        print(performance_df.dtypes)
        
        # 主要指標を辞書に変換して詳細確認
        print("\n🔍 指標別詳細:")
        for _, row in performance_df.iterrows():
            key = row['指標']
            value = row['値']
            print(f"   {key}: {repr(value)} (型: {type(value)})")
        
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    check_performance_sheet()
