#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excel出力データ確認スクリプト
修正後のExcel出力が正しく582.86%を表示しているか確認
"""

# openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl
from pathlib import Path
from typing import List

# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: def check_excel_output():
    """最新のExcelファイルの内容を確認"""
    
    # 最新のExcelファイルを特定
    excel_dir = Path("output/dssms_integration")
    excel_files = list(excel_dir.glob("backtest_results_*.xlsx"))
    
    if not excel_files:
        print("[ERROR] Excelファイルが見つかりません")
        return
    
    # 最新ファイルを選択（ファイル名でソート）
    latest_file = sorted(excel_files, key=lambda x: x.name)[-1]
    print(f"[CHART] 確認対象: {latest_file}")
    
    try:
        # Excelファイルを開く
        workbook = openpyxl.load_workbook(latest_file)
        
        print(f"\n=== シート一覧 ===")
        for sheet_name in workbook.sheetnames:
            print(f"  - {sheet_name}")
        
        # サマリーシートの確認
        if "サマリー" in workbook.sheetnames:
            ws = workbook["サマリー"]
            print(f"\n=== サマリーシート内容 ===")
            
            # 重要な指標を探す
            for row in range(1, 30):  # 最初の30行をチェック
                for col in range(1, 4):  # A-C列をチェック
                    cell_value = ws.cell(row=row, column=col).value
                    if cell_value:
                        # 次のセルの値も取得
                        next_cell = ws.cell(row=row, column=col+1).value if col < 3 else None
                        
                        if isinstance(cell_value, str):
                            if "最終ポートフォリオ価値" in cell_value:
                                print(f"[OK] {cell_value}: {next_cell}")
                            elif "総リターン" in cell_value and "年率" not in cell_value:
                                print(f"[OK] {cell_value}: {next_cell}")
                            elif "年率リターン" in cell_value:
                                print(f"[OK] {cell_value}: {next_cell}")
                            elif "銘柄切替回数" in cell_value:
                                print(f"[OK] {cell_value}: {next_cell}")
                            elif "シャープレシオ" in cell_value:
                                print(f"[OK] {cell_value}: {next_cell}")
                            elif "最大ドローダウン" in cell_value:
                                print(f"[OK] {cell_value}: {next_cell}")
        
        # パフォーマンス指標シートの確認
        if "パフォーマンス指標" in workbook.sheetnames:
            ws = workbook["パフォーマンス指標"]
            print(f"\n=== パフォーマンス指標シート内容 ===")
            
            for row in range(1, 20):
                for col in range(1, 5):
                    cell_value = ws.cell(row=row, column=col).value
                    if cell_value and isinstance(cell_value, str):
                        if "総リターン" in cell_value or "シャープレシオ" in cell_value:
                            # その行の他の値も表示
                            row_data: List[str] = []
                            for c in range(1, 5):
                                val = ws.cell(row=row, column=c).value
                                row_data.append(str(val) if val else "")
                            print(f"  {' | '.join(row_data)}")
        
        # 取引履歴シートの確認
        if "取引履歴" in workbook.sheetnames:
            ws = workbook["取引履歴"]
            print(f"\n=== 取引履歴シート概要 ===")
            
            # 総行数を確認
            max_row = ws.max_row
            print(f"  総行数: {max_row}行")
            
            # 最初の数行と最後の数行を表示
            if max_row > 1:
                print(f"  最初の取引履歴:")
                for row in range(2, min(5, max_row+1)):  # ヘッダーを除く最初の3行
                    date_val = ws.cell(row=row, column=1).value
                    profit_val = ws.cell(row=row, column=8).value  # 損益列
                    cumulative_val = ws.cell(row=row, column=9).value  # 累積損益列
                    print(f"    {date_val}: 損益={profit_val}, 累積={cumulative_val}")
                
                print(f"  最後の取引履歴:")
                for row in range(max(max_row-2, 2), max_row+1):
                    date_val = ws.cell(row=row, column=1).value
                    profit_val = ws.cell(row=row, column=8).value
                    cumulative_val = ws.cell(row=row, column=9).value
                    print(f"    {date_val}: 損益={profit_val}, 累積={cumulative_val}")
        
        workbook.close()
        print(f"\n[OK] Excel出力確認完了!")
        
    except Exception as e:
        print(f"[ERROR] Excel読み込みエラー: {e}")

if __name__ == "__main__":
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: check_excel_output()
