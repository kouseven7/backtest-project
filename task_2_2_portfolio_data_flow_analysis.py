#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 2.2: ポートフォリオ価値データフローの追跡

DSSMSBacktester.portfolio_values の生データから
Excel出力までの変換過程を詳細追跡し、
データ変換時の問題点を特定する
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import traceback

def analyze_portfolio_data_flow():
    """ポートフォリオデータの変換過程を追跡"""
    
    results = {
        "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "findings": {},
        "data_flow_analysis": {},
        "problems_identified": [],
        "recommendations": []
    }
    
    print("🔍 Task 2.2: ポートフォリオ価値データフロー追跡を開始")
    print("=" * 60)
    
    # 1. DSSMSBacktester.portfolio_values の生データ確認
    print("\n📊 Step 1: DSSMSBacktesterの生データ分析")
    try:
        # DSSMSBacktesterクラスを調査
        backtester_path = Path("src/dssms/dssms_backtester.py")
        if backtester_path.exists():
            with open(backtester_path, 'r', encoding='utf-8') as f:
                backtester_content = f.read()
            
            # portfolio_valuesの使用箇所を検索
            portfolio_usage = []
            for i, line in enumerate(backtester_content.split('\n'), 1):
                if 'portfolio_values' in line.lower():
                    portfolio_usage.append(f"Line {i}: {line.strip()}")
            
            results["findings"]["backtester_portfolio_usage"] = {
                "total_occurrences": len(portfolio_usage),
                "usage_lines": portfolio_usage[:10]  # 最初の10件
            }
            
            print(f"✅ DSSMSBacktesterでのportfolio_values使用箇所: {len(portfolio_usage)}件")
            for usage in portfolio_usage[:5]:
                print(f"   {usage}")
        else:
            results["problems_identified"].append("DSSMSBacktesterファイルが見つからない")
            print("❌ DSSMSBacktesterファイルが見つかりません")
            
    except Exception as e:
        print(f"❌ DSSMSBacktester分析エラー: {e}")
        results["problems_identified"].append(f"DSSMSBacktester分析エラー: {e}")
    
    # 2. _convert_backtester_results での変換過程分析
    print("\n🔄 Step 2: _convert_backtester_results変換過程分析")
    try:
        # 各統一エンジンでの変換ロジックを比較
        engine_files = [
            "dssms_unified_output_engine.py",
            "dssms_unified_output_engine_fixed.py", 
            "dssms_unified_output_engine_fixed_v3.py",
            "dssms_unified_output_engine_fixed_v4.py"
        ]
        
        conversion_analysis = {}
        for engine_file in engine_files:
            engine_path = Path(engine_file)
            if engine_path.exists():
                with open(engine_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # _convert_backtester_resultsメソッドを検索
                method_lines = []
                in_method = False
                for i, line in enumerate(content.split('\n'), 1):
                    if '_convert_backtester_results' in line and 'def' in line:
                        in_method = True
                        method_lines.append(f"Line {i}: {line.strip()}")
                    elif in_method and line.strip().startswith('def ') and '_convert_backtester_results' not in line:
                        break
                    elif in_method:
                        method_lines.append(f"Line {i}: {line.strip()}")
                
                conversion_analysis[engine_file] = {
                    "method_found": len(method_lines) > 0,
                    "method_lines_count": len(method_lines),
                    "first_10_lines": method_lines[:10]
                }
                
                print(f"📁 {engine_file}: {'✅' if len(method_lines) > 0 else '❌'} _convert_backtester_results")
                
        results["data_flow_analysis"]["conversion_methods"] = conversion_analysis
        
    except Exception as e:
        print(f"❌ 変換過程分析エラー: {e}")
        results["problems_identified"].append(f"変換過程分析エラー: {e}")
    
    # 3. _fix_date_inconsistencies での修正過程分析
    print("\n🛠️ Step 3: _fix_date_inconsistencies修正過程分析")
    try:
        date_fix_analysis = {}
        for engine_file in engine_files:
            engine_path = Path(engine_file)
            if engine_path.exists():
                with open(engine_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # _fix_date_inconsistenciesメソッドを検索
                fix_lines = []
                in_method = False
                for i, line in enumerate(content.split('\n'), 1):
                    if '_fix_date_inconsistencies' in line and 'def' in line:
                        in_method = True
                        fix_lines.append(f"Line {i}: {line.strip()}")
                    elif in_method and line.strip().startswith('def ') and '_fix_date_inconsistencies' not in line:
                        break
                    elif in_method:
                        fix_lines.append(f"Line {i}: {line.strip()}")
                
                # pd.to_datetimeの使用箇所を特別に検索
                datetime_usage = []
                for i, line in enumerate(content.split('\n'), 1):
                    if 'pd.to_datetime' in line or 'pd.to_datetime' in line:
                        datetime_usage.append(f"Line {i}: {line.strip()}")
                
                date_fix_analysis[engine_file] = {
                    "fix_method_found": len(fix_lines) > 0,
                    "fix_method_lines": len(fix_lines),
                    "datetime_conversions": len(datetime_usage),
                    "datetime_usage_sample": datetime_usage[:5]
                }
                
                print(f"📁 {engine_file}:")
                print(f"   _fix_date_inconsistencies: {'✅' if len(fix_lines) > 0 else '❌'}")
                print(f"   pd.to_datetime使用箇所: {len(datetime_usage)}件")
                
        results["data_flow_analysis"]["date_fix_methods"] = date_fix_analysis
        
    except Exception as e:
        print(f"❌ 日付修正分析エラー: {e}")
        results["problems_identified"].append(f"日付修正分析エラー: {e}")
    
    # 4. Excel出力での最終データ確認
    print("\n📊 Step 4: Excel出力最終データ確認")
    try:
        # 最新のExcelファイルを検索
        excel_files = list(Path("backtest_results/dssms_results").glob("*.xlsx"))
        excel_files = [f for f in excel_files if not f.name.startswith("~$")]  # ロックファイル除外
        
        if excel_files:
            latest_excel = max(excel_files, key=lambda x: x.stat().st_mtime)
            print(f"📄 最新Excelファイル: {latest_excel.name}")
            
            # Excelファイルの損益推移シートを分析
            try:
                # openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl
                wb = openpyxl.load_workbook(latest_excel)
                
                excel_analysis = {
                    "file_name": latest_excel.name,
                    "sheet_names": wb.sheetnames,
                    "sheets_analysis": {}
                }
                
                # 損益推移シートの分析
                if "損益推移" in wb.sheetnames:
                    ws = wb["損益推移"]
                    # 最初の10行のデータを確認
                    portfolio_data = []
                    for row in range(2, min(12, ws.max_row + 1)):  # ヘッダー除く最初10行
                        row_data = {}
                        for col in range(1, min(5, ws.max_column + 1)):  # 最初4列
                            cell_value = ws.cell(row=row, column=col).value
                            row_data[f"col_{col}"] = str(cell_value) if cell_value is not None else ""
                        portfolio_data.append(row_data)
                    
                    excel_analysis["sheets_analysis"]["損益推移"] = {
                        "max_row": ws.max_row,
                        "max_column": ws.max_column,
                        "sample_data": portfolio_data
                    }
                    
                    print(f"✅ 損益推移シート: {ws.max_row}行 x {ws.max_column}列")
                
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: results["data_flow_analysis"]["excel_output"] = excel_analysis
                
            except Exception as excel_error:
                print(f"❌ Excel分析エラー: {excel_error}")
                results["problems_identified"].append(f"Excel分析エラー: {excel_error}")
                
        else:
            print("❌ Excelファイルが見つかりません")
            results["problems_identified"].append("Excelファイルが見つからない")
            
    except Exception as e:
        print(f"❌ Excel出力分析エラー: {e}")
        results["problems_identified"].append(f"Excel出力分析エラー: {e}")
    
    # 5. データフロー問題の特定
    print("\n🎯 Step 5: データフロー問題の特定")
    
    # Task 2.1の結果と連携して問題を特定
    task_2_1_results_file = None
    for file in Path(".").glob("task_2_1_results_*.json"):
        task_2_1_results_file = file
        break
    
    if task_2_1_results_file:
        try:
            with open(task_2_1_results_file, 'r', encoding='utf-8') as f:
                task_2_1_data = json.load(f)
            
            # Task 2.1で発見された日付ループ問題との関連性を分析
            date_loop_impact = {
                "task_2_1_findings": task_2_1_data.get("problems_identified", []),
                "portfolio_data_impact": []
            }
            
            # 日付ループがポートフォリオデータに与える影響を分析
            if "origin='2023-01-01'" in str(task_2_1_data):
                date_loop_impact["portfolio_data_impact"].append(
                    "Task 2.1で発見されたorigin='2023-01-01'固定問題がポートフォリオ価値データにも影響"
                )
            
            results["data_flow_analysis"]["date_loop_correlation"] = date_loop_impact
            
        except Exception as e:
            print(f"⚠️ Task 2.1結果読み込みエラー: {e}")
    
    # 推奨事項の生成
    print("\n💡 推奨事項:")
    recommendations = [
        "DSSMSBacktester.portfolio_valuesの生成ロジック詳細調査",
        "各エンジンでの変換ロジック統一",
        "日付変換時のorigin問題解決（Task 2.1連携）",
        "Excel出力時のデータ整合性検証強化"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
        results["recommendations"].append(rec)
    
    # 結果をJSONファイルに保存
    output_file = f"task_2_2_results_{results['analysis_timestamp']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 分析結果を保存: {output_file}")
    print("=" * 60)
    print("🔍 Task 2.2: ポートフォリオ価値データフロー追跡完了")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_portfolio_data_flow()
        
        # 重要な発見事項を表示
        print("\n🎯 重要な発見事項:")
        for problem in results["problems_identified"]:
            print(f"❌ {problem}")
            
    except Exception as e:
        print(f"❌ Task 2.2実行エラー: {e}")
        traceback.print_exc()
