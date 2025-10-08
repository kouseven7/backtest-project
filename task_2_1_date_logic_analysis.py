#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 2.1: 日付処理ロジックの検証
「2023-12-31 → 2023-01-01」の不正ループ原因を特定
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import re

def analyze_date_processing_logic():
    """日付処理ロジックの詳細検証"""
    print("[SEARCH] Task 2.1: 日付処理ロジックの検証")
    print("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'engine_date_fixes': {},
        'excel_data_analysis': {},
        'date_loop_detection': {}
    }
    
    # 1. 各エンジンの日付修正ロジック比較
    analyze_date_fixing_methods(results)
    
    # 2. 問題のExcelファイルの日付データ分析
    analyze_excel_date_data(results)
    
    # 3. 日付ループパターンの検出
    detect_date_loop_patterns(results)
    
    return results

def analyze_date_fixing_methods(results):
    """各エンジンの日付修正ロジック比較"""
    print("\n📅 1. 各エンジンの日付修正ロジック比較")
    print("-" * 40)
    
    engines_to_analyze = [
        'dssms_unified_output_engine_fixed.py',
        'dssms_unified_output_engine_fixed_v4.py'
    ]
    
    date_fix_analysis = {}
    
    for engine_name in engines_to_analyze:
        if Path(engine_name).exists():
            print(f"\n[LIST] {engine_name} の日付修正分析:")
            
            try:
                with open(engine_name, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # _fix_date_inconsistencies メソッドの抽出
                method_pattern = r'def _fix_date_inconsistencies.*?(?=def |\Z)'
                matches = re.findall(method_pattern, content, re.DOTALL)
                
                if matches:
                    method_content = matches[0]
                    lines = method_content.split('\n')
                    
                    # 重要な処理を特定
                    important_lines = []
                    for line in lines[:50]:  # 最初の50行
                        if any(keyword in line.lower() for keyword in ['expected_year', '2023', 'year', 'pd.to_datetime', 'loop']):
                            important_lines.append(line.strip())
                    
                    print(f"  [CHART] メソッド行数: {len(lines)}")
                    print(f"  [TARGET] 重要な処理: {len(important_lines)}件")
                    
                    for i, line in enumerate(important_lines[:10]):
                        print(f"    {i+1}: {line}")
                    
                    # expected_year = 2023 の無限ループ可能性チェック
                    expected_year_lines = [line for line in lines if 'expected_year' in line and '2023' in line]
                    if expected_year_lines:
                        print(f"  [WARNING] expected_year = 2023 の設定発見:")
                        for line in expected_year_lines:
                            print(f"    → {line.strip()}")
                    
                    # 年末→年始境界処理の確認
                    boundary_keywords = ['december', 'january', '12-31', '01-01', 'year_end', 'year_start']
                    boundary_lines = []
                    for line in lines:
                        if any(keyword in line.lower() for keyword in boundary_keywords):
                            boundary_lines.append(line.strip())
                    
                    if boundary_lines:
                        print(f"  📅 年末年始境界処理: {len(boundary_lines)}件")
                        for line in boundary_lines[:3]:
                            print(f"    → {line}")
                    
                    date_fix_analysis[engine_name] = {
                        'method_lines': len(lines),
                        'important_lines': important_lines,
                        'expected_year_usage': expected_year_lines,
                        'boundary_processing': boundary_lines
                    }
                    
                else:
                    print(f"  [ERROR] _fix_date_inconsistencies メソッドが見つかりません")
                    date_fix_analysis[engine_name] = {'error': 'method_not_found'}
                    
            except Exception as e:
                print(f"  [ERROR] 分析エラー: {e}")
                date_fix_analysis[engine_name] = {'error': str(e)}
        else:
            print(f"[ERROR] {engine_name} が見つかりません")
            date_fix_analysis[engine_name] = {'exists': False}
    
    results['engine_date_fixes'] = date_fix_analysis

def analyze_excel_date_data(results):
    """問題のExcelファイルの日付データ分析"""
    print(f"\n[CHART] 2. Excelファイルの日付データ分析")
    print("-" * 40)
    
    excel_file = "backtest_results/dssms_results/dssms_unified_backtest_20250910_213413.xlsx"
    
    try:
        if Path(excel_file).exists():
            print(f"[LIST] 分析対象: {excel_file}")
            
            # Excelファイルのシート一覧を取得
            excel_file_obj = pd.ExcelFile(excel_file)
            sheet_names = excel_file_obj.sheet_names
            print(f"[CHART] シート数: {len(sheet_names)}")
            print(f"[LIST] シート名: {', '.join(sheet_names)}")
            
            date_analysis = {}
            
            # 各シートの日付データを分析
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    print(f"\n[LIST] {sheet_name} シート分析:")
                    print(f"  行数: {len(df)}, 列数: {len(df.columns)}")
                    
                    # 日付列を特定
                    date_columns = []
                    for col in df.columns:
                        if any(keyword in col.lower() for keyword in ['date', '日付', 'time', '時刻']):
                            date_columns.append(col)
                    
                    print(f"  日付関連列: {date_columns}")
                    
                    # 日付データの詳細分析
                    if date_columns:
                        for date_col in date_columns:
                            if date_col in df.columns:
                                date_values = df[date_col].dropna()
                                if len(date_values) > 0:
                                    print(f"    {date_col}:")
                                    print(f"      データ数: {len(date_values)}")
                                    print(f"      最初の3件: {list(date_values.head(3))}")
                                    print(f"      最後の3件: {list(date_values.tail(3))}")
                                    
                                    # 年末年始のループパターンを検索
                                    date_strings = date_values.astype(str)
                                    loop_patterns = []
                                    
                                    for i, date_str in enumerate(date_strings):
                                        if '2023-12-31' in date_str or '2023-01-01' in date_str:
                                            loop_patterns.append((i, date_str))
                                    
                                    if loop_patterns:
                                        print(f"      [WARNING] 年末年始パターン: {len(loop_patterns)}件")
                                        for idx, pattern in loop_patterns[:5]:
                                            print(f"        {idx}: {pattern}")
                    
                    date_analysis[sheet_name] = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'date_columns': date_columns,
                        'date_patterns': loop_patterns if 'loop_patterns' in locals() else []
                    }
                    
                except Exception as e:
                    print(f"  [ERROR] {sheet_name} 読み込みエラー: {e}")
                    date_analysis[sheet_name] = {'error': str(e)}
            
            results['excel_data_analysis'] = date_analysis
            
        else:
            print(f"[ERROR] Excelファイルが見つかりません: {excel_file}")
            results['excel_data_analysis'] = {'error': 'file_not_found'}
            
    except Exception as e:
        print(f"[ERROR] Excel分析エラー: {e}")
        results['excel_data_analysis'] = {'error': str(e)}

def detect_date_loop_patterns(results):
    """日付ループパターンの検出"""
    print(f"\n🔄 3. 日付ループパターンの検出")
    print("-" * 40)
    
    try:
        # DSSMSBacktesterのポートフォリオ履歴生成部分を分析
        backtester_path = "src/dssms/dssms_backtester.py"
        if Path(backtester_path).exists():
            with open(backtester_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # ポートフォリオ履歴関連の処理を検索
            portfolio_methods = re.findall(r'def.*portfolio.*\(.*\):', content, re.IGNORECASE)
            print(f"[CHART] ポートフォリオ関連メソッド: {len(portfolio_methods)}件")
            for method in portfolio_methods:
                print(f"  → {method}")
            
            # 日付生成・変換処理を検索
            date_processing_lines = []
            lines = content.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['pd.to_datetime', 'datetime', 'date_range', 'daterange']):
                    date_processing_lines.append(line.strip())
            
            print(f"📅 日付処理: {len(date_processing_lines)}件")
            for line in date_processing_lines[:5]:
                print(f"  → {line}")
            
            # ポートフォリオ値の生成ロジックを特定
            portfolio_value_lines = []
            for line in lines:
                if 'portfolio_value' in line.lower() and ('append' in line or '=' in line):
                    portfolio_value_lines.append(line.strip())
            
            print(f"[MONEY] ポートフォリオ値生成: {len(portfolio_value_lines)}件")
            for line in portfolio_value_lines[:5]:
                print(f"  → {line}")
            
            results['date_loop_detection'] = {
                'portfolio_methods': portfolio_methods,
                'date_processing_count': len(date_processing_lines),
                'portfolio_value_generation': portfolio_value_lines[:10]
            }
        else:
            print(f"[ERROR] DSSMSBacktesterファイルが見つかりません")
            results['date_loop_detection'] = {'error': 'backtester_not_found'}
            
    except Exception as e:
        print(f"[ERROR] ループパターン検出エラー: {e}")
        results['date_loop_detection'] = {'error': str(e)}

def main():
    """Task 2.1 メイン実行"""
    print("[ROCKET] Task 2.1: 日付処理ロジック検証 開始")
    print("=" * 80)
    
    # 日付処理ロジック分析実行
    analysis_results = analyze_date_processing_logic()
    
    # 結果サマリー
    print(f"\n[CHART] Task 2.1 実行結果サマリー")
    print("=" * 50)
    
    engine_count = len([k for k, v in analysis_results.get('engine_date_fixes', {}).items() if not v.get('error')])
    excel_sheets = len(analysis_results.get('excel_data_analysis', {}))
    
    print(f"[OK] 分析完了エンジン: {engine_count}件")
    print(f"[OK] Excel分析シート: {excel_sheets}件")
    print(f"[OK] ループパターン検出: {'成功' if not analysis_results.get('date_loop_detection', {}).get('error') else '失敗'}")
    
    # 重要な発見事項
    print(f"\n[TARGET] 重要な発見事項:")
    
    # expected_year問題の確認
    for engine, analysis in analysis_results.get('engine_date_fixes', {}).items():
        expected_year_usage = analysis.get('expected_year_usage', [])
        if expected_year_usage:
            print(f"  [WARNING] {engine}: expected_year=2023の固定設定発見")
    
    # Excel内のループパターン確認
    for sheet, data in analysis_results.get('excel_data_analysis', {}).items():
        date_patterns = data.get('date_patterns', [])
        if date_patterns:
            print(f"  📅 {sheet}: 年末年始ループパターン {len(date_patterns)}件検出")
    
    # 結果保存
    output_file = f"task_2_1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 結果保存: {output_file}")
    
    # 次のアクション
    print(f"\n[TARGET] 次のアクション:")
    print("1. Task 2.1結果をroadmap2.mdに記録")
    print("2. Task 2.2: ポートフォリオ価値データフロー追跡の実行")
    print("3. 日付修正ロジックの改善提案")
    
    return analysis_results

if __name__ == "__main__":
    main()
