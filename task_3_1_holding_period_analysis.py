#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 3.1: 保有期間計算ロジックの比較分析

各統一エンジンでの保有期間計算方法を比較し、
24時間固定問題の根本原因を特定する
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import traceback
import re

def analyze_holding_period_calculation():
    """保有期間計算ロジックの比較分析"""
    
    results = {
        "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "findings": {},
        "calculation_methods": {},
        "problems_identified": [],
        "recommendations": []
    }
    
    print("🔍 Task 3.1: 保有期間計算ロジックの比較分析を開始")
    print("=" * 60)
    
    # 1. 各エンジンでの保有期間計算方法比較
    print("\n📊 Step 1: 各エンジンの保有期間計算ロジック分析")
    
    engine_files = [
        "dssms_unified_output_engine.py",
        "dssms_unified_output_engine_fixed.py", 
        "dssms_unified_output_engine_fixed_v3.py",
        "dssms_unified_output_engine_fixed_v4.py"
    ]
    
    calculation_analysis = {}
    
    for engine_file in engine_files:
        engine_path = Path(engine_file)
        if engine_path.exists():
            with open(engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"\n📁 {engine_file} の分析:")
            
            # 保有期間関連のキーワードを検索
            holding_keywords = [
                "holding_period", "保有期間", "actual_holding", 
                "24.0", "24時間", "timedelta", "timestamp"
            ]
            
            keyword_occurrences = {}
            for keyword in holding_keywords:
                occurrences = []
                for i, line in enumerate(content.split('\n'), 1):
                    if keyword.lower() in line.lower():
                        occurrences.append(f"Line {i}: {line.strip()}")
                keyword_occurrences[keyword] = occurrences[:5]  # 最初の5件
            
            # 特に重要な計算ロジックを抽出
            calculation_methods = []
            
            # actual_holding_hoursの計算ロジックを検索
            if "actual_holding_hours" in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "actual_holding_hours" in line:
                        # 前後5行のコンテキストを取得
                        start = max(0, i-3)
                        end = min(len(lines), i+4)
                        context = lines[start:end]
                        calculation_methods.append({
                            "line_number": i+1,
                            "context": context,
                            "calculation_line": line.strip()
                        })
            
            # 24.0や24時間の固定値を検索
            fixed_24_hour_usage = []
            for i, line in enumerate(content.split('\n'), 1):
                if re.search(r'24\.0|24時間|24\s*hour', line, re.IGNORECASE):
                    fixed_24_hour_usage.append(f"Line {i}: {line.strip()}")
            
            calculation_analysis[engine_file] = {
                "file_exists": True,
                "file_size": len(content),
                "keyword_occurrences": keyword_occurrences,
                "calculation_methods": calculation_methods,
                "fixed_24_hour_usage": fixed_24_hour_usage[:5],
                "has_actual_holding_calculation": "actual_holding_hours" in content
            }
            
            print(f"   ✅ ファイルサイズ: {len(content)}文字")
            print(f"   📍 actual_holding_hours計算: {'✅' if 'actual_holding_hours' in content else '❌'}")
            print(f"   🔢 24時間固定値使用: {len(fixed_24_hour_usage)}箇所")
            
            if calculation_methods:
                print(f"   🧮 保有期間計算ロジック: {len(calculation_methods)}箇所発見")
                for method in calculation_methods[:2]:  # 最初の2件を表示
                    print(f"      Line {method['line_number']}: {method['calculation_line']}")
        else:
            calculation_analysis[engine_file] = {
                "file_exists": False,
                "error": "ファイルが見つからない"
            }
            print(f"❌ {engine_file}: ファイルが見つかりません")
    
    results["calculation_methods"] = calculation_analysis
    
    # 2. SymbolSwitchオブジェクトのtimestamp取得分析
    print("\n🔍 Step 2: SymbolSwitchオブジェクトの調査")
    
    try:
        # SymbolSwitchクラスの定義を検索
        symbol_switch_files = []
        for file_path in Path(".").rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if "class SymbolSwitch" in content or "SymbolSwitch" in content:
                        symbol_switch_files.append(str(file_path))
                except Exception:
                    continue
        
        symbol_switch_analysis = {
            "found_files": symbol_switch_files[:5],
            "timestamp_usage": {}
        }
        
        # 主要なSymbolSwitchファイルを分析
        for file_path in symbol_switch_files[:3]:  # 最初の3ファイル
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                timestamp_lines = []
                for i, line in enumerate(content.split('\n'), 1):
                    if 'timestamp' in line.lower():
                        timestamp_lines.append(f"Line {i}: {line.strip()}")
                
                symbol_switch_analysis["timestamp_usage"][file_path] = timestamp_lines[:5]
                
            except Exception as e:
                symbol_switch_analysis["timestamp_usage"][file_path] = f"読み込みエラー: {e}"
        
        results["findings"]["symbol_switch_analysis"] = symbol_switch_analysis
        print(f"✅ SymbolSwitch関連ファイル: {len(symbol_switch_files)}件発見")
        
    except Exception as e:
        print(f"❌ SymbolSwitch分析エラー: {e}")
        results["problems_identified"].append(f"SymbolSwitch分析エラー: {e}")
    
    # 3. 日付差分計算の実装比較
    print("\n📅 Step 3: 日付差分計算の実装比較")
    
    date_calculation_patterns = {
        "timedelta": [],
        "pd.Timestamp": [],
        "datetime": [],
        "date difference": []
    }
    
    for engine_file in engine_files:
        engine_path = Path(engine_file)
        if engine_path.exists():
            with open(engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 日付計算パターンを検索
            for i, line in enumerate(content.split('\n'), 1):
                line_lower = line.lower()
                
                if 'timedelta' in line_lower:
                    date_calculation_patterns["timedelta"].append(f"{engine_file} Line {i}: {line.strip()}")
                
                if 'pd.timestamp' in line_lower or 'pandas.timestamp' in line_lower:
                    date_calculation_patterns["pd.Timestamp"].append(f"{engine_file} Line {i}: {line.strip()}")
                
                if 'datetime' in line_lower and ('delta' in line_lower or '-' in line):
                    date_calculation_patterns["datetime"].append(f"{engine_file} Line {i}: {line.strip()}")
                
                if re.search(r'(\w+)\s*-\s*(\w+)', line) and ('date' in line_lower or 'time' in line_lower):
                    date_calculation_patterns["date difference"].append(f"{engine_file} Line {i}: {line.strip()}")
    
    # 結果を制限
    for pattern_type in date_calculation_patterns:
        date_calculation_patterns[pattern_type] = date_calculation_patterns[pattern_type][:5]
    
    results["findings"]["date_calculation_patterns"] = date_calculation_patterns
    
    print("📅 日付計算パターン発見:")
    for pattern_type, occurrences in date_calculation_patterns.items():
        print(f"   {pattern_type}: {len(occurrences)}件")
    
    # 4. 最後のスイッチでのデフォルト値処理
    print("\n🔚 Step 4: 最後のスイッチでのデフォルト値処理")
    
    default_value_analysis = {}
    
    for engine_file in engine_files:
        engine_path = Path(engine_file)
        if engine_path.exists():
            with open(engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            default_patterns = []
            
            # デフォルト値関連のパターンを検索
            default_keywords = ["default", "デフォルト", "if not", "is None", "get(", "last"]
            
            for i, line in enumerate(content.split('\n'), 1):
                line_lower = line.lower()
                for keyword in default_keywords:
                    if keyword in line_lower and ("24" in line or "hour" in line_lower or "period" in line_lower):
                        default_patterns.append(f"Line {i}: {line.strip()}")
            
            default_value_analysis[engine_file] = default_patterns[:3]
            
            if default_patterns:
                print(f"📁 {engine_file}: {len(default_patterns)}件のデフォルト値処理")
    
    results["findings"]["default_value_analysis"] = default_value_analysis
    
    # 5. 問題の特定と推奨事項
    print("\n🎯 Step 5: 問題の特定と推奨事項")
    
    identified_problems = []
    
    # 24時間固定値の使用を確認
    total_fixed_24_usage = sum(len(data.get("fixed_24_hour_usage", [])) 
                              for data in calculation_analysis.values() 
                              if isinstance(data, dict) and data.get("file_exists"))
    
    if total_fixed_24_usage > 3:
        identified_problems.append(f"24時間固定値の過剰使用: {total_fixed_24_usage}箇所で発見")
    
    # actual_holding_hours計算の有無
    engines_with_calculation = [engine for engine, data in calculation_analysis.items() 
                               if isinstance(data, dict) and data.get("has_actual_holding_calculation")]
    
    if len(engines_with_calculation) < len([e for e in engine_files if Path(e).exists()]):
        identified_problems.append("一部エンジンでactual_holding_hours計算が未実装")
    
    # v3エンジンの空ファイル問題
    v3_data = calculation_analysis.get("dssms_unified_output_engine_fixed_v3.py", {})
    if isinstance(v3_data, dict) and v3_data.get("file_size", 0) < 100:
        identified_problems.append("v3エンジンが空ファイル状態で保有期間計算不可能")
    
    results["problems_identified"] = identified_problems
    
    # 推奨事項
    recommendations = [
        "v3エンジンの実装完了（現在空ファイル）",
        "24時間固定値を動的計算に変更",
        "actual_holding_hours計算ロジックの統一",
        "SymbolSwitchのtimestamp取得方法統一",
        "最後のスイッチでのデフォルト値処理改善"
    ]
    
    results["recommendations"] = recommendations
    
    print("💡 推奨事項:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n❌ 特定された問題:")
    for problem in identified_problems:
        print(f"   {problem}")
    
    # 結果をJSONファイルに保存
    output_file = f"task_3_1_results_{results['analysis_timestamp']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 分析結果を保存: {output_file}")
    print("=" * 60)
    print("🔍 Task 3.1: 保有期間計算ロジックの比較分析完了")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_holding_period_calculation()
        
        # 重要な発見事項を表示
        print("\n🎯 重要な発見事項:")
        for problem in results["problems_identified"]:
            print(f"❌ {problem}")
            
    except Exception as e:
        print(f"❌ Task 3.1実行エラー: {e}")
        traceback.print_exc()
