#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 4.2: 計算ロジック実装状況の確認

勝率計算、平均利益/損失、プロフィットファクター等の
具体的な計算ロジックの実装状況を各エンジンで詳細確認し、
実装レベルを比較分析する
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import traceback
import re

def analyze_calculation_logic_implementation():
    """計算ロジック実装状況の詳細確認"""
    
    results = {
        "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "detailed_implementations": {},
        "calculation_formulas": {},
        "implementation_quality": {},
        "problems_identified": [],
        "recommendations": []
    }
    
    print("🔍 Task 4.2: 計算ロジック実装状況の確認を開始")
    print("=" * 60)
    
    engine_files = [
        "dssms_unified_output_engine.py",
        "dssms_unified_output_engine_fixed.py", 
        "dssms_unified_output_engine_fixed_v3.py",
        "dssms_unified_output_engine_fixed_v4.py"
    ]
    
    # 分析対象の計算ロジック
    target_calculations = {
        "win_rate": {
            "name": "勝率計算",
            "expected_formula": "profitable_trades / total_trades",
            "keywords": ["win_rate", "勝率", "profitable", "winning"]
        },
        "profit_factor": {
            "name": "プロフィットファクター",
            "expected_formula": "total_profit / abs(total_loss)",
            "keywords": ["profit_factor", "プロフィットファクター", "profit", "factor"]
        },
        "average_profit": {
            "name": "平均利益",
            "expected_formula": "profit_trades.mean()",
            "keywords": ["average_profit", "平均利益", "mean", "profit"]
        },
        "average_loss": {
            "name": "平均損失",
            "expected_formula": "loss_trades.mean()",
            "keywords": ["average_loss", "平均損失", "mean", "loss"]
        },
        "total_trades": {
            "name": "総取引数",
            "expected_formula": "len(trades)",
            "keywords": ["total_trades", "総取引", "count", "len"]
        },
        "profitable_trades": {
            "name": "利益取引数",
            "expected_formula": "len([t for t in trades if t > 0])",
            "keywords": ["profitable_trades", "利益取引", "positive", "win"]
        }
    }
    
    # 1. 各エンジンでの計算ロジック詳細分析
    print("\n🧮 Step 1: 各エンジンの計算ロジック詳細分析")
    
    for engine_file in engine_files:
        engine_path = Path(engine_file)
        if not engine_path.exists():
            results["detailed_implementations"][engine_file] = {
                "file_exists": False,
                "error": "ファイルが存在しない"
            }
            print(f"❌ {engine_file}: ファイルが見つかりません")
            continue
            
        with open(engine_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\n📁 {engine_file} の詳細分析:")
        
        engine_analysis = {
            "file_exists": True,
            "file_size": len(content),
            "calculations": {}
        }
        
        # 各計算ロジックの実装状況を詳細調査
        for calc_id, calc_info in target_calculations.items():
            calc_analysis = {
                "implemented": False,
                "implementation_quality": "未実装",
                "found_implementations": [],
                "code_snippets": [],
                "formula_matches": False
            }
            
            # キーワードベースで実装箇所を検索
            implementations = []
            for i, line in enumerate(content.split('\n'), 1):
                line_lower = line.lower().strip()
                
                # キーワードマッチング
                keyword_matches = [kw for kw in calc_info["keywords"] if kw.lower() in line_lower]
                if keyword_matches:
                    implementations.append({
                        "line": i,
                        "content": line.strip(),
                        "matched_keywords": keyword_matches
                    })
            
            if implementations:
                calc_analysis["implemented"] = True
                calc_analysis["found_implementations"] = implementations
                
                # 実装品質の評価
                quality_score = 0
                for impl in implementations:
                    # 完全な計算式があるかチェック
                    if '=' in impl["content"] and any(op in impl["content"] for op in ['+', '-', '*', '/', 'len', 'mean']):
                        quality_score += 2
                    # 変数定義があるかチェック
                    elif any(keyword in impl["content"].lower() for keyword in ['def ', 'return', 'calculate']):
                        quality_score += 1
                
                # 品質評価
                if quality_score >= 3:
                    calc_analysis["implementation_quality"] = "高品質"
                elif quality_score >= 1:
                    calc_analysis["implementation_quality"] = "部分実装"
                else:
                    calc_analysis["implementation_quality"] = "低品質"
                
                # 期待される公式との一致度チェック
                expected_elements = calc_info["expected_formula"].lower().split()
                for impl in implementations:
                    impl_content = impl["content"].lower()
                    if any(element in impl_content for element in expected_elements):
                        calc_analysis["formula_matches"] = True
                        break
            
            engine_analysis["calculations"][calc_id] = calc_analysis
            
            # 結果表示
            status = "✅" if calc_analysis["implemented"] else "❌"
            quality = calc_analysis["implementation_quality"]
            count = len(calc_analysis["found_implementations"])
            print(f"   {status} {calc_info['name']}: {quality} ({count}箇所)")
        
        results["detailed_implementations"][engine_file] = engine_analysis
    
    # 2. 計算式の具体的な実装内容分析
    print("\n📐 Step 2: 計算式の具体的な実装内容分析")
    
    formula_analysis = {}
    for engine_file, engine_data in results["detailed_implementations"].items():
        if not engine_data.get("file_exists", False):
            continue
            
        print(f"\n📁 {engine_file} の計算式分析:")
        
        engine_formulas = {}
        for calc_id, calc_data in engine_data.get("calculations", {}).items():
            if calc_data["implemented"]:
                # 実際の計算式を抽出
                actual_formulas = []
                for impl in calc_data["found_implementations"]:
                    content = impl["content"]
                    if '=' in content:
                        # 代入文から右辺を抽出
                        parts = content.split('=')
                        if len(parts) >= 2:
                            formula = parts[-1].strip()
                            actual_formulas.append(formula)
                
                engine_formulas[calc_id] = {
                    "target_name": target_calculations[calc_id]["name"],
                    "expected_formula": target_calculations[calc_id]["expected_formula"],
                    "actual_formulas": actual_formulas,
                    "formula_quality": "一致" if calc_data["formula_matches"] else "不一致"
                }
                
                print(f"   📊 {target_calculations[calc_id]['name']}:")
                print(f"      期待式: {target_calculations[calc_id]['expected_formula']}")
                if actual_formulas:
                    print(f"      実装式: {actual_formulas[0] if actual_formulas else 'なし'}")
                else:
                    print(f"      実装式: 検出されず")
        
        formula_analysis[engine_file] = engine_formulas
    
    results["calculation_formulas"] = formula_analysis
    
    # 3. 実装品質の総合評価
    print("\n⭐ Step 3: 実装品質の総合評価")
    
    quality_summary = {}
    for engine_file, engine_data in results["detailed_implementations"].items():
        if not engine_data.get("file_exists", False):
            continue
            
        total_calcs = len(target_calculations)
        implemented_calcs = sum(1 for calc in engine_data.get("calculations", {}).values() 
                               if calc["implemented"])
        high_quality_calcs = sum(1 for calc in engine_data.get("calculations", {}).values() 
                                if calc["implementation_quality"] == "高品質")
        formula_matched_calcs = sum(1 for calc in engine_data.get("calculations", {}).values() 
                                   if calc["formula_matches"])
        
        # 総合スコア計算
        implementation_score = (implemented_calcs / total_calcs) * 40
        quality_score = (high_quality_calcs / total_calcs) * 30
        formula_score = (formula_matched_calcs / total_calcs) * 30
        total_score = implementation_score + quality_score + formula_score
        
        quality_summary[engine_file] = {
            "total_calculations": total_calcs,
            "implemented_count": implemented_calcs,
            "high_quality_count": high_quality_calcs,
            "formula_matched_count": formula_matched_calcs,
            "implementation_percentage": round((implemented_calcs / total_calcs) * 100, 1),
            "quality_percentage": round((high_quality_calcs / total_calcs) * 100, 1),
            "formula_match_percentage": round((formula_matched_calcs / total_calcs) * 100, 1),
            "total_score": round(total_score, 1)
        }
        
        print(f"📁 {engine_file}:")
        print(f"   実装率: {quality_summary[engine_file]['implementation_percentage']}% ({implemented_calcs}/{total_calcs})")
        print(f"   高品質率: {quality_summary[engine_file]['quality_percentage']}% ({high_quality_calcs}/{total_calcs})")
        print(f"   公式一致率: {quality_summary[engine_file]['formula_match_percentage']}% ({formula_matched_calcs}/{total_calcs})")
        print(f"   総合スコア: {quality_summary[engine_file]['total_score']}/100")
    
    results["implementation_quality"] = quality_summary
    
    # 4. 問題の特定
    print("\n🎯 Step 4: 実装問題の特定")
    
    problems = []
    
    # 実装率が50%未満のエンジン
    low_implementation_engines = [
        engine for engine, data in quality_summary.items()
        if data["implementation_percentage"] < 50
    ]
    if low_implementation_engines:
        problems.append(f"実装率50%未満のエンジン: {', '.join(low_implementation_engines)}")
    
    # 高品質実装が25%未満のエンジン
    low_quality_engines = [
        engine for engine, data in quality_summary.items()
        if data["quality_percentage"] < 25
    ]
    if low_quality_engines:
        problems.append(f"高品質実装25%未満のエンジン: {', '.join(low_quality_engines)}")
    
    # 公式一致率が30%未満のエンジン
    low_formula_engines = [
        engine for engine, data in quality_summary.items()
        if data["formula_match_percentage"] < 30
    ]
    if low_formula_engines:
        problems.append(f"公式一致率30%未満のエンジン: {', '.join(low_formula_engines)}")
    
    # v3エンジンの特別チェック
    v3_data = quality_summary.get("dssms_unified_output_engine_fixed_v3.py")
    if v3_data and v3_data["total_score"] == 0:
        problems.append("v3エンジンで計算ロジックが完全未実装（スコア0点）")
    
    results["problems_identified"] = problems
    
    # 推奨事項
    recommendations = [
        "v3エンジンの計算ロジック実装（現在完全未実装）",
        "低品質実装エンジンの計算式修正",
        "期待公式との一致率向上",
        "統一的な計算ロジック実装ガイドライン策定",
        "計算精度検証テストの追加"
    ]
    
    results["recommendations"] = recommendations
    
    print("\n💡 推奨事項:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n❌ 特定された問題:")
    for problem in problems:
        print(f"   {problem}")
    
    # 5. 最優秀エンジンと最低エンジンの特定
    print("\n🏆 Step 5: エンジンランキング")
    
    if quality_summary:
        sorted_engines = sorted(quality_summary.items(), 
                               key=lambda x: x[1]["total_score"], reverse=True)
        
        print("📊 総合スコアランキング:")
        for i, (engine, data) in enumerate(sorted_engines, 1):
            rank_icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
            print(f"{rank_icon} {i}位: {engine.replace('dssms_unified_output_engine', 'engine').replace('.py', '')} - {data['total_score']}点")
    
    # 結果をJSONファイルに保存
    output_file = f"task_4_2_results_{results['analysis_timestamp']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 分析結果を保存: {output_file}")
    print("=" * 60)
    print("🔍 Task 4.2: 計算ロジック実装状況の確認完了")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_calculation_logic_implementation()
        
        # 重要な発見事項を表示
        print("\n🎯 重要な発見事項:")
        for problem in results["problems_identified"]:
            print(f"❌ {problem}")
            
    except Exception as e:
        print(f"❌ Task 4.2実行エラー: {e}")
        traceback.print_exc()