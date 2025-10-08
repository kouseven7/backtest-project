#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 5.3: エンジン共存分析
DSSMS内の複数エンジン（v1,v2,v3,v4）の共存状況と相互影響を調査
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import ast

def analyze_engine_coexistence():
    """複数エンジンの共存分析実行"""
    print("[SEARCH] Task 5.3: DSSMS エンジン共存分析開始")
    
    analysis_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "task": "5.3",
        "purpose": "エンジン共存状況と相互影響の分析",
        "engines_analyzed": [],
        "coexistence_issues": [],
        "performance_conflicts": [],
        "integration_problems": [],
        "resource_competition": [],
        "execution_runtime_conflicts": [],
        "version_inconsistencies": [],
        "summary_findings": {}
    }
    
    # 1. エンジンファイル検出
    print("\n📁 Step 1: エンジンファイル検出")
    engine_files = detect_engine_files()
    analysis_results["engines_analyzed"] = engine_files
    
    # 2. 共存状況分析
    print("\n🔄 Step 2: エンジン共存状況分析")
    coexistence_analysis = analyze_coexistence_patterns(engine_files)
    analysis_results["coexistence_issues"] = coexistence_analysis
    
    # 3. 実行時競合分析
    print("\n⚡ Step 3: 実行時競合分析")
    runtime_conflicts = analyze_runtime_conflicts(engine_files)
    analysis_results["execution_runtime_conflicts"] = runtime_conflicts
    
    # 4. リソース競合分析
    print("\n💾 Step 4: リソース競合分析")
    resource_analysis = analyze_resource_competition(engine_files)
    analysis_results["resource_competition"] = resource_analysis
    
    # 5. バージョン一貫性分析
    print("\n🔢 Step 5: バージョン一貫性分析")
    version_analysis = analyze_version_consistency(engine_files)
    analysis_results["version_inconsistencies"] = version_analysis
    
    # 6. 統合問題分析
    print("\n🔗 Step 6: 統合問題分析")
    integration_analysis = analyze_integration_problems(engine_files)
    analysis_results["integration_problems"] = integration_analysis
    
    # 7. パフォーマンス競合分析
    print("\n[CHART] Step 7: パフォーマンス競合分析")
    performance_analysis = analyze_performance_conflicts(engine_files)
    analysis_results["performance_conflicts"] = performance_analysis
    
    # 8. 総合分析と結論
    print("\n[LIST] Step 8: 総合分析")
    summary = generate_summary_analysis(analysis_results)
    analysis_results["summary_findings"] = summary
    
    # 結果保存
    output_file = "task_5_3_engine_coexistence_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Task 5.3 完了: {output_file}")
    print(f"[SEARCH] 検出エンジン数: {len(engine_files)}")
    print(f"[WARNING] 共存問題: {len(analysis_results['coexistence_issues'])}")
    print(f"⚡ 実行時競合: {len(analysis_results['execution_runtime_conflicts'])}")
    print(f"💾 リソース競合: {len(analysis_results['resource_competition'])}")
    
    return analysis_results

def detect_engine_files() -> List[Dict[str, Any]]:
    """エンジンファイルの検出と基本情報収集"""
    engine_patterns = [
        r".*backtester.*\.py$",
        r".*engine.*\.py$", 
        r".*excel.*exporter.*\.py$",
        r".*output.*engine.*\.py$",
        r".*unified.*engine.*\.py$"
    ]
    
    engines = []
    
    # srcディレクトリとルートディレクトリをスキャン
    search_dirs = ["src", ".", "output"]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        
                        # エンジンパターンマッチング
                        for pattern in engine_patterns:
                            if re.match(pattern, file, re.IGNORECASE):
                                engine_info = analyze_engine_file(file_path)
                                if engine_info:
                                    engines.append(engine_info)
                                break
    
    return engines

def analyze_engine_file(file_path: str) -> Optional[Dict[str, Any]]:
    """個別エンジンファイルの分析"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基本情報
        file_size = os.path.getsize(file_path)
        line_count = len(content.splitlines())
        
        # エンジンタイプ判定
        engine_type = determine_engine_type(file_path, content)
        
        # クラス・メソッド検出
        classes = extract_classes(content)
        methods = extract_methods(content)
        
        # 依存関係分析
        imports = extract_imports(content)
        
        # パフォーマンス関連コード検出
        performance_code = extract_performance_indicators(content)
        
        return {
            "file_path": file_path,
            "engine_type": engine_type,
            "file_size_bytes": file_size,
            "line_count": line_count,
            "classes": classes,
            "methods": methods,
            "imports": imports,
            "performance_indicators": performance_code,
            "is_active": check_if_active(content),
            "version_indicators": extract_version_indicators(content)
        }
        
    except Exception as e:
        print(f"[WARNING] エンジンファイル解析エラー {file_path}: {e}")
        return None

def determine_engine_type(file_path: str, content: str) -> str:
    """エンジンタイプの判定"""
    path_lower = file_path.lower()
    content_lower = content.lower()
    
    if "v4" in path_lower or "v4" in content_lower:
        return "v4"
    elif "v3" in path_lower or "v3" in content_lower:
        return "v3"
    elif "v2" in path_lower or "v2" in content_lower:
        return "v2"
    elif "v1" in path_lower or "v1" in content_lower:
        return "v1"
    elif "unified" in path_lower:
        return "unified"
    elif "excel" in path_lower and "exporter" in path_lower:
        return "excel_exporter"
    elif "backtester" in path_lower:
        return "backtester"
    elif "engine" in path_lower:
        return "engine"
    else:
        return "unknown"

def extract_classes(content: str) -> List[str]:
    """クラス名の抽出"""
    classes = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
    except:
        # ASTパースが失敗した場合は正規表現でフォールバック
        class_pattern = r'^class\s+(\w+)'
        matches = re.findall(class_pattern, content, re.MULTILINE)
        classes = matches
    
    return classes

def extract_methods(content: str) -> List[str]:
    """メソッド名の抽出"""
    methods = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)
    except:
        # ASTパースが失敗した場合は正規表現でフォールバック
        method_pattern = r'def\s+(\w+)'
        matches = re.findall(method_pattern, content)
        methods = matches
    
    return methods

def extract_imports(content: str) -> List[str]:
    """import文の抽出"""
    imports = []
    import_patterns = [
        r'^import\s+([\w\.]+)',
        r'^from\s+([\w\.]+)\s+import'
    ]
    
    for pattern in import_patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        imports.extend(matches)
    
    return list(set(imports))

def extract_performance_indicators(content: str) -> Dict[str, List[str]]:
    """パフォーマンス関連コードの検出"""
    indicators = {
        "excel_operations": [],
        "data_processing": [],
        "memory_operations": [],
        "file_operations": [],
        "calculation_operations": []
    }
    
    # Excel操作
    excel_patterns = [
        r'\.to_excel\(',
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: r'pd\.ExcelWriter',
        r'xlsxwriter',
        r'openpyxl',
        r'\.save\(',
        r'workbook\.'
    ]
    
    for pattern in excel_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            indicators["excel_operations"].extend(matches)
    
    # データ処理
    data_patterns = [
        r'pd\.DataFrame',
        r'\.groupby\(',
        r'\.apply\(',
        r'\.merge\(',
        r'\.concat\('
    ]
    
    for pattern in data_patterns:
        matches = re.findall(pattern, content)
        if matches:
            indicators["data_processing"].extend(matches)
    
    # メモリ操作
    memory_patterns = [
        r'\.copy\(\)',
        r'\.deepcopy\(',
        r'del\s+\w+',
        r'gc\.collect\('
    ]
    
    for pattern in memory_patterns:
        matches = re.findall(pattern, content)
        if matches:
            indicators["memory_operations"].extend(matches)
    
    return indicators

def check_if_active(content: str) -> bool:
    """アクティブなエンジンかどうかの判定"""
    # 空ファイルまたは非常に小さいファイルは非アクティブ
    if len(content.strip()) < 100:
        return False
    
    # 実装されているかの簡易判定
    active_indicators = [
        "def __init__",
        "def run",
        "def execute",
        "def process",
        "def create_"
    ]
    
    for indicator in active_indicators:
        if indicator in content:
            return True
    
    return False

def extract_version_indicators(content: str) -> List[str]:
    """バージョン指示子の抽出"""
    version_patterns = [
        r'version\s*=\s*["\']([^"\']+)["\']',
        r'VERSION\s*=\s*["\']([^"\']+)["\']',
        r'__version__\s*=\s*["\']([^"\']+)["\']',
        r'# Version:\s*([^\n]+)',
        r'# v(\d+(?:\.\d+)*)'
    ]
    
    versions = []
    for pattern in version_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        versions.extend(matches)
    
    return versions

def analyze_coexistence_patterns(engines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """エンジン共存パターンの分析"""
    issues = []
    
    # エンジンタイプ別グループ化
    engine_groups = {}
    for engine in engines:
        engine_type = engine["engine_type"]
        if engine_type not in engine_groups:
            engine_groups[engine_type] = []
        engine_groups[engine_type].append(engine)
    
    # 1. 同一タイプの重複検出
    for engine_type, group in engine_groups.items():
        if len(group) > 1:
            issues.append({
                "issue_type": "duplicate_engines",
                "engine_type": engine_type,
                "count": len(group),
                "files": [engine["file_path"] for engine in group],
                "severity": "high" if len(group) > 2 else "medium",
                "description": f"{engine_type}エンジンが{len(group)}個重複存在"
            })
    
    # 2. アクティブ/非アクティブ混在
    active_engines = [e for e in engines if e["is_active"]]
    inactive_engines = [e for e in engines if not e["is_active"]]
    
    if len(inactive_engines) > 0:
        issues.append({
            "issue_type": "inactive_engines",
            "count": len(inactive_engines),
            "files": [engine["file_path"] for engine in inactive_engines],
            "severity": "medium",
            "description": f"{len(inactive_engines)}個の非アクティブエンジンが存在"
        })
    
    # 3. サイズ格差問題
    if active_engines:
        sizes = [engine["file_size_bytes"] for engine in active_engines]
        max_size = max(sizes)
        min_size = min(sizes)
        
        if max_size > min_size * 10:  # 10倍以上の差
            issues.append({
                "issue_type": "size_disparity",
                "max_size": max_size,
                "min_size": min_size,
                "ratio": max_size / min_size,
                "severity": "high",
                "description": f"エンジンサイズに{max_size/min_size:.1f}倍の格差"
            })
    
    return issues

def analyze_runtime_conflicts(engines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """実行時競合の分析"""
    conflicts = []
    
    # 1. 同時実行可能性の検証
    active_engines = [e for e in engines if e["is_active"]]
    
    if len(active_engines) > 1:
        conflicts.append({
            "conflict_type": "concurrent_execution",
            "engine_count": len(active_engines),
            "engines": [engine["file_path"] for engine in active_engines],
            "severity": "high",
            "description": f"{len(active_engines)}個のアクティブエンジンが同時実行可能"
        })
    
    # 2. 出力ファイル競合
    excel_engines = []
    for engine in engines:
        if any("excel" in indicator for indicators in engine["performance_indicators"].values() 
               for indicator in indicators):
            excel_engines.append(engine)
    
    if len(excel_engines) > 1:
        conflicts.append({
            "conflict_type": "output_file_conflict",
            "engine_count": len(excel_engines),
            "engines": [engine["file_path"] for engine in excel_engines],
            "severity": "medium",
            "description": f"{len(excel_engines)}個のエンジンがExcel出力を実行"
        })
    
    # 3. メモリ使用競合
    memory_intensive_engines = []
    for engine in engines:
        if len(engine["performance_indicators"]["data_processing"]) > 5:
            memory_intensive_engines.append(engine)
    
    if len(memory_intensive_engines) > 1:
        conflicts.append({
            "conflict_type": "memory_competition",
            "engine_count": len(memory_intensive_engines),
            "engines": [engine["file_path"] for engine in memory_intensive_engines],
            "severity": "medium",
            "description": f"{len(memory_intensive_engines)}個のメモリ集約型エンジン"
        })
    
    return conflicts

def analyze_resource_competition(engines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """リソース競合の分析"""
    competitions = []
    
    # 1. ファイル操作競合
    file_operation_engines = []
    for engine in engines:
        file_ops = engine["performance_indicators"]["file_operations"]
        if len(file_ops) > 0:
            file_operation_engines.append(engine)
    
    if len(file_operation_engines) > 1:
        competitions.append({
            "resource_type": "file_operations",
            "competing_engines": len(file_operation_engines),
            "engines": [engine["file_path"] for engine in file_operation_engines],
            "severity": "medium",
            "description": f"{len(file_operation_engines)}個のエンジンがファイル操作を実行"
        })
    
    # 2. 計算リソース競合
    calculation_engines = []
    for engine in engines:
        calc_ops = engine["performance_indicators"]["calculation_operations"]
        if len(calc_ops) > 3:
            calculation_engines.append(engine)
    
    if len(calculation_engines) > 1:
        competitions.append({
            "resource_type": "calculation_resources",
            "competing_engines": len(calculation_engines),
            "engines": [engine["file_path"] for engine in calculation_engines],
            "severity": "low",
            "description": f"{len(calculation_engines)}個のエンジンが集約計算を実行"
        })
    
    return competitions

def analyze_version_consistency(engines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """バージョン一貫性の分析"""
    inconsistencies = []
    
    # バージョン情報収集
    version_info = {}
    for engine in engines:
        engine_type = engine["engine_type"]
        versions = engine["version_indicators"]
        
        if versions:
            if engine_type not in version_info:
                version_info[engine_type] = []
            version_info[engine_type].extend(versions)
    
    # バージョン不整合検出
    for engine_type, versions in version_info.items():
        unique_versions = list(set(versions))
        if len(unique_versions) > 1:
            inconsistencies.append({
                "inconsistency_type": "version_mismatch",
                "engine_type": engine_type,
                "versions": unique_versions,
                "severity": "medium",
                "description": f"{engine_type}で複数バージョン検出: {unique_versions}"
            })
    
    return inconsistencies

def analyze_integration_problems(engines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """統合問題の分析"""
    problems = []
    
    # 1. インポート依存関係の分析
    import_graph = {}
    for engine in engines:
        engine_path = engine["file_path"]
        imports = engine["imports"]
        import_graph[engine_path] = imports
    
    # 循環依存の検出
    circular_deps = detect_circular_dependencies(import_graph)
    if circular_deps:
        problems.append({
            "problem_type": "circular_dependencies",
            "dependencies": circular_deps,
            "severity": "high",
            "description": f"{len(circular_deps)}個の循環依存を検出"
        })
    
    # 2. 共通インターフェースの欠如
    active_engines = [e for e in engines if e["is_active"]]
    if len(active_engines) > 1:
        common_methods = find_common_methods(active_engines)
        interface_coverage = len(common_methods) / max(len(engine["methods"]) for engine in active_engines)
        
        if interface_coverage < 0.3:  # 30%未満の共通性
            problems.append({
                "problem_type": "interface_inconsistency",
                "interface_coverage": interface_coverage,
                "common_methods": common_methods,
                "severity": "high",
                "description": f"共通インターフェース率{interface_coverage*100:.1f}%と低い"
            })
    
    return problems

def detect_circular_dependencies(import_graph: Dict[str, List[str]]) -> List[List[str]]:
    """循環依存の検出"""
    # 簡易実装：完全なグラフ解析は複雑なため基本的なケースのみ検出
    circular_deps = []
    
    for file_a, imports_a in import_graph.items():
        for import_a in imports_a:
            for file_b, imports_b in import_graph.items():
                if file_a != file_b and import_a in file_b:
                    for import_b in imports_b:
                        if import_b in file_a:
                            circular_deps.append([file_a, file_b])
    
    return circular_deps

def find_common_methods(engines: List[Dict[str, Any]]) -> List[str]:
    """共通メソッドの検出"""
    if not engines:
        return []
    
    # 最初のエンジンのメソッドから開始
    common_methods = set(engines[0]["methods"])
    
    # 他のエンジンとの共通部分を取る
    for engine in engines[1:]:
        common_methods &= set(engine["methods"])
    
    return list(common_methods)

def analyze_performance_conflicts(engines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """パフォーマンス競合の分析"""
    conflicts = []
    
    # 1. Excel出力競合
    excel_engines = []
    for engine in engines:
        excel_ops = engine["performance_indicators"]["excel_operations"]
        if len(excel_ops) > 0:
            excel_engines.append({
                "engine": engine,
                "excel_operations": len(excel_ops)
            })
    
    if len(excel_engines) > 1:
        conflicts.append({
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: "conflict_type": "excel_output_competition",
            "competing_engines": len(excel_engines),
            "engines": [e["engine"]["file_path"] for e in excel_engines],
            "total_operations": sum(e["excel_operations"] for e in excel_engines),
            "severity": "high",
            "description": f"{len(excel_engines)}個のエンジンがExcel出力で競合"
        })
    
    # 2. データ処理負荷競合
    data_engines = []
    for engine in engines:
        data_ops = engine["performance_indicators"]["data_processing"]
        if len(data_ops) > 3:
            data_engines.append({
                "engine": engine,
                "data_operations": len(data_ops)
            })
    
    if len(data_engines) > 1:
        conflicts.append({
            "conflict_type": "data_processing_load",
            "competing_engines": len(data_engines),
            "engines": [e["engine"]["file_path"] for e in data_engines],
            "total_operations": sum(e["data_operations"] for e in data_engines),
            "severity": "medium",
            "description": f"{len(data_engines)}個のエンジンがデータ処理で競合"
        })
    
    return conflicts

def generate_summary_analysis(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """総合分析と結論の生成"""
    engines = analysis_results["engines_analyzed"]
    coexistence_issues = analysis_results["coexistence_issues"]
    runtime_conflicts = analysis_results["execution_runtime_conflicts"]
    resource_competition = analysis_results["resource_competition"]
    integration_problems = analysis_results["integration_problems"]
    performance_conflicts = analysis_results["performance_conflicts"]
    
    # 重要度計算
    total_issues = len(coexistence_issues) + len(runtime_conflicts) + len(resource_competition) + \
                  len(integration_problems) + len(performance_conflicts)
    
    # アクティブエンジン統計
    active_engines = [e for e in engines if e["is_active"]]
    inactive_engines = [e for e in engines if not e["is_active"]]
    
    # エンジンタイプ分布
    engine_types = {}
    for engine in engines:
        engine_type = engine["engine_type"]
        engine_types[engine_type] = engine_types.get(engine_type, 0) + 1
    
    # 深刻度評価
    high_severity_count = sum(1 for issue_list in [coexistence_issues, runtime_conflicts, 
                                                  resource_competition, integration_problems, 
                                                  performance_conflicts]
                             for issue in issue_list if issue.get("severity") == "high")
    
    severity_level = "critical" if high_severity_count >= 3 else \
                    "high" if high_severity_count >= 2 else \
                    "medium" if total_issues >= 5 else "low"
    
    # 改善推奨事項
    recommendations = generate_improvement_recommendations(analysis_results)
    
    return {
        "total_engines_detected": len(engines),
        "active_engines": len(active_engines),
        "inactive_engines": len(inactive_engines),
        "engine_type_distribution": engine_types,
        "total_issues_detected": total_issues,
        "high_severity_issues": high_severity_count,
        "overall_severity_level": severity_level,
        "coexistence_health_score": calculate_coexistence_health_score(analysis_results),
        "primary_concerns": identify_primary_concerns(analysis_results),
        "improvement_recommendations": recommendations,
        "impact_on_switch_frequency": analyze_switch_frequency_impact(analysis_results)
    }

def calculate_coexistence_health_score(analysis_results: Dict[str, Any]) -> float:
    """共存健全度スコアの計算"""
    engines = analysis_results["engines_analyzed"]
    issues = analysis_results["coexistence_issues"]
    conflicts = analysis_results["execution_runtime_conflicts"]
    
    # 基本スコア
    base_score = 100.0
    
    # ペナルティ計算
    penalty = 0
    
    # 重複エンジンペナルティ
    duplicate_issues = [i for i in issues if i["issue_type"] == "duplicate_engines"]
    penalty += len(duplicate_issues) * 15
    
    # 非アクティブエンジンペナルティ
    inactive_issues = [i for i in issues if i["issue_type"] == "inactive_engines"]
    penalty += len(inactive_issues) * 10
    
    # 実行時競合ペナルティ
    penalty += len(conflicts) * 20
    
    # サイズ格差ペナルティ
    size_issues = [i for i in issues if i["issue_type"] == "size_disparity"]
    penalty += len(size_issues) * 25
    
    final_score = max(0, base_score - penalty)
    return final_score

def identify_primary_concerns(analysis_results: Dict[str, Any]) -> List[str]:
    """主要な懸念事項の特定"""
    concerns = []
    
    # 高重要度の問題を特定
    all_issues = (analysis_results["coexistence_issues"] + 
                 analysis_results["execution_runtime_conflicts"] +
                 analysis_results["resource_competition"] +
                 analysis_results["integration_problems"] +
                 analysis_results["performance_conflicts"])
    
    high_severity_issues = [issue for issue in all_issues if issue.get("severity") == "high"]
    
    for issue in high_severity_issues:
        if issue.get("issue_type") == "duplicate_engines":
            concerns.append(f"重複エンジン: {issue['engine_type']}が{issue['count']}個存在")
        elif issue.get("conflict_type") == "concurrent_execution":
            concerns.append(f"同時実行競合: {issue['engine_count']}個のアクティブエンジン")
        elif issue.get("problem_type") == "circular_dependencies":
            concerns.append("循環依存: エンジン間で循環インポート")
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: elif issue.get("conflict_type") == "excel_output_competition":
            concerns.append("Excel出力競合: 複数エンジンが同時出力")
    
    return concerns

def generate_improvement_recommendations(analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """改善推奨事項の生成"""
    recommendations = []
    
    engines = analysis_results["engines_analyzed"]
    active_engines = [e for e in engines if e["is_active"]]
    inactive_engines = [e for e in engines if not e["is_active"]]
    
    # 1. 非アクティブエンジンの整理
    if len(inactive_engines) > 0:
        recommendations.append({
            "priority": "high",
            "category": "cleanup",
            "title": "非アクティブエンジンの整理",
            "description": f"{len(inactive_engines)}個の非アクティブエンジンを削除またはアーカイブ",
            "files": [engine["file_path"] for engine in inactive_engines],
            "effort": "low",
            "impact": "medium"
        })
    
    # 2. 重複エンジンの統合
    duplicate_issues = [i for i in analysis_results["coexistence_issues"] 
                       if i["issue_type"] == "duplicate_engines"]
    if duplicate_issues:
        recommendations.append({
            "priority": "high",
            "category": "consolidation",
            "title": "重複エンジンの統合",
            "description": "同一タイプの重複エンジンを統合し、最適なものを選択",
            "files": [file for issue in duplicate_issues for file in issue["files"]],
            "effort": "medium",
            "impact": "high"
        })
    
    # 3. 共通インターフェースの導入
    if len(active_engines) > 1:
        recommendations.append({
            "priority": "medium",
            "category": "standardization",
            "title": "共通インターフェースの導入",
            "description": "全エンジンで共通のインターフェースを定義・実装",
            "effort": "high",
            "impact": "high"
        })
    
    # 4. パフォーマンス最適化
    excel_conflicts = [c for c in analysis_results["performance_conflicts"] 
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: if c.get("conflict_type") == "excel_output_competition"]
    if excel_conflicts:
        recommendations.append({
            "priority": "medium",
            "category": "optimization",
            "title": "Excel出力の一元化",
            "description": "複数エンジンのExcel出力を単一エンジンに集約",
            "effort": "medium",
            "impact": "medium"
        })
    
    return recommendations

def analyze_switch_frequency_impact(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """切替頻度への影響分析"""
    engines = analysis_results["engines_analyzed"]
    conflicts = analysis_results["execution_runtime_conflicts"]
    
    # 実行時競合による影響評価
    concurrent_engines = sum(1 for conflict in conflicts 
                           if conflict.get("conflict_type") == "concurrent_execution")
    
    # パフォーマンス低下予測
    performance_impact = 0
    if concurrent_engines > 0:
        performance_impact = min(50, concurrent_engines * 15)  # 最大50%の影響
    
    # 非アクティブエンジンによるコード複雑性影響
    inactive_engines = [e for e in engines if not e["is_active"]]
    complexity_impact = min(30, len(inactive_engines) * 5)  # 最大30%の影響
    
    total_impact = performance_impact + complexity_impact
    
    impact_level = "critical" if total_impact >= 40 else \
                  "high" if total_impact >= 25 else \
                  "medium" if total_impact >= 15 else "low"
    
    return {
        "performance_degradation_percent": performance_impact,
        "code_complexity_impact_percent": complexity_impact,
        "total_switch_frequency_impact_percent": total_impact,
        "impact_level": impact_level,
        "contributing_factors": {
            "concurrent_execution_engines": concurrent_engines,
            "inactive_engines_count": len(inactive_engines),
            "total_engines": len(engines)
        },
        "mitigation_potential": {
            "engine_consolidation": min(25, total_impact * 0.6),
            "interface_standardization": min(15, total_impact * 0.4),
            "performance_optimization": min(20, performance_impact * 0.8)
        }
    }

if __name__ == "__main__":
    try:
        results = analyze_engine_coexistence()
        
        # 結果サマリー表示
        summary = results["summary_findings"]
        print(f"\n[CHART] 分析サマリー:")
        print(f"総エンジン数: {summary['total_engines_detected']}")
        print(f"アクティブ: {summary['active_engines']}, 非アクティブ: {summary['inactive_engines']}")
        print(f"総問題数: {summary['total_issues_detected']}")
        print(f"重要問題数: {summary['high_severity_issues']}")
        print(f"深刻度レベル: {summary['overall_severity_level']}")
        print(f"共存健全度: {summary['coexistence_health_score']:.1f}/100")
        print(f"切替頻度影響: {summary['impact_on_switch_frequency']['impact_level']}")
        
        if summary["primary_concerns"]:
            print(f"\n[WARNING] 主要懸念事項:")
            for concern in summary["primary_concerns"]:
                print(f"  - {concern}")
        
    except Exception as e:
        print(f"[ERROR] Task 5.3 エラー: {e}")
        import traceback
        traceback.print_exc()