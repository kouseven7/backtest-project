"""
Task 5.1: IntelligentSwitchManager統合状況の詳細調査

目的:
1. IntelligentSwitchManagerクラスの実装状況分析
2. DSSMS各バックテスターでの統合・利用状況調査
3. 初期化、設定、実際の銘柄切替処理での利用状況調査
4. 問題点・改善点の特定

Author: GitHub Copilot Agent
Date: 2025-09-14
"""

import os
import re
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json


class IntelligentSwitchManagerAnalyzer:
    """IntelligentSwitchManager統合状況分析クラス"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.analysis_results = {
            "file_analysis": {},
            "integration_patterns": {},
            "usage_statistics": {},
            "implementation_issues": [],
            "recommendations": []
        }
        
    def analyze_intelligent_switch_manager_integration(self) -> Dict[str, Any]:
        """IntelligentSwitchManager統合状況の包括的分析"""
        print("=== Task 5.1: IntelligentSwitchManager統合状況調査開始 ===")
        
        # 1. IntelligentSwitchManagerファイルの存在確認
        isw_files = self._find_intelligent_switch_manager_files()
        
        # 2. DSSMS関連ファイルでの利用状況調査
        dssms_usage = self._analyze_dssms_usage_patterns()
        
        # 3. 統合パターン分析
        integration_analysis = self._analyze_integration_patterns(isw_files, dssms_usage)
        
        # 4. 問題点特定
        issues = self._identify_integration_issues(integration_analysis)
        
        # 5. 推奨改善策
        recommendations = self._generate_recommendations(issues)
        
        # 結果コンパイル
        results = {
            "timestamp": datetime.now().isoformat(),
            "intelligent_switch_manager_files": isw_files,
            "dssms_usage_patterns": dssms_usage,
            "integration_analysis": integration_analysis,
            "identified_issues": issues,
            "recommendations": recommendations,
            "summary_statistics": self._calculate_summary_statistics(isw_files, dssms_usage, issues)
        }
        
        self._generate_detailed_report(results)
        return results
    
    def _find_intelligent_switch_manager_files(self) -> List[Dict[str, Any]]:
        """IntelligentSwitchManager関連ファイルの検索"""
        print("1. IntelligentSwitchManager関連ファイル検索中...")
        
        isw_files = []
        search_patterns = [
            "*intelligent_switch_manager*",
            "*switch_manager*",
            "*intelligent_switch*"
        ]
        
        for pattern in search_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file() and file_path.suffix == '.py':
                    file_info = self._analyze_file_structure(file_path)
                    isw_files.append(file_info)
        
        print(f"   発見ファイル数: {len(isw_files)}")
        for file_info in isw_files:
            print(f"   - {file_info['relative_path']} ({file_info['size_bytes']} bytes)")
        
        return isw_files
    
    def _analyze_file_structure(self, file_path: Path) -> Dict[str, Any]:
        """ファイル構造分析"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # AST解析
            try:
                tree = ast.parse(content)
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                
            except SyntaxError:
                classes, functions, imports = [], [], []
            
            return {
                "absolute_path": str(file_path),
                "relative_path": str(file_path.relative_to(self.project_root)),
                "size_bytes": file_path.stat().st_size,
                "line_count": len(content.splitlines()),
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "contains_intelligent_switch_manager": "IntelligentSwitchManager" in content,
                "implementation_quality": self._assess_implementation_quality(content),
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            return {
                "absolute_path": str(file_path),
                "relative_path": str(file_path.relative_to(self.project_root)),
                "error": str(e),
                "accessible": False
            }
    
    def _assess_implementation_quality(self, content: str) -> Dict[str, Any]:
        """実装品質評価"""
        return {
            "has_docstrings": '"""' in content or "'''" in content,
            "has_type_hints": "->" in content or ":" in content,
            "has_error_handling": "try:" in content and "except" in content,
            "has_logging": any(log_pattern in content for log_pattern in ["logger", "logging", "log"]),
            "has_tests": "test" in content.lower() or "Test" in content,
            "complexity_score": len(re.findall(r'def |class |if |for |while ', content))
        }
    
    def _analyze_dssms_usage_patterns(self) -> Dict[str, Any]:
        """DSSMS関連ファイルでのIntelligentSwitchManager利用パターン分析"""
        print("2. DSSMS関連ファイルでの利用状況調査中...")
        
        dssms_files = list(self.project_root.rglob("*dssms*.py"))
        usage_patterns = {
            "files_using_isw": [],
            "files_not_using_isw": [],
            "import_patterns": {},
            "usage_contexts": {}
        }
        
        for file_path in dssms_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_analysis = {
                    "file_path": str(file_path.relative_to(self.project_root)),
                    "size_bytes": file_path.stat().st_size,
                    "imports_isw": self._check_isw_imports(content),
                    "uses_isw": self._check_isw_usage(content),
                    "initialization_patterns": self._extract_initialization_patterns(content),
                    "method_calls": self._extract_method_calls(content),
                    "integration_quality": self._assess_integration_quality(content)
                }
                
                if file_analysis["imports_isw"] or file_analysis["uses_isw"]:
                    usage_patterns["files_using_isw"].append(file_analysis)
                else:
                    usage_patterns["files_not_using_isw"].append(file_analysis)
                    
            except Exception as e:
                print(f"   エラー: {file_path} - {e}")
        
        print(f"   ISW利用ファイル: {len(usage_patterns['files_using_isw'])}")
        print(f"   ISW未利用ファイル: {len(usage_patterns['files_not_using_isw'])}")
        
        return usage_patterns
    
    def _check_isw_imports(self, content: str) -> List[str]:
        """IntelligentSwitchManagerのインポート確認"""
        import_patterns = [
            r"from\s+.*intelligent_switch_manager\s+import\s+.*IntelligentSwitchManager",
            r"import\s+.*intelligent_switch_manager",
            r"from\s+.*IntelligentSwitchManager",
            r"import\s+.*IntelligentSwitchManager"
        ]
        
        found_imports = []
        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_imports.extend(matches)
        
        return found_imports
    
    def _check_isw_usage(self, content: str) -> List[str]:
        """IntelligentSwitchManagerの使用確認"""
        usage_patterns = [
            r"IntelligentSwitchManager\(",
            r"intelligent_switch_manager\.",
            r"switch_manager\s*=",
            r"\.switch_manager",
            r"switch_decision"
        ]
        
        found_usage = []
        for pattern in usage_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_usage.extend(matches)
        
        return found_usage
    
    def _extract_initialization_patterns(self, content: str) -> List[str]:
        """初期化パターンの抽出"""
        init_patterns = [
            r"IntelligentSwitchManager\([^)]*\)",
            r"switch_manager\s*=\s*[^\\n]*",
            r"self\.switch_manager\s*=\s*[^\\n]*"
        ]
        
        found_patterns = []
        for pattern in init_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            found_patterns.extend(matches)
        
        return found_patterns
    
    def _extract_method_calls(self, content: str) -> List[str]:
        """メソッド呼び出しパターンの抽出"""
        method_patterns = [
            r"switch_manager\.[a-zA-Z_][a-zA-Z0-9_]*\(",
            r"intelligent_switch_manager\.[a-zA-Z_][a-zA-Z0-9_]*\(",
            r"IntelligentSwitchManager\.[a-zA-Z_][a-zA-Z0-9_]*\("
        ]
        
        found_calls = []
        for pattern in method_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_calls.extend(matches)
        
        return found_calls
    
    def _assess_integration_quality(self, content: str) -> Dict[str, Any]:
        """統合品質評価"""
        return {
            "proper_initialization": "IntelligentSwitchManager(" in content,
            "configuration_setup": any(pattern in content for pattern in ["config", "setup", "initialize"]),
            "error_handling": "try:" in content and "except" in content,
            "logging_integration": any(log_pattern in content for log_pattern in ["logger", "logging"]),
            "method_utilization_score": len(re.findall(r"switch_manager\.", content, re.IGNORECASE))
        }
    
    def _analyze_integration_patterns(self, isw_files: List[Dict], dssms_usage: Dict) -> Dict[str, Any]:
        """統合パターン分析"""
        print("3. 統合パターン分析中...")
        
        analysis = {
            "implementation_distribution": {},
            "usage_distribution": {},
            "integration_health": {},
            "version_consistency": {}
        }
        
        # 実装分布
        total_isw_implementations = len(isw_files)
        analysis["implementation_distribution"] = {
            "total_implementations": total_isw_implementations,
            "functional_implementations": len([f for f in isw_files if f.get("contains_intelligent_switch_manager", False)]),
            "average_file_size": sum(f.get("size_bytes", 0) for f in isw_files) / max(len(isw_files), 1),
            "implementation_complexity": sum(f.get("implementation_quality", {}).get("complexity_score", 0) for f in isw_files)
        }
        
        # 利用分布
        total_dssms_files = len(dssms_usage["files_using_isw"]) + len(dssms_usage["files_not_using_isw"])
        analysis["usage_distribution"] = {
            "total_dssms_files": total_dssms_files,
            "files_using_isw": len(dssms_usage["files_using_isw"]),
            "files_not_using_isw": len(dssms_usage["files_not_using_isw"]),
            "usage_rate": len(dssms_usage["files_using_isw"]) / max(total_dssms_files, 1) * 100
        }
        
        # 統合健全性
        analysis["integration_health"] = {
            "has_implementations": total_isw_implementations > 0,
            "has_usage": len(dssms_usage["files_using_isw"]) > 0,
            "implementation_usage_ratio": len(dssms_usage["files_using_isw"]) / max(total_isw_implementations, 1),
            "average_integration_quality": self._calculate_average_integration_quality(dssms_usage["files_using_isw"])
        }
        
        return analysis
    
    def _calculate_average_integration_quality(self, using_files: List[Dict]) -> float:
        """平均統合品質計算"""
        if not using_files:
            return 0.0
        
        total_score = 0
        for file_info in using_files:
            quality = file_info.get("integration_quality", {})
            score = sum(1 for key, value in quality.items() if key != "method_utilization_score" and value)
            total_score += score / 4  # 4つの基本品質指標
        
        return total_score / len(using_files) * 100
    
    def _identify_integration_issues(self, integration_analysis: Dict) -> List[Dict[str, Any]]:
        """統合問題の特定"""
        print("4. 統合問題特定中...")
        
        issues = []
        
        # 実装不足の問題
        if integration_analysis["implementation_distribution"]["total_implementations"] == 0:
            issues.append({
                "type": "missing_implementation",
                "severity": "critical",
                "description": "IntelligentSwitchManagerの実装が見つからない",
                "impact": "銘柄切替機能が完全に動作しない可能性"
            })
        
        # 利用率の問題
        usage_rate = integration_analysis["usage_distribution"]["usage_rate"]
        if usage_rate < 50:
            issues.append({
                "type": "low_usage_rate",
                "severity": "high",
                "description": f"DSSMSファイルでのISW利用率が低い ({usage_rate:.1f}%)",
                "impact": "一貫性のない銘柄切替処理"
            })
        
        # 統合品質の問題
        avg_quality = integration_analysis["integration_health"]["average_integration_quality"]
        if avg_quality < 60:
            issues.append({
                "type": "poor_integration_quality",
                "severity": "medium",
                "description": f"統合品質が低い (平均{avg_quality:.1f}%)",
                "impact": "エラーハンドリングや設定不備"
            })
        
        # 実装と利用のアンバランス
        impl_usage_ratio = integration_analysis["integration_health"]["implementation_usage_ratio"]
        if impl_usage_ratio < 0.5:
            issues.append({
                "type": "implementation_usage_imbalance",
                "severity": "medium",
                "description": f"実装数に対して利用が少ない (比率: {impl_usage_ratio:.2f})",
                "impact": "未使用の重複実装または利用漏れ"
            })
        
        return issues
    
    def _generate_recommendations(self, issues: List[Dict]) -> List[Dict[str, Any]]:
        """推奨改善策の生成"""
        print("5. 推奨改善策生成中...")
        
        recommendations = []
        
        for issue in issues:
            if issue["type"] == "missing_implementation":
                recommendations.append({
                    "priority": "highest",
                    "action": "IntelligentSwitchManager実装の作成",
                    "description": "標準的なISMクラスを実装し、基本的な銘柄切替機能を提供",
                    "estimated_effort": "高（3-5日）"
                })
            
            elif issue["type"] == "low_usage_rate":
                recommendations.append({
                    "priority": "high",
                    "action": "DSSMS統合の標準化",
                    "description": "全DSSMSバックテスターでISMを統一利用する標準パターンを確立",
                    "estimated_effort": "中（2-3日）"
                })
            
            elif issue["type"] == "poor_integration_quality":
                recommendations.append({
                    "priority": "medium",
                    "action": "統合品質の向上",
                    "description": "エラーハンドリング、ログ機能、設定管理の改善",
                    "estimated_effort": "低（1-2日）"
                })
            
            elif issue["type"] == "implementation_usage_imbalance":
                recommendations.append({
                    "priority": "medium",
                    "action": "重複実装の整理",
                    "description": "不要な重複実装を削除し、統一実装の利用を促進",
                    "estimated_effort": "低（1日）"
                })
        
        return recommendations
    
    def _calculate_summary_statistics(self, isw_files: List, dssms_usage: Dict, issues: List) -> Dict[str, Any]:
        """サマリー統計計算"""
        return {
            "total_isw_implementations": len(isw_files),
            "total_dssms_files": len(dssms_usage["files_using_isw"]) + len(dssms_usage["files_not_using_isw"]),
            "files_with_isw": len(dssms_usage["files_using_isw"]),
            "integration_coverage": len(dssms_usage["files_using_isw"]) / max(len(dssms_usage["files_using_isw"]) + len(dssms_usage["files_not_using_isw"]), 1) * 100,
            "critical_issues": len([i for i in issues if i["severity"] == "critical"]),
            "high_issues": len([i for i in issues if i["severity"] == "high"]),
            "medium_issues": len([i for i in issues if i["severity"] == "medium"]),
            "total_issues": len(issues),
            "overall_health_score": max(0, 100 - len(issues) * 25)  # 問題1つにつき25点減点
        }
    
    def _generate_detailed_report(self, results: Dict[str, Any]):
        """詳細レポート生成"""
        print("\\n=== Task 5.1分析結果サマリー ===")
        stats = results["summary_statistics"]
        
        print(f"ISM実装数: {stats['total_isw_implementations']}")
        print(f"DSSMS総ファイル数: {stats['total_dssms_files']}")
        print(f"ISM利用ファイル数: {stats['files_with_isw']}")
        print(f"統合カバレッジ: {stats['integration_coverage']:.1f}%")
        print(f"健全性スコア: {stats['overall_health_score']:.1f}/100")
        
        print(f"\\n問題数:")
        print(f"  Critical: {stats['critical_issues']}")
        print(f"  High: {stats['high_issues']}")
        print(f"  Medium: {stats['medium_issues']}")
        print(f"  Total: {stats['total_issues']}")
        
        print("\\n=== 主要な発見事項 ===")
        for issue in results["identified_issues"]:
            print(f"- [{issue['severity'].upper()}] {issue['description']}")
        
        print("\\n=== 推奨改善策 ===")
        for rec in results["recommendations"]:
            print(f"- [{rec['priority'].upper()}] {rec['action']}")
            print(f"  {rec['description']} (工数: {rec['estimated_effort']})")


def main():
    """Task 5.1メイン実行"""
    analyzer = IntelligentSwitchManagerAnalyzer()
    results = analyzer.analyze_intelligent_switch_manager_integration()
    
    # 結果をJSONで保存
    output_file = "task_5_1_intelligent_switch_manager_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n詳細結果を {output_file} に保存しました")
    return results


if __name__ == "__main__":
    main()