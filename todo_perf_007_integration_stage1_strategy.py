#!/usr/bin/env python3
"""
TODO-PERF-007 Stage 1: 統合対象確認・統合戦略策定

目的:
- 実際のScreenerファイル構造確認・最適化対象メソッド特定
- todo_perf_007_stage*.py実装済みコンポーネントの詳細分析
- ParallelDataFetcher・SmartCache・OptimizedAlgorithmEngine統合箇所設計
- 統合順序・依存関係・リスク評価・ロールバック戦略策定
- SystemFallbackPolicy統合・エラーハンドリング設計

実行時間: 20分で完了・統合戦略確定
"""

import os
import sys
import ast
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import re

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class IntegrationTargetAnalyzer:
    """統合対象確認・統合戦略策定システム"""
    
    def __init__(self):
        self.analysis_results = {
            "screener_analysis": {},
            "optimization_components": {},
            "integration_strategy": {},
            "risk_assessment": {},
            "implementation_plan": {}
        }
        
        # 統合対象ファイル
        self.target_files = {
            "screener": "src/dssms/nikkei225_screener.py",
            "stage_2_parallel": "todo_perf_007_stage2_parallel_cache.py",
            "stage_3_algorithm": "todo_perf_007_stage3_algorithm_optimization.py",
            "stage_4_validation": "todo_perf_007_stage4_final_validation.py"
        }
        
        # 統合対象コンポーネント
        self.optimization_components = {
            "ParallelDataFetcher": {
                "source_file": "todo_perf_007_stage2_parallel_cache.py",
                "target_methods": ["market_cap_filter", "affordability_filter", "volume_filter", "price_filter"],
                "expected_improvement": "70%削減（52.5秒→15秒目標）"
            },
            "SmartCache": {
                "source_file": "todo_perf_007_stage2_parallel_cache.py",
                "target_methods": ["yfinance API呼び出し全般"],
                "expected_improvement": "2回目以降80%+削減（150秒以上削減）"
            },
            "OptimizedAlgorithmEngine": {
                "source_file": "todo_perf_007_stage3_algorithm_optimization.py",
                "target_methods": ["final_selection"],
                "expected_improvement": "67%削減（45.7秒→15秒目標）"
            }
        }
        
    def analyze_integration_targets(self):
        """統合対象分析実行"""
        print("[SEARCH] Stage 1: 統合対象確認・統合戦略策定開始")
        print("="*70)
        
        try:
            # 1. Screenerファイル構造分析
            screener_analysis = self._analyze_screener_structure()
            
            # 2. 最適化コンポーネント詳細分析
            components_analysis = self._analyze_optimization_components()
            
            # 3. 統合戦略策定
            integration_strategy = self._design_integration_strategy()
            
            # 4. リスク評価・ロールバック戦略
            risk_assessment = self._assess_integration_risks()
            
            # 5. 実装計画策定
            implementation_plan = self._create_implementation_plan()
            
            # 結果統合
            self.analysis_results.update({
                "screener_analysis": screener_analysis,
                "optimization_components": components_analysis,
                "integration_strategy": integration_strategy,
                "risk_assessment": risk_assessment,
                "implementation_plan": implementation_plan
            })
            
            return self.analysis_results
            
        except Exception as e:
            print(f"[ERROR] Stage 1 分析エラー: {e}")
            return {"error": str(e)}
    
    def _analyze_screener_structure(self) -> Dict[str, Any]:
        """Screenerファイル構造詳細分析"""
        
        screener_path = project_root / self.target_files["screener"]
        
        if not screener_path.exists():
            return {"error": f"Screenerファイル不存在: {screener_path}"}
        
        try:
            with open(screener_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # AST解析でクラス・メソッド構造取得
            tree = ast.parse(content)
            
            classes = []
            methods = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes.append({
                        "name": node.name,
                        "methods": class_methods,
                        "line": node.lineno
                    })
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    methods.append({
                        "name": node.name,
                        "line": node.lineno
                    })
                elif isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.extend([f"{module}.{alias.name}" for alias in node.names])
            
            # 統合対象メソッド特定
            target_methods_analysis = self._identify_target_methods(content)
            
            # 既存最適化チェック
            existing_optimizations = self._check_existing_optimizations(content)
            
            return {
                "file_info": {
                    "path": str(screener_path),
                    "size": len(content),
                    "lines": len(content.split('\n'))
                },
                "structure": {
                    "classes": classes,
                    "methods": methods,
                    "imports": imports
                },
                "target_methods": target_methods_analysis,
                "existing_optimizations": existing_optimizations,
                "integration_readiness": "準備完了" if classes else "要調査"
            }
            
        except Exception as e:
            return {"error": f"Screener分析エラー: {e}"}
    
    def _identify_target_methods(self, content: str) -> Dict[str, Any]:
        """統合対象メソッド特定"""
        
        target_patterns = {
            "market_cap_filter": {
                "pattern": r"def.*market.*cap.*filter|def.*filter.*market.*cap",
                "description": "市場キャップフィルタリング（最大ボトルネック52.5秒）",
                "optimization": "ParallelDataFetcher統合"
            },
            "final_selection": {
                "pattern": r"def.*final.*selection|def.*select.*final",
                "description": "最終銘柄選択（第2ボトルネック45.7秒）",
                "optimization": "OptimizedAlgorithmEngine統合"
            },
            "affordability_filter": {
                "pattern": r"def.*affordability.*filter|def.*affordable",
                "description": "購入可能性フィルタ（33.1秒）",
                "optimization": "ParallelDataFetcher統合"
            },
            "volume_filter": {
                "pattern": r"def.*volume.*filter|def.*filter.*volume",
                "description": "出来高フィルタ（28.5秒）",
                "optimization": "ParallelDataFetcher統合"
            },
            "price_filter": {
                "pattern": r"def.*price.*filter|def.*filter.*price",
                "description": "価格フィルタ（23.4秒）",
                "optimization": "ParallelDataFetcher統合"
            }
        }
        
        found_methods = {}
        
        for method_name, info in target_patterns.items():
            matches = re.findall(info["pattern"], content, re.IGNORECASE)
            if matches:
                found_methods[method_name] = {
                    "found": True,
                    "matches": matches,
                    "description": info["description"],
                    "optimization": info["optimization"]
                }
            else:
                found_methods[method_name] = {
                    "found": False,
                    "description": info["description"],
                    "optimization": info["optimization"],
                    "search_required": True
                }
        
        # yfinance呼び出し箇所特定
        yfinance_calls = len(re.findall(r"yf\.|yfinance\.|\.download\(|\.info\(|\.history\(", content, re.IGNORECASE))
        found_methods["yfinance_calls"] = {
            "count": yfinance_calls,
            "description": "yfinance API呼び出し箇所",
            "optimization": "SmartCache統合"
        }
        
        return found_methods
    
    def _check_existing_optimizations(self, content: str) -> Dict[str, Any]:
        """既存最適化実装チェック"""
        
        optimization_indicators = {
            "parallel_processing": [
                "ThreadPoolExecutor", "concurrent.futures", "multiprocessing",
                "Pool", "parallel", "async", "await"
            ],
            "caching": [
                "cache", "Cache", "lru_cache", "functools.lru_cache",
                "redis", "memcached", "pickle"
            ],
            "numpy_optimization": [
                "numpy", "np.array", "vectorize", "broadcasting",
                "np.where", "np.apply"
            ],
            "api_optimization": [
                "batch", "bulk", "rate_limit", "throttle",
                "retry", "backoff"
            ]
        }
        
        found_optimizations = {}
        
        for category, indicators in optimization_indicators.items():
            found = []
            for indicator in indicators:
                if indicator in content:
                    found.append(indicator)
            
            found_optimizations[category] = {
                "found": found,
                "count": len(found),
                "status": "[OK] 実装済み" if found else "[ERROR] 未実装"
            }
        
        return found_optimizations
    
    def _analyze_optimization_components(self) -> Dict[str, Any]:
        """最適化コンポーネント詳細分析"""
        
        components_analysis = {}
        
        for component_name, component_info in self.optimization_components.items():
            source_path = project_root / component_info["source_file"]
            
            if not source_path.exists():
                components_analysis[component_name] = {
                    "status": "[ERROR] ファイル不存在",
                    "path": str(source_path)
                }
                continue
            
            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # クラス定義検索
                class_pattern = rf"class\s+{component_name}\s*[\(:]"
                class_match = re.search(class_pattern, content)
                
                if class_match:
                    # クラス詳細分析
                    class_analysis = self._analyze_component_class(content, component_name)
                    
                    components_analysis[component_name] = {
                        "status": "[OK] 実装済み",
                        "path": str(source_path),
                        "file_size": len(content),
                        "class_analysis": class_analysis,
                        "target_methods": component_info["target_methods"],
                        "expected_improvement": component_info["expected_improvement"],
                        "integration_ready": True
                    }
                else:
                    components_analysis[component_name] = {
                        "status": "[ERROR] クラス未発見",
                        "path": str(source_path),
                        "search_pattern": class_pattern
                    }
                    
            except Exception as e:
                components_analysis[component_name] = {
                    "status": "[ERROR] 分析エラー",
                    "error": str(e)
                }
        
        return components_analysis
    
    def _analyze_component_class(self, content: str, class_name: str) -> Dict[str, Any]:
        """コンポーネントクラス詳細分析"""
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    methods = []
                    attributes = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append({
                                "name": item.name,
                                "line": item.lineno,
                                "args": [arg.arg for arg in item.args.args]
                            })
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    attributes.append(target.id)
                    
                    return {
                        "methods": methods,
                        "attributes": attributes,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node)
                    }
            
            return {"error": f"クラス {class_name} 未発見"}
            
        except Exception as e:
            return {"error": f"AST解析エラー: {e}"}
    
    def _design_integration_strategy(self) -> Dict[str, Any]:
        """統合戦略策定"""
        
        integration_strategy = {
            "overall_approach": {
                "strategy": "段階的統合・個別効果測定・安全優先",
                "sequence": [
                    "ParallelDataFetcher統合（最大効果期待）",
                    "SmartCache統合（長期効果基盤）",
                    "OptimizedAlgorithmEngine統合（第2効果源）"
                ],
                "safety_measures": [
                    "既存Screenerの完全バックアップ",
                    "各段階での機能完全性確認",
                    "パフォーマンス測定・効果検証",
                    "即座ロールバック機構準備"
                ]
            },
            "technical_integration": {
                "ParallelDataFetcher": {
                    "integration_points": [
                        "market_cap_filter内のfor loop並列化",
                        "affordability_filter内のyfinance呼び出し並列化",
                        "volume_filter・price_filter並列処理統合"
                    ],
                    "implementation": "ThreadPoolExecutor(max_workers=8)統合",
                    "fallback": "並列処理失敗時の逐次処理フォールバック"
                },
                "SmartCache": {
                    "integration_points": [
                        "yfinance.download()呼び出し前後",
                        "株価・時価総額・出来高データ取得時",
                        "24時間キャッシュ有効期限設定"
                    ],
                    "implementation": "メモリ+JSON永続化統合",
                    "fallback": "キャッシュ失敗時の直接API呼び出し"
                },
                "OptimizedAlgorithmEngine": {
                    "integration_points": [
                        "final_selection内のスコア計算部分",
                        "銘柄比較・ソート・選択アルゴリズム",
                        "numpy配列ベクトル化処理"
                    ],
                    "implementation": "既存ロジック置換・計算最適化",
                    "fallback": "最適化失敗時の既存アルゴリズム維持"
                }
            },
            "quality_assurance": {
                "testing_strategy": [
                    "各統合段階でのユニットテスト実行",
                    "銘柄選択結果の一致性確認",
                    "パフォーマンス測定・改善効果検証",
                    "E2E統合テスト・実用性確認"
                ],
                "quality_gates": [
                    "銘柄選択精度100%維持",
                    "データ品質完全保持",
                    "既存機能完全性確保",
                    "SystemFallbackPolicy統合"
                ]
            }
        }
        
        return integration_strategy
    
    def _assess_integration_risks(self) -> Dict[str, Any]:
        """リスク評価・ロールバック戦略策定"""
        
        risk_assessment = {
            "high_risks": [
                {
                    "risk": "既存Screener機能破壊",
                    "probability": "低",
                    "impact": "高",
                    "mitigation": "完全バックアップ・段階的統合・即座ロールバック"
                },
                {
                    "risk": "パフォーマンス劣化",
                    "probability": "低",
                    "impact": "高",
                    "mitigation": "詳細測定・最適化調整・フォールバック実装"
                }
            ],
            "medium_risks": [
                {
                    "risk": "API制限違反",
                    "probability": "中",
                    "impact": "中",
                    "mitigation": "レート制限実装・エラーハンドリング強化"
                },
                {
                    "risk": "キャッシュデータ整合性",
                    "probability": "中",
                    "impact": "中",
                    "mitigation": "検証機構・自動更新・手動クリア機能"
                }
            ],
            "low_risks": [
                {
                    "risk": "統合複雑度",
                    "probability": "中",
                    "impact": "低",
                    "mitigation": "段階的アプローチ・詳細ドキュメント"
                }
            ],
            "rollback_strategy": {
                "backup_approach": [
                    "統合前の完全ファイルバックアップ",
                    "Git commit・branch作成",
                    "設定値バックアップ"
                ],
                "rollback_triggers": [
                    "銘柄選択結果の不一致",
                    "パフォーマンス20%以上劣化",
                    "システムエラー・例外発生",
                    "品質テスト失敗"
                ],
                "recovery_process": [
                    "即座の処理中断",
                    "バックアップファイル復元",
                    "システム動作確認",
                    "原因分析・再統合計画"
                ]
            }
        }
        
        return risk_assessment
    
    def _create_implementation_plan(self) -> Dict[str, Any]:
        """実装計画策定"""
        
        implementation_plan = {
            "stage_2_parallel": {
                "duration": "25分",
                "priority": "最高（52.5秒ボトルネック解決）",
                "tasks": [
                    "ParallelDataFetcher クラスの実際のScreenerへの統合",
                    "market_cap_filter並列化実装（ThreadPoolExecutor）",
                    "affordability_filter・volume_filter・price_filter並列処理統合",
                    "レート制限・エラーハンドリング実装",
                    "SystemFallbackPolicy統合・障害時処理",
                    "パフォーマンステスト・52.5秒→15秒達成確認"
                ],
                "success_criteria": "market_cap_filter実行時間70%削減確認"
            },
            "stage_3_cache_algorithm": {
                "duration": "30分",
                "priority": "高（45.7秒+長期効果）",
                "tasks": [
                    "SmartCache統合・yfinance API呼び出し最適化",
                    "24時間キャッシュ有効期限・JSON永続化実装",
                    "OptimizedAlgorithmEngine統合・final_selection最適化",
                    "numpy配列ベクトル化・計算パイプライン統合",
                    "重複計算排除・効率化実装",
                    "統合エラーハンドリング・品質保証"
                ],
                "success_criteria": "final_selection実行時間67%削減＋キャッシュ効果確認"
            },
            "stage_4_validation": {
                "duration": "15分", 
                "priority": "中（統合効果検証）",
                "tasks": [
                    "統合後Screener E2Eテスト実行",
                    "183秒→45-50秒削減達成確認",
                    "各フィルター段階別パフォーマンス測定",
                    "銘柄選択品質・データ品質維持確認",
                    "SystemFallbackPolicy統合動作テスト",
                    "実用レベル達成評価・文書化"
                ],
                "success_criteria": "74.6%削減達成・実用レベル確認"
            },
            "overall_timeline": {
                "total_duration": "70分（Stage 2-4）",
                "critical_path": "Stage 2 ParallelDataFetcher統合",
                "contingency": "90分上限・Stage単位分割可能",
                "success_definition": "183秒→46.5秒以下達成・実測値確認"
            }
        }
        
        return implementation_plan
    
    def generate_integration_strategy_report(self):
        """統合戦略レポート生成"""
        
        try:
            analysis_results = self.analyze_integration_targets()
            
            if "error" in analysis_results:
                return analysis_results, None
            
            # 戦略サマリー生成
            strategy_summary = {
                "stage_1_completion": {
                    "execution_date": datetime.now().isoformat(),
                    "status": "[OK] 完了",
                    "screener_readiness": analysis_results["screener_analysis"].get("integration_readiness", "要調査"),
                    "components_ready": len([c for c in analysis_results["optimization_components"].values() if c.get("integration_ready", False)]),
                    "total_components": len(self.optimization_components)
                },
                "integration_feasibility": {
                    "overall_assessment": "高（実装済みコンポーネント統合中心）",
                    "technical_risk": "低（段階的統合・ロールバック準備）",
                    "expected_performance": "183秒→45-50秒（74.6%削減）",
                    "implementation_time": "70分（Stage 2-4）"
                },
                "critical_success_factors": [
                    "既存Screener機能完全性維持",
                    "段階的統合・個別効果測定",
                    "SystemFallbackPolicy統合継続",
                    "パフォーマンス劣化防止・品質保証"
                ],
                "next_steps": [
                    "Stage 2: ParallelDataFetcher統合実装（25分）",
                    "Stage 3: SmartCache・OptimizedAlgorithmEngine統合（30分）", 
                    "Stage 4: 統合効果検証・実用レベル達成確認（15分）"
                ]
            }
            
            # 完全レポート統合
            complete_report = {
                "summary": strategy_summary,
                "detailed_analysis": analysis_results
            }
            
            # レポート保存
            report_file = f"TODO_PERF_007_Stage1_Integration_Strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(complete_report, f, ensure_ascii=False, indent=2)
            
            print(f"📄 統合戦略レポート保存: {report_file}")
            
            return complete_report, report_file
            
        except Exception as e:
            print(f"[ERROR] Stage 1 レポート生成エラー: {e}")
            return {"error": str(e)}, None

def main():
    """Stage 1 メイン実行"""
    print("[SEARCH] TODO-PERF-007 Stage 1: 統合対象確認・統合戦略策定開始")
    print("目標: 20分で統合戦略確定・Stage 2-4実装準備完了")
    print("="*80)
    
    try:
        analyzer = IntegrationTargetAnalyzer()
        results, report_file = analyzer.generate_integration_strategy_report()
        
        if "error" not in results:
            print("\n" + "="*80)
            print("[TARGET] Stage 1: 統合対象確認・統合戦略策定完了")
            print("="*80)
            
            summary = results["summary"]["stage_1_completion"]
            feasibility = results["summary"]["integration_feasibility"]
            
            print(f"\n[SEARCH] Stage 1完了状況:")
            print(f"  ステータス: {summary['status']}")
            print(f"  Screener準備: {summary['screener_readiness']}")
            print(f"  コンポーネント準備: {summary['components_ready']}/{summary['total_components']}")
            
            print(f"\n[CHART] 統合実現可能性:")
            print(f"  総合評価: {feasibility['overall_assessment']}")
            print(f"  技術リスク: {feasibility['technical_risk']}")
            print(f"  期待パフォーマンス: {feasibility['expected_performance']}")
            print(f"  実装時間: {feasibility['implementation_time']}")
            
            next_steps = results["summary"]["next_steps"]
            print(f"\n[ROCKET] 次ステップ:")
            for step in next_steps:
                print(f"  [OK] {step}")
            
            critical_factors = results["summary"]["critical_success_factors"]
            print(f"\n[WARNING] 重要成功要因:")
            for factor in critical_factors:
                print(f"  [TARGET] {factor}")
            
            print(f"\n📄 詳細戦略レポート: {report_file}")
            
            print("\n" + "="*80)
            print("[OK] Stage 1完了 → Stage 2 ParallelDataFetcher統合実装開始準備完了")
            print("[TARGET] 次作業: market_cap_filter並列化（52.5秒→15秒目標）")
            print("⏱️ 予定時間: 25分（ThreadPoolExecutor統合・レート制限・SystemFallbackPolicy）")
            print("="*80)
            
            return True
        else:
            print(f"\n[ERROR] Stage 1 失敗: {results.get('error', '不明なエラー')}")
            return False
            
    except Exception as e:
        print(f"\n💥 Stage 1 実行エラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)