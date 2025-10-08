#!/usr/bin/env python3
"""
hierarchical_ranking_system ボトルネック詳細分析ツール

TODO-PERF-001 Stage 1: 2422msの巨大ボトルネック根本原因特定
"""

import time
import importlib
import importlib.util
import sys
import traceback
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import ast
import os

class HierarchicalRankingBottleneckAnalyzer:
    """hierarchical_ranking_system ボトルネック分析器"""
    
    def __init__(self):
        self.analysis_results = {}
        self.import_chain = {}
        self.dependency_tree = {}
        self.bottleneck_sources = []
        
    def analyze_import_dependencies(self, module_path: str) -> Dict[str, Any]:
        """インポート依存関係の詳細分析"""
        print("[SEARCH] hierarchical_ranking_system.py 依存関係分析中...")
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # ASTパースによるインポート分析
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'type': 'import',
                            'module': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    imports.append({
                        'type': 'from_import',
                        'module': node.module,
                        'names': [alias.name for alias in node.names],
                        'line': node.lineno
                    })
            
            # 各インポートのコスト測定
            import_costs = {}
            heavy_imports = []
            
            for imp in imports:
                if imp['type'] == 'import':
                    module_name = imp['module']
                elif imp['type'] == 'from_import':
                    module_name = imp['module']
                else:
                    continue
                
                if module_name and not module_name.startswith('.'):
                    cost = self._measure_import_cost(module_name)
                    import_costs[module_name] = cost
                    
                    if cost > 100:  # 100ms以上を重いインポートと判定
                        heavy_imports.append({
                            'module': module_name,
                            'cost_ms': cost,
                            'line': imp['line'],
                            'severity': 'critical' if cost > 500 else 'high' if cost > 200 else 'medium'
                        })
            
            # 結果整理
            analysis = {
                'total_imports': len(imports),
                'import_details': imports,
                'import_costs': import_costs,
                'heavy_imports': sorted(heavy_imports, key=lambda x: x['cost_ms'], reverse=True),
                'total_import_cost': sum(import_costs.values()),
                'optimization_potential': sum(cost for cost in import_costs.values() if cost > 100)
            }
            
            print(f"  [CHART] 総インポート数: {analysis['total_imports']}")
            print(f"  ⏱️ 総インポートコスト: {analysis['total_import_cost']:.1f}ms")
            print(f"  [TARGET] 最適化可能コスト: {analysis['optimization_potential']:.1f}ms")
            print(f"  🔴 重いインポート: {len(heavy_imports)}個")
            
            for heavy in heavy_imports[:5]:  # Top 5表示
                print(f"    - {heavy['module']}: {heavy['cost_ms']:.1f}ms ({heavy['severity']})")
            
            return analysis
            
        except Exception as e:
            print(f"[ERROR] 依存関係分析エラー: {e}")
            return {}
    
    def _measure_import_cost(self, module_name: str) -> float:
        """個別モジュールのインポートコスト測定"""
        try:
            start_time = time.perf_counter()
            
            # 新しいPythonプロセスでインポート時間を測定
            # 既存のインポートキャッシュの影響を回避
            cmd = [sys.executable, '-c', f'import {module_name}']
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            if result.returncode == 0:
                end_time = time.perf_counter()
                return (end_time - start_time) * 1000
            else:
                # モジュールが見つからない場合は0ms
                return 0.0
                
        except subprocess.TimeoutExpired:
            return 10000.0  # タイムアウトは10秒として記録
        except Exception:
            return 0.0
    
    def analyze_class_initialization_overhead(self, module_path: str) -> Dict[str, Any]:
        """クラス初期化・メソッド定義のオーバーヘッド分析"""
        print("🏗️ クラス構造・初期化オーバーヘッド分析中...")
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            classes = []
            functions = []
            
            # クラス・関数定義の分析
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes.append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods_count': len(methods),
                        'methods': [m.name for m in methods],
                        'has_init': '__init__' in [m.name for m in methods]
                    })
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:  # トップレベル関数
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args_count': len(node.args.args)
                    })
            
            # 初期化コスト推定
            initialization_complexity = {
                'total_classes': len(classes),
                'total_methods': sum(c['methods_count'] for c in classes),
                'total_functions': len(functions),
                'complex_classes': [c for c in classes if c['methods_count'] > 10],
                'large_init_classes': [c for c in classes if c['has_init'] and c['methods_count'] > 5]
            }
            
            print(f"  [CHART] 総クラス数: {initialization_complexity['total_classes']}")
            print(f"  [TOOL] 総メソッド数: {initialization_complexity['total_methods']}")
            print(f"  [UP] 複雑クラス: {len(initialization_complexity['complex_classes'])}個")
            
            for complex_class in initialization_complexity['complex_classes']:
                print(f"    - {complex_class['name']}: {complex_class['methods_count']}メソッド")
            
            return {
                'classes': classes,
                'functions': functions,
                'complexity_analysis': initialization_complexity
            }
            
        except Exception as e:
            print(f"[ERROR] クラス分析エラー: {e}")
            return {}
    
    def analyze_init_py_impact(self) -> Dict[str, Any]:
        """__init__.py 自動インポート影響度測定"""
        print("📦 __init__.py 自動インポート影響度分析中...")
        
        init_files = [
            'src/dssms/__init__.py',
            'src/__init__.py',
            'src/config/__init__.py'
        ]
        
        init_analysis = {}
        
        for init_file in init_files:
            if Path(init_file).exists():
                try:
                    # __init__.pyの内容分析
                    with open(init_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # __init__.pyインポート時間測定
                    start_time = time.perf_counter()
                    
                    # インポートテスト（モジュール名を構築）
                    module_name = init_file.replace('/', '.').replace('\\', '.').replace('.py', '').replace('src.', '')
                    
                    try:
                        importlib.import_module(module_name)
                        end_time = time.perf_counter()
                        import_time = (end_time - start_time) * 1000
                    except Exception:
                        import_time = 0.0
                    
                    # __all__ や * インポートの検出
                    has_all = '__all__' in content
                    has_star_imports = 'from' in content and '*' in content
                    line_count = len(content.splitlines())
                    
                    init_analysis[init_file] = {
                        'import_time_ms': import_time,
                        'has_all_definition': has_all,
                        'has_star_imports': has_star_imports,
                        'line_count': line_count,
                        'content_preview': content[:200] + '...' if len(content) > 200 else content
                    }
                    
                    print(f"  📄 {init_file}: {import_time:.1f}ms")
                    
                except Exception as e:
                    print(f"  [ERROR] {init_file} 分析エラー: {e}")
        
        return init_analysis
    
    def identify_lazy_loading_opportunities(self, dependency_analysis: Dict) -> Dict[str, Any]:
        """遅延ローディング機会の特定"""
        print("[TARGET] 遅延ローディング機会特定中...")
        
        # 重いインポートを遅延可能カテゴリに分類
        heavy_imports = dependency_analysis.get('heavy_imports', [])
        
        categorization = {
            'visualization': [],    # 可視化関連（遅延可能度: 高）
            'data_analysis': [],    # データ分析関連（遅延可能度: 中）
            'numerical': [],        # 数値計算関連（遅延可能度: 低）
            'io_operations': [],    # I/O操作関連（遅延可能度: 中）
            'core_system': []       # コアシステム（遅延可能度: 低）
        }
        
        # カテゴリ分類ルール
        classification_rules = {
            'visualization': ['matplotlib', 'seaborn', 'plotly', 'bokeh'],
            'data_analysis': ['scipy', 'sklearn', 'statsmodels'],
            'numerical': ['numpy', 'pandas'],  # 基本的な数値処理は即座必要
            'io_operations': ['yfinance', 'requests', 'urllib'],
            'core_system': ['datetime', 'os', 'sys', 'pathlib']
        }
        
        for heavy in heavy_imports:
            module = heavy['module']
            categorized = False
            
            for category, keywords in classification_rules.items():
                if any(keyword in module.lower() for keyword in keywords):
                    categorization[category].append(heavy)
                    categorized = True
                    break
            
            if not categorized:
                # 不明なモジュールはdata_analysisに分類
                categorization['data_analysis'].append(heavy)
        
        # 最適化優先度計算
        optimization_plan = []
        
        priority_weights = {
            'visualization': 0.9,      # 高い遅延可能性
            'io_operations': 0.8,      # 高い遅延可能性
            'data_analysis': 0.6,      # 中程度の遅延可能性
            'numerical': 0.3,          # 低い遅延可能性
            'core_system': 0.1         # 最低の遅延可能性
        }
        
        for category, imports in categorization.items():
            for imp in imports:
                priority_score = imp['cost_ms'] * priority_weights[category]
                optimization_plan.append({
                    'module': imp['module'],
                    'category': category,
                    'cost_ms': imp['cost_ms'],
                    'priority_score': priority_score,
                    'lazy_loading_feasibility': priority_weights[category]
                })
        
        # 優先度順にソート
        optimization_plan.sort(key=lambda x: x['priority_score'], reverse=True)
        
        print(f"  [TARGET] 最適化対象モジュール: {len(optimization_plan)}個")
        for i, plan in enumerate(optimization_plan[:5]):  # Top 5表示
            print(f"    {i+1}. {plan['module']} ({plan['category']}): "
                  f"{plan['cost_ms']:.1f}ms, 遅延可能度: {plan['lazy_loading_feasibility']:.1%}")
        
        return {
            'categorization': categorization,
            'optimization_plan': optimization_plan,
            'total_optimizable_cost': sum(p['cost_ms'] * p['lazy_loading_feasibility'] 
                                        for p in optimization_plan)
        }
    
    def generate_comprehensive_analysis_report(self) -> Dict[str, Any]:
        """総合分析レポート生成"""
        print("[LIST] hierarchical_ranking_system 総合ボトルネック分析実行中...")
        
        module_path = 'src/dssms/hierarchical_ranking_system.py'
        
        if not Path(module_path).exists():
            print(f"[ERROR] {module_path} が見つかりません")
            return {}
        
        # Stage 1: 各種分析実行
        dependency_analysis = self.analyze_import_dependencies(module_path)
        class_analysis = self.analyze_class_initialization_overhead(module_path)
        init_analysis = self.analyze_init_py_impact()
        lazy_loading_analysis = self.identify_lazy_loading_opportunities(dependency_analysis)
        
        # 最適化効果予測
        predicted_savings = lazy_loading_analysis.get('total_optimizable_cost', 0)
        current_cost = dependency_analysis.get('total_import_cost', 2422)
        
        optimization_projection = {
            'current_import_time': current_cost,
            'predicted_savings': predicted_savings,
            'optimized_time': current_cost - predicted_savings,
            'improvement_percentage': (predicted_savings / current_cost * 100) if current_cost > 0 else 0,
            'meets_target': (current_cost - predicted_savings) <= 50  # 50ms目標
        }
        
        # 総合レポート
        comprehensive_report = {
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'analysis_target': module_path,
            'current_bottleneck': {
                'total_import_time_ms': current_cost,
                'target_time_ms': 50,
                'reduction_needed_ms': current_cost - 50,
                'reduction_needed_percentage': ((current_cost - 50) / current_cost * 100) if current_cost > 0 else 0
            },
            'dependency_analysis': dependency_analysis,
            'class_structure_analysis': class_analysis,
            'init_py_impact': init_analysis,
            'lazy_loading_opportunities': lazy_loading_analysis,
            'optimization_projection': optimization_projection,
            'stage2_implementation_plan': self._generate_stage2_plan(lazy_loading_analysis),
            'stage3_implementation_plan': self._generate_stage3_plan(class_analysis)
        }
        
        return comprehensive_report
    
    def _generate_stage2_plan(self, lazy_loading_analysis: Dict) -> Dict[str, Any]:
        """Stage 2実装計画生成"""
        optimization_plan = lazy_loading_analysis.get('optimization_plan', [])
        
        # 遅延インポート実装計画
        stage2_plan = {
            'high_priority_modules': [p for p in optimization_plan if p['lazy_loading_feasibility'] >= 0.7],
            'medium_priority_modules': [p for p in optimization_plan if 0.4 <= p['lazy_loading_feasibility'] < 0.7],
            'low_priority_modules': [p for p in optimization_plan if p['lazy_loading_feasibility'] < 0.4],
            'implementation_strategy': {
                'step1': 'visualization系モジュール（matplotlib等）の完全遅延化',
                'step2': 'I/O系モジュール（yfinance等）の条件付きインポート',
                'step3': 'data_analysis系モジュール（scipy等）の on-demand ロード',
                'step4': '数値計算系モジュールの最小限インポート最適化'
            }
        }
        
        return stage2_plan
    
    def _generate_stage3_plan(self, class_analysis: Dict) -> Dict[str, Any]:
        """Stage 3実装計画生成"""
        complexity = class_analysis.get('complexity_analysis', {})
        
        stage3_plan = {
            'class_optimization_targets': complexity.get('complex_classes', []),
            'initialization_optimization_targets': complexity.get('large_init_classes', []),
            'optimization_strategies': {
                'lazy_initialization': '重いクラスの遅延初期化実装',
                'lightweight_alternatives': '軽量版クラスの作成',
                'on_demand_methods': 'メソッドの on-demand バインド',
                'cache_optimization': 'データキャッシュの効率化'
            }
        }
        
        return stage3_plan

def main():
    """メイン実行関数"""
    print("[ROCKET] TODO-PERF-001 Stage 1: hierarchical_ranking_system ボトルネック詳細分析開始")
    print("=" * 80)
    
    analyzer = HierarchicalRankingBottleneckAnalyzer()
    
    try:
        # 総合分析実行
        report = analyzer.generate_comprehensive_analysis_report()
        
        # レポート保存
        report_path = Path("hierarchical_ranking_bottleneck_analysis.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("[CHART] Stage 1 分析結果サマリー")
        print("=" * 80)
        
        current_time = report['current_bottleneck']['total_import_time_ms']
        target_time = report['current_bottleneck']['target_time_ms']
        needed_reduction = report['current_bottleneck']['reduction_needed_ms']
        
        print(f"[TARGET] 現在のインポート時間: {current_time:.1f}ms")
        print(f"[TARGET] 目標時間: {target_time}ms")
        print(f"[TARGET] 必要削減量: {needed_reduction:.1f}ms ({report['current_bottleneck']['reduction_needed_percentage']:.1f}%)")
        
        optimization = report['optimization_projection']
        print(f"\n[UP] 最適化効果予測:")
        print(f"  予測削減量: {optimization['predicted_savings']:.1f}ms")
        print(f"  最適化後時間: {optimization['optimized_time']:.1f}ms")
        print(f"  改善率: {optimization['improvement_percentage']:.1f}%")
        print(f"  目標達成可能性: {'[OK] 可能' if optimization['meets_target'] else '[WARNING] 追加最適化必要'}")
        
        # Stage 2計画
        stage2_plan = report['stage2_implementation_plan']
        print(f"\n🔄 Stage 2 実装計画:")
        print(f"  高優先度モジュール: {len(stage2_plan['high_priority_modules'])}個")
        print(f"  中優先度モジュール: {len(stage2_plan['medium_priority_modules'])}個")
        
        for i, module in enumerate(stage2_plan['high_priority_modules'][:3]):
            print(f"    {i+1}. {module['module']}: {module['cost_ms']:.1f}ms削減可能")
        
        print(f"\n📄 詳細分析レポート: {report_path}")
        
        # Stage 1完了・Stage 2準備完了の確認
        print("\n" + "=" * 80)
        print("[OK] Stage 1: ボトルネック詳細分析・根本原因特定 完了")
        print("[ROCKET] Stage 2: 重いライブラリ遅延インポート実装 準備完了")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Stage 1 分析実行エラー: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
