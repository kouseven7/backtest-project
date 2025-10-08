#!/usr/bin/env python3
"""
TODO-PERF-001: Phase 1即効性対策実装
Stage 1: 現状ボトルネック実測・統合計画策定

重いライブラリ（yfinance, openpyxl）の使用箇所特定と
lazy_loader残存参照の完全分析を行い、遅延インポート統合戦略を策定する。
"""

import os
import sys
import time
import importlib.util
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re
import json

class Phase1BottleneckAnalyzer:
    """Phase 1ボトルネック詳細分析・統合計画策定"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analysis_results = {}
        self.heavy_libraries = ['yfinance', 'openpyxl']
        self.lazy_loader_patterns = [
            r'@lazy_import',
            r'@lazy_class_import', 
            r'lazy_loader\.',
            r'from.*lazy_loader',
            r'import.*lazy_loader'
        ]
    
    def measure_import_times_detailed(self) -> Dict[str, float]:
        """重いライブラリの詳細インポート時間測定"""
        print("📊 重いライブラリ詳細インポート時間測定開始...")
        import_times = {}
        
        for library in self.heavy_libraries:
            print(f"  📈 {library} インポート時間測定中...")
            
            # 複数回測定で精度向上
            times = []
            for i in range(3):
                start_time = time.perf_counter()
                try:
                    # 新しいPythonプロセスで純粋なインポート時間測定
                    result = subprocess.run([
                        sys.executable, '-c', 
                        f'import time; start=time.perf_counter(); import {library}; print(f"{{(time.perf_counter()-start)*1000:.1f}}")'
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        import_time = float(result.stdout.strip())
                        times.append(import_time)
                        print(f"    試行{i+1}: {import_time:.1f}ms")
                    else:
                        print(f"    試行{i+1}: エラー - {result.stderr.strip()}")
                        
                except subprocess.TimeoutExpired:
                    print(f"    試行{i+1}: タイムアウト (30秒)")
                except Exception as e:
                    print(f"    試行{i+1}: 例外 - {str(e)}")
            
            if times:
                avg_time = sum(times) / len(times)
                import_times[library] = {
                    'average_ms': avg_time,
                    'measurements': times,
                    'min_ms': min(times),
                    'max_ms': max(times)
                }
                print(f"  ✅ {library}: 平均 {avg_time:.1f}ms (範囲: {min(times):.1f}-{max(times):.1f}ms)")
            else:
                import_times[library] = {
                    'average_ms': 0,
                    'measurements': [],
                    'error': 'All measurements failed'
                }
                print(f"  ❌ {library}: 測定失敗")
        
        return import_times
    
    def find_heavy_library_usages(self) -> Dict[str, List[Dict]]:
        """重いライブラリの使用箇所特定"""
        print("🔍 重いライブラリ使用箇所特定中...")
        usages = {lib: [] for lib in self.heavy_libraries}
        
        # Pythonファイルを再帰的に検索
        python_files = list(self.project_root.rglob('*.py'))
        print(f"  📁 対象ファイル数: {len(python_files)}")
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for lib in self.heavy_libraries:
                    # インポート文の検索
                    import_patterns = [
                        rf'^import\s+{lib}',
                        rf'^from\s+{lib}',
                        rf'import.*{lib}',
                        rf'{lib}\.',  # 使用箇所
                    ]
                    
                    for line_num, line in enumerate(lines, 1):
                        for pattern in import_patterns:
                            if re.search(pattern, line.strip()):
                                usages[lib].append({
                                    'file': str(file_path.relative_to(self.project_root)),
                                    'line': line_num,
                                    'content': line.strip(),
                                    'type': 'import' if 'import' in line else 'usage'
                                })
                                
            except Exception as e:
                print(f"  ⚠️ ファイル読み取りエラー: {file_path} - {str(e)}")
        
        # 結果サマリー
        for lib, usage_list in usages.items():
            import_count = len([u for u in usage_list if u['type'] == 'import'])
            usage_count = len([u for u in usage_list if u['type'] == 'usage'])
            print(f"  📋 {lib}: インポート {import_count}箇所, 使用 {usage_count}箇所")
        
        return usages
    
    def find_lazy_loader_remnants(self) -> List[Dict]:
        """lazy_loader残存参照の完全特定"""
        print("🔍 lazy_loader残存参照完全特定中...")
        remnants = []
        
        # Pythonファイルを再帰的に検索
        python_files = list(self.project_root.rglob('*.py'))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.lazy_loader_patterns:
                        if re.search(pattern, line):
                            remnants.append({
                                'file': str(file_path.relative_to(self.project_root)),
                                'line': line_num,
                                'content': line.strip(),
                                'pattern': pattern
                            })
                            
            except Exception as e:
                print(f"  ⚠️ ファイル読み取りエラー: {file_path} - {str(e)}")
        
        print(f"  📋 lazy_loader残存参照: {len(remnants)}箇所発見")
        for remnant in remnants:
            print(f"    - {remnant['file']}:{remnant['line']} | {remnant['content']}")
        
        return remnants
    
    def analyze_lazy_loading_strategy(self, import_times: Dict, usages: Dict) -> Dict:
        """遅延インポート戦略分析"""
        print("📈 遅延インポート戦略分析中...")
        strategy = {}
        
        for lib, time_data in import_times.items():
            if 'average_ms' not in time_data:
                continue
                
            avg_time = time_data['average_ms']
            lib_usages = usages.get(lib, [])
            
            # インポート箇所とメイン使用箇所の分析
            import_files = set(u['file'] for u in lib_usages if u['type'] == 'import')
            usage_files = set(u['file'] for u in lib_usages if u['type'] == 'usage')
            
            # 遅延化の効果予測
            if avg_time > 100:  # 100ms以上なら遅延化候補
                expected_reduction = avg_time * 0.8  # 80%削減を期待
                strategy[lib] = {
                    'current_time_ms': avg_time,
                    'expected_reduction_ms': expected_reduction,
                    'optimization_priority': 'HIGH' if avg_time > 500 else 'MEDIUM',
                    'import_files': list(import_files),
                    'usage_files': list(usage_files),
                    'lazy_loading_approach': self._determine_lazy_approach(lib, lib_usages),
                    'risks': self._assess_lazy_loading_risks(lib, lib_usages),
                    'implementation_complexity': self._assess_implementation_complexity(lib, lib_usages)
                }
        
        return strategy
    
    def _determine_lazy_approach(self, lib: str, usages: List[Dict]) -> str:
        """ライブラリごとの最適な遅延アプローチ決定"""
        if lib == 'yfinance':
            return 'conditional_import_in_data_fetcher'
        elif lib == 'openpyxl':
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: return 'conditional_import_in_excel_output'
        else:
            return 'general_lazy_import'
    
    def _assess_lazy_loading_risks(self, lib: str, usages: List[Dict]) -> List[str]:
        """遅延読み込みのリスク評価"""
        risks = []
        
        if lib == 'yfinance':
            risks.extend([
                'network_timeout_during_lazy_load',
                'market_data_cache_invalidation',
                'real_time_data_delay_impact'
            ])
        elif lib == 'openpyxl':
            risks.extend([
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: 'excel_output_first_time_delay',
                'memory_allocation_spike',
                'file_format_compatibility_check_delay'
            ])
        
        # 共通リスク
        import_count = len([u for u in usages if u['type'] == 'import'])
        if import_count > 3:
            risks.append('multiple_import_locations_complexity')
        
        return risks
    
    def _assess_implementation_complexity(self, lib: str, usages: List[Dict]) -> str:
        """実装複雑度評価"""
        import_files = set(u['file'] for u in usages if u['type'] == 'import')
        usage_files = set(u['file'] for u in usages if u['type'] == 'usage')
        
        if len(import_files) <= 2 and len(usage_files) <= 5:
            return 'LOW'
        elif len(import_files) <= 5 and len(usage_files) <= 15:
            return 'MEDIUM'  
        else:
            return 'HIGH'
    
    def generate_implementation_plan(self, strategy: Dict, remnants: List[Dict]) -> Dict:
        """実装計画生成"""
        print("📋 Phase 1実装計画生成中...")
        
        plan = {
            'phase1_overview': {
                'target_libraries': list(strategy.keys()),
                'expected_total_reduction_ms': sum(s['expected_reduction_ms'] for s in strategy.values()),
                'lazy_loader_remnants_count': len(remnants),
                'implementation_stages': 4
            },
            'stage_details': {},
            'risk_mitigation': {},
            'success_criteria': {}
        }
        
        # Stage 2: yfinance実装詳細
        if 'yfinance' in strategy:
            yf_strategy = strategy['yfinance']
            plan['stage_details']['stage2_yfinance'] = {
                'target_files': yf_strategy['import_files'],
                'approach': yf_strategy['lazy_loading_approach'],
                'expected_reduction': yf_strategy['expected_reduction_ms'],
                'implementation_steps': [
                    'create_conditional_import_wrapper',
                    'modify_data_fetcher_imports',
                    'integrate_with_dssms_data_manager',
                    'add_systemfallbackpolicy_handling',
                    'measure_performance_improvement'
                ]
            }
        
        # Stage 3: openpyxl + lazy_loader除去実装詳細
        if 'openpyxl' in strategy:
            openpyxl_strategy = strategy['openpyxl']
            plan['stage_details']['stage3_openpyxl_cleanup'] = {
                'target_files': openpyxl_strategy['import_files'],
                'approach': openpyxl_strategy['lazy_loading_approach'],
                'expected_reduction': openpyxl_strategy['expected_reduction_ms'],
                'lazy_loader_cleanup': {
                    'remnant_files': [r['file'] for r in remnants],
                    'cleanup_approach': 'direct_import_replacement'
                },
                'implementation_steps': [
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: 'create_excel_output_lazy_wrapper',
                    'modify_simulation_handler_imports',
                    'remove_all_lazy_loader_remnants',
                    'convert_to_direct_imports',
                    'verify_functionality_maintained'
                ]
            }
        
        # リスク軽減策
        plan['risk_mitigation'] = {
            'functionality_preservation': [
                'comprehensive_regression_testing',
                'function_signature_compatibility_check',
                'data_integrity_validation'
            ],
            'performance_monitoring': [
                'before_after_measurement',
                'memory_usage_tracking',
                'first_access_delay_measurement'
            ],
            'error_handling': [
                'systemfallbackpolicy_integration',
                'graceful_degradation_on_import_failure',
                'detailed_error_logging'
            ]
        }
        
        # 成功判定基準
        total_expected = sum(s['expected_reduction_ms'] for s in strategy.values())
        plan['success_criteria'] = {
            'quantitative': {
                'minimum_reduction_ms': max(800, total_expected * 0.8),
                'yfinance_target_ms': 200,
                'openpyxl_target_ms': 50,
                'lazy_loader_remnants': 0
            },
            'qualitative': {
                'functionality_preservation': '100%',
                'regression_test_pass_rate': '100%',
                'system_stability': 'maintained_or_improved'
            }
        }
        
        return plan
    
    def run_complete_analysis(self) -> Dict:
        """Phase 1完全分析実行"""
        print("🚀 TODO-PERF-001 Phase 1: 現状ボトルネック完全分析開始")
        print("=" * 80)
        
        start_time = time.time()
        
        # Step 1: 重いライブラリインポート時間詳細測定
        import_times = self.measure_import_times_detailed()
        
        # Step 2: 使用箇所特定
        usages = self.find_heavy_library_usages()
        
        # Step 3: lazy_loader残存参照特定
        remnants = self.find_lazy_loader_remnants()
        
        # Step 4: 遅延インポート戦略分析
        strategy = self.analyze_lazy_loading_strategy(import_times, usages)
        
        # Step 5: 実装計画生成
        implementation_plan = self.generate_implementation_plan(strategy, remnants)
        
        # 結果統合
        analysis_results = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_duration_seconds': time.time() - start_time,
            'import_times': import_times,
            'library_usages': usages,
            'lazy_loader_remnants': remnants,
            'optimization_strategy': strategy,
            'implementation_plan': implementation_plan
        }
        
        self.analysis_results = analysis_results
        return analysis_results
    
    def save_analysis_results(self) -> str:
        """分析結果保存"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = self.project_root / f'phase1_bottleneck_analysis_{timestamp}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 分析結果保存完了: {output_file}")
        return str(output_file)
    
    def print_executive_summary(self):
        """重要発見事項サマリー出力"""
        if not self.analysis_results:
            return
        
        print("\n" + "="*80)
        print("📊 TODO-PERF-001 Phase 1: 重要発見事項サマリー")
        print("="*80)
        
        # インポート時間結果
        import_times = self.analysis_results['import_times']
        total_reduction = 0
        
        print("🎯 重いライブラリ最適化ポテンシャル:")
        for lib, data in import_times.items():
            if 'average_ms' in data:
                avg_time = data['average_ms']
                expected_reduction = avg_time * 0.8
                total_reduction += expected_reduction
                print(f"  • {lib}: {avg_time:.1f}ms → ~{avg_time*0.2:.1f}ms (削減: {expected_reduction:.1f}ms)")
        
        print(f"\n🏆 Phase 1期待削減効果: {total_reduction:.1f}ms")
        
        # lazy_loader残存状況
        remnants = self.analysis_results['lazy_loader_remnants']
        print(f"🧹 lazy_loader残存参照: {len(remnants)}箇所")
        
        # 実装計画概要
        plan = self.analysis_results['implementation_plan']
        expected_total = plan['phase1_overview']['expected_total_reduction_ms']
        print(f"📋 実装計画期待効果: {expected_total:.1f}ms削減")
        
        # 成功判定基準
        criteria = plan['success_criteria']['quantitative']
        print(f"✅ 合格判定基準: {criteria['minimum_reduction_ms']:.1f}ms以上削減")
        
        print(f"\n⏱️ 分析時間: {self.analysis_results['analysis_duration_seconds']:.1f}秒")
        print("="*80)

def main():
    """メイン実行"""
    project_root = os.getcwd()
    analyzer = Phase1BottleneckAnalyzer(project_root)
    
    try:
        # Phase 1完全分析実行
        results = analyzer.run_complete_analysis()
        
        # 結果保存
        output_file = analyzer.save_analysis_results()
        
        # 重要発見事項サマリー
        analyzer.print_executive_summary()
        
        print(f"\n🎉 Phase 1 Stage 1完了 - 次は Stage 2 yfinance遅延インポート実装に進行")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 Stage 1分析エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)