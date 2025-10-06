#!/usr/bin/env python3
"""
TODO-PERF-001 Phase 1 Stage 1: 現状ボトルネック実測・統合計画策定

yfinance(866ms)とopenpyxl(212ms)の詳細分析と遅延インポート統合計画の策定
"""

import time
import sys
import os
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
import json

class Phase1BottleneckAnalyzer:
    """Phase 1用ボトルネック詳細分析"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'timestamp': self.timestamp,
            'yfinance_analysis': {},
            'openpyxl_analysis': {},
            'lazy_loader_analysis': {},
            'integration_plan': {},
            'risk_assessment': {}
        }
        
    def measure_import_time(self, module_name: str, description: str = "") -> float:
        """モジュールインポート時間の精密測定"""
        try:
            # クリーンな環境でのインポート測定
            result = subprocess.run([
                sys.executable, '-c',
                f'''
import time
start = time.perf_counter()
import {module_name}
end = time.perf_counter()
print(f"{{(end - start) * 1000:.2f}}")
                '''
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import_time = float(result.stdout.strip())
                print(f"  📊 {description or module_name}: {import_time:.1f}ms")
                return import_time
            else:
                print(f"  ❌ {description or module_name}: インポートエラー")
                return 0.0
                
        except Exception as e:
            print(f"  ❌ {description or module_name}: 測定エラー ({e})")
            return 0.0
    
    def analyze_yfinance_usage(self):
        """yfinance使用箇所の詳細分析"""
        print("🔍 yfinance使用箇所詳細分析中...")
        
        # yfinanceインポート時間測定
        yfinance_time = self.measure_import_time('yfinance', 'yfinance基本インポート')
        
        # yfinance使用ファイル検索
        yfinance_files = []
        project_root = Path('.')
        
        for py_file in project_root.rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8')
                if 'yfinance' in content or 'import yf' in content:
                    yfinance_files.append(str(py_file))
            except:
                continue
        
        # 主要使用箇所の分析
        key_usage_files = []
        for file_path in yfinance_files:
            if any(key in file_path for key in ['data_fetcher', 'dssms', 'src']):
                key_usage_files.append(file_path)
        
        self.results['yfinance_analysis'] = {
            'import_time_ms': yfinance_time,
            'usage_files_total': len(yfinance_files),
            'key_usage_files': key_usage_files[:10],  # 主要10ファイル
            'expected_reduction': yfinance_time * 0.75,  # 75%削減期待
            'lazy_import_priority': 'high' if yfinance_time > 500 else 'medium'
        }
        
        print(f"  📊 yfinance使用ファイル: {len(yfinance_files)}個")
        print(f"  📊 主要使用箇所: {len(key_usage_files)}個")
        print(f"  📊 期待削減効果: {yfinance_time * 0.75:.0f}ms")
    
    def analyze_openpyxl_usage(self):
        """openpyxl使用箇所の詳細分析"""
        print("🔍 openpyxl使用箇所詳細分析中...")
        
        # openpyxlインポート時間測定
        openpyxl_time = self.measure_import_time('openpyxl', 'openpyxl基本インポート')
        
        # openpyxl使用ファイル検索
        openpyxl_files = []
        project_root = Path('.')
        
        for py_file in project_root.rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8')
                if 'openpyxl' in content or 'from openpyxl' in content:
                    openpyxl_files.append(str(py_file))
            except:
                continue
        
        # Excel出力関連ファイルの特定
        excel_output_files = []
        for file_path in openpyxl_files:
            if any(key in file_path for key in ['simulation_handler', 'output', 'excel', 'export']):
                excel_output_files.append(file_path)
        
        self.results['openpyxl_analysis'] = {
            'import_time_ms': openpyxl_time,
            'usage_files_total': len(openpyxl_files),
            'excel_output_files': excel_output_files,
            'expected_reduction': openpyxl_time * 0.76,  # 76%削減期待
            'lazy_import_priority': 'high' if openpyxl_time > 100 else 'medium'
        }
        
        print(f"  📊 openpyxl使用ファイル: {len(openpyxl_files)}個")
        print(f"  📊 Excel出力ファイル: {len(excel_output_files)}個")
        print(f"  📊 期待削減効果: {openpyxl_time * 0.76:.0f}ms")
    
    def analyze_lazy_loader_remnants(self):
        """lazy_loader残存参照の完全分析"""
        print("🔍 lazy_loader残存参照完全分析中...")
        
        # lazy_loader関連ファイル検索
        lazy_loader_files = []
        lazy_loader_references = []
        project_root = Path('.')
        
        for py_file in project_root.rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8')
                if 'lazy_loader' in content or 'lazy_import' in content:
                    lazy_loader_files.append(str(py_file))
                    
                    # 具体的な参照箇所を特定
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if 'lazy_loader' in line or 'lazy_import' in line:
                            lazy_loader_references.append({
                                'file': str(py_file),
                                'line': i,
                                'content': line.strip()
                            })
            except:
                continue
        
        # lazy_import_manager.pyの存在確認
        lazy_manager_exists = Path('src/utils/lazy_import_manager.py').exists()
        
        self.results['lazy_loader_analysis'] = {
            'remnant_files': lazy_loader_files,
            'reference_count': len(lazy_loader_references),
            'specific_references': lazy_loader_references[:20],  # 最初の20個
            'lazy_manager_exists': lazy_manager_exists,
            'removal_priority': 'high' if len(lazy_loader_references) > 0 else 'low'
        }
        
        print(f"  📊 lazy_loader参照ファイル: {len(lazy_loader_files)}個")
        print(f"  📊 具体的参照箇所: {len(lazy_loader_references)}箇所")
        print(f"  💡 lazy_import_manager存在: {'✅' if lazy_manager_exists else '❌'}")
    
    def create_integration_plan(self):
        """遅延インポート統合計画の策定"""
        print("📋 遅延インポート統合計画策定中...")
        
        yf_time = self.results['yfinance_analysis']['import_time_ms']
        xl_time = self.results['openpyxl_analysis']['import_time_ms']
        
        integration_plan = {
            'phase1_targets': {
                'yfinance': {
                    'current_ms': yf_time,
                    'target_ms': 200,
                    'reduction_ms': yf_time - 200,
                    'strategy': 'conditional_import_in_data_fetcher',
                    'implementation_files': ['data_fetcher.py', 'src/dssms/dssms_data_manager.py']
                },
                'openpyxl': {
                    'current_ms': xl_time,
                    'target_ms': 50,
                    'reduction_ms': xl_time - 50,
                    'strategy': 'excel_output_lazy_loading',
                    'implementation_files': ['output/simulation_handler.py']
                }
            },
            'total_expected_reduction': (yf_time - 200) + (xl_time - 50),
            'implementation_sequence': [
                'Stage 2: yfinance遅延インポート統合実装',
                'Stage 3: openpyxl遅延インポート・lazy_loader除去',
                'Stage 4: 統合効果検証・実用性確認'
            ],
            'success_criteria': {
                'performance': 'total_reduction >= 800ms',
                'functionality': 'no_feature_degradation',
                'stability': 'lazy_loader_completely_removed'
            }
        }
        
        self.results['integration_plan'] = integration_plan
        
        print(f"  🎯 yfinance削減目標: {yf_time:.0f}ms → 200ms ({yf_time-200:.0f}ms削減)")
        print(f"  🎯 openpyxl削減目標: {xl_time:.0f}ms → 50ms ({xl_time-50:.0f}ms削減)")
        print(f"  🎯 総削減期待値: {integration_plan['total_expected_reduction']:.0f}ms")
    
    def assess_implementation_risks(self):
        """実装リスクの評価"""
        print("⚠️ 実装リスク評価中...")
        
        risks = {
            'high_risk': [],
            'medium_risk': [],
            'low_risk': [],
            'mitigation_strategies': {}
        }
        
        # yfinanceリスク評価
        yf_files = len(self.results['yfinance_analysis']['key_usage_files'])
        if yf_files > 5:
            risks['high_risk'].append('yfinance多数ファイル使用によるインポート順序依存')
            risks['mitigation_strategies']['yfinance_dependency'] = 'importlib.util使用・エラーハンドリング強化'
        
        # openpyxlリスク評価  
        xl_files = len(self.results['openpyxl_analysis']['excel_output_files'])
        if xl_files > 3:
            risks['medium_risk'].append('openpyxl複数出力ファイルでの初回遅延')
            risks['mitigation_strategies']['openpyxl_delay'] = '初回Excel出力時の明示的待機メッセージ'
        
        # lazy_loaderリスク評価
        lazy_refs = self.results['lazy_loader_analysis']['reference_count']
        if lazy_refs > 10:
            risks['high_risk'].append('lazy_loader大量参照による除去作業複雑化')
            risks['mitigation_strategies']['lazy_removal'] = '段階的除去・テスト重点実施'
        elif lazy_refs > 0:
            risks['medium_risk'].append('lazy_loader残存参照による不安定化')
            risks['mitigation_strategies']['lazy_cleanup'] = '完全除去・直接インポート化'
        
        self.results['risk_assessment'] = risks
        
        print(f"  🔴 高リスク: {len(risks['high_risk'])}項目")
        print(f"  🟡 中リスク: {len(risks['medium_risk'])}項目")
        print(f"  🟢 低リスク: {len(risks['low_risk'])}項目")
    
    def generate_stage1_report(self):
        """Stage 1完了レポート生成"""
        report_file = f"phase1_stage1_analysis_{self.timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # サマリー表示
        print("\n" + "="*60)
        print("📊 Stage 1: 現状ボトルネック実測・統合計画策定 完了")
        print("="*60)
        
        yf_reduction = self.results['integration_plan']['phase1_targets']['yfinance']['reduction_ms']
        xl_reduction = self.results['integration_plan']['phase1_targets']['openpyxl']['reduction_ms']
        total_reduction = self.results['integration_plan']['total_expected_reduction']
        
        print(f"🎯 Phase 1期待効果:")
        print(f"  • yfinance最適化: {yf_reduction:.0f}ms削減")
        print(f"  • openpyxl最適化: {xl_reduction:.0f}ms削減")
        print(f"  • 総削減効果: {total_reduction:.0f}ms")
        print(f"  • 実用性評価: {'✅ 目標800ms達成可能' if total_reduction >= 800 else '⚠️ 目標未達のリスク'}")
        
        print(f"\n📄 詳細分析結果: {report_file}")
        print("🚀 Stage 2: yfinance遅延インポート統合実装 準備完了")
        
        return report_file

def main():
    """Stage 1メイン実行"""
    print("🚀 TODO-PERF-001 Phase 1 Stage 1: 現状ボトルネック実測開始")
    print("="*80)
    
    analyzer = Phase1BottleneckAnalyzer()
    
    # 各種分析の実行
    analyzer.analyze_yfinance_usage()
    analyzer.analyze_openpyxl_usage()
    analyzer.analyze_lazy_loader_remnants()
    analyzer.create_integration_plan()
    analyzer.assess_implementation_risks()
    
    # 完了レポート生成
    report_file = analyzer.generate_stage1_report()
    
    return analyzer.results

if __name__ == "__main__":
    results = main()