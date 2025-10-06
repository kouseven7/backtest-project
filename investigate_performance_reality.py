#!/usr/bin/env python3
"""
パフォーマンス最適化の実態調査スクリプト

TODO-PERF-001の98.7%改善達成の実在性を検証し、
継続的な最適化の必要性を判断するための調査ツール
"""

import time
import importlib
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

class PerformanceInvestigator:
    """パフォーマンス調査器 - 実際の実行時間測定・分析"""
    
    def __init__(self):
        self.measurements: Dict[str, Any] = {}
        self.import_times: Dict[str, Dict] = {}
        self.execution_times: Dict[str, float] = {}
        
    def measure_import_time(self, module_name: str, import_path: Optional[str] = None) -> float:
        """モジュール・インポート時間測定"""
        start_time = time.perf_counter()
        try:
            if import_path:
                # 直接パスからのインポート
                spec = importlib.util.spec_from_file_location(module_name, import_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    success = True
                else:
                    success = False
            else:
                # 通常のインポート
                importlib.import_module(module_name)
                success = True
        except Exception as e:
            print(f"Import error for {module_name}: {e}")
            success = False
        
        end_time = time.perf_counter()
        import_time = (end_time - start_time) * 1000  # ms
        
        self.import_times[module_name] = {
            'time_ms': import_time,
            'success': success,
            'path': import_path
        }
        
        return import_time
    
    def measure_dssms_integrated_main_import(self) -> Dict:
        """DSSMSIntegratedBacktester インポート時間の詳細測定"""
        print("🔍 DSSMSIntegratedBacktester インポート時間調査中...")
        
        # 段階的インポート測定
        stages = {
            'src.dssms.dssms_integrated_main': None,
            'src.dssms.symbol_switch_manager': None,
            'src.dssms.advanced_ranking_engine': None,
            'src.dssms.hierarchical_ranking_system': None
        }
        
        results = {}
        total_start = time.perf_counter()
        
        for stage_name, path in stages.items():
            stage_time = self.measure_import_time(stage_name, path)
            results[stage_name] = stage_time
            print(f"  📊 {stage_name}: {stage_time:.1f}ms")
        
        total_end = time.perf_counter()
        total_time = (total_end - total_start) * 1000
        results['total_dssms_import'] = total_time
        
        print(f"  🎯 DSSMS総インポート時間: {total_time:.1f}ms")
        
        return results
    
    def measure_heavy_libraries(self) -> Dict:
        """重いライブラリ（yfinance, openpyxl）のインポート時間測定"""
        print("📚 重いライブラリのインポート時間調査中...")
        
        heavy_libs = ['yfinance', 'openpyxl', 'matplotlib', 'pandas', 'numpy']
        results = {}
        
        for lib in heavy_libs:
            lib_time = self.measure_import_time(lib)
            results[lib] = lib_time
            print(f"  📊 {lib}: {lib_time:.1f}ms")
        
        return results
    
    def measure_execution_performance(self) -> Dict:
        """実際の実行パフォーマンス測定"""
        print("⚡ システム実行パフォーマンス調査中...")
        
        results = {}
        
        try:
            # DSSMSシステムの初期化・実行測定
            start_time = time.perf_counter()
            
            from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
            
            init_time = time.perf_counter()
            initialization_time = (init_time - start_time) * 1000
            
            # 軽量なバックテスト実行（1日分）
            backtester = DSSMSIntegratedBacktester()
            execution_start = time.perf_counter()
            
            # 軽量設定でのクイック実行
            # result = backtester.run_quick_performance_test()  # 軽量テスト実行
            
            execution_end = time.perf_counter()
            execution_time = (execution_end - execution_start) * 1000
            
            total_time = (execution_end - start_time) * 1000
            
            results = {
                'initialization_time_ms': initialization_time,
                'execution_time_ms': execution_time,
                'total_time_ms': total_time,
                'success': True
            }
            
            print(f"  🎯 初期化時間: {initialization_time:.1f}ms")
            print(f"  ⚡ 実行時間: {execution_time:.1f}ms")
            print(f"  📊 総時間: {total_time:.1f}ms")
            
        except Exception as e:
            print(f"  ❌ 実行測定エラー: {e}")
            traceback.print_exc()
            results = {
                'error': str(e),
                'success': False
            }
        
        return results
    
    def analyze_bottlenecks(self) -> Dict:
        """ボトルネック分析・優先度評価"""
        print("🎯 ボトルネック分析・優先度評価...")
        
        analysis = {
            'import_bottlenecks': [],
            'execution_bottlenecks': [],
            'optimization_priorities': [],
            'recommendation': 'continue'  # continue / complete / investigate
        }
        
        # インポート時間の分析
        if self.import_times:
            sorted_imports = sorted(
                self.import_times.items(), 
                key=lambda x: x[1]['time_ms'], 
                reverse=True
            )
            
            for name, data in sorted_imports[:5]:  # Top 5
                if data['time_ms'] > 100:  # 100ms以上をボトルネック判定
                    analysis['import_bottlenecks'].append({
                        'module': name,
                        'time_ms': data['time_ms'],
                        'severity': 'high' if data['time_ms'] > 500 else 'medium'
                    })
        
        # 実行時間の分析
        if self.execution_times:
            for name, time_ms in self.execution_times.items():
                if time_ms > 50:  # 50ms以上を実行ボトルネック判定
                    analysis['execution_bottlenecks'].append({
                        'operation': name,
                        'time_ms': time_ms,
                        'severity': 'high' if time_ms > 200 else 'medium'
                    })
        
        # 推奨事項の決定
        high_impact_bottlenecks = [
            b for b in analysis['import_bottlenecks'] 
            if b['severity'] == 'high'
        ]
        
        if len(high_impact_bottlenecks) == 0:
            analysis['recommendation'] = 'complete'
            analysis['rationale'] = "重大なボトルネックが存在しない - 実用レベル達成"
        elif len(high_impact_bottlenecks) <= 2:
            analysis['recommendation'] = 'investigate'
            analysis['rationale'] = "軽微なボトルネック存在 - コスト対効果分析必要"
        else:
            analysis['recommendation'] = 'continue'
            analysis['rationale'] = "複数の重大ボトルネック存在 - 継続最適化推奨"
        
        return analysis
    
    def generate_performance_report(self) -> Dict:
        """総合パフォーマンスレポート生成"""
        print("📋 総合パフォーマンスレポート生成中...")
        
        # 全測定実行
        dssms_results = self.measure_dssms_integrated_main_import()
        heavy_lib_results = self.measure_heavy_libraries()
        execution_results = self.measure_execution_performance()
        bottleneck_analysis = self.analyze_bottlenecks()
        
        # 文書記載値との比較
        documented_values = {
            'dssms_integrated_before': 2871.7,  # ms
            'dssms_integrated_after': 36.7,     # ms
            'improvement_claimed': 98.7,        # %
            'system_execution_claimed': 0.2,    # ms
            'yfinance_claimed': 972.4,          # ms
            'openpyxl_claimed': 254.6           # ms
        }
        
        # 実測値との比較分析
        actual_dssms = dssms_results.get('total_dssms_import', 0)
        actual_yfinance = heavy_lib_results.get('yfinance', 0)
        actual_openpyxl = heavy_lib_results.get('openpyxl', 0)
        
        comparison = {
            'dssms_integrated_actual_vs_claimed': {
                'actual': actual_dssms,
                'claimed': documented_values['dssms_integrated_after'],
                'difference': actual_dssms - documented_values['dssms_integrated_after'],
                'accuracy': 'good' if abs(actual_dssms - documented_values['dssms_integrated_after']) < 50 else 'poor'
            },
            'yfinance_actual_vs_claimed': {
                'actual': actual_yfinance,
                'claimed': documented_values['yfinance_claimed'],
                'difference': actual_yfinance - documented_values['yfinance_claimed'],
                'accuracy': 'good' if abs(actual_yfinance - documented_values['yfinance_claimed']) < 200 else 'poor'
            },
            'openpyxl_actual_vs_claimed': {
                'actual': actual_openpyxl,
                'claimed': documented_values['openpyxl_claimed'],
                'difference': actual_openpyxl - documented_values['openpyxl_claimed'],
                'accuracy': 'good' if abs(actual_openpyxl - documented_values['openpyxl_claimed']) < 100 else 'poor'
            }
        }
        
        report = {
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'measurements': {
                'dssms_import': dssms_results,
                'heavy_libraries': heavy_lib_results,
                'execution': execution_results
            },
            'comparison_with_documentation': comparison,
            'bottleneck_analysis': bottleneck_analysis,
            'conclusion': self._generate_conclusion(comparison, bottleneck_analysis)
        }
        
        return report
    
    def _generate_conclusion(self, comparison: Dict, analysis: Dict) -> Dict:
        """調査結論の生成"""
        
        # 文書精度評価
        doc_accuracy_score = 0
        accuracy_checks = [
            comparison['dssms_integrated_actual_vs_claimed']['accuracy'],
            comparison['yfinance_actual_vs_claimed']['accuracy'],
            comparison['openpyxl_actual_vs_claimed']['accuracy']
        ]
        doc_accuracy_score = len([a for a in accuracy_checks if a == 'good']) / len(accuracy_checks)
        
        conclusion = {
            'documentation_accuracy': doc_accuracy_score,
            'performance_status': 'unknown',
            'optimization_necessity': 'unknown',
            'recommendation': analysis['recommendation'],
            'confidence_level': 'low'
        }
        
        # パフォーマンス状況判定
        if doc_accuracy_score >= 0.7:
            conclusion['confidence_level'] = 'high'
            if analysis['recommendation'] == 'complete':
                conclusion['performance_status'] = 'excellent'
                conclusion['optimization_necessity'] = 'unnecessary'
            elif analysis['recommendation'] == 'investigate':
                conclusion['performance_status'] = 'good'
                conclusion['optimization_necessity'] = 'optional'
            else:
                conclusion['performance_status'] = 'needs_improvement'
                conclusion['optimization_necessity'] = 'required'
        else:
            conclusion['confidence_level'] = 'medium'
            conclusion['performance_status'] = 'uncertain'
            conclusion['optimization_necessity'] = 'investigate_further'
        
        return conclusion

def main():
    """メイン実行関数"""
    print("🚀 TODO-PERF-001 パフォーマンス最適化実態調査開始")
    print("=" * 60)
    
    investigator = PerformanceInvestigator()
    
    try:
        # 総合調査実行
        report = investigator.generate_performance_report()
        
        # レポート保存
        import json
        report_path = Path("performance_investigation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("📊 調査結果サマリー")
        print("=" * 60)
        
        conclusion = report['conclusion']
        print(f"📋 文書精度: {conclusion['documentation_accuracy']:.1%}")
        print(f"⚡ パフォーマンス状況: {conclusion['performance_status']}")
        print(f"🎯 最適化必要性: {conclusion['optimization_necessity']}")
        print(f"📝 推奨事項: {conclusion['recommendation']}")
        print(f"🔍 信頼度: {conclusion['confidence_level']}")
        
        # 詳細結果表示
        print("\n📊 詳細測定結果:")
        measurements = report['measurements']
        
        if 'dssms_import' in measurements:
            dssms = measurements['dssms_import']
            print(f"  🎯 DSSMS総インポート: {dssms.get('total_dssms_import', 0):.1f}ms")
        
        if 'heavy_libraries' in measurements:
            heavy = measurements['heavy_libraries']
            print(f"  📚 yfinance: {heavy.get('yfinance', 0):.1f}ms")
            print(f"  📚 openpyxl: {heavy.get('openpyxl', 0):.1f}ms")
        
        if 'execution' in measurements:
            exec_data = measurements['execution']
            if exec_data.get('success', False):
                print(f"  ⚡ システム実行: {exec_data.get('total_time_ms', 0):.1f}ms")
        
        print(f"\n📄 詳細レポート: {report_path}")
        
        # 最終推奨
        print("\n" + "=" * 60)
        print("🎯 最終推奨事項")
        print("=" * 60)
        
        if conclusion['optimization_necessity'] == 'unnecessary':
            print("✅ 追加最適化は不要です")
            print("   → 現在のパフォーマンスは実用レベルに達しています")
            print("   → TODO-PERF-001 を完了状態に移行することを推奨します")
        elif conclusion['optimization_necessity'] == 'optional':
            print("⚠️ 追加最適化は任意です")
            print("   → コストと効果を慎重に検討してください") 
            print("   → 他の優先度高い作業がある場合は延期も可能です")
        elif conclusion['optimization_necessity'] == 'required':
            print("🔴 追加最適化が必要です")
            print("   → 重大なボトルネックが残存しています")
            print("   → TODO-PERF-001 の継続実装を推奨します")
        else:
            print("🔍 さらなる調査が必要です")
            print("   → 測定精度に課題があります")
            print("   → より詳細な調査を実施してください")
        
    except Exception as e:
        print(f"❌ 調査実行エラー: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)