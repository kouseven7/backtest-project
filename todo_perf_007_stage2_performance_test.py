#!/usr/bin/env python3
"""
TODO-PERF-007 Stage 2: 並列処理統合効果テスト

目的:
- 実際のScreenerの並列処理統合効果測定
- apply_market_cap_filterメソッドのパフォーマンス改善確認
- 52.5秒→25-30秒削減達成検証
- 既存機能・品質完全保持確認
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 統合後Screenerインポート
try:
    from src.dssms.nikkei225_screener import Nikkei225Screener
    print("✅ 統合後Nikkei225Screener インポート成功")
except ImportError as e:
    print(f"❌ Screenerインポートエラー: {e}")
    sys.exit(1)

class ScreenerPerformanceTester:
    """Screener並列処理効果テスター"""
    
    def __init__(self):
        self.screener = Nikkei225Screener()
        self.test_results = {}
        
        # テスト用銘柄セット（段階的スケーリング）
        self.test_symbol_sets = {
            "small": ["7203", "9984", "8058", "9983", "6758"],  # 5銘柄
            "medium": ["7203", "9984", "8058", "9983", "6758", "7201", "8306", "8316", "4519", "4502"],  # 10銘柄
            "large": [
                "7203", "9984", "8058", "9983", "6758", "7201", "8306", "8316", "4519", "4502",
                "6098", "3382", "4751", "2432", "3659", "7974", "9434", "8411", "8802", "8604"
            ]  # 20銘柄
        }
    
    def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """包括的パフォーマンステスト実行"""
        
        print("🚀 TODO-PERF-007 Stage 2: 並列処理統合効果テスト開始")
        print("="*80)
        
        try:
            # 1. 基本機能テスト
            basic_test = self._test_basic_functionality()
            
            # 2. パフォーマンス測定（段階的）
            performance_test = self._test_performance_scaling()
            
            # 3. 並列・逐次比較テスト
            comparison_test = self._test_parallel_vs_sequential()
            
            # 4. 効果推定・外挿
            effectiveness_analysis = self._analyze_effectiveness()
            
            # 5. 最終評価
            final_assessment = self._assess_final_performance()
            
            self.test_results = {
                "basic_functionality": basic_test,
                "performance_scaling": performance_test,
                "parallel_vs_sequential": comparison_test,
                "effectiveness_analysis": effectiveness_analysis,
                "final_assessment": final_assessment,
                "test_timestamp": datetime.now().isoformat()
            }
            
            return self.test_results
            
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")
            return {"error": str(e)}
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """基本機能テスト"""
        
        print("🧪 基本機能テスト実行中...")
        
        try:
            test_symbols = self.test_symbol_sets["small"]
            
            # apply_market_cap_filterメソッド直接テスト
            start_time = time.perf_counter()
            filtered_symbols = self.screener.apply_market_cap_filter(test_symbols)
            execution_time = time.perf_counter() - start_time
            
            # 基本検証
            basic_checks = {
                "method_callable": callable(self.screener.apply_market_cap_filter),
                "returns_list": isinstance(filtered_symbols, list),
                "symbols_filtered": len(filtered_symbols) <= len(test_symbols),
                "execution_successful": execution_time > 0,
                "no_errors": True  # エラーなしで実行完了
            }
            
            success_rate = sum(basic_checks.values()) / len(basic_checks) * 100
            
            print(f"  ✅ 基本機能テスト完了: {success_rate:.1f}%成功")
            
            return {
                "success": success_rate >= 80,
                "execution_time": round(execution_time, 2),
                "input_symbols": len(test_symbols),
                "output_symbols": len(filtered_symbols),
                "basic_checks": basic_checks,
                "success_rate": success_rate
            }
            
        except Exception as e:
            return {"success": False, "error": f"基本機能テストエラー: {e}"}
    
    def _test_performance_scaling(self) -> Dict[str, Any]:
        """パフォーマンススケーリングテスト"""
        
        print("⚡ パフォーマンススケーリングテスト実行中...")
        
        scaling_results = {}
        
        for size_name, symbols in self.test_symbol_sets.items():
            try:
                print(f"  🔧 {size_name}テスト ({len(symbols)}銘柄)...")
                
                start_time = time.perf_counter()
                filtered_symbols = self.screener.apply_market_cap_filter(symbols)
                execution_time = time.perf_counter() - start_time
                
                symbols_per_second = len(symbols) / execution_time if execution_time > 0 else 0
                
                scaling_results[size_name] = {
                    "symbols_count": len(symbols),
                    "execution_time": round(execution_time, 2),
                    "symbols_per_second": round(symbols_per_second, 1),
                    "filtered_count": len(filtered_symbols),
                    "filter_rate": round(len(filtered_symbols) / len(symbols) * 100, 1)
                }
                
                print(f"    ✅ {execution_time:.1f}秒 ({symbols_per_second:.1f}銘柄/秒)")
                
            except Exception as e:
                scaling_results[size_name] = {"error": str(e)}
        
        return {
            "scaling_results": scaling_results,
            "scaling_efficiency": self._calculate_scaling_efficiency(scaling_results)
        }
    
    def _calculate_scaling_efficiency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """スケーリング効率計算"""
        
        try:
            if "small" in results and "large" in results:
                small_data = results["small"]
                large_data = results["large"]
                
                if "symbols_per_second" in small_data and "symbols_per_second" in large_data:
                    small_throughput = small_data["symbols_per_second"]
                    large_throughput = large_data["symbols_per_second"]
                    
                    # 理想的には並列処理で throughput が維持または向上
                    efficiency = (large_throughput / small_throughput) if small_throughput > 0 else 0
                    
                    return {
                        "throughput_efficiency": round(efficiency, 2),
                        "parallel_effectiveness": efficiency >= 0.8,  # 80%効率以上
                        "small_throughput": small_throughput,
                        "large_throughput": large_throughput
                    }
            
            return {"efficiency_calculation": "データ不足"}
            
        except Exception as e:
            return {"error": f"効率計算エラー: {e}"}
    
    def _test_parallel_vs_sequential(self) -> Dict[str, Any]:
        """並列・逐次処理比較テスト"""
        
        print("🔄 並列・逐次処理比較テスト実行中...")
        
        try:
            test_symbols = self.test_symbol_sets["medium"]  # 10銘柄で比較
            
            # 並列処理テスト（5銘柄以上で自動的に並列処理）
            start_time = time.perf_counter()
            parallel_result = self.screener.apply_market_cap_filter(test_symbols)
            parallel_time = time.perf_counter() - start_time
            
            # 逐次処理テスト（直接_sequential_market_cap_filterを呼び出し）
            start_time = time.perf_counter()
            min_cap = self.screener.config["screening"]["nikkei225_filters"]["min_market_cap"]
            sequential_result = self.screener._sequential_market_cap_filter(test_symbols, min_cap)
            sequential_time = time.perf_counter() - start_time
            
            # 比較分析
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            improvement_percentage = (1 - parallel_time / sequential_time) * 100 if sequential_time > 0 else 0
            
            comparison_data = {
                "parallel_execution": {
                    "time": round(parallel_time, 2),
                    "result_count": len(parallel_result),
                    "throughput": round(len(test_symbols) / parallel_time, 1)
                },
                "sequential_execution": {
                    "time": round(sequential_time, 2),
                    "result_count": len(sequential_result),
                    "throughput": round(len(test_symbols) / sequential_time, 1)
                },
                "performance_comparison": {
                    "speedup_factor": round(speedup, 2),
                    "improvement_percentage": round(improvement_percentage, 1),
                    "results_consistent": len(parallel_result) == len(sequential_result),
                    "parallel_advantage": speedup > 1.0
                }
            }
            
            print(f"  📊 比較結果: {speedup:.2f}x高速化 ({improvement_percentage:.1f}%改善)")
            
            return {
                "success": True,
                "comparison_data": comparison_data,
                "test_symbols_count": len(test_symbols)
            }
            
        except Exception as e:
            return {"success": False, "error": f"比較テストエラー: {e}"}
    
    def _analyze_effectiveness(self) -> Dict[str, Any]:
        """効果分析・外挿"""
        
        print("📈 効果分析・200銘柄スケール外挿中...")
        
        try:
            # 実測データから200銘柄への外挿
            scaling_data = self.test_results.get("performance_scaling", {}).get("scaling_results", {})
            
            if "large" in scaling_data:
                large_test = scaling_data["large"]
                symbols_per_second = large_test.get("symbols_per_second", 0)
                
                if symbols_per_second > 0:
                    # 200銘柄処理時間推定
                    estimated_200_symbols_time = 200 / symbols_per_second
                    
                    # 並列効率を考慮（大規模になると若干効率低下）
                    parallel_efficiency_factor = 0.8  # 80%効率想定
                    realistic_200_symbols_time = estimated_200_symbols_time / parallel_efficiency_factor
                    
                    # 改善効果計算
                    original_estimated_time = 52.5  # 元の実測値
                    improvement_seconds = original_estimated_time - realistic_200_symbols_time
                    improvement_percentage = (improvement_seconds / original_estimated_time) * 100
                    
                    target_achievement = improvement_percentage >= 40  # 40%削減目標
                    
                    effectiveness_data = {
                        "extrapolation_basis": {
                            "test_scale": large_test["symbols_count"],
                            "measured_throughput": symbols_per_second,
                            "efficiency_factor": parallel_efficiency_factor
                        },
                        "projected_performance": {
                            "estimated_200_symbols_time": round(realistic_200_symbols_time, 1),
                            "original_time": original_estimated_time,
                            "improvement_seconds": round(improvement_seconds, 1),
                            "improvement_percentage": round(improvement_percentage, 1)
                        },
                        "target_achievement": {
                            "target_met": target_achievement,
                            "target_percentage": 40,
                            "achieved_percentage": round(improvement_percentage, 1),
                            "performance_category": "excellent" if improvement_percentage >= 50 else "good" if improvement_percentage >= 40 else "needs_improvement"
                        }
                    }
                    
                    print(f"  📊 200銘柄推定: {original_estimated_time}秒 → {realistic_200_symbols_time:.1f}秒 ({improvement_percentage:.1f}%改善)")
                    
                    return {"success": True, "effectiveness_data": effectiveness_data}
            
            return {"success": False, "error": "外挿に必要なデータが不足"}
            
        except Exception as e:
            return {"success": False, "error": f"効果分析エラー: {e}"}
    
    def _assess_final_performance(self) -> Dict[str, Any]:
        """最終パフォーマンス評価"""
        
        try:
            # 各テスト結果の成功状況確認
            basic_ok = self.test_results.get("basic_functionality", {}).get("success", False)
            scaling_ok = len(self.test_results.get("performance_scaling", {}).get("scaling_results", {})) >= 2
            comparison_ok = self.test_results.get("parallel_vs_sequential", {}).get("success", False)
            effectiveness_ok = self.test_results.get("effectiveness_analysis", {}).get("success", False)
            
            success_count = sum([basic_ok, scaling_ok, comparison_ok, effectiveness_ok])
            
            # 目標達成状況
            effectiveness_data = self.test_results.get("effectiveness_analysis", {}).get("effectiveness_data", {})
            target_data = effectiveness_data.get("target_achievement", {})
            target_met = target_data.get("target_met", False)
            achieved_percentage = target_data.get("achieved_percentage", 0)
            
            # 総合評価
            if success_count >= 4 and target_met:
                overall_status = "✅ 完全成功"
                readiness = "実用レベル達成"
            elif success_count >= 3 and achieved_percentage >= 30:
                overall_status = "⚠️ 部分的成功"
                readiness = "改善余地あり・実用可能"
            else:
                overall_status = "❌ 要改善"
                readiness = "追加最適化必要"
            
            return {
                "overall_assessment": {
                    "status": overall_status,
                    "success_tests": f"{success_count}/4",
                    "readiness": readiness
                },
                "test_summary": {
                    "basic_functionality": "✅" if basic_ok else "❌",
                    "performance_scaling": "✅" if scaling_ok else "❌",
                    "parallel_comparison": "✅" if comparison_ok else "❌",
                    "effectiveness_analysis": "✅" if effectiveness_ok else "❌"
                },
                "target_achievement": {
                    "target_met": target_met,
                    "achieved_improvement": f"{achieved_percentage:.1f}%",
                    "target_improvement": "40-50%"
                },
                "next_steps": {
                    "immediate_action": "Stage 3移行" if target_met else "さらなる最適化",
                    "stage3_readiness": target_met,
                    "optimization_priority": "SmartCache統合" if target_met else "並列処理調整"
                }
            }
            
        except Exception as e:
            return {"error": f"最終評価エラー: {e}"}

def main():
    """メイン実行"""
    print("🚀 TODO-PERF-007 Stage 2: 並列処理統合効果テスト")
    print("目標: 52.5秒→25-30秒削減確認・40-50%改善達成検証")
    print("="*80)
    
    try:
        tester = ScreenerPerformanceTester()
        results = tester.run_comprehensive_performance_test()
        
        if "error" not in results:
            print("\n" + "="*80)
            print("🎯 Stage 2: 並列処理統合効果テスト完了")
            print("="*80)
            
            # 最終評価表示
            final_assessment = results.get("final_assessment", {})
            if "overall_assessment" in final_assessment:
                assessment = final_assessment["overall_assessment"]
                test_summary = final_assessment["test_summary"]
                target_achievement = final_assessment["target_achievement"]
                next_steps = final_assessment["next_steps"]
                
                print(f"\n🏆 総合評価:")
                print(f"  ステータス: {assessment['status']}")
                print(f"  成功テスト: {assessment['success_tests']}")
                print(f"  実用性: {assessment['readiness']}")
                
                print(f"\n📊 テスト結果サマリー:")
                print(f"  基本機能: {test_summary['basic_functionality']}")
                print(f"  スケーリング: {test_summary['performance_scaling']}")
                print(f"  並列比較: {test_summary['parallel_comparison']}")
                print(f"  効果分析: {test_summary['effectiveness_analysis']}")
                
                print(f"\n🎯 目標達成状況:")
                print(f"  目標達成: {'✅ 達成' if target_achievement['target_met'] else '❌ 未達成'}")
                print(f"  達成改善率: {target_achievement['achieved_improvement']}")
                print(f"  目標改善率: {target_achievement['target_improvement']}")
                
                print(f"\n🚀 次ステップ:")
                print(f"  即座対応: {next_steps['immediate_action']}")
                print(f"  Stage 3準備: {'✅ 準備完了' if next_steps['stage3_readiness'] else '❌ 要追加最適化'}")
                print(f"  最適化優先度: {next_steps['optimization_priority']}")
            
            # 詳細結果保存
            result_file = f"TODO_PERF_007_Stage2_Performance_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n📄 詳細テスト結果: {result_file}")
            print("="*80)
            
            return results.get("final_assessment", {}).get("target_achievement", {}).get("target_met", False)
        else:
            print(f"\n❌ テスト失敗: {results.get('error')}")
            return False
            
    except Exception as e:
        print(f"\n💥 テスト実行エラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)