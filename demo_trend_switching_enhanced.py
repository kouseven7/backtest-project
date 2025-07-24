"""
Module: Demo Enhanced Trend Switching Test System
File: demo_trend_switching_enhanced.py
Description: 
  改良版トレンド切替テストシステムのデモとベンチマーク実行
  包括的な結果分析と可視化機能付き

Author: imega
Created: 2025-01-22
Modified: 2025-01-22
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# プロジェクトパスの追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# 自作モジュールのインポート
from trend_switching_test_enhanced import (
    TrendSwitchingTester, 
    TrendScenario, 
    MarketCondition, 
    StrategyType,
    TestResult
)

# 設定
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class EnhancedTrendSwitchingDemo:
    """拡張トレンド切替テストデモ"""
    
    def __init__(self, output_dir: str = "demo_trend_switching_enhanced_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 設定読み込み
        self.config = self._load_config()
        
        # テスターの初期化
        self.tester = TrendSwitchingTester(str(self.output_dir))
        
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            config_file = Path("trend_switching_test_config.json")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning("Config file not found, using default settings")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "test_config": {
                "test_scenarios": {
                    "enabled": True,
                    "scenario_count": 4,
                    "data_points_per_scenario": 1000,
                    "switch_frequency": 50
                },
                "performance_criteria": {
                    "min_sharpe_ratio": 0.5,
                    "max_drawdown": 0.2,
                    "min_win_rate": 0.45,
                    "min_total_return": 0.0
                }
            }
        }
    
    def create_extended_scenarios(self) -> List[TrendScenario]:
        """拡張シナリオセット作成"""
        scenarios = [
            # 基本シナリオ
            TrendScenario(
                scenario_id="bull_to_bear",
                name="強気相場から弱気相場への転換",
                description="長期上昇トレンドから急激な下降トレンドへの転換テスト",
                initial_condition=MarketCondition.STRONG_UPTREND,
                target_condition=MarketCondition.STRONG_DOWNTREND,
                transition_period=8,
                volatility_factor=1.5,
                data_points=1200
            ),
            TrendScenario(
                scenario_id="bear_to_bull",
                name="弱気相場から強気相場への転換",
                description="長期下降トレンドから力強い上昇トレンドへの転換テスト",
                initial_condition=MarketCondition.STRONG_DOWNTREND,
                target_condition=MarketCondition.STRONG_UPTREND,
                transition_period=12,
                volatility_factor=1.3,
                data_points=1200
            ),
            TrendScenario(
                scenario_id="sideways_consolidation",
                name="レンジ相場での戦略効果",
                description="横ばいレンジ相場での各戦略の効果測定",
                initial_condition=MarketCondition.SIDEWAYS,
                target_condition=MarketCondition.SIDEWAYS,
                transition_period=30,
                volatility_factor=0.8,
                data_points=800
            ),
            TrendScenario(
                scenario_id="volatility_storm",
                name="ボラティリティ・ストーム",
                description="高ボラティリティ環境での戦略切替効果",
                initial_condition=MarketCondition.LOW_VOLATILITY,
                target_condition=MarketCondition.HIGH_VOLATILITY,
                transition_period=5,
                volatility_factor=3.0,
                data_points=600
            ),
            TrendScenario(
                scenario_id="multi_phase_transition",
                name="多段階トレンド変化",
                description="複数段階のトレンド変化における適応性テスト",
                initial_condition=MarketCondition.MODERATE_UPTREND,
                target_condition=MarketCondition.MODERATE_DOWNTREND,
                transition_period=20,
                volatility_factor=1.2,
                data_points=1500
            ),
            TrendScenario(
                scenario_id="flash_crash_recovery",
                name="フラッシュクラッシュ回復",
                description="急激な下落からの回復過程での戦略効果",
                initial_condition=MarketCondition.MODERATE_UPTREND,
                target_condition=MarketCondition.HIGH_VOLATILITY,
                transition_period=3,
                volatility_factor=2.5,
                data_points=400
            )
        ]
        
        return scenarios
    
    def run_comprehensive_test(self) -> Dict[str, TestResult]:
        """包括的テスト実行"""
        logger.info("=== Enhanced Trend Switching Test Demo Started ===")
        
        # 拡張シナリオでテスト実行
        original_scenarios = self.tester.create_test_scenarios
        self.tester.create_test_scenarios = self.create_extended_scenarios
        
        start_time = time.time()
        results = self.tester.run_all_tests()
        execution_time = time.time() - start_time
        
        # 統計情報出力
        self._print_execution_summary(results, execution_time)
        
        # 詳細分析実行
        self._perform_detailed_analysis(results)
        
        return results
    
    def _print_execution_summary(self, results: Dict[str, TestResult], execution_time: float):
        """実行サマリー出力"""
        print("\n" + "="*60)
        print("     ENHANCED TREND SWITCHING TEST RESULTS")
        print("="*60)
        
        total_scenarios = len(results)
        successful_scenarios = sum(1 for result in results.values() if not result.errors)
        success_rate = (successful_scenarios / total_scenarios) * 100 if total_scenarios > 0 else 0
        
        print(f"📊 Test Overview:")
        print(f"   Total Scenarios: {total_scenarios}")
        print(f"   Successful Tests: {successful_scenarios}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Execution Time: {execution_time:.2f} seconds")
        print(f"   Average Time per Test: {execution_time/total_scenarios:.2f} seconds")
        
        print(f"\n📈 Performance Summary:")
        
        for scenario_id, result in results.items():
            status = "✅" if not result.errors else "❌"
            switches = len(result.switching_events)
            
            print(f"   {status} {scenario_id}:")
            print(f"      Execution: {result.execution_time:.3f}s")
            print(f"      Switches: {switches}")
            
            if result.overall_performance:
                total_return = result.overall_performance.get('avg_total_return', 0)
                sharpe = result.overall_performance.get('avg_sharpe_ratio', 0)
                drawdown = result.overall_performance.get('avg_max_drawdown', 0)
                confidence = result.overall_performance.get('avg_confidence', 0)
                
                print(f"      Return: {total_return:.4f}")
                print(f"      Sharpe: {sharpe:.4f}")
                print(f"      Drawdown: {drawdown:.4f}")
                print(f"      Confidence: {confidence:.4f}")
            
            if result.errors:
                print(f"      ⚠️  Errors: {len(result.errors)}")
        
        print("\n" + "="*60)
    
    def _perform_detailed_analysis(self, results: Dict[str, TestResult]):
        """詳細分析実行"""
        try:
            logger.info("Performing detailed analysis...")
            
            # パフォーマンス分析
            self._analyze_performance_metrics(results)
            
            # 戦略切替分析
            self._analyze_switching_patterns(results)
            
            # 成功要因分析
            self._analyze_success_factors(results)
            
            # ベンチマーク比較
            self._benchmark_comparison(results)
            
            logger.info("Detailed analysis completed")
            
        except Exception as e:
            logger.error(f"Error in detailed analysis: {e}")
    
    def _analyze_performance_metrics(self, results: Dict[str, TestResult]):
        """パフォーマンス指標分析"""
        print(f"\n📊 Performance Metrics Analysis:")
        
        # 全体統計
        all_returns = []
        all_sharpes = []
        all_drawdowns = []
        all_switches = []
        
        for result in results.values():
            if result.overall_performance:
                all_returns.append(result.overall_performance.get('avg_total_return', 0))
                all_sharpes.append(result.overall_performance.get('avg_sharpe_ratio', 0))
                all_drawdowns.append(result.overall_performance.get('avg_max_drawdown', 0))
                all_switches.append(len(result.switching_events))
        
        if all_returns:
            print(f"   Average Total Return: {np.mean(all_returns):.4f} (±{np.std(all_returns):.4f})")
            print(f"   Average Sharpe Ratio: {np.mean(all_sharpes):.4f} (±{np.std(all_sharpes):.4f})")
            print(f"   Average Max Drawdown: {np.mean(all_drawdowns):.4f} (±{np.std(all_drawdowns):.4f})")
            print(f"   Average Switches: {np.mean(all_switches):.1f} (±{np.std(all_switches):.1f})")
            
            # 最高・最低パフォーマンス
            best_return_idx = np.argmax(all_returns)
            worst_return_idx = np.argmin(all_returns)
            
            scenario_names = list(results.keys())
            print(f"   Best Performer: {scenario_names[best_return_idx]} ({all_returns[best_return_idx]:.4f})")
            print(f"   Worst Performer: {scenario_names[worst_return_idx]} ({all_returns[worst_return_idx]:.4f})")
    
    def _analyze_switching_patterns(self, results: Dict[str, TestResult]):
        """戦略切替パターン分析"""
        print(f"\n🔄 Strategy Switching Analysis:")
        
        # 切替頻度分析
        switching_frequencies = {}
        strategy_transitions = {}
        
        for scenario_id, result in results.items():
            switches = len(result.switching_events)
            switching_frequencies[scenario_id] = switches
            
            # 戦略遷移パターン
            transitions = []
            for event in result.switching_events:
                transition = f"{event.from_strategy} → {event.to_strategy}"
                transitions.append(transition)
            
            strategy_transitions[scenario_id] = transitions
        
        if switching_frequencies:
            print(f"   Switch Frequency by Scenario:")
            for scenario, freq in switching_frequencies.items():
                print(f"     {scenario}: {freq} switches")
            
            # 共通の遷移パターン
            all_transitions = [t for transitions in strategy_transitions.values() for t in transitions]
            if all_transitions:
                from collections import Counter
                common_transitions = Counter(all_transitions).most_common(3)
                
                print(f"   Most Common Transitions:")
                for transition, count in common_transitions:
                    print(f"     {transition}: {count} times")
    
    def _analyze_success_factors(self, results: Dict[str, TestResult]):
        """成功要因分析"""
        print(f"\n🎯 Success Factors Analysis:")
        
        criteria_performance = {}
        
        for scenario_id, result in results.items():
            if result.success_metrics:
                for criterion, success in result.success_metrics.items():
                    if criterion not in criteria_performance:
                        criteria_performance[criterion] = {'passed': 0, 'total': 0}
                    
                    criteria_performance[criterion]['total'] += 1
                    if success:
                        criteria_performance[criterion]['passed'] += 1
        
        print(f"   Success Criteria Performance:")
        for criterion, perf in criteria_performance.items():
            success_rate = (perf['passed'] / perf['total']) * 100 if perf['total'] > 0 else 0
            print(f"     {criterion}: {perf['passed']}/{perf['total']} ({success_rate:.1f}%)")
    
    def _benchmark_comparison(self, results: Dict[str, TestResult]):
        """ベンチマーク比較"""
        print(f"\n🏆 Benchmark Comparison:")
        
        # 設定基準と比較
        config_criteria = self.config.get('test_config', {}).get('performance_criteria', {})
        
        compliant_scenarios = 0
        
        for scenario_id, result in results.items():
            if not result.overall_performance:
                continue
                
            meets_criteria = True
            
            # 各基準との比較
            sharpe = result.overall_performance.get('avg_sharpe_ratio', 0)
            min_sharpe = config_criteria.get('min_sharpe_ratio', 0.5)
            if sharpe < min_sharpe:
                meets_criteria = False
            
            drawdown = result.overall_performance.get('avg_max_drawdown', 1)
            max_drawdown = config_criteria.get('max_drawdown', 0.2)
            if drawdown > max_drawdown:
                meets_criteria = False
            
            total_return = result.overall_performance.get('avg_total_return', 0)
            min_return = config_criteria.get('min_total_return', 0)
            if total_return < min_return:
                meets_criteria = False
            
            if meets_criteria:
                compliant_scenarios += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            print(f"     {status} {scenario_id}")
        
        total_scenarios = len([r for r in results.values() if r.overall_performance])
        compliance_rate = (compliant_scenarios / total_scenarios) * 100 if total_scenarios > 0 else 0
        
        print(f"   Overall Compliance: {compliant_scenarios}/{total_scenarios} ({compliance_rate:.1f}%)")
        
    def save_detailed_report(self, results: Dict[str, TestResult]):
        """詳細レポート保存"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.output_dir / f"enhanced_trend_switching_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("ENHANCED TREND SWITCHING TEST DETAILED REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for scenario_id, result in results.items():
                    f.write(f"SCENARIO: {scenario_id}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Name: {result.scenario.name}\n")
                    f.write(f"Description: {result.scenario.description}\n")
                    f.write(f"Initial Condition: {result.scenario.initial_condition.value}\n")
                    f.write(f"Target Condition: {result.scenario.target_condition.value}\n")
                    f.write(f"Data Points: {result.scenario.data_points}\n")
                    f.write(f"Execution Time: {result.execution_time:.4f}s\n")
                    f.write(f"Switching Events: {len(result.switching_events)}\n")
                    
                    if result.overall_performance:
                        f.write("\nPerformance Metrics:\n")
                        for key, value in result.overall_performance.items():
                            f.write(f"  {key}: {value:.6f}\n")
                    
                    if result.success_metrics:
                        f.write("\nSuccess Criteria:\n")
                        for criterion, success in result.success_metrics.items():
                            status = "PASS" if success else "FAIL"
                            f.write(f"  {criterion}: {status}\n")
                    
                    if result.errors:
                        f.write(f"\nErrors ({len(result.errors)}):\n")
                        for error in result.errors:
                            f.write(f"  - {error}\n")
                    
                    f.write("\n" + "="*50 + "\n")
            
            logger.info(f"Detailed report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving detailed report: {e}")

def main():
    """メイン実行関数"""
    try:
        print("🚀 Enhanced Trend Switching Test Demo Starting...")
        
        # デモ実行
        demo = EnhancedTrendSwitchingDemo()
        
        # 包括的テスト実行
        results = demo.run_comprehensive_test()
        
        # 詳細レポート保存
        demo.save_detailed_report(results)
        
        print("\n✅ Enhanced Trend Switching Test Demo completed successfully!")
        print(f"📁 Results saved to: {demo.output_dir}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    main()
