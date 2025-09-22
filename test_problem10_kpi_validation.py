"""
Problem 10 Phase 3: KPI測定・統合検証スクリプト

目標KPI:
- 計算エラー率 < 5%
- Statistical indicators 100% success
- ZeroDivisionError elimination
- 85.0-point品質維持確認

実行方法: python test_problem10_kpi_validation.py
"""

import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import traceback

# プロジェクトルート追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from src.dssms.dssms_backtester import DSSMSBacktester

@dataclass
class KPITestResult:
    """KPI測定結果"""
    test_name: str
    success: bool
    error_rate: float
    execution_time: float
    details: Dict[str, Any]
    quality_score: float

class Problem10KPIValidator:
    """Problem 10 KPI検証システム"""
    
    def __init__(self):
        self.logger = setup_logger('problem10.kpi.validator')
        self.results: List[KPITestResult] = []
        self.total_tests = 0
        self.successful_tests = 0
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """包括的KPI検証実行"""
        self.logger.info("Problem 10 Phase 3: KPI測定開始")
        start_time = time.time()
        
        # テストスイート実行
        test_results = []
        
        # Test 1: Critical attributes初期化テスト
        test_results.append(self._test_critical_attributes_initialization())
        
        # Test 2: Performance calculation安全性テスト
        test_results.append(self._test_performance_calculation_safety())
        
        # Test 3: ZeroDivisionError prevention テスト
        test_results.append(self._test_zero_division_prevention())
        
        # Test 4: Statistical indicators計算テスト
        test_results.append(self._test_statistical_indicators())
        
        # Test 5: 品質エンジン出力テスト（85.0-point維持確認）
        test_results.append(self._test_quality_engine_output())
        
        # Test 6: ストレステスト（エラー率計算）
        test_results.append(self._test_stress_conditions())
        
        total_time = time.time() - start_time
        
        # KPI計算
        kpi_summary = self._calculate_kpi_summary(test_results, total_time)
        
        self.logger.info(f"KPI検証完了: {total_time:.2f}秒")
        return kpi_summary
    
    def _test_critical_attributes_initialization(self) -> KPITestResult:
        """Test 1: Critical attributes安全初期化テスト"""
        test_name = "Critical_Attributes_Initialization"
        start_time = time.time()
        errors = 0
        total_attempts = 10
        
        self.logger.info(f"テスト開始: {test_name}")
        
        try:
            details = {
                'initial_capital_tests': [],
                'daily_returns_tests': [],
                'performance_metrics_tests': []
            }
            
            for i in range(total_attempts):
                try:
                    # 様々な設定でDSSMSBacktester初期化テスト
                    configs = [
                        {'initial_capital': 1000000},
                        {'initial_capital': 500000},
                        {'initial_capital': -1000000},  # 異常値テスト
                        {},  # デフォルト設定テスト
                        {'initial_capital': 'invalid'}  # 型エラーテスト
                    ]
                    
                    config = configs[i % len(configs)]
                    backtester = DSSMSBacktester(config)
                    
                    # Critical attributes検証
                    if hasattr(backtester, '_validate_critical_attributes'):
                        validation_result = backtester._validate_critical_attributes()
                        details['initial_capital_tests'].append({
                            'config': str(config),
                            'validation_success': validation_result,
                            'initial_capital_value': getattr(backtester, 'initial_capital', 'missing')
                        })
                    else:
                        errors += 1
                        details['initial_capital_tests'].append({
                            'config': str(config),
                            'error': '_validate_critical_attributes method missing'
                        })
                    
                except Exception as e:
                    errors += 1
                    details['initial_capital_tests'].append({
                        'config': str(config) if 'config' in locals() else 'unknown',
                        'error': str(e)
                    })
            
            error_rate = errors / total_attempts
            success = error_rate < 0.05  # 5%以下ならOK
            execution_time = time.time() - start_time
            
            return KPITestResult(
                test_name=test_name,
                success=success,
                error_rate=error_rate,
                execution_time=execution_time,
                details=details,
                quality_score=85.0 if success else 60.0
            )
            
        except Exception as e:
            self.logger.error(f"テスト{test_name}で致命的エラー: {e}")
            return KPITestResult(
                test_name=test_name,
                success=False,
                error_rate=1.0,
                execution_time=time.time() - start_time,
                details={'fatal_error': str(e)},
                quality_score=0.0
            )
    
    def _test_performance_calculation_safety(self) -> KPITestResult:
        """Test 2: Performance calculation安全性テスト"""
        test_name = "Performance_Calculation_Safety"
        start_time = time.time()
        errors = 0
        total_attempts = 20
        
        self.logger.info(f"テスト開始: {test_name}")
        
        try:
            details = {'calculation_tests': []}
            
            for i in range(total_attempts):
                try:
                    backtester = DSSMSBacktester({'initial_capital': 1000000})
                    
                    # 様々な異常データでパフォーマンス計算テスト
                    test_data = [
                        {'portfolio_value': [1000000, 1050000, 1100000], 'daily_returns': [0.05, 0.047619]},
                        {'portfolio_value': [], 'daily_returns': []},  # 空データ
                        {'portfolio_value': [1000000], 'daily_returns': []},  # 不十分データ
                        {'portfolio_value': [0, 0, 0], 'daily_returns': [0, 0]},  # ゼロ値
                        {'portfolio_value': [1000000, float('inf')], 'daily_returns': [float('inf')]},  # 無限値
                    ]
                    
                    test_case = test_data[i % len(test_data)]
                    backtester.performance_history.update(test_case)
                    
                    # パフォーマンス計算実行
                    simulation_result = {'test': True}
                    result = backtester.calculate_dssms_performance(simulation_result)
                    
                    details['calculation_tests'].append({
                        'test_case': str(test_case),
                        'success': True,
                        'total_return': result.total_return,
                        'volatility': result.volatility,
                        'max_drawdown': result.max_drawdown
                    })
                    
                except Exception as e:
                    errors += 1
                    details['calculation_tests'].append({
                        'test_case': str(test_case) if 'test_case' in locals() else 'unknown',
                        'error': str(e),
                        'success': False
                    })
            
            error_rate = errors / total_attempts
            success = error_rate < 0.05
            execution_time = time.time() - start_time
            
            return KPITestResult(
                test_name=test_name,
                success=success,
                error_rate=error_rate,
                execution_time=execution_time,
                details=details,
                quality_score=85.0 if success else 70.0
            )
            
        except Exception as e:
            self.logger.error(f"テスト{test_name}で致命的エラー: {e}")
            return KPITestResult(
                test_name=test_name,
                success=False,
                error_rate=1.0,
                execution_time=time.time() - start_time,
                details={'fatal_error': str(e)},
                quality_score=0.0
            )
    
    def _test_zero_division_prevention(self) -> KPITestResult:
        """Test 3: ZeroDivisionError prevention テスト"""
        test_name = "ZeroDivisionError_Prevention"
        start_time = time.time()
        zero_division_errors = 0
        total_attempts = 15
        
        self.logger.info(f"テスト開始: {test_name}")
        
        details = {'zero_division_tests': []}
        
        for i in range(total_attempts):
            try:
                backtester = DSSMSBacktester({'initial_capital': 1000000})
                
                # ZeroDivisionErrorを引き起こしやすい条件を設定
                problematic_cases = [
                    {'portfolio_value': [0, 0, 0], 'daily_returns': [0, 0]},
                    {'portfolio_value': [1000000, 1000000, 1000000], 'daily_returns': [0, 0]},  # 変動なし
                    {'portfolio_value': [1000000], 'daily_returns': []},  # 単一値
                    {'initial_capital': 0},  # ゼロ資本
                ]
                
                case = problematic_cases[i % len(problematic_cases)]
                
                if 'initial_capital' in case:
                    backtester.initial_capital = case['initial_capital']
                else:
                    backtester.performance_history.update(case)
                
                # パフォーマンス計算実行（ZeroDivisionErrorが起きるはず）
                simulation_result = {'test': True}
                result = backtester.calculate_dssms_performance(simulation_result)
                
                details['zero_division_tests'].append({
                    'case': str(case),
                    'result_obtained': True,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio
                })
                
            except ZeroDivisionError:
                zero_division_errors += 1
                details['zero_division_tests'].append({
                    'case': str(case) if 'case' in locals() else 'unknown',
                    'zero_division_error': True
                })
                
            except Exception as e:
                details['zero_division_tests'].append({
                    'case': str(case) if 'case' in locals() else 'unknown',
                    'other_error': str(e)
                })
        
        # ZeroDivisionErrorが発生しないことを確認
        success = zero_division_errors == 0
        error_rate = zero_division_errors / total_attempts
        execution_time = time.time() - start_time
        
        return KPITestResult(
            test_name=test_name,
            success=success,
            error_rate=error_rate,
            execution_time=execution_time,
            details=details,
            quality_score=85.0 if success else 50.0
        )
    
    def _test_statistical_indicators(self) -> KPITestResult:
        """Test 4: Statistical indicators 100% success テスト"""
        test_name = "Statistical_Indicators_Success"
        start_time = time.time()
        failed_indicators = 0
        total_indicators = 0
        
        self.logger.info(f"テスト開始: {test_name}")
        
        details = {'indicator_tests': []}
        
        try:
            backtester = DSSMSBacktester({'initial_capital': 1000000})
            
            # 正常なデータでの統計指標計算テスト
            normal_data = {
                'portfolio_value': [1000000, 1020000, 1050000, 1030000, 1080000],
                'daily_returns': [0.02, 0.029412, -0.019048, 0.048544]
            }
            backtester.performance_history.update(normal_data)
            
            simulation_result = {'test': True}
            result = backtester.calculate_dssms_performance(simulation_result)
            
            # 統計指標の妥当性確認
            indicators_to_check = [
                ('total_return', result.total_return),
                ('volatility', result.volatility),
                ('max_drawdown', result.max_drawdown),
                ('sharpe_ratio', result.sharpe_ratio),
                ('sortino_ratio', result.sortino_ratio)
            ]
            
            for indicator_name, value in indicators_to_check:
                total_indicators += 1
                
                # 妥当性チェック
                is_valid = (
                    isinstance(value, (int, float)) and
                    not (value == float('inf') or value == float('-inf')) and
                    not (value != value)  # NaNチェック
                )
                
                if not is_valid:
                    failed_indicators += 1
                
                details['indicator_tests'].append({
                    'indicator': indicator_name,
                    'value': str(value),
                    'valid': is_valid,
                    'type': str(type(value))
                })
            
            success_rate = (total_indicators - failed_indicators) / total_indicators if total_indicators > 0 else 0
            success = success_rate == 1.0  # 100% success required
            error_rate = failed_indicators / total_indicators if total_indicators > 0 else 0
            execution_time = time.time() - start_time
            
            return KPITestResult(
                test_name=test_name,
                success=success,
                error_rate=error_rate,
                execution_time=execution_time,
                details=details,
                quality_score=85.0 if success else 75.0
            )
            
        except Exception as e:
            self.logger.error(f"テスト{test_name}で致命的エラー: {e}")
            return KPITestResult(
                test_name=test_name,
                success=False,
                error_rate=1.0,
                execution_time=time.time() - start_time,
                details={'fatal_error': str(e)},
                quality_score=0.0
            )
    
    def _test_quality_engine_output(self) -> KPITestResult:
        """Test 5: 品質エンジン出力テスト（85.0-point維持確認）"""
        test_name = "Quality_Engine_Output_85Point"
        start_time = time.time()
        
        self.logger.info(f"テスト開始: {test_name}")
        
        try:
            backtester = DSSMSBacktester({'initial_capital': 1000000})
            
            # 品質スコア計算に必要なデータを設定
            quality_data = {
                'portfolio_value': [1000000, 1050000, 1080000, 1120000, 1100000],
                'daily_returns': [0.05, 0.02857, 0.037037, -0.017857]
            }
            backtester.performance_history.update(quality_data)
            
            simulation_result = {'test': True}
            result = backtester.calculate_dssms_performance(simulation_result)
            
            # 品質スコア算出（簡易版）
            quality_factors = {
                'calculation_success': 1.0 if result.total_return is not None else 0.0,
                'data_validity': 1.0 if abs(result.total_return) < 10 else 0.0,  # 妥当な範囲
                'ratio_validity': 1.0 if -5 <= result.sharpe_ratio <= 5 else 0.0,
                'no_infinity': 1.0 if result.volatility != float('inf') else 0.0,
                'no_nan': 1.0 if result.volatility == result.volatility else 0.0  # NaNチェック
            }
            
            average_quality = sum(quality_factors.values()) / len(quality_factors)
            quality_score = average_quality * 85.0  # 85.0ポイント満点
            
            success = quality_score >= 85.0
            execution_time = time.time() - start_time
            
            details = {
                'quality_factors': quality_factors,
                'quality_score': quality_score,
                'performance_metrics': {
                    'total_return': result.total_return,
                    'volatility': result.volatility,
                    'sharpe_ratio': result.sharpe_ratio,
                    'sortino_ratio': result.sortino_ratio
                }
            }
            
            return KPITestResult(
                test_name=test_name,
                success=success,
                error_rate=0.0 if success else 1.0,
                execution_time=execution_time,
                details=details,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"テスト{test_name}で致命的エラー: {e}")
            return KPITestResult(
                test_name=test_name,
                success=False,
                error_rate=1.0,
                execution_time=time.time() - start_time,
                details={'fatal_error': str(e)},
                quality_score=0.0
            )
    
    def _test_stress_conditions(self) -> KPITestResult:
        """Test 6: ストレステスト（エラー率計算）"""
        test_name = "Stress_Conditions_Error_Rate"
        start_time = time.time()
        errors = 0
        total_attempts = 50  # 多数回テスト
        
        self.logger.info(f"テスト開始: {test_name} - {total_attempts}回実行")
        
        details = {'stress_tests': []}
        
        for i in range(total_attempts):
            try:
                # ランダムなストレス条件生成
                import random
                
                stress_configs = [
                    {'initial_capital': random.uniform(1, 10000000)},
                    {'initial_capital': 0},
                    {'initial_capital': -random.uniform(1, 1000000)},
                    {},
                    {'initial_capital': float('inf')},
                    {'initial_capital': 'invalid_type'}
                ]
                
                config = stress_configs[i % len(stress_configs)]
                backtester = DSSMSBacktester(config)
                
                # ランダムな異常データ
                random_data = {
                    'portfolio_value': [random.uniform(-1000000, 2000000) for _ in range(random.randint(0, 10))],
                    'daily_returns': [random.uniform(-0.5, 0.5) for _ in range(random.randint(0, 9))]
                }
                
                backtester.performance_history.update(random_data)
                
                simulation_result = {'stress_test': True}
                result = backtester.calculate_dssms_performance(simulation_result)
                
                details['stress_tests'].append({
                    'attempt': i,
                    'config': str(config),
                    'success': True,
                    'total_return': str(result.total_return)[:50]  # 文字列制限
                })
                
            except Exception as e:
                errors += 1
                details['stress_tests'].append({
                    'attempt': i,
                    'config': str(config) if 'config' in locals() else 'unknown',
                    'error': str(e)[:100],  # エラー文字列制限
                    'success': False
                })
        
        error_rate = errors / total_attempts
        success = error_rate < 0.05  # 5%未満ならOK
        execution_time = time.time() - start_time
        
        return KPITestResult(
            test_name=test_name,
            success=success,
            error_rate=error_rate,
            execution_time=execution_time,
            details=details,
            quality_score=85.0 if success else max(0, 85.0 - (error_rate * 100))
        )
    
    def _calculate_kpi_summary(self, test_results: List[KPITestResult], total_time: float) -> Dict[str, Any]:
        """KPI要約計算"""
        
        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results if r.success)
        overall_error_rate = sum(r.error_rate for r in test_results) / total_tests if total_tests > 0 else 1.0
        average_quality_score = sum(r.quality_score for r in test_results) / total_tests if total_tests > 0 else 0.0
        
        # KPI判定
        kpi_achievement = {
            'error_rate_below_5_percent': overall_error_rate < 0.05,
            'statistical_indicators_100_percent': any(r.test_name == "Statistical_Indicators_Success" and r.success for r in test_results),
            'zero_division_elimination': any(r.test_name == "ZeroDivisionError_Prevention" and r.success for r in test_results),
            'quality_score_85_point': average_quality_score >= 85.0
        }
        
        all_kpis_achieved = all(kpi_achievement.values())
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'problem_10_phase_3_results': {
                'total_execution_time': total_time,
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'overall_error_rate': overall_error_rate,
                'average_quality_score': average_quality_score,
                'all_kpis_achieved': all_kpis_achieved
            },
            'kpi_achievement': kpi_achievement,
            'individual_test_results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'error_rate': r.error_rate,
                    'quality_score': r.quality_score,
                    'execution_time': r.execution_time
                }
                for r in test_results
            ],
            'recommendations': self._generate_recommendations(kpi_achievement, overall_error_rate, average_quality_score)
        }
        
        return summary
    
    def _generate_recommendations(self, kpi_achievement: Dict[str, bool], error_rate: float, quality_score: float) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []
        
        if not kpi_achievement['error_rate_below_5_percent']:
            recommendations.append(f"エラー率{error_rate:.1%}が目標5%を超過 - 例外処理の強化が必要")
        
        if not kpi_achievement['statistical_indicators_100_percent']:
            recommendations.append("統計指標計算の100%成功達成未達 - 計算ロジックの見直しが必要")
        
        if not kpi_achievement['zero_division_elimination']:
            recommendations.append("ZeroDivisionError撲滅未達成 - ゼロ除算対策の強化が必要")
        
        if not kpi_achievement['quality_score_85_point']:
            recommendations.append(f"品質スコア{quality_score:.1f}が目標85.0未達 - 品質管理システムの改善が必要")
        
        if all(kpi_achievement.values()):
            recommendations.append("🎉 Problem 10: 数学的エラー修正 - 全KPI達成完了！")
        
        return recommendations

def main():
    """メイン実行関数"""
    print("=== Problem 10 Phase 3: KPI測定・統合検証開始 ===")
    
    validator = Problem10KPIValidator()
    results = validator.run_comprehensive_validation()
    
    # 結果出力
    print("\n=== KPI検証結果サマリー ===")
    print(f"実行時間: {results['problem_10_phase_3_results']['total_execution_time']:.2f}秒")
    print(f"総テスト数: {results['problem_10_phase_3_results']['total_tests']}")
    print(f"成功テスト数: {results['problem_10_phase_3_results']['successful_tests']}")
    print(f"総合エラー率: {results['problem_10_phase_3_results']['overall_error_rate']:.1%}")
    print(f"平均品質スコア: {results['problem_10_phase_3_results']['average_quality_score']:.1f}")
    print(f"全KPI達成: {'✅' if results['problem_10_phase_3_results']['all_kpis_achieved'] else '❌'}")
    
    print("\n=== KPI達成状況 ===")
    for kpi, achieved in results['kpi_achievement'].items():
        print(f"{kpi}: {'✅' if achieved else '❌'}")
    
    print("\n=== 推奨事項 ===")
    for rec in results['recommendations']:
        print(f"• {rec}")
    
    # 詳細結果をJSONファイルに保存
    output_file = Path("problem10_kpi_validation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n詳細結果保存: {output_file}")
    
    return results['problem_10_phase_3_results']['all_kpis_achieved']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)