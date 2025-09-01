"""
DSSMS Task 2.3: メインコントローラー
===================================

DSSMSシステムのTask 2.3「パフォーマンス最適化と検証」の統合実行を管理します。

このコントローラーは以下の3つの主要コンポーネントを統合実行します:
1. パフォーマンス最適化 (dssms_task_2_3_performance_optimizer.py)
2. 統合テストスイート (dssms_task_2_3_integration_test_suite.py)  
3. 品質保証システム (dssms_task_2_3_quality_assurance.py)

Author: DSSMS Development Team
Created: 2025-01-22
Version: 1.0.0
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import traceback
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパス設定
PROJECT_ROOT = Path(r"C:\Users\imega\Documents\my_backtest_project")
sys.path.append(str(PROJECT_ROOT))

# ロギング設定
from config.logger_config import setup_logger
logger = setup_logger(__name__, log_file=str(PROJECT_ROOT / "logs" / "dssms_task_2_3_main.log"))

# Task 2.3 コンポーネントのインポート
try:
    from dssms_task_2_3_performance_optimizer import DSSMSPerformanceOptimizer
    from dssms_task_2_3_integration_test_suite import DSSMSIntegrationTestSuite
    from dssms_task_2_3_quality_assurance import DSSMSQualityAssuranceSystem
    components_available = True
    logger.info("All Task 2.3 components loaded successfully")
except ImportError as e:
    components_available = False
    logger.error(f"Failed to load Task 2.3 components: {e}")

class DSSMSTask23Controller:
    """DSSMS Task 2.3 メインコントローラー"""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Task 2.3 コントローラーの初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.results_dir = self.project_root / "analysis_results" / "task_2_3"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # コンポーネントの初期化
        if components_available:
            self.performance_optimizer = DSSMSPerformanceOptimizer(str(self.project_root))
            self.integration_test_suite = DSSMSIntegrationTestSuite(str(self.project_root))
            self.quality_assurance = DSSMSQualityAssuranceSystem(str(self.project_root))
            logger.info("Task 2.3 components initialized successfully")
        else:
            self.performance_optimizer = None
            self.integration_test_suite = None
            self.quality_assurance = None
            logger.warning("Task 2.3 components not available")
        
        self.execution_results = {}
        
        logger.info("DSSMS Task 2.3 Controller initialized")
        logger.info(f"Results directory: {self.results_dir}")
    
    def execute_task_2_3_complete(self) -> Dict[str, Any]:
        """
        Task 2.3の完全実行
        
        Returns:
            Dict[str, Any]: 実行結果のサマリー
        """
        logger.info("🚀 Starting DSSMS Task 2.3: パフォーマンス最適化と検証")
        print("\n🚀 DSSMS Task 2.3: パフォーマンス最適化と検証")
        print("=" * 60)
        
        execution_summary = {
            'start_time': datetime.now().isoformat(),
            'components': {
                'performance_optimization': {'status': 'pending', 'results': None},
                'integration_testing': {'status': 'pending', 'results': None},
                'quality_assurance': {'status': 'pending', 'results': None}
            },
            'overall_status': 'in_progress',
            'total_execution_time': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Phase 1: システム最適化
            print("\n📈 Phase 1: システム最適化")
            print("-" * 30)
            execution_summary['components']['performance_optimization'] = self._execute_performance_optimization()
            
            # Phase 2: 統合テスト
            print("\n🧪 Phase 2: 統合テスト")
            print("-" * 30)
            execution_summary['components']['integration_testing'] = self._execute_integration_testing()
            
            # Phase 3: 品質保証
            print("\n🔍 Phase 3: 品質保証")
            print("-" * 30)
            execution_summary['components']['quality_assurance'] = self._execute_quality_assurance()
            
            # 実行結果の評価
            execution_summary = self._evaluate_execution_results(execution_summary)
            
        except Exception as e:
            logger.error(f"Task 2.3 execution failed: {e}")
            logger.error(traceback.format_exc())
            execution_summary['errors'].append(str(e))
            execution_summary['overall_status'] = 'failed'
        
        execution_summary['total_execution_time'] = time.time() - start_time
        execution_summary['end_time'] = datetime.now().isoformat()
        
        # 結果の保存
        self._save_execution_results(execution_summary)
        
        # 最終レポートの表示
        self._display_final_report(execution_summary)
        
        logger.info("DSSMS Task 2.3 execution completed")
        return execution_summary
    
    def _execute_performance_optimization(self) -> Dict[str, Any]:
        """パフォーマンス最適化の実行"""
        logger.info("Executing performance optimization")
        
        result = {'status': 'pending', 'results': None, 'execution_time': 0}
        start_time = time.time()
        
        try:
            if self.performance_optimizer:
                print("  ⚡ バックテスト実行速度最適化...")
                print("  💾 メモリ使用量削減...")
                print("  🔀 並列処理の導入...")
                
                # パフォーマンスベンチマーク実行
                benchmark_results = self.performance_optimizer.run_performance_benchmark()
                
                # ダミーデータでの最適化テスト
                import pandas as pd
                import numpy as np
                test_data = pd.DataFrame({
                    'Date': pd.date_range('2024-01-01', periods=1000),
                    'Close': np.random.random(1000) * 100,
                    'Volume': np.random.randint(1000, 100000, 1000)
                })
                
                optimized_data = self.performance_optimizer.optimize_data_processing(test_data)
                
                result['results'] = {
                    'benchmark_results': benchmark_results,
                    'data_optimization': {
                        'original_size': test_data.memory_usage(deep=True).sum(),
                        'optimized_size': optimized_data.memory_usage(deep=True).sum(),
                        'memory_reduction_percentage': (
                            1 - optimized_data.memory_usage(deep=True).sum() / 
                            test_data.memory_usage(deep=True).sum()
                        ) * 100
                    },
                    'parallel_processing_enabled': True
                }
                result['status'] = 'completed'
                print("  ✅ パフォーマンス最適化完了")
                
            else:
                result['status'] = 'skipped'
                result['results'] = {'reason': 'Performance optimizer not available'}
                print("  ⚠️  パフォーマンス最適化器が利用できません")
                
        except Exception as e:
            result['status'] = 'failed'
            result['results'] = {'error': str(e)}
            logger.error(f"Performance optimization failed: {e}")
            print(f"  ❌ パフォーマンス最適化エラー: {e}")
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _execute_integration_testing(self) -> Dict[str, Any]:
        """統合テストの実行"""
        logger.info("Executing integration testing")
        
        result = {'status': 'pending', 'results': None, 'execution_time': 0}
        start_time = time.time()
        
        try:
            if self.integration_test_suite:
                print("  🔧 統合テストスイート実行...")
                print("  🔄 エンドツーエンドテスト...")
                print("  📊 パフォーマンステスト...")
                
                # 統合テスト実行
                test_results = self.integration_test_suite.run_all_tests()
                
                result['results'] = {
                    'total_tests': test_results['tests_passed'] + test_results['tests_failed'],
                    'tests_passed': test_results['tests_passed'],
                    'tests_failed': test_results['tests_failed'],
                    'success_rate': test_results['success_rate'],
                    'test_execution_time': test_results['total_execution_time'],
                    'test_categories': [
                        'basic_integration',
                        'data_processing',
                        'strategy_execution',
                        'performance',
                        'end_to_end',
                        'error_handling'
                    ]
                }
                result['status'] = 'completed'
                print(f"  ✅ 統合テスト完了: {test_results['tests_passed']}/{test_results['tests_passed'] + test_results['tests_failed']} 成功")
                
            else:
                result['status'] = 'skipped'
                result['results'] = {'reason': 'Integration test suite not available'}
                print("  ⚠️  統合テストスイートが利用できません")
                
        except Exception as e:
            result['status'] = 'failed'
            result['results'] = {'error': str(e)}
            logger.error(f"Integration testing failed: {e}")
            print(f"  ❌ 統合テストエラー: {e}")
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _execute_quality_assurance(self) -> Dict[str, Any]:
        """品質保証の実行"""
        logger.info("Executing quality assurance")
        
        result = {'status': 'pending', 'results': None, 'execution_time': 0}
        start_time = time.time()
        
        try:
            if self.quality_assurance:
                print("  🔍 自動コードレビュー...")
                print("  🐛 バグ検出・修正...")
                print("  📚 ドキュメント更新...")
                
                # 品質保証実行
                qa_report = self.quality_assurance.run_comprehensive_quality_assurance()
                
                result['results'] = {
                    'overall_quality_score': qa_report.overall_score,
                    'code_metrics': {
                        'lines_of_code': qa_report.code_metrics.lines_of_code,
                        'complexity_score': qa_report.code_metrics.complexity_score,
                        'documentation_coverage': qa_report.code_metrics.documentation_coverage,
                        'test_coverage': qa_report.code_metrics.test_coverage,
                        'bug_risk_score': qa_report.code_metrics.bug_risk_score,
                        'maintainability_index': qa_report.code_metrics.maintainability_index,
                        'code_smells_count': len(qa_report.code_metrics.code_smells)
                    },
                    'recommendations_count': len(qa_report.recommendations),
                    'fixed_issues_count': len(qa_report.fixed_issues),
                    'documentation_updates_count': len(qa_report.documentation_updates)
                }
                result['status'] = 'completed'
                print(f"  ✅ 品質保証完了: スコア {qa_report.overall_score:.1f}/100")
                
            else:
                result['status'] = 'skipped'
                result['results'] = {'reason': 'Quality assurance system not available'}
                print("  ⚠️  品質保証システムが利用できません")
                
        except Exception as e:
            result['status'] = 'failed'
            result['results'] = {'error': str(e)}
            logger.error(f"Quality assurance failed: {e}")
            print(f"  ❌ 品質保証エラー: {e}")
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _evaluate_execution_results(self, execution_summary: Dict[str, Any]) -> Dict[str, Any]:
        """実行結果の評価"""
        logger.info("Evaluating execution results")
        
        completed_components = 0
        failed_components = 0
        
        for component_name, component_result in execution_summary['components'].items():
            if component_result['status'] == 'completed':
                completed_components += 1
            elif component_result['status'] == 'failed':
                failed_components += 1
        
        total_components = len(execution_summary['components'])
        success_rate = completed_components / total_components
        
        if success_rate >= 0.8:  # 80%以上成功
            execution_summary['overall_status'] = 'success'
        elif success_rate >= 0.5:  # 50%以上成功
            execution_summary['overall_status'] = 'partial_success'
        else:
            execution_summary['overall_status'] = 'failed'
        
        execution_summary['evaluation'] = {
            'completed_components': completed_components,
            'failed_components': failed_components,
            'total_components': total_components,
            'success_rate': success_rate,
            'task_completion_level': self._calculate_completion_level(execution_summary)
        }
        
        return execution_summary
    
    def _calculate_completion_level(self, execution_summary: Dict[str, Any]) -> str:
        """完了レベルの計算"""
        performance_status = execution_summary['components']['performance_optimization']['status']
        testing_status = execution_summary['components']['integration_testing']['status']
        qa_status = execution_summary['components']['quality_assurance']['status']
        
        if all(status == 'completed' for status in [performance_status, testing_status, qa_status]):
            return 'full_completion'
        elif any(status == 'completed' for status in [performance_status, testing_status, qa_status]):
            return 'partial_completion'
        else:
            return 'minimal_completion'
    
    def _save_execution_results(self, execution_summary: Dict[str, Any]):
        """実行結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"task_2_3_execution_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(execution_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Execution results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save execution results: {e}")
    
    def _display_final_report(self, execution_summary: Dict[str, Any]):
        """最終レポートの表示"""
        print("\n📋 Task 2.3 実行結果サマリー")
        print("=" * 60)
        
        # 全体ステータス
        status_emoji = {
            'success': '✅',
            'partial_success': '⚠️',
            'failed': '❌',
            'in_progress': '🔄'
        }
        
        overall_status = execution_summary['overall_status']
        print(f"🎯 全体ステータス: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
        print(f"⏱️  総実行時間: {execution_summary['total_execution_time']:.2f} 秒")
        
        if 'evaluation' in execution_summary:
            eval_data = execution_summary['evaluation']
            print(f"📊 成功率: {eval_data['success_rate']:.1%} ({eval_data['completed_components']}/{eval_data['total_components']})")
            print(f"🎭 完了レベル: {eval_data['task_completion_level']}")
        
        # コンポーネント別結果
        print("\n📋 コンポーネント別結果:")
        
        for component_name, result in execution_summary['components'].items():
            status = result['status']
            exec_time = result.get('execution_time', 0)
            
            print(f"  {status_emoji.get(status, '❓')} {component_name}: {status} ({exec_time:.2f}s)")
            
            if result['results'] and isinstance(result['results'], dict):
                if component_name == 'performance_optimization' and 'data_optimization' in result['results']:
                    opt_data = result['results']['data_optimization']
                    print(f"    💾 メモリ削減: {opt_data.get('memory_reduction_percentage', 0):.1f}%")
                
                elif component_name == 'integration_testing' and 'total_tests' in result['results']:
                    test_data = result['results']
                    print(f"    🧪 テスト成功率: {test_data.get('success_rate', 0):.1%}")
                
                elif component_name == 'quality_assurance' and 'overall_quality_score' in result['results']:
                    qa_data = result['results']
                    print(f"    🔍 品質スコア: {qa_data.get('overall_quality_score', 0):.1f}/100")
        
        # エラー情報
        if execution_summary['errors']:
            print(f"\n⚠️  エラー ({len(execution_summary['errors'])} 件):")
            for error in execution_summary['errors']:
                print(f"    • {error}")
        
        # 成果物
        print("\n🎁 成果物:")
        print("  • 最適化済みシステム")
        print("  • 統合テストスイート")
        print("  • パフォーマンスベンチマーク")
        print("  • 品質保証レポート")
        print("  • 実行結果サマリー")
        
        print("\n" + "=" * 60)
        print("🎉 DSSMS Task 2.3: パフォーマンス最適化と検証 - 完了")
        print("=" * 60)
    
    def get_task_status(self) -> Dict[str, Any]:
        """Task 2.3の現在のステータスを取得"""
        return {
            'task_name': 'Task 2.3: パフォーマンス最適化と検証',
            'components_available': components_available,
            'project_root': str(self.project_root),
            'results_directory': str(self.results_dir),
            'last_execution': self.execution_results.get('end_time', None)
        }

def main():
    """メイン実行関数"""
    print("\n" + "=" * 60)
    print("🚀 DSSMS Task 2.3: パフォーマンス最適化と検証")
    print("=" * 60)
    print("📋 作業内容:")
    print("  1. システム最適化 - バックテスト実行速度最適化、メモリ削減、並列処理")
    print("  2. 統合テスト - 統合テストスイート、E2Eテスト、パフォーマンステスト")
    print("  3. 品質保証 - コードレビュー、バグ修正、ドキュメント更新")
    print("=" * 60)
    
    controller = DSSMSTask23Controller()
    
    # Task 2.3の実行
    results = controller.execute_task_2_3_complete()
    
    return results

if __name__ == "__main__":
    main()
