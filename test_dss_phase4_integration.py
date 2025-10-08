#!/usr/bin/env python3
"""
DSS Core V3 Phase 4: 統合テスト・デバッグ

Phase 4 統合テスト内容:
1. 全体フロー動作確認
2. エラーハンドリング確認  
3. パフォーマンス測定
4. ログ出力妥当性確認
5. 設定ファイル影響確認

成功基準: 統合テスト成功率 95%以上
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import json

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__)))

# DSS V3インポート
from src.dssms.dssms_backtester_v3 import DSSBacktesterV3

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DSSPhase4IntegrationTester:
    """DSS Core V3 Phase 4 統合テスト"""
    
    def __init__(self):
        self.test_results = {}
        self.test_passed = 0
        self.test_total = 0
        
    def run_all_tests(self) -> Dict[str, Any]:
        """全統合テスト実行"""
        logger.info("=== DSS Core V3 Phase 4: 統合テスト開始 ===")
        
        # テスト1: 全体フロー動作確認
        self.test_complete_daily_selection_flow()
        
        # テスト2: エラーハンドリング確認
        self.test_error_handling()
        
        # テスト3: パフォーマンス測定
        self.test_performance_measurement()
        
        # テスト4: ログ出力妥当性確認
        self.test_logging_validation()
        
        # テスト5: 設定ファイル影響確認
        self.test_configuration_impact()
        
        # 総合結果
        success_rate = (self.test_passed / self.test_total) * 100 if self.test_total > 0 else 0
        
        summary = {
            'phase': 'Phase 4: 統合テスト・デバッグ',
            'test_results': self.test_results,
            'test_passed': self.test_passed,
            'test_total': self.test_total,
            'success_rate': success_rate,
            'target_success_rate': 95.0,
            'phase4_completed': success_rate >= 95.0,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"=== Phase 4 統合テスト完了: {success_rate:.1f}% ({self.test_passed}/{self.test_total}) ===")
        
        return summary
    
    def test_complete_daily_selection_flow(self):
        """テスト1: 全体フロー動作確認"""
        logger.info("--- テスト1: 全体フロー動作確認 ---")
        
        try:
            # DSS V3初期化
            dss = DSSBacktesterV3()
            
            # 単日実行テスト
            target_date = datetime(2023, 1, 15)
            result = dss.run_daily_selection(target_date)
            
            # 結果検証
            required_keys = ['date', 'selected_symbol', 'ranking', 'execution_time_ms']
            
            checks = {
                'dss_initialization': True,
                'result_contains_required_keys': all(key in result for key in required_keys),
                'selected_symbol_valid': result['selected_symbol'] in dss.symbol_universe,
                'ranking_count_correct': len(result['ranking']) == len(dss.symbol_universe),
                'execution_time_reasonable': 0 < result['execution_time_ms'] < 10000,  # 0-10秒
                'ranking_structure_valid': all(
                    all(key in entry for key in ['symbol', 'score', 'rank']) 
                    for entry in result['ranking']
                )
            }
            
            passed_checks = sum(checks.values())
            total_checks = len(checks)
            
            self.test_results['complete_flow'] = {
                'checks': checks,
                'passed': passed_checks,
                'total': total_checks,
                'success': passed_checks == total_checks,
                'selected_symbol': result['selected_symbol'],
                'execution_time_ms': result['execution_time_ms'],
                'ranking_top3': result['ranking'][:3]
            }
            
            if passed_checks == total_checks:
                self.test_passed += 1
                logger.info(f"[OK] テスト1成功: 全項目パス ({passed_checks}/{total_checks})")
            else:
                logger.warning(f"⚠ テスト1部分成功: ({passed_checks}/{total_checks})")
                
            self.test_total += 1
            
        except Exception as e:
            logger.error(f"[ERROR] テスト1失敗: {e}")
            self.test_results['complete_flow'] = {'error': str(e), 'success': False}
            self.test_total += 1
    
    def test_error_handling(self):
        """テスト2: エラーハンドリング確認"""
        logger.info("--- テスト2: エラーハンドリング確認 ---")
        
        error_tests = {}
        
        try:
            dss = DSSBacktesterV3()
            
            # 2.1 古すぎる日付でのテスト（データ不足）
            try:
                old_date = datetime(1990, 1, 1)  # 古い日付
                result = dss.run_daily_selection(old_date)
                error_tests['old_date_handling'] = {
                    'success': True,
                    'result_type': type(result).__name__,
                    'has_fallback': 'selected_symbol' in result
                }
            except Exception as e:
                error_tests['old_date_handling'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # 2.2 未来日付でのテスト
            try:
                future_date = datetime(2030, 1, 1)  # 未来日付
                result = dss.run_daily_selection(future_date)
                error_tests['future_date_handling'] = {
                    'success': True,
                    'result_type': type(result).__name__
                }
            except Exception as e:
                error_tests['future_date_handling'] = {
                    'success': False,
                    'error': str(e)
                }
            
            successful_tests = sum(1 for test in error_tests.values() if test.get('success', False))
            total_tests = len(error_tests)
            
            self.test_results['error_handling'] = {
                'tests': error_tests,
                'passed': successful_tests,
                'total': total_tests,
                'success': successful_tests >= total_tests * 0.5  # 50%以上成功で合格
            }
            
            if successful_tests >= total_tests * 0.5:
                self.test_passed += 1
                logger.info(f"[OK] テスト2成功: エラーハンドリング良好 ({successful_tests}/{total_tests})")
            else:
                logger.warning(f"⚠ テスト2要改善: エラーハンドリング不十分 ({successful_tests}/{total_tests})")
                
            self.test_total += 1
            
        except Exception as e:
            logger.error(f"[ERROR] テスト2失敗: {e}")
            self.test_results['error_handling'] = {'error': str(e), 'success': False}
            self.test_total += 1
    
    def test_performance_measurement(self):
        """テスト3: パフォーマンス測定"""
        logger.info("--- テスト3: パフォーマンス測定 ---")
        
        try:
            dss = DSSBacktesterV3()
            
            # 複数回実行して平均測定
            execution_times = []
            test_dates = [
                datetime(2023, 1, 15),
                datetime(2023, 2, 15),
                datetime(2023, 3, 15)
            ]
            
            for date in test_dates:
                start_time = time.time()
                result = dss.run_daily_selection(date)
                end_time = time.time()
                
                actual_time = (end_time - start_time) * 1000  # ミリ秒
                reported_time = result.get('execution_time_ms', 0)
                
                execution_times.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'actual_time_ms': actual_time,
                    'reported_time_ms': reported_time,
                    'selected_symbol': result['selected_symbol']
                })
            
            avg_time = sum(t['actual_time_ms'] for t in execution_times) / len(execution_times)
            max_time = max(t['actual_time_ms'] for t in execution_times)
            min_time = min(t['actual_time_ms'] for t in execution_times)
            
            # 目標: 5銘柄処理で1秒（1000ms）以内
            performance_ok = avg_time <= 1000
            
            self.test_results['performance'] = {
                'execution_times': execution_times,
                'avg_time_ms': avg_time,
                'max_time_ms': max_time,
                'min_time_ms': min_time,
                'target_time_ms': 1000,
                'performance_ok': performance_ok,
                'success': performance_ok
            }
            
            if performance_ok:
                self.test_passed += 1
                logger.info(f"[OK] テスト3成功: パフォーマンス良好 (平均: {avg_time:.1f}ms)")
            else:
                logger.warning(f"⚠ テスト3要改善: パフォーマンス改善必要 (平均: {avg_time:.1f}ms)")
                
            self.test_total += 1
            
        except Exception as e:
            logger.error(f"[ERROR] テスト3失敗: {e}")
            self.test_results['performance'] = {'error': str(e), 'success': False}
            self.test_total += 1
    
    def test_logging_validation(self):
        """テスト4: ログ出力妥当性確認"""
        logger.info("--- テスト4: ログ出力妥当性確認 ---")
        
        try:
            # ログキャプチャ用のハンドラ設定
            log_messages = []
            
            class LogCapture(logging.Handler):
                def emit(self, record):
                    log_messages.append(record.getMessage())
            
            # 一時的にログハンドラ追加
            capture_handler = LogCapture()
            capture_handler.setLevel(logging.INFO)
            
            dss_logger = logging.getLogger('src.dssms.dssms_backtester_v3')
            dss_logger.addHandler(capture_handler)
            
            try:
                # DSS実行してログキャプチャ
                dss = DSSBacktesterV3()
                result = dss.run_daily_selection(datetime(2023, 1, 15))
                
                # ログ分析
                log_analysis = {
                    'total_messages': len(log_messages),
                    'info_messages': len([m for m in log_messages if 'INFO' in str(m) or any(
                        keyword in m for keyword in ['✓', '===', '[TARGET]', '🏆']
                    )]),
                    'error_messages': len([m for m in log_messages if 'ERROR' in str(m) or '[ERROR]' in m]),
                    'warning_messages': len([m for m in log_messages if 'WARNING' in str(m) or '⚠' in m]),
                    'has_initialization_logs': any('初期化' in m for m in log_messages),
                    'has_data_fetch_logs': any('データ取得' in m for m in log_messages),
                    'has_ranking_logs': any('ランキング' in m for m in log_messages),
                    'has_selection_logs': any('選択' in m for m in log_messages)
                }
                
                # ログ品質評価
                required_log_types = [
                    log_analysis['has_initialization_logs'],
                    log_analysis['has_data_fetch_logs'],
                    log_analysis['has_ranking_logs'],
                    log_analysis['has_selection_logs'],
                    log_analysis['total_messages'] >= 10  # 最低10個のログメッセージ
                ]
                
                log_quality_score = sum(required_log_types) / len(required_log_types)
                logging_ok = log_quality_score >= 0.8  # 80%以上で合格
                
            finally:
                # ログハンドラ除去
                dss_logger.removeHandler(capture_handler)
            
            self.test_results['logging'] = {
                'log_analysis': log_analysis,
                'log_quality_score': log_quality_score,
                'logging_ok': logging_ok,
                'success': logging_ok,
                'sample_messages': log_messages[:5]  # 最初の5個のサンプル
            }
            
            if logging_ok:
                self.test_passed += 1
                logger.info(f"[OK] テスト4成功: ログ出力良好 (品質スコア: {log_quality_score:.1%})")
            else:
                logger.warning(f"⚠ テスト4要改善: ログ不十分 (品質スコア: {log_quality_score:.1%})")
                
            self.test_total += 1
            
        except Exception as e:
            logger.error(f"[ERROR] テスト4失敗: {e}")
            self.test_results['logging'] = {'error': str(e), 'success': False}
            self.test_total += 1
    
    def test_configuration_impact(self):
        """テスト5: 設定ファイル影響確認"""
        logger.info("--- テスト5: 設定ファイル影響確認 ---")
        
        try:
            # 通常動作での結果
            dss = DSSBacktesterV3()
            normal_result = dss.run_daily_selection(datetime(2023, 1, 15))
            
            # 設定ファイル確認
            config_checks = {
                'dss_initialization_success': True,
                'components_loaded': hasattr(dss, 'perfect_order_detector'),
                'ranking_system_loaded': hasattr(dss, 'ranking_system'),
                'scoring_engine_loaded': hasattr(dss, 'scoring_engine'),
                'symbol_universe_defined': len(dss.symbol_universe) > 0,
                'normal_execution_success': 'selected_symbol' in normal_result
            }
            
            passed_config_checks = sum(config_checks.values())
            total_config_checks = len(config_checks)
            
            config_ok = passed_config_checks == total_config_checks
            
            self.test_results['configuration'] = {
                'config_checks': config_checks,
                'passed': passed_config_checks,
                'total': total_config_checks,
                'config_ok': config_ok,
                'success': config_ok,
                'normal_result_symbol': normal_result.get('selected_symbol', 'N/A')
            }
            
            if config_ok:
                self.test_passed += 1
                logger.info(f"[OK] テスト5成功: 設定ファイル問題なし ({passed_config_checks}/{total_config_checks})")
            else:
                logger.warning(f"⚠ テスト5要確認: 設定に問題あり ({passed_config_checks}/{total_config_checks})")
                
            self.test_total += 1
            
        except Exception as e:
            logger.error(f"[ERROR] テスト5失敗: {e}")
            self.test_results['configuration'] = {'error': str(e), 'success': False}
            self.test_total += 1


def main():
    """Phase 4 統合テスト実行"""
    print("=== DSS Core V3 Phase 4: 統合テスト・デバッグ ===")
    
    tester = DSSPhase4IntegrationTester()
    results = tester.run_all_tests()
    
    # 結果表示
    print(f"\n=== Phase 4 統合テスト結果 ===")
    print(f"成功率: {results['success_rate']:.1f}% ({results['test_passed']}/{results['test_total']})")
    print(f"目標成功率: {results['target_success_rate']:.1f}%")
    print(f"Phase 4 完了: {'[OK] Yes' if results['phase4_completed'] else '[ERROR] No'}")
    
    # 詳細結果
    print(f"\n=== テスト詳細結果 ===")
    for test_name, test_result in results['test_results'].items():
        status = "[OK]" if test_result.get('success', False) else "[ERROR]"
        print(f"{status} {test_name}: {test_result.get('success', False)}")
    
    # 結果保存
    results_file = f"DSS_CORE_PHASE4_TEST_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n結果詳細: {results_file}")
    
    return results


if __name__ == "__main__":
    main()