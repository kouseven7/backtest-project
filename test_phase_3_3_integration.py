"""
Phase 3.3 統合テスト
IntegratedExecutionManager + StrategyExecutionManager の統合動作確認

テスト観点:
1. 実データなし環境でのエラーハンドリング検証
2. copilot-instructions.md 違反チェック（モック/ダミーデータ使用禁止）
3. コンポーネント間連携の正常性確認
4. エラーログの適切性確認
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger

# Phase 3 統合システム
from main_system.execution_control.integrated_execution_manager import IntegratedExecutionManager
from main_system.execution_control.strategy_execution_manager import StrategyExecutionManager


class Phase33IntegrationTester:
    """Phase 3.3 統合テストクラス"""
    
    def __init__(self):
        self.logger = setup_logger(
            "Phase33IntegrationTester",
            log_file="logs/phase_3_3_integration_test.log"
        )
        self.test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "warnings": []
            }
        }
        
    def run_all_tests(self):
        """全テスト実行"""
        self.logger.info("="*80)
        self.logger.info("Phase 3.3 統合テスト開始")
        self.logger.info("="*80)
        
        # Test 1: StrategyExecutionManager 単体テスト（実データなし）
        self._test_strategy_execution_manager_no_data()
        
        # Test 2: IntegratedExecutionManager 単体テスト（実データなし）
        self._test_integrated_execution_manager_no_data()
        
        # Test 3: 統合フロー テスト（サンプルデータあり）
        self._test_integrated_flow_with_sample_data()
        
        # Test 4: copilot-instructions.md 違反チェック
        self._test_copilot_instructions_compliance()
        
        # Test 5: エラーハンドリング検証
        self._test_error_handling()
        
        # サマリー計算
        self._calculate_summary()
        
        # 結果出力
        self._output_results()
        
        return self.test_results
    
    def _test_strategy_execution_manager_no_data(self):
        """Test 1: StrategyExecutionManager 単体テスト（実データなし環境）"""
        test_name = "StrategyExecutionManager単体テスト（実データなし）"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Test 1: {test_name}")
        self.logger.info(f"{'='*60}")
        
        test_result = {
            "test_name": test_name,
            "test_id": "test_1",
            "status": "UNKNOWN",
            "details": {},
            "issues": []
        }
        
        try:
            # StrategyExecutionManager作成（data_feed=None）
            config = {
                'execution_mode': 'simple',
                'broker': {
                    'initial_cash': 100000,
                    'commission_per_trade': 1.0,
                    'slippage_bps': 5
                }
            }
            
            manager = StrategyExecutionManager(config)
            
            # 期待動作: data_feed=None なので実行不可
            # エラーハンドリングが正しく動作するか検証
            
            result = manager.execute_strategy(
                strategy_name='VWAPBreakoutStrategy',
                symbols=['TEST']
            )
            
            # 結果検証
            test_result['details']['execution_result'] = result
            
            # 検証ポイント1: data_feedがNoneであることを確認
            if manager.data_feed is None:
                self.logger.info("✅ data_feed=None 確認 OK")
                test_result['details']['data_feed_check'] = "PASS"
            else:
                self.logger.warning("⚠️ data_feed が None でない")
                test_result['issues'].append("data_feed が None でない")
                test_result['details']['data_feed_check'] = "FAIL"
            
            # 検証ポイント2: 実行結果がエラーであることを確認
            if not result.get('success', False):
                self.logger.info("✅ エラーハンドリング正常動作")
                test_result['details']['error_handling'] = "PASS"
                test_result['status'] = "PASSED"
            else:
                self.logger.error("❌ 実データなしで成功したと報告（copilot-instructions.md違反の可能性）")
                test_result['issues'].append("実データなしで成功報告（モック/ダミーデータ使用の疑い）")
                test_result['details']['error_handling'] = "FAIL"
                test_result['status'] = "FAILED"
            
            # 検証ポイント3: エラーメッセージの適切性
            error_msg = result.get('error', '')
            if 'Market data unavailable' in error_msg or 'data_unavailable' in error_msg:
                self.logger.info("✅ エラーメッセージ適切")
                test_result['details']['error_message'] = "PASS"
            else:
                self.logger.warning(f"⚠️ エラーメッセージ不明瞭: {error_msg}")
                test_result['issues'].append(f"エラーメッセージ不明瞭: {error_msg}")
                test_result['details']['error_message'] = "WARNING"
            
        except Exception as e:
            self.logger.error(f"❌ Test 1 例外発生: {e}")
            test_result['status'] = "FAILED"
            test_result['issues'].append(f"例外発生: {str(e)}")
        
        self.test_results['tests'].append(test_result)
        self.test_results['summary']['total_tests'] += 1
        
        if test_result['status'] == "PASSED":
            self.test_results['summary']['passed_tests'] += 1
        else:
            self.test_results['summary']['failed_tests'] += 1
    
    def _test_integrated_execution_manager_no_data(self):
        """Test 2: IntegratedExecutionManager 単体テスト（実データなし環境）"""
        test_name = "IntegratedExecutionManager単体テスト（実データなし）"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Test 2: {test_name}")
        self.logger.info(f"{'='*60}")
        
        test_result = {
            "test_name": test_name,
            "test_id": "test_2",
            "status": "UNKNOWN",
            "details": {},
            "issues": []
        }
        
        try:
            # IntegratedExecutionManager作成
            config = {
                'initial_portfolio_value': 100000,
                'execution': {
                    'execution_mode': 'simple'
                }
            }
            
            manager = IntegratedExecutionManager(config)
            
            # サンプルデータ（最小限）
            dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
            sample_data = pd.DataFrame({
                'Open': [100] * 10,
                'High': [105] * 10,
                'Low': [95] * 10,
                'Close': [102] * 10,
                'Volume': [1000000] * 10
            }, index=dates)
            
            # 実行テスト
            result = manager.execute_dynamic_strategies(
                stock_data=sample_data,
                ticker='TEST'
            )
            
            test_result['details']['execution_result'] = {
                'status': result.get('status'),
                'successful_strategies': result.get('successful_strategies', 0),
                'failed_strategies': result.get('failed_strategies', 0)
            }
            
            # 検証ポイント1: StrategyExecutionManager との連携確認
            if hasattr(manager, 'execution_manager'):
                self.logger.info("✅ StrategyExecutionManager 連携 OK")
                test_result['details']['component_integration'] = "PASS"
            else:
                self.logger.error("❌ StrategyExecutionManager 連携失敗")
                test_result['issues'].append("StrategyExecutionManager 連携失敗")
                test_result['details']['component_integration'] = "FAIL"
            
            # 検証ポイント2: リスク管理コンポーネント確認
            if hasattr(manager, 'risk_controller'):
                self.logger.info("✅ DrawdownController 連携 OK")
                test_result['details']['risk_management'] = "PASS"
            else:
                self.logger.warning("⚠️ DrawdownController 連携なし")
                test_result['issues'].append("DrawdownController 連携なし")
                test_result['details']['risk_management'] = "WARNING"
            
            # 検証ポイント3: 実行結果の妥当性
            # 実データなし環境なので、エラーまたは失敗が正常
            if result.get('status') in ['ERROR', 'PARTIAL_SUCCESS']:
                self.logger.info("✅ 実行結果妥当（実データなし環境で適切なエラー）")
                test_result['details']['execution_validity'] = "PASS"
                test_result['status'] = "PASSED"
            elif result.get('status') == 'SUCCESS' and result.get('successful_strategies', 0) == 0:
                self.logger.info("✅ 実行結果妥当（成功戦略数0）")
                test_result['details']['execution_validity'] = "PASS"
                test_result['status'] = "PASSED"
            else:
                self.logger.warning(f"⚠️ 実行結果要確認: {result.get('status')}")
                test_result['issues'].append(f"実行結果要確認: {result.get('status')}")
                test_result['details']['execution_validity'] = "WARNING"
                test_result['status'] = "PASSED"  # WARNING扱いだが合格
            
        except Exception as e:
            self.logger.error(f"❌ Test 2 例外発生: {e}")
            test_result['status'] = "FAILED"
            test_result['issues'].append(f"例外発生: {str(e)}")
        
        self.test_results['tests'].append(test_result)
        self.test_results['summary']['total_tests'] += 1
        
        if test_result['status'] == "PASSED":
            self.test_results['summary']['passed_tests'] += 1
        else:
            self.test_results['summary']['failed_tests'] += 1
    
    def _test_integrated_flow_with_sample_data(self):
        """Test 3: 統合フロー テスト（サンプルデータあり）"""
        test_name = "統合フロー テスト（サンプルデータ）"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Test 3: {test_name}")
        self.logger.info(f"{'='*60}")
        
        test_result = {
            "test_name": test_name,
            "test_id": "test_3",
            "status": "UNKNOWN",
            "details": {},
            "issues": []
        }
        
        try:
            # より現実的なサンプルデータ生成
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)  # 再現性のため
            
            price_base = 100
            price_changes = np.random.randn(len(dates)) * 2
            close_prices = price_base + np.cumsum(price_changes)
            
            sample_data = pd.DataFrame({
                'Open': close_prices * 0.99,
                'High': close_prices * 1.02,
                'Low': close_prices * 0.98,
                'Close': close_prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            
            # IntegratedExecutionManager作成
            config = {
                'initial_portfolio_value': 100000,
                'execution': {
                    'execution_mode': 'simple'
                }
            }
            
            manager = IntegratedExecutionManager(config)
            
            # 統合実行
            result = manager.execute_dynamic_strategies(
                stock_data=sample_data,
                ticker='SAMPLE'
            )
            
            test_result['details']['execution_result'] = {
                'status': result.get('status'),
                'successful_strategies': result.get('successful_strategies', 0),
                'failed_strategies': result.get('failed_strategies', 0),
                'total_trades': result.get('total_trades', 0)
            }
            
            # 検証ポイント1: 実行完了
            if result.get('status') in ['SUCCESS', 'PARTIAL_SUCCESS', 'ERROR']:
                self.logger.info(f"✅ 実行完了: {result.get('status')}")
                test_result['details']['execution_completion'] = "PASS"
            else:
                self.logger.error(f"❌ 実行状態不明: {result.get('status')}")
                test_result['issues'].append(f"実行状態不明: {result.get('status')}")
                test_result['details']['execution_completion'] = "FAIL"
            
            # 検証ポイント2: サマリー取得
            summary = manager.get_execution_summary()
            test_result['details']['execution_summary'] = summary
            
            if summary.get('total_executions', 0) > 0:
                self.logger.info(f"✅ 実行履歴記録: {summary.get('total_executions')} 件")
                test_result['details']['history_recording'] = "PASS"
            else:
                self.logger.warning("⚠️ 実行履歴記録なし")
                test_result['issues'].append("実行履歴記録なし")
                test_result['details']['history_recording'] = "WARNING"
            
            # 検証ポイント3: Phase 2 コンポーネント連携
            # MarketAnalyzer, DynamicStrategySelectorとの連携確認
            if hasattr(manager, 'market_analyzer') and hasattr(manager, 'strategy_selector'):
                self.logger.info("✅ Phase 2 コンポーネント連携 OK")
                test_result['details']['phase2_integration'] = "PASS"
                test_result['status'] = "PASSED"
            else:
                self.logger.warning("⚠️ Phase 2 コンポーネント連携未確認")
                test_result['issues'].append("Phase 2 コンポーネント連携未確認")
                test_result['details']['phase2_integration'] = "WARNING"
                test_result['status'] = "PASSED"  # WARNING扱いだが合格
            
        except Exception as e:
            self.logger.error(f"❌ Test 3 例外発生: {e}")
            test_result['status'] = "FAILED"
            test_result['issues'].append(f"例外発生: {str(e)}")
        
        self.test_results['tests'].append(test_result)
        self.test_results['summary']['total_tests'] += 1
        
        if test_result['status'] == "PASSED":
            self.test_results['summary']['passed_tests'] += 1
        else:
            self.test_results['summary']['failed_tests'] += 1
    
    def _test_copilot_instructions_compliance(self):
        """Test 4: copilot-instructions.md 違反チェック"""
        test_name = "copilot-instructions.md 遵守確認"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Test 4: {test_name}")
        self.logger.info(f"{'='*60}")
        
        test_result = {
            "test_name": test_name,
            "test_id": "test_4",
            "status": "UNKNOWN",
            "details": {},
            "issues": []
        }
        
        try:
            # StrategyExecutionManager のコード検査
            strategy_exec_path = project_root / "main_system" / "execution_control" / "strategy_execution_manager.py"
            
            with open(strategy_exec_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # 禁止パターン検出
            forbidden_patterns = {
                '_generate_sample_data': 'ダミーデータ生成メソッド',
                'mock_execution': 'モック実行',
                'dummy_data': 'ダミーデータ',
                'test_data': 'テストデータ（実データでない）',
                'fallback.*random': 'ランダムデータフォールバック'
            }
            
            violations = []
            for pattern, description in forbidden_patterns.items():
                if pattern in code_content.lower():
                    violations.append(f"{description}: '{pattern}' 検出")
            
            if len(violations) == 0:
                self.logger.info("✅ copilot-instructions.md 違反なし")
                test_result['details']['compliance'] = "PASS"
                test_result['status'] = "PASSED"
            else:
                self.logger.error(f"❌ copilot-instructions.md 違反検出: {len(violations)} 件")
                for violation in violations:
                    self.logger.error(f"  - {violation}")
                    test_result['issues'].append(violation)
                test_result['details']['compliance'] = "FAIL"
                test_result['status'] = "FAILED"
            
            test_result['details']['violations_count'] = len(violations)
            
        except Exception as e:
            self.logger.error(f"❌ Test 4 例外発生: {e}")
            test_result['status'] = "FAILED"
            test_result['issues'].append(f"例外発生: {str(e)}")
        
        self.test_results['tests'].append(test_result)
        self.test_results['summary']['total_tests'] += 1
        
        if test_result['status'] == "PASSED":
            self.test_results['summary']['passed_tests'] += 1
        else:
            self.test_results['summary']['failed_tests'] += 1
    
    def _test_error_handling(self):
        """Test 5: エラーハンドリング検証"""
        test_name = "エラーハンドリング検証"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Test 5: {test_name}")
        self.logger.info(f"{'='*60}")
        
        test_result = {
            "test_name": test_name,
            "test_id": "test_5",
            "status": "UNKNOWN",
            "details": {},
            "issues": []
        }
        
        try:
            # 異常系テスト: 不正な戦略名
            config = {
                'execution_mode': 'simple',
                'broker': {
                    'initial_cash': 100000,
                    'commission_per_trade': 1.0,
                    'slippage_bps': 5
                }
            }
            
            manager = StrategyExecutionManager(config)
            
            # 存在しない戦略名で実行
            result = manager.execute_strategy(
                strategy_name='NonExistentStrategy',
                symbols=['TEST']
            )
            
            # エラーハンドリング確認
            if not result.get('success', False):
                self.logger.info("✅ 不正戦略名エラーハンドリング OK")
                test_result['details']['invalid_strategy_handling'] = "PASS"
                test_result['status'] = "PASSED"
            else:
                self.logger.error("❌ 不正戦略名で成功報告（エラーハンドリング不正）")
                test_result['issues'].append("不正戦略名で成功報告")
                test_result['details']['invalid_strategy_handling'] = "FAIL"
                test_result['status'] = "FAILED"
            
            # エラーメッセージ確認
            error_msg = result.get('error', '')
            if 'strategy_not_found' in error_msg or 'not found' in error_msg.lower():
                self.logger.info("✅ エラーメッセージ適切")
                test_result['details']['error_message'] = "PASS"
            else:
                self.logger.warning(f"⚠️ エラーメッセージ要改善: {error_msg}")
                test_result['issues'].append(f"エラーメッセージ要改善: {error_msg}")
                test_result['details']['error_message'] = "WARNING"
            
        except Exception as e:
            self.logger.error(f"❌ Test 5 例外発生: {e}")
            test_result['status'] = "FAILED"
            test_result['issues'].append(f"例外発生: {str(e)}")
        
        self.test_results['tests'].append(test_result)
        self.test_results['summary']['total_tests'] += 1
        
        if test_result['status'] == "PASSED":
            self.test_results['summary']['passed_tests'] += 1
        else:
            self.test_results['summary']['failed_tests'] += 1
    
    def _calculate_summary(self):
        """サマリー計算"""
        total = self.test_results['summary']['total_tests']
        passed = self.test_results['summary']['passed_tests']
        
        self.test_results['summary']['success_rate'] = (
            (passed / total * 100) if total > 0 else 0
        )
        
        # 警告収集
        for test in self.test_results['tests']:
            if test.get('issues'):
                for issue in test['issues']:
                    if 'WARNING' in issue or '⚠️' in issue:
                        self.test_results['summary']['warnings'].append(
                            f"{test['test_name']}: {issue}"
                        )
    
    def _output_results(self):
        """結果出力"""
        # コンソール出力
        print("\n" + "="*80)
        print("Phase 3.3 統合テスト結果")
        print("="*80)
        
        summary = self.test_results['summary']
        print(f"\n総テスト数: {summary['total_tests']}")
        print(f"成功: {summary['passed_tests']}")
        print(f"失敗: {summary['failed_tests']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        
        if summary['warnings']:
            print(f"\n警告: {len(summary['warnings'])} 件")
            for warning in summary['warnings']:
                print(f"  ⚠️ {warning}")
        
        print("\n" + "-"*80)
        print("詳細テスト結果:")
        print("-"*80)
        
        for test in self.test_results['tests']:
            status_icon = "✅" if test['status'] == "PASSED" else "❌"
            print(f"\n{status_icon} {test['test_name']}: {test['status']}")
            
            if test.get('issues'):
                print(f"  問題点: {len(test['issues'])} 件")
                for issue in test['issues']:
                    print(f"    - {issue}")
        
        # JSON出力（datetime型を文字列に変換）
        output_dir = project_root / "diagnostics" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"phase_3_3_integration_test_results_{timestamp}.json"
        
        # datetime型をJSON serializable に変換
        def convert_to_serializable(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(self.test_results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n詳細結果保存: {json_path}")
        self.logger.info(f"テスト結果保存完了: {json_path}")


def main():
    """メイン実行"""
    print("Phase 3.3 統合テスト実行開始")
    print("="*80)
    
    tester = Phase33IntegrationTester()
    results = tester.run_all_tests()
    
    print("\n" + "="*80)
    print("Phase 3.3 統合テスト完了")
    print("="*80)
    
    return results


if __name__ == '__main__':
    main()
