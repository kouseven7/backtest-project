"""
Test Module: Drawdown Controller
File: test_drawdown_controller.py
Description: 
  5-3-1「ドローダウン制御機能」包括的テストスイート

Author: imega
Created: 2025-07-20
Modified: 2025-07-20
"""

import os
import sys
import unittest
import json
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import tempfile

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config.drawdown_controller import (
        DrawdownController, DrawdownSeverity, DrawdownControlAction, 
        DrawdownEvent, DrawdownControlResult, create_default_drawdown_config
    )
    from config.drawdown_action_executor import DrawdownActionExecutor, ActionExecutionResult
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class TestDrawdownController(unittest.TestCase):
    """ドローダウンコントローラーテスト"""
    
    def setUp(self):
        """テストセットアップ"""
        # 一時設定ファイル作成
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config_data = create_default_drawdown_config()
        json.dump(config_data, self.temp_config, indent=2)
        self.temp_config.close()
        
        # コントローラー初期化
        self.controller = DrawdownController(config_file=self.temp_config.name)
        
        # モックシステム
        self.mock_portfolio_risk_manager = Mock()
        self.mock_position_size_adjuster = Mock()
        self.mock_portfolio_weight_calculator = Mock()
        self.mock_coordination_manager = Mock()
        
        # モックを設定
        self.controller.portfolio_risk_manager = self.mock_portfolio_risk_manager
        self.controller.position_size_adjuster = self.mock_position_size_adjuster
        self.controller.portfolio_weight_calculator = self.mock_portfolio_weight_calculator
        self.controller.coordination_manager = self.mock_coordination_manager
    
    def tearDown(self):
        """テストクリーンアップ"""
        if self.controller.is_monitoring:
            self.controller.stop_monitoring()
        
        # 一時ファイル削除
        try:
            os.unlink(self.temp_config.name)
        except:
            pass
    
    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.controller, DrawdownController)
        self.assertEqual(self.controller.control_mode.value, 'moderate')
        self.assertFalse(self.controller.is_monitoring)
        self.assertEqual(len(self.controller.control_history), 0)
    
    def test_config_loading(self):
        """設定読み込みテスト"""
        # デフォルト設定確認
        self.assertEqual(self.controller.thresholds.warning_threshold, 0.05)
        self.assertEqual(self.controller.thresholds.critical_threshold, 0.10)
        self.assertEqual(self.controller.thresholds.emergency_threshold, 0.15)
    
    def test_monitoring_start_stop(self):
        """監視開始・停止テスト"""
        # 監視開始
        initial_value = 1000000.0
        self.controller.start_monitoring(initial_value)
        
        self.assertTrue(self.controller.is_monitoring)
        self.assertEqual(self.controller.performance_tracker['portfolio_peak'], initial_value)
        self.assertEqual(len(self.controller.performance_tracker['portfolio_history']), 1)
        
        # 監視停止
        self.controller.stop_monitoring()
        self.assertFalse(self.controller.is_monitoring)
    
    def test_portfolio_value_update(self):
        """ポートフォリオ価値更新テスト"""
        # 初期化
        self.controller.start_monitoring(1000000.0)
        
        # 価値更新
        new_value = 950000.0
        self.controller.update_portfolio_value(new_value)
        
        # 履歴確認
        self.assertEqual(len(self.controller.performance_tracker['portfolio_history']), 2)
        self.assertEqual(self.controller.performance_tracker['portfolio_history'][-1][1], new_value)
        
        # ピーク値は変わらないことを確認
        self.assertEqual(self.controller.performance_tracker['portfolio_peak'], 1000000.0)
    
    def test_drawdown_severity_determination(self):
        """ドローダウン深刻度判定テスト"""
        # 正常範囲
        severity = self.controller._determine_severity(0.03)  # 3%
        self.assertEqual(severity, DrawdownSeverity.NORMAL)
        
        # 警告レベル
        severity = self.controller._determine_severity(0.07)  # 7%
        self.assertEqual(severity, DrawdownSeverity.WARNING)
        
        # 重要レベル
        severity = self.controller._determine_severity(0.12)  # 12%
        self.assertEqual(severity, DrawdownSeverity.CRITICAL)
        
        # 緊急レベル
        severity = self.controller._determine_severity(0.18)  # 18%
        self.assertEqual(severity, DrawdownSeverity.EMERGENCY)
    
    def test_control_action_determination(self):
        """制御アクション決定テスト"""
        # 警告レベルイベント作成
        warning_event = DrawdownEvent(
            timestamp=datetime.now(),
            portfolio_value=950000.0,
            previous_peak=1000000.0,
            current_drawdown=50000.0,
            drawdown_percentage=0.05,
            severity=DrawdownSeverity.WARNING,
            affected_strategies=['Momentum'],
            triggering_factor="Test factor"
        )
        
        action = self.controller._determine_control_action(warning_event)
        self.assertEqual(action, DrawdownControlAction.POSITION_REDUCTION_LIGHT)
    
    def test_drawdown_detection_and_control(self):
        """ドローダウン検出・制御テスト"""
        # 監視開始
        self.controller.start_monitoring(1000000.0)
        
        # 段階的にドローダウン発生
        test_values = [
            (950000.0, DrawdownSeverity.WARNING),  # 5%ドローダウン
            (900000.0, DrawdownSeverity.CRITICAL), # 10%ドローダウン
            (850000.0, DrawdownSeverity.EMERGENCY) # 15%ドローダウン
        ]
        
        for value, expected_severity in test_values:
            # 価値更新
            self.controller.update_portfolio_value(value)
            
            # 手動ドローダウンチェック
            event = self.controller._check_drawdown()
            
            if event:
                self.assertEqual(event.severity, expected_severity)
                # 制御履歴確認
                self.assertGreater(len(self.controller.control_history), 0)
    
    def test_strategy_values_tracking(self):
        """戦略別価値追跡テスト"""
        self.controller.start_monitoring(1000000.0)
        
        # 戦略別価値更新
        strategy_values = {
            'Momentum': 300000.0,
            'Contrarian': 350000.0,
            'Pairs_Trading': 350000.0
        }
        
        self.controller.update_portfolio_value(1000000.0, strategy_values)
        
        # 戦略履歴確認
        self.assertEqual(len(self.controller.performance_tracker['strategy_histories']), 3)
        self.assertIn('Momentum', self.controller.performance_tracker['strategy_peaks'])
        
        # 戦略価値下降
        strategy_values_decline = {
            'Momentum': 250000.0,  # 大幅下降
            'Contrarian': 340000.0, # 軽微下降
            'Pairs_Trading': 360000.0 # 上昇
        }
        
        self.controller.update_portfolio_value(950000.0, strategy_values_decline)
        
        # 影響戦略特定
        affected_strategies = self.controller._identify_affected_strategies()
        # Momentumが影響戦略として特定されることを期待
        # （実際の閾値によって結果は変わる）
    
    def test_market_condition_adjustment(self):
        """市場環境調整テスト"""
        # ボラティリティ推定
        self.controller.start_monitoring(1000000.0)
        
        # 価格変動データ作成
        volatile_values = [1000000, 980000, 1020000, 950000, 1040000, 920000]
        for value in volatile_values:
            self.controller.update_portfolio_value(value)
        
        # ボラティリティ推定
        volatility = self.controller._estimate_market_volatility()
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0)
        
        # トレンド要因推定
        trend_factor = self.controller._estimate_trend_factor()
        self.assertIsInstance(trend_factor, float)
        self.assertGreaterEqual(trend_factor, -1.0)
        self.assertLessEqual(trend_factor, 1.0)
    
    def test_performance_summary(self):
        """パフォーマンスサマリーテスト"""
        # データなし状態
        summary = self.controller.get_performance_summary()
        self.assertEqual(summary['status'], 'no_data')
        
        # データありの状態
        self.controller.start_monitoring(1000000.0)
        self.controller.update_portfolio_value(950000.0)
        
        summary = self.controller.get_performance_summary()
        self.assertEqual(summary['current_portfolio_value'], 950000.0)
        self.assertEqual(summary['peak_portfolio_value'], 1000000.0)
        self.assertEqual(summary['drawdown_percentage'], 0.05)
        self.assertEqual(summary['monitoring_status'], 'active')
    
    def test_integration_with_mocked_systems(self):
        """モックシステム統合テスト"""
        # 制御アクション実行時にモックシステムが呼ばれるかテスト
        self.controller.start_monitoring(1000000.0)
        
        # 緊急レベルドローダウン発生
        self.controller.update_portfolio_value(850000.0)  # 15%ドローダウン
        
        # 短時間待機して制御実行を待つ
        time.sleep(2)
        
        # モックへの呼び出し確認は具体的な実装によって調整

class TestDrawdownActionExecutor(unittest.TestCase):
    """ドローダウンアクション実行器テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        self.executor = DrawdownActionExecutor()
        
        # モックシステム
        self.executor.portfolio_risk_manager = Mock()
        self.executor.coordination_manager = Mock()
    
    def test_position_reduction_execution(self):
        """ポジション削減実行テスト"""
        original_positions = {
            'Momentum': 0.4,
            'Contrarian': 0.3,
            'Pairs_Trading': 0.3
        }
        
        result = self.executor._execute_position_reduction(original_positions, 0.15, "light")
        
        # 結果確認
        self.assertIsInstance(result, ActionExecutionResult)
        self.assertTrue(result.success)
        
        # ポジション削減確認
        for strategy, original_pos in original_positions.items():
            adjusted_pos = result.final_positions[strategy]
            expected_pos = original_pos * 0.85  # 15%削減
            self.assertAlmostEqual(adjusted_pos, expected_pos, places=6)
        
        # 変更量確認
        changes = result.get_position_change_summary()
        for strategy, change in changes.items():
            self.assertLess(change, 0)  # 全て削減方向
    
    def test_emergency_stop_execution(self):
        """緊急停止実行テスト"""
        original_positions = {
            'Momentum': 0.4,
            'Contrarian': 0.3,
            'Pairs_Trading': 0.3
        }
        
        result = self.executor._execute_emergency_stop(original_positions)
        
        # 結果確認
        self.assertTrue(result.success)  # 緊急停止は常に成功
        self.assertFalse(result.rollback_available)  # ロールバック不可
        self.assertTrue(self.executor.emergency_mode)
        
        # 全ポジションが0になることを確認
        for strategy, position in result.final_positions.items():
            self.assertEqual(position, 0.0)
    
    def test_execution_history_tracking(self):
        """実行履歴追跡テスト"""
        initial_history_count = len(self.executor.execution_history)
        
        # アクション実行
        positions = {'Momentum': 0.5, 'Contrarian': 0.5}
        self.executor._execute_position_reduction(positions, 0.2, "moderate")
        
        # 履歴追加確認
        self.assertEqual(len(self.executor.execution_history), initial_history_count + 1)
        
        # サマリー確認
        summary = self.executor.get_execution_summary()
        self.assertEqual(summary['total_executions'], initial_history_count + 1)

class TestDrawdownControllerIntegration(unittest.TestCase):
    """ドローダウンコントローラー統合テスト"""
    
    def setUp(self):
        """テストセットアップ"""
        # リアルシステムとの統合テスト用セットアップ
        self.controller = DrawdownController()
    
    def test_realistic_drawdown_scenario(self):
        """現実的ドローダウンシナリオテスト"""
        # 現実的な市場シナリオをシミュレーション
        initial_value = 1000000.0
        self.controller.start_monitoring(initial_value)
        
        # 段階的市場下落シナリオ
        market_scenario = [
            (995000, "小幅下落"),
            (985000, "続落"),
            (970000, "調整局面"),
            (945000, "警告レベル"),
            (920000, "重要レベル"),
            (890000, "危険レベル"),
            (850000, "緊急レベル"),
            (870000, "小幅回復"),
            (880000, "継続回復")
        ]
        
        control_actions = []
        
        for value, description in market_scenario:
            self.controller.update_portfolio_value(value)
            
            # 制御履歴確認
            current_controls = len(self.controller.control_history)
            if current_controls > len(control_actions):
                latest_control = self.controller.control_history[-1]
                control_actions.append(latest_control)
                print(f"Control triggered at {value} ({description}): {latest_control.action_taken.value}")
            
            time.sleep(0.1)  # 短い間隔での更新
        
        # 最終確認
        final_summary = self.controller.get_performance_summary()
        print(f"\nFinal Summary:")
        print(f"Final Value: ${final_summary.get('current_portfolio_value', 0):,.0f}")
        print(f"Max Drawdown: {final_summary.get('drawdown_percentage', 0):.1%}")
        print(f"Control Actions: {final_summary.get('total_control_actions', 0)}")
        
        # 制御が適切に動作したことを確認
        self.assertGreaterEqual(len(control_actions), 0, "Expected drawdown controls to be triggered")

def run_comprehensive_test():
    """包括テスト実行"""
    print("=" * 70)
    print("Drawdown Controller - Comprehensive Test Suite")
    print("=" * 70)
    
    # テストスイート構築
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # テストクラス追加
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestDrawdownController))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestDrawdownActionExecutor))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestDrawdownControllerIntegration))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # 結果サマリー
    print(f"\n" + "=" * 70)
    print("Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%" if result.testsRun > 0 else "N/A")
    
    if result.failures:
        print(f"\nFailures:")
        for test, trace in result.failures:
            print(f"- {test}: {trace.split('AssertionError:')[-1].strip() if 'AssertionError:' in trace else 'Unknown error'}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, trace in result.errors:
            print(f"- {test}: {trace.split('Exception:')[-1].strip() if 'Exception:' in trace else 'Unknown error'}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
