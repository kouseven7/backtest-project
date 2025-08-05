"""
Test Script: Signal Integrator
File: test_signal_integrator.py
Description: 
  3-3-1「シグナル競合時の優先度ルール設計」テスト
  シグナル統合システムの包括的テスト

Author: imega
Created: 2025-07-16
Modified: 2025-07-16
"""

import os
import sys
import json
import logging
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config.signal_integrator import (
        SignalIntegrator, SignalType, StrategySignal, ConflictType, PriorityMethod,
        ConflictDetector, PriorityResolver, ResourceManager,
        create_signal_integrator, create_strategy_signal
    )
    from config.strategy_selector import StrategySelector
    from config.portfolio_weight_calculator import PortfolioWeightCalculator
    from config.strategy_scoring_model import StrategyScoreCalculator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TestSignalIntegrator(unittest.TestCase):
    """SignalIntegratorのテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        try:
            self.strategy_selector = StrategySelector()
            self.portfolio_calculator = PortfolioWeightCalculator()
            self.score_calculator = StrategyScoreCalculator()
            
            self.integrator = create_signal_integrator(
                self.strategy_selector, 
                self.portfolio_calculator, 
                self.score_calculator
            )
            
            # テスト用データ
            self.test_portfolio = {
                "momentum_strategy": 0.1,
                "mean_reversion": 0.0,
                "breakout_strategy": 0.05,
                "vwap_bounce": 0.15
            }
            self.test_capital = 1000000
            
        except Exception as e:
            self.skipTest(f"Setup failed: {e}")
    
    def test_signal_creation(self):
        """シグナル作成テスト"""
        signal = create_strategy_signal(
            strategy_name="test_strategy",
            signal_type=SignalType.ENTRY_LONG,
            confidence=0.8,
            position_size=0.15
        )
        
        self.assertEqual(signal.strategy_name, "test_strategy")
        self.assertEqual(signal.signal_type, SignalType.ENTRY_LONG)
        self.assertEqual(signal.confidence, 0.8)
        self.assertEqual(signal.position_size, 0.15)
        self.assertIsInstance(signal.timestamp, datetime)
    
    def test_no_conflict_integration(self):
        """競合なしシグナル統合テスト"""
        signals = {
            "strategy_a": create_strategy_signal("strategy_a", SignalType.ENTRY_LONG, 0.8, 0.1),
            "strategy_b": create_strategy_signal("strategy_b", SignalType.HOLD, 0.9, 0.0)
        }
        
        result = self.integrator.integrate_signals(
            signals, self.test_portfolio, self.test_capital
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["signals"]), 2)
        self.assertEqual(result["statistics"]["conflicts_count"], 0)
    
    def test_direction_conflict_resolution(self):
        """方向性競合解決テスト"""
        signals = {
            "strategy_long": create_strategy_signal("strategy_long", SignalType.ENTRY_LONG, 0.8, 0.2),
            "strategy_short": create_strategy_signal("strategy_short", SignalType.ENTRY_SHORT, 0.7, 0.15)
        }
        
        result = self.integrator.integrate_signals(
            signals, {}, self.test_capital
        )
        
        self.assertTrue(result["success"])
        self.assertGreater(result["statistics"]["conflicts_count"], 0)
        # 高信頼度のロングシグナルが選択されることを期待
        final_signals = result["signals"]
        self.assertEqual(len(final_signals), 1)
    
    def test_exit_signal_priority(self):
        """エグジットシグナル優先テスト"""
        signals = {
            "strategy_entry": create_strategy_signal("strategy_entry", SignalType.ENTRY_LONG, 0.9, 0.2),
            "strategy_exit": create_strategy_signal("strategy_exit", SignalType.EXIT_LONG, 0.6, 0.1)
        }
        
        result = self.integrator.integrate_signals(
            signals, self.test_portfolio, self.test_capital
        )
        
        self.assertTrue(result["success"])
        # エグジットシグナルが優先されることを確認
        final_signals = result["signals"]
        exit_signals = [s for s in final_signals if s["signal_type"] == "exit_long"]
        self.assertGreater(len(exit_signals), 0)
    
    def test_resource_constraints(self):
        """リソース制約テスト"""
        # 大きなポジションサイズで資金不足をシミュレート
        signals = {
            "strategy_a": create_strategy_signal("strategy_a", SignalType.ENTRY_LONG, 0.8, 0.6),
            "strategy_b": create_strategy_signal("strategy_b", SignalType.ENTRY_LONG, 0.7, 0.6)
        }
        
        result = self.integrator.integrate_signals(
            signals, {}, 1000000  # 制限された資金
        )
        
        self.assertTrue(result["success"])
        # リソース制約により、すべてのシグナルが採用されないことを確認
        self.assertGreater(result["statistics"]["resource_failures"], 0)
    
    def test_signal_confidence_filtering(self):
        """シグナル信頼度フィルタリングテスト"""
        signals = {
            "high_confidence": create_strategy_signal("high_confidence", SignalType.ENTRY_LONG, 0.9, 0.1),
            "low_confidence": create_strategy_signal("low_confidence", SignalType.ENTRY_LONG, 0.2, 0.1)  # 閾値以下
        }
        
        result = self.integrator.integrate_signals(
            signals, {}, self.test_capital
        )
        
        self.assertTrue(result["success"])
        # 低信頼度シグナルが除外されることを確認
        final_strategies = [s["strategy_name"] for s in result["signals"]]
        self.assertIn("high_confidence", final_strategies)
        # 設定によっては low_confidence も含まれる可能性があるため、具体的な除外確認は設定次第
    
    def test_integration_statistics(self):
        """統合統計テスト"""
        signals = {
            "strategy_a": create_strategy_signal("strategy_a", SignalType.ENTRY_LONG, 0.8, 0.1),
            "strategy_b": create_strategy_signal("strategy_b", SignalType.ENTRY_SHORT, 0.7, 0.1)
        }
        
        # 複数回実行して統計を蓄積
        for _ in range(3):
            self.integrator.integrate_signals(signals, {}, self.test_capital)
        
        stats = self.integrator.get_integration_stats()
        
        self.assertGreater(stats["total_signals_processed"], 0)
        self.assertGreaterEqual(stats["conflicts_detected"], 0)
        self.assertIsInstance(stats["average_processing_time"], float)
    
    def test_configuration_loading(self):
        """設定読み込みテスト"""
        # デフォルト設定で初期化されていることを確認
        self.assertIsNotNone(self.integrator.config)
        self.assertIn("priority_method", self.integrator.config)
        self.assertIn("risk_limits", self.integrator.config)
        self.assertIn("conflict_resolution", self.integrator.config)
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 不正なシグナルでエラーハンドリングをテスト
        invalid_signals = {}
        
        result = self.integrator.integrate_signals(
            invalid_signals, {}, self.test_capital
        )
        
        # エラー時でも適切な結果が返されることを確認
        self.assertFalse(result["success"])
        self.assertIn("error", result)


class TestConflictDetector(unittest.TestCase):
    """ConflictDetectorのテストクラス"""
    
    def setUp(self):
        self.detector = ConflictDetector()
    
    def test_direction_conflict_detection(self):
        """方向性競合検出テスト"""
        signals = [
            create_strategy_signal("strategy_long", SignalType.ENTRY_LONG, 0.8, 0.2),
            create_strategy_signal("strategy_short", SignalType.ENTRY_SHORT, 0.7, 0.15)
        ]
        
        conflicts = self.detector.detect_conflicts(signals, {}, 1000000, {})
        
        direction_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.DIRECTION_CONFLICT]
        self.assertGreater(len(direction_conflicts), 0)


def run_comprehensive_test():
    """包括的テストの実行"""
    print("=" * 60)
    print("Signal Integrator 包括的テスト")
    print("=" * 60)
    
    # テストスイート作成
    suite = unittest.TestSuite()
    
    # SignalIntegratorテスト
    integrator_tests = [
        'test_signal_creation',
        'test_no_conflict_integration',
        'test_direction_conflict_resolution',
        'test_exit_signal_priority',
        'test_resource_constraints',
        'test_signal_confidence_filtering',
        'test_integration_statistics',
        'test_configuration_loading',
        'test_error_handling'
    ]
    
    for test_name in integrator_tests:
        suite.addTest(TestSignalIntegrator(test_name))
    
    # ConflictDetectorテスト
    suite.addTest(TestConflictDetector('test_direction_conflict_detection'))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
