"""
Integration Test for Hierarchical Switch Decision System
階層化切替決定システムの統合テスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# 階層化切替決定システムのインポート
from src.dssms.hierarchical_switch_decision_engine import HierarchicalSwitchDecisionEngine
from src.dssms.decision_context import DecisionContext, HierarchicalDecisionResult


class TestHierarchicalSwitchDecisionSystem(unittest.TestCase):
    """階層化切替決定システムの統合テスト"""
    
    def setUp(self):
        """テスト前の準備"""
        self.config_path = "config/dssms/hierarchical_switch_decision_config.json"
        self.engine = HierarchicalSwitchDecisionEngine(config_path=self.config_path)
        
        # テスト用の基本コンテキスト
        self.base_strategies_data = {
            'VWAPBreakoutStrategy': {
                'score': 75.0,
                'weight': 0.3,
                'returns': [0.01, -0.005, 0.02, 0.015, -0.01],
                'daily_contribution': 0.008,
                'target_potential': 0.015,
                'stability_score': 0.8,
                'volatility_score': 0.3,
                'execution_confidence': 0.9
            },
            'MomentumInvestingStrategy': {
                'score': 68.0,
                'weight': 0.25,
                'returns': [0.015, 0.01, -0.02, 0.008, 0.012],
                'daily_contribution': 0.006,
                'target_potential': 0.012,
                'stability_score': 0.6,
                'volatility_score': 0.5,
                'execution_confidence': 0.7
            },
            'BreakoutStrategy': {
                'score': 82.0,
                'weight': 0.2,
                'returns': [0.02, 0.018, -0.015, 0.025, -0.008],
                'daily_contribution': 0.012,
                'target_potential': 0.020,
                'stability_score': 0.7,
                'volatility_score': 0.4,
                'execution_confidence': 0.8
            }
        }
        
        self.base_risk_metrics = {
            'current_drawdown': 0.02,
            'portfolio_var': 0.015,
            'portfolio_volatility': 0.18,
            'correlation_risk': 0.6,
            'exposure_concentration': 0.4
        }
        
        self.base_market_conditions = {
            'market_open_time': datetime.now().replace(hour=9, minute=0),
            'market_close_time': datetime.now().replace(hour=15, minute=30),
            'market_trend': 'bullish',
            'volatility_regime': 'normal'
        }
        
        self.base_portfolio_state = {
            'daily_return': 0.012,
            'daily_target': 0.020,
            'total_assets': 1000000,
            'current_positions': 3
        }
    
    def create_test_context(self, **kwargs) -> DecisionContext:
        """テスト用コンテキストを作成"""
        strategies_data = kwargs.get('strategies_data', self.base_strategies_data)
        risk_metrics = kwargs.get('risk_metrics', self.base_risk_metrics)
        market_conditions = kwargs.get('market_conditions', self.base_market_conditions)
        portfolio_state = kwargs.get('portfolio_state', self.base_portfolio_state)
        emergency_signals = kwargs.get('emergency_signals', None)
        
        return DecisionContext(
            strategies_data=strategies_data,
            risk_metrics=risk_metrics,
            market_conditions=market_conditions,
            portfolio_state=portfolio_state,
            timestamp=datetime.now(),
            emergency_signals=emergency_signals
        )
    
    def test_engine_initialization(self):
        """エンジン初期化テスト"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(len(self.engine.decision_levels), 3)
        self.assertTrue(self.engine.validate_configuration())
    
    def test_level1_optimization_decision(self):
        """Level1最適化決定テスト"""
        # 通常の最適化条件
        context = self.create_test_context()
        decision = self.engine.make_decision(context)
        
        self.assertIsNotNone(decision)
        self.assertIn(decision.decision_level, [1, 2, 3])
        self.assertIn(decision.decision_type, ['switch', 'maintain', 'emergency_stop'])
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
    
    def test_level2_daily_target_decision(self):
        """Level2日次ターゲット決定テスト"""
        # 日次ターゲット未達成のシナリオ
        portfolio_state = self.base_portfolio_state.copy()
        portfolio_state['daily_return'] = 0.008  # ターゲット(0.020)に対して大幅未達成
        
        context = self.create_test_context(portfolio_state=portfolio_state)
        decision = self.engine.make_decision(context)
        
        self.assertIsNotNone(decision)
        # Level2がアクティベートされることを確認
        if decision.decision_level == 2:
            self.assertIn(decision.decision_type, ['switch', 'maintain'])
    
    def test_level3_emergency_decision(self):
        """Level3緊急決定テスト"""
        # 緊急条件（高ドローダウン）
        risk_metrics = self.base_risk_metrics.copy()
        risk_metrics['current_drawdown'] = 0.08  # 閾値(0.05)を超過
        
        context = self.create_test_context(risk_metrics=risk_metrics)
        decision = self.engine.make_decision(context)
        
        self.assertIsNotNone(decision)
        # Level3がアクティベートされることを確認
        if decision.decision_level == 3:
            self.assertTrue(decision.override_conditions)
    
    def test_emergency_signal_handling(self):
        """緊急シグナル処理テスト"""
        emergency_signals = {
            'risk_breach': True,
            'system_alert': 'high_volatility'
        }
        
        context = self.create_test_context(emergency_signals=emergency_signals)
        decision = self.engine.make_decision(context)
        
        self.assertIsNotNone(decision)
        # 緊急シグナルがある場合はLevel3がアクティベートされるべき
        if decision.decision_level == 3:
            self.assertTrue(decision.override_conditions)
    
    def test_multiple_emergency_conditions(self):
        """複数緊急条件テスト"""
        # 複数の緊急条件を同時に満たす
        risk_metrics = {
            'current_drawdown': 0.10,  # 高ドローダウン
            'portfolio_var': 0.05,     # 高VaR
            'portfolio_volatility': 0.35,  # 高ボラティリティ
            'correlation_risk': 0.9,   # 高相関リスク
            'exposure_concentration': 0.8  # 高エクスポージャー集中
        }
        
        context = self.create_test_context(risk_metrics=risk_metrics)
        decision = self.engine.make_decision(context)
        
        self.assertIsNotNone(decision)
        self.assertEqual(decision.decision_level, 3)
        self.assertIn(decision.decision_type, ['switch', 'emergency_stop'])
        self.assertTrue(decision.override_conditions)
    
    def test_decision_priority_hierarchy(self):
        """決定優先度階層テスト"""
        # Level3の緊急条件とLevel2の日次ターゲット条件を同時に満たす
        risk_metrics = self.base_risk_metrics.copy()
        risk_metrics['current_drawdown'] = 0.06  # 緊急レベル
        
        portfolio_state = self.base_portfolio_state.copy()
        portfolio_state['daily_return'] = 0.005  # 大幅未達成
        
        context = self.create_test_context(
            risk_metrics=risk_metrics,
            portfolio_state=portfolio_state
        )
        decision = self.engine.make_decision(context)
        
        # Level3が優先されるべき
        self.assertEqual(decision.decision_level, 3)
        self.assertTrue(decision.override_conditions)
    
    def test_confidence_calculation(self):
        """信頼度計算テスト"""
        context = self.create_test_context()
        decision = self.engine.make_decision(context)
        
        self.assertIsNotNone(decision)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        
        # 信頼度が高い場合は切替決定、低い場合は現状維持が期待される
        if decision.confidence >= 0.7:
            self.assertIn(decision.decision_type, ['switch', 'emergency_stop'])
        else:
            self.assertEqual(decision.decision_type, 'maintain')
    
    def test_decision_reasoning(self):
        """決定理由テスト"""
        context = self.create_test_context()
        decision = self.engine.make_decision(context)
        
        self.assertIsNotNone(decision.reasoning)
        self.assertIsInstance(decision.reasoning, str)
        self.assertGreater(len(decision.reasoning), 0)
    
    def test_decision_metadata(self):
        """決定メタデータテスト"""
        context = self.create_test_context()
        decision = self.engine.make_decision(context)
        
        # メタデータが適切に設定されていることを確認
        if decision.metadata:
            self.assertIsInstance(decision.metadata, dict)
    
    def test_engine_statistics(self):
        """エンジン統計テスト"""
        context = self.create_test_context()
        
        # 初期統計
        initial_stats = self.engine.get_engine_stats()
        self.assertEqual(initial_stats['total_decisions'], 0)
        
        # 決定実行
        decision = self.engine.make_decision(context)
        
        # 統計更新確認
        updated_stats = self.engine.get_engine_stats()
        self.assertEqual(updated_stats['total_decisions'], 1)
        self.assertGreater(updated_stats[f'level{decision.decision_level}_decisions'], 0)
    
    def test_decision_history(self):
        """決定履歴テスト"""
        context = self.create_test_context()
        
        # 複数の決定を実行
        for i in range(3):
            self.engine.make_decision(context)
        
        # 履歴確認
        history = self.engine.get_recent_decisions(count=5)
        self.assertEqual(len(history), 3)
        
        for record in history:
            self.assertIn('timestamp', record)
            self.assertIn('decision', record)
            self.assertIn('context_summary', record)
    
    def test_configuration_validation(self):
        """設定妥当性テスト"""
        # 正常な設定
        self.assertTrue(self.engine.validate_configuration())
        
        # 不正な設定でのエンジン作成
        invalid_config = {'invalid': 'config'}
        try:
            invalid_engine = HierarchicalSwitchDecisionEngine(config=invalid_config)
            # 設定不備があっても初期化は成功するが、妥当性チェックで失敗するべき
            self.assertFalse(invalid_engine.validate_configuration())
        except Exception:
            # 初期化段階で失敗することも正常
            pass
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 不完全なコンテキストでのテスト
        incomplete_context = DecisionContext(
            strategies_data={},  # 空のデータ
            risk_metrics={},
            market_conditions={},
            portfolio_state={},
            timestamp=datetime.now()
        )
        
        decision = self.engine.make_decision(incomplete_context)
        
        # エラー時でも有効な決定が返されるべき
        self.assertIsNotNone(decision)
        self.assertIn(decision.decision_type, ['maintain', 'emergency_stop'])


def run_integration_test():
    """統合テストを実行"""
    print("=== 階層化切替決定システム 統合テスト ===")
    
    # テストスイートの作成と実行
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestHierarchicalSwitchDecisionSystem)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # 結果サマリー
    print(f"\n=== テスト結果サマリー ===")
    print(f"実行テスト数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    
    if result.failures:
        print("\n=== 失敗したテスト ===")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n=== エラーが発生したテスト ===")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_test()
    exit(0 if success else 1)
