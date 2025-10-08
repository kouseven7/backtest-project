"""
Test Script: 3-1-3 Strategy Selection Rule Engine
File: test_rule_engine_implementation.py
Description: 
  3-1-3「選択ルールの抽象化（差し替え可能に）」実装テスト
  各コンポーネントの統合テストとサンプル利用例

Author: imega
Created: 2025-07-13
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

def test_basic_rule_engine():
    """基本ルールエンジンのテスト"""
    print("=" * 60)
    print("Testing Basic Rule Engine")
    print("=" * 60)
    
    try:
        from config.strategy_selection_rule_engine import (
            StrategySelectionRuleEngine, RuleContext, SelectionCriteria,
            TrendBasedSelectionRule, ScoreBasedSelectionRule, RiskAdjustedSelectionRule
        )
        
        # ルールエンジンの初期化
        engine = StrategySelectionRuleEngine()
        print(f"✓ Rule engine initialized with {len(engine.rules)} rules")
        
        # テストコンテキストの作成
        context = RuleContext(
            strategy_scores={
                'momentum': 0.8,
                'mean_reversion': 0.6,
                'breakout': 0.9,
                'pairs': 0.5,
                'defensive': 0.4
            },
            trend_analysis={
                'trend_type': 'uptrend',
                'confidence': 0.85,
                'strength': 0.7
            },
            selection_criteria=SelectionCriteria(
                max_strategies=3,
                min_score_threshold=0.6
            ),
            available_strategies={'momentum', 'mean_reversion', 'breakout', 'pairs', 'defensive'},
            ticker='AAPL',
            timestamp=datetime.now(),
            data_quality=0.9,
            risk_metrics={
                'momentum': {'volatility': 0.15, 'sharpe_ratio': 1.2, 'max_drawdown': 0.08},
                'breakout': {'volatility': 0.25, 'sharpe_ratio': 1.0, 'max_drawdown': 0.12},
                'pairs': {'volatility': 0.10, 'sharpe_ratio': 1.5, 'max_drawdown': 0.05}
            }
        )
        
        # ルールの個別実行テスト
        print("\n--- Individual Rule Tests ---")
        
        # 1. トレンドベースルール
        trend_rule = TrendBasedSelectionRule()
        if trend_rule.can_execute(context):
            trend_result = trend_rule.execute(context)
            print(f"TrendBased Rule: {trend_result.execution_status.value}")
            print(f"  Selected: {trend_result.selected_strategies}")
            print(f"  Confidence: {trend_result.confidence:.2f}")
        
        # 2. スコアベースルール
        score_rule = ScoreBasedSelectionRule()
        if score_rule.can_execute(context):
            score_result = score_rule.execute(context)
            print(f"ScoreBased Rule: {score_result.execution_status.value}")
            print(f"  Selected: {score_result.selected_strategies}")
            print(f"  Confidence: {score_result.confidence:.2f}")
        
        # 3. リスク調整ルール
        risk_rule = RiskAdjustedSelectionRule()
        if risk_rule.can_execute(context):
            risk_result = risk_rule.execute(context)
            print(f"RiskAdjusted Rule: {risk_result.execution_status.value}")
            print(f"  Selected: {risk_result.selected_strategies}")
            print(f"  Confidence: {risk_result.confidence:.2f}")
        
        # 全ルール実行
        print("\n--- Engine Execution ---")
        results = engine.execute_rules(context)
        
        print(f"Executed {len(results)} rules:")
        for result in results:
            print(f"  {result.rule_name}: {result.execution_status.value}")
            print(f"    Selected: {result.selected_strategies}")
            print(f"    Confidence: {result.confidence:.2f}")
            print(f"    Time: {result.execution_time_ms:.1f}ms")
        
        # 最適結果の選択
        best_result = engine.select_best_result(results)
        if best_result:
            print(f"\nBest Result: {best_result.rule_name}")
            print(f"  Strategies: {best_result.selected_strategies}")
            print(f"  Weights: {best_result.strategy_weights}")
        
        # パフォーマンス統計
        print("\n--- Performance Summary ---")
        perf_summary = engine.get_performance_summary()
        for rule_name, stats in perf_summary.get('rule_statistics', {}).items():
            print(f"  {rule_name}: Success Rate {stats['success_rate']:.1%}, "
                  f"Avg Time {stats['average_time_ms']:.1f}ms")
        
        print("✓ Basic Rule Engine test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Basic Rule Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configurable_rules():
    """設定可能ルールのテスト"""
    print("\n" + "=" * 60)
    print("Testing Configurable Rules")
    print("=" * 60)
    
    try:
        from config.strategy_selection_rule_engine import (
            StrategySelectionRuleEngine, RuleContext, SelectionCriteria,
            ConfigurableSelectionRule
        )
        
        # カスタムルール設定
        custom_config = {
            'type': 'custom',
            'required_fields': ['strategy_scores', 'trend_analysis'],
            'conditions': [
                {
                    'type': 'trend_confidence',
                    'threshold': 0.8,
                    'operator': '>='
                },
                {
                    'type': 'trend_type',
                    'value': 'uptrend'
                }
            ],
            'actions': {
                'type': 'select_by_trend',
                'trend_mappings': {
                    'uptrend': ['momentum', 'breakout'],
                    'downtrend': ['defensive', 'short_selling'],
                    'sideways': ['mean_reversion', 'pairs']
                },
                'base_confidence': 0.8
            }
        }
        
        # カスタムルールの作成
        custom_rule = ConfigurableSelectionRule(
            name="CustomUptrendRule",
            config=custom_config,
            priority=25
        )
        
        # ルールエンジンに追加
        engine = StrategySelectionRuleEngine()
        engine.add_rule(custom_rule)
        
        print(f"✓ Custom rule added. Total rules: {len(engine.rules)}")
        
        # テストコンテキスト（上昇トレンド、高信頼度）
        context = RuleContext(
            strategy_scores={
                'momentum': 0.8,
                'mean_reversion': 0.6,
                'breakout': 0.9,
                'pairs': 0.5,
                'defensive': 0.4
            },
            trend_analysis={
                'trend_type': 'uptrend',
                'confidence': 0.85,  # 高信頼度
                'strength': 0.7
            },
            selection_criteria=SelectionCriteria(),
            available_strategies={'momentum', 'mean_reversion', 'breakout', 'pairs', 'defensive'},
            ticker='TEST',
            timestamp=datetime.now(),
            data_quality=0.9
        )
        
        # カスタムルールの実行
        if custom_rule.can_execute(context):
            result = custom_rule.execute(context)
            print(f"Custom Rule Execution: {result.execution_status.value}")
            print(f"  Selected: {result.selected_strategies}")
            print(f"  Reasoning: {result.reasoning}")
            print(f"  Confidence: {result.confidence:.2f}")
        else:
            print("Custom rule cannot execute with current context")
        
        # 低信頼度テスト（条件を満たさない場合）
        context.trend_analysis['confidence'] = 0.5  # 低信頼度
        
        if custom_rule.can_execute(context):
            result = custom_rule.execute(context)
            print(f"Low Confidence Test: {result.execution_status.value}")
            if result.execution_status.value == 'skipped':
                print("  ✓ Rule correctly skipped due to low confidence")
        
        print("✓ Configurable Rules test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Configurable Rules test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rule_configuration_manager():
    """ルール設定管理のテスト"""
    print("\n" + "=" * 60)
    print("Testing Rule Configuration Manager")
    print("=" * 60)
    
    try:
        from config.rule_configuration_manager import RuleConfigurationManager
        
        # 設定管理の初期化
        config_manager = RuleConfigurationManager()
        print("✓ Rule Configuration Manager initialized")
        
        # 現在の設定サマリー
        summary = config_manager.get_configuration_summary()
        print(f"Configuration Summary:")
        print(f"  Total Rules: {summary['total_rules']}")
        print(f"  Enabled Rules: {summary['enabled_rules']}")
        print(f"  Rule Types: {summary['rule_types']}")
        
        # カスタムルール設定の追加
        new_rule_config = {
            "type": "Configurable",
            "name": "TestSidewaysRule",
            "priority": 30,
            "enabled": True,
            "config": {
                "required_fields": ["strategy_scores"],
                "conditions": [
                    {
                        "type": "trend_type",
                        "value": "sideways"
                    }
                ],
                "actions": {
                    "type": "select_top",
                    "count": 2,
                    "threshold": 0.5,
                    "base_confidence": 0.7
                }
            }
        }
        
        # ルール追加
        if config_manager.add_rule_configuration(new_rule_config):
            print("✓ Custom rule configuration added successfully")
        
        # 設定の検証
        current_config = config_manager.load_configuration()
        validation_result = config_manager.validate_configuration(current_config)
        
        print(f"Configuration Validation:")
        print(f"  Valid: {validation_result.is_valid}")
        print(f"  Status: {validation_result.status.value}")
        print(f"  Errors: {len(validation_result.errors)}")
        print(f"  Warnings: {len(validation_result.warnings)}")
        
        if validation_result.warnings:
            print("  Warning Details:")
            for warning in validation_result.warnings:
                print(f"    - {warning}")
        
        # ルール一覧の表示
        rules = config_manager.list_rule_configurations()
        print(f"\nConfigured Rules ({len(rules)}):")
        for rule in rules:
            print(f"  - {rule['name']} ({rule['type']}, priority: {rule['priority']}, "
                  f"enabled: {rule['enabled']})")
        
        print("✓ Rule Configuration Manager test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Rule Configuration Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_scenario():
    """統合シナリオのテスト"""
    print("\n" + "=" * 60)
    print("Testing Integration Scenario")
    print("=" * 60)
    
    try:
        # 実際の使用例を模擬
        print("Simulating real-world usage scenario...")
        
        # 1. 市場状況の変化をシミュレート
        market_scenarios = [
            {
                'name': 'Bull Market',
                'trend_type': 'uptrend',
                'confidence': 0.9,
                'strategy_scores': {
                    'momentum': 0.9, 'breakout': 0.8, 'mean_reversion': 0.4,
                    'pairs': 0.5, 'defensive': 0.3
                }
            },
            {
                'name': 'Bear Market',
                'trend_type': 'downtrend',
                'confidence': 0.8,
                'strategy_scores': {
                    'momentum': 0.3, 'breakout': 0.2, 'mean_reversion': 0.7,
                    'pairs': 0.6, 'defensive': 0.9
                }
            },
            {
                'name': 'Sideways Market',
                'trend_type': 'sideways',
                'confidence': 0.7,
                'strategy_scores': {
                    'momentum': 0.4, 'breakout': 0.3, 'mean_reversion': 0.8,
                    'pairs': 0.9, 'defensive': 0.6
                }
            }
        ]
        
        from config.strategy_selection_rule_engine import (
            StrategySelectionRuleEngine, RuleContext, SelectionCriteria
        )
        
        engine = StrategySelectionRuleEngine()
        
        for scenario in market_scenarios:
            print(f"\n--- {scenario['name']} Scenario ---")
            
            context = RuleContext(
                strategy_scores=scenario['strategy_scores'],
                trend_analysis={
                    'trend_type': scenario['trend_type'],
                    'confidence': scenario['confidence'],
                    'strength': 0.7
                },
                selection_criteria=SelectionCriteria(max_strategies=2),
                available_strategies=set(scenario['strategy_scores'].keys()),
                ticker='MARKET_TEST',
                timestamp=datetime.now(),
                data_quality=0.9
            )
            
            # ルール実行
            results = engine.execute_rules(context)
            best_result = engine.select_best_result(results)
            
            if best_result:
                print(f"  Best Rule: {best_result.rule_name}")
                print(f"  Selected Strategies: {best_result.selected_strategies}")
                print(f"  Confidence: {best_result.confidence:.2f}")
                print(f"  Reasoning: {best_result.reasoning}")
            
        print("\n✓ Integration scenario test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Integration scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_configs():
    """サンプル設定ファイルの作成"""
    print("\n" + "=" * 60)
    print("Creating Sample Configuration Files")
    print("=" * 60)
    
    try:
        # 設定ディレクトリの作成
        config_dir = Path("config/rule_engine")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # 高度なルール設定例
        advanced_config = {
            "rules": [
                {
                    "type": "TrendBased",
                    "name": "PrimaryTrendBased",
                    "priority": 5,
                    "enabled": True,
                    "config": {}
                },
                {
                    "type": "Hybrid",
                    "name": "MainHybrid",
                    "priority": 10,
                    "enabled": True,
                    "config": {}
                },
                {
                    "type": "Configurable",
                    "name": "HighConfidenceUptrend",
                    "priority": 8,
                    "enabled": True,
                    "config": {
                        "required_fields": ["strategy_scores", "trend_analysis"],
                        "conditions": [
                            {
                                "type": "trend_type",
                                "value": "uptrend"
                            },
                            {
                                "type": "trend_confidence",
                                "threshold": 0.8,
                                "operator": ">="
                            }
                        ],
                        "actions": {
                            "type": "select_by_trend",
                            "trend_mappings": {
                                "uptrend": ["momentum", "breakout", "trend_following"]
                            },
                            "base_confidence": 0.9
                        }
                    }
                },
                {
                    "type": "Configurable",
                    "name": "ConservativeSelection",
                    "priority": 25,
                    "enabled": True,
                    "config": {
                        "required_fields": ["strategy_scores", "risk_metrics"],
                        "conditions": [
                            {
                                "type": "data_quality",
                                "threshold": 0.6,
                                "operator": ">="
                            }
                        ],
                        "actions": {
                            "type": "select_top",
                            "count": 1,
                            "threshold": 0.8,
                            "base_confidence": 0.7
                        }
                    }
                }
            ],
            "global_settings": {
                "default_priority": 50,
                "max_execution_time_ms": 3000,
                "enable_parallel_execution": False,
                "cache_enabled": True
            },
            "last_updated": datetime.now().isoformat(),
            "version": "1.1"
        }
        
        # 設定ファイルの保存
        config_file = config_dir / "advanced_rules_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(advanced_config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Advanced configuration saved: {config_file}")
        
        # スキーマファイルの確認
        schema_file = config_dir / "rule_schema.json"
        if schema_file.exists():
            print(f"✓ Schema file exists: {schema_file}")
        else:
            print(f"ℹ Schema file will be created automatically on first run")
        
        print("✓ Sample configuration files created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create sample configs: {e}")
        return False

def main():
    """メインテスト実行"""
    print("3-1-3 Strategy Selection Rule Engine Implementation Test")
    print("=" * 80)
    
    test_results = []
    
    # 各テストの実行
    test_results.append(("Basic Rule Engine", test_basic_rule_engine()))
    test_results.append(("Configurable Rules", test_configurable_rules()))
    test_results.append(("Rule Configuration Manager", test_rule_configuration_manager()))
    test_results.append(("Integration Scenario", test_integration_scenario()))
    test_results.append(("Sample Configurations", create_sample_configs()))
    
    # テスト結果の集計
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        color = "✓" if result else "✗"
        print(f"{color} {test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(test_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(test_results)*100:.1f}%")
    
    if failed == 0:
        print("\n[SUCCESS] All tests passed! 3-1-3 implementation is ready for use.")
        print("\nNext steps:")
        print("  1. Run with real market data")
        print("  2. Customize rule configurations")
        print("  3. Monitor performance metrics")
        print("  4. Integrate with existing backtesting system")
    else:
        print(f"\n[WARNING] {failed} test(s) failed. Please check the error messages above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
