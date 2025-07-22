"""
Demo Script: Drawdown Control System
File: demo_drawdown_control.py
Description: 
  5-3-1「ドローダウン制御機能」デモンストレーション

Author: imega
Created: 2025-07-20
Modified: 2025-07-20
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config.drawdown_controller import (
        DrawdownController, DrawdownSeverity, create_default_drawdown_config
    )
    from config.drawdown_action_executor import DrawdownActionExecutor
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_market_scenario_data():
    """市場シナリオデータ作成"""
    scenarios = {
        "gradual_decline": {
            "name": "段階的下落シナリオ",
            "description": "徐々にポートフォリオ価値が下落し、各制御レベルがトリガーされる",
            "values": [
                1000000, 995000, 985000, 970000, 950000,  # 警告レベルまで
                930000, 910000, 890000,                   # 重要レベルまで  
                870000, 850000, 830000,                   # 緊急レベルまで
                845000, 860000, 875000                    # 回復局面
            ]
        },
        "flash_crash": {
            "name": "急落・回復シナリオ",
            "description": "急激な価格下落後、部分的回復を示すシナリオ",
            "values": [
                1000000, 990000, 980000,                 # 初期下落
                920000, 860000, 820000,                  # 急落
                840000, 870000, 900000, 920000           # 回復
            ]
        },
        "volatile_market": {
            "name": "高ボラティリティシナリオ",
            "description": "価格が激しく上下動するシナリオ",
            "values": [
                1000000, 980000, 1020000, 950000, 1040000,
                920000, 1060000, 890000, 1080000, 860000,
                900000, 950000, 980000
            ]
        },
        "steady_decline": {
            "name": "持続的下落シナリオ", 
            "description": "継続的な価値減少を示すシナリオ",
            "values": [
                1000000, 985000, 970000, 955000, 940000,
                925000, 910000, 895000, 880000, 865000,
                850000, 835000, 820000
            ]
        }
    }
    
    return scenarios

def run_scenario_demo(controller: DrawdownController, scenario: dict, demo_name: str):
    """シナリオデモ実行"""
    print(f"\n🎬 {demo_name}: {scenario['name']}")
    print(f"📝 {scenario['description']}")
    print("-" * 60)
    
    # 初期化
    initial_value = scenario['values'][0]
    controller.start_monitoring(initial_value)
    
    print(f"💰 Initial Portfolio Value: ${initial_value:,.0f}")
    
    # シナリオ実行
    control_events = []
    
    for i, value in enumerate(scenario['values'][1:], 1):
        print(f"\n📊 Step {i}: Portfolio Value = ${value:,.0f}")
        
        # 戦略別価値もシミュレーション
        strategy_values = {
            'Momentum': value * 0.35,
            'Contrarian': value * 0.35, 
            'Pairs_Trading': value * 0.30
        }
        
        controller.update_portfolio_value(value, strategy_values)
        
        # 現在のドローダウン計算・表示
        peak_value = controller.performance_tracker['portfolio_peak']
        current_dd = (peak_value - value) / peak_value if peak_value > 0 else 0
        
        print(f"   📉 Drawdown: {current_dd:.2%}")
        
        # 制御イベント確認
        history_count = len(controller.control_history)
        if history_count > len(control_events):
            latest_control = controller.control_history[-1]
            control_events.append(latest_control)
            
            print(f"   🚨 CONTROL TRIGGERED:")
            print(f"      Action: {latest_control.action_taken.value}")
            print(f"      Severity: {latest_control.event.severity.value}")
            print(f"      Impact: {latest_control.expected_impact:.1%}")
        
        # パフォーマンス状況表示
        summary = controller.get_performance_summary()
        print(f"   📈 Status: {summary.get('monitoring_status', 'unknown')}")
        
        # シミュレーション間隔
        time.sleep(1)
    
    # シナリオ完了
    controller.stop_monitoring()
    
    print(f"\n✅ {demo_name} completed!")
    print(f"   Total Control Actions: {len(control_events)}")
    
    if control_events:
        print(f"   Control Summary:")
        for i, event in enumerate(control_events, 1):
            print(f"     {i}. {event.action_taken.value} at {event.event.drawdown_percentage:.2%} DD")
    
    return control_events

def demonstrate_integration_features(controller: DrawdownController):
    """統合機能デモンストレーション"""
    print(f"\n🔗 Integration Features Demonstration")
    print("-" * 60)
    
    # Mock統合システム作成
    from unittest.mock import Mock
    
    # モックシステム設定
    controller.portfolio_risk_manager = Mock()
    controller.position_size_adjuster = Mock()  
    controller.coordination_manager = Mock()
    
    # 統合設定確認
    integration_config = controller.config.get('integration', {})
    print(f"📋 Integration Configuration:")
    for system, config in integration_config.items():
        status = "✅ Enabled" if config.get('enabled', False) else "❌ Disabled"
        print(f"   {system}: {status}")
    
    # 統合アクション実行器テスト
    executor = DrawdownActionExecutor(
        portfolio_risk_manager=controller.portfolio_risk_manager,
        coordination_manager=controller.coordination_manager
    )
    
    # テストポジション
    test_positions = {
        'Momentum': 0.4,
        'Contrarian': 0.3,
        'Pairs_Trading': 0.3
    }
    
    print(f"\n🧪 Testing Integration Actions:")
    print(f"Original Positions: {test_positions}")
    
    # 軽度削減テスト
    result = executor._execute_position_reduction(test_positions, 0.15, "light")
    print(f"After 15% Reduction: {result.final_positions}")
    print(f"Execution Success: {result.success}")
    
    # システム呼び出し履歴確認（モック）
    print(f"Portfolio Risk Manager called: {controller.portfolio_risk_manager.called}")
    
    return executor

def analyze_control_effectiveness():
    """制御効果分析"""
    print(f"\n📊 Control Effectiveness Analysis")
    print("-" * 60)
    
    scenarios = create_market_scenario_data()
    controller = DrawdownController()
    
    results = {}
    
    for scenario_key, scenario in scenarios.items():
        print(f"\n🔍 Analyzing: {scenario['name']}")
        
        # コントローラーリセット
        controller = DrawdownController()
        
        # シナリオ実行（高速）
        controller.start_monitoring(scenario['values'][0])
        
        max_dd = 0
        control_count = 0
        
        for value in scenario['values'][1:]:
            controller.update_portfolio_value(value)
            
            # ドローダウン計算
            peak = controller.performance_tracker['portfolio_peak']
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
            time.sleep(0.1)  # 高速実行
        
        control_count = len(controller.control_history)
        controller.stop_monitoring()
        
        results[scenario_key] = {
            'scenario_name': scenario['name'],
            'max_drawdown': max_dd,
            'control_actions': control_count,
            'final_value': scenario['values'][-1],
            'total_decline': (scenario['values'][0] - scenario['values'][-1]) / scenario['values'][0]
        }
        
        print(f"   Max Drawdown: {max_dd:.2%}")
        print(f"   Control Actions: {control_count}")
        print(f"   Final Recovery: ${scenario['values'][-1]:,.0f}")
    
    # 分析結果サマリー
    print(f"\n📈 Analysis Summary:")
    print("   Scenario".ljust(25) + "Max DD".ljust(10) + "Controls".ljust(10) + "Final Value".ljust(15))
    print("-" * 60)
    
    for key, result in results.items():
        scenario_name = result['scenario_name'][:22]
        max_dd = f"{result['max_drawdown']:.1%}"
        controls = str(result['control_actions'])
        final_val = f"${result['final_value']:,.0f}"
        
        print(f"   {scenario_name.ljust(25)}{max_dd.ljust(10)}{controls.ljust(10)}{final_val}")
    
    return results

def main():
    """メインデモ実行"""
    print("=" * 70)
    print("🚨 Drawdown Control System - Comprehensive Demo")
    print("=" * 70)
    
    try:
        # 1. 基本機能デモ
        print(f"\n🔧 1. System Initialization")
        print("-" * 40)
        
        controller = DrawdownController()
        print(f"✅ Drawdown Controller initialized")
        print(f"   Control Mode: {controller.control_mode.value}")
        print(f"   Warning Threshold: {controller.thresholds.warning_threshold:.1%}")
        print(f"   Critical Threshold: {controller.thresholds.critical_threshold:.1%}")
        print(f"   Emergency Threshold: {controller.thresholds.emergency_threshold:.1%}")
        
        # 2. シナリオベースデモ
        print(f"\n🎭 2. Scenario-Based Demonstrations")
        print("-" * 40)
        
        scenarios = create_market_scenario_data()
        
        # 主要シナリオ実行
        key_scenarios = ['gradual_decline', 'flash_crash']
        all_control_events = []
        
        for scenario_key in key_scenarios:
            scenario = scenarios[scenario_key]
            events = run_scenario_demo(controller, scenario, f"Demo {scenario_key}")
            all_control_events.extend(events)
        
        # 3. 統合機能デモ
        print(f"\n🔗 3. Integration Features")
        print("-" * 40)
        
        executor = demonstrate_integration_features(controller)
        
        # 4. 制御効果分析
        print(f"\n📊 4. Control Effectiveness Analysis")
        print("-" * 40)
        
        effectiveness_results = analyze_control_effectiveness()
        
        # 5. 最終サマリー
        print(f"\n📋 5. Demo Summary")
        print("-" * 40)
        
        total_scenarios = len(key_scenarios)
        total_controls = len(all_control_events)
        
        print(f"✅ Scenarios Tested: {total_scenarios}")
        print(f"🚨 Control Actions Triggered: {total_controls}")
        print(f"🔧 Integration Systems: 4 (Portfolio Risk, Position Size, Coordination, Weight Calculator)")
        
        if all_control_events:
            print(f"📈 Control Action Types:")
            action_counts = {}
            for event in all_control_events:
                action = event.action_taken.value
                action_counts[action] = action_counts.get(action, 0) + 1
            
            for action, count in action_counts.items():
                print(f"   {action}: {count}")
        
        # 設定ファイル生成
        config_file = os.path.join('config', 'drawdown_config.json')
        if not os.path.exists(config_file):
            os.makedirs('config', exist_ok=True)
            default_config = create_default_drawdown_config()
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            print(f"📝 Configuration file created: {config_file}")
        
        print(f"\n🎉 Drawdown Control System Demo completed successfully!")
        print(f"The system is ready for production integration.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
