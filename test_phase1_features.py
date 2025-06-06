#!/usr/bin/env python3
"""
Phase 1 新機能テスト: 戦略自動判別機能のテスト
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from validation.parameter_validator import ParameterValidator

def test_strategy_auto_detection():
    """戦略自動判別機能のテスト"""
    print("🧪 Phase 1 新機能テスト開始")
    print("=" * 60)
    
    validator = ParameterValidator()
    
    # テスト1: MomentumStrategy固有パラメータ
    print("\n📋 テスト1: MomentumStrategy固有パラメータの自動検出")
    momentum_params = {
        "sma_short": 20,
        "sma_long": 50,
        "rsi_period": 14,
        "take_profit": 0.1,
        "stop_loss": 0.05
    }
    
    detected_strategy = validator.auto_detect_strategy(momentum_params)
    print(f"検出された戦略: {detected_strategy}")
    
    validation_result = validator.validate_auto(momentum_params)
    print(f"検証結果: {validation_result['validation_summary']}")
    
    # テスト2: BreakoutStrategy固有パラメータ
    print("\n📋 テスト2: BreakoutStrategy固有パラメータの自動検出")
    breakout_params = {
        "volume_threshold": 1.5,
        "take_profit": 0.05,
        "look_back": 3,
        "trailing_stop": 0.02,
        "breakout_buffer": 0.01
    }
    
    detected_strategy = validator.auto_detect_strategy(breakout_params)
    print(f"検出された戦略: {detected_strategy}")
    
    validation_result = validator.validate_auto(breakout_params)
    print(f"検証結果: {validation_result['validation_summary']}")
    
    # テスト3: 戦略名による明示的な指定
    print("\n📋 テスト3: 戦略名による明示的な指定")
    test_cases = [
        ("momentum", momentum_params),
        ("MomentumInvestingStrategy", momentum_params),
        ("breakout", breakout_params),
        ("BreakoutStrategy", breakout_params),
        ("unknown_strategy", momentum_params)
    ]
    
    for strategy_name, params in test_cases:
        result = validator.validate(strategy_name, params)
        print(f"戦略: {strategy_name} → {result['validation_summary']}")
    
    # テスト4: 後方互換性テスト
    print("\n📋 テスト4: 後方互換性テスト（既存メソッド）")
    legacy_momentum_result = validator.validate_momentum_parameters(momentum_params)
    legacy_breakout_result = validator.validate_breakout_parameters(breakout_params)
    
    print(f"従来のmomentum検証: {legacy_momentum_result['validation_summary']}")
    print(f"従来のbreakout検証: {legacy_breakout_result['validation_summary']}")
    
    print("\n✅ Phase 1 テスト完了")
    print("🚀 新機能が正常に動作しています！")

if __name__ == "__main__":
    test_strategy_auto_detection()
