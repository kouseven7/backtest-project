"""
Phase2: バリデータ分割後の動作確認テスト
"""
from validation.parameter_validator import ParameterValidator

def test_breakout_validator():
    validator = ParameterValidator()
    # 正常系
    params = {
        "volume_threshold": 1.8,
        "take_profit": 0.05,
        "look_back": 3,
        "trailing_stop": 0.02,
        "breakout_buffer": 0.01
    }
    result = validator.validate("breakout", params)
    assert result["valid"], f"正常パラメータで失敗: {result}"
    # 異常系
    params_err = {
        "volume_threshold": 0.5,  # 小さすぎ
        "take_profit": 0.001,  # 小さすぎ
        "look_back": 0,        # 小さすぎ
        "trailing_stop": 0.2,  # 大きすぎ
        "breakout_buffer": 0.1 # 大きすぎ
    }
    result = validator.validate("breakout", params_err)
    assert not result["valid"], "異常パラメータでvalidになっている"
    assert len(result["errors"]) > 0, "エラーが検出されない"
    print("✅ Breakoutバリデータ分割テスト合格")

def test_momentum_validator():
    validator = ParameterValidator()
    params = {
        "sma_short": 10,
        "sma_long": 50,
        "rsi_period": 14,
        "rsi_lower": 30,
        "rsi_upper": 70,
        "take_profit": 0.1,
        "stop_loss": 0.05,
        "trailing_stop": 0.02,
        "volume_threshold": 1.5,
        "max_hold_days": 10,
        "atr_multiple": 2.0,
        "partial_exit_pct": 0.5,
        "partial_exit_threshold": 0.1,
        "momentum_exit_threshold": -0.05,
        "volume_exit_threshold": 0.5
    }
    result = validator.validate("momentum", params)
    assert result["valid"], f"正常パラメータで失敗: {result}"
    params_err = {
        "sma_short": 100,
        "sma_long": 10,
        "rsi_period": 5,
        "rsi_lower": 80,
        "rsi_upper": 60,
        "take_profit": 0.001,
        "stop_loss": 0.5
    }
    result = validator.validate("momentum", params_err)
    assert not result["valid"], "異常パラメータでvalidになっている"
    assert len(result["errors"]) > 0, "エラーが検出されない"
    print("✅ Momentumバリデータ分割テスト合格")

def main():
    test_breakout_validator()
    test_momentum_validator()
    print("✅ Phase2 バリデータ分割テスト完了")

if __name__ == "__main__":
    main()
