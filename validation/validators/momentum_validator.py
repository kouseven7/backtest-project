"""
Momentum戦略専用パラメータバリデータ
"""
from typing import Dict, Any, List

class MomentumParameterValidator:
    rules = {
        "sma_short": {"min": 5, "max": 50, "type": int, "description": "短期移動平均期間"},
        "sma_long": {"min": 20, "max": 200, "type": int, "description": "長期移動平均期間"},
        "rsi_period": {"min": 10, "max": 30, "type": int, "description": "RSI計算期間"},
        "rsi_lower": {"min": 20, "max": 50, "type": int, "description": "RSI下限閾値"},
        "rsi_upper": {"min": 60, "max": 90, "type": int, "description": "RSI上限閾値"},
        "take_profit": {"min": 0.02, "max": 0.5, "type": float, "description": "利確レベル"},
        "stop_loss": {"min": 0.01, "max": 0.2, "type": float, "description": "損切りレベル"},
        "trailing_stop": {"min": 0.01, "max": 0.15, "type": float, "description": "トレーリングストップ"},
        "volume_threshold": {"min": 1.0, "max": 3.0, "type": float, "description": "出来高閾値"},
        "max_hold_days": {"min": 1, "max": 60, "type": int, "description": "最大保有期間"},
        "atr_multiple": {"min": 1.0, "max": 5.0, "type": float, "description": "ATR倍率"},
        "partial_exit_pct": {"min": 0.0, "max": 1.0, "type": float, "description": "部分利確率"},
        "partial_exit_threshold": {"min": 0.02, "max": 0.3, "type": float, "description": "部分利確閾値"},
        "momentum_exit_threshold": {"min": -0.1, "max": 0.0, "type": float, "description": "モメンタム失速閾値"},
        "volume_exit_threshold": {"min": 0.3, "max": 1.0, "type": float, "description": "出来高減少閾値"}
    }

    @classmethod
    def validate(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        warnings = []
        for param_name, param_value in params.items():
            if param_name in cls.rules:
                rule = cls.rules[param_name]
                if not isinstance(param_value, rule["type"]):
                    try:
                        if rule["type"] == int:
                            param_value = int(param_value)
                        elif rule["type"] == float:
                            param_value = float(param_value)
                        params[param_name] = param_value
                    except (ValueError, TypeError):
                        errors.append(f"{param_name} ({rule['description']}): 型が不正 (期待: {rule['type'].__name__}, 実際: {type(param_value).__name__})")
                        continue
                if param_value < rule["min"]:
                    errors.append(f"{param_name} ({rule['description']}): 最小値{rule['min']}未満です (実際: {param_value})")
                elif param_value > rule["max"]:
                    errors.append(f"{param_name} ({rule['description']}): 最大値{rule['max']}超過です (実際: {param_value})")
        # 論理チェック
        if 'sma_short' in params and 'sma_long' in params:
            if params['sma_short'] >= params['sma_long']:
                errors.append("短期移動平均期間は長期移動平均期間より小さくする必要があります")
        if 'rsi_lower' in params and 'rsi_upper' in params:
            if params['rsi_lower'] >= params['rsi_upper']:
                errors.append("RSI下限閾値はRSI上限閾値より小さくする必要があります")
        if 'take_profit' in params and 'stop_loss' in params:
            if params['take_profit'] <= params['stop_loss']:
                errors.append("利確レベルは損切りレベルより大きくする必要があります")
        # 推奨値警告
        if 'rsi_period' in params:
            if params['rsi_period'] < 14 or params['rsi_period'] > 21:
                warnings.append("RSI期間は14-21の範囲が推奨されます")
        if 'take_profit' in params and 'stop_loss' in params:
            ratio = params['take_profit'] / params['stop_loss']
            if ratio < 1.5:
                warnings.append("利確/損切り比率は1.5以上が推奨されます")
        return {
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0,
            "validation_summary": cls._generate_validation_summary(errors, warnings)
        }

    @staticmethod
    def _generate_validation_summary(errors: List[str], warnings: List[str]) -> str:
        if errors:
            return f"❌ エラー{len(errors)}件"
        elif warnings:
            return f"⚠️ 警告{len(warnings)}件"
        else:
            return "✅ 検証完了"
