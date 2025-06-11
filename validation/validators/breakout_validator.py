"""
Breakout戦略専用パラメータバリデータ
"""
from typing import Dict, Any, List

class BreakoutParameterValidator:
    rules = {
        "volume_threshold": {"min": 1.0, "max": 3.0, "type": float, "description": "出来高増加率の閾値"},
        "take_profit": {"min": 0.01, "max": 0.2, "type": float, "description": "利益確定レベル"},
        "look_back": {"min": 1, "max": 10, "type": int, "description": "ブレイクアウト判定期間"},
        "trailing_stop": {"min": 0.005, "max": 0.1, "type": float, "description": "トレーリングストップ"},
        "breakout_buffer": {"min": 0.001, "max": 0.05, "type": float, "description": "ブレイクアウト判定の閾値"}
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
        if 'take_profit' in params and 'trailing_stop' in params:
            if params['trailing_stop'] >= params['take_profit']:
                errors.append("トレーリングストップは利確レベルより小さくする必要があります")
        if 'look_back' in params:
            if params['look_back'] < 1:
                errors.append("ブレイクアウト判定期間は1以上である必要があります")
        # 推奨値警告
        if 'volume_threshold' in params:
            if params['volume_threshold'] < 1.5:
                warnings.append("出来高閾値は1.5以上が推奨されます")
        if 'look_back' in params:
            if params['look_back'] > 5:
                warnings.append("ブレイクアウト判定期間は5以下が推奨されます")
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
