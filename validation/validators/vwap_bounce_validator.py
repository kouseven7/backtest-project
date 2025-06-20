"""
VWAPBounce戦略専用パラメータバリデータ
"""
from typing import Dict, Any

class VWAPBounceParameterValidator:
    rules = {
        "vwap_lower_threshold": {"min": 0.97, "max": 1.0, "type": float, "description": "VWAP下限閾値"},
        "vwap_upper_threshold": {"min": 1.0, "max": 1.05, "type": float, "description": "VWAP上限閾値"},
        "volume_increase_threshold": {"min": 1.0, "max": 2.0, "type": float, "description": "出来高増加閾値"},
        "stop_loss": {"min": 0.005, "max": 0.05, "type": float, "description": "ストップロス幅"},
        "take_profit": {"min": 0.01, "max": 0.2, "type": float, "description": "利益確定幅"},
        "trailing_stop_pct": {"min": 0.005, "max": 0.05, "type": float, "description": "トレーリングストップ率"},
        "bullish_candle_min_pct": {"min": 0.001, "max": 0.02, "type": float, "description": "陽線最小サイズ"},
        "max_hold_days": {"min": 1, "max": 20, "type": int, "description": "最大保有日数"},
        "cool_down_period": {"min": 0, "max": 10, "type": int, "description": "クールダウン期間"},
        "partial_exit_portion": {"min": 0.1, "max": 0.9, "type": float, "description": "一部利確割合"},
    }

    @classmethod
    def validate(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        for param_name, rule in cls.rules.items():
            value = params.get(param_name)
            if value is None:
                continue  # 未指定はデフォルト値で許容
            try:
                # 型変換（文字列で渡された場合も考慮）
                if rule["type"] == int:
                    value = int(float(value))
                elif rule["type"] == float:
                    value = float(value)
            except Exception:
                errors.append(f"{param_name}の型が不正です: {value}")
                continue
            if "min" in rule and value < rule["min"]:
                errors.append(f"{param_name}が最小値未満です: {value} < {rule['min']}")
            if "max" in rule and value > rule["max"]:
                errors.append(f"{param_name}が最大値超過です: {value} > {rule['max']}")
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "validation_summary": "OK" if len(errors) == 0 else "NG"
        }
