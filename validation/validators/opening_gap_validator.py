"""
OpeningGap戦略専用パラメータバリデータ
"""
from typing import Dict, Any

class OpeningGapParameterValidator:
    rules = {
        "atr_threshold": {"min": 0.5, "max": 5.0, "type": float, "description": "高ボラティリティ判定閾値"},
        "stop_loss": {"min": 0.005, "max": 0.1, "type": float, "description": "ストップロス幅"},
        "take_profit": {"min": 0.01, "max": 0.2, "type": float, "description": "利益確定幅"},
        "gap_threshold": {"min": 0.005, "max": 0.05, "type": float, "description": "ギャップ判定閾値"},
        "entry_delay": {"min": 0, "max": 3, "type": int, "description": "エントリー遅延日数"},
        "gap_direction": {"choices": ["up", "down", "both"], "type": str, "description": "ギャップ方向"},
        "dow_filter_enabled": {"choices": [True, False], "type": bool, "description": "ダウフィルター有効"},
        "dow_trend_days": {"min": 1, "max": 20, "type": int, "description": "ダウトレンド判定期間"},
        "min_vol_ratio": {"min": 0.5, "max": 5.0, "type": float, "description": "最小出来高倍率"},
        "volatility_filter": {"choices": [True, False], "type": bool, "description": "高ボラ環境のみ取引"},
        "max_hold_days": {"min": 1, "max": 20, "type": int, "description": "最大保有日数"},
        "consecutive_down_days": {"min": 1, "max": 5, "type": int, "description": "連続下落日数"},
        "trailing_stop_pct": {"min": 0.005, "max": 0.1, "type": float, "description": "トレーリングストップ率"},
        "atr_stop_multiple": {"min": 0.5, "max": 5.0, "type": float, "description": "ATRストップ乗数"},
        "partial_exit_enabled": {"choices": [True, False], "type": bool, "description": "一部利確有効"},
        "partial_exit_threshold": {"min": 0.005, "max": 0.1, "type": float, "description": "一部利確閾値"},
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
                # 型変換
                if rule["type"] == int:
                    value = int(float(value))
                elif rule["type"] == float:
                    value = float(value)
                elif rule["type"] == bool:
                    if isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes")
                    else:
                        value = bool(value)
                elif rule["type"] == str:
                    value = str(value)
            except Exception:
                errors.append(f"{param_name}の型が不正です: {value}")
                continue
            if "min" in rule and value < rule["min"]:
                errors.append(f"{param_name}が最小値未満です: {value} < {rule['min']}")
            if "max" in rule and value > rule["max"]:
                errors.append(f"{param_name}が最大値超過です: {value} > {rule['max']}")
            if "choices" in rule and value not in rule["choices"]:
                errors.append(f"{param_name}が許容値ではありません: {value}（許容: {rule['choices']}）")
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "validation_summary": "OK" if len(errors) == 0 else "NG"
        }
