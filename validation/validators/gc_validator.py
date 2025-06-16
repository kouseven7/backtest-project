"""
GC戦略用パラメータバリデータ
Breakout/Momentumのバリデータを踏襲
"""
from typing import Dict, Any, List

class GCParameterValidator:
    @classmethod
    def validate(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        warnings = []
        # 必須パラメータの存在チェック
        required = [
            "short_window", "long_window", "take_profit", "stop_loss",
            "trailing_stop_pct", "max_hold_days", "exit_on_death_cross",
            "trend_lookback_period", "trend_short_period", "trend_medium_period", "trend_long_period", "trend_up_score"
        ]
        for key in required:
            if key not in params:
                errors.append(f"パラメータ '{key}' が指定されていません")
        # 型・範囲チェック
        if "short_window" in params and params["short_window"] < 1:
            errors.append("short_windowは1以上")
        if "long_window" in params and params["long_window"] < 5:
            errors.append("long_windowは5以上")
        if "take_profit" in params and (params["take_profit"] <= 0 or params["take_profit"] > 0.5):
            errors.append("take_profitは0より大きく0.5以下")
        if "stop_loss" in params and (params["stop_loss"] <= 0 or params["stop_loss"] > 0.5):
            errors.append("stop_lossは0より大きく0.5以下")
        if "trailing_stop_pct" in params and (params["trailing_stop_pct"] < 0 or params["trailing_stop_pct"] > 0.5):
            errors.append("trailing_stop_pctは0以上0.5以下")
        if "max_hold_days" in params and params["max_hold_days"] < 1:
            errors.append("max_hold_daysは1以上")
        if "exit_on_death_cross" in params and not isinstance(params["exit_on_death_cross"], bool):
            errors.append("exit_on_death_crossはbool型")
        if "trend_lookback_period" in params and params["trend_lookback_period"] < 1:
            errors.append("trend_lookback_periodは1以上")
        if "trend_up_score" in params and not (1 <= params["trend_up_score"] <= 6):
            errors.append("trend_up_scoreは1～6の整数")
        # 推奨値警告例
        if "short_window" in params and "long_window" in params:
            if params["short_window"] >= params["long_window"]:
                errors.append("short_windowはlong_windowより小さくする必要があります")
            elif params["long_window"] > 200:
                warnings.append("long_windowは200以下が推奨されます")
        if "take_profit" in params and "stop_loss" in params:
            if params["take_profit"] <= params["stop_loss"]:
                warnings.append("take_profitはstop_lossより大きい値が推奨されます")
        valid = len(errors) == 0
        validation_summary = (
            f"❌ エラー{len(errors)}件" if errors else
            f"⚠️ 警告{len(warnings)}件" if warnings else
            "✅ 検証完了"
        )
        return {
            "errors": errors,
            "warnings": warnings,
            "valid": valid,
            "validation_summary": validation_summary
        }
