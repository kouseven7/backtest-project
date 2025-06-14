"""
contrarian_validator.py
逆張り戦略（ContrarianStrategy）用の最適化パラメータ承認・バリデーションロジック。
型安全・例外対策済み。
"""
from typing import Any, Dict
from config.optimized_parameters import OptimizedParameterManager

class ContrarianParameterValidator:
    def __init__(self, strategy_name: str = "ContrarianStrategy"):
        self.strategy_name = strategy_name
        self.param_manager = OptimizedParameterManager()

    @classmethod
    def validate(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        warnings = []
        # 必須キーと型・範囲チェック
        rules = {
            "entry_threshold": (float, 0, 1),
            "exit_threshold": (float, 0, 1),
            "lookback_period": (int, 1, 100)
        }
        for key, (tp, vmin, vmax) in rules.items():
            if key not in params:
                errors.append(f"{key} が未設定です")
                continue
            try:
                val = tp(params[key])
            except Exception:
                errors.append(f"{key} の型が不正です (期待: {tp.__name__})")
                continue
            if not (vmin <= val <= vmax):
                errors.append(f"{key} の値域が不正です (許容: {vmin}～{vmax}, 実際: {val})")
        valid = len(errors) == 0
        return {
            "errors": errors,
            "warnings": warnings,
            "valid": valid,
            "validation_summary": ("✅ 妥当" if valid else "❌ 不正")
        }

    def get_latest_parameters(self, ticker: str = None) -> Dict[str, Any]:
        # 最新の承認済みパラメータを取得
        return self.param_manager.load_approved_params(self.strategy_name, ticker)
