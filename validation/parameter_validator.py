"""
パラメータ妥当性検証（モメンタム戦略用）
"""
from typing import Dict, List, Any

class ParameterValidator:
    def __init__(self):
        self.momentum_rules = {
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
    
    def validate_momentum_parameters(self, params: Dict[str, Any]) -> Dict:
        """モメンタム戦略パラメータの妥当性検証"""
        errors = []
        warnings = []
        
        # 個別パラメータチェック
        for param_name, param_value in params.items():
            if param_name in self.momentum_rules:
                rule = self.momentum_rules[param_name]
                
                # 型チェック
                if not isinstance(param_value, rule["type"]):
                    try:
                        # 型変換を試行
                        if rule["type"] == int:
                            param_value = int(param_value)
                        elif rule["type"] == float:
                            param_value = float(param_value)
                        params[param_name] = param_value  # 変換成功時は更新
                    except (ValueError, TypeError):
                        errors.append(f"{param_name} ({rule['description']}): 型が不正 (期待: {rule['type'].__name__}, 実際: {type(param_value).__name__})")
                        continue
                
                # 範囲チェック
                if param_value < rule["min"]:
                    errors.append(f"{param_name} ({rule['description']}): 最小値{rule['min']}未満です (実際: {param_value})")
                elif param_value > rule["max"]:
                    errors.append(f"{param_name} ({rule['description']}): 最大値{rule['max']}超過です (実際: {param_value})")
        
        # 論理的整合性チェック
        logical_errors = self._check_logical_consistency(params)
        errors.extend(logical_errors)
        
        # 推奨値からの乖離チェック（警告レベル）
        logical_warnings = self._check_recommended_ranges(params)
        warnings.extend(logical_warnings)
        
        return {
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0,
            "validation_summary": self._generate_validation_summary(errors, warnings)
        }
    
    def _check_logical_consistency(self, params: Dict[str, Any]) -> List[str]:
        """論理的整合性をチェック"""
        errors = []
        
        # 移動平均期間の整合性
        if "sma_short" in params and "sma_long" in params:
            if params["sma_short"] >= params["sma_long"]:
                errors.append("短期移動平均期間は長期移動平均期間より小さくしてください")
        
        # RSI閾値の整合性
        if "rsi_lower" in params and "rsi_upper" in params:
            if params["rsi_lower"] >= params["rsi_upper"]:
                errors.append("RSI下限は上限より小さくしてください")
        
        # 利確・損切りの整合性
        if "take_profit" in params and "stop_loss" in params:
            if params["take_profit"] <= params["stop_loss"]:
                errors.append("利確レベルは損切りレベルより大きくしてください")
        
        # トレーリングストップの整合性
        if "trailing_stop" in params and "stop_loss" in params:
            if params["trailing_stop"] > params["stop_loss"]:
                errors.append("トレーリングストップは初期損切りレベル以下にしてください")
        
        # 部分利確の整合性
        if "partial_exit_threshold" in params and "take_profit" in params:
            if params["partial_exit_threshold"] >= params["take_profit"]:
                errors.append("部分利確閾値は最終利確レベル未満にしてください")
        
        return errors
    
    def _check_recommended_ranges(self, params: Dict[str, Any]) -> List[str]:
        """推奨範囲からの乖離をチェック（警告レベル）"""
        warnings = []
        
        # リスク・リワード比チェック
        if "take_profit" in params and "stop_loss" in params:
            risk_reward_ratio = params["take_profit"] / params["stop_loss"]
            if risk_reward_ratio < 1.5:
                warnings.append(f"リスク・リワード比が低いです: {risk_reward_ratio:.2f} (推奨: 1.5以上)")
            elif risk_reward_ratio > 5.0:
                warnings.append(f"リスク・リワード比が高すぎる可能性があります: {risk_reward_ratio:.2f}")
        
        # 移動平均期間の妥当性
        if "sma_short" in params and "sma_long" in params:
            ratio = params["sma_long"] / params["sma_short"]
            if ratio < 2.0:
                warnings.append(f"長期・短期移動平均の比率が小さいです: {ratio:.1f} (推奨: 2.0以上)")
        
        # RSI範囲の妥当性
        if "rsi_lower" in params and "rsi_upper" in params:
            rsi_range = params["rsi_upper"] - params["rsi_lower"]
            if rsi_range < 15:
                warnings.append(f"RSI範囲が狭すぎる可能性があります: {rsi_range}")
            elif rsi_range > 40:
                warnings.append(f"RSI範囲が広すぎる可能性があります: {rsi_range}")
        
        # 保有期間の妥当性
        if "max_hold_days" in params:
            if params["max_hold_days"] < 5:
                warnings.append("最大保有期間が短すぎる可能性があります (推奨: 5日以上)")
            elif params["max_hold_days"] > 30:
                warnings.append("最大保有期間が長すぎる可能性があります (推奨: 30日以下)")
        
        return warnings
    
    def _generate_validation_summary(self, errors: List[str], warnings: List[str]) -> str:
        """検証結果のサマリーを生成"""
        if not errors and not warnings:
            return "✅ すべてのパラメータが妥当です"
        
        summary = []
        if errors:
            summary.append(f"❌ エラー: {len(errors)}件")
        if warnings:
            summary.append(f"⚠️ 警告: {len(warnings)}件")
        
        return " | ".join(summary)
    
    def generate_validation_report(self, validation_result: Dict) -> str:
        """詳細な検証レポートを生成"""
        report = f"""
=== パラメータ妥当性検証レポート ===
🎯 検証結果: {'✅ 合格' if validation_result['valid'] else '❌ 不合格'}
📋 サマリー: {validation_result['validation_summary']}

"""
        
        if validation_result['errors']:
            report += "❌ エラー項目:\n"
            for i, error in enumerate(validation_result['errors'], 1):
                report += f"  {i}. {error}\n"
            report += "\n"
        
        if validation_result['warnings']:
            report += "⚠️ 警告項目:\n"
            for i, warning in enumerate(validation_result['warnings'], 1):
                report += f"  {i}. {warning}\n"
            report += "\n"
        
        if not validation_result['errors'] and not validation_result['warnings']:
            report += "✅ 問題は検出されませんでした。\n\n"
        
        report += "💡 推奨事項:\n"
        if validation_result['errors']:
            report += "  - エラー項目を修正してから再実行してください\n"
        if validation_result['warnings']:
            report += "  - 警告項目を確認し、必要に応じて調整を検討してください\n"
        if validation_result['valid'] and not validation_result['warnings']:
            report += "  - パラメータは適切です。安心して使用できます\n"
        
        return report
