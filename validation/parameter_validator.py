"""
パラメータ妥当性検証（戦略対応版）
"""
from typing import Dict, List, Any, Union

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
          # BreakoutStrategy用のパラメータ検証ルール
        self.breakout_rules = {
            "volume_threshold": {"min": 1.0, "max": 3.0, "type": float, "description": "出来高増加率の閾値"},
            "take_profit": {"min": 0.01, "max": 0.2, "type": float, "description": "利益確定レベル"},
            "look_back": {"min": 1, "max": 10, "type": int, "description": "ブレイクアウト判定期間"},
            "trailing_stop": {"min": 0.005, "max": 0.1, "type": float, "description": "トレーリングストップ"},
            "breakout_buffer": {"min": 0.001, "max": 0.05, "type": float, "description": "ブレイクアウト判定の閾値"}
        }
        
        # 戦略名マッピング（自動判別用）
        self.strategy_mapping = {
            'momentum': 'momentum',
            'MomentumInvestingStrategy': 'momentum',
            'breakout': 'breakout', 
            'BreakoutStrategy': 'breakout'
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
    
    def validate_breakout_parameters(self, params: Dict[str, Any]) -> Dict:
        """BreakoutStrategy パラメータの妥当性検証"""
        errors = []
        warnings = []
        
        # 個別パラメータチェック
        for param_name, param_value in params.items():
            if param_name in self.breakout_rules:
                rule = self.breakout_rules[param_name]
                
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
        logical_errors = self._check_breakout_logical_consistency(params)
        errors.extend(logical_errors)
        
        # 推奨値からの乖離チェック（警告レベル）
        logical_warnings = self._check_breakout_recommended_ranges(params)
        warnings.extend(logical_warnings)
        
        return {
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0,
            "validation_summary": self._generate_validation_summary(errors, warnings)
        }
    
    def validate(self, strategy_name: str, params: Dict[str, Any]) -> Dict:
        """
        戦略名に基づいて適切な検証メソッドを自動選択
        
        Args:
            strategy_name: 戦略名 ('momentum', 'breakout', 'MomentumInvestingStrategy', 'BreakoutStrategy')
            params: 検証するパラメータ辞書
            
        Returns:
            検証結果辞書
        """
        # 戦略名を正規化
        normalized_strategy = self._normalize_strategy_name(strategy_name)
        
        if normalized_strategy == 'momentum':
            return self.validate_momentum_parameters(params)
        elif normalized_strategy == 'breakout':
            return self.validate_breakout_parameters(params)
        else:
            # 未対応戦略の場合はエラーを返す
            return {
                "errors": [f"未対応の戦略です: {strategy_name}"],
                "warnings": [],
                "valid": False,
                "validation_summary": "❌ 未対応戦略"
            }
    
    def _normalize_strategy_name(self, strategy_name: str) -> str:
        """
        戦略名を正規化して標準名に変換
        
        Args:
            strategy_name: 入力された戦略名
            
        Returns:
            正規化された戦略名
        """
        # 戦略名マッピングから検索
        normalized = self.strategy_mapping.get(strategy_name.lower(), None)
        if normalized:
            return normalized
            
        # クラス名からの推定
        if 'momentum' in strategy_name.lower():
            return 'momentum'
        elif 'breakout' in strategy_name.lower():
            return 'breakout'
        
        # デフォルトは空文字（未対応として扱う）
        return ''
    
    def auto_detect_strategy(self, params: Dict[str, Any]) -> str:
        """
        パラメータ内容から戦略を自動推定
        
        Args:
            params: パラメータ辞書
            
        Returns:
            推定された戦略名
        """
        momentum_specific = {'sma_short', 'sma_long', 'rsi_period', 'rsi_lower', 'rsi_upper'}
        breakout_specific = {'look_back', 'breakout_buffer'}
        
        param_keys = set(params.keys())
        
        # BreakoutStrategy固有パラメータがある場合
        if param_keys & breakout_specific:
            return 'breakout'
        
        # MomentumStrategy固有パラメータがある場合
        if param_keys & momentum_specific:
            return 'momentum'
        
        # 共通パラメータのみの場合はデフォルトでmomentum
        return 'momentum'
    
    def validate_auto(self, params: Dict[str, Any], strategy_hint: str = None) -> Dict:
        """
        パラメータから戦略を自動判別して検証
        
        Args:
            params: 検証するパラメータ辞書
            strategy_hint: 戦略のヒント（オプション）
            
        Returns:
            検証結果辞書
        """
        if strategy_hint:
            # ヒントがある場合はそれを使用
            strategy_name = self._normalize_strategy_name(strategy_hint)
        else:
            # パラメータから自動推定
            strategy_name = self.auto_detect_strategy(params)
        
        return self.validate(strategy_name, params)

    # 既存メソッドに後方互換性のための@deprecatedマークを追加（コメントのみ）
    def validate_momentum_parameters_deprecated(self, params: Dict[str, Any]) -> Dict:
        """@deprecated モメンタム戦略パラメータの妥当性検証（後方互換性のため）"""
        return self.validate_momentum_parameters(params)
    
    def validate_breakout_parameters_deprecated(self, params: Dict[str, Any]) -> Dict:
        """@deprecated BreakoutStrategy パラメータの妥当性検証（後方互換性のため）"""
        return self.validate_breakout_parameters(params)
    
    def _check_logical_consistency(self, params: Dict[str, Any]) -> List[str]:
        """モメンタム戦略パラメータの論理的整合性チェック"""
        errors = []
        
        # SMA期間の論理性チェック
        if 'sma_short' in params and 'sma_long' in params:
            if params['sma_short'] >= params['sma_long']:
                errors.append("短期移動平均期間は長期移動平均期間より小さくする必要があります")
        
        # RSI閾値の論理性チェック
        if 'rsi_lower' in params and 'rsi_upper' in params:
            if params['rsi_lower'] >= params['rsi_upper']:
                errors.append("RSI下限閾値はRSI上限閾値より小さくする必要があります")
        
        # 利確・損切りの論理性チェック
        if 'take_profit' in params and 'stop_loss' in params:
            if params['take_profit'] <= params['stop_loss']:
                errors.append("利確レベルは損切りレベルより大きくする必要があります")
        
        return errors
    
    def _check_recommended_ranges(self, params: Dict[str, Any]) -> List[str]:
        """モメンタム戦略パラメータの推奨値からの乖離チェック"""
        warnings = []
        
        # RSI期間の推奨値チェック
        if 'rsi_period' in params:
            if params['rsi_period'] < 14 or params['rsi_period'] > 21:
                warnings.append("RSI期間は14-21の範囲が推奨されます")
        
        # 利確・損切り比率の推奨値チェック
        if 'take_profit' in params and 'stop_loss' in params:
            ratio = params['take_profit'] / params['stop_loss']
            if ratio < 1.5:
                warnings.append("利確/損切り比率は1.5以上が推奨されます")
        
        return warnings
    
    def _check_breakout_logical_consistency(self, params: Dict[str, Any]) -> List[str]:
        """ブレイクアウト戦略パラメータの論理的整合性チェック"""
        errors = []
        
        # 利確とトレーリングストップの関係チェック
        if 'take_profit' in params and 'trailing_stop' in params:
            if params['trailing_stop'] >= params['take_profit']:
                errors.append("トレーリングストップは利確レベルより小さくする必要があります")
        
        # ブレイクアウト判定期間の合理性チェック
        if 'look_back' in params:
            if params['look_back'] < 1:
                errors.append("ブレイクアウト判定期間は1以上である必要があります")
        
        return errors
    
    def _check_breakout_recommended_ranges(self, params: Dict[str, Any]) -> List[str]:
        """ブレイクアウト戦略パラメータの推奨値からの乖離チェック"""
        warnings = []
        
        # 出来高閾値の推奨値チェック
        if 'volume_threshold' in params:
            if params['volume_threshold'] < 1.5:
                warnings.append("出来高閾値は1.5以上が推奨されます")
        
        # ブレイクアウト判定期間の推奨値チェック
        if 'look_back' in params:
            if params['look_back'] > 5:
                warnings.append("ブレイクアウト判定期間は5以下が推奨されます")
        
        return warnings
    
    def _generate_validation_summary(self, errors: List[str], warnings: List[str]) -> str:
        """検証結果のサマリーを生成"""
        if errors:
            return f"❌ エラー{len(errors)}件"
        elif warnings:
            return f"⚠️ 警告{len(warnings)}件"
        else:
            return "✅ 検証完了"
