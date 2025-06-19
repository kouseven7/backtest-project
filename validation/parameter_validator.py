"""
パラメータ妥当性検証（戦略対応版）
"""
from typing import Dict, Any, Optional
from validation.validators.momentum_validator import MomentumParameterValidator
from validation.validators.breakout_validator import BreakoutParameterValidator
from validation.validators.contrarian_validator import ContrarianParameterValidator
from validation.validators.opening_gap_validator import OpeningGapParameterValidator

class ParameterValidator:
    def __init__(self):
        # 戦略名マッピング（自動判別用）
        self.strategy_mapping = {
            'momentum': 'momentum',
            'MomentumInvestingStrategy': 'momentum',
            'breakout': 'breakout',
            'BreakoutStrategy': 'breakout',
            'contrarian': 'contrarian',
            'ContrarianStrategy': 'contrarian',
            'gc': 'gc',
            'GCStrategy': 'gc',
            'gc_strategy': 'gc',
            'opening_gap': 'opening_gap',
            'OpeningGapStrategy': 'opening_gap',
        }
        from validation.validators.gc_validator import GCParameterValidator
        self.validator_map: Dict[str, Any] = {
            'momentum': MomentumParameterValidator,
            'breakout': BreakoutParameterValidator,
            'contrarian': ContrarianParameterValidator,
            'gc': GCParameterValidator,
            'opening_gap': OpeningGapParameterValidator
        }

    def validate_momentum_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return MomentumParameterValidator.validate(params)

    def validate_breakout_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return BreakoutParameterValidator.validate(params)

    def validate_contrarian_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return ContrarianParameterValidator.validate(params)

    def validate_gc_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from validation.validators.gc_validator import GCParameterValidator
        return GCParameterValidator().validate(params)

    def validate(self, strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        戦略名に基づいて適切な検証メソッドを自動選択
        
        Args:
            strategy_name: 戦略名 ('momentum', 'breakout', 'MomentumInvestingStrategy', 'BreakoutStrategy')
            params: 検証するパラメータ辞書
            
        Returns:
            検証結果辞書
        """
        normalized_strategy = self._normalize_strategy_name(strategy_name)
        validator_cls = self.validator_map.get(normalized_strategy)
        if validator_cls:
            return validator_cls.validate(params)
        else:
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
        elif 'contrarian' in strategy_name.lower():
            return 'contrarian'
        elif 'gc' in strategy_name.lower():
            return 'gc'
        elif 'opening_gap' in strategy_name.lower():
            return 'opening_gap'
        
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
        contrarian_specific = {'entry_threshold', 'exit_threshold'}
        gc_specific = {'gc_param1', 'gc_param2'}  # GCStrategy固有パラメータの例
        opening_gap_specific = {'gap_size', 'slippage'}  # OpeningGapStrategy固有パラメータの例
        
        param_keys = set(params.keys())
        
        # BreakoutStrategy固有パラメータがある場合
        if param_keys & breakout_specific:
            return 'breakout'
        
        # MomentumStrategy固有パラメータがある場合
        if param_keys & momentum_specific:
            return 'momentum'
        
        # ContrarianStrategy固有パラメータがある場合
        if param_keys & contrarian_specific:
            return 'contrarian'
        
        # GCStrategy固有パラメータがある場合
        if param_keys & gc_specific:
            return 'gc'
        
        # OpeningGapStrategy固有パラメータがある場合
        if param_keys & opening_gap_specific:
            return 'opening_gap'
        
        # 共通パラメータのみの場合はデフォルトでmomentum
        return 'momentum'
    
    def validate_auto(self, params: Dict[str, Any], strategy_hint: Optional[str] = None) -> Dict[str, Any]:
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
    def validate_momentum_parameters_deprecated(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """@deprecated モメンタム戦略パラメータの妥当性検証（後方互換性のため）"""
        return self.validate_momentum_parameters(params)
    
    def validate_breakout_parameters_deprecated(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """@deprecated BreakoutStrategy パラメータの妥当性検証（後方互換性のため）"""
        return self.validate_breakout_parameters(params)
    
    def validate_contrarian_parameters_deprecated(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """@deprecated ContrarianStrategy パラメータの妥当性検証（後方互換性のため）"""
        return self.validate_contrarian_parameters(params)
