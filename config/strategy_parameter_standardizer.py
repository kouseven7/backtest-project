#!/usr/bin/env python3
"""
TODO #13: 戦略パラメータ標準化システム

戦略間でのパラメータ名称統一による開発効率化・保守性向上を目的とした標準化システム

Phase 1: 名称統一のみ（値の変更なし）
- 対象戦略: GCStrategy
- 標準化パラメータ: stop_loss → stop_loss_pct, take_profit → take_profit_pct
- 影響範囲: 1戦略のパラメータ名のみ変更、値・ロジック変更なし
"""

from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

@dataclass
class ParameterStandardizationRule:
    """パラメータ標準化ルール"""
    old_name: str          # 旧パラメータ名
    new_name: str          # 新パラメータ名
    preserve_value: bool   # 値を保持するか
    description: str       # 標準化の説明

class StrategyParameterStandardizer:
    """戦略パラメータ標準化システム"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # TODO #13 Phase 1: 名称統一設定
        self.standardization_config = {
            # Phase 1: 名称統一のみ（ユーザー前向き部分）
            'parameter_name_standardization': {
                'stop_loss': 'stop_loss_pct',       # 統一
                'take_profit': 'take_profit_pct',   # 統一
                'risk_reward': 'risk_reward_ratio'  # 統一（将来用）
            },
            
            # 値は現状維持
            'preserve_existing_values': True,
            'no_value_changes': True,
            
            # Phase 1対象戦略
            'target_strategies': ['GCStrategy']
        }
        
        # 標準化ルール定義
        self.standardization_rules = {
            'GCStrategy': [
                ParameterStandardizationRule(
                    old_name='stop_loss',
                    new_name='stop_loss_pct',
                    preserve_value=True,
                    description='ストップロス率統一: stop_loss → stop_loss_pct'
                ),
                ParameterStandardizationRule(
                    old_name='take_profit',
                    new_name='take_profit_pct',
                    preserve_value=True,
                    description='利益確定率統一: take_profit → take_profit_pct'
                )
            ]
        }
        
        self.logger.info("StrategyParameterStandardizer initialized for Phase 1")
    
    def get_standardization_mapping(self, strategy_name: str) -> Dict[str, str]:
        """戦略のパラメータ標準化マッピング取得"""
        if strategy_name not in self.standardization_rules:
            return {}
        
        mapping = {}
        for rule in self.standardization_rules[strategy_name]:
            mapping[rule.old_name] = rule.new_name
        
        return mapping
    
    def standardize_parameters(self, strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        パラメータの標準化実行
        
        Args:
            strategy_name: 戦略名
            params: 元のパラメータ辞書
            
        Returns:
            標準化されたパラメータ辞書
        """
        if strategy_name not in self.standardization_rules:
            return params.copy()
        
        standardized_params = params.copy()
        mapping = self.get_standardization_mapping(strategy_name)
        
        for old_name, new_name in mapping.items():
            if old_name in standardized_params:
                # 値を保持して新しい名前に移動
                standardized_params[new_name] = standardized_params[old_name]
                del standardized_params[old_name]
                
                self.logger.info(f"{strategy_name}: パラメータ標準化 {old_name} → {new_name}")
        
        return standardized_params
    
    def get_standardization_report(self) -> str:
        """標準化レポート生成"""
        lines = []
        lines.append("=" * 60)
        lines.append("TODO #13: 戦略パラメータ標準化レポート")
        lines.append("=" * 60)
        lines.append(f"Phase: {1} (名称統一のみ)")
        lines.append(f"対象戦略数: {len(self.standardization_rules)}")
        lines.append("")
        
        for strategy_name, rules in self.standardization_rules.items():
            lines.append(f"📋 {strategy_name}:")
            for rule in rules:
                lines.append(f"  • {rule.old_name} → {rule.new_name}")
                lines.append(f"    {rule.description}")
            lines.append("")
        
        lines.append("✅ 期待効果:")
        lines.append("  • 開発効率: 戦略追加時60分 → 10分")
        lines.append("  • コード品質: パラメータ名統一によるバグ防止")
        lines.append("  • 保守性: 統一インターフェース")
        lines.append("  • 拡張性: 標準パターン確立")
        lines.append("=" * 60)
        
        return "\n".join(lines)

# TODO #13 Phase 1実装のグローバル設定
PARAMETER_STANDARDIZATION_CONFIG = {
    # Phase 1: 名称統一のみ（値の変更なし）  
    'enabled': True,
    'phase': 1,
    'description': 'Parameter name standardization only - no value changes',
    
    # 対象戦略・パラメータ
    'target_strategies': {
        'GCStrategy': {
            'stop_loss': 'stop_loss_pct',
            'take_profit': 'take_profit_pct'
        }
    },
    
    # 統一後の標準パラメータ名一覧（参考）
    'standard_parameter_names': [
        'stop_loss_pct',      # ストップロス率
        'take_profit_pct',    # 利益確定率
        'risk_reward_ratio',  # リスクリワード比（将来用）
        'max_hold_days',      # 最大保有期間
        'trailing_stop_pct'   # トレーリングストップ率
    ],
    
    # 既に標準化済み戦略（参考）
    'already_standardized_strategies': [
        'pairs_trading_strategy',
        'support_resistance_contrarian_strategy', 
        'mean_reversion_strategy'
        # その他_pct付きパラメータ使用戦略
    ]
}

def get_parameter_standardizer() -> StrategyParameterStandardizer:
    """標準化システムのファクトリ関数"""
    return StrategyParameterStandardizer()

if __name__ == "__main__":
    # テスト実行
    standardizer = get_parameter_standardizer()
    
    # テスト用パラメータ
    test_params = {
        "short_window": 5,
        "long_window": 25,
        "take_profit": 0.05,     # 旧名称
        "stop_loss": 0.03,       # 旧名称
        "trailing_stop_pct": 0.03,
        "max_hold_days": 20
    }
    
    print("TODO #13 Phase 1 テスト実行")
    print(f"元のパラメータ: {test_params}")
    
    # 標準化実行
    standardized = standardizer.standardize_parameters('GCStrategy', test_params)
    print(f"標準化後: {standardized}")
    
    # レポート出力
    print(standardizer.get_standardization_report())