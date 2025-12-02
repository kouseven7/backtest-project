"""
5-3-3 戦略間相関を考慮した配分最適化 - 設定ファイル

システム全体のデフォルト設定を管理

Author: imega
Created: 2025-01-27
Task: 5-3-3
"""

from dataclasses import dataclass, field
from typing import Any

# メインクラスのインポート
from ..correlation_based_allocator import AllocationConfig
from ..optimization_engine import OptimizationConfig
from ..constraint_manager import ConstraintConfig
from ..integration_bridge import IntegrationConfig

@dataclass
class SystemConfig:
    """システム全体設定"""
    # 各コンポーネント設定
    allocation_config: AllocationConfig = field(default_factory=AllocationConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    constraint_config: ConstraintConfig = field(default_factory=ConstraintConfig)
    integration_config: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    # システムレベル設定
    system_name: str = "5-3-3_correlation_based_optimizer"
    version: str = "1.0.0"
    debug_mode: bool = False
    performance_monitoring: bool = True

# デフォルト設定の定義

# 保守的設定（リスク重視）
CONSERVATIVE_CONFIG = SystemConfig(
    allocation_config=AllocationConfig(
        correlation_timeframes={
            'short_term': 30,
            'medium_term': 90,
            'long_term': 252
        },
        timeframe_weights={
            'short_term': 0.2,
            'medium_term': 0.3,
            'long_term': 0.5  # 長期重視
        },
        optimization_methods={
            'mean_variance': 0.30,
            'risk_parity': 0.50,  # リスクパリティ重視
            'hierarchical_risk_parity': 0.20
        },
        min_weight=0.10,  # より高い最小重み
        max_weight=0.25,  # より低い最大重み
        max_concentration=0.50,
        correlation_penalty_threshold=0.6,  # より厳しい相関制限
        turnover_limit=0.15,  # より低いターンオーバー
        risk_aversion=3.0  # より高いリスク回避度
    ),
    optimization_config=OptimizationConfig(
        risk_aversion=3.0,
        max_weight=0.25,
        max_turnover=0.15,
        max_concentration=0.50,
        regularization=1e-6
    ),
    constraint_config=ConstraintConfig(
        min_weight=0.10,
        max_weight=0.25,
        max_turnover=0.15,
        max_single_weight=0.25,
        max_top_n_concentration={3: 0.50, 5: 0.75},
        max_correlation_exposure=0.6,
        correlation_penalty_threshold=0.6,
        max_portfolio_volatility=0.15  # 年率15%制限
    ),
    integration_config=IntegrationConfig(
        integration_level="basic",
        correlation_data_priority=0.8,
        weight_adjustment_factor=0.9
    )
)

# バランス設定（デフォルト）
BALANCED_CONFIG = SystemConfig(
    allocation_config=AllocationConfig(
        correlation_timeframes={
            'short_term': 30,
            'medium_term': 90,
            'long_term': 252
        },
        timeframe_weights={
            'short_term': 0.3,
            'medium_term': 0.4,
            'long_term': 0.3
        },
        optimization_methods={
            'mean_variance': 0.40,
            'risk_parity': 0.35,
            'hierarchical_risk_parity': 0.25
        },
        min_weight=0.05,
        max_weight=0.40,
        max_concentration=0.60,
        correlation_penalty_threshold=0.7,
        turnover_limit=0.20,
        risk_aversion=2.0
    ),
    optimization_config=OptimizationConfig(
        risk_aversion=2.0,
        max_weight=0.40,
        max_turnover=0.20,
        max_concentration=0.60,
        regularization=1e-8
    ),
    constraint_config=ConstraintConfig(
        min_weight=0.05,
        max_weight=0.40,
        max_turnover=0.20,
        max_single_weight=0.40,
        max_top_n_concentration={3: 0.60, 5: 0.80},
        max_correlation_exposure=0.7,
        correlation_penalty_threshold=0.7
    ),
    integration_config=IntegrationConfig(
        integration_level="moderate",
        correlation_data_priority=0.7,
        weight_adjustment_factor=0.8
    )
)

# アグレッシブ設定（リターン重視）
AGGRESSIVE_CONFIG = SystemConfig(
    allocation_config=AllocationConfig(
        correlation_timeframes={
            'short_term': 21,  # より短期
            'medium_term': 63,
            'long_term': 189
        },
        timeframe_weights={
            'short_term': 0.5,  # 短期重視
            'medium_term': 0.3,
            'long_term': 0.2
        },
        optimization_methods={
            'mean_variance': 0.60,  # 平均分散重視
            'risk_parity': 0.20,
            'hierarchical_risk_parity': 0.20
        },
        min_weight=0.02,  # より低い最小重み
        max_weight=0.60,  # より高い最大重み
        max_concentration=0.75,
        correlation_penalty_threshold=0.8,  # より緩い相関制限
        turnover_limit=0.30,  # より高いターンオーバー
        risk_aversion=1.0,  # より低いリスク回避度
        expected_return_adjustment=True,
        score_weight=0.5  # スコア重視
    ),
    optimization_config=OptimizationConfig(
        risk_aversion=1.0,
        max_weight=0.60,
        max_turnover=0.30,
        max_concentration=0.75,
        regularization=1e-10
    ),
    constraint_config=ConstraintConfig(
        min_weight=0.02,
        max_weight=0.60,
        max_turnover=0.30,
        max_single_weight=0.60,
        max_top_n_concentration={3: 0.75, 5: 0.90},
        max_correlation_exposure=0.8,
        correlation_penalty_threshold=0.8,
        soft_constraint_penalty=500.0  # より緩いペナルティ
    ),
    integration_config=IntegrationConfig(
        integration_level="advanced",
        correlation_data_priority=0.6,
        weight_adjustment_factor=0.7,
        score_integration_weight=0.5
    )
)

# 多様化重視設定
DIVERSIFICATION_CONFIG = SystemConfig(
    allocation_config=AllocationConfig(
        correlation_timeframes={
            'short_term': 30,
            'medium_term': 90,
            'long_term': 252
        },
        timeframe_weights={
            'short_term': 0.25,
            'medium_term': 0.35,
            'long_term': 0.40
        },
        optimization_methods={
            'mean_variance': 0.20,
            'risk_parity': 0.30,
            'hierarchical_risk_parity': 0.50  # 階層化重視
        },
        min_weight=0.08,  # より均等
        max_weight=0.20,  # より制限的
        max_concentration=0.45,  # より厳しい
        correlation_penalty_threshold=0.5,  # 非常に厳しい
        turnover_limit=0.12,
        risk_aversion=2.5
    ),
    optimization_config=OptimizationConfig(
        risk_aversion=2.5,
        max_weight=0.20,
        max_turnover=0.12,
        max_concentration=0.45,
        clustering_method="complete"  # より厳密なクラスタリング
    ),
    constraint_config=ConstraintConfig(
        min_weight=0.08,
        max_weight=0.20,
        max_turnover=0.12,
        max_single_weight=0.20,
        max_top_n_concentration={3: 0.45, 5: 0.65},
        max_correlation_exposure=0.5,
        correlation_penalty_threshold=0.5,
        correlation_penalty_factor=3.0  # 高い相関ペナルティ
    ),
    integration_config=IntegrationConfig(
        integration_level="moderate",
        correlation_data_priority=0.8,  # 相関データ重視
        weight_adjustment_factor=0.9
    )
)

# 設定プリセット辞書
CONFIG_PRESETS = {
    'conservative': CONSERVATIVE_CONFIG,
    'balanced': BALANCED_CONFIG,
    'aggressive': AGGRESSIVE_CONFIG,
    'diversification': DIVERSIFICATION_CONFIG
}

def get_config_preset(preset_name: str) -> SystemConfig:
    """
    設定プリセット取得
    
    Args:
        preset_name: プリセット名
        
    Returns:
        システム設定
    """
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(CONFIG_PRESETS.keys())}")
    
    return CONFIG_PRESETS[preset_name]

def create_custom_config(
    base_preset: str = 'balanced',
    **customizations: Any
) -> SystemConfig:
    """
    カスタム設定作成
    
    Args:
        base_preset: ベースプリセット
        **customizations: カスタマイズパラメータ
        
    Returns:
        カスタム設定
    """
    
    base_config = get_config_preset(base_preset)
    
    # カスタマイズ適用
    if 'risk_aversion' in customizations:
        base_config.allocation_config.risk_aversion = customizations['risk_aversion']
        base_config.optimization_config.risk_aversion = customizations['risk_aversion']
    
    if 'max_weight' in customizations:
        base_config.allocation_config.max_weight = customizations['max_weight']
        base_config.optimization_config.max_weight = customizations['max_weight']
        base_config.constraint_config.max_weight = customizations['max_weight']
    
    if 'turnover_limit' in customizations:
        base_config.allocation_config.turnover_limit = customizations['turnover_limit']
        base_config.optimization_config.max_turnover = customizations['turnover_limit']
        base_config.constraint_config.max_turnover = customizations['turnover_limit']
    
    if 'integration_level' in customizations:
        base_config.integration_config.integration_level = customizations['integration_level']
    
    return base_config
