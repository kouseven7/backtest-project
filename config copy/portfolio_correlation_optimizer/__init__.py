"""
5-3-3「戦略間相関を考慮した配分最適化」パッケージ

このパッケージは、戦略間の相関関係を考慮した高度な
ポートフォリオ配分最適化システムを提供します。

Components:
- CorrelationBasedAllocator: メイン配分エンジン
- OptimizationEngine: 最適化計算エンジン  
- ConstraintManager: 制約管理システム
- IntegrationBridge: 既存システム統合

Author: imega
Created: 2025-01-27
Task: 5-3-3
"""

# メインクラスのインポート
try:
    from .correlation_based_allocator import (
        CorrelationBasedAllocator,
        AllocationConfig,
        AllocationResult,
        OptimizationStatus
    )
    
    from .optimization_engine import (
        HybridOptimizationEngine,
        OptimizationMethod,
        OptimizationConfig,
        OptimizationResult
    )
    
    from .constraint_manager import (
        CorrelationConstraintManager,
        ConstraintType,
        ConstraintResult,
        ConstraintViolation
    )
    
    from .integration_bridge import (
        SystemIntegrationBridge,
        IntegrationConfig,
        BridgeResult
    )
    
    __all__ = [
        # Main allocator
        'CorrelationBasedAllocator',
        'AllocationConfig', 
        'AllocationResult',
        'OptimizationStatus',
        
        # Optimization engine
        'HybridOptimizationEngine',
        'OptimizationMethod',
        'OptimizationConfig', 
        'OptimizationResult',
        
        # Constraint management
        'CorrelationConstraintManager',
        'ConstraintType',
        'ConstraintResult',
        'ConstraintViolation',
        
        # System integration
        'SystemIntegrationBridge',
        'IntegrationConfig',
        'BridgeResult'
    ]

except ImportError as e:
    print(f"Warning: Failed to import some components: {e}")
    __all__ = []

__version__ = "1.0.0"
