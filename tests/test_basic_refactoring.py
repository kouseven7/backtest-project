"""
基本的な統合テスト
リファクタリング後の基本機能確認
"""

import sys
import os
import pytest
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_src_imports():
    """srcパッケージの基本インポートテスト"""
    import src
    assert src is not None

def test_config_imports():
    """設定モジュールのインポートテスト"""
    from src.config.logger_config import setup_logger
    from src.config.risk_management import RiskManagement
    from src.config.optimized_parameters import OptimizedParameterManager
    
    # 基本的なインスタンス作成
    logger = setup_logger("test")
    risk_manager = RiskManagement(total_assets=1000000)
    param_manager = OptimizedParameterManager()
    
    assert logger is not None
    assert risk_manager is not None
    assert param_manager is not None

def test_strategies_imports():
    """戦略モジュールのインポートテスト"""
    from src.strategies.VWAP_Breakout import VWAPBreakoutStrategy
    from src.strategies.Momentum_Investing import MomentumInvestingStrategy
    from src.strategies.Breakout import BreakoutStrategy
    from src.strategies.VWAP_Bounce import VWAPBounceStrategy
    
    assert VWAPBreakoutStrategy is not None
    assert MomentumInvestingStrategy is not None
    assert BreakoutStrategy is not None
    assert VWAPBounceStrategy is not None

def test_main_function_imports():
    """main.pyの主要関数インポートテスト"""
    from src.main import load_optimized_parameters
    from src.main import get_default_parameters
    
    assert load_optimized_parameters is not None
    assert get_default_parameters is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
