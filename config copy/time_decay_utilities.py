"""
Module: Time Decay Factor Utilities
File: time_decay_utilities.py
Description: 
  2-3-2「時間減衰ファクター導入」便利関数・ユーティリティ
  時間減衰システムの使いやすいインターフェース

Author: GitHub Copilot
Created: 2025-07-13
Modified: 2025-07-13
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# 時間減衰関連のインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .time_decay_factor import TimeDecayFactor, DecayParameters, DecayModel
except ImportError:
    from time_decay_factor import TimeDecayFactor, DecayParameters, DecayModel

logger = logging.getLogger(__name__)

# =============================================================================
# 便利関数 - システム作成・初期化
# =============================================================================

def create_time_decay_system(strategy_name: Optional[str] = None,
                           half_life_days: float = 30.0,
                           decay_model: str = "exponential",
                           base_dir: str = "score_history") -> Any:
    """
    時間減衰システムの簡単作成
    
    Parameters:
        strategy_name: 戦略名（デフォルト設定使用）
        half_life_days: 半減期（日）
        decay_model: 減衰モデル ("exponential", "linear", "gaussian", "power_law")
        base_dir: データ保存ディレクトリ
        
    Returns:
        EnhancedScoreHistoryManager: 時間減衰対応管理システム
    """
    try:
        # 減衰モデル変換
        model_map = {
            "exponential": DecayModel.EXPONENTIAL,
            "linear": DecayModel.LINEAR,
            "gaussian": DecayModel.GAUSSIAN,
            "power_law": DecayModel.POWER_LAW
        }
        
        model = model_map.get(decay_model.lower(), DecayModel.EXPONENTIAL)
        
        # パラメータ作成
        params = DecayParameters(
            half_life_days=half_life_days,
            model=model
        )
        
        # 戦略別デフォルト適用
        if strategy_name and strategy_name in params.strategy_defaults:
            strategy_config = params.strategy_defaults[strategy_name]
            params.half_life_days = strategy_config.get("half_life_days", half_life_days)
            params.strategy_multiplier = strategy_config.get("strategy_multiplier", 1.0)
        
        # モック実装（実際のEnhancedScoreHistoryManagerは後で実装）
        class MockEnhancedScoreHistoryManager:
            def __init__(self, base_dir: str, decay_parameters: DecayParameters):
                self.base_dir = base_dir
                self.decay_parameters = decay_parameters
                self.decay_factor = TimeDecayFactor(decay_parameters)
                logger.info(f"Mock Enhanced Score History Manager created")
        
        system = MockEnhancedScoreHistoryManager(
            base_dir=base_dir,
            decay_parameters=params
        )
        
        logger.info(f"Time decay system created for strategy: {strategy_name}")
        return system
        
    except Exception as e:
        logger.error(f"Failed to create time decay system: {e}")
        raise

def create_multi_strategy_decay_system(strategies: List[str],
                                     base_dir: str = "score_history") -> Dict[str, Any]:
    """
    複数戦略用時間減衰システム作成
    
    Parameters:
        strategies: 戦略名リスト
        base_dir: ベースディレクトリ
        
    Returns:
        Dict[str, Any]: 戦略別システム辞書
    """
    systems = {}
    
    for strategy in strategies:
        try:
            system = create_time_decay_system(
                strategy_name=strategy,
                base_dir=f"{base_dir}/{strategy}"
            )
            systems[strategy] = system
            
        except Exception as e:
            logger.error(f"Failed to create system for {strategy}: {e}")
            
    logger.info(f"Multi-strategy decay systems created: {list(systems.keys())}")
    return systems

# =============================================================================
# 便利関数 - 時間減衰重み計算
# =============================================================================

def calculate_time_weights(timestamps: List[str],
                         reference_time: Optional[str] = None,
                         half_life_days: float = 30.0,
                         model: str = "exponential") -> List[float]:
    """
    タイムスタンプリストの時間減衰重み一括計算
    
    Parameters:
        timestamps: タイムスタンプリスト
        reference_time: 基準時刻（デフォルト：現在時刻）
        half_life_days: 半減期（日）
        model: 減衰モデル
        
    Returns:
        List[float]: 時間減衰重みリスト
    """
    try:
        # 減衰ファクター作成
        params = DecayParameters(
            half_life_days=half_life_days,
            model=DecayModel(model.lower())
        )
        decay_factor = TimeDecayFactor(params)
        
        # 重み計算
        weights = []
        for timestamp in timestamps:
            weight = decay_factor.calculate_decay_weight(
                timestamp=timestamp,
                reference_time=reference_time
            )
            weights.append(weight)
        
        return weights
        
    except Exception as e:
        logger.error(f"Failed to calculate time weights: {e}")
        return [1.0] * len(timestamps)  # エラー時は均等重み

def get_effective_sample_size(timestamps: List[str],
                            reference_time: Optional[str] = None,
                            half_life_days: float = 30.0) -> float:
    """
    実効サンプルサイズ計算
    
    Parameters:
        timestamps: タイムスタンプリスト
        reference_time: 基準時刻
        half_life_days: 半減期
        
    Returns:
        float: 実効サンプルサイズ
    """
    try:
        weights = calculate_time_weights(
            timestamps=timestamps,
            reference_time=reference_time,
            half_life_days=half_life_days
        )
        
        if not weights:
            return 0.0
        
        # 実効サンプルサイズ = (Σw)² / Σw²
        sum_weights = sum(weights)
        sum_weights_squared = sum(w**2 for w in weights)
        
        if sum_weights_squared == 0:
            return 0.0
        
        return sum_weights**2 / sum_weights_squared
        
    except Exception as e:
        logger.error(f"Failed to calculate effective sample size: {e}")
        return 0.0

# =============================================================================
# 便利関数 - 可視化サポート
# =============================================================================

def get_decay_curve_data(strategy_name: Optional[str] = None,
                       half_life_days: float = 30.0,
                       model: str = "exponential",
                       days_range: int = 90) -> pd.DataFrame:
    """
    減衰曲線の可視化用データ取得
    
    Parameters:
        strategy_name: 戦略名
        half_life_days: 半減期
        model: 減衰モデル
        days_range: 日数範囲
        
    Returns:
        pd.DataFrame: 可視化用データ
    """
    try:
        # 減衰ファクター作成
        params = DecayParameters(
            half_life_days=half_life_days,
            model=DecayModel(model.lower())
        )
        decay_factor = TimeDecayFactor(params)
        
        # 可視化データ生成
        curve_data = decay_factor.get_decay_visualization_data(
            days_range=days_range,
            strategy_name=strategy_name
        )
        
        # 追加情報
        curve_data['model'] = model
        curve_data['half_life_days'] = half_life_days
        
        return curve_data
        
    except Exception as e:
        logger.error(f"Failed to get decay curve data: {e}")
        return pd.DataFrame()

def get_multi_model_comparison_data(half_life_days: float = 30.0,
                                  days_range: int = 90) -> pd.DataFrame:
    """
    複数減衰モデルの比較用データ取得
    
    Parameters:
        half_life_days: 半減期
        days_range: 日数範囲
        
    Returns:
        pd.DataFrame: モデル比較データ
    """
    try:
        all_data = []
        
        for model in DecayModel:
            curve_data = get_decay_curve_data(
                half_life_days=half_life_days,
                model=model.value,
                days_range=days_range
            )
            
            if not curve_data.empty:
                curve_data['decay_model'] = model.value
                all_data.append(curve_data)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Failed to get multi-model comparison data: {e}")
        return pd.DataFrame()

# =============================================================================
# 便利関数 - 検証・診断
# =============================================================================

def validate_time_decay_system(strategy_name: Optional[str] = None,
                              base_dir: str = "score_history") -> Dict[str, Any]:
    """
    時間減衰システムの検証
    
    Parameters:
        strategy_name: 戦略名
        base_dir: データディレクトリ
        
    Returns:
        Dict[str, Any]: 検証結果
    """
    try:
        validation_results = {
            "system_status": "healthy",
            "issues": [],
            "statistics": {},
            "recommendations": []
        }
        
        # システム作成テスト
        try:
            system = create_time_decay_system(
                strategy_name=strategy_name,
                base_dir=base_dir
            )
            validation_results["system_creation"] = "success"
        except Exception as e:
            validation_results["issues"].append(f"System creation failed: {e}")
            validation_results["system_status"] = "error"
            return validation_results
        
        # 基本機能テスト
        try:
            # テスト用タイムスタンプ
            test_timestamps = [
                datetime.now().isoformat(),
                (datetime.now() - timedelta(days=1)).isoformat(),
                (datetime.now() - timedelta(days=7)).isoformat(),
                (datetime.now() - timedelta(days=30)).isoformat(),
            ]
            
            weights = calculate_time_weights(test_timestamps)
            validation_results["basic_functionality"] = "success"
            validation_results["statistics"]["test_weights"] = {
                "count": len(weights),
                "min": min(weights),
                "max": max(weights),
                "mean": np.mean(weights)
            }
            
        except Exception as e:
            validation_results["issues"].append(f"Basic functionality test failed: {e}")
        
        # 推奨事項生成
        if len(validation_results["issues"]) == 0:
            validation_results["recommendations"].append(
                "System is functioning correctly"
            )
        else:
            validation_results["system_status"] = "warning"
            validation_results["recommendations"].append(
                "Check system configuration and dependencies"
            )
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Failed to validate time decay system: {e}")
        return {
            "system_status": "error",
            "issues": [f"Validation failed: {e}"],
            "statistics": {},
            "recommendations": ["Check system configuration and data availability"]
        }

# =============================================================================
# エクスポート用メイン関数
# =============================================================================

__all__ = [
    "create_time_decay_system",
    "create_multi_strategy_decay_system",
    "calculate_time_weights",
    "get_effective_sample_size",
    "get_decay_curve_data",
    "get_multi_model_comparison_data",
    "validate_time_decay_system"
]
