"""
Module: Enhanced Score History Manager
File: enhanced_score_history_manager.py
Description: 
  2-3-2「時間減衰ファクター導入」統合実装
  ScoreHistoryManagerと時間減衰ファクターの統合システム

Author: GitHub Copilot
Created: 2025-07-13
Modified: 2025-07-13
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

# 関連モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .score_history_manager import ScoreHistoryManager, ScoreHistoryEntry, ScoreHistoryConfig
    from .time_decay_factor import TimeDecayFactor, DecayParameters, DecayModel
    from .strategy_scoring_model import StrategyScore
except ImportError:
    from score_history_manager import ScoreHistoryManager, ScoreHistoryEntry, ScoreHistoryConfig
    from time_decay_factor import TimeDecayFactor, DecayParameters, DecayModel
    from strategy_scoring_model import StrategyScore

logger = logging.getLogger(__name__)

# =============================================================================
# 統合設定データクラス
# =============================================================================

@dataclass
class EnhancedScoreHistoryConfig:
    """時間減衰対応スコア履歴設定"""
    
    # 基本設定
    base_dir: str = "score_history"
    max_entries: int = 10000
    auto_cleanup: bool = True
    cleanup_days: int = 365
    
    # 時間減衰設定
    enable_time_decay: bool = True
    default_half_life_days: float = 30.0
    default_decay_model: DecayModel = DecayModel.EXPONENTIAL
    
    # 戦略別設定
    strategy_specific_decay: Optional[Dict[str, DecayParameters]] = None
    
    # パフォーマンス設定
    cache_decay_weights: bool = True
    cache_ttl_hours: int = 24
    batch_processing_size: int = 1000
    
    # ログ設定
    enable_debug_logging: bool = False
    log_decay_calculations: bool = False
    
    def __post_init__(self):
        if self.strategy_specific_decay is None:
            self.strategy_specific_decay = {}

# =============================================================================
# 時間減衰対応スコア履歴エントリ
# =============================================================================

@dataclass
class EnhancedScoreHistoryEntry(ScoreHistoryEntry):
    """時間減衰重み付きスコア履歴エントリ"""
    
    # 時間減衰関連
    decay_weight: Optional[float] = None
    reference_time: Optional[str] = None
    decay_parameters: Optional[Dict[str, Any]] = None
    
    # 統計情報
    weighted_score: Optional[float] = None
    effective_weight: Optional[float] = None
    
    def calculate_decay_weight(self, 
                             reference_time: Optional[str] = None,
                             decay_params: Optional[DecayParameters] = None) -> float:
        """時間減衰重み計算"""
        try:
            if not decay_params:
                # デフォルトパラメータ使用
                decay_params = DecayParameters()
            
            decay_factor = TimeDecayFactor(decay_params)
            weight = decay_factor.calculate_decay_weight(
                timestamp=self.timestamp,
                reference_time=reference_time
            )
            
            self.decay_weight = weight
            self.reference_time = reference_time or datetime.now().isoformat()
            self.decay_parameters = decay_params.to_dict()
            
            return weight
            
        except Exception as e:
            logger.error(f"Failed to calculate decay weight: {e}")
            return 1.0
    
    def get_weighted_score(self) -> Optional[float]:
        """重み付きスコア取得"""
        if self.decay_weight is None or self.strategy_score is None:
            return None
        
        if hasattr(self.strategy_score, 'total_score'):
            return self.strategy_score.total_score * self.decay_weight
        
        return None

# =============================================================================
# メイン：Enhanced Score History Manager
# =============================================================================

class EnhancedScoreHistoryManager(ScoreHistoryManager):
    """時間減衰対応スコア履歴管理システム"""
    
    def __init__(self, config: Optional[EnhancedScoreHistoryConfig] = None):
        """
        初期化
        
        Parameters:
            config: 統合設定
        """
        self.enhanced_config = config or EnhancedScoreHistoryConfig()
        
        # 基本設定を継承
        base_config = ScoreHistoryConfig(
            base_dir=self.enhanced_config.base_dir,
            max_entries=self.enhanced_config.max_entries,
            auto_cleanup=self.enhanced_config.auto_cleanup,
            cleanup_days=self.enhanced_config.cleanup_days
        )
        
        super().__init__(base_config)
        
        # 時間減衰システム初期化
        self.time_decay_systems: Dict[str, TimeDecayFactor] = {}
        self.decay_weight_cache: Dict[str, Dict[str, float]] = {}
        
        # デバッグログ設定
        if self.enhanced_config.enable_debug_logging:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
        
        logger.info("Enhanced Score History Manager initialized")
    
    # =========================================================================
    # 時間減衰システム管理
    # =========================================================================
    
    def get_decay_factor(self, strategy_name: str) -> TimeDecayFactor:
        """戦略別時間減衰ファクター取得"""
        if strategy_name not in self.time_decay_systems:
            # 戦略別設定確認
            if (self.enhanced_config.strategy_specific_decay and 
                strategy_name in self.enhanced_config.strategy_specific_decay):
                params = self.enhanced_config.strategy_specific_decay[strategy_name]
            else:
                # デフォルト設定使用
                params = DecayParameters(
                    half_life_days=self.enhanced_config.default_half_life_days,
                    model=self.enhanced_config.default_decay_model
                )
        
        self.time_decay_systems[strategy_name] = TimeDecayFactor(params)
        logger.debug(f"Created decay factor for strategy: {strategy_name}")
        
        return self.time_decay_systems[strategy_name]
    
    def set_strategy_decay_parameters(self, 
                                    strategy_name: str, 
                                    decay_params: DecayParameters):
        """戦略別減衰パラメータ設定"""
        if self.enhanced_config.strategy_specific_decay is None:
            self.enhanced_config.strategy_specific_decay = {}
        
        self.enhanced_config.strategy_specific_decay[strategy_name] = decay_params
        
        # キャッシュ無効化
        if strategy_name in self.time_decay_systems:
            del self.time_decay_systems[strategy_name]
        
        if strategy_name in self.decay_weight_cache:
            del self.decay_weight_cache[strategy_name]
        
        logger.info(f"Updated decay parameters for strategy: {strategy_name}")
    
    # =========================================================================
    # 時間減衰対応データ追加・更新
    # =========================================================================
    
    def add_enhanced_entry(self, 
                         strategy_name: str,
                         strategy_score: StrategyScore,
                         timestamp: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         calculate_decay: bool = True) -> str:
        """
        時間減衰対応エントリ追加
        
        Parameters:
            strategy_name: 戦略名
            strategy_score: 戦略スコア
            timestamp: タイムスタンプ
            metadata: メタデータ
            calculate_decay: 時間減衰計算実行フラグ
            
        Returns:
            str: エントリID
        """
        try:
            # 拡張エントリ作成
            entry = EnhancedScoreHistoryEntry(
                strategy_name=strategy_name,
                strategy_score=strategy_score,
                timestamp=timestamp or datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            # 時間減衰計算
            if calculate_decay and self.enhanced_config.enable_time_decay:
                decay_factor = self.get_decay_factor(strategy_name)
                entry.calculate_decay_weight(
                    reference_time=None,  # 現在時刻を基準
                    decay_params=decay_factor.parameters
                )
            
            # 基本システムに追加
            entry_id = super().add_entry(
                strategy_name=strategy_name,
                strategy_score=strategy_score,
                timestamp=entry.timestamp,
                metadata=entry.metadata
            )
            
            # 拡張情報保存
            self._save_enhanced_entry_data(entry_id, entry)
            
            logger.debug(f"Added enhanced entry: {entry_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Failed to add enhanced entry: {e}")
            raise
    
    def _save_enhanced_entry_data(self, entry_id: str, entry: EnhancedScoreHistoryEntry):
        """拡張エントリデータ保存"""
        try:
            enhanced_data = {
                "decay_weight": entry.decay_weight,
                "reference_time": entry.reference_time,
                "decay_parameters": entry.decay_parameters,
                "weighted_score": entry.get_weighted_score()
            }
            
            enhanced_file = self.base_dir / "enhanced" / f"{entry_id}.json"
            enhanced_file.parent.mkdir(exist_ok=True)
            
            with open(enhanced_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to save enhanced entry data: {e}")
    
    # =========================================================================
    # 時間減衰対応データ取得
    # =========================================================================
    
    def get_weighted_scores(self, 
                          strategy_name: str,
                          reference_time: Optional[str] = None,
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        時間減衰重み付きスコア取得
        
        Parameters:
            strategy_name: 戦略名
            reference_time: 基準時刻
            limit: 最大取得数
            
        Returns:
            List[Dict[str, Any]]: 重み付きスコアリスト
        """
        try:
            # 基本エントリ取得
            entries = self.get_entries(strategy_name, limit=limit)
            
            if not entries:
                return []
            
            # 時間減衰ファクター取得
            decay_factor = self.get_decay_factor(strategy_name)
            
            weighted_scores = []
            
            for entry in entries:
                # 時間減衰重み計算
                weight = decay_factor.calculate_decay_weight(
                    timestamp=entry.timestamp,
                    reference_time=reference_time
                )
                
                # 重み付きスコア計算
                if hasattr(entry.strategy_score, 'total_score'):
                    weighted_score = entry.strategy_score.total_score * weight
                else:
                    weighted_score = None
                
                weighted_data = {
                    "entry_id": entry.entry_id,
                    "timestamp": entry.timestamp,
                    "original_score": entry.strategy_score.total_score if hasattr(entry.strategy_score, 'total_score') else None,
                    "decay_weight": weight,
                    "weighted_score": weighted_score,
                    "reference_time": reference_time or datetime.now().isoformat()
                }
                
                weighted_scores.append(weighted_data)
            
            return weighted_scores
            
        except Exception as e:
            logger.error(f"Failed to get weighted scores: {e}")
            return []
    
    def calculate_aggregated_metrics(self, 
                                   strategy_name: str,
                                   reference_time: Optional[str] = None,
                                   metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        時間減衰集約メトリクス計算
        
        Parameters:
            strategy_name: 戦略名
            reference_time: 基準時刻
            metrics: 計算対象メトリクス
            
        Returns:
            Dict[str, float]: 集約メトリクス
        """
        try:
            weighted_scores = self.get_weighted_scores(
                strategy_name=strategy_name,
                reference_time=reference_time
            )
            
            if not weighted_scores:
                return {}
            
            # 重みと重み付きスコア抽出
            weights = [ws["decay_weight"] for ws in weighted_scores if ws["decay_weight"] is not None]
            w_scores = [ws["weighted_score"] for ws in weighted_scores if ws["weighted_score"] is not None]
            
            if not weights or not w_scores:
                return {}
            
            # 基本統計計算
            aggregated = {
                "total_entries": len(weighted_scores),
                "valid_weights": len(weights),
                "sum_weights": sum(weights),
                "weighted_average_score": sum(w_scores) / sum(weights) if sum(weights) > 0 else 0.0,
                "effective_sample_size": sum(weights)**2 / sum(w**2 for w in weights) if weights else 0.0
            }
            
            # 追加メトリクス
            if metrics:
                if "weighted_std" in metrics:
                    mean_score = aggregated["weighted_average_score"]
                    weighted_var = sum(w * (score/w - mean_score)**2 for w, score in zip(weights, w_scores) if w > 0) / sum(weights)
                    aggregated["weighted_std"] = np.sqrt(weighted_var) if weighted_var >= 0 else 0.0
                
                if "decay_efficiency" in metrics:
                    # 減衰効率 = 重み付き平均 / 単純平均
                    simple_avg = np.mean([ws["original_score"] for ws in weighted_scores if ws["original_score"] is not None])
                    aggregated["decay_efficiency"] = aggregated["weighted_average_score"] / simple_avg if simple_avg != 0 else 1.0
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to calculate aggregated metrics: {e}")
            return {}
    
    # =========================================================================
    # ユーティリティ・診断
    # =========================================================================
    
    def get_decay_statistics(self, strategy_name: str) -> Dict[str, Any]:
        """時間減衰統計情報取得"""
        try:
            decay_factor = self.get_decay_factor(strategy_name)
            
            # パラメータ情報
            params_info = {
                "half_life_days": decay_factor.parameters.half_life_days,
                "model": decay_factor.parameters.model.value,
                "strategy_multiplier": decay_factor.parameters.strategy_multiplier
            }
            
            # 最近のエントリ数
            recent_entries = len(self.get_entries(strategy_name, limit=100))
            
            # 可視化データ
            viz_data = decay_factor.get_decay_visualization_data(
                days_range=90,
                strategy_name=strategy_name
            )
            
            statistics = {
                "strategy_name": strategy_name,
                "parameters": params_info,
                "recent_entries_count": recent_entries,
                "visualization_data": viz_data.to_dict() if not viz_data.empty else {}
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Failed to get decay statistics: {e}")
            return {}
    
    def validate_system_health(self) -> Dict[str, Any]:
        """システムヘルス検証"""
        try:
            health_report = {
                "overall_status": "healthy",
                "issues": [],
                "strategies": {},
                "performance": {}
            }
            
            # 戦略別チェック
            for strategy_name in self.time_decay_systems.keys():
                try:
                    stats = self.get_decay_statistics(strategy_name)
                    health_report["strategies"][strategy_name] = {
                        "status": "healthy",
                        "entries_count": stats.get("recent_entries_count", 0)
                    }
                except Exception as e:
                    health_report["issues"].append(f"Strategy {strategy_name}: {e}")
                    health_report["strategies"][strategy_name] = {"status": "error"}
            
            # パフォーマンス統計
            health_report["performance"] = {
                "cached_strategies": len(self.time_decay_systems),
                "cache_entries": sum(len(cache) for cache in self.decay_weight_cache.values())
            }
            
            # 全体ステータス判定
            if health_report["issues"]:
                health_report["overall_status"] = "warning"
            
            return health_report
            
        except Exception as e:
            logger.error(f"Failed to validate system health: {e}")
            return {
                "overall_status": "error",
                "issues": [f"Health check failed: {e}"]
            }

# =============================================================================
# エクスポート
# =============================================================================

__all__ = [
    "EnhancedScoreHistoryConfig",
    "EnhancedScoreHistoryEntry", 
    "EnhancedScoreHistoryManager"
]
