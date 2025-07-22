"""
Module: Time Decay Factor
File: time_decay_factor.py
Description: 
  2-3-2「時間減衰ファクター導入」
  新しいデータを重視する指数減衰機能

Author: GitHub Copilot
Created: 2025-07-13
Modified: 2025-07-13
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DecayModel(Enum):
    """減衰モデルの種類"""
    EXPONENTIAL = "exponential"      # 指数減衰
    LINEAR = "linear"               # 線形減衰
    GAUSSIAN = "gaussian"           # ガウシアン減衰
    POWER_LAW = "power_law"         # べき乗減衰

@dataclass
class DecayParameters:
    """戦略別減衰パラメータ"""
    half_life_days: float = 30.0           # 半減期（日）
    model: DecayModel = DecayModel.EXPONENTIAL  # 減衰モデル
    min_weight: float = 0.01               # 最小重み
    volatility_adjustment: bool = True      # ボラティリティ調整
    strategy_multiplier: float = 1.0       # 戦略別倍率
    
    # 戦略別デフォルト設定
    strategy_defaults: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "VWAP_Bounce": {
            "half_life_days": 30.0,
            "strategy_multiplier": 1.0,
            "volatility_adjustment": True
        },
        "Momentum_Investing": {
            "half_life_days": 45.0,
            "strategy_multiplier": 1.1,
            "volatility_adjustment": True
        },
        "Mean_Reversion": {
            "half_life_days": 21.0,
            "strategy_multiplier": 0.9,
            "volatility_adjustment": False
        },
        "Breakout": {
            "half_life_days": 35.0,
            "strategy_multiplier": 1.05,
            "volatility_adjustment": True
        },
        "Golden_Cross": {
            "half_life_days": 60.0,
            "strategy_multiplier": 1.15,
            "volatility_adjustment": True
        }
    })

class TimeDecayFactor:
    """
    時間減衰ファクター計算クラス
    新しいデータを重視する重み付け機能
    """
    
    def __init__(self, parameters: Optional[DecayParameters] = None):
        """
        Parameters:
            parameters: 減衰パラメータ設定
        """
        self.parameters = parameters or DecayParameters()
        self._cache = {}  # 計算結果キャッシュ
        
    def calculate_decay_weight(self, 
                             timestamp: str, 
                             reference_time: Optional[str] = None,
                             strategy_name: Optional[str] = None) -> float:
        """
        時間減衰重みの計算
        
        Parameters:
            timestamp: 対象データのタイムスタンプ
            reference_time: 基準時刻（デフォルト：現在時刻）
            strategy_name: 戦略名（戦略別調整用）
            
        Returns:
            float: 減衰重み（0.0-1.0）
        """
        try:
            # 基準時刻設定
            if reference_time is None:
                reference_time = datetime.now().isoformat()
            
            # 時刻解析
            target_dt = self._parse_timestamp(timestamp)
            ref_dt = self._parse_timestamp(reference_time)
            
            # 時間差計算（日数）
            time_diff = (ref_dt - target_dt).total_seconds() / 86400  # 日数
            
            if time_diff < 0:
                # 未来のデータは重み1.0
                return 1.0
            
            # 戦略別パラメータ取得
            params = self._get_strategy_parameters(strategy_name)
            
            # 減衰重み計算
            decay_weight = self._calculate_decay_by_model(
                time_diff, 
                params["half_life_days"],
                params["model"]
            )
            
            # 戦略別倍率適用
            decay_weight *= params["strategy_multiplier"]
            
            # 最小重み制限
            decay_weight = max(decay_weight, params["min_weight"])
            
            return min(1.0, decay_weight)
            
        except Exception as e:
            logger.error(f"Failed to calculate decay weight: {e}")
            return 1.0  # エラー時はフル重み
    
    def _calculate_decay_by_model(self, 
                                time_diff: float, 
                                half_life: float, 
                                model: DecayModel) -> float:
        """減衰モデル別計算"""
        
        if model == DecayModel.EXPONENTIAL:
            # 指数減衰: w = exp(-λt), λ = ln(2)/half_life
            decay_constant = np.log(2) / half_life
            return np.exp(-decay_constant * time_diff)
            
        elif model == DecayModel.LINEAR:
            # 線形減衰: w = max(0, 1 - t/half_life)
            return max(0.0, 1.0 - time_diff / half_life)
            
        elif model == DecayModel.GAUSSIAN:
            # ガウシアン減衰: w = exp(-(t/σ)²), σ = half_life/sqrt(2ln2)
            sigma = half_life / np.sqrt(2 * np.log(2))
            return np.exp(-(time_diff / sigma) ** 2)
            
        elif model == DecayModel.POWER_LAW:
            # べき乗減衰: w = (1 + t/half_life)^(-α), α=1
            alpha = 1.0
            return (1 + time_diff / half_life) ** (-alpha)
            
        else:
            # デフォルト：指数減衰
            decay_constant = np.log(2) / half_life
            return np.exp(-decay_constant * time_diff)
    
    def calculate_weighted_scores(self, 
                                score_entries: List[Any],
                                reference_time: Optional[str] = None,
                                strategy_name: Optional[str] = None) -> Dict[str, float]:
        """
        時間減衰重みを適用したスコア計算
        
        Parameters:
            score_entries: スコア履歴エントリリスト
            reference_time: 基準時刻
            strategy_name: 戦略名
            
        Returns:
            Dict[str, float]: 重み付きスコア統計
        """
        if not score_entries:
            return {}
        
        try:
            weighted_scores = []
            total_weight = 0.0
            
            for entry in score_entries:
                # タイムスタンプ取得（エントリ形式に対応）
                if hasattr(entry, 'strategy_score'):
                    timestamp = entry.strategy_score.calculated_at.isoformat()
                    score = entry.strategy_score.total_score
                    entry_strategy = entry.strategy_score.strategy_name
                elif hasattr(entry, 'calculated_at'):
                    timestamp = entry.calculated_at.isoformat()
                    score = entry.total_score
                    entry_strategy = entry.strategy_name
                else:
                    # 辞書形式の場合
                    timestamp = entry.get('calculated_at', entry.get('timestamp'))
                    score = entry.get('total_score', 0.0)
                    entry_strategy = entry.get('strategy_name')
                
                # 減衰重み計算
                weight = self.calculate_decay_weight(
                    timestamp, 
                    reference_time, 
                    strategy_name or entry_strategy
                )
                
                # 重み付きスコア
                weighted_score = score * weight
                weighted_scores.append(weighted_score)
                total_weight += weight
            
            if total_weight == 0:
                return {}
            
            # 統計計算
            weighted_mean = sum(weighted_scores) / total_weight
            
            # 重み付き分散
            weights = [
                self.calculate_decay_weight(
                    entry.strategy_score.calculated_at.isoformat() if hasattr(entry, 'strategy_score') 
                    else entry.calculated_at.isoformat() if hasattr(entry, 'calculated_at')
                    else entry.get('calculated_at', entry.get('timestamp')),
                    reference_time, 
                    strategy_name
                ) for entry in score_entries
            ]
            
            scores = [
                entry.strategy_score.total_score if hasattr(entry, 'strategy_score')
                else entry.total_score if hasattr(entry, 'total_score') 
                else entry.get('total_score', 0.0) for entry in score_entries
            ]
            
            weighted_variance = sum([
                w * (score - weighted_mean) ** 2 
                for w, score in zip(weights, scores)
            ]) / total_weight
            
            weighted_std = np.sqrt(weighted_variance)
            
            return {
                "weighted_mean": weighted_mean,
                "weighted_std": weighted_std,
                "total_weight": total_weight,
                "entry_count": len(score_entries),
                "effective_sample_size": total_weight ** 2 / sum([w ** 2 for w in weights])
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate weighted scores: {e}")
            return {}
    
    def _get_strategy_parameters(self, strategy_name: Optional[str]) -> Dict[str, Any]:
        """戦略別パラメータ取得"""
        if strategy_name and strategy_name in self.parameters.strategy_defaults:
            defaults = self.parameters.strategy_defaults[strategy_name]
            return {
                "half_life_days": defaults.get("half_life_days", self.parameters.half_life_days),
                "strategy_multiplier": defaults.get("strategy_multiplier", self.parameters.strategy_multiplier),
                "model": self.parameters.model,
                "min_weight": self.parameters.min_weight,
                "volatility_adjustment": defaults.get("volatility_adjustment", self.parameters.volatility_adjustment)
            }
        else:
            return {
                "half_life_days": self.parameters.half_life_days,
                "strategy_multiplier": self.parameters.strategy_multiplier,
                "model": self.parameters.model,
                "min_weight": self.parameters.min_weight,
                "volatility_adjustment": self.parameters.volatility_adjustment
            }
    
    def _parse_timestamp(self, timestamp: str) -> datetime:
        """タイムスタンプ解析"""
        try:
            # datetime オブジェクトの場合
            if isinstance(timestamp, datetime):
                return timestamp
            
            # ISO形式対応
            if 'T' in timestamp:
                if timestamp.endswith('Z'):
                    timestamp = timestamp[:-1] + '+00:00'
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                # 日付のみの場合
                return datetime.strptime(timestamp, '%Y-%m-%d')
        except Exception as e:
            logger.error(f"Failed to parse timestamp {timestamp}: {e}")
            return datetime.now()
    
    def get_decay_visualization_data(self, 
                                   days_range: int = 90,
                                   strategy_name: Optional[str] = None) -> pd.DataFrame:
        """
        減衰曲線の可視化用データ生成
        
        Parameters:
            days_range: 日数範囲
            strategy_name: 戦略名
            
        Returns:
            pd.DataFrame: 可視化用データ
        """
        days = np.arange(0, days_range + 1)
        reference_time = datetime.now().isoformat()
        
        weights = []
        for day in days:
            timestamp = (datetime.now() - timedelta(days=float(day))).isoformat()
            weight = self.calculate_decay_weight(timestamp, reference_time, strategy_name)
            weights.append(weight)
        
        return pd.DataFrame({
            'days_ago': days,
            'decay_weight': weights,
            'strategy': strategy_name or 'default'
        })
