"""
Module: Score Corrector
File: score_corrector.py
Description: 
  5-2-1「戦略実績に基づくスコア補正機能」
  実績ベースのスコア補正エンジン - 指数移動平均 + 適応的学習

Author: imega
Created: 2025-07-22
Modified: 2025-07-22
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from .performance_tracker import PerformanceTracker, StrategyPerformanceRecord
except ImportError:
    from performance_tracker import PerformanceTracker, StrategyPerformanceRecord

# ロガー設定
logger = logging.getLogger(__name__)

@dataclass
class CorrectionResult:
    """補正結果"""
    correction_factor: float
    confidence: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'correction_factor': self.correction_factor,
            'confidence': self.confidence,
            'reason': self.reason,
            'metadata': self.metadata
        }

class PerformanceBasedScoreCorrector:
    """実績ベースのスコア補正エンジン"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.performance_tracker = PerformanceTracker(config.get('tracker', {}))
        self.correction_config = config.get('correction', {})
        
        # 指数移動平均パラメータ
        self.alpha = self.correction_config.get('ema_alpha', 0.3)
        self.lookback_periods = self.correction_config.get('lookback_periods', 20)
        
        # 補正制限
        self.max_correction_factor = self.correction_config.get('max_correction', 0.3)
        self.min_confidence_threshold = self.correction_config.get('min_confidence', 0.6)
        self.min_records = self.correction_config.get('min_records', 5)
        self.learning_rate = self.correction_config.get('learning_rate', 0.1)
        
        # 適応的学習設定
        self.adaptive_learning_enabled = self.correction_config.get('adaptive_learning_enabled', True)
        self.trend_adjustment_weight = self.correction_config.get('trend_adjustment_weight', 0.1)
        
        logger.info("PerformanceBasedScoreCorrector initialized")
    
    def calculate_correction_factor(self,
                                  strategy_name: str,
                                  ticker: str,
                                  current_score: float) -> CorrectionResult:
        """
        補正ファクターを計算
        
        Args:
            strategy_name: 戦略名
            ticker: ティッカー
            current_score: 現在のスコア
            
        Returns:
            CorrectionResult: 補正結果
        """
        try:
            # 1. パフォーマンス履歴を取得
            history = self.performance_tracker.get_performance_history(
                strategy_name, ticker, self.lookback_periods
            )
            
            if len(history) < self.min_records:
                return CorrectionResult(
                    correction_factor=1.0,
                    confidence=0.0,
                    reason="insufficient_data",
                    metadata={'records_available': len(history), 'min_required': self.min_records}
                )
            
            # 2. 指数移動平均ベース補正の計算
            ema_correction = self._calculate_ema_correction(history)
            
            # 3. 適応的学習要素の追加
            adaptive_adjustment = 0.0
            if self.adaptive_learning_enabled:
                adaptive_adjustment = self._calculate_adaptive_adjustment(history)
            
            # 4. 最終補正ファクターの計算
            final_correction = self._combine_corrections(ema_correction, adaptive_adjustment)
            
            # 5. 信頼度の計算
            confidence = self._calculate_correction_confidence(history)
            
            # 6. メタデータの作成
            metadata = {
                'records_used': len(history),
                'ema_correction': ema_correction,
                'adaptive_adjustment': adaptive_adjustment,
                'raw_correction': final_correction,
                'confidence_raw': confidence,
                'avg_accuracy': np.mean([r.prediction_accuracy for r in history])
            }
            
            return CorrectionResult(
                correction_factor=final_correction,
                confidence=confidence,
                reason="performance_based_correction",
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate correction factor for {strategy_name}/{ticker}: {e}")
            return CorrectionResult(
                correction_factor=1.0,
                confidence=0.0,
                reason="calculation_error",
                metadata={'error': str(e)}
            )
    
    def _calculate_ema_correction(self, history: List[StrategyPerformanceRecord]) -> float:
        """
        指数移動平均ベース補正を計算
        
        Args:
            history: パフォーマンス履歴
            
        Returns:
            float: 補正ファクター
        """
        if not history:
            return 1.0
        
        try:
            # 予測誤差の指数移動平均を計算
            errors = []
            for record in history:
                # 予測スコアと実際のパフォーマンスの乖離を計算
                error = record.actual_performance - record.predicted_score
                errors.append(error)
            
            if not errors:
                return 1.0
            
            # 指数移動平均による補正ファクター計算
            ema_error = errors[0]  # 初期値
            for error in errors[1:]:
                ema_error = self.alpha * error + (1 - self.alpha) * ema_error
            
            # 補正ファクターに変換（制限あり）
            correction_factor = 1.0 + np.clip(ema_error, -self.max_correction_factor, self.max_correction_factor)
            
            return max(0.1, correction_factor)  # 最小値0.1で制限
            
        except Exception as e:
            logger.error(f"Failed to calculate EMA correction: {e}")
            return 1.0
    
    def _calculate_adaptive_adjustment(self, history: List[StrategyPerformanceRecord]) -> float:
        """
        適応的学習による調整を計算
        
        Args:
            history: パフォーマンス履歴
            
        Returns:
            float: 適応的調整値
        """
        if len(history) < 10:
            return 0.0
        
        try:
            # 最近のトレンドを分析
            recent_history = history[-10:]
            older_history = history[:-10] if len(history) > 10 else []
            
            if not older_history:
                return 0.0
            
            # 最近と過去のパフォーマンス差を計算
            recent_avg = np.mean([r.actual_performance for r in recent_history])
            older_avg = np.mean([r.actual_performance for r in older_history])
            
            # トレンド調整
            trend_adjustment = (recent_avg - older_avg) * self.trend_adjustment_weight
            
            return np.clip(trend_adjustment, -0.1, 0.1)  # ±10%に制限
            
        except Exception as e:
            logger.error(f"Failed to calculate adaptive adjustment: {e}")
            return 0.0
    
    def _combine_corrections(self, ema_correction: float, adaptive_adjustment: float) -> float:
        """
        複数の補正を組み合わせ
        
        Args:
            ema_correction: EMA補正
            adaptive_adjustment: 適応的調整
            
        Returns:
            float: 最終補正ファクター
        """
        try:
            # EMA補正に適応的調整を加算
            combined_correction = ema_correction + adaptive_adjustment
            
            # 最終的な制限を適用
            final_correction = np.clip(
                combined_correction, 
                1.0 - self.max_correction_factor,  # 下限
                1.0 + self.max_correction_factor   # 上限
            )
            
            return max(0.1, final_correction)  # 最小値0.1
            
        except Exception as e:
            logger.error(f"Failed to combine corrections: {e}")
            return 1.0
    
    def _calculate_correction_confidence(self, history: List[StrategyPerformanceRecord]) -> float:
        """
        補正の信頼度を計算
        
        Args:
            history: パフォーマンス履歴
            
        Returns:
            float: 信頼度 (0.0-1.0)
        """
        if not history:
            return 0.0
        
        try:
            # データ量ベースの信頼度
            data_confidence = min(1.0, len(history) / (self.min_records * 2))
            
            # 予測精度ベースの信頼度
            accuracies = [r.prediction_accuracy for r in history if r.prediction_accuracy is not None]
            if accuracies:
                accuracy_confidence = np.mean(accuracies)
            else:
                accuracy_confidence = 0.0
            
            # データの一貫性ベースの信頼度
            performances = [r.actual_performance for r in history]
            if len(performances) > 1:
                consistency_confidence = 1.0 - min(1.0, np.std(performances))
            else:
                consistency_confidence = 0.0
            
            # 最近のデータの重み付き信頼度
            recent_records = [r for r in history if r.timestamp >= datetime.now() - timedelta(days=7)]
            recency_confidence = min(1.0, len(recent_records) / max(1, len(history) * 0.3))
            
            # 総合信頼度（各要素の重み付き平均）
            confidence = (
                data_confidence * 0.3 +
                accuracy_confidence * 0.4 +
                consistency_confidence * 0.2 +
                recency_confidence * 0.1
            )
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.0
    
    def update_performance_record(self,
                                strategy_name: str,
                                ticker: str,
                                predicted_score: float,
                                actual_performance: float,
                                market_context: Dict[str, Any] = None) -> str:
        """
        パフォーマンス記録を更新
        
        Args:
            strategy_name: 戦略名
            ticker: ティッカー
            predicted_score: 予測スコア
            actual_performance: 実際のパフォーマンス
            market_context: 市場コンテキスト
            
        Returns:
            str: 記録ID
        """
        try:
            record_id = self.performance_tracker.record_strategy_performance(
                strategy_name=strategy_name,
                ticker=ticker,
                predicted_score=predicted_score,
                actual_performance=actual_performance,
                market_context=market_context or {}
            )
            
            logger.info(f"Updated performance record: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to update performance record: {e}")
            raise
    
    def get_correction_statistics(self, strategy_name: str, days: int = 30) -> Dict[str, Any]:
        """
        補正統計情報を取得
        
        Args:
            strategy_name: 戦略名
            days: 分析日数
            
        Returns:
            Dict[str, Any]: 統計情報
        """
        try:
            stats = self.performance_tracker.get_strategy_statistics(strategy_name, days)
            
            # 補正関連統計を追加
            correction_stats = {
                'correction_enabled': True,
                'ema_alpha': self.alpha,
                'lookback_periods': self.lookback_periods,
                'max_correction_factor': self.max_correction_factor,
                'min_confidence_threshold': self.min_confidence_threshold,
                'adaptive_learning_enabled': self.adaptive_learning_enabled
            }
            
            stats.update(correction_stats)
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get correction statistics: {e}")
            return {}

# エクスポート
__all__ = [
    "CorrectionResult",
    "PerformanceBasedScoreCorrector"
]
