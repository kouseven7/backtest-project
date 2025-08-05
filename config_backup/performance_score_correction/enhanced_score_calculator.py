"""
Module: Enhanced Score Calculator
File: enhanced_score_calculator.py
Description: 
  5-2-1「戦略実績に基づくスコア補正機能」
  実績補正付き戦略スコア計算器 - 既存StrategyScoreCalculatorとの統合

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
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from .score_corrector import PerformanceBasedScoreCorrector, CorrectionResult
    from .performance_tracker import PerformanceTracker
except ImportError:
    from score_corrector import PerformanceBasedScoreCorrector, CorrectionResult
    from performance_tracker import PerformanceTracker

try:
    from config.strategy_scoring_model import StrategyScoreCalculator, StrategyScore
except ImportError:
    # 基本的なStrategyScoreクラスの代替実装
    @dataclass
    class StrategyScore:
        strategy_name: str
        ticker: str
        total_score: float
        component_scores: Dict[str, float]
        trend_fitness: float
        confidence: float
        metadata: Dict[str, Any]
        calculated_at: datetime
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'strategy_name': self.strategy_name,
                'ticker': self.ticker,
                'total_score': self.total_score,
                'component_scores': self.component_scores,
                'trend_fitness': self.trend_fitness,
                'confidence': self.confidence,
                'metadata': self.metadata,
                'calculated_at': self.calculated_at.isoformat()
            }

    class StrategyScoreCalculator:
        def __init__(self):
            pass
        
        def calculate_strategy_score(self, strategy_name: str, ticker: str, 
                                   market_data: pd.DataFrame = None,
                                   trend_context: Dict[str, Any] = None) -> Optional[StrategyScore]:
            # 基本スコア計算のダミー実装
            return StrategyScore(
                strategy_name=strategy_name,
                ticker=ticker,
                total_score=0.5,  # ダミー値
                component_scores={'performance': 0.5, 'stability': 0.5},
                trend_fitness=0.5,
                confidence=0.7,
                metadata={'source': 'dummy'},
                calculated_at=datetime.now()
            )

# ロガー設定
logger = logging.getLogger(__name__)

@dataclass
class CorrectedStrategyScore:
    """補正済み戦略スコア"""
    base_score: StrategyScore
    corrected_total_score: float
    correction_factor: float
    correction_confidence: float
    correction_reason: str
    correction_metadata: Dict[str, Any] = field(default_factory=dict)
    calculated_at: datetime = field(default_factory=datetime.now)
    
    def get_improvement_ratio(self) -> float:
        """改善比率を取得"""
        if self.base_score.total_score == 0:
            return 0.0
        return (self.corrected_total_score - self.base_score.total_score) / self.base_score.total_score
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'base_score': self.base_score.to_dict(),
            'corrected_total_score': self.corrected_total_score,
            'correction_factor': self.correction_factor,
            'correction_confidence': self.correction_confidence,
            'correction_reason': self.correction_reason,
            'correction_metadata': self.correction_metadata,
            'improvement_ratio': self.get_improvement_ratio(),
            'calculated_at': self.calculated_at.isoformat()
        }

class EnhancedStrategyScoreCalculator:
    """実績補正付き戦略スコア計算器"""
    
    def __init__(self, 
                 base_calculator: StrategyScoreCalculator = None,
                 score_corrector: PerformanceBasedScoreCorrector = None,
                 config_file: Optional[str] = None):
        """
        初期化
        
        Args:
            base_calculator: ベースとなる戦略スコア計算器
            score_corrector: スコア補正器
            config_file: 設定ファイルパス
        """
        self.base_calculator = base_calculator or StrategyScoreCalculator()
        self.enhanced_config = self._load_config(config_file)
        
        # スコア補正器の初期化
        if score_corrector:
            self.score_corrector = score_corrector
        else:
            self.score_corrector = PerformanceBasedScoreCorrector(self.enhanced_config)
        
        # 統計追跡
        self.correction_stats = {
            'total_calculations': 0,
            'corrections_applied': 0,
            'avg_improvement': 0.0,
            'high_confidence_corrections': 0
        }
        
        logger.info("EnhancedStrategyScoreCalculator initialized")
    
    def calculate_corrected_strategy_score(self,
                                         strategy_name: str,
                                         ticker: str,
                                         market_data: pd.DataFrame = None,
                                         trend_context: Dict[str, Any] = None,
                                         apply_correction: bool = True) -> Optional[CorrectedStrategyScore]:
        """
        補正付き戦略スコアを計算
        
        Args:
            strategy_name: 戦略名
            ticker: ティッカー
            market_data: 市場データ
            trend_context: トレンドコンテキスト
            apply_correction: 補正を適用するか
            
        Returns:
            CorrectedStrategyScore: 補正済みスコア
        """
        try:
            self.correction_stats['total_calculations'] += 1
            
            # 1. 基本スコアを計算
            base_score = self.base_calculator.calculate_strategy_score(
                strategy_name, ticker, market_data, trend_context
            )
            
            if not base_score:
                logger.warning(f"Base score calculation failed for {strategy_name}/{ticker}")
                return None
            
            # 2. 補正ファクターを計算
            correction_result = CorrectionResult(
                correction_factor=1.0,
                confidence=0.0,
                reason="correction_disabled"
            )
            
            if apply_correction:
                correction_result = self.score_corrector.calculate_correction_factor(
                    strategy_name, ticker, base_score.total_score
                )
            
            # 3. 補正を適用
            corrected_score = base_score.total_score
            correction_applied = False
            
            if (apply_correction and 
                correction_result.confidence >= self.score_corrector.min_confidence_threshold):
                
                corrected_score = base_score.total_score * correction_result.correction_factor
                corrected_score = np.clip(corrected_score, 0.0, 1.0)
                correction_applied = True
                
                self.correction_stats['corrections_applied'] += 1
                
                if correction_result.confidence >= 0.8:
                    self.correction_stats['high_confidence_corrections'] += 1
            else:
                correction_result.correction_factor = 1.0
            
            # 4. 補正結果オブジェクトを作成
            corrected_strategy_score = CorrectedStrategyScore(
                base_score=base_score,
                corrected_total_score=corrected_score,
                correction_factor=correction_result.correction_factor,
                correction_confidence=correction_result.confidence,
                correction_reason=correction_result.reason,
                correction_metadata={
                    **correction_result.metadata,
                    'correction_applied': correction_applied,
                    'original_score': base_score.total_score,
                    'calculation_timestamp': datetime.now().isoformat()
                },
                calculated_at=datetime.now()
            )
            
            # 5. 統計更新
            if correction_applied:
                improvement = corrected_strategy_score.get_improvement_ratio()
                current_avg = self.correction_stats['avg_improvement']
                corrections_count = self.correction_stats['corrections_applied']
                
                # 移動平均で改善比率を更新
                self.correction_stats['avg_improvement'] = (
                    (current_avg * (corrections_count - 1) + improvement) / corrections_count
                )
            
            logger.debug(f"Calculated corrected score for {strategy_name}/{ticker}: "
                        f"base={base_score.total_score:.3f}, corrected={corrected_score:.3f}, "
                        f"factor={correction_result.correction_factor:.3f}")
            
            return corrected_strategy_score
            
        except Exception as e:
            logger.error(f"Failed to calculate corrected strategy score for {strategy_name}/{ticker}: {e}")
            return None
    
    def update_performance_feedback(self,
                                  strategy_name: str,
                                  ticker: str,
                                  predicted_score: float,
                                  actual_performance: float,
                                  market_context: Dict[str, Any] = None) -> str:
        """
        パフォーマンスフィードバックを更新
        
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
            record_id = self.score_corrector.update_performance_record(
                strategy_name=strategy_name,
                ticker=ticker,
                predicted_score=predicted_score,
                actual_performance=actual_performance,
                market_context=market_context or {}
            )
            
            logger.info(f"Updated performance feedback: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to update performance feedback: {e}")
            raise
    
    def get_correction_performance(self) -> Dict[str, Any]:
        """補正パフォーマンスの統計を取得"""
        try:
            total_calcs = self.correction_stats['total_calculations']
            corrections_applied = self.correction_stats['corrections_applied']
            
            performance = {
                **self.correction_stats,
                'correction_rate': corrections_applied / max(1, total_calcs),
                'high_confidence_rate': self.correction_stats['high_confidence_corrections'] / max(1, corrections_applied),
                'system_config': {
                    'min_confidence_threshold': self.score_corrector.min_confidence_threshold,
                    'max_correction_factor': self.score_corrector.max_correction_factor,
                    'lookback_periods': self.score_corrector.lookback_periods,
                    'adaptive_learning_enabled': self.score_corrector.adaptive_learning_enabled
                }
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to get correction performance: {e}")
            return {}
    
    def get_strategy_correction_history(self, strategy_name: str, days: int = 30) -> Dict[str, Any]:
        """戦略の補正履歴を取得"""
        try:
            return self.score_corrector.get_correction_statistics(strategy_name, days)
        except Exception as e:
            logger.error(f"Failed to get strategy correction history: {e}")
            return {}
    
    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            if config_file and os.path.exists(config_file):
                config_path = config_file
            else:
                # デフォルト設定ファイルパス
                config_dir = Path(__file__).parent.parent / "score_correction"
                config_path = config_dir / "correction_config.json"
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Enhanced configuration loaded from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file not found: {config_path}, using defaults")
                return self._get_default_config()
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            'tracker': {
                'tracking_window_days': 30,
                'min_records': 10,
                'performance_threshold': 0.1
            },
            'correction': {
                'ema_alpha': 0.3,
                'lookback_periods': 20,
                'max_correction': 0.3,
                'min_confidence': 0.6,
                'min_records': 5,
                'adaptive_learning_enabled': True
            }
        }

# エクスポート
__all__ = [
    "CorrectedStrategyScore",
    "EnhancedStrategyScoreCalculator"
]
