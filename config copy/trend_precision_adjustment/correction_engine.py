"""
Module: Correction Engine
File: correction_engine.py
Description: 
  5-2-2「トレンド判定精度の自動補正」
  ハイブリッド補正エンジン - パラメータ調整と信頼度較正の統合

Author: imega
Created: 2025-07-22
Modified: 2025-07-22
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 内部モジュールのインポート
from .parameter_adjuster import ParameterAdjuster
from .confidence_calibrator import ConfidenceCalibrator

@dataclass
class CorrectedTrendResult:
    """補正済みトレンド判定結果"""
    original_trend: str
    original_confidence: float
    corrected_trend: str
    corrected_confidence: float
    parameter_adjustments: Dict[str, Any]
    calibration_factor: float
    correction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_improvement_ratio(self) -> float:
        """改善率を計算"""
        if self.original_confidence == 0:
            return 0.0
        return (self.corrected_confidence - self.original_confidence) / self.original_confidence

class TrendPrecisionCorrectionEngine:
    """トレンド判定精度補正エンジン"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # 設定の読み込み
        self.correction_strength = config.get('correction_strength', 0.3)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.enable_parameter_adjustment = config.get('enable_parameter_adjustment', True)
        self.enable_confidence_calibration = config.get('enable_confidence_calibration', True)
        self.hybrid_combination_weight = config.get('hybrid_combination_weight', 0.7)
        
        # サブシステムの初期化
        try:
            self.parameter_adjuster = ParameterAdjuster(config.get('parameter_adjustment', {}))
            self.confidence_calibrator = ConfidenceCalibrator(config.get('confidence_calibration', {}))
            self.logger.info("TrendPrecisionCorrectionEngine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize correction engine: {e}")
            raise
        
        # 補正履歴
        self._correction_history: List[CorrectedTrendResult] = []
    
    def apply_precision_correction(self,
                                 detector: Any,
                                 ticker: str,
                                 precision_tracker: Any = None) -> CorrectedTrendResult:
        """精度補正を適用"""
        
        try:
            # 1. 基本トレンド判定を実行
            base_trend, base_confidence = self._get_base_trend_prediction(detector)
            
            self.logger.debug(f"Base prediction: {base_trend} (confidence: {base_confidence:.3f})")
            
            # 2. パラメータ自動調整
            adjusted_params = {}
            adjusted_trend = base_trend
            adjusted_confidence = base_confidence
            
            if self.enable_parameter_adjustment:
                adjusted_params = self.parameter_adjuster.get_optimized_parameters(
                    detector.strategy_name, detector.method, ticker, precision_tracker
                )
                
                if adjusted_params:
                    # 調整後のパラメータで再判定
                    adjusted_detector = self._create_adjusted_detector(detector, adjusted_params)
                    if adjusted_detector:
                        adjusted_trend, adjusted_confidence = self._get_base_trend_prediction(adjusted_detector)
                        self.logger.debug(f"Adjusted prediction: {adjusted_trend} (confidence: {adjusted_confidence:.3f})")
            
            # 3. 信頼度較正
            calibrated_confidence = adjusted_confidence
            calibration_factor = 1.0
            
            if self.enable_confidence_calibration:
                calibrated_confidence = self.confidence_calibrator.calibrate_confidence(
                    adjusted_confidence, detector.strategy_name, detector.method, ticker, precision_tracker
                )
                calibration_factor = calibrated_confidence / adjusted_confidence if adjusted_confidence > 0 else 1.0
                self.logger.debug(f"Calibrated confidence: {calibrated_confidence:.3f} (factor: {calibration_factor:.3f})")
            
            # 4. ハイブリッド補正結果を統合
            final_result = self._combine_corrections(
                base_trend, base_confidence,
                adjusted_trend, adjusted_confidence,
                calibrated_confidence
            )
            
            # 5. 結果オブジェクトの作成
            correction_result = CorrectedTrendResult(
                original_trend=base_trend,
                original_confidence=base_confidence,
                corrected_trend=final_result['trend'],
                corrected_confidence=final_result['confidence'],
                parameter_adjustments=adjusted_params,
                calibration_factor=calibration_factor,
                correction_metadata={
                    'base_prediction': (base_trend, base_confidence),
                    'parameter_adjusted': (adjusted_trend, adjusted_confidence),
                    'final_calibrated': (final_result['trend'], final_result['confidence']),
                    'adjustments_applied': len(adjusted_params) > 0,
                    'calibration_applied': self.enable_confidence_calibration,
                    'correction_timestamp': datetime.now().isoformat(),
                    'detector_method': detector.method,
                    'detector_strategy': detector.strategy_name
                }
            )
            
            # 履歴に追加
            self._correction_history.append(correction_result)
            self._cleanup_correction_history()
            
            self.logger.info(f"Applied correction for {detector.strategy_name}_{detector.method}_{ticker}: "
                           f"{base_confidence:.3f} -> {final_result['confidence']:.3f}")
            
            return correction_result
            
        except Exception as e:
            self.logger.error(f"Failed to apply precision correction: {e}")
            # エラー時はオリジナル結果を返す
            base_trend, base_confidence = self._get_base_trend_prediction(detector)
            return CorrectedTrendResult(
                original_trend=base_trend,
                original_confidence=base_confidence,
                corrected_trend=base_trend,
                corrected_confidence=base_confidence,
                parameter_adjustments={},
                calibration_factor=1.0,
                correction_metadata={'error': str(e)}
            )
    
    def _get_base_trend_prediction(self, detector: Any) -> Tuple[str, float]:
        """基本トレンド判定を取得"""
        try:
            if hasattr(detector, 'detect_trend_with_confidence'):
                return detector.detect_trend_with_confidence()
            elif hasattr(detector, 'detect_trend'):
                trend = detector.detect_trend()
                confidence = getattr(detector, 'get_confidence_score', lambda: 0.5)()
                return trend, confidence
            else:
                self.logger.warning("Detector does not have expected methods, using defaults")
                return "unknown", 0.5
        except Exception as e:
            self.logger.error(f"Failed to get base trend prediction: {e}")
            return "unknown", 0.0
    
    def _create_adjusted_detector(self, original_detector: Any, adjusted_params: Dict[str, Any]) -> Optional[Any]:
        """調整されたパラメータで新しい検出器を作成"""
        try:
            if not adjusted_params:
                return None
            
            # 元の検出器のパラメータをコピー
            new_params = getattr(original_detector, 'params', {}).copy()
            new_params.update(adjusted_params)
            
            # 新しい検出器を作成（簡易実装）
            # 実際の実装では、UnifiedTrendDetectorの新しいインスタンスを作成
            if hasattr(original_detector, '__class__'):
                try:
                    adjusted_detector = original_detector.__class__(
                        data=original_detector.data,
                        strategy_name=original_detector.strategy_name,
                        method=original_detector.method,
                        params=new_params,
                        price_column=getattr(original_detector, 'price_column', 'Adj Close')
                    )
                    return adjusted_detector
                except Exception as e:
                    self.logger.warning(f"Could not create adjusted detector: {e}")
                    return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create adjusted detector: {e}")
            return None
    
    def _combine_corrections(self,
                           base_trend: str,
                           base_confidence: float,
                           adjusted_trend: str,
                           adjusted_confidence: float,
                           calibrated_confidence: float) -> Dict[str, Any]:
        """ハイブリッド補正結果を統合"""
        
        try:
            # トレンド判定の統合
            if base_trend == adjusted_trend:
                final_trend = base_trend
                trend_confidence_boost = 0.1  # 一致した場合は信頼度を少しブースト
            else:
                # 不一致の場合は信頼度に基づいて選択
                if adjusted_confidence > base_confidence * 1.1:
                    final_trend = adjusted_trend
                    trend_confidence_boost = 0.0
                else:
                    final_trend = base_trend
                    trend_confidence_boost = -0.05  # 不一致による信頼度ペナルティ
            
            # 信頼度の統合
            # ベース信頼度、調整後信頼度、較正後信頼度の加重平均
            weight_base = 0.3
            weight_adjusted = 0.3 if adjusted_confidence != base_confidence else 0.0
            weight_calibrated = 0.4 + (0.3 if weight_adjusted == 0.0 else 0.0)
            
            combined_confidence = (
                weight_base * base_confidence +
                weight_adjusted * adjusted_confidence +
                weight_calibrated * calibrated_confidence
            )
            
            # トレンド一致によるブーストを適用
            combined_confidence += trend_confidence_boost
            
            # 補正強度を適用
            final_confidence = (
                (1 - self.correction_strength) * base_confidence +
                self.correction_strength * combined_confidence
            )
            
            # 0-1の範囲にクリッピング
            final_confidence = max(0.0, min(1.0, final_confidence))
            
            return {
                'trend': final_trend,
                'confidence': final_confidence,
                'combination_weights': {
                    'base': weight_base,
                    'adjusted': weight_adjusted,
                    'calibrated': weight_calibrated
                },
                'trend_agreement': base_trend == adjusted_trend,
                'confidence_improvement': final_confidence - base_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Failed to combine corrections: {e}")
            return {
                'trend': base_trend,
                'confidence': base_confidence,
                'combination_weights': {'base': 1.0, 'adjusted': 0.0, 'calibrated': 0.0},
                'trend_agreement': True,
                'confidence_improvement': 0.0
            }
    
    def _cleanup_correction_history(self, max_entries: int = 1000):
        """補正履歴のクリーンアップ"""
        try:
            if len(self._correction_history) > max_entries:
                # 最新のエントリを保持
                self._correction_history = self._correction_history[-max_entries:]
                self.logger.debug(f"Cleaned up correction history, keeping {max_entries} entries")
        except Exception as e:
            self.logger.error(f"Failed to cleanup correction history: {e}")
    
    def get_correction_performance(self, days: int = 30) -> Dict[str, Any]:
        """補正パフォーマンスの統計を取得"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.isoformat()
            
            recent_corrections = [
                corr for corr in self._correction_history
                if corr.correction_metadata.get('correction_timestamp', '') >= cutoff_str
            ]
            
            if not recent_corrections:
                return {
                    'period_days': days,
                    'total_corrections': 0,
                    'message': 'No recent corrections found'
                }
            
            # 統計計算
            improvements = [corr.get_improvement_ratio() for corr in recent_corrections]
            confidence_improvements = [
                corr.corrected_confidence - corr.original_confidence 
                for corr in recent_corrections
            ]
            
            parameter_adjustment_rate = sum(
                1 for corr in recent_corrections 
                if len(corr.parameter_adjustments) > 0
            ) / len(recent_corrections)
            
            trend_change_rate = sum(
                1 for corr in recent_corrections 
                if corr.original_trend != corr.corrected_trend
            ) / len(recent_corrections)
            
            return {
                'period_days': days,
                'total_corrections': len(recent_corrections),
                'avg_improvement_ratio': float(np.mean(improvements)),
                'avg_confidence_improvement': float(np.mean(confidence_improvements)),
                'parameter_adjustment_rate': float(parameter_adjustment_rate),
                'trend_change_rate': float(trend_change_rate),
                'positive_improvements': sum(1 for imp in improvements if imp > 0),
                'negative_improvements': sum(1 for imp in improvements if imp < 0),
                'max_improvement': float(max(improvements)) if improvements else 0.0,
                'min_improvement': float(min(improvements)) if improvements else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get correction performance: {e}")
            return {'error': str(e)}
    
    def adjust_correction_parameters(self, performance_feedback: Dict[str, Any]):
        """パフォーマンスフィードバックに基づいて補正パラメータを調整"""
        
        try:
            avg_improvement = performance_feedback.get('avg_improvement_ratio', 0.0)
            
            if avg_improvement < -0.1:
                # パフォーマンスが悪化している場合は補正強度を下げる
                self.correction_strength = max(0.1, self.correction_strength * 0.9)
                self.logger.info(f"Reduced correction strength to {self.correction_strength:.3f}")
                
            elif avg_improvement > 0.1:
                # パフォーマンスが向上している場合は補正強度を上げる
                self.correction_strength = min(0.5, self.correction_strength * 1.1)
                self.logger.info(f"Increased correction strength to {self.correction_strength:.3f}")
            
            # 学習率の調整
            trend_change_rate = performance_feedback.get('trend_change_rate', 0.0)
            if trend_change_rate > 0.3:
                # トレンド変更が多すぎる場合は学習率を下げる
                self.learning_rate = max(0.05, self.learning_rate * 0.95)
                self.logger.info(f"Reduced learning rate to {self.learning_rate:.3f}")
                
        except Exception as e:
            self.logger.error(f"Failed to adjust correction parameters: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """システムステータスを取得"""
        
        try:
            return {
                'correction_engine_active': True,
                'parameter_adjustment_enabled': self.enable_parameter_adjustment,
                'confidence_calibration_enabled': self.enable_confidence_calibration,
                'correction_strength': self.correction_strength,
                'learning_rate': self.learning_rate,
                'hybrid_weight': self.hybrid_combination_weight,
                'total_corrections_applied': len(self._correction_history),
                'parameter_adjuster_status': 'active' if self.parameter_adjuster else 'inactive',
                'confidence_calibrator_status': 'active' if self.confidence_calibrator else 'inactive'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e), 'status': 'error'}

if __name__ == "__main__":
    # テスト用コード
    print("TrendPrecisionCorrectionEngine モジュールが正常にロードされました")
