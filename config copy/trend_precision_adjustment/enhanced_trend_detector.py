"""
Module: Enhanced Trend Detector
File: enhanced_trend_detector.py
Description: 
  5-2-2「トレンド判定精度の自動補正」
  精度補正機能付きトレンド判定器 - 統合インターフェース

Author: imega
Created: 2025-07-22
Modified: 2025-07-22
"""

import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 内部モジュールのインポート
from .correction_engine import TrendPrecisionCorrectionEngine, CorrectedTrendResult

@dataclass
class EnhancedTrendResult:
    """拡張トレンド判定結果"""
    trend: str
    confidence: float
    is_corrected: bool
    correction_details: Optional[CorrectedTrendResult] = None
    original_result: Optional[Tuple[str, float]] = None
    improvement_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_confidence_improvement(self) -> float:
        """信頼度改善率を取得"""
        if not self.is_corrected or not self.original_result:
            return 0.0
        
        original_conf = self.original_result[1]
        if original_conf == 0:
            return 0.0
        
        return (self.confidence - original_conf) / original_conf
    
    def get_summary(self) -> Dict[str, Any]:
        """結果サマリーを取得"""
        return {
            'trend': self.trend,
            'confidence': round(self.confidence, 3),
            'is_corrected': self.is_corrected,
            'improvement_ratio': round(self.improvement_ratio or 0.0, 3),
            'confidence_improvement': round(self.get_confidence_improvement(), 3),
            'original_trend': self.original_result[0] if self.original_result else None,
            'original_confidence': round(self.original_result[1], 3) if self.original_result else None
        }

class EnhancedTrendDetector:
    """精度補正機能付きトレンド判定器"""
    
    def __init__(self,
                 base_detector: Any,
                 correction_engine: TrendPrecisionCorrectionEngine,
                 enable_correction: bool = True,
                 precision_tracker: Any = None):
        """
        初期化
        
        Args:
            base_detector: 基本トレンド判定器（UnifiedTrendDetector等）
            correction_engine: 精度補正エンジン
            enable_correction: 補正を有効にするかどうか
            precision_tracker: 精度追跡器（オプション）
        """
        self.base_detector = base_detector
        self.correction_engine = correction_engine
        self.enable_correction = enable_correction
        self.precision_tracker = precision_tracker
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"EnhancedTrendDetector initialized for {getattr(base_detector, 'strategy_name', 'unknown')}")
    
    def detect_enhanced_trend(self, ticker: str = 'UNKNOWN') -> EnhancedTrendResult:
        """拡張トレンド判定（補正付き）"""
        
        try:
            # 補正が無効な場合は基本判定のみ
            if not self.enable_correction:
                trend, confidence = self._get_base_prediction()
                return EnhancedTrendResult(
                    trend=trend,
                    confidence=confidence,
                    is_corrected=False,
                    metadata={
                        'correction_disabled': True,
                        'timestamp': datetime.now().isoformat()
                    }
                )
            
            # 補正適用
            correction_result = self.correction_engine.apply_precision_correction(
                self.base_detector, ticker, self.precision_tracker
            )
            
            # 改善率の計算
            improvement_ratio = self._calculate_improvement_ratio(correction_result)
            
            return EnhancedTrendResult(
                trend=correction_result.corrected_trend,
                confidence=correction_result.corrected_confidence,
                is_corrected=True,
                correction_details=correction_result,
                original_result=(correction_result.original_trend, correction_result.original_confidence),
                improvement_ratio=improvement_ratio,
                metadata={
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat(),
                    'parameter_adjustments_applied': len(correction_result.parameter_adjustments) > 0,
                    'calibration_factor': correction_result.calibration_factor
                }
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced trend detection failed: {e}")
            # エラー時は基本判定にフォールバック
            trend, confidence = self._get_base_prediction()
            return EnhancedTrendResult(
                trend=trend,
                confidence=confidence,
                is_corrected=False,
                metadata={
                    'error': str(e),
                    'fallback_used': True,
                    'timestamp': datetime.now().isoformat()
                }
            )
    
    def _get_base_prediction(self) -> Tuple[str, float]:
        """基本予測を取得"""
        try:
            if hasattr(self.base_detector, 'detect_trend_with_confidence'):
                return self.base_detector.detect_trend_with_confidence()
            elif hasattr(self.base_detector, 'detect_trend'):
                trend = self.base_detector.detect_trend()
                confidence = getattr(self.base_detector, 'get_confidence_score', lambda: 0.5)()
                return trend, confidence
            else:
                self.logger.warning("Base detector lacks expected methods")
                return "unknown", 0.5
        except Exception as e:
            self.logger.error(f"Failed to get base prediction: {e}")
            return "unknown", 0.0
    
    def _calculate_improvement_ratio(self, correction_result: CorrectedTrendResult) -> float:
        """改善率を計算"""
        try:
            if correction_result.original_confidence == 0:
                return 0.0
            
            confidence_improvement = (
                correction_result.corrected_confidence - correction_result.original_confidence
            ) / correction_result.original_confidence
            
            # パラメータ調整ボーナス
            param_bonus = 0.05 if len(correction_result.parameter_adjustments) > 0 else 0.0
            
            # 較正ファクターによる調整
            calibration_adjustment = abs(correction_result.calibration_factor - 1.0) * 0.1
            
            total_improvement = confidence_improvement + param_bonus + calibration_adjustment
            return round(total_improvement, 4)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate improvement ratio: {e}")
            return 0.0
    
    def detect_trend(self) -> str:
        """シンプルなトレンド判定インターフェース（後方互換性）"""
        try:
            result = self.detect_enhanced_trend()
            return result.trend
        except Exception as e:
            self.logger.error(f"Simple trend detection failed: {e}")
            return "unknown"
    
    def detect_trend_with_confidence(self) -> Tuple[str, float]:
        """信頼度付きトレンド判定インターフェース（後方互換性）"""
        try:
            result = self.detect_enhanced_trend()
            return result.trend, result.confidence
        except Exception as e:
            self.logger.error(f"Confidence trend detection failed: {e}")
            return "unknown", 0.0
    
    def get_confidence_score(self) -> float:
        """信頼度スコア取得（後方互換性）"""
        try:
            result = self.detect_enhanced_trend()
            return result.confidence
        except Exception as e:
            self.logger.error(f"Failed to get confidence score: {e}")
            return 0.0
    
    def record_prediction_feedback(self,
                                 ticker: str,
                                 predicted_trend: str,
                                 predicted_confidence: float,
                                 actual_outcome: str,
                                 market_context: Dict[str, Any] = None):
        """予測フィードバックを記録"""
        try:
            if self.precision_tracker is None:
                self.logger.warning("No precision tracker available for feedback recording")
                return
            
            if market_context is None:
                market_context = {}
            
            # パラメータ情報の取得
            parameters = getattr(self.base_detector, 'params', {})
            method = getattr(self.base_detector, 'method', 'unknown')
            strategy_name = getattr(self.base_detector, 'strategy_name', 'unknown')
            
            # フィードバックを記録
            record_id = self.precision_tracker.record_trend_prediction(
                ticker=ticker,
                strategy_name=strategy_name,
                method=method,
                predicted_trend=predicted_trend,
                confidence_score=predicted_confidence,
                parameters=parameters,
                market_context=market_context
            )
            
            self.logger.info(f"Recorded prediction feedback: {record_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record prediction feedback: {e}")
    
    def get_correction_statistics(self, days: int = 30) -> Dict[str, Any]:
        """補正統計を取得"""
        try:
            if self.correction_engine:
                return self.correction_engine.get_correction_performance(days)
            else:
                return {'message': 'No correction engine available'}
        except Exception as e:
            self.logger.error(f"Failed to get correction statistics: {e}")
            return {'error': str(e)}
    
    def enable_corrections(self):
        """補正を有効化"""
        self.enable_correction = True
        self.logger.info("Trend corrections enabled")
    
    def disable_corrections(self):
        """補正を無効化"""
        self.enable_correction = False
        self.logger.info("Trend corrections disabled")
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        try:
            base_info = {
                'detector_type': type(self.base_detector).__name__ if self.base_detector else 'None',
                'strategy_name': getattr(self.base_detector, 'strategy_name', 'unknown'),
                'method': getattr(self.base_detector, 'method', 'unknown'),
                'correction_enabled': self.enable_correction,
                'has_precision_tracker': self.precision_tracker is not None,
                'has_correction_engine': self.correction_engine is not None
            }
            
            if self.correction_engine:
                engine_status = self.correction_engine.get_system_status()
                base_info.update({'correction_engine': engine_status})
            
            return base_info
            
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {'error': str(e)}
    
    def validate_setup(self) -> Dict[str, Any]:
        """セットアップの妥当性を検証"""
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # 基本検出器の検証
            if self.base_detector is None:
                validation_results['errors'].append("Base detector is None")
                validation_results['is_valid'] = False
            elif not hasattr(self.base_detector, 'detect_trend'):
                validation_results['warnings'].append("Base detector lacks detect_trend method")
            
            # 補正エンジンの検証
            if self.enable_correction:
                if self.correction_engine is None:
                    validation_results['errors'].append("Correction enabled but engine is None")
                    validation_results['is_valid'] = False
                
                if self.precision_tracker is None:
                    validation_results['warnings'].append("No precision tracker provided - limited functionality")
                    validation_results['recommendations'].append("Consider providing precision tracker for better corrections")
            
            # パフォーマンス関連の推奨事項
            if self.enable_correction and self.precision_tracker:
                validation_results['recommendations'].append("Regular validation of predictions recommended for optimal performance")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Setup validation failed: {e}")
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
            return validation_results

if __name__ == "__main__":
    # テスト用コード
    print("EnhancedTrendDetector モジュールが正常にロードされました")
