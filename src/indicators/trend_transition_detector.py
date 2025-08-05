"""
Module: Trend Transition Detector
File: trend_transition_detector.py
Description: 
  トレンド移行期検出エンジン - 複数指標による高精度移行期判定
  2-2-2「トレンド移行期の特別処理ルール」の検出コンポーネント

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - pandas
  - numpy
  - indicators.unified_trend_detector
  - preprocessing.volatility
  - preprocessing.returns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

# プロジェクトパスの追加
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 内部モジュールのインポート
try:
    from indicators.unified_trend_detector import UnifiedTrendDetector
    from preprocessing.volatility import add_volatility
    from preprocessing.returns import add_returns
except ImportError as e:
    print(f"Import warning: {e}")

# ロガー設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class TransitionDetectionResult:
    """移行期検出結果"""
    is_transition_period: bool
    transition_type: str  # 'unknown_to_trend', 'trend_to_trend', 'trend_to_range', 'range_to_trend'
    confidence_score: float  # 0-1
    volatility_factor: float
    trend_strength_change: float
    detection_timestamp: datetime
    indicators_used: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    recommended_actions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'is_transition_period': self.is_transition_period,
            'transition_type': self.transition_type,
            'confidence_score': self.confidence_score,
            'volatility_factor': self.volatility_factor,
            'trend_strength_change': self.trend_strength_change,
            'detection_timestamp': self.detection_timestamp.isoformat(),
            'indicators_used': self.indicators_used,
            'risk_level': self.risk_level,
            'recommended_actions': self.recommended_actions
        }

class TrendTransitionDetector:
    """
    トレンド移行期検出エンジン
    
    機能:
    1. 複数指標による移行期検出
    2. リスクレベル評価
    3. 推奨アクション生成
    """
    
    def __init__(self, 
                 volatility_threshold: float = 1.5,
                 confidence_change_threshold: float = 0.3,
                 trend_strength_threshold: float = 0.4,
                 lookback_period: int = 10,
                 detection_sensitivity: str = "medium"):
        """
        初期化
        
        Parameters:
            volatility_threshold: ボラティリティ閾値（平均の何倍で移行期とするか）
            confidence_change_threshold: 信頼度変化閾値
            trend_strength_threshold: トレンド強度変化閾値
            lookback_period: 比較用遡り期間
            detection_sensitivity: 検出感度 ('low', 'medium', 'high')
        """
        self.volatility_threshold = volatility_threshold
        self.confidence_change_threshold = confidence_change_threshold
        self.trend_strength_threshold = trend_strength_threshold
        self.lookback_period = lookback_period
        
        # 感度に応じた閾値調整
        sensitivity_multipliers = {
            'low': 1.5,
            'medium': 1.0,
            'high': 0.7
        }
        multiplier = sensitivity_multipliers.get(detection_sensitivity, 1.0)
        
        self.volatility_threshold *= multiplier
        self.confidence_change_threshold *= multiplier
        self.trend_strength_threshold *= multiplier
        
        # 内部状態
        self._cache = {}
        self._last_detection_time = None
        
        logger.info(f"TrendTransitionDetector initialized with sensitivity: {detection_sensitivity}")
    
    def detect_transition(self, 
                         data: pd.DataFrame, 
                         strategy_name: str = "default",
                         price_column: str = "Adj Close") -> TransitionDetectionResult:
        """
        トレンド移行期を検出
        
        Parameters:
            data: 株価データ
            strategy_name: 戦略名
            price_column: 価格カラム名
            
        Returns:
            TransitionDetectionResult: 検出結果
        """
        try:
            # データの前処理
            processed_data = self._preprocess_data(data, price_column)
            
            # 複数指標による検出
            detection_results = self._run_multiple_indicators(
                processed_data, strategy_name, price_column
            )
            
            # 統合判定
            final_result = self._integrate_detection_results(detection_results)
            
            # キャッシュ更新
            self._update_cache(final_result)
            
            logger.debug(f"Transition detection completed: {final_result.transition_type}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in transition detection: {e}")
            return self._create_error_result()
    
    def _preprocess_data(self, data: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """データ前処理"""
        processed = data.copy()
        
        # リターン計算
        try:
            processed = add_returns(processed, price_column)
        except Exception:
            processed['Daily Return'] = processed[price_column].pct_change()
        
        # ボラティリティ計算
        try:
            processed = add_volatility(processed)
        except Exception:
            processed['Volatility'] = processed['Daily Return'].rolling(20).std() * np.sqrt(252)
        
        return processed
    
    def _run_multiple_indicators(self, 
                                data: pd.DataFrame, 
                                strategy_name: str, 
                                price_column: str) -> Dict[str, Any]:
        """複数指標による検出実行"""
        results = {}
        
        # 1. ボラティリティ変化検出
        results['volatility'] = self._detect_volatility_change(data)
        
        # 2. トレンド信頼度変化検出
        results['confidence'] = self._detect_confidence_change(data, strategy_name, price_column)
        
        # 3. トレンド強度変化検出
        results['trend_strength'] = self._detect_trend_strength_change(data, strategy_name, price_column)
        
        # 4. 価格パターン変化検出
        results['price_pattern'] = self._detect_price_pattern_change(data, price_column)
        
        return results
    
    def _detect_volatility_change(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ボラティリティ変化検出"""
        if 'Volatility' not in data.columns or len(data) < self.lookback_period * 2:
            return {'detected': False, 'factor': 1.0, 'confidence': 0.0}
        
        # 最近の期間と過去の期間のボラティリティを比較
        recent_vol = data['Volatility'].tail(self.lookback_period).mean()
        past_vol = data['Volatility'].iloc[-(self.lookback_period*2):-self.lookback_period].mean()
        
        if past_vol == 0:
            return {'detected': False, 'factor': 1.0, 'confidence': 0.0}
        
        vol_factor = recent_vol / past_vol
        is_detected = vol_factor > self.volatility_threshold
        
        # 信頼度計算（変化の明確さに基づく）
        confidence = min(1.0, (vol_factor - 1.0) / self.volatility_threshold)
        
        return {
            'detected': is_detected,
            'factor': vol_factor,
            'confidence': max(0.0, confidence),
            'change_direction': 'increase' if vol_factor > 1.0 else 'decrease'
        }
    
    def _detect_confidence_change(self, 
                                 data: pd.DataFrame, 
                                 strategy_name: str, 
                                 price_column: str) -> Dict[str, Any]:
        """信頼度変化検出"""
        try:
            detector = UnifiedTrendDetector(data, price_column, strategy_name)
            
            # 現在と過去の信頼度を比較
            current_confidence = detector.get_confidence_score(0)
            past_confidence = detector.get_confidence_score(self.lookback_period)
            
            confidence_change = abs(current_confidence - past_confidence)
            is_detected = confidence_change > self.confidence_change_threshold
            
            return {
                'detected': is_detected,
                'change': confidence_change,
                'current': current_confidence,
                'past': past_confidence,
                'confidence': confidence_change / self.confidence_change_threshold
            }
            
        except Exception as e:
            logger.warning(f"Confidence change detection failed: {e}")
            return {'detected': False, 'change': 0.0, 'confidence': 0.0}
    
    def _detect_trend_strength_change(self, 
                                    data: pd.DataFrame, 
                                    strategy_name: str, 
                                    price_column: str) -> Dict[str, Any]:
        """トレンド強度変化検出"""
        try:
            # 移動平均の傾きでトレンド強度を測定
            short_ma = data[price_column].rolling(10).mean()
            medium_ma = data[price_column].rolling(20).mean()
            
            # 最近の傾き
            recent_short_slope = (short_ma.iloc[-1] - short_ma.iloc[-self.lookback_period]) / self.lookback_period
            recent_medium_slope = (medium_ma.iloc[-1] - medium_ma.iloc[-self.lookback_period]) / self.lookback_period
            
            # 過去の傾き
            past_short_slope = (short_ma.iloc[-self.lookback_period] - short_ma.iloc[-self.lookback_period*2]) / self.lookback_period
            past_medium_slope = (medium_ma.iloc[-self.lookback_period] - medium_ma.iloc[-self.lookback_period*2]) / self.lookback_period
            
            # 変化量計算
            short_change = abs(recent_short_slope - past_short_slope)
            medium_change = abs(recent_medium_slope - past_medium_slope)
            
            avg_change = (short_change + medium_change) / 2
            is_detected = avg_change > self.trend_strength_threshold
            
            return {
                'detected': is_detected,
                'change': avg_change,
                'confidence': min(1.0, avg_change / self.trend_strength_threshold),
                'recent_strength': (abs(recent_short_slope) + abs(recent_medium_slope)) / 2,
                'past_strength': (abs(past_short_slope) + abs(past_medium_slope)) / 2
            }
            
        except Exception as e:
            logger.warning(f"Trend strength detection failed: {e}")
            return {'detected': False, 'change': 0.0, 'confidence': 0.0}
    
    def _detect_price_pattern_change(self, data: pd.DataFrame, price_column: str) -> Dict[str, Any]:
        """価格パターン変化検出"""
        try:
            if len(data) < self.lookback_period * 2:
                return {'detected': False, 'confidence': 0.0}
            
            # 最近の価格範囲
            recent_data = data[price_column].tail(self.lookback_period)
            recent_range = recent_data.max() - recent_data.min()
            recent_mean = recent_data.mean()
            
            # 過去の価格範囲
            past_data = data[price_column].iloc[-(self.lookback_period*2):-self.lookback_period]
            past_range = past_data.max() - past_data.min()
            past_mean = past_data.mean()
            
            # パターン変化指標
            range_change_ratio = recent_range / past_range if past_range > 0 else 1.0
            mean_change_ratio = abs(recent_mean - past_mean) / past_mean if past_mean > 0 else 0.0
            
            # 検出判定（範囲変化または平均変化が大きい）
            is_detected = range_change_ratio > 1.5 or mean_change_ratio > 0.1
            
            confidence = min(1.0, max(range_change_ratio - 1.0, mean_change_ratio * 10))
            
            return {
                'detected': is_detected,
                'range_change_ratio': range_change_ratio,
                'mean_change_ratio': mean_change_ratio,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.warning(f"Price pattern detection failed: {e}")
            return {'detected': False, 'confidence': 0.0}
    
    def _integrate_detection_results(self, results: Dict[str, Any]) -> TransitionDetectionResult:
        """検出結果の統合判定"""
        # 各指標の検出状況
        detections = []
        confidences = []
        indicators_used = []
        
        for indicator, result in results.items():
            if result.get('detected', False):
                detections.append(True)
                confidences.append(result.get('confidence', 0.0))
                indicators_used.append(indicator)
            else:
                detections.append(False)
                confidences.append(0.0)
        
        # 総合判定
        detection_count = sum(detections)
        is_transition = detection_count >= 2  # 2つ以上の指標で検出
        
        # 信頼度スコア計算
        avg_confidence = np.mean(confidences) if confidences else 0.0
        confidence_score = min(1.0, avg_confidence * (detection_count / len(results)))
        
        # 移行タイプ判定
        transition_type = self._determine_transition_type(results)
        
        # リスクレベル評価
        risk_level = self._evaluate_risk_level(results, confidence_score)
        
        # 推奨アクション生成
        recommended_actions = self._generate_recommendations(results, is_transition, risk_level)
        
        return TransitionDetectionResult(
            is_transition_period=is_transition,
            transition_type=transition_type,
            confidence_score=confidence_score,
            volatility_factor=results.get('volatility', {}).get('factor', 1.0),
            trend_strength_change=results.get('trend_strength', {}).get('change', 0.0),
            detection_timestamp=datetime.now(),
            indicators_used=indicators_used,
            risk_level=risk_level,
            recommended_actions=recommended_actions
        )
    
    def _determine_transition_type(self, results: Dict[str, Any]) -> str:
        """移行タイプ判定"""
        confidence_result = results.get('confidence', {})
        trend_result = results.get('trend_strength', {})
        
        if not confidence_result.get('detected', False):
            return 'stable'
        
        current_conf = confidence_result.get('current', 0.5)
        past_conf = confidence_result.get('past', 0.5)
        
        recent_strength = trend_result.get('recent_strength', 0.0)
        past_strength = trend_result.get('past_strength', 0.0)
        
        # 移行タイプの判定ロジック
        if current_conf < 0.5 and past_conf >= 0.5:
            return 'trend_to_range'
        elif current_conf >= 0.5 and past_conf < 0.5:
            return 'range_to_trend'
        elif abs(recent_strength - past_strength) > 0.5:
            return 'trend_to_trend'
        else:
            return 'unknown_transition'
    
    def _evaluate_risk_level(self, results: Dict[str, Any], confidence_score: float) -> str:
        """リスクレベル評価"""
        vol_factor = results.get('volatility', {}).get('factor', 1.0)
        
        # 複合リスク評価
        risk_score = 0.0
        
        # ボラティリティ要因
        if vol_factor > 2.0:
            risk_score += 0.4
        elif vol_factor > 1.5:
            risk_score += 0.2
        
        # 信頼度要因
        if confidence_score > 0.7:
            risk_score += 0.3
        elif confidence_score > 0.5:
            risk_score += 0.2
        
        # 検出指標数要因
        detection_count = sum(1 for r in results.values() if r.get('detected', False))
        if detection_count >= 3:
            risk_score += 0.3
        elif detection_count >= 2:
            risk_score += 0.2
        
        # リスクレベル判定
        if risk_score >= 0.7:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, 
                                results: Dict[str, Any], 
                                is_transition: bool, 
                                risk_level: str) -> List[str]:
        """推奨アクション生成"""
        recommendations = []
        
        if not is_transition:
            recommendations.append("normal_operation")
            return recommendations
        
        # リスクレベル別推奨
        if risk_level == 'high':
            recommendations.extend([
                "restrict_new_entries",
                "reduce_position_sizes", 
                "tighten_stop_losses",
                "increase_monitoring_frequency"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "cautious_entry_only",
                "maintain_current_positions",
                "monitor_closely"
            ])
        else:
            recommendations.extend([
                "allow_selective_entries",
                "standard_risk_management"
            ])
        
        # 指標別推奨
        if results.get('volatility', {}).get('detected', False):
            recommendations.append("adjust_for_volatility")
        
        if results.get('confidence', {}).get('detected', False):
            recommendations.append("verify_trend_signals")
        
        return recommendations
    
    def _create_error_result(self) -> TransitionDetectionResult:
        """エラー時のデフォルト結果"""
        return TransitionDetectionResult(
            is_transition_period=False,
            transition_type='error',
            confidence_score=0.0,
            volatility_factor=1.0,
            trend_strength_change=0.0,
            detection_timestamp=datetime.now(),
            indicators_used=[],
            risk_level='unknown',
            recommended_actions=['error_state']
        )
    
    def _update_cache(self, result: TransitionDetectionResult):
        """キャッシュ更新"""
        self._cache['last_result'] = result
        self._last_detection_time = result.detection_timestamp
    
    def get_last_result(self) -> Optional[TransitionDetectionResult]:
        """最後の検出結果を取得"""
        return self._cache.get('last_result')

# 便利関数
def detect_trend_transition(data: pd.DataFrame, 
                          strategy_name: str = "default",
                          price_column: str = "Adj Close",
                          **kwargs) -> TransitionDetectionResult:
    """
    トレンド移行期検出の便利関数
    
    Parameters:
        data: 株価データ
        strategy_name: 戦略名
        price_column: 価格カラム名
        **kwargs: TrendTransitionDetectorの初期化パラメータ
        
    Returns:
        TransitionDetectionResult: 検出結果
    """
    detector = TrendTransitionDetector(**kwargs)
    return detector.detect_transition(data, strategy_name, price_column)

if __name__ == "__main__":
    # テスト用のサンプル実行
    print("TrendTransitionDetector - 開発版テスト")
    
    # サンプルデータ作成
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Adj Close': prices
    })
    
    # 検出テスト
    result = detect_trend_transition(sample_data)
    print(f"検出結果: {result.transition_type}, 信頼度: {result.confidence_score:.3f}")
