"""
市場状況検出システム - A→B市場分類システム基盤
リアルタイム市場状況の自動検出と分類機能を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings

# 既存の市場分類システムとの統合
from .market_conditions import (
    SimpleMarketCondition, DetailedMarketCondition,
    MarketMetrics, ClassificationResult, MarketConditions
)
from .market_classifier import MarketClassifier, ClassificationResult

class DetectionMethod(Enum):
    """検出手法の種類"""
    VOLATILITY_BASED = "volatility_based"
    MOMENTUM_BASED = "momentum_based"
    VOLUME_BASED = "volume_based"
    TECHNICAL_BASED = "technical_based"
    COMPOSITE = "composite"

@dataclass
class DetectionResult:
    """市場状況検出結果"""
    condition: MarketCondition
    strength: MarketStrength
    confidence: float
    method: DetectionMethod
    supporting_indicators: Dict[str, float]
    detection_time: datetime
    additional_info: Dict[str, any] = None

    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}

class MarketConditionDetector:
    """
    市場状況検出システムのメインクラス
    複数の手法を組み合わせて市場状況を自動検出・分類
    """
    
    def __init__(self, 
                 default_method: DetectionMethod = DetectionMethod.COMPOSITE,
                 volatility_threshold: float = 0.02,
                 momentum_threshold: float = 0.01,
                 confidence_threshold: float = 0.6,
                 lookback_period: int = 20):
        """
        市場状況検出器の初期化
        
        Args:
            default_method: デフォルト検出手法
            volatility_threshold: ボラティリティ閾値
            momentum_threshold: モメンタム閾値
            confidence_threshold: 信頼度閾値
            lookback_period: 分析期間
        """
        self.default_method = default_method
        self.volatility_threshold = volatility_threshold
        self.momentum_threshold = momentum_threshold
        self.confidence_threshold = confidence_threshold
        self.lookback_period = lookback_period
        
        # 市場分類器との統合
        self.market_classifier = MarketClassifier()
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # 検出結果キャッシュ
        self._cache = {}
        self._cache_timeout = timedelta(minutes=5)
        
        self.logger.info(f"MarketConditionDetector初期化完了 - 手法: {default_method}")

    def detect_market_condition(self, 
                               data: pd.DataFrame,
                               method: Optional[DetectionMethod] = None,
                               custom_params: Optional[Dict] = None) -> DetectionResult:
        """
        市場状況の検出
        
        Args:
            data: 市場データ (OHLCV形式)
            method: 検出手法 (None=デフォルト使用)
            custom_params: カスタムパラメータ
            
        Returns:
            DetectionResult: 検出結果
        """
        try:
            method = method or self.default_method
            cache_key = self._generate_cache_key(data, method)
            
            # キャッシュチェック
            if self._is_cache_valid(cache_key):
                self.logger.debug(f"キャッシュから結果を返却: {cache_key}")
                return self._cache[cache_key]['result']
            
            # データ検証
            if not self._validate_data(data):
                raise ValueError("無効なデータフォーマット")
            
            # 手法別検出実行
            if method == DetectionMethod.VOLATILITY_BASED:
                result = self._detect_by_volatility(data, custom_params)
            elif method == DetectionMethod.MOMENTUM_BASED:
                result = self._detect_by_momentum(data, custom_params)
            elif method == DetectionMethod.VOLUME_BASED:
                result = self._detect_by_volume(data, custom_params)
            elif method == DetectionMethod.TECHNICAL_BASED:
                result = self._detect_by_technical(data, custom_params)
            elif method == DetectionMethod.COMPOSITE:
                result = self._detect_by_composite(data, custom_params)
            else:
                raise ValueError(f"未対応の検出手法: {method}")
            
            # 結果をキャッシュ
            self._cache_result(cache_key, result)
            
            self.logger.info(f"市場状況検出完了: {result.condition} (信頼度: {result.confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"市場状況検出エラー: {e}")
            # フォールバック結果を返却
            return self._create_fallback_result(method)

    def _detect_by_volatility(self, data: pd.DataFrame, custom_params: Optional[Dict] = None) -> DetectionResult:
        """ボラティリティベース検出"""
        try:
            # パラメータ設定
            threshold = custom_params.get('threshold', self.volatility_threshold) if custom_params else self.volatility_threshold
            period = custom_params.get('period', self.lookback_period) if custom_params else self.lookback_period
            
            # ボラティリティ計算
            returns = data['Close'].pct_change().dropna()
            current_vol = returns.rolling(period).std().iloc[-1] * np.sqrt(252)  # 年率化
            historical_vol = returns.rolling(period * 3).std().mean() * np.sqrt(252)
            
            # 相対ボラティリティ
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            # 市場状況判定
            if vol_ratio > 1.5:
                condition = MarketCondition.HIGH_VOLATILITY
                strength = MarketStrength.STRONG if vol_ratio > 2.0 else MarketStrength.MODERATE
            elif vol_ratio < 0.7:
                condition = MarketCondition.LOW_VOLATILITY
                strength = MarketStrength.MODERATE
            else:
                condition = MarketCondition.NORMAL_VOLATILITY
                strength = MarketStrength.WEAK
            
            # 信頼度計算
            confidence = min(abs(vol_ratio - 1.0), 1.0)
            
            # 支持指標
            supporting_indicators = {
                'current_volatility': current_vol,
                'historical_volatility': historical_vol,
                'volatility_ratio': vol_ratio,
                'annualized_vol': current_vol
            }
            
            return DetectionResult(
                condition=condition,
                strength=strength,
                confidence=confidence,
                method=DetectionMethod.VOLATILITY_BASED,
                supporting_indicators=supporting_indicators,
                detection_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"ボラティリティベース検出エラー: {e}")
            return self._create_fallback_result(DetectionMethod.VOLATILITY_BASED)

    def _detect_by_momentum(self, data: pd.DataFrame, custom_params: Optional[Dict] = None) -> DetectionResult:
        """モメンタムベース検出"""
        try:
            # パラメータ設定
            threshold = custom_params.get('threshold', self.momentum_threshold) if custom_params else self.momentum_threshold
            period = custom_params.get('period', self.lookback_period) if custom_params else self.lookback_period
            
            # モメンタム計算
            returns = data['Close'].pct_change().dropna()
            momentum = returns.rolling(period).mean().iloc[-1]
            momentum_std = returns.rolling(period).std().iloc[-1]
            
            # Z-score計算
            z_score = abs(momentum / momentum_std) if momentum_std > 0 else 0
            
            # 市場状況判定
            if abs(momentum) > threshold:
                if momentum > 0:
                    condition = MarketCondition.BULLISH_TREND
                    strength = MarketStrength.STRONG if z_score > 2 else MarketStrength.MODERATE
                else:
                    condition = MarketCondition.BEARISH_TREND
                    strength = MarketStrength.STRONG if z_score > 2 else MarketStrength.MODERATE
            else:
                condition = MarketCondition.SIDEWAYS
                strength = MarketStrength.WEAK
            
            # 信頼度計算
            confidence = min(z_score / 3.0, 1.0)  # 3σまでで正規化
            
            # 支持指標
            supporting_indicators = {
                'momentum': momentum,
                'momentum_zscore': z_score,
                'momentum_std': momentum_std,
                'direction': 'bullish' if momentum > 0 else 'bearish'
            }
            
            return DetectionResult(
                condition=condition,
                strength=strength,
                confidence=confidence,
                method=DetectionMethod.MOMENTUM_BASED,
                supporting_indicators=supporting_indicators,
                detection_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"モメンタムベース検出エラー: {e}")
            return self._create_fallback_result(DetectionMethod.MOMENTUM_BASED)

    def _detect_by_volume(self, data: pd.DataFrame, custom_params: Optional[Dict] = None) -> DetectionResult:
        """出来高ベース検出"""
        try:
            period = custom_params.get('period', self.lookback_period) if custom_params else self.lookback_period
            
            # 出来高分析
            if 'Volume' not in data.columns:
                # 出来高データがない場合
                return self._create_fallback_result(DetectionMethod.VOLUME_BASED)
            
            volume = data['Volume']
            avg_volume = volume.rolling(period).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 価格と出来高の関係分析
            price_change = data['Close'].pct_change().iloc[-1]
            
            # 市場状況判定
            if volume_ratio > 1.5:
                if abs(price_change) > 0.02:
                    condition = MarketCondition.HIGH_ACTIVITY
                    strength = MarketStrength.STRONG
                else:
                    condition = MarketCondition.CONSOLIDATION
                    strength = MarketStrength.MODERATE
            elif volume_ratio < 0.7:
                condition = MarketCondition.LOW_ACTIVITY
                strength = MarketStrength.WEAK
            else:
                condition = MarketCondition.NORMAL_ACTIVITY
                strength = MarketStrength.MODERATE
            
            # 信頼度計算
            confidence = min(abs(volume_ratio - 1.0), 1.0)
            
            # 支持指標
            supporting_indicators = {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'price_change': price_change
            }
            
            return DetectionResult(
                condition=condition,
                strength=strength,
                confidence=confidence,
                method=DetectionMethod.VOLUME_BASED,
                supporting_indicators=supporting_indicators,
                detection_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"出来高ベース検出エラー: {e}")
            return self._create_fallback_result(DetectionMethod.VOLUME_BASED)

    def _detect_by_technical(self, data: pd.DataFrame, custom_params: Optional[Dict] = None) -> DetectionResult:
        """テクニカル指標ベース検出"""
        try:
            period = custom_params.get('period', self.lookback_period) if custom_params else self.lookback_period
            
            # 移動平均線分析
            close = data['Close']
            sma_short = close.rolling(period // 2).mean().iloc[-1]
            sma_long = close.rolling(period).mean().iloc[-1]
            current_price = close.iloc[-1]
            
            # RSI計算
            rsi = self._calculate_rsi(close, period)
            
            # ボリンジャーバンド
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(close, period)
            
            # 総合判定
            signals = []
            
            # 移動平均シグナル
            if current_price > sma_short > sma_long:
                signals.append(('bullish', 0.7))
            elif current_price < sma_short < sma_long:
                signals.append(('bearish', 0.7))
            else:
                signals.append(('sideways', 0.5))
            
            # RSIシグナル
            if rsi > 70:
                signals.append(('overbought', 0.6))
            elif rsi < 30:
                signals.append(('oversold', 0.6))
            else:
                signals.append(('neutral', 0.4))
            
            # ボリンジャーバンドシグナル
            if current_price > bb_upper:
                signals.append(('breakout_high', 0.8))
            elif current_price < bb_lower:
                signals.append(('breakout_low', 0.8))
            else:
                signals.append(('range_bound', 0.5))
            
            # 総合判定
            condition, strength, confidence = self._aggregate_technical_signals(signals)
            
            # 支持指標
            supporting_indicators = {
                'sma_short': sma_short,
                'sma_long': sma_long,
                'rsi': rsi,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'current_price': current_price,
                'signals': [s[0] for s in signals]
            }
            
            return DetectionResult(
                condition=condition,
                strength=strength,
                confidence=confidence,
                method=DetectionMethod.TECHNICAL_BASED,
                supporting_indicators=supporting_indicators,
                detection_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"テクニカル指標ベース検出エラー: {e}")
            return self._create_fallback_result(DetectionMethod.TECHNICAL_BASED)

    def _detect_by_composite(self, data: pd.DataFrame, custom_params: Optional[Dict] = None) -> DetectionResult:
        """複合手法による検出"""
        try:
            # 各手法で検出実行
            vol_result = self._detect_by_volatility(data, custom_params)
            mom_result = self._detect_by_momentum(data, custom_params)
            vol_data_result = self._detect_by_volume(data, custom_params)
            tech_result = self._detect_by_technical(data, custom_params)
            
            # 結果統合
            results = [vol_result, mom_result, vol_data_result, tech_result]
            valid_results = [r for r in results if r.confidence > 0.3]
            
            if not valid_results:
                return self._create_fallback_result(DetectionMethod.COMPOSITE)
            
            # 重み付き投票による統合
            condition_votes = {}
            strength_votes = {}
            total_confidence = 0
            
            for result in valid_results:
                weight = result.confidence
                condition_key = result.condition
                strength_key = result.strength
                
                condition_votes[condition_key] = condition_votes.get(condition_key, 0) + weight
                strength_votes[strength_key] = strength_votes.get(strength_key, 0) + weight
                total_confidence += weight
            
            # 最多得票の条件と強度を選択
            final_condition = max(condition_votes, key=condition_votes.get)
            final_strength = max(strength_votes, key=strength_votes.get)
            final_confidence = total_confidence / len(valid_results)
            
            # 全ての支持指標を統合
            all_indicators = {}
            for i, result in enumerate(valid_results):
                method_name = result.method.value
                all_indicators[f"{method_name}_confidence"] = result.confidence
                for key, value in result.supporting_indicators.items():
                    all_indicators[f"{method_name}_{key}"] = value
            
            return DetectionResult(
                condition=final_condition,
                strength=final_strength,
                confidence=min(final_confidence, 1.0),
                method=DetectionMethod.COMPOSITE,
                supporting_indicators=all_indicators,
                detection_time=datetime.now(),
                additional_info={
                    'component_results': [
                        {
                            'method': r.method.value,
                            'condition': r.condition.value,
                            'confidence': r.confidence
                        } for r in valid_results
                    ]
                }
            )
            
        except Exception as e:
            self.logger.error(f"複合手法検出エラー: {e}")
            return self._create_fallback_result(DetectionMethod.COMPOSITE)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI計算"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """ボリンジャーバンド計算"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper.iloc[-1], lower.iloc[-1], sma.iloc[-1]
        except:
            current_price = prices.iloc[-1]
            return current_price * 1.02, current_price * 0.98, current_price

    def _aggregate_technical_signals(self, signals: List[Tuple[str, float]]) -> Tuple[MarketCondition, MarketStrength, float]:
        """テクニカルシグナルの統合"""
        try:
            # シグナル分類
            bullish_signals = ['bullish', 'oversold', 'breakout_high']
            bearish_signals = ['bearish', 'overbought', 'breakout_low']
            neutral_signals = ['sideways', 'neutral', 'range_bound']
            
            bullish_score = sum(conf for sig, conf in signals if sig in bullish_signals)
            bearish_score = sum(conf for sig, conf in signals if sig in bearish_signals)
            neutral_score = sum(conf for sig, conf in signals if sig in neutral_signals)
            
            total_score = bullish_score + bearish_score + neutral_score
            
            # 条件判定
            if bullish_score > bearish_score and bullish_score > neutral_score:
                condition = MarketCondition.BULLISH_TREND
                strength = MarketStrength.STRONG if bullish_score / total_score > 0.6 else MarketStrength.MODERATE
            elif bearish_score > bullish_score and bearish_score > neutral_score:
                condition = MarketCondition.BEARISH_TREND
                strength = MarketStrength.STRONG if bearish_score / total_score > 0.6 else MarketStrength.MODERATE
            else:
                condition = MarketCondition.SIDEWAYS
                strength = MarketStrength.WEAK
            
            # 信頼度
            max_score = max(bullish_score, bearish_score, neutral_score)
            confidence = max_score / total_score if total_score > 0 else 0.3
            
            return condition, strength, confidence
            
        except:
            return MarketCondition.SIDEWAYS, MarketStrength.WEAK, 0.3

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """データ検証"""
        required_columns = ['Open', 'High', 'Low', 'Close']
        return all(col in data.columns for col in required_columns) and len(data) >= self.lookback_period

    def _generate_cache_key(self, data: pd.DataFrame, method: DetectionMethod) -> str:
        """キャッシュキー生成"""
        try:
            last_timestamp = str(data.index[-1]) if hasattr(data.index, '__getitem__') else str(len(data))
            return f"{method.value}_{last_timestamp}_{len(data)}"
        except:
            return f"{method.value}_{datetime.now().isoformat()}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """キャッシュ有効性チェック"""
        if cache_key not in self._cache:
            return False
        
        cache_time = self._cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self._cache_timeout

    def _cache_result(self, cache_key: str, result: DetectionResult):
        """結果をキャッシュ"""
        self._cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        # キャッシュサイズ制限
        if len(self._cache) > 100:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]

    def _create_fallback_result(self, method: DetectionMethod) -> DetectionResult:
        """フォールバック結果生成"""
        return DetectionResult(
            condition=MarketCondition.UNKNOWN,
            strength=MarketStrength.WEAK,
            confidence=0.1,
            method=method,
            supporting_indicators={'error': 'fallback_result'},
            detection_time=datetime.now(),
            additional_info={'is_fallback': True}
        )

    def get_detection_summary(self, data: pd.DataFrame) -> Dict[str, DetectionResult]:
        """全手法での検出結果サマリー"""
        try:
            summary = {}
            for method in DetectionMethod:
                result = self.detect_market_condition(data, method)
                summary[method.value] = result
            
            return summary
            
        except Exception as e:
            self.logger.error(f"検出サマリー作成エラー: {e}")
            return {}

    def clear_cache(self):
        """キャッシュクリア"""
        self._cache.clear()
        self.logger.info("検出結果キャッシュをクリアしました")

    def get_cache_info(self) -> Dict[str, any]:
        """キャッシュ情報取得"""
        return {
            'cache_size': len(self._cache),
            'cache_timeout_minutes': self._cache_timeout.total_seconds() / 60,
            'cache_keys': list(self._cache.keys())
        }

# 利便性関数
def detect_market_condition_simple(data: pd.DataFrame, 
                                  method: DetectionMethod = DetectionMethod.COMPOSITE) -> Dict[str, any]:
    """
    簡単な市場状況検出関数
    
    Args:
        data: 市場データ
        method: 検出手法
        
    Returns:
        Dict: 検出結果の辞書形式
    """
    detector = MarketConditionDetector()
    result = detector.detect_market_condition(data, method)
    
    return {
        'condition': result.condition.value,
        'strength': result.strength.value,
        'confidence': result.confidence,
        'method': result.method.value,
        'detection_time': result.detection_time.isoformat(),
        'supporting_indicators': result.supporting_indicators
    }

if __name__ == "__main__":
    # テスト用コード
    import sys
    import os
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== 市場状況検出システム テスト ===")
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    
    # トレンド相場データ
    trend_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.normal(0.1, 0.02, 100)),
        'High': 100 + np.cumsum(np.random.normal(0.1, 0.02, 100)) + np.random.uniform(0, 0.5, 100),
        'Low': 100 + np.cumsum(np.random.normal(0.1, 0.02, 100)) - np.random.uniform(0, 0.5, 100),
        'Close': 100 + np.cumsum(np.random.normal(0.1, 0.02, 100)),
        'Volume': np.random.uniform(1000000, 5000000, 100)
    }, index=dates)
    
    # 検出器テスト
    detector = MarketConditionDetector()
    
    print("\n1. 複合手法による検出")
    result = detector.detect_market_condition(trend_data)
    print(f"条件: {result.condition.value}")
    print(f"強度: {result.strength.value}")
    print(f"信頼度: {result.confidence:.3f}")
    
    print("\n2. 全手法サマリー")
    summary = detector.get_detection_summary(trend_data)
    for method, result in summary.items():
        print(f"{method}: {result.condition.value} (信頼度: {result.confidence:.3f})")
    
    print("\n3. 簡単検出関数テスト")
    simple_result = detect_market_condition_simple(trend_data)
    print(f"簡単検出結果: {simple_result['condition']} (信頼度: {simple_result['confidence']:.3f})")
    
    print("\n=== テスト完了 ===")
