"""
テクニカル指標分析システム - A→B市場分類システム基盤
複数のテクニカル指標を統合した市場分析機能を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import warnings

# 既存システムとの統合
from .market_conditions import MarketCondition, MarketStrength

class IndicatorType(Enum):
    """テクニカル指標の種類"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"

@dataclass
class IndicatorResult:
    """テクニカル指標の分析結果"""
    indicator_name: str
    indicator_type: IndicatorType
    value: float
    signal: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}

@dataclass
class TechnicalAnalysisResult:
    """テクニカル分析の総合結果"""
    overall_signal: str
    market_condition: MarketCondition
    market_strength: MarketStrength
    confidence: float
    individual_indicators: List[IndicatorResult]
    analysis_time: datetime
    summary_statistics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.summary_statistics is None:
            self.summary_statistics = {}

class TechnicalIndicatorAnalyzer:
    """
    テクニカル指標分析システムのメインクラス
    複数のテクニカル指標を計算・分析し、統合判定を提供
    """
    
    def __init__(self, 
                 default_periods: Dict[str, int] = None,
                 signal_threshold: float = 0.6,
                 confidence_threshold: float = 0.5):
        """
        テクニカル指標分析器の初期化
        
        Args:
            default_periods: デフォルト期間設定
            signal_threshold: シグナル閾値
            confidence_threshold: 信頼度閾値
        """
        self.default_periods = default_periods or {
            'sma_short': 5,
            'sma_medium': 20,
            'sma_long': 50,
            'ema_short': 12,
            'ema_long': 26,
            'rsi': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger': 20,
            'stochastic': 14,
            'atr': 14,
            'volume_sma': 20
        }
        
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        self.logger.info("TechnicalIndicatorAnalyzer初期化完了")

    def analyze_technical_indicators(self, 
                                   data: pd.DataFrame,
                                   indicators: Optional[List[str]] = None,
                                   custom_periods: Optional[Dict[str, int]] = None) -> TechnicalAnalysisResult:
        """
        テクニカル指標の総合分析
        
        Args:
            data: 市場データ (OHLCV形式)
            indicators: 分析対象指標 (None=全指標)
            custom_periods: カスタム期間設定
            
        Returns:
            TechnicalAnalysisResult: 分析結果
        """
        try:
            # データ検証
            if not self._validate_data(data):
                raise ValueError("無効なデータフォーマット")
            
            # 期間設定の統合
            periods = {**self.default_periods}
            if custom_periods:
                periods.update(custom_periods)
            
            # 指標リストの設定
            if indicators is None:
                indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic', 'atr', 'volume']
            
            # 各指標の分析実行
            results = []
            
            if 'sma' in indicators:
                results.extend(self._analyze_sma(data, periods))
            if 'ema' in indicators:
                results.extend(self._analyze_ema(data, periods))
            if 'rsi' in indicators:
                results.append(self._analyze_rsi(data, periods))
            if 'macd' in indicators:
                results.append(self._analyze_macd(data, periods))
            if 'bollinger' in indicators:
                results.extend(self._analyze_bollinger_bands(data, periods))
            if 'stochastic' in indicators:
                results.append(self._analyze_stochastic(data, periods))
            if 'atr' in indicators:
                results.append(self._analyze_atr(data, periods))
            if 'volume' in indicators and 'Volume' in data.columns:
                results.extend(self._analyze_volume(data, periods))
            
            # 結果統合
            overall_result = self._integrate_results(results)
            
            self.logger.info(f"テクニカル分析完了: {overall_result.overall_signal} (信頼度: {overall_result.confidence:.3f})")
            return overall_result
            
        except Exception as e:
            self.logger.error(f"テクニカル分析エラー: {e}")
            return self._create_fallback_result()

    def _analyze_sma(self, data: pd.DataFrame, periods: Dict[str, int]) -> List[IndicatorResult]:
        """移動平均線分析"""
        try:
            close = data['Close']
            results = []
            
            # 短期移動平均
            sma_short = close.rolling(periods['sma_short']).mean()
            # 中期移動平均
            sma_medium = close.rolling(periods['sma_medium']).mean()
            # 長期移動平均
            sma_long = close.rolling(periods['sma_long']).mean()
            
            current_price = close.iloc[-1]
            
            # 短期vs中期
            if not pd.isna(sma_short.iloc[-1]) and not pd.isna(sma_medium.iloc[-1]):
                if sma_short.iloc[-1] > sma_medium.iloc[-1]:
                    signal = 'bullish'
                    strength = min((sma_short.iloc[-1] - sma_medium.iloc[-1]) / sma_medium.iloc[-1] * 10, 1.0)
                elif sma_short.iloc[-1] < sma_medium.iloc[-1]:
                    signal = 'bearish'
                    strength = min((sma_medium.iloc[-1] - sma_short.iloc[-1]) / sma_medium.iloc[-1] * 10, 1.0)
                else:
                    signal = 'neutral'
                    strength = 0.3
                
                results.append(IndicatorResult(
                    indicator_name='SMA_Cross_Short_Medium',
                    indicator_type=IndicatorType.TREND,
                    value=sma_short.iloc[-1] / sma_medium.iloc[-1] - 1,
                    signal=signal,
                    strength=strength,
                    confidence=0.7,
                    additional_data={
                        'sma_short': sma_short.iloc[-1],
                        'sma_medium': sma_medium.iloc[-1]
                    }
                ))
            
            # 中期vs長期
            if not pd.isna(sma_medium.iloc[-1]) and not pd.isna(sma_long.iloc[-1]):
                if sma_medium.iloc[-1] > sma_long.iloc[-1]:
                    signal = 'bullish'
                    strength = min((sma_medium.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1] * 10, 1.0)
                elif sma_medium.iloc[-1] < sma_long.iloc[-1]:
                    signal = 'bearish'
                    strength = min((sma_long.iloc[-1] - sma_medium.iloc[-1]) / sma_long.iloc[-1] * 10, 1.0)
                else:
                    signal = 'neutral'
                    strength = 0.3
                
                results.append(IndicatorResult(
                    indicator_name='SMA_Cross_Medium_Long',
                    indicator_type=IndicatorType.TREND,
                    value=sma_medium.iloc[-1] / sma_long.iloc[-1] - 1,
                    signal=signal,
                    strength=strength,
                    confidence=0.8,
                    additional_data={
                        'sma_medium': sma_medium.iloc[-1],
                        'sma_long': sma_long.iloc[-1]
                    }
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"SMA分析エラー: {e}")
            return []

    def _analyze_ema(self, data: pd.DataFrame, periods: Dict[str, int]) -> List[IndicatorResult]:
        """指数移動平均線分析"""
        try:
            close = data['Close']
            results = []
            
            # EMA計算
            ema_short = close.ewm(span=periods['ema_short']).mean()
            ema_long = close.ewm(span=periods['ema_long']).mean()
            
            # EMAAクロス分析
            if not pd.isna(ema_short.iloc[-1]) and not pd.isna(ema_long.iloc[-1]):
                cross_value = ema_short.iloc[-1] / ema_long.iloc[-1] - 1
                
                if cross_value > 0.001:  # 0.1%以上の乖離
                    signal = 'bullish'
                    strength = min(abs(cross_value) * 50, 1.0)
                elif cross_value < -0.001:
                    signal = 'bearish'
                    strength = min(abs(cross_value) * 50, 1.0)
                else:
                    signal = 'neutral'
                    strength = 0.3
                
                results.append(IndicatorResult(
                    indicator_name='EMA_Cross',
                    indicator_type=IndicatorType.TREND,
                    value=cross_value,
                    signal=signal,
                    strength=strength,
                    confidence=0.75,
                    additional_data={
                        'ema_short': ema_short.iloc[-1],
                        'ema_long': ema_long.iloc[-1],
                        'cross_value': cross_value
                    }
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"EMA分析エラー: {e}")
            return []

    def _analyze_rsi(self, data: pd.DataFrame, periods: Dict[str, int]) -> IndicatorResult:
        """RSI分析"""
        try:
            close = data['Close']
            
            # RSI計算
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods['rsi']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods['rsi']).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            
            # シグナル判定
            if current_rsi > 70:
                signal = 'bearish'  # 売られすぎ
                strength = min((current_rsi - 70) / 30, 1.0)
            elif current_rsi < 30:
                signal = 'bullish'  # 買われすぎ
                strength = min((30 - current_rsi) / 30, 1.0)
            else:
                signal = 'neutral'
                strength = 1 - abs(current_rsi - 50) / 50  # 50に近いほど中立
            
            return IndicatorResult(
                indicator_name='RSI',
                indicator_type=IndicatorType.MOMENTUM,
                value=current_rsi,
                signal=signal,
                strength=strength,
                confidence=0.8,
                additional_data={
                    'rsi_value': current_rsi,
                    'overbought_threshold': 70,
                    'oversold_threshold': 30
                }
            )
            
        except Exception as e:
            self.logger.error(f"RSI分析エラー: {e}")
            return IndicatorResult('RSI', IndicatorType.MOMENTUM, 50.0, 'neutral', 0.3, 0.3)

    def _analyze_macd(self, data: pd.DataFrame, periods: Dict[str, int]) -> IndicatorResult:
        """MACD分析"""
        try:
            close = data['Close']
            
            # MACD計算
            ema_fast = close.ewm(span=periods['macd_fast']).mean()
            ema_slow = close.ewm(span=periods['macd_slow']).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=periods['macd_signal']).mean()
            histogram = macd_line - signal_line
            
            current_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
            current_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
            current_histogram = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
            
            # シグナル判定
            if current_macd > current_signal and current_histogram > 0:
                signal = 'bullish'
                strength = min(abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0, 1.0)
            elif current_macd < current_signal and current_histogram < 0:
                signal = 'bearish'
                strength = min(abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0, 1.0)
            else:
                signal = 'neutral'
                strength = 0.4
            
            return IndicatorResult(
                indicator_name='MACD',
                indicator_type=IndicatorType.MOMENTUM,
                value=current_histogram,
                signal=signal,
                strength=strength,
                confidence=0.7,
                additional_data={
                    'macd_line': current_macd,
                    'signal_line': current_signal,
                    'histogram': current_histogram
                }
            )
            
        except Exception as e:
            self.logger.error(f"MACD分析エラー: {e}")
            return IndicatorResult('MACD', IndicatorType.MOMENTUM, 0.0, 'neutral', 0.3, 0.3)

    def _analyze_bollinger_bands(self, data: pd.DataFrame, periods: Dict[str, int]) -> List[IndicatorResult]:
        """ボリンジャーバンド分析"""
        try:
            close = data['Close']
            results = []
            
            # ボリンジャーバンド計算
            sma = close.rolling(window=periods['bollinger']).mean()
            std = close.rolling(window=periods['bollinger']).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            current_price = close.iloc[-1]
            current_upper = upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else current_price * 1.02
            current_lower = lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else current_price * 0.98
            current_middle = sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else current_price
            
            # バンド内位置分析
            band_width = current_upper - current_lower
            price_position = (current_price - current_lower) / band_width if band_width > 0 else 0.5
            
            # 位置ベースシグナル
            if price_position > 0.8:
                signal = 'bearish'  # 上限近く
                strength = (price_position - 0.8) / 0.2
            elif price_position < 0.2:
                signal = 'bullish'  # 下限近く
                strength = (0.2 - price_position) / 0.2
            else:
                signal = 'neutral'
                strength = 1 - abs(price_position - 0.5) * 2
            
            results.append(IndicatorResult(
                indicator_name='Bollinger_Position',
                indicator_type=IndicatorType.VOLATILITY,
                value=price_position,
                signal=signal,
                strength=strength,
                confidence=0.7,
                additional_data={
                    'upper_band': current_upper,
                    'lower_band': current_lower,
                    'middle_band': current_middle,
                    'band_width': band_width,
                    'price_position': price_position
                }
            ))
            
            # バンド幅分析（ボラティリティ）
            if len(std) >= 20:
                avg_std = std.rolling(20).mean().iloc[-1]
                current_std = std.iloc[-1]
                volatility_ratio = current_std / avg_std if avg_std > 0 else 1.0
                
                if volatility_ratio > 1.5:
                    vol_signal = 'high_volatility'
                    vol_strength = min((volatility_ratio - 1.5) / 0.5, 1.0)
                elif volatility_ratio < 0.7:
                    vol_signal = 'low_volatility'
                    vol_strength = min((0.7 - volatility_ratio) / 0.3, 1.0)
                else:
                    vol_signal = 'normal_volatility'
                    vol_strength = 0.5
                
                results.append(IndicatorResult(
                    indicator_name='Bollinger_Volatility',
                    indicator_type=IndicatorType.VOLATILITY,
                    value=volatility_ratio,
                    signal=vol_signal,
                    strength=vol_strength,
                    confidence=0.6,
                    additional_data={
                        'current_std': current_std,
                        'average_std': avg_std,
                        'volatility_ratio': volatility_ratio
                    }
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"ボリンジャーバンド分析エラー: {e}")
            return []

    def _analyze_stochastic(self, data: pd.DataFrame, periods: Dict[str, int]) -> IndicatorResult:
        """ストキャスティクス分析"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # ストキャスティクス計算
            lowest_low = low.rolling(window=periods['stochastic']).min()
            highest_high = high.rolling(window=periods['stochastic']).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=3).mean()
            
            current_k = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50.0
            current_d = d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50.0
            
            # シグナル判定
            if current_k > 80 and current_d > 80:
                signal = 'bearish'  # 買われすぎ
                strength = min((current_k - 80) / 20, 1.0)
            elif current_k < 20 and current_d < 20:
                signal = 'bullish'  # 売られすぎ
                strength = min((20 - current_k) / 20, 1.0)
            else:
                signal = 'neutral'
                strength = 0.4
            
            return IndicatorResult(
                indicator_name='Stochastic',
                indicator_type=IndicatorType.MOMENTUM,
                value=current_k,
                signal=signal,
                strength=strength,
                confidence=0.65,
                additional_data={
                    'k_percent': current_k,
                    'd_percent': current_d,
                    'overbought_threshold': 80,
                    'oversold_threshold': 20
                }
            )
            
        except Exception as e:
            self.logger.error(f"ストキャスティクス分析エラー: {e}")
            return IndicatorResult('Stochastic', IndicatorType.MOMENTUM, 50.0, 'neutral', 0.3, 0.3)

    def _analyze_atr(self, data: pd.DataFrame, periods: Dict[str, int]) -> IndicatorResult:
        """ATR（Average True Range）分析"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # True Range計算
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR計算
            atr = true_range.rolling(window=periods['atr']).mean()
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
            
            # 相対ATR（価格に対する比率）
            current_price = close.iloc[-1]
            relative_atr = current_atr / current_price if current_price > 0 else 0
            
            # ボラティリティレベル判定
            if len(atr) >= 20:
                avg_atr = atr.rolling(20).mean().iloc[-1]
                atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
                
                if atr_ratio > 1.5:
                    signal = 'high_volatility'
                    strength = min((atr_ratio - 1.5) / 0.5, 1.0)
                elif atr_ratio < 0.7:
                    signal = 'low_volatility'
                    strength = min((0.7 - atr_ratio) / 0.3, 1.0)
                else:
                    signal = 'normal_volatility'
                    strength = 0.5
            else:
                signal = 'normal_volatility'
                strength = 0.5
                atr_ratio = 1.0
            
            return IndicatorResult(
                indicator_name='ATR',
                indicator_type=IndicatorType.VOLATILITY,
                value=relative_atr,
                signal=signal,
                strength=strength,
                confidence=0.6,
                additional_data={
                    'atr_value': current_atr,
                    'relative_atr': relative_atr,
                    'atr_ratio': atr_ratio if 'atr_ratio' in locals() else 1.0
                }
            )
            
        except Exception as e:
            self.logger.error(f"ATR分析エラー: {e}")
            return IndicatorResult('ATR', IndicatorType.VOLATILITY, 0.02, 'normal_volatility', 0.3, 0.3)

    def _analyze_volume(self, data: pd.DataFrame, periods: Dict[str, int]) -> List[IndicatorResult]:
        """出来高分析"""
        try:
            volume = data['Volume']
            close = data['Close']
            results = []
            
            # 出来高移動平均
            volume_sma = volume.rolling(window=periods['volume_sma']).mean()
            current_volume = volume.iloc[-1]
            avg_volume = volume_sma.iloc[-1] if not pd.isna(volume_sma.iloc[-1]) else current_volume
            
            # 出来高比率
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 出来高シグナル
            if volume_ratio > 1.5:
                vol_signal = 'high_volume'
                vol_strength = min((volume_ratio - 1.5) / 0.5, 1.0)
            elif volume_ratio < 0.7:
                vol_signal = 'low_volume'
                vol_strength = min((0.7 - volume_ratio) / 0.3, 1.0)
            else:
                vol_signal = 'normal_volume'
                vol_strength = 0.5
            
            results.append(IndicatorResult(
                indicator_name='Volume_Ratio',
                indicator_type=IndicatorType.VOLUME,
                value=volume_ratio,
                signal=vol_signal,
                strength=vol_strength,
                confidence=0.6,
                additional_data={
                    'current_volume': current_volume,
                    'average_volume': avg_volume,
                    'volume_ratio': volume_ratio
                }
            ))
            
            # 価格出来高分析
            price_change = close.pct_change().iloc[-1]
            if not pd.isna(price_change):
                if volume_ratio > 1.2 and abs(price_change) > 0.02:
                    if price_change > 0:
                        pv_signal = 'bullish'
                    else:
                        pv_signal = 'bearish'
                    pv_strength = min(volume_ratio * abs(price_change) * 10, 1.0)
                else:
                    pv_signal = 'neutral'
                    pv_strength = 0.4
                
                results.append(IndicatorResult(
                    indicator_name='Price_Volume',
                    indicator_type=IndicatorType.VOLUME,
                    value=price_change * volume_ratio,
                    signal=pv_signal,
                    strength=pv_strength,
                    confidence=0.7,
                    additional_data={
                        'price_change': price_change,
                        'volume_ratio': volume_ratio,
                        'combined_signal': price_change * volume_ratio
                    }
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"出来高分析エラー: {e}")
            return []

    def _integrate_results(self, results: List[IndicatorResult]) -> TechnicalAnalysisResult:
        """個別指標結果の統合"""
        try:
            if not results:
                return self._create_fallback_result()
            
            # シグナル重み付き集計
            bullish_score = 0
            bearish_score = 0
            neutral_score = 0
            total_weight = 0
            
            for result in results:
                weight = result.confidence * result.strength
                total_weight += weight
                
                if result.signal == 'bullish':
                    bullish_score += weight
                elif result.signal == 'bearish':
                    bearish_score += weight
                else:
                    neutral_score += weight
            
            # 全体シグナル決定
            if total_weight > 0:
                bullish_ratio = bullish_score / total_weight
                bearish_ratio = bearish_score / total_weight
                neutral_ratio = neutral_score / total_weight
                
                if bullish_ratio > self.signal_threshold:
                    overall_signal = 'bullish'
                    market_condition = MarketCondition.BULLISH_TREND
                    market_strength = MarketStrength.STRONG if bullish_ratio > 0.8 else MarketStrength.MODERATE
                    confidence = bullish_ratio
                elif bearish_ratio > self.signal_threshold:
                    overall_signal = 'bearish'
                    market_condition = MarketCondition.BEARISH_TREND
                    market_strength = MarketStrength.STRONG if bearish_ratio > 0.8 else MarketStrength.MODERATE
                    confidence = bearish_ratio
                else:
                    overall_signal = 'neutral'
                    market_condition = MarketCondition.SIDEWAYS
                    market_strength = MarketStrength.WEAK
                    confidence = max(neutral_ratio, 0.3)
            else:
                overall_signal = 'neutral'
                market_condition = MarketCondition.SIDEWAYS
                market_strength = MarketStrength.WEAK
                confidence = 0.3
            
            # サマリー統計
            summary_stats = {
                'total_indicators': len(results),
                'bullish_signals': sum(1 for r in results if r.signal == 'bullish'),
                'bearish_signals': sum(1 for r in results if r.signal == 'bearish'),
                'neutral_signals': sum(1 for r in results if r.signal == 'neutral'),
                'average_confidence': sum(r.confidence for r in results) / len(results),
                'average_strength': sum(r.strength for r in results) / len(results),
                'bullish_ratio': bullish_score / total_weight if total_weight > 0 else 0,
                'bearish_ratio': bearish_score / total_weight if total_weight > 0 else 0,
                'neutral_ratio': neutral_score / total_weight if total_weight > 0 else 0
            }
            
            return TechnicalAnalysisResult(
                overall_signal=overall_signal,
                market_condition=market_condition,
                market_strength=market_strength,
                confidence=confidence,
                individual_indicators=results,
                analysis_time=datetime.now(),
                summary_statistics=summary_stats
            )
            
        except Exception as e:
            self.logger.error(f"結果統合エラー: {e}")
            return self._create_fallback_result()

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """データ検証"""
        required_columns = ['Open', 'High', 'Low', 'Close']
        return all(col in data.columns for col in required_columns) and len(data) >= 20

    def _create_fallback_result(self) -> TechnicalAnalysisResult:
        """フォールバック結果生成"""
        return TechnicalAnalysisResult(
            overall_signal='neutral',
            market_condition=MarketCondition.SIDEWAYS,
            market_strength=MarketStrength.WEAK,
            confidence=0.1,
            individual_indicators=[],
            analysis_time=datetime.now(),
            summary_statistics={'error': 'fallback_result'}
        )

    def get_indicator_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """指標サマリー取得"""
        try:
            result = self.analyze_technical_indicators(data)
            return {
                'overall_signal': result.overall_signal,
                'market_condition': result.market_condition.value,
                'confidence': result.confidence,
                'total_indicators': len(result.individual_indicators),
                'summary_statistics': result.summary_statistics,
                'analysis_time': result.analysis_time.isoformat()
            }
        except Exception as e:
            self.logger.error(f"指標サマリー取得エラー: {e}")
            return {'error': str(e)}

# 利便性関数
def analyze_technical_simple(data: pd.DataFrame, 
                           indicators: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    簡単なテクニカル分析関数
    
    Args:
        data: 市場データ
        indicators: 分析対象指標
        
    Returns:
        Dict: 分析結果の辞書形式
    """
    analyzer = TechnicalIndicatorAnalyzer()
    result = analyzer.analyze_technical_indicators(data, indicators)
    
    return {
        'overall_signal': result.overall_signal,
        'market_condition': result.market_condition.value,
        'market_strength': result.market_strength.value,
        'confidence': result.confidence,
        'indicator_count': len(result.individual_indicators),
        'analysis_time': result.analysis_time.isoformat(),
        'summary': result.summary_statistics
    }

if __name__ == "__main__":
    # テスト用コード
    import sys
    import os
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== テクニカル指標分析システム テスト ===")
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    
    # トレンド相場データ
    base_price = 100
    trend_data = []
    for i in range(100):
        trend_data.append(base_price + i * 0.1 + np.random.normal(0, 0.5))
    
    test_data = pd.DataFrame({
        'Open': [p + np.random.normal(0, 0.1) for p in trend_data],
        'High': [p + abs(np.random.normal(0, 0.3)) for p in trend_data],
        'Low': [p - abs(np.random.normal(0, 0.3)) for p in trend_data],
        'Close': trend_data,
        'Volume': [np.random.uniform(1000000, 5000000) for _ in range(100)]
    }, index=dates)
    
    # 分析器テスト
    analyzer = TechnicalIndicatorAnalyzer()
    
    print("\n1. 全指標分析")
    result = analyzer.analyze_technical_indicators(test_data)
    print(f"総合シグナル: {result.overall_signal}")
    print(f"市場状況: {result.market_condition.value}")
    print(f"信頼度: {result.confidence:.3f}")
    print(f"分析指標数: {len(result.individual_indicators)}")
    
    print("\n2. 個別指標結果")
    for indicator in result.individual_indicators:
        print(f"  {indicator.indicator_name}: {indicator.signal} (強度: {indicator.strength:.2f})")
    
    print("\n3. 簡単分析関数テスト")
    simple_result = analyze_technical_simple(test_data, ['sma', 'rsi', 'macd'])
    print(f"簡単分析結果: {simple_result['overall_signal']} (信頼度: {simple_result['confidence']:.3f})")
    
    print("\n=== テスト完了 ===")
