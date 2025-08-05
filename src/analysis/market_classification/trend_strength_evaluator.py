"""
トレンド強度評価システム - A→B市場分類システム基盤
多面的なトレンド分析と強度評価機能を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.signal import argrelextrema

# 既存システムとの統合
from .market_conditions import MarketCondition, MarketStrength

class TrendDirection(Enum):
    """トレンド方向"""
    STRONG_UPTREND = "strong_uptrend"
    MODERATE_UPTREND = "moderate_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    MODERATE_DOWNTREND = "moderate_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

class TrendQuality(Enum):
    """トレンド品質"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"

class TrendMethod(Enum):
    """トレンド分析手法"""
    LINEAR_REGRESSION = "linear_regression"
    MOVING_AVERAGE = "moving_average"
    PEAK_TROUGH = "peak_trough"
    FRACTAL = "fractal"
    MOMENTUM = "momentum"
    COMPOSITE = "composite"

@dataclass
class TrendAnalysisResult:
    """トレンド分析結果"""
    method: TrendMethod
    direction: TrendDirection
    strength: float  # 0.0 - 1.0
    quality: TrendQuality
    confidence: float  # 0.0 - 1.0
    duration_estimate: int  # 推定継続期間（日数）
    supporting_metrics: Dict[str, float]
    analysis_time: datetime
    
    def __post_init__(self):
        if self.supporting_metrics is None:
            self.supporting_metrics = {}

@dataclass
class TrendStrengthResult:
    """トレンド強度評価の総合結果"""
    primary_direction: TrendDirection
    overall_strength: float
    overall_quality: TrendQuality
    confidence: float
    trend_analyses: List[TrendAnalysisResult]
    consensus_metrics: Dict[str, float]
    trend_characteristics: Dict[str, Any]
    evaluation_time: datetime

class TrendStrengthEvaluator:
    """
    トレンド強度評価システムのメインクラス
    複数の手法を用いてトレンドの方向と強度を総合評価
    """
    
    def __init__(self, 
                 default_lookback: int = 50,
                 short_ma_period: int = 5,
                 medium_ma_period: int = 20,
                 long_ma_period: int = 50,
                 strength_threshold: float = 0.6):
        """
        トレンド強度評価器の初期化
        
        Args:
            default_lookback: デフォルト分析期間
            short_ma_period: 短期移動平均期間
            medium_ma_period: 中期移動平均期間  
            long_ma_period: 長期移動平均期間
            strength_threshold: 強度閾値
        """
        self.default_lookback = default_lookback
        self.short_ma_period = short_ma_period
        self.medium_ma_period = medium_ma_period
        self.long_ma_period = long_ma_period
        self.strength_threshold = strength_threshold
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # トレンド強度レベル
        self.strength_levels = {
            'very_weak': 0.2,
            'weak': 0.4,
            'moderate': 0.6,
            'strong': 0.8,
            'very_strong': 1.0
        }
        
        # 分析結果キャッシュ
        self._trend_cache = {}
        self._cache_timeout = timedelta(minutes=5)
        
        self.logger.info("TrendStrengthEvaluator初期化完了")

    def evaluate_trend_strength(self, 
                               data: pd.DataFrame,
                               methods: Optional[List[TrendMethod]] = None,
                               custom_params: Optional[Dict] = None) -> TrendStrengthResult:
        """
        トレンド強度の総合評価
        
        Args:
            data: 市場データ (OHLCV形式)
            methods: 使用する分析手法 (None=全手法)
            custom_params: カスタムパラメータ
            
        Returns:
            TrendStrengthResult: 評価結果
        """
        try:
            # データ検証
            if not self._validate_data(data):
                raise ValueError("無効なデータフォーマット")
            
            # キャッシュチェック
            cache_key = self._generate_cache_key(data, methods)
            if self._is_cache_valid(cache_key):
                self.logger.debug(f"キャッシュから結果を返却: {cache_key}")
                return self._trend_cache[cache_key]['result']
            
            # 手法設定
            if methods is None:
                methods = [TrendMethod.LINEAR_REGRESSION, TrendMethod.MOVING_AVERAGE, 
                          TrendMethod.PEAK_TROUGH, TrendMethod.MOMENTUM]
            
            # パラメータ統合
            params = self._merge_params(custom_params)
            
            # 分析期間データの取得
            analysis_data = data.tail(self.default_lookback) if len(data) > self.default_lookback else data
            
            # 各手法でトレンド分析
            trend_analyses = []
            
            for method in methods:
                try:
                    analysis_result = self._analyze_trend(analysis_data, method, params)
                    if analysis_result:
                        trend_analyses.append(analysis_result)
                except Exception as e:
                    self.logger.warning(f"手法 {method.value} でエラー: {e}")
                    continue
            
            if not trend_analyses:
                return self._create_fallback_result()
            
            # 総合評価
            overall_result = self._synthesize_trend_results(trend_analyses)
            
            # 結果をキャッシュ
            self._cache_result(cache_key, overall_result)
            
            self.logger.info(f"トレンド強度評価完了: {overall_result.primary_direction.value} (強度: {overall_result.overall_strength:.3f})")
            return overall_result
            
        except Exception as e:
            self.logger.error(f"トレンド強度評価エラー: {e}")
            return self._create_fallback_result()

    def _analyze_trend(self, 
                      data: pd.DataFrame, 
                      method: TrendMethod, 
                      params: Dict) -> Optional[TrendAnalysisResult]:
        """個別手法でのトレンド分析"""
        try:
            if method == TrendMethod.LINEAR_REGRESSION:
                return self._linear_regression_trend(data, params)
            elif method == TrendMethod.MOVING_AVERAGE:
                return self._moving_average_trend(data, params)
            elif method == TrendMethod.PEAK_TROUGH:
                return self._peak_trough_trend(data, params)
            elif method == TrendMethod.FRACTAL:
                return self._fractal_trend(data, params)
            elif method == TrendMethod.MOMENTUM:
                return self._momentum_trend(data, params)
            elif method == TrendMethod.COMPOSITE:
                return self._composite_trend(data, params)
            else:
                self.logger.warning(f"未対応のトレンド分析手法: {method}")
                return None
                
        except Exception as e:
            self.logger.error(f"{method.value} トレンド分析エラー: {e}")
            return None

    def _linear_regression_trend(self, data: pd.DataFrame, params: Dict) -> TrendAnalysisResult:
        """線形回帰によるトレンド分析"""
        try:
            close = data['Close']
            
            # 線形回帰
            x = np.arange(len(close))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, close)
            
            # トレンド方向判定
            relative_slope = slope / close.mean() if close.mean() > 0 else 0
            
            # 強度計算（R二乗値とスロープの組み合わせ）
            r_squared = r_value ** 2
            slope_strength = min(abs(relative_slope) * 100, 1.0)  # 相対スロープを0-1に正規化
            strength = (r_squared + slope_strength) / 2
            
            # 方向分類
            if relative_slope > 0.002:  # 0.2%以上の日次上昇
                if strength > 0.8:
                    direction = TrendDirection.STRONG_UPTREND
                elif strength > 0.6:
                    direction = TrendDirection.MODERATE_UPTREND
                else:
                    direction = TrendDirection.WEAK_UPTREND
            elif relative_slope < -0.002:  # 0.2%以上の日次下降
                if strength > 0.8:
                    direction = TrendDirection.STRONG_DOWNTREND
                elif strength > 0.6:
                    direction = TrendDirection.MODERATE_DOWNTREND
                else:
                    direction = TrendDirection.WEAK_DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS
            
            # 品質評価
            if r_squared > 0.8:
                quality = TrendQuality.EXCELLENT
            elif r_squared > 0.6:
                quality = TrendQuality.GOOD
            elif r_squared > 0.4:
                quality = TrendQuality.FAIR
            elif r_squared > 0.2:
                quality = TrendQuality.POOR
            else:
                quality = TrendQuality.VERY_POOR
            
            # 信頼度（p値とR二乗の組み合わせ）
            confidence = r_squared * (1 - p_value)
            
            # 継続期間推定（現在のトレンドがいつまで続くか）
            if strength > 0.7:
                duration_estimate = min(int(30 * strength), 60)
            else:
                duration_estimate = max(int(15 * strength), 5)
            
            # 支持メトリクス
            supporting_metrics = {
                'slope': slope,
                'relative_slope': relative_slope,
                'r_squared': r_squared,
                'p_value': p_value,
                'standard_error': std_err,
                'trend_angle': np.degrees(np.arctan(relative_slope))
            }
            
            return TrendAnalysisResult(
                method=TrendMethod.LINEAR_REGRESSION,
                direction=direction,
                strength=strength,
                quality=quality,
                confidence=confidence,
                duration_estimate=duration_estimate,
                supporting_metrics=supporting_metrics,
                analysis_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"線形回帰トレンド分析エラー: {e}")
            raise

    def _moving_average_trend(self, data: pd.DataFrame, params: Dict) -> TrendAnalysisResult:
        """移動平均によるトレンド分析"""
        try:
            close = data['Close']
            
            # 移動平均計算
            ma_short = close.rolling(window=self.short_ma_period).mean()
            ma_medium = close.rolling(window=self.medium_ma_period).mean()
            ma_long = close.rolling(window=self.long_ma_period).mean()
            
            current_price = close.iloc[-1]
            
            # 移動平均の配列確認
            ma_alignment_score = 0
            total_comparisons = 0
            
            # 価格と短期MA
            if not pd.isna(ma_short.iloc[-1]):
                if current_price > ma_short.iloc[-1]:
                    ma_alignment_score += 1
                elif current_price < ma_short.iloc[-1]:
                    ma_alignment_score -= 1
                total_comparisons += 1
            
            # 短期MAと中期MA
            if not pd.isna(ma_short.iloc[-1]) and not pd.isna(ma_medium.iloc[-1]):
                if ma_short.iloc[-1] > ma_medium.iloc[-1]:
                    ma_alignment_score += 1
                elif ma_short.iloc[-1] < ma_medium.iloc[-1]:
                    ma_alignment_score -= 1
                total_comparisons += 1
            
            # 中期MAと長期MA
            if not pd.isna(ma_medium.iloc[-1]) and not pd.isna(ma_long.iloc[-1]):
                if ma_medium.iloc[-1] > ma_long.iloc[-1]:
                    ma_alignment_score += 1
                elif ma_medium.iloc[-1] < ma_long.iloc[-1]:
                    ma_alignment_score -= 1
                total_comparisons += 1
            
            # 正規化されたアライメントスコア
            normalized_alignment = ma_alignment_score / total_comparisons if total_comparisons > 0 else 0
            
            # MAの傾き計算
            ma_slopes = []
            for ma in [ma_short, ma_medium, ma_long]:
                if len(ma) >= 10 and not pd.isna(ma.iloc[-1]) and not pd.isna(ma.iloc[-10]):
                    slope = (ma.iloc[-1] - ma.iloc[-10]) / ma.iloc[-10]
                    ma_slopes.append(slope)
            
            avg_slope = np.mean(ma_slopes) if ma_slopes else 0
            
            # 強度計算
            alignment_strength = abs(normalized_alignment)
            slope_strength = min(abs(avg_slope) * 50, 1.0)
            strength = (alignment_strength + slope_strength) / 2
            
            # 方向判定
            if normalized_alignment > 0.6:
                if strength > 0.8:
                    direction = TrendDirection.STRONG_UPTREND
                elif strength > 0.6:
                    direction = TrendDirection.MODERATE_UPTREND
                else:
                    direction = TrendDirection.WEAK_UPTREND
            elif normalized_alignment < -0.6:
                if strength > 0.8:
                    direction = TrendDirection.STRONG_DOWNTREND
                elif strength > 0.6:
                    direction = TrendDirection.MODERATE_DOWNTREND
                else:
                    direction = TrendDirection.WEAK_DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS
            
            # 品質評価（MAの一貫性）
            ma_consistency = 1 - np.std(ma_slopes) if len(ma_slopes) > 1 else strength
            
            if ma_consistency > 0.8:
                quality = TrendQuality.EXCELLENT
            elif ma_consistency > 0.6:
                quality = TrendQuality.GOOD
            elif ma_consistency > 0.4:
                quality = TrendQuality.FAIR
            else:
                quality = TrendQuality.POOR
            
            # 信頼度
            confidence = strength * ma_consistency
            
            # 継続期間推定
            duration_estimate = int(20 + 30 * strength)
            
            # 支持メトリクス
            supporting_metrics = {
                'ma_alignment_score': normalized_alignment,
                'average_slope': avg_slope,
                'ma_consistency': ma_consistency,
                'short_ma': ma_short.iloc[-1] if not pd.isna(ma_short.iloc[-1]) else 0,
                'medium_ma': ma_medium.iloc[-1] if not pd.isna(ma_medium.iloc[-1]) else 0,
                'long_ma': ma_long.iloc[-1] if not pd.isna(ma_long.iloc[-1]) else 0
            }
            
            return TrendAnalysisResult(
                method=TrendMethod.MOVING_AVERAGE,
                direction=direction,
                strength=strength,
                quality=quality,
                confidence=confidence,
                duration_estimate=duration_estimate,
                supporting_metrics=supporting_metrics,
                analysis_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"移動平均トレンド分析エラー: {e}")
            raise

    def _peak_trough_trend(self, data: pd.DataFrame, params: Dict) -> TrendAnalysisResult:
        """ピーク・トラフ分析によるトレンド判定"""
        try:
            close = data['Close']
            
            # 極値検出（ローカルミニマ・マキシマ）
            window = params.get('peak_window', 5)
            
            # 高値・安値の極値を見つける
            highs = argrelextrema(close.values, np.greater, order=window)[0]
            lows = argrelextrema(close.values, np.less, order=window)[0]
            
            if len(highs) < 2 or len(lows) < 2:
                # 十分な極値がない場合
                return self._create_neutral_trend_result(TrendMethod.PEAK_TROUGH)
            
            # 最近の極値（直近3個ずつ）
            recent_highs = highs[-3:] if len(highs) >= 3 else highs
            recent_lows = lows[-3:] if len(lows) >= 3 else lows
            
            # 高値の傾向
            high_trend = 0
            if len(recent_highs) >= 2:
                high_prices = close.iloc[recent_highs]
                high_slope, _, high_r, _, _ = stats.linregress(recent_highs, high_prices)
                high_trend = high_slope / close.mean() if close.mean() > 0 else 0
            
            # 安値の傾向
            low_trend = 0
            if len(recent_lows) >= 2:
                low_prices = close.iloc[recent_lows]
                low_slope, _, low_r, _, _ = stats.linregress(recent_lows, low_prices)
                low_trend = low_slope / close.mean() if close.mean() > 0 else 0
            
            # トレンド方向決定
            avg_trend = (high_trend + low_trend) / 2
            trend_consistency = 1 - abs(high_trend - low_trend) / (abs(high_trend) + abs(low_trend) + 1e-6)
            
            # 強度計算
            strength = min(abs(avg_trend) * 100, 1.0) * trend_consistency
            
            # 方向分類
            if avg_trend > 0.001:
                if strength > 0.7:
                    direction = TrendDirection.STRONG_UPTREND
                elif strength > 0.5:
                    direction = TrendDirection.MODERATE_UPTREND
                else:
                    direction = TrendDirection.WEAK_UPTREND
            elif avg_trend < -0.001:
                if strength > 0.7:
                    direction = TrendDirection.STRONG_DOWNTREND
                elif strength > 0.5:
                    direction = TrendDirection.MODERATE_DOWNTREND
                else:
                    direction = TrendDirection.WEAK_DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS
            
            # 品質評価
            if trend_consistency > 0.8 and len(recent_highs) >= 3 and len(recent_lows) >= 3:
                quality = TrendQuality.EXCELLENT
            elif trend_consistency > 0.6:
                quality = TrendQuality.GOOD
            elif trend_consistency > 0.4:
                quality = TrendQuality.FAIR
            else:
                quality = TrendQuality.POOR
            
            # 信頼度
            confidence = strength * trend_consistency
            
            # 継続期間推定
            duration_estimate = int(15 + 25 * strength)
            
            # 支持メトリクス
            supporting_metrics = {
                'high_trend': high_trend,
                'low_trend': low_trend,
                'trend_consistency': trend_consistency,
                'peak_count': len(highs),
                'trough_count': len(lows),
                'avg_trend': avg_trend
            }
            
            return TrendAnalysisResult(
                method=TrendMethod.PEAK_TROUGH,
                direction=direction,
                strength=strength,
                quality=quality,
                confidence=confidence,
                duration_estimate=duration_estimate,
                supporting_metrics=supporting_metrics,
                analysis_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"ピーク・トラフ分析エラー: {e}")
            raise

    def _fractal_trend(self, data: pd.DataFrame, params: Dict) -> TrendAnalysisResult:
        """フラクタル分析によるトレンド判定"""
        try:
            close = data['Close']
            
            # フラクタル検出（5期間パターン）
            fractal_up = []
            fractal_down = []
            
            for i in range(2, len(close) - 2):
                # 上向きフラクタル（中央が最高値）
                if (close.iloc[i] > close.iloc[i-2] and 
                    close.iloc[i] > close.iloc[i-1] and 
                    close.iloc[i] > close.iloc[i+1] and 
                    close.iloc[i] > close.iloc[i+2]):
                    fractal_up.append(i)
                
                # 下向きフラクタル（中央が最安値）
                if (close.iloc[i] < close.iloc[i-2] and 
                    close.iloc[i] < close.iloc[i-1] and 
                    close.iloc[i] < close.iloc[i+1] and 
                    close.iloc[i] < close.iloc[i+2]):
                    fractal_down.append(i)
            
            if len(fractal_up) < 2 and len(fractal_down) < 2:
                return self._create_neutral_trend_result(TrendMethod.FRACTAL)
            
            # 最近のフラクタル分析
            recent_fractals = []
            
            # 上向きフラクタル
            for idx in fractal_up[-3:]:  # 直近3個
                recent_fractals.append(('up', idx, close.iloc[idx]))
            
            # 下向きフラクタル
            for idx in fractal_down[-3:]:  # 直近3個
                recent_fractals.append(('down', idx, close.iloc[idx]))
            
            # 時間順にソート
            recent_fractals.sort(key=lambda x: x[1])
            
            # フラクタルパターン分析
            if len(recent_fractals) >= 3:
                pattern_score = 0
                for i in range(1, len(recent_fractals)):
                    prev_fractal = recent_fractals[i-1]
                    curr_fractal = recent_fractals[i]
                    
                    if prev_fractal[0] == curr_fractal[0]:
                        continue  # 同種類のフラクタルは無視
                    
                    # 価格レベルの比較
                    if curr_fractal[2] > prev_fractal[2]:
                        pattern_score += 1
                    elif curr_fractal[2] < prev_fractal[2]:
                        pattern_score -= 1
                
                # 正規化
                max_comparisons = len(recent_fractals) - 1
                normalized_pattern = pattern_score / max_comparisons if max_comparisons > 0 else 0
            else:
                normalized_pattern = 0
            
            # 強度計算
            strength = min(abs(normalized_pattern), 1.0)
            
            # 方向判定
            if normalized_pattern > 0.3:
                if strength > 0.7:
                    direction = TrendDirection.STRONG_UPTREND
                elif strength > 0.5:
                    direction = TrendDirection.MODERATE_UPTREND
                else:
                    direction = TrendDirection.WEAK_UPTREND
            elif normalized_pattern < -0.3:
                if strength > 0.7:
                    direction = TrendDirection.STRONG_DOWNTREND
                elif strength > 0.5:
                    direction = TrendDirection.MODERATE_DOWNTREND
                else:
                    direction = TrendDirection.WEAK_DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS
            
            # 品質評価
            fractal_density = (len(fractal_up) + len(fractal_down)) / len(close)
            
            if 0.05 <= fractal_density <= 0.15:  # 適切なフラクタル密度
                quality = TrendQuality.GOOD
            elif 0.03 <= fractal_density <= 0.2:
                quality = TrendQuality.FAIR
            else:
                quality = TrendQuality.POOR
            
            # 信頼度
            confidence = strength * min(fractal_density / 0.1, 1.0)
            
            # 継続期間推定
            duration_estimate = int(10 + 20 * strength)
            
            # 支持メトリクス
            supporting_metrics = {
                'pattern_score': normalized_pattern,
                'fractal_up_count': len(fractal_up),
                'fractal_down_count': len(fractal_down),
                'fractal_density': fractal_density,
                'recent_fractal_count': len(recent_fractals)
            }
            
            return TrendAnalysisResult(
                method=TrendMethod.FRACTAL,
                direction=direction,
                strength=strength,
                quality=quality,
                confidence=confidence,
                duration_estimate=duration_estimate,
                supporting_metrics=supporting_metrics,
                analysis_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"フラクタル分析エラー: {e}")
            raise

    def _momentum_trend(self, data: pd.DataFrame, params: Dict) -> TrendAnalysisResult:
        """モメンタム分析によるトレンド判定"""
        try:
            close = data['Close']
            
            # 複数期間のモメンタム計算
            momentum_periods = [5, 10, 20]
            momentum_values = []
            
            for period in momentum_periods:
                if len(close) > period:
                    momentum = (close.iloc[-1] / close.iloc[-period-1] - 1) * 100
                    momentum_values.append(momentum)
            
            if not momentum_values:
                return self._create_neutral_trend_result(TrendMethod.MOMENTUM)
            
            # 平均モメンタム
            avg_momentum = np.mean(momentum_values)
            momentum_consistency = 1 - (np.std(momentum_values) / (abs(avg_momentum) + 1))
            
            # ROC (Rate of Change) 分析
            roc_values = []
            for i in range(1, min(10, len(close))):
                roc = (close.iloc[-1] / close.iloc[-i-1] - 1) * 100
                roc_values.append(roc)
            
            # モメンタムの加速度（2次微分的分析）
            if len(close) >= 15:
                short_momentum = (close.iloc[-1] / close.iloc[-6] - 1) * 100
                long_momentum = (close.iloc[-6] / close.iloc[-11] - 1) * 100
                momentum_acceleration = short_momentum - long_momentum
            else:
                momentum_acceleration = 0
            
            # 強度計算
            base_strength = min(abs(avg_momentum) / 10, 1.0)  # 10%を最大として正規化
            acceleration_factor = min(abs(momentum_acceleration) / 5, 0.3)  # 加速度ボーナス
            strength = min(base_strength + acceleration_factor, 1.0) * momentum_consistency
            
            # 方向判定
            if avg_momentum > 2:  # 2%以上
                if strength > 0.7:
                    direction = TrendDirection.STRONG_UPTREND
                elif strength > 0.5:
                    direction = TrendDirection.MODERATE_UPTREND
                else:
                    direction = TrendDirection.WEAK_UPTREND
            elif avg_momentum < -2:  # -2%以下
                if strength > 0.7:
                    direction = TrendDirection.STRONG_DOWNTREND
                elif strength > 0.5:
                    direction = TrendDirection.MODERATE_DOWNTREND
                else:
                    direction = TrendDirection.WEAK_DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS
            
            # 品質評価（モメンタムの一貫性）
            if momentum_consistency > 0.8:
                quality = TrendQuality.EXCELLENT
            elif momentum_consistency > 0.6:
                quality = TrendQuality.GOOD
            elif momentum_consistency > 0.4:
                quality = TrendQuality.FAIR
            else:
                quality = TrendQuality.POOR
            
            # 信頼度
            confidence = strength * momentum_consistency
            
            # 継続期間推定（モメンタムの持続性）
            if abs(momentum_acceleration) > 3:  # 強い加速
                duration_estimate = int(15 + 20 * strength)
            else:
                duration_estimate = int(10 + 15 * strength)
            
            # 支持メトリクス
            supporting_metrics = {
                'average_momentum': avg_momentum,
                'momentum_consistency': momentum_consistency,
                'momentum_acceleration': momentum_acceleration,
                'short_term_roc': roc_values[0] if roc_values else 0,
                'medium_term_roc': roc_values[4] if len(roc_values) > 4 else 0
            }
            
            return TrendAnalysisResult(
                method=TrendMethod.MOMENTUM,
                direction=direction,
                strength=strength,
                quality=quality,
                confidence=confidence,
                duration_estimate=duration_estimate,
                supporting_metrics=supporting_metrics,
                analysis_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"モメンタム分析エラー: {e}")
            raise

    def _composite_trend(self, data: pd.DataFrame, params: Dict) -> TrendAnalysisResult:
        """複合手法によるトレンド分析"""
        try:
            # 主要手法を実行
            methods = [TrendMethod.LINEAR_REGRESSION, TrendMethod.MOVING_AVERAGE, TrendMethod.MOMENTUM]
            results = []
            
            for method in methods:
                try:
                    result = self._analyze_trend(data, method, params)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"複合分析中の{method.value}でエラー: {e}")
                    continue
            
            if not results:
                return self._create_neutral_trend_result(TrendMethod.COMPOSITE)
            
            # 重み付き統合
            direction_scores = {direction: 0 for direction in TrendDirection}
            total_weight = 0
            
            for result in results:
                weight = result.confidence * result.strength
                direction_scores[result.direction] += weight
                total_weight += weight
            
            # 最高スコアの方向を選択
            primary_direction = max(direction_scores, key=direction_scores.get)
            
            # 統合強度
            direction_strength = direction_scores[primary_direction] / total_weight if total_weight > 0 else 0
            avg_strength = np.mean([r.strength for r in results])
            strength = (direction_strength + avg_strength) / 2
            
            # 統合品質（結果の一致度）
            same_direction_count = sum(1 for r in results if r.direction == primary_direction)
            agreement_ratio = same_direction_count / len(results)
            
            if agreement_ratio >= 0.8:
                quality = TrendQuality.EXCELLENT
            elif agreement_ratio >= 0.6:
                quality = TrendQuality.GOOD
            elif agreement_ratio >= 0.4:
                quality = TrendQuality.FAIR
            else:
                quality = TrendQuality.POOR
            
            # 統合信頼度
            confidence = np.mean([r.confidence for r in results]) * agreement_ratio
            
            # 継続期間推定（最長の推定値を採用）
            duration_estimate = max([r.duration_estimate for r in results])
            
            # 支持メトリクス
            supporting_metrics = {
                'method_count': len(results),
                'agreement_ratio': agreement_ratio,
                'direction_strength': direction_strength,
                'component_methods': [r.method.value for r in results]
            }
            
            return TrendAnalysisResult(
                method=TrendMethod.COMPOSITE,
                direction=primary_direction,
                strength=strength,
                quality=quality,
                confidence=confidence,
                duration_estimate=duration_estimate,
                supporting_metrics=supporting_metrics,
                analysis_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"複合トレンド分析エラー: {e}")
            raise

    def _synthesize_trend_results(self, analyses: List[TrendAnalysisResult]) -> TrendStrengthResult:
        """個別分析結果の統合"""
        try:
            if not analyses:
                return self._create_fallback_result()
            
            # 方向別重み付きスコア計算
            direction_scores = {direction: 0 for direction in TrendDirection}
            total_weight = 0
            
            for analysis in analyses:
                weight = analysis.confidence * analysis.strength
                direction_scores[analysis.direction] += weight
                total_weight += weight
            
            # 主要方向決定
            primary_direction = max(direction_scores, key=direction_scores.get)
            
            # 全体強度計算
            direction_consensus = direction_scores[primary_direction] / total_weight if total_weight > 0 else 0
            avg_strength = np.mean([a.strength for a in analyses])
            overall_strength = (direction_consensus + avg_strength) / 2
            
            # 全体品質評価
            same_direction_analyses = [a for a in analyses if a.direction == primary_direction]
            quality_scores = [a.quality for a in same_direction_analyses]
            
            if quality_scores:
                # 品質スコアの数値化
                quality_values = {'very_poor': 1, 'poor': 2, 'fair': 3, 'good': 4, 'excellent': 5}
                avg_quality_value = np.mean([quality_values[q.value] for q in quality_scores])
                
                if avg_quality_value >= 4.5:
                    overall_quality = TrendQuality.EXCELLENT
                elif avg_quality_value >= 3.5:
                    overall_quality = TrendQuality.GOOD
                elif avg_quality_value >= 2.5:
                    overall_quality = TrendQuality.FAIR
                elif avg_quality_value >= 1.5:
                    overall_quality = TrendQuality.POOR
                else:
                    overall_quality = TrendQuality.VERY_POOR
            else:
                overall_quality = TrendQuality.FAIR
            
            # 全体信頼度
            confidence = np.mean([a.confidence for a in analyses]) * direction_consensus
            
            # コンセンサスメトリクス
            consensus_metrics = {
                'direction_agreement': len(same_direction_analyses) / len(analyses),
                'strength_consistency': 1 - np.std([a.strength for a in analyses]),
                'confidence_average': np.mean([a.confidence for a in analyses]),
                'method_diversity': len(set(a.method for a in analyses)),
                'duration_consensus': np.mean([a.duration_estimate for a in analyses])
            }
            
            # トレンド特性
            trend_characteristics = {
                'dominant_method': max(analyses, key=lambda x: x.confidence).method.value,
                'trend_persistence': max([a.duration_estimate for a in analyses]),
                'trend_volatility': np.std([a.strength for a in analyses]),
                'method_results': {a.method.value: {
                    'direction': a.direction.value,
                    'strength': a.strength,
                    'confidence': a.confidence
                } for a in analyses}
            }
            
            return TrendStrengthResult(
                primary_direction=primary_direction,
                overall_strength=overall_strength,
                overall_quality=overall_quality,
                confidence=confidence,
                trend_analyses=analyses,
                consensus_metrics=consensus_metrics,
                trend_characteristics=trend_characteristics,
                evaluation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"トレンド結果統合エラー: {e}")
            return self._create_fallback_result()

    def _create_neutral_trend_result(self, method: TrendMethod) -> TrendAnalysisResult:
        """中立トレンド結果生成"""
        return TrendAnalysisResult(
            method=method,
            direction=TrendDirection.SIDEWAYS,
            strength=0.3,
            quality=TrendQuality.POOR,
            confidence=0.2,
            duration_estimate=10,
            supporting_metrics={'insufficient_data': True},
            analysis_time=datetime.now()
        )

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """データ検証"""
        required_columns = ['Open', 'High', 'Low', 'Close']
        return all(col in data.columns for col in required_columns) and len(data) >= 20

    def _merge_params(self, custom_params: Optional[Dict]) -> Dict:
        """パラメータ統合"""
        default_params = {
            'lookback_period': self.default_lookback,
            'short_ma': self.short_ma_period,
            'medium_ma': self.medium_ma_period,
            'long_ma': self.long_ma_period,
            'peak_window': 5
        }
        
        if custom_params:
            default_params.update(custom_params)
        
        return default_params

    def _generate_cache_key(self, data: pd.DataFrame, methods: Optional[List[TrendMethod]]) -> str:
        """キャッシュキー生成"""
        try:
            last_timestamp = str(data.index[-1]) if hasattr(data.index, '__getitem__') else str(len(data))
            methods_str = '_'.join([m.value for m in methods]) if methods else 'all'
            return f"trend_{last_timestamp}_{len(data)}_{methods_str}"
        except:
            return f"trend_{datetime.now().isoformat()}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """キャッシュ有効性チェック"""
        if cache_key not in self._trend_cache:
            return False
        
        cache_time = self._trend_cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self._cache_timeout

    def _cache_result(self, cache_key: str, result: TrendStrengthResult):
        """結果をキャッシュ"""
        self._trend_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        # キャッシュサイズ制限
        if len(self._trend_cache) > 50:
            oldest_key = min(self._trend_cache.keys(), key=lambda k: self._trend_cache[k]['timestamp'])
            del self._trend_cache[oldest_key]

    def _create_fallback_result(self) -> TrendStrengthResult:
        """フォールバック結果生成"""
        fallback_analysis = self._create_neutral_trend_result(TrendMethod.LINEAR_REGRESSION)
        
        return TrendStrengthResult(
            primary_direction=TrendDirection.SIDEWAYS,
            overall_strength=0.1,
            overall_quality=TrendQuality.POOR,
            confidence=0.1,
            trend_analyses=[fallback_analysis],
            consensus_metrics={'error': 'fallback'},
            trend_characteristics={'is_fallback': True},
            evaluation_time=datetime.now()
        )

    def clear_cache(self):
        """キャッシュクリア"""
        self._trend_cache.clear()
        self.logger.info("トレンド強度評価キャッシュをクリアしました")

    def get_trend_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """トレンドサマリー取得"""
        try:
            result = self.evaluate_trend_strength(data)
            return {
                'primary_direction': result.primary_direction.value,
                'overall_strength': result.overall_strength,
                'overall_quality': result.overall_quality.value,
                'confidence': result.confidence,
                'analysis_count': len(result.trend_analyses),
                'consensus_agreement': result.consensus_metrics.get('direction_agreement', 0),
                'evaluation_time': result.evaluation_time.isoformat()
            }
        except Exception as e:
            self.logger.error(f"トレンドサマリー取得エラー: {e}")
            return {'error': str(e)}

# 利便性関数
def evaluate_trend_strength_simple(data: pd.DataFrame, methods: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    簡単なトレンド強度評価関数
    
    Args:
        data: 市場データ
        methods: 使用手法名のリスト
        
    Returns:
        Dict: 評価結果の辞書形式
    """
    # 手法名を列挙型に変換
    if methods:
        method_enums = []
        for method_name in methods:
            try:
                method_enums.append(TrendMethod(method_name))
            except ValueError:
                continue
    else:
        method_enums = None
    
    evaluator = TrendStrengthEvaluator()
    result = evaluator.evaluate_trend_strength(data, method_enums)
    
    return {
        'direction': result.primary_direction.value,
        'strength': result.overall_strength,
        'quality': result.overall_quality.value,
        'confidence': result.confidence,
        'method_count': len(result.trend_analyses),
        'evaluation_time': result.evaluation_time.isoformat()
    }

if __name__ == "__main__":
    # テスト用コード
    import sys
    import os
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== トレンド強度評価システム テスト ===")
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    
    # 明確なトレンドデータ
    base_price = 100
    trend_prices = []
    
    for i in range(100):
        # 上昇トレンド + ノイズ
        trend_component = base_price * (1 + 0.002 * i)  # 日次0.2%上昇
        noise = np.random.normal(0, trend_component * 0.01)  # 1%のノイズ
        trend_prices.append(trend_component + noise)
    
    test_data = pd.DataFrame({
        'Open': [p + np.random.normal(0, p*0.005) for p in trend_prices],
        'High': [p + abs(np.random.normal(0, p*0.008)) for p in trend_prices],
        'Low': [p - abs(np.random.normal(0, p*0.008)) for p in trend_prices],
        'Close': trend_prices,
        'Volume': [np.random.uniform(1000000, 5000000) for _ in range(100)]
    }, index=dates)
    
    # 評価器テスト
    evaluator = TrendStrengthEvaluator()
    
    print("\n1. 全手法評価")
    result = evaluator.evaluate_trend_strength(test_data)
    print(f"主要方向: {result.primary_direction.value}")
    print(f"全体強度: {result.overall_strength:.3f}")
    print(f"全体品質: {result.overall_quality.value}")
    print(f"信頼度: {result.confidence:.3f}")
    
    print("\n2. 個別手法結果")
    for analysis in result.trend_analyses:
        print(f"  {analysis.method.value}: {analysis.direction.value} (強度: {analysis.strength:.2f})")
    
    print("\n3. コンセンサスメトリクス")
    for metric, value in result.consensus_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\n4. 簡単評価関数テスト")
    simple_result = evaluate_trend_strength_simple(test_data, ['linear_regression', 'moving_average'])
    print(f"簡単評価結果: {simple_result['direction']} (強度: {simple_result['strength']:.3f})")
    
    print("\n=== テスト完了 ===")
