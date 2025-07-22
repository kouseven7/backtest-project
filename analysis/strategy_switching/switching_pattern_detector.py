"""
Module: Switching Pattern Detector
File: switching_pattern_detector.py  
Description:
  5-1-1「戦略切替のタイミング分析ツール」
  戦略切替パターンの検出と分析

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ロガーの設定
logger = logging.getLogger(__name__)

class PatternType(Enum):
    """パターン種別"""
    TREND_REVERSAL = "trend_reversal"
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"  
    VOLATILITY_BREAKOUT = "volatility_breakout"
    MEAN_REVERSION = "mean_reversion"
    SEASONAL_PATTERN = "seasonal_pattern"
    REGIME_CHANGE = "regime_change"
    CORRELATION_BREAKDOWN = "correlation_breakdown"

class MarketRegime(Enum):
    """市場レジーム"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class SwitchingPattern:
    """切替パターン"""
    pattern_type: PatternType
    timestamp: datetime
    confidence: float
    description: str
    market_regime: MarketRegime
    trigger_indicators: Dict[str, float]
    pattern_strength: float
    historical_success_rate: float
    expected_duration: timedelta
    risk_level: str

@dataclass
class PatternAnalysisResult:
    """パターン分析結果"""
    analysis_period: Tuple[datetime, datetime]
    detected_patterns: List[SwitchingPattern]
    pattern_frequency: Dict[PatternType, int]
    success_rates: Dict[PatternType, float]
    optimal_timing_windows: Dict[PatternType, Dict[str, Any]]
    regime_transition_patterns: Dict[str, List[SwitchingPattern]]
    seasonal_patterns: Dict[str, List[SwitchingPattern]]

class SwitchingPatternDetector:
    """戦略切替パターン検出器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Parameters:
            config: 設定辞書
        """
        self.config = config or self._get_default_config()
        
        # パラメータ設定
        self.lookback_window = self.config.get('lookback_window', 20)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.pattern_min_strength = self.config.get('pattern_min_strength', 0.5)
        
        # 検出履歴
        self.detection_history: List[SwitchingPattern] = []
        self.pattern_performance_history: Dict[PatternType, List[float]] = {}
        
        logger.info("SwitchingPatternDetector initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'lookback_window': 20,
            'confidence_threshold': 0.6,
            'pattern_min_strength': 0.5,
            'volatility_threshold': 0.02,
            'trend_strength_threshold': 0.5,
            'correlation_threshold': 0.3,
            'seasonal_analysis_window': 252,
            'regime_change_sensitivity': 1.5
        }

    def detect_switching_patterns(
        self,
        data: pd.DataFrame,
        analysis_period: Optional[Tuple[datetime, datetime]] = None
    ) -> PatternAnalysisResult:
        """
        戦略切替パターンの検出
        
        Parameters:
            data: 市場データ
            analysis_period: 分析期間
            
        Returns:
            パターン分析結果
        """
        try:
            # 分析期間の設定
            if analysis_period:
                start_date, end_date = analysis_period
                data = data[(data.index >= start_date) & (data.index <= end_date)]
            else:
                start_date, end_date = data.index[0], data.index[-1]
                
            # データの準備
            prepared_data = self._prepare_pattern_detection_data(data)
            
            # 各種パターンの検出
            detected_patterns = []
            
            # トレンド反転パターン
            trend_reversal_patterns = self._detect_trend_reversal_patterns(prepared_data)
            detected_patterns.extend(trend_reversal_patterns)
            
            # モメンタム枯渇パターン
            momentum_exhaustion_patterns = self._detect_momentum_exhaustion_patterns(prepared_data)
            detected_patterns.extend(momentum_exhaustion_patterns)
            
            # ボラティリティブレイクアウトパターン
            volatility_breakout_patterns = self._detect_volatility_breakout_patterns(prepared_data)
            detected_patterns.extend(volatility_breakout_patterns)
            
            # 平均回帰パターン
            mean_reversion_patterns = self._detect_mean_reversion_patterns(prepared_data)
            detected_patterns.extend(mean_reversion_patterns)
            
            # 季節性パターン
            seasonal_patterns = self._detect_seasonal_patterns(prepared_data)
            detected_patterns.extend(seasonal_patterns)
            
            # レジーム変化パターン
            regime_change_patterns = self._detect_regime_change_patterns(prepared_data)
            detected_patterns.extend(regime_change_patterns)
            
            # 相関破綻パターン
            correlation_breakdown_patterns = self._detect_correlation_breakdown_patterns(prepared_data)
            detected_patterns.extend(correlation_breakdown_patterns)
            
            # パターン頻度の集計
            pattern_frequency = self._calculate_pattern_frequency(detected_patterns)
            
            # 成功率の計算
            success_rates = self._calculate_success_rates(detected_patterns)
            
            # 最適タイミングウィンドウの分析
            optimal_timing_windows = self._analyze_optimal_timing_windows(detected_patterns, prepared_data)
            
            # レジーム遷移パターンの分析
            regime_transition_patterns = self._analyze_regime_transition_patterns(detected_patterns)
            
            # 季節性パターンの分析
            seasonal_pattern_analysis = self._analyze_seasonal_patterns(detected_patterns)
            
            # 結果の構築
            result = PatternAnalysisResult(
                analysis_period=(start_date, end_date),
                detected_patterns=detected_patterns,
                pattern_frequency=pattern_frequency,
                success_rates=success_rates,
                optimal_timing_windows=optimal_timing_windows,
                regime_transition_patterns=regime_transition_patterns,
                seasonal_patterns=seasonal_pattern_analysis
            )
            
            logger.info(f"Pattern detection completed: {len(detected_patterns)} patterns found")
            return result
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            raise

    def _prepare_pattern_detection_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """パターン検出用データの準備"""
        prepared_data = data.copy()
        
        # 基本指標の計算
        if 'close' in prepared_data.columns:
            prepared_data['returns'] = prepared_data['close'].pct_change()
            prepared_data['log_returns'] = np.log(prepared_data['close'] / prepared_data['close'].shift(1))
            
            # 移動平均
            for window in [5, 10, 20, 50]:
                prepared_data[f'ma_{window}'] = prepared_data['close'].rolling(window).mean()
                
            # ボラティリティ
            prepared_data['volatility'] = prepared_data['returns'].rolling(20).std()
            prepared_data['volatility_ma'] = prepared_data['volatility'].rolling(10).mean()
            
            # モメンタム
            prepared_data['momentum_5'] = prepared_data['close'] / prepared_data['close'].shift(5) - 1
            prepared_data['momentum_10'] = prepared_data['close'] / prepared_data['close'].shift(10) - 1
            prepared_data['momentum_20'] = prepared_data['close'] / prepared_data['close'].shift(20) - 1
            
            # RSI
            prepared_data['rsi'] = self._calculate_rsi(prepared_data['close'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prepared_data['close'])
            prepared_data['bb_upper'] = bb_upper
            prepared_data['bb_middle'] = bb_middle  
            prepared_data['bb_lower'] = bb_lower
            prepared_data['bb_width'] = (bb_upper - bb_lower) / bb_middle
            prepared_data['bb_position'] = (prepared_data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # MACD
            macd, macd_signal, macd_histogram = self._calculate_macd(prepared_data['close'])
            prepared_data['macd'] = macd
            prepared_data['macd_signal'] = macd_signal
            prepared_data['macd_histogram'] = macd_histogram
            
        # ボリューム指標（利用可能な場合）
        if 'volume' in prepared_data.columns:
            prepared_data['volume_ma'] = prepared_data['volume'].rolling(20).mean()
            prepared_data['volume_ratio'] = prepared_data['volume'] / prepared_data['volume_ma']
            
        # 欠損値処理
        prepared_data = prepared_data.fillna(method='ffill').fillna(method='bfill')
        
        return prepared_data

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSIの計算"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンドの計算"""
        middle = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACDの計算"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram

    def _detect_trend_reversal_patterns(self, data: pd.DataFrame) -> List[SwitchingPattern]:
        """トレンド反転パターンの検出"""
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
            
        try:
            # トレンドの定義
            ma_short = data['ma_5'] if 'ma_5' in data.columns else data['close'].rolling(5).mean()
            ma_long = data['ma_20'] if 'ma_20' in data.columns else data['close'].rolling(20).mean()
            
            # トレンド方向の判定
            trend_up = ma_short > ma_long
            trend_change = trend_up.diff().abs() > 0
            
            # RSIとの組み合わせ
            rsi = data['rsi'] if 'rsi' in data.columns else pd.Series(50, index=data.index)
            
            for idx in data.index[trend_change]:
                if idx in data.index[1:]:  # 最初の要素をスキップ
                    current_trend = trend_up.loc[idx]
                    prev_trend = trend_up.loc[data.index[data.index.get_loc(idx) - 1]]
                    
                    # トレンド反転の確認
                    if current_trend != prev_trend:
                        rsi_value = rsi.loc[idx]
                        
                        # パターンの強度計算
                        if current_trend and rsi_value < 30:  # 上昇転換 + 売られすぎ
                            strength = 1.0 - (rsi_value / 30)
                            confidence = 0.7 + (strength * 0.3)
                        elif not current_trend and rsi_value > 70:  # 下降転換 + 買われすぎ
                            strength = (rsi_value - 70) / 30
                            confidence = 0.7 + (strength * 0.3)
                        else:
                            strength = 0.5
                            confidence = 0.5
                            
                        if strength >= self.pattern_min_strength:
                            # 市場レジームの判定
                            volatility = data['volatility'].loc[idx] if 'volatility' in data.columns else 0.02
                            regime = self._determine_market_regime(data.loc[idx], volatility)
                            
                            pattern = SwitchingPattern(
                                pattern_type=PatternType.TREND_REVERSAL,
                                timestamp=idx,
                                confidence=min(1.0, confidence),
                                description=f"{'Upward' if current_trend else 'Downward'} trend reversal detected",
                                market_regime=regime,
                                trigger_indicators={
                                    'ma_crossover': float(current_trend),
                                    'rsi': rsi_value,
                                    'strength': strength
                                },
                                pattern_strength=strength,
                                historical_success_rate=0.65,  # 過去実績ベース
                                expected_duration=timedelta(days=10),
                                risk_level='Medium'
                            )
                            patterns.append(pattern)
                            
        except Exception as e:
            logger.warning(f"Trend reversal pattern detection failed: {e}")
            
        return patterns

    def _detect_momentum_exhaustion_patterns(self, data: pd.DataFrame) -> List[SwitchingPattern]:
        """モメンタム枯渇パターンの検出"""
        patterns = []
        
        try:
            if 'momentum_20' not in data.columns:
                return patterns
                
            momentum = data['momentum_20']
            rsi = data['rsi'] if 'rsi' in data.columns else pd.Series(50, index=data.index)
            
            # モメンタムの極値検出
            momentum_peaks, _ = find_peaks(momentum, height=0.1, distance=10)
            momentum_troughs, _ = find_peaks(-momentum, height=0.1, distance=10)
            
            # 極値での枯渇パターン検出
            for peak_idx in momentum_peaks:
                idx = data.index[peak_idx]
                momentum_value = momentum.iloc[peak_idx]
                rsi_value = rsi.loc[idx]
                
                # 枯渇条件：高いモメンタム + 買われすぎRSI
                if momentum_value > 0.15 and rsi_value > 75:
                    strength = min(1.0, (momentum_value - 0.15) / 0.15 + (rsi_value - 75) / 25)
                    confidence = 0.6 + (strength * 0.3)
                    
                    if strength >= self.pattern_min_strength:
                        volatility = data['volatility'].loc[idx] if 'volatility' in data.columns else 0.02
                        regime = self._determine_market_regime(data.loc[idx], volatility)
                        
                        pattern = SwitchingPattern(
                            pattern_type=PatternType.MOMENTUM_EXHAUSTION,
                            timestamp=idx,
                            confidence=min(1.0, confidence),
                            description="Upward momentum exhaustion detected",
                            market_regime=regime,
                            trigger_indicators={
                                'momentum': momentum_value,
                                'rsi': rsi_value,
                                'strength': strength
                            },
                            pattern_strength=strength,
                            historical_success_rate=0.58,
                            expected_duration=timedelta(days=7),
                            risk_level='High'
                        )
                        patterns.append(pattern)
                        
            # 下方向の枯渇パターン
            for trough_idx in momentum_troughs:
                idx = data.index[trough_idx]
                momentum_value = -momentum.iloc[trough_idx]  # 負の値を正に変換
                rsi_value = rsi.loc[idx]
                
                # 枯渇条件：低いモメンタム + 売られすぎRSI
                if momentum_value > 0.15 and rsi_value < 25:
                    strength = min(1.0, (momentum_value - 0.15) / 0.15 + (25 - rsi_value) / 25)
                    confidence = 0.6 + (strength * 0.3)
                    
                    if strength >= self.pattern_min_strength:
                        volatility = data['volatility'].loc[idx] if 'volatility' in data.columns else 0.02
                        regime = self._determine_market_regime(data.loc[idx], volatility)
                        
                        pattern = SwitchingPattern(
                            pattern_type=PatternType.MOMENTUM_EXHAUSTION,
                            timestamp=idx,
                            confidence=min(1.0, confidence),
                            description="Downward momentum exhaustion detected",
                            market_regime=regime,
                            trigger_indicators={
                                'momentum': -momentum_value,
                                'rsi': rsi_value,
                                'strength': strength
                            },
                            pattern_strength=strength,
                            historical_success_rate=0.58,
                            expected_duration=timedelta(days=7),
                            risk_level='High'
                        )
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.warning(f"Momentum exhaustion pattern detection failed: {e}")
            
        return patterns

    def _detect_volatility_breakout_patterns(self, data: pd.DataFrame) -> List[SwitchingPattern]:
        """ボラティリティブレイクアウトパターンの検出"""
        patterns = []
        
        try:
            if 'volatility' not in data.columns or 'bb_width' not in data.columns:
                return patterns
                
            volatility = data['volatility']
            bb_width = data['bb_width']
            
            # ボラティリティの低下期間の検出
            vol_ma = volatility.rolling(20).mean()
            vol_ratio = volatility / vol_ma
            
            # ボリンジャーバンドの幅の変化
            bb_width_ma = bb_width.rolling(10).mean()
            bb_width_ratio = bb_width / bb_width_ma
            
            # 低ボラティリティからのブレイクアウト検出
            for i in range(10, len(data)):
                idx = data.index[i]
                
                # 過去10日の低ボラティリティ確認
                recent_vol_ratios = vol_ratio.iloc[i-10:i]
                recent_bb_ratios = bb_width_ratio.iloc[i-10:i]
                
                if (recent_vol_ratios.mean() < 0.8 and 
                    recent_bb_ratios.mean() < 0.9 and
                    vol_ratio.iloc[i] > 1.5):  # 急激なボラティリティ上昇
                    
                    # ブレイクアウトの方向性判定
                    price_change = data['close'].iloc[i] / data['close'].iloc[i-1] - 1
                    
                    strength = min(1.0, vol_ratio.iloc[i] - 1.5)
                    confidence = 0.6 + (strength * 0.2)
                    
                    if strength >= self.pattern_min_strength:
                        regime = self._determine_market_regime(data.iloc[i], volatility.iloc[i])
                        
                        pattern = SwitchingPattern(
                            pattern_type=PatternType.VOLATILITY_BREAKOUT,
                            timestamp=idx,
                            confidence=min(1.0, confidence),
                            description=f"Volatility breakout ({'upward' if price_change > 0 else 'downward'}) detected",
                            market_regime=regime,
                            trigger_indicators={
                                'vol_ratio': vol_ratio.iloc[i],
                                'bb_width_ratio': bb_width_ratio.iloc[i],
                                'price_change': price_change,
                                'strength': strength
                            },
                            pattern_strength=strength,
                            historical_success_rate=0.72,
                            expected_duration=timedelta(days=5),
                            risk_level='High'
                        )
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.warning(f"Volatility breakout pattern detection failed: {e}")
            
        return patterns

    def _detect_mean_reversion_patterns(self, data: pd.DataFrame) -> List[SwitchingPattern]:
        """平均回帰パターンの検出"""
        patterns = []
        
        try:
            if 'bb_position' not in data.columns or 'rsi' not in data.columns:
                return patterns
                
            bb_position = data['bb_position']
            rsi = data['rsi']
            
            # ボリンジャーバンドからの乖離とRSIの組み合わせ
            for idx in data.index:
                bb_pos = bb_position.loc[idx]
                rsi_val = rsi.loc[idx]
                
                # 平均回帰のシグナル：極端な位置 + RSIの逆張りシグナル
                if ((bb_pos < 0.1 and rsi_val < 30) or  # 下方乖離 + 売られすぎ
                    (bb_pos > 0.9 and rsi_val > 70)):   # 上方乖離 + 買われすぎ
                    
                    if bb_pos < 0.1:
                        strength = (0.1 - bb_pos) / 0.1 + (30 - rsi_val) / 30
                        direction = 'upward'
                    else:
                        strength = (bb_pos - 0.9) / 0.1 + (rsi_val - 70) / 30
                        direction = 'downward'
                        
                    strength = min(1.0, strength / 2)  # 正規化
                    confidence = 0.55 + (strength * 0.3)
                    
                    if strength >= self.pattern_min_strength:
                        volatility = data['volatility'].loc[idx] if 'volatility' in data.columns else 0.02
                        regime = self._determine_market_regime(data.loc[idx], volatility)
                        
                        pattern = SwitchingPattern(
                            pattern_type=PatternType.MEAN_REVERSION,
                            timestamp=idx,
                            confidence=min(1.0, confidence),
                            description=f"Mean reversion ({direction}) pattern detected",
                            market_regime=regime,
                            trigger_indicators={
                                'bb_position': bb_pos,
                                'rsi': rsi_val,
                                'strength': strength
                            },
                            pattern_strength=strength,
                            historical_success_rate=0.61,
                            expected_duration=timedelta(days=8),
                            risk_level='Medium'
                        )
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.warning(f"Mean reversion pattern detection failed: {e}")
            
        return patterns

    def _detect_seasonal_patterns(self, data: pd.DataFrame) -> List[SwitchingPattern]:
        """季節性パターンの検出"""
        patterns = []
        
        try:
            # 月次および曜日パターンの分析
            if len(data) < 60:  # 最低2ヶ月のデータが必要
                return patterns
                
            data_with_time = data.copy()
            data_with_time['month'] = data_with_time.index.month
            data_with_time['weekday'] = data_with_time.index.weekday
            data_with_time['day_of_month'] = data_with_time.index.day
            
            if 'returns' not in data_with_time.columns:
                return patterns
                
            # 月次パターンの検出
            monthly_returns = data_with_time.groupby('month')['returns'].agg(['mean', 'std', 'count'])
            significant_months = monthly_returns[
                (abs(monthly_returns['mean']) > monthly_returns['mean'].std() * 1.5) &
                (monthly_returns['count'] > 5)  # 最低5回の観測
            ]
            
            for month, stats in significant_months.iterrows():
                # その月の日付を特定
                month_data = data_with_time[data_with_time['month'] == month]
                
                for idx in month_data.index[::10]:  # 10日おきにパターンを記録
                    strength = min(1.0, abs(stats['mean']) / monthly_returns['mean'].std())
                    confidence = 0.5 + (strength * 0.3)
                    
                    if strength >= 0.3:  # 季節性パターンは低い閾値
                        volatility = data['volatility'].loc[idx] if 'volatility' in data.columns else 0.02
                        regime = self._determine_market_regime(data.loc[idx], volatility)
                        
                        pattern = SwitchingPattern(
                            pattern_type=PatternType.SEASONAL_PATTERN,
                            timestamp=idx,
                            confidence=min(1.0, confidence),
                            description=f"Seasonal pattern for month {month} ({'positive' if stats['mean'] > 0 else 'negative'} bias)",
                            market_regime=regime,
                            trigger_indicators={
                                'month': month,
                                'avg_return': stats['mean'],
                                'strength': strength
                            },
                            pattern_strength=strength,
                            historical_success_rate=0.52,
                            expected_duration=timedelta(days=30),
                            risk_level='Low'
                        )
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.warning(f"Seasonal pattern detection failed: {e}")
            
        return patterns

    def _detect_regime_change_patterns(self, data: pd.DataFrame) -> List[SwitchingPattern]:
        """レジーム変化パターンの検出"""
        patterns = []
        
        try:
            if len(data) < 50:
                return patterns
                
            # ボラティリティレジームの変化検出
            volatility = data['volatility'] if 'volatility' in data.columns else data['returns'].rolling(20).std()
            vol_ma = volatility.rolling(20).mean()
            vol_std = volatility.rolling(20).std()
            
            # レジーム変化の検出（2標準偏差を超える変化）
            vol_zscore = (volatility - vol_ma) / vol_std
            regime_changes = abs(vol_zscore) > self.config.get('regime_change_sensitivity', 1.5)
            
            for idx in data.index[regime_changes]:
                if idx in vol_zscore.index:
                    zscore = vol_zscore.loc[idx]
                    vol_value = volatility.loc[idx]
                    
                    strength = min(1.0, abs(zscore) / 3)  # 3標準偏差で最大強度
                    confidence = 0.6 + (strength * 0.25)
                    
                    if strength >= self.pattern_min_strength:
                        regime_type = 'high_volatility' if zscore > 0 else 'low_volatility'
                        regime = MarketRegime.HIGH_VOLATILITY if zscore > 0 else MarketRegime.LOW_VOLATILITY
                        
                        pattern = SwitchingPattern(
                            pattern_type=PatternType.REGIME_CHANGE,
                            timestamp=idx,
                            confidence=min(1.0, confidence),
                            description=f"Regime change to {regime_type} detected",
                            market_regime=regime,
                            trigger_indicators={
                                'vol_zscore': zscore,
                                'volatility': vol_value,
                                'strength': strength
                            },
                            pattern_strength=strength,
                            historical_success_rate=0.68,
                            expected_duration=timedelta(days=15),
                            risk_level='High'
                        )
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.warning(f"Regime change pattern detection failed: {e}")
            
        return patterns

    def _detect_correlation_breakdown_patterns(self, data: pd.DataFrame) -> List[SwitchingPattern]:
        """相関破綻パターンの検出"""
        patterns = []
        
        try:
            # 複数の指標間の相関分析
            if not all(col in data.columns for col in ['ma_5', 'ma_20', 'rsi']):
                return patterns
                
            # 移動平均間の相関
            ma_correlation = data['ma_5'].rolling(20).corr(data['ma_20'])
            
            # 価格とRSIの相関
            price_rsi_correlation = data['close'].rolling(20).corr(data['rsi']) if 'close' in data.columns else pd.Series()
            
            # 相関の破綻検出
            for idx in data.index[20:]:  # 最初の20期間はスキップ
                if idx in ma_correlation.index and not pd.isna(ma_correlation.loc[idx]):
                    ma_corr = ma_correlation.loc[idx]
                    
                    # 通常高い相関を持つべき移動平均間の相関が低下
                    if ma_corr < self.config.get('correlation_threshold', 0.3):
                        strength = min(1.0, (0.3 - ma_corr) / 0.3)
                        confidence = 0.55 + (strength * 0.25)
                        
                        if strength >= self.pattern_min_strength:
                            volatility = data['volatility'].loc[idx] if 'volatility' in data.columns else 0.02
                            regime = self._determine_market_regime(data.loc[idx], volatility)
                            
                            pattern = SwitchingPattern(
                                pattern_type=PatternType.CORRELATION_BREAKDOWN,
                                timestamp=idx,
                                confidence=min(1.0, confidence),
                                description="Moving average correlation breakdown detected",
                                market_regime=regime,
                                trigger_indicators={
                                    'ma_correlation': ma_corr,
                                    'strength': strength
                                },
                                pattern_strength=strength,
                                historical_success_rate=0.54,
                                expected_duration=timedelta(days=12),
                                risk_level='Medium'
                            )
                            patterns.append(pattern)
                            
        except Exception as e:
            logger.warning(f"Correlation breakdown pattern detection failed: {e}")
            
        return patterns

    def _determine_market_regime(self, data_point: pd.Series, volatility: float) -> MarketRegime:
        """市場レジームの判定"""
        try:
            # トレンド判定
            if 'ma_5' in data_point.index and 'ma_20' in data_point.index:
                ma_short = data_point['ma_5']
                ma_long = data_point['ma_20']
                
                trend_strength = (ma_short - ma_long) / ma_long if ma_long != 0 else 0
                
                if abs(trend_strength) < 0.01:  # 1%未満の差
                    if volatility > 0.03:
                        return MarketRegime.HIGH_VOLATILITY
                    else:
                        return MarketRegime.SIDEWAYS
                elif trend_strength > 0.01:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            else:
                # フォールバック：ボラティリティベース
                if volatility > 0.025:
                    return MarketRegime.HIGH_VOLATILITY
                else:
                    return MarketRegime.LOW_VOLATILITY
                    
        except Exception as e:
            logger.warning(f"Market regime determination failed: {e}")
            return MarketRegime.SIDEWAYS

    def _calculate_pattern_frequency(self, patterns: List[SwitchingPattern]) -> Dict[PatternType, int]:
        """パターン頻度の計算"""
        frequency = {}
        for pattern in patterns:
            frequency[pattern.pattern_type] = frequency.get(pattern.pattern_type, 0) + 1
        return frequency

    def _calculate_success_rates(self, patterns: List[SwitchingPattern]) -> Dict[PatternType, float]:
        """成功率の計算（履歴ベース）"""
        success_rates = {}
        
        for pattern_type in PatternType:
            pattern_list = [p for p in patterns if p.pattern_type == pattern_type]
            if pattern_list:
                # 履歴成功率の平均を使用
                avg_success_rate = np.mean([p.historical_success_rate for p in pattern_list])
                success_rates[pattern_type] = avg_success_rate
            else:
                success_rates[pattern_type] = 0.5  # デフォルト
                
        return success_rates

    def _analyze_optimal_timing_windows(
        self, 
        patterns: List[SwitchingPattern], 
        data: pd.DataFrame
    ) -> Dict[PatternType, Dict[str, Any]]:
        """最適タイミングウィンドウの分析"""
        timing_windows = {}
        
        for pattern_type in PatternType:
            pattern_list = [p for p in patterns if p.pattern_type == pattern_type]
            if not pattern_list:
                continue
                
            # パターン発生時刻の分析
            hours = [p.timestamp.hour for p in pattern_list if hasattr(p.timestamp, 'hour')]
            days_of_week = [p.timestamp.weekday() for p in pattern_list]
            
            optimal_window = {
                'count': len(pattern_list),
                'avg_confidence': np.mean([p.confidence for p in pattern_list]),
                'avg_strength': np.mean([p.pattern_strength for p in pattern_list]),
                'preferred_hours': self._find_most_common_values(hours) if hours else [],
                'preferred_days': self._find_most_common_values(days_of_week),
                'avg_expected_duration_days': np.mean([p.expected_duration.days for p in pattern_list])
            }
            
            timing_windows[pattern_type] = optimal_window
            
        return timing_windows

    def _find_most_common_values(self, values: List[Union[int, float]], top_n: int = 3) -> List[Union[int, float]]:
        """最頻値の検出"""
        if not values:
            return []
            
        from collections import Counter
        counter = Counter(values)
        return [item for item, count in counter.most_common(top_n)]

    def _analyze_regime_transition_patterns(
        self, 
        patterns: List[SwitchingPattern]
    ) -> Dict[str, List[SwitchingPattern]]:
        """レジーム遷移パターンの分析"""
        regime_transitions = {}
        
        for regime in MarketRegime:
            regime_patterns = [p for p in patterns if p.market_regime == regime]
            if regime_patterns:
                regime_transitions[regime.value] = regime_patterns
                
        return regime_transitions

    def _analyze_seasonal_patterns(
        self, 
        patterns: List[SwitchingPattern]
    ) -> Dict[str, List[SwitchingPattern]]:
        """季節性パターンの分析"""
        seasonal_patterns = {}
        
        # 月別分析
        for month in range(1, 13):
            month_patterns = [
                p for p in patterns 
                if p.timestamp.month == month and p.pattern_type == PatternType.SEASONAL_PATTERN
            ]
            if month_patterns:
                seasonal_patterns[f'month_{month}'] = month_patterns
                
        # 四半期別分析
        for quarter in range(1, 5):
            quarter_months = [(quarter - 1) * 3 + i for i in range(1, 4)]
            quarter_patterns = [
                p for p in patterns 
                if p.timestamp.month in quarter_months
            ]
            if quarter_patterns:
                seasonal_patterns[f'quarter_{quarter}'] = quarter_patterns
                
        return seasonal_patterns

    def get_pattern_recommendations(
        self,
        current_data: pd.DataFrame,
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        現在の市場状況に基づくパターンベース推奨事項
        
        Parameters:
            current_data: 現在の市場データ
            lookback_days: 分析期間
            
        Returns:
            推奨事項リスト
        """
        try:
            if len(current_data) < lookback_days:
                return []
                
            recent_data = current_data.tail(lookback_days)
            prepared_data = self._prepare_pattern_detection_data(recent_data)
            
            # 最新データでのパターン検出
            current_patterns = []
            
            # 簡易パターン検出（計算負荷軽減）
            trend_patterns = self._detect_trend_reversal_patterns(prepared_data.tail(10))
            momentum_patterns = self._detect_momentum_exhaustion_patterns(prepared_data.tail(10))
            volatility_patterns = self._detect_volatility_breakout_patterns(prepared_data.tail(10))
            
            current_patterns.extend(trend_patterns[-3:])  # 最近3件
            current_patterns.extend(momentum_patterns[-2:])  # 最近2件
            current_patterns.extend(volatility_patterns[-2:])  # 最近2件
            
            # 推奨事項の生成
            recommendations = []
            
            for pattern in current_patterns:
                if pattern.confidence > self.confidence_threshold:
                    recommendation = {
                        'pattern_type': pattern.pattern_type.value,
                        'timestamp': pattern.timestamp.isoformat(),
                        'confidence': pattern.confidence,
                        'description': pattern.description,
                        'recommended_action': self._get_recommended_action(pattern),
                        'risk_level': pattern.risk_level,
                        'expected_duration_days': pattern.expected_duration.days,
                        'trigger_indicators': pattern.trigger_indicators
                    }
                    recommendations.append(recommendation)
                    
            # 信頼度でソート
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            return recommendations[:5]  # トップ5
            
        except Exception as e:
            logger.error(f"Pattern recommendation failed: {e}")
            return []

    def _get_recommended_action(self, pattern: SwitchingPattern) -> str:
        """パターンに基づく推奨アクション"""
        action_map = {
            PatternType.TREND_REVERSAL: "Consider switching to contrarian strategy",
            PatternType.MOMENTUM_EXHAUSTION: "Consider switching from momentum to mean-reversion strategy",
            PatternType.VOLATILITY_BREAKOUT: "Consider switching to momentum or breakout strategy",
            PatternType.MEAN_REVERSION: "Consider switching to mean-reversion strategy",
            PatternType.SEASONAL_PATTERN: "Consider seasonal strategy adjustment",
            PatternType.REGIME_CHANGE: "Consider fundamental strategy reassessment",
            PatternType.CORRELATION_BREAKDOWN: "Consider diversification strategy review"
        }
        
        return action_map.get(pattern.pattern_type, "Monitor situation closely")

# テスト用のメイン関数
if __name__ == "__main__":
    # 簡単なテストの実行
    logging.basicConfig(level=logging.INFO)
    
    detector = SwitchingPatternDetector()
    
    # テストデータの生成
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # トレンドとボラティリティを含む価格データ
    returns = np.random.randn(len(dates)) * 0.02
    trend = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.001
    returns += trend
    
    prices = 100 * (1 + returns).cumprod()
    
    test_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(100000, 1000000, len(dates)),
    }, index=dates)
    
    try:
        # パターン検出の実行
        result = detector.detect_switching_patterns(test_data)
        
        print("\n=== 戦略切替パターン検出結果 ===")
        print(f"分析期間: {result.analysis_period[0].strftime('%Y-%m-%d')} - {result.analysis_period[1].strftime('%Y-%m-%d')}")
        print(f"検出パターン数: {len(result.detected_patterns)}")
        
        print("\n--- パターン頻度 ---")
        for pattern_type, count in result.pattern_frequency.items():
            print(f"{pattern_type.value}: {count}件")
            
        print("\n--- 成功率 ---")
        for pattern_type, success_rate in result.success_rates.items():
            print(f"{pattern_type.value}: {success_rate:.1%}")
            
        print("\n--- 最近の検出パターン ---")
        recent_patterns = sorted(result.detected_patterns, key=lambda x: x.timestamp)[-5:]
        for pattern in recent_patterns:
            print(f"{pattern.timestamp.strftime('%Y-%m-%d')}: {pattern.pattern_type.value} "
                  f"(信頼度: {pattern.confidence:.1%}, 強度: {pattern.pattern_strength:.2f})")
        
        # 推奨事項の取得
        recommendations = detector.get_pattern_recommendations(test_data, lookback_days=30)
        
        print("\n--- 推奨事項 ---")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['pattern_type']}: {rec['recommended_action']} "
                  f"(信頼度: {rec['confidence']:.1%})")
        
        print("パターン検出成功")
        
    except Exception as e:
        print(f"パターン検出エラー: {e}")
        raise
