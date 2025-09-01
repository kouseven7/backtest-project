"""
DSSMS Phase 3 Task 3.1: Multi-Dimensional Analyzer
多次元分析器

様々な次元からの市場分析を統合し、包括的な評価を提供します。
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import warnings

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 設定とロガー
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

class AnalysisDimension(Enum):
    """分析次元定義"""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    TREND = "trend"
    RELATIVE_STRENGTH = "relative_strength"

class TimeFrame(Enum):
    """時間枠定義"""
    SHORT = "short"      # 5-10日
    MEDIUM = "medium"    # 20-30日
    LONG = "long"        # 60-90日
    ULTRA_LONG = "ultra_long"  # 120-252日

@dataclass
class DimensionalAnalysisConfig:
    """多次元分析設定"""
    enabled_dimensions: List[AnalysisDimension] = field(default_factory=lambda: list(AnalysisDimension))
    time_frames: List[TimeFrame] = field(default_factory=lambda: [TimeFrame.SHORT, TimeFrame.MEDIUM, TimeFrame.LONG])
    normalization_method: str = "minmax"  # "minmax", "zscore", "robust"
    correlation_threshold: float = 0.7
    pca_components: int = 5
    enable_cross_dimensional_analysis: bool = True
    enable_regime_detection: bool = True
    outlier_detection_threshold: float = 2.0

@dataclass
class DimensionalScore:
    """次元スコア"""
    dimension: AnalysisDimension
    time_frame: TimeFrame
    raw_score: float
    normalized_score: float
    confidence: float
    components: Dict[str, float]
    timestamp: datetime

@dataclass
class MultiDimensionalResult:
    """多次元分析結果"""
    symbol: str
    dimensional_scores: Dict[str, DimensionalScore]
    composite_score: float
    pca_score: float
    correlation_matrix: Optional[pd.DataFrame]
    regime_classification: str
    outlier_status: bool
    confidence_level: float
    analysis_timestamp: datetime

@dataclass
class AnalysisConfig:
    """分析設定"""
    enable_momentum_analysis: bool = True
    enable_volatility_analysis: bool = True
    enable_volume_analysis: bool = True
    enable_technical_analysis: bool = True
    enable_fundamental_analysis: bool = True
    lookback_period: int = 252
    log_level: str = "INFO"

class MultiDimensionalAnalyzer:
    """
    多次元分析器
    
    以下の分析次元を統合:
    - モメンタム分析
    - ボラティリティ分析
    - 出来高分析
    - テクニカル分析
    - ファンダメンタル分析（利用可能な場合）
    - センチメント分析（利用可能な場合）
    - トレンド分析
    - 相対強度分析
    """
    
    def __init__(self, config: Optional[DimensionalAnalysisConfig] = None):
        """
        初期化
        
        Args:
            config: 多次元分析設定
        """
        self.config = config or DimensionalAnalysisConfig()
        self.logger = logger
        
        # スケーラー初期化
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=self.config.pca_components)
        
        # 分析キャッシュ
        self._analysis_cache = {}
        
        # 初期化ログ
        self.logger.info(f"Multi-Dimensional Analyzer initialized with {len(self.config.enabled_dimensions)} dimensions")
    
    def analyze_multi_dimensional(
        self, 
        symbol: str, 
        data: pd.DataFrame,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None
    ) -> MultiDimensionalResult:
        """
        多次元分析実行
        
        Args:
            symbol: 銘柄コード
            data: 価格・出来高データ
            market_data: 市場全体データ（相対分析用）
            fundamental_data: ファンダメンタルデータ
            
        Returns:
            多次元分析結果
        """
        start_time = datetime.now()
        self.logger.debug(f"Starting multi-dimensional analysis for {symbol}")
        
        try:
            # 各次元の分析実行
            dimensional_scores = {}
            
            for dimension in self.config.enabled_dimensions:
                for time_frame in self.config.time_frames:
                    try:
                        score = self._analyze_dimension(
                            dimension, time_frame, symbol, data, market_data, fundamental_data
                        )
                        key = f"{dimension.value}_{time_frame.value}"
                        dimensional_scores[key] = score
                    except Exception as e:
                        self.logger.warning(f"Dimension analysis failed for {dimension.value}_{time_frame.value}: {e}")
                        continue
            
            # 複合スコア計算
            composite_score = self._calculate_composite_score(dimensional_scores)
            
            # PCA分析
            pca_score = self._calculate_pca_score(dimensional_scores)
            
            # 相関分析
            correlation_matrix = self._calculate_correlation_matrix(dimensional_scores)
            
            # レジーム分類
            regime_classification = self._classify_regime(dimensional_scores, data)
            
            # 外れ値検出
            outlier_status = self._detect_outlier(dimensional_scores)
            
            # 信頼度計算
            confidence_level = self._calculate_confidence_level(dimensional_scores)
            
            # 結果構築
            result = MultiDimensionalResult(
                symbol=symbol,
                dimensional_scores=dimensional_scores,
                composite_score=composite_score,
                pca_score=pca_score,
                correlation_matrix=correlation_matrix,
                regime_classification=regime_classification,
                outlier_status=outlier_status,
                confidence_level=confidence_level,
                analysis_timestamp=start_time
            )
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"Multi-dimensional analysis completed for {symbol} in {analysis_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-dimensional analysis failed for {symbol}: {e}")
            raise
    
    def _analyze_dimension(
        self, 
        dimension: AnalysisDimension,
        time_frame: TimeFrame,
        symbol: str,
        data: pd.DataFrame,
        market_data: Optional[Dict[str, pd.DataFrame]],
        fundamental_data: Optional[Dict[str, Any]]
    ) -> DimensionalScore:
        """次元別分析実行"""
        
        # 時間枠データ準備
        frame_data = self._prepare_timeframe_data(data, time_frame)
        
        # 次元別分析
        if dimension == AnalysisDimension.MOMENTUM:
            analysis_result = self._analyze_momentum(frame_data, time_frame)
        elif dimension == AnalysisDimension.VOLATILITY:
            analysis_result = self._analyze_volatility(frame_data, time_frame)
        elif dimension == AnalysisDimension.VOLUME:
            analysis_result = self._analyze_volume(frame_data, time_frame)
        elif dimension == AnalysisDimension.TECHNICAL:
            analysis_result = self._analyze_technical(frame_data, time_frame)
        elif dimension == AnalysisDimension.FUNDAMENTAL:
            analysis_result = self._analyze_fundamental(fundamental_data, time_frame)
        elif dimension == AnalysisDimension.SENTIMENT:
            analysis_result = self._analyze_sentiment(frame_data, time_frame)
        elif dimension == AnalysisDimension.TREND:
            analysis_result = self._analyze_trend(frame_data, time_frame)
        elif dimension == AnalysisDimension.RELATIVE_STRENGTH:
            analysis_result = self._analyze_relative_strength(frame_data, market_data, time_frame)
        else:
            raise ValueError(f"Unknown dimension: {dimension}")
        
        # スコア正規化
        normalized_score = self._normalize_score(analysis_result['raw_score'])
        
        # 信頼度計算
        confidence = self._calculate_dimension_confidence(analysis_result, frame_data)
        
        return DimensionalScore(
            dimension=dimension,
            time_frame=time_frame,
            raw_score=analysis_result['raw_score'],
            normalized_score=normalized_score,
            confidence=confidence,
            components=analysis_result['components'],
            timestamp=datetime.now()
        )
    
    def _prepare_timeframe_data(self, data: pd.DataFrame, time_frame: TimeFrame) -> pd.DataFrame:
        """時間枠データ準備"""
        
        frame_lengths = {
            TimeFrame.SHORT: 10,
            TimeFrame.MEDIUM: 30,
            TimeFrame.LONG: 90,
            TimeFrame.ULTRA_LONG: 252
        }
        
        length = frame_lengths[time_frame]
        return data.tail(length).copy()
    
    def _analyze_momentum(self, data: pd.DataFrame, time_frame: TimeFrame) -> Dict[str, Any]:
        """モメンタム分析"""
        try:
            components = {}
            
            # 価格モメンタム
            price_momentum = data['Close'].pct_change().mean()
            components['price_momentum'] = price_momentum
            
            # 加速度
            acceleration = data['Close'].pct_change().diff().mean()
            components['acceleration'] = acceleration
            
            # RSI
            rsi = self._calculate_rsi(data['Close'])
            components['rsi'] = (rsi - 50) / 50  # 正規化
            
            # MACD
            macd_signal = self._calculate_macd_momentum(data['Close'])
            components['macd'] = macd_signal
            
            # 相対モメンタム
            relative_momentum = self._calculate_relative_momentum(data)
            components['relative_momentum'] = relative_momentum
            
            # 複合スコア
            raw_score = np.mean(list(components.values()))
            
            return {
                'raw_score': raw_score,
                'components': components
            }
            
        except Exception as e:
            self.logger.warning(f"Momentum analysis failed: {e}")
            return {'raw_score': 0.0, 'components': {}}
    
    def _analyze_volatility(self, data: pd.DataFrame, time_frame: TimeFrame) -> Dict[str, Any]:
        """ボラティリティ分析"""
        try:
            components = {}
            
            # 実現ボラティリティ
            returns = data['Close'].pct_change()
            realized_vol = returns.std() * np.sqrt(252)
            components['realized_volatility'] = realized_vol
            
            # ボラティリティトレンド
            vol_short = returns.rolling(5).std()
            vol_long = returns.rolling(20).std()
            vol_trend = (vol_short.mean() - vol_long.mean()) / vol_long.mean()
            components['volatility_trend'] = vol_trend
            
            # ATR
            atr = self._calculate_atr(data)
            atr_normalized = atr / data['Close'].mean()
            components['atr_normalized'] = atr_normalized
            
            # VIX風指標
            vix_like = self._calculate_vix_like(data)
            components['vix_like'] = vix_like
            
            # ボラティリティ非対称性
            asymmetry = self._calculate_volatility_asymmetry(returns)
            components['asymmetry'] = asymmetry
            
            # 複合スコア（ボラティリティは反転）
            raw_score = -np.mean([realized_vol, atr_normalized]) + np.mean([vol_trend, asymmetry])
            
            return {
                'raw_score': raw_score,
                'components': components
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility analysis failed: {e}")
            return {'raw_score': 0.0, 'components': {}}
    
    def _analyze_volume(self, data: pd.DataFrame, time_frame: TimeFrame) -> Dict[str, Any]:
        """出来高分析"""
        try:
            components = {}
            
            # 出来高トレンド
            volume_trend = np.polyfit(range(len(data)), data['Volume'], 1)[0]
            components['volume_trend'] = volume_trend / data['Volume'].mean()
            
            # 価格出来高相関
            price_volume_corr = data['Close'].pct_change().corr(data['Volume'].pct_change())
            components['price_volume_correlation'] = price_volume_corr
            
            # 出来高レート
            volume_rate = data['Volume'].mean() / data['Volume'].rolling(len(data)).mean().iloc[-1]
            components['volume_rate'] = volume_rate - 1
            
            # OBV（On Balance Volume）
            obv = self._calculate_obv(data)
            obv_trend = np.polyfit(range(len(obv)), obv, 1)[0]
            components['obv_trend'] = obv_trend / abs(obv.mean()) if obv.mean() != 0 else 0
            
            # VWAP乖離
            vwap_divergence = self._calculate_vwap_divergence(data)
            components['vwap_divergence'] = vwap_divergence
            
            # 複合スコア
            raw_score = np.mean(list(components.values()))
            
            return {
                'raw_score': raw_score,
                'components': components
            }
            
        except Exception as e:
            self.logger.warning(f"Volume analysis failed: {e}")
            return {'raw_score': 0.0, 'components': {}}
    
    def _analyze_technical(self, data: pd.DataFrame, time_frame: TimeFrame) -> Dict[str, Any]:
        """テクニカル分析"""
        try:
            components = {}
            
            # 移動平均
            ma_signal = self._calculate_ma_signals(data)
            components.update(ma_signal)
            
            # ボリンジャーバンド
            bb_signal = self._calculate_bollinger_signals(data)
            components.update(bb_signal)
            
            # ストキャスティクス
            stoch_signal = self._calculate_stochastic_signals(data)
            components.update(stoch_signal)
            
            # 一目均衡表
            ichimoku_signal = self._calculate_ichimoku_signals(data)
            components.update(ichimoku_signal)
            
            # ADX
            adx_signal = self._calculate_adx_signals(data)
            components.update(adx_signal)
            
            # 複合スコア
            raw_score = np.mean(list(components.values()))
            
            return {
                'raw_score': raw_score,
                'components': components
            }
            
        except Exception as e:
            self.logger.warning(f"Technical analysis failed: {e}")
            return {'raw_score': 0.0, 'components': {}}
    
    def _analyze_fundamental(self, fundamental_data: Optional[Dict[str, Any]], time_frame: TimeFrame) -> Dict[str, Any]:
        """ファンダメンタル分析"""
        try:
            if not fundamental_data:
                return {'raw_score': 0.0, 'components': {}}
            
            components = {}
            
            # PER
            if 'per' in fundamental_data:
                per_score = self._normalize_per(fundamental_data['per'])
                components['per_score'] = per_score
            
            # PBR
            if 'pbr' in fundamental_data:
                pbr_score = self._normalize_pbr(fundamental_data['pbr'])
                components['pbr_score'] = pbr_score
            
            # ROE
            if 'roe' in fundamental_data:
                roe_score = self._normalize_roe(fundamental_data['roe'])
                components['roe_score'] = roe_score
            
            # 売上高成長率
            if 'revenue_growth' in fundamental_data:
                growth_score = fundamental_data['revenue_growth'] / 100
                components['revenue_growth_score'] = growth_score
            
            # 複合スコア
            raw_score = np.mean(list(components.values())) if components else 0.0
            
            return {
                'raw_score': raw_score,
                'components': components
            }
            
        except Exception as e:
            self.logger.warning(f"Fundamental analysis failed: {e}")
            return {'raw_score': 0.0, 'components': {}}
    
    def _analyze_sentiment(self, data: pd.DataFrame, time_frame: TimeFrame) -> Dict[str, Any]:
        """センチメント分析（簡易版）"""
        try:
            components = {}
            
            # 価格ギャップ分析
            gaps = self._calculate_price_gaps(data)
            components['gap_sentiment'] = gaps
            
            # 値幅分析
            range_sentiment = self._calculate_range_sentiment(data)
            components['range_sentiment'] = range_sentiment
            
            # 終値位置
            close_position = self._calculate_close_position(data)
            components['close_position'] = close_position
            
            # 複合スコア
            raw_score = np.mean(list(components.values()))
            
            return {
                'raw_score': raw_score,
                'components': components
            }
            
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
            return {'raw_score': 0.0, 'components': {}}
    
    def _analyze_trend(self, data: pd.DataFrame, time_frame: TimeFrame) -> Dict[str, Any]:
        """トレンド分析"""
        try:
            components = {}
            
            # 線形トレンド
            linear_trend = np.polyfit(range(len(data)), data['Close'], 1)[0]
            components['linear_trend'] = linear_trend / data['Close'].mean()
            
            # 指数トレンド
            exp_trend = self._calculate_exponential_trend(data['Close'])
            components['exponential_trend'] = exp_trend
            
            # トレンド強度
            trend_strength = self._calculate_trend_strength(data)
            components['trend_strength'] = trend_strength
            
            # トレンド一貫性
            trend_consistency = self._calculate_trend_consistency(data)
            components['trend_consistency'] = trend_consistency
            
            # 複合スコア
            raw_score = np.mean(list(components.values()))
            
            return {
                'raw_score': raw_score,
                'components': components
            }
            
        except Exception as e:
            self.logger.warning(f"Trend analysis failed: {e}")
            return {'raw_score': 0.0, 'components': {}}
    
    def _analyze_relative_strength(
        self, 
        data: pd.DataFrame, 
        market_data: Optional[Dict[str, pd.DataFrame]], 
        time_frame: TimeFrame
    ) -> Dict[str, Any]:
        """相対強度分析"""
        try:
            components = {}
            
            if market_data:
                # 市場との相対パフォーマンス
                market_returns = self._calculate_market_returns(market_data)
                symbol_returns = data['Close'].pct_change()
                
                relative_performance = symbol_returns.mean() - market_returns.mean()
                components['relative_performance'] = relative_performance
                
                # ベータ
                beta = self._calculate_beta(symbol_returns, market_returns)
                components['beta'] = beta
                
                # アルファ
                alpha = self._calculate_alpha(symbol_returns, market_returns, beta)
                components['alpha'] = alpha
                
                # 相関係数
                correlation = symbol_returns.corr(market_returns)
                components['correlation'] = correlation
            
            # 相対強度指数
            rsi = self._calculate_rsi(data['Close'])
            components['rsi_relative'] = (rsi - 50) / 50
            
            # 複合スコア
            raw_score = np.mean(list(components.values()))
            
            return {
                'raw_score': raw_score,
                'components': components
            }
            
        except Exception as e:
            self.logger.warning(f"Relative strength analysis failed: {e}")
            return {'raw_score': 0.0, 'components': {}}
    
    # 技術指標計算メソッド
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI計算"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return 50.0
    
    def _calculate_macd_momentum(self, prices: pd.Series) -> float:
        """MACDモメンタム計算"""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            return (macd.iloc[-1] - signal.iloc[-1]) / prices.iloc[-1]
        except:
            return 0.0
    
    def _calculate_relative_momentum(self, data: pd.DataFrame) -> float:
        """相対モメンタム計算"""
        try:
            high_low_ratio = (data['High'] - data['Low']) / data['Close']
            return high_low_ratio.mean()
        except:
            return 0.0
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """ATR計算"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.rolling(period).mean().iloc[-1]
        except:
            return 0.0
    
    def _calculate_vix_like(self, data: pd.DataFrame) -> float:
        """VIX風指標計算"""
        try:
            returns = data['Close'].pct_change()
            return returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        except:
            return 0.0
    
    def _calculate_volatility_asymmetry(self, returns: pd.Series) -> float:
        """ボラティリティ非対称性計算"""
        try:
            return returns.skew()
        except:
            return 0.0
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """OBV計算"""
        try:
            obv = (data['Volume'] * np.sign(data['Close'].diff())).cumsum()
            return obv
        except:
            return pd.Series([0] * len(data))
    
    def _calculate_vwap_divergence(self, data: pd.DataFrame) -> float:
        """VWAP乖離計算"""
        try:
            vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            current_price = data['Close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            return (current_price - current_vwap) / current_vwap
        except:
            return 0.0
    
    def _calculate_ma_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """移動平均シグナル計算"""
        try:
            signals = {}
            close = data['Close']
            
            # SMA
            sma5 = close.rolling(5).mean()
            sma25 = close.rolling(25).mean()
            signals['sma_signal'] = (sma5.iloc[-1] - sma25.iloc[-1]) / sma25.iloc[-1]
            
            # EMA
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            signals['ema_signal'] = (ema12.iloc[-1] - ema26.iloc[-1]) / ema26.iloc[-1]
            
            return signals
        except:
            return {}
    
    def _calculate_bollinger_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """ボリンジャーバンドシグナル計算"""
        try:
            close = data['Close']
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            current_price = close.iloc[-1]
            band_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            
            return {'bollinger_position': band_position - 0.5}
        except:
            return {}
    
    def _calculate_stochastic_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """ストキャスティクスシグナル計算"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(3).mean()
            
            return {
                'stoch_k': (k_percent.iloc[-1] - 50) / 50,
                'stoch_d': (d_percent.iloc[-1] - 50) / 50
            }
        except:
            return {}
    
    def _calculate_ichimoku_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """一目均衡表シグナル計算"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # 転換線
            tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
            
            # 基準線
            kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
            
            # 雲
            senkou_a = ((tenkan + kijun) / 2).shift(26)
            senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
            
            current_price = close.iloc[-1]
            signals = {}
            
            if len(tenkan) > 0 and len(kijun) > 0:
                signals['tenkan_kijun'] = (tenkan.iloc[-1] - kijun.iloc[-1]) / kijun.iloc[-1]
            
            return signals
        except:
            return {}
    
    def _calculate_adx_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """ADXシグナル計算"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # TR計算
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # DM計算
            dm_plus = high.diff()
            dm_minus = -low.diff()
            
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            
            # ADX計算（簡易版）
            adx_value = abs(dm_plus.mean() - dm_minus.mean()) / tr.mean()
            
            return {'adx': adx_value}
        except:
            return {}
    
    def _normalize_per(self, per: float) -> float:
        """PER正規化"""
        if per <= 0:
            return -1.0
        elif per <= 15:
            return 1.0
        elif per <= 25:
            return 0.5
        else:
            return -0.5
    
    def _normalize_pbr(self, pbr: float) -> float:
        """PBR正規化"""
        if pbr <= 0:
            return -1.0
        elif pbr <= 1:
            return 1.0
        elif pbr <= 2:
            return 0.5
        else:
            return -0.5
    
    def _normalize_roe(self, roe: float) -> float:
        """ROE正規化"""
        return min(1.0, max(-1.0, roe / 20))
    
    def _calculate_price_gaps(self, data: pd.DataFrame) -> float:
        """価格ギャップ計算"""
        try:
            gaps = data['Open'] - data['Close'].shift()
            return gaps.mean() / data['Close'].mean()
        except:
            return 0.0
    
    def _calculate_range_sentiment(self, data: pd.DataFrame) -> float:
        """値幅センチメント計算"""
        try:
            daily_range = (data['High'] - data['Low']) / data['Close']
            return daily_range.mean()
        except:
            return 0.0
    
    def _calculate_close_position(self, data: pd.DataFrame) -> float:
        """終値位置計算"""
        try:
            close_position = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
            return close_position.mean() - 0.5
        except:
            return 0.0
    
    def _calculate_exponential_trend(self, prices: pd.Series) -> float:
        """指数トレンド計算"""
        try:
            log_prices = np.log(prices)
            trend = np.polyfit(range(len(log_prices)), log_prices, 1)[0]
            return trend
        except:
            return 0.0
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """トレンド強度計算"""
        try:
            prices = data['Close']
            returns = prices.pct_change()
            trend_strength = abs(returns.mean()) / returns.std()
            return trend_strength
        except:
            return 0.0
    
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """トレンド一貫性計算"""
        try:
            prices = data['Close']
            returns = prices.pct_change()
            positive_returns = (returns > 0).sum()
            total_returns = len(returns) - 1
            consistency = (positive_returns / total_returns) - 0.5
            return consistency * 2  # -1 to 1 range
        except:
            return 0.0
    
    def _calculate_market_returns(self, market_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """市場リターン計算"""
        try:
            # 市場インデックス取得（例：TOPIX）
            if 'TOPIX' in market_data:
                return market_data['TOPIX']['Close'].pct_change()
            elif '^N225' in market_data:
                return market_data['^N225']['Close'].pct_change()
            else:
                # フォールバック：全銘柄の平均
                all_returns = []
                for symbol, data in market_data.items():
                    returns = data['Close'].pct_change()
                    all_returns.append(returns)
                return pd.concat(all_returns, axis=1).mean(axis=1)
        except:
            return pd.Series([0.0])
    
    def _calculate_beta(self, symbol_returns: pd.Series, market_returns: pd.Series) -> float:
        """ベータ計算"""
        try:
            covariance = symbol_returns.cov(market_returns)
            market_variance = market_returns.var()
            return covariance / market_variance if market_variance > 0 else 1.0
        except:
            return 1.0
    
    def _calculate_alpha(self, symbol_returns: pd.Series, market_returns: pd.Series, beta: float) -> float:
        """アルファ計算"""
        try:
            symbol_mean = symbol_returns.mean()
            market_mean = market_returns.mean()
            return symbol_mean - (beta * market_mean)
        except:
            return 0.0
    
    def _normalize_score(self, score: float) -> float:
        """スコア正規化"""
        return max(-1.0, min(1.0, score))
    
    def _calculate_dimension_confidence(self, analysis_result: Dict[str, Any], data: pd.DataFrame) -> float:
        """次元信頼度計算"""
        try:
            components = analysis_result.get('components', {})
            if not components:
                return 0.0
            
            # コンポーネントの一貫性チェック
            component_values = list(components.values())
            consistency = 1 - np.std(component_values)
            
            # データ品質チェック
            data_quality = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            
            return max(0.0, min(1.0, (consistency + data_quality) / 2))
        except:
            return 0.5
    
    def _calculate_composite_score(self, dimensional_scores: Dict[str, DimensionalScore]) -> float:
        """複合スコア計算"""
        try:
            if not dimensional_scores:
                return 0.0
            
            total_score = 0.0
            total_weight = 0.0
            
            for score in dimensional_scores.values():
                weight = score.confidence
                total_score += score.normalized_score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_pca_score(self, dimensional_scores: Dict[str, DimensionalScore]) -> float:
        """PCA分析スコア計算"""
        try:
            if not dimensional_scores:
                return 0.0
            
            # スコア行列準備
            scores_matrix = np.array([[score.normalized_score] for score in dimensional_scores.values()])
            
            if len(scores_matrix) < 2:
                return dimensional_scores[list(dimensional_scores.keys())[0]].normalized_score
            
            # PCA実行
            pca_result = PCA(n_components=1).fit_transform(scores_matrix.reshape(1, -1))
            
            return float(pca_result[0][0])
        except:
            return 0.0
    
    def _calculate_correlation_matrix(self, dimensional_scores: Dict[str, DimensionalScore]) -> Optional[pd.DataFrame]:
        """相関行列計算"""
        try:
            if len(dimensional_scores) < 2:
                return None
            
            score_data = {key: [score.normalized_score] for key, score in dimensional_scores.items()}
            df = pd.DataFrame(score_data)
            
            return df.corr()
        except:
            return None
    
    def _classify_regime(self, dimensional_scores: Dict[str, DimensionalScore], data: pd.DataFrame) -> str:
        """レジーム分類"""
        try:
            # モメンタムスコア
            momentum_scores = [score for key, score in dimensional_scores.items() if 'momentum' in key]
            avg_momentum = np.mean([s.normalized_score for s in momentum_scores]) if momentum_scores else 0
            
            # ボラティリティスコア
            volatility_scores = [score for key, score in dimensional_scores.items() if 'volatility' in key]
            avg_volatility = np.mean([s.normalized_score for s in volatility_scores]) if volatility_scores else 0
            
            # レジーム分類
            if avg_momentum > 0.3 and avg_volatility < 0.3:
                return "bullish_stable"
            elif avg_momentum > 0.3 and avg_volatility > 0.3:
                return "bullish_volatile"
            elif avg_momentum < -0.3 and avg_volatility < 0.3:
                return "bearish_stable"
            elif avg_momentum < -0.3 and avg_volatility > 0.3:
                return "bearish_volatile"
            else:
                return "neutral"
                
        except:
            return "unknown"
    
    def _detect_outlier(self, dimensional_scores: Dict[str, DimensionalScore]) -> bool:
        """外れ値検出"""
        try:
            scores = [score.normalized_score for score in dimensional_scores.values()]
            if len(scores) < 3:
                return False
            
            z_scores = np.abs(stats.zscore(scores))
            return any(z > self.config.outlier_detection_threshold for z in z_scores)
        except:
            return False
    
    def _calculate_confidence_level(self, dimensional_scores: Dict[str, DimensionalScore]) -> float:
        """全体信頼度計算"""
        try:
            if not dimensional_scores:
                return 0.0
            
            confidences = [score.confidence for score in dimensional_scores.values()]
            return np.mean(confidences)
        except:
            return 0.0
