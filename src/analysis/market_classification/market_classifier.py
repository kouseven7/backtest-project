"""
市場分類器クラス
A→B段階的市場分類システムのメイン分類エンジン
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

from .market_conditions import (
    SimpleMarketCondition, DetailedMarketCondition, 
    MarketMetrics, ClassificationResult, MarketConditions
)

logger = logging.getLogger(__name__)


class MarketClassifier:
    """市場分類器メインクラス"""
    
    def __init__(self, lookback_periods: int = 20, 
                 volatility_threshold: float = 0.02,
                 trend_threshold: float = 0.001,
                 confidence_threshold: float = 0.6):
        """
        Args:
            lookback_periods: 分析期間（日数）
            volatility_threshold: ボラティリティ閾値
            trend_threshold: トレンド閾値
            confidence_threshold: 分類信頼度閾値
        """
        self.lookback_periods = lookback_periods
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.confidence_threshold = confidence_threshold
        
        # 分類履歴
        self.classification_history: List[ClassificationResult] = []
        
    def calculate_market_metrics(self, data: pd.DataFrame, 
                               symbol: str = "Unknown") -> MarketMetrics:
        """市場メトリクスを計算"""
        try:
            # 基本価格データの確認
            required_cols = ['Close', 'High', 'Low', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            if len(data) < self.lookback_periods:
                logger.warning(f"Insufficient data for {symbol}: {len(data)} < {self.lookback_periods}")
                
            # 最新のlookback_periods分のデータを使用
            recent_data = data.tail(self.lookback_periods).copy()
            
            # 1. トレンド強度計算
            close_prices = recent_data['Close'].values
            trend_strength = self._calculate_trend_strength(close_prices)
            
            # 2. ボラティリティ計算
            returns = np.diff(np.log(close_prices))
            volatility = np.std(returns) if len(returns) > 1 else 0.0
            
            # 3. モメンタム計算
            momentum = self._calculate_momentum(close_prices)
            
            # 4. 出来高トレンド
            volume_trend = self._calculate_volume_trend(recent_data['Volume'].values)
            
            # 5. 価格モメンタム
            price_momentum = self._calculate_price_momentum(close_prices)
            
            # 6. リスクレベル
            risk_level = self._calculate_risk_level(volatility, close_prices)
            
            # 7. 追加メトリクス
            rsi = self._calculate_rsi(close_prices)
            ma_slope = self._calculate_ma_slope(close_prices)
            atr_ratio = self._calculate_atr_ratio(recent_data)
            volume_ratio = self._calculate_volume_ratio(recent_data['Volume'].values)
            
            return MarketMetrics(
                trend_strength=trend_strength,
                volatility=volatility,
                momentum=momentum,
                volume_trend=volume_trend,
                price_momentum=price_momentum,
                risk_level=risk_level,
                rsi=rsi,
                ma_slope=ma_slope,
                atr_ratio=atr_ratio,
                volume_ratio=volume_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating market metrics for {symbol}: {e}")
            # デフォルト値を返す
            return MarketMetrics(
                trend_strength=0.0,
                volatility=0.02,
                momentum=0.0,
                volume_trend=0.0,
                price_momentum=0.0,
                risk_level=0.5
            )
    
    def classify_market_simple(self, metrics: MarketMetrics) -> Tuple[SimpleMarketCondition, float]:
        """シンプル市場分類"""
        confidence_scores = {}
        
        # 1. TRENDING_BULL判定
        bull_score = 0.0
        if metrics.trend_strength > 0.01:
            bull_score += 0.4
        if metrics.momentum > 0.02:
            bull_score += 0.3
        if metrics.price_momentum > 0.01:
            bull_score += 0.3
        confidence_scores[SimpleMarketCondition.TRENDING_BULL] = bull_score
        
        # 2. TRENDING_BEAR判定
        bear_score = 0.0
        if metrics.trend_strength < -0.01:
            bear_score += 0.4
        if metrics.momentum < -0.02:
            bear_score += 0.3
        if metrics.price_momentum < -0.01:
            bear_score += 0.3
        confidence_scores[SimpleMarketCondition.TRENDING_BEAR] = bear_score
        
        # 3. SIDEWAYS判定
        sideways_score = 0.0
        if abs(metrics.trend_strength) <= 0.005:
            sideways_score += 0.5
        if metrics.volatility < self.volatility_threshold:
            sideways_score += 0.3
        if abs(metrics.momentum) < 0.01:
            sideways_score += 0.2
        confidence_scores[SimpleMarketCondition.SIDEWAYS] = sideways_score
        
        # 4. VOLATILE判定
        volatile_score = 0.0
        if metrics.volatility > self.volatility_threshold * 1.5:
            volatile_score += 0.6
        if metrics.risk_level > 0.6:
            volatile_score += 0.4
        confidence_scores[SimpleMarketCondition.VOLATILE] = volatile_score
        
        # 5. RECOVERY判定
        recovery_score = 0.0
        if metrics.trend_strength > 0.005 and metrics.trend_strength < 0.02:
            recovery_score += 0.4
        if metrics.volume_trend > 0.1:
            recovery_score += 0.3
        if metrics.momentum > 0.01 and metrics.volatility > self.volatility_threshold:
            recovery_score += 0.3
        confidence_scores[SimpleMarketCondition.RECOVERY] = recovery_score
        
        # 最高スコアの分類を選択
        best_condition = max(confidence_scores, key=confidence_scores.get)
        best_confidence = confidence_scores[best_condition]
        
        return best_condition, best_confidence
    
    def classify_market_detailed(self, metrics: MarketMetrics) -> Tuple[DetailedMarketCondition, float]:
        """詳細市場分類"""
        confidence_scores = {}
        
        # 1. STRONG_BULL
        if metrics.trend_strength > 0.02 and metrics.momentum > 0.03:
            confidence_scores[DetailedMarketCondition.STRONG_BULL] = min(
                0.8 + metrics.trend_strength * 10, 1.0
            )
        else:
            confidence_scores[DetailedMarketCondition.STRONG_BULL] = 0.0
            
        # 2. MODERATE_BULL
        if 0.005 < metrics.trend_strength <= 0.02 and metrics.momentum > 0.01:
            confidence_scores[DetailedMarketCondition.MODERATE_BULL] = 0.6 + abs(metrics.momentum) * 5
        else:
            confidence_scores[DetailedMarketCondition.MODERATE_BULL] = 0.0
            
        # 3. SIDEWAYS_BULL
        if -0.005 <= metrics.trend_strength <= 0.01 and metrics.price_momentum > 0:
            confidence_scores[DetailedMarketCondition.SIDEWAYS_BULL] = 0.5 + metrics.price_momentum * 10
        else:
            confidence_scores[DetailedMarketCondition.SIDEWAYS_BULL] = 0.0
            
        # 4. NEUTRAL_SIDEWAYS
        if abs(metrics.trend_strength) <= 0.005 and abs(metrics.momentum) <= 0.01:
            confidence_scores[DetailedMarketCondition.NEUTRAL_SIDEWAYS] = 0.7 - abs(metrics.trend_strength) * 20
        else:
            confidence_scores[DetailedMarketCondition.NEUTRAL_SIDEWAYS] = 0.0
            
        # 5. SIDEWAYS_BEAR
        if -0.01 <= metrics.trend_strength <= 0.005 and metrics.price_momentum < 0:
            confidence_scores[DetailedMarketCondition.SIDEWAYS_BEAR] = 0.5 + abs(metrics.price_momentum) * 10
        else:
            confidence_scores[DetailedMarketCondition.SIDEWAYS_BEAR] = 0.0
            
        # 6. MODERATE_BEAR
        if -0.02 <= metrics.trend_strength < -0.005 and metrics.momentum < -0.01:
            confidence_scores[DetailedMarketCondition.MODERATE_BEAR] = 0.6 + abs(metrics.momentum) * 5
        else:
            confidence_scores[DetailedMarketCondition.MODERATE_BEAR] = 0.0
            
        # 7. STRONG_BEAR
        if metrics.trend_strength < -0.02 and metrics.momentum < -0.03:
            confidence_scores[DetailedMarketCondition.STRONG_BEAR] = min(
                0.8 + abs(metrics.trend_strength) * 10, 1.0
            )
        else:
            confidence_scores[DetailedMarketCondition.STRONG_BEAR] = 0.0
        
        # 最高スコアの分類を選択
        best_condition = max(confidence_scores, key=confidence_scores.get)
        best_confidence = confidence_scores[best_condition]
        
        return best_condition, best_confidence
    
    def classify(self, data: pd.DataFrame, symbol: str = "Unknown", 
                mode: str = "hybrid") -> ClassificationResult:
        """
        市場分類の実行
        
        Args:
            data: 価格データ
            symbol: シンボル名
            mode: 分類モード ("simple", "detailed", "hybrid")
        """
        try:
            # メトリクス計算
            metrics = self.calculate_market_metrics(data, symbol)
            
            # 分類実行
            if mode == "simple":
                simple_condition, simple_confidence = self.classify_market_simple(metrics)
                # 詳細分類はシンプルから推定
                possible_detailed = MarketConditions.get_possible_detailed_from_simple(simple_condition)
                detailed_condition = possible_detailed[0]  # とりあえず最初の候補
                confidence = simple_confidence
            
            elif mode == "detailed":
                detailed_condition, detailed_confidence = self.classify_market_detailed(metrics)
                simple_condition = MarketConditions.get_simple_from_detailed(detailed_condition)
                confidence = detailed_confidence
                
            else:  # hybrid
                simple_condition, simple_confidence = self.classify_market_simple(metrics)
                detailed_condition, detailed_confidence = self.classify_market_detailed(metrics)
                
                # 互換性チェック
                if MarketConditions.is_compatible(simple_condition, detailed_condition):
                    confidence = (simple_confidence + detailed_confidence) / 2
                else:
                    # 互換性がない場合はシンプル分類を優先
                    possible_detailed = MarketConditions.get_possible_detailed_from_simple(simple_condition)
                    detailed_condition = possible_detailed[0]
                    confidence = simple_confidence * 0.8  # 信頼度を少し下げる
            
            # 分類結果の作成
            result = ClassificationResult(
                simple_condition=simple_condition,
                detailed_condition=detailed_condition,
                confidence=confidence,
                metrics=metrics,
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                classification_reason={
                    'mode': mode,
                    'trend_strength': metrics.trend_strength,
                    'volatility': metrics.volatility,
                    'momentum': metrics.momentum,
                    'confidence_threshold': self.confidence_threshold
                }
            )
            
            # 履歴に追加
            self.classification_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Classification error for {symbol}: {e}")
            # デフォルト結果を返す
            return ClassificationResult(
                simple_condition=SimpleMarketCondition.SIDEWAYS,
                detailed_condition=DetailedMarketCondition.NEUTRAL_SIDEWAYS,
                confidence=0.1,
                metrics=MarketMetrics(0.0, 0.02, 0.0, 0.0, 0.0, 0.5),
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                classification_reason={'error': str(e)}
            )
    
    # ヘルパーメソッド群
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """トレンド強度計算"""
        if len(prices) < 2:
            return 0.0
        
        # 線形回帰でトレンドを計算
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # 価格で正規化
        normalized_slope = slope / np.mean(prices) if np.mean(prices) != 0 else 0.0
        
        return float(normalized_slope)
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """モメンタム計算"""
        if len(prices) < 10:
            return 0.0
        
        # 短期と長期の平均を比較
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-10:])
        
        return float((short_ma - long_ma) / long_ma) if long_ma != 0 else 0.0
    
    def _calculate_volume_trend(self, volumes: np.ndarray) -> float:
        """出来高トレンド計算"""
        if len(volumes) < 2:
            return 0.0
        
        # 最近の出来高変化
        recent_avg = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
        past_avg = np.mean(volumes[:-5]) if len(volumes) > 5 else volumes[0]
        
        return float((recent_avg - past_avg) / past_avg) if past_avg != 0 else 0.0
    
    def _calculate_price_momentum(self, prices: np.ndarray) -> float:
        """価格モメンタム計算"""
        if len(prices) < 2:
            return 0.0
        
        return float((prices[-1] - prices[0]) / prices[0]) if prices[0] != 0 else 0.0
    
    def _calculate_risk_level(self, volatility: float, prices: np.ndarray) -> float:
        """リスクレベル計算"""
        # ボラティリティベースのリスク計算
        volatility_risk = min(volatility / self.volatility_threshold, 1.0)
        
        # 価格変動リスク
        if len(prices) > 1:
            price_changes = np.abs(np.diff(prices) / prices[:-1])
            price_risk = min(np.mean(price_changes) / 0.02, 1.0)
        else:
            price_risk = 0.5
        
        return float((volatility_risk + price_risk) / 2)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> Optional[float]:
        """RSI計算"""
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_ma_slope(self, prices: np.ndarray, period: int = 10) -> Optional[float]:
        """移動平均の傾き計算"""
        if len(prices) < period:
            return None
        
        ma = np.convolve(prices, np.ones(period)/period, mode='valid')
        if len(ma) < 2:
            return None
        
        return float((ma[-1] - ma[-2]) / ma[-2]) if ma[-2] != 0 else 0.0
    
    def _calculate_atr_ratio(self, data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """ATR比率計算"""
        if len(data) < period:
            return None
        
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift(1))
        low_close = np.abs(data['Low'] - data['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        current_price = data['Close'].iloc[-1]
        
        return float(atr / current_price) if current_price != 0 else None
    
    def _calculate_volume_ratio(self, volumes: np.ndarray, period: int = 20) -> Optional[float]:
        """出来高比率計算"""
        if len(volumes) < period:
            return None
        
        avg_volume = np.mean(volumes[-period:])
        current_volume = volumes[-1]
        
        return float(current_volume / avg_volume) if avg_volume != 0 else None
