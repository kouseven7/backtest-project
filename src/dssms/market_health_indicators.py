"""
DSSMS Market Health Indicators System
市場ヘルスチェック指標計算システム

日経225指数ベースの市場健全性評価
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

class MarketHealthIndicators:
    """市場ヘルスチェック指標計算システム"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 市場監視設定
        """
        self.logger = setup_logger('dssms.market_health')
        self.config = config
        self.weights = config.get("health_scoring", {}).get("weights", {})
        self.thresholds = config.get("health_scoring", {}).get("thresholds", {})
        
        self.logger.info("MarketHealthIndicators initialized")
    
    def calculate_perfect_order_health(self, nikkei_data: pd.DataFrame) -> float:
        """
        パーフェクトオーダー健全性スコア計算
        
        Args:
            nikkei_data: 日経225データ
            
        Returns:
            0.0-1.0のスコア
        """
        try:
            if len(nikkei_data) < 75:  # 最低限のデータ数チェック
                return 0.5
            
            # SBI証券準拠の移動平均
            timeframes = self.config.get("nikkei225_analysis", {}).get("timeframes", {})
            daily_periods = timeframes.get("daily", {"short": 5, "medium": 25, "long": 75})
            
            short_ma = nikkei_data['Close'].rolling(window=daily_periods["short"]).mean()
            medium_ma = nikkei_data['Close'].rolling(window=daily_periods["medium"]).mean()
            long_ma = nikkei_data['Close'].rolling(window=daily_periods["long"]).mean()
            
            current_price = nikkei_data['Close'].iloc[-1]
            current_short = short_ma.iloc[-1]
            current_medium = medium_ma.iloc[-1]
            current_long = long_ma.iloc[-1]
            
            # パーフェクトオーダー判定
            is_perfect_order = (
                current_price > current_short and
                current_short > current_medium and
                current_medium > current_long
            )
            
            if is_perfect_order:
                # パーフェクトオーダーの強度計算
                price_above_short = (current_price - current_short) / current_short
                short_above_medium = (current_short - current_medium) / current_medium
                medium_above_long = (current_medium - current_long) / current_long
                
                # 強度スコア（各間隔の比率を評価）
                strength = float(np.mean([
                    min(price_above_short * 100, 1.0),
                    min(short_above_medium * 100, 1.0),
                    min(medium_above_long * 100, 1.0)
                ]))
                
                return min(0.7 + strength * 0.3, 1.0)  # 0.7-1.0の範囲
            else:
                # パーフェクトオーダーでない場合は距離に基づく減点
                return 0.3
                
        except Exception as e:
            self.logger.error(f"Perfect order health calculation error: {e}")
            return 0.5
    
    def calculate_volatility_health(self, nikkei_data: pd.DataFrame) -> float:
        """
        ボラティリティ健全性スコア計算
        
        Args:
            nikkei_data: 日経225データ
            
        Returns:
            0.0-1.0のスコア（適正範囲内で高スコア）
        """
        try:
            if len(nikkei_data) < 20:
                return 0.5
            
            # 20日ボラティリティ計算
            returns = nikkei_data['Close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
            
            vol_thresholds = self.thresholds.get("volatility", {"low": 0.01, "high": 0.03})
            low_threshold = vol_thresholds["low"]
            high_threshold = vol_thresholds["high"]
            
            if low_threshold <= volatility <= high_threshold:
                # 適正範囲内
                return 1.0
            elif volatility < low_threshold:
                # 低すぎるボラティリティ（流動性不足リスク）
                return max(0.3, volatility / low_threshold)
            else:
                # 高すぎるボラティリティ（リスク増大）
                excess = volatility - high_threshold
                penalty = min(excess / high_threshold, 0.7)  # 最大70%減点
                return max(0.1, 1.0 - penalty)
                
        except Exception as e:
            self.logger.error(f"Volatility health calculation error: {e}")
            return 0.5
    
    def calculate_volume_health(self, nikkei_data: pd.DataFrame) -> float:
        """
        出来高健全性スコア計算
        
        Args:
            nikkei_data: 日経225データ
            
        Returns:
            0.0-1.0のスコア（流動性確保で高スコア）
        """
        try:
            if len(nikkei_data) < 20:
                return 0.5
            
            # 出来高変化率分析
            recent_volume = nikkei_data['Volume'].iloc[-5:].mean()  # 直近5日平均
            historical_volume = nikkei_data['Volume'].iloc[-20:-5].mean()  # 過去15日平均
            
            if historical_volume == 0:
                return 0.5
            
            volume_ratio = recent_volume / historical_volume
            
            vol_change_thresholds = self.thresholds.get("volume_change", {
                "significant_increase": 1.5,
                "significant_decrease": 0.7
            })
            
            increase_threshold = vol_change_thresholds["significant_increase"]
            decrease_threshold = vol_change_thresholds["significant_decrease"]
            
            if decrease_threshold <= volume_ratio <= increase_threshold:
                # 安定した出来高
                return 1.0
            elif volume_ratio < decrease_threshold:
                # 出来高減少（流動性懸念）
                return max(0.2, volume_ratio / decrease_threshold)
            else:
                # 出来高急増（過熱感）
                excess = volume_ratio - increase_threshold
                penalty = min(excess / increase_threshold, 0.5)  # 最大50%減点
                return max(0.5, 1.0 - penalty)
                
        except Exception as e:
            self.logger.error(f"Volume health calculation error: {e}")
            return 0.5
    
    def calculate_trend_strength_health(self, nikkei_data: pd.DataFrame) -> float:
        """
        トレンド強度健全性スコア計算
        
        Args:
            nikkei_data: 日経225データ
            
        Returns:
            0.0-1.0のスコア（強いトレンドで高スコア）
        """
        try:
            if len(nikkei_data) < 30:
                return 0.5
            
            # ADX風の計算（簡易版）
            high = nikkei_data['High']
            low = nikkei_data['Low']
            close = nikkei_data['Close']
            
            # True Range計算
            hl = high - low
            hc = abs(high - close.shift(1))
            lc = abs(low - close.shift(1))
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            
            # Directional Movement計算
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # 14期間平均
            period = 14
            plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / tr.rolling(window=period).mean()
            minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / tr.rolling(window=period).mean()
            
            # ADX計算
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean().iloc[-1]
            
            # ADXベースのスコア
            if adx > 25:  # 強いトレンド
                return min(1.0, adx / 50)  # 最大1.0
            elif adx > 15:  # 中程度のトレンド
                return 0.4 + (adx - 15) / 10 * 0.3  # 0.4-0.7
            else:  # 弱いトレンド
                return max(0.1, adx / 15 * 0.4)  # 0.1-0.4
                
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            # RSIベースのフォールバック
            try:
                rsi_period = self.config.get("nikkei225_analysis", {}).get("trend_strength_indicators", {}).get("rsi_period", 14)
                close_prices = nikkei_data['Close'].astype(float)
                delta = close_prices.diff()
                gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = float(rsi.iloc[-1])
                
                # RSIから強度を評価
                if 30 <= current_rsi <= 70:
                    return 0.8  # 健全な範囲
                else:
                    return 0.4  # 過買い・過売り
            except:
                return 0.5
    
    def get_composite_health_score(self, nikkei_data: pd.DataFrame) -> Dict[str, float]:
        """
        総合ヘルススコア計算
        
        Args:
            nikkei_data: 日経225データ
            
        Returns:
            各指標スコアと総合スコアの辞書
        """
        try:
            # 各指標計算
            perfect_order_score = self.calculate_perfect_order_health(nikkei_data)
            volatility_score = self.calculate_volatility_health(nikkei_data)
            volume_score = self.calculate_volume_health(nikkei_data)
            trend_strength_score = self.calculate_trend_strength_health(nikkei_data)
            
            # 重み付き総合スコア
            composite_score = (
                perfect_order_score * self.weights.get("perfect_order_status", 0.40) +
                volatility_score * self.weights.get("volatility_level", 0.25) +
                volume_score * self.weights.get("volume_profile", 0.20) +
                trend_strength_score * self.weights.get("trend_strength", 0.15)
            )
            
            result: Dict[str, float] = {
                "perfect_order": perfect_order_score,
                "volatility": volatility_score,
                "volume": volume_score,
                "trend_strength": trend_strength_score,
                "composite": composite_score
            }
            
            self.logger.debug(f"Health scores: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Composite health score calculation error: {e}")
            return {
                "perfect_order": 0.5,
                "volatility": 0.5,
                "volume": 0.5,
                "trend_strength": 0.5,
                "composite": 0.5
            }
