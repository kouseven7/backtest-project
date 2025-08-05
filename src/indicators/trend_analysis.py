"""
Module: Trend Analysis (Enhanced)
File: trend_analysis.py
Description: 
  トレンド分析を行うためのモジュールです。
  SMAやATRを用いてトレンドやボラティリティを判定します。
  精度検証機能と信頼度スコア付きの判定機能を提供します。

Author: imega
Created: 2023-04-01
Modified: 2025-07-03

Dependencies:
  - pandas
  - numpy
  - basic_indicators
  - bollinger_atr
  - momentum_indicators
"""

import pandas as pd
import numpy as np
from typing import Literal, Optional, Dict, Any, Tuple
from .basic_indicators import calculate_sma
from .bollinger_atr import calculate_atr

# MACD計算のためのインポート追加
try:
    from .momentum_indicators import calculate_macd
except ImportError:
    # momentum_indicatorsがない場合の代替実装
    def calculate_macd(data: pd.DataFrame, price_column: str, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
        short_ema = data[price_column].ewm(span=short_window, adjust=False).mean()
        long_ema = data[price_column].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal_line

def detect_trend(data: pd.DataFrame, price_column: str, lookback_period: int = 10, 
                short_period: int = 10, medium_period: int = 20, long_period: int = 50, 
                up_score: int = 4, volatility_threshold: float = 0.02) -> str:
    """
    改善されたトレンド判定関数。レンジ相場の検出精度を向上させます。
    VWAP Bounce戦略のために特化した判定ロジックを含みます。
    """
    # データのコピーを作成して元のデータを変更しないようにする
    data_copy = data.copy()
    
    # 短期、中期、長期のSMAを計算
    data_copy.loc[:, 'SMA_short'] = calculate_sma(data_copy, price_column, short_period)
    data_copy.loc[:, 'SMA_medium'] = calculate_sma(data_copy, price_column, medium_period)
    data_copy.loc[:, 'SMA_long'] = calculate_sma(data_copy, price_column, long_period)
    
    # 十分なデータがない場合はレンジ相場と判定
    if len(data_copy) < max(lookback_period, long_period):
        return "range-bound"
    
    # 直近の値を取得
    latest_data = data_copy.iloc[-lookback_period:]
    current_price = latest_data[price_column].iloc[-1]
    
    # SMAの傾き（方向性）を計算
    short_slope = (latest_data['SMA_short'].iloc[-1] - latest_data['SMA_short'].iloc[0]) / lookback_period
    medium_slope = (latest_data['SMA_medium'].iloc[-1] - latest_data['SMA_medium'].iloc[0]) / lookback_period
    long_slope = (latest_data['SMA_long'].iloc[-1] - latest_data['SMA_long'].iloc[0]) / lookback_period
    
    # 価格のボラティリティを計算（レンジ相場判定用）
    price_volatility = latest_data[price_column].std() / latest_data[price_column].mean()
    
    # レンジ相場の判定を強化
    sma_convergence = abs(latest_data['SMA_short'].iloc[-1] - latest_data['SMA_long'].iloc[-1]) / latest_data['SMA_long'].iloc[-1]
    
    # レンジ相場の条件：
    # 1. SMAが収束している（短期と長期の差が小さい）
    # 2. 傾きが小さい
    # 3. ボラティリティが低い
    if (sma_convergence < 0.05 and 
        abs(short_slope) < volatility_threshold and 
        abs(medium_slope) < volatility_threshold and
        price_volatility < volatility_threshold):
        return "range-bound"
    
    # トレンド判定のスコアリングシステム（閾値を緩和）
    uptrend_score = 0
    downtrend_score = 0
    
    # 位置関係のスコア
    if latest_data['SMA_short'].iloc[-1] > latest_data['SMA_medium'].iloc[-1]:
        uptrend_score += 1
    else:
        downtrend_score += 1
        
    if latest_data['SMA_medium'].iloc[-1] > latest_data['SMA_long'].iloc[-1]:
        uptrend_score += 1
    else:
        downtrend_score += 1
        
    if current_price > latest_data['SMA_short'].iloc[-1]:
        uptrend_score += 1
    else:
        downtrend_score += 1
    
    # 傾きのスコア（閾値を設定）
    slope_threshold = volatility_threshold / 2
    if short_slope > slope_threshold:
        uptrend_score += 1
    elif short_slope < -slope_threshold:
        downtrend_score += 1
        
    if medium_slope > slope_threshold:
        uptrend_score += 1
    elif medium_slope < -slope_threshold:
        downtrend_score += 1
        
    if long_slope > slope_threshold:
        uptrend_score += 1
    elif long_slope < -slope_threshold:
        downtrend_score += 1
    
    # スコアに基づくトレンド判定（より厳格に）
    if uptrend_score >= up_score and uptrend_score > downtrend_score:
        return "uptrend"
    elif downtrend_score >= up_score and downtrend_score > uptrend_score:
        return "downtrend"
    else:
        return "range-bound"

def detect_high_volatility(data: pd.DataFrame, price_column: str, atr_threshold: float) -> str:
    """
    ATRを用いて高ボラティリティ相場を判定する関数。

    Args:
        data (pd.DataFrame): 株価データを含むDataFrame。
        price_column (str): ATR計算に使用する価格カラム名。
        atr_threshold (float): 高ボラティリティを判定するATRの閾値。

    Returns:
        str: "high volatility"（高ボラティリティ）または"normal volatility"（通常のボラティリティ）。
    """
    # ATRを計算
    data = calculate_atr(data, price_column)

    # 最新のATR値を取得
    latest_atr = data['ATR'].iloc[-1]

    # 高ボラティリティ判定
    if latest_atr > atr_threshold:
        return "high volatility"
    else:
        return "normal volatility"

class EnhancedTrendAnalyzer:
    """強化されたトレンド分析クラス - 精度検証対応"""
    
    def __init__(self, data: pd.DataFrame, price_column: str = "Adj Close"):
        self.data = data.copy()
        self.price_column = price_column
        self.confidence_scores = {}
    
    def detect_trend_with_confidence(self, 
                                   method: Literal["sma", "macd", "combined", "advanced"] = "advanced",
                                   **kwargs) -> Tuple[str, float]:
        """
        トレンド判定と信頼度を返す
        
        Returns:
            Tuple[str, float]: (トレンド, 信頼度スコア 0-1)
        """
        try:
            if method == "sma":
                # SMA用パラメータだけ抽出
                sma_params = {
                    k: v for k, v in kwargs.items() 
                    if k in ["short_period", "medium_period", "long_period"]
                }
                return self._detect_trend_sma_with_confidence(**sma_params)
                
            elif method == "macd":
                # MACD用パラメータだけ抽出
                macd_params = {
                    k: v for k, v in kwargs.items()
                    if k in ["short_window", "long_window", "signal_window"]
                }
                return self._detect_trend_macd_with_confidence(**macd_params)
                
            elif method == "combined" or method == "advanced":
                return self._detect_trend_advanced_with_confidence(**kwargs)
                
            else:
                return "unknown", 0.0
                
        except Exception as e:
            print(f"トレンド判定エラー: {e}, メソッド: {method}")
            return "unknown", 0.0
    
    def _detect_trend_sma_with_confidence(self, 
                                        short_period: int = 10,
                                        medium_period: int = 20, 
                                        long_period: int = 50) -> Tuple[str, float]:
        """SMAベースのトレンド判定（信頼度付き）"""
        if len(self.data) < max(short_period, medium_period, long_period):
            return "unknown", 0.0
        
        short_sma = calculate_sma(self.data, self.price_column, short_period).iloc[-1]
        medium_sma = calculate_sma(self.data, self.price_column, medium_period).iloc[-1]
        long_sma = calculate_sma(self.data, self.price_column, long_period).iloc[-1]
        current_price = float(self.data[self.price_column].iloc[-1])
        
        # SMAの順序による基本判定
        if short_sma > medium_sma > long_sma:
            trend = "uptrend"
            # 信頼度計算：価格とSMAの乖離具合
            confidence = min(1.0, (current_price - float(long_sma)) / float(long_sma) * 10)
        elif short_sma < medium_sma < long_sma:
            trend = "downtrend"
            confidence = min(1.0, (float(long_sma) - current_price) / float(long_sma) * 10)
        else:
            trend = "range-bound"
            # レンジの信頼度：SMA間の近接度
            sma_spread = abs(float(short_sma) - float(long_sma)) / ((float(short_sma) + float(long_sma)) / 2)
            confidence = max(0.3, 1.0 - sma_spread * 20)
        
        return trend, max(0.0, min(1.0, confidence))
    
    def _detect_trend_macd_with_confidence(self, 
                                         short_window: int = 12,
                                         long_window: int = 26, 
                                         signal_window: int = 9) -> Tuple[str, float]:
        """MACDベースのトレンド判定（信頼度付き）"""
        if len(self.data) < max(short_window, long_window) + signal_window:
            return "unknown", 0.0
        
        macd, signal = calculate_macd(self.data, self.price_column, 
                                    short_window, long_window, signal_window)
        
        latest_macd = float(macd.iloc[-1])
        latest_signal = float(signal.iloc[-1])
        
        # MACD線とシグナル線の関係
        macd_diff = latest_macd - latest_signal
        
        if latest_macd > latest_signal and latest_macd > 0:
            trend = "uptrend"
            confidence = min(1.0, abs(macd_diff) * 100)
        elif latest_macd < latest_signal and latest_macd < 0:
            trend = "downtrend"
            confidence = min(1.0, abs(macd_diff) * 100)
        else:
            trend = "range-bound"
            confidence = max(0.3, 1.0 - abs(macd_diff) * 200)
        
        return trend, max(0.0, min(1.0, confidence))
    
    def _detect_trend_advanced_with_confidence(self, **kwargs) -> Tuple[str, float]:
        """
        高度なトレンド判定（複数指標の組み合わせ）
        """
        # SMA用パラメータ
        sma_params = {
            "short_period": kwargs.get("short_period", 10),
            "medium_period": kwargs.get("medium_period", 20),
            "long_period": kwargs.get("long_period", 50)
        }
        
        # MACD用パラメータ
        macd_params = {
            "short_window": kwargs.get("short_window", 12),
            "long_window": kwargs.get("long_window", 26),
            "signal_window": kwargs.get("signal_window", 9)
        }
        
        # 各手法で判定
        sma_trend, sma_conf = self._detect_trend_sma_with_confidence(**sma_params)
        macd_trend, macd_conf = self._detect_trend_macd_with_confidence(**macd_params)
        
        # 価格モメンタムの追加
        momentum_trend, momentum_conf = self._detect_momentum_trend()
        
        # 投票システム
        trends = [sma_trend, macd_trend, momentum_trend]
        confidences = [sma_conf, macd_conf, momentum_conf]
        
        # 重み付き投票
        trend_scores = {"uptrend": 0.0, "downtrend": 0.0, "range-bound": 0.0}
        total_weight = 0.0
        
        for trend, conf in zip(trends, confidences):
            if trend in trend_scores:
                trend_scores[trend] += conf
                total_weight += conf
        
        if total_weight == 0:
            return "unknown", 0.0
        
        # 最も支持されたトレンド
        best_trend = max(trend_scores.keys(), key=lambda k: trend_scores[k])
        overall_confidence = trend_scores[best_trend] / total_weight
        
        return best_trend, overall_confidence
    
    def _detect_momentum_trend(self) -> Tuple[str, float]:
        """価格モメンタムによるトレンド判定"""
        if len(self.data) < 10:
            return "unknown", 0.0
        
        # 短期価格変化
        current_price = float(self.data[self.price_column].iloc[-1])
        past_price = float(self.data[self.price_column].iloc[-10])
        
        momentum = (current_price / past_price) - 1
        
        if momentum > 0.02:  # 2%以上の上昇
            return "uptrend", min(1.0, momentum * 10)
        elif momentum < -0.02:  # 2%以上の下落
            return "downtrend", min(1.0, abs(momentum) * 10)
        else:
            return "range-bound", max(0.3, 1.0 - abs(momentum) * 25)
    
    def _detect_trend_combined_with_confidence(self, **kwargs) -> Tuple[str, float]:
        """複合的なトレンド判定"""
        # SMA用パラメータ
        sma_params = {
            "short_period": kwargs.get("short_period", 10),
            "medium_period": kwargs.get("medium_period", 20),
            "long_period": kwargs.get("long_period", 50)
        }
        
        # MACD用パラメータ
        macd_params = {
            "short_window": kwargs.get("short_window", 12),
            "long_window": kwargs.get("long_window", 26),
            "signal_window": kwargs.get("signal_window", 9)
        }
        
        # 個別に判定して結果を結合
        sma_trend, sma_conf = self._detect_trend_sma_with_confidence(**sma_params)
        macd_trend, macd_conf = self._detect_trend_macd_with_confidence(**macd_params)
        
        # 両方が一致した場合のみ明確なトレンド
        if sma_trend == macd_trend:
            return sma_trend, (sma_conf + macd_conf) / 2
        else:
            return "range-bound", max(sma_conf, macd_conf) * 0.5
    
    def detect_market_trend(self, market_data: pd.DataFrame, 
                           trend_days: int = 5) -> str:
        """市場全体のトレンド判定（ダウ平均等）"""
        if len(market_data) < trend_days:
            return "neutral"
        
        first_close = float(market_data['Close'].iloc[-trend_days])
        last_close = float(market_data['Close'].iloc[-1])
        
        change_pct = (last_close / first_close - 1)
        
        if change_pct > 0.01:  # 1%以上上昇
            return "up"
        elif change_pct < -0.01:  # 1%以上下落
            return "down"
        else:
            return "neutral"

# 互換性維持のための関数
def detect_trend(data: pd.DataFrame, price_column: str = "Adj Close", 
                lookback_period: int = 20, **kwargs) -> str:
    """既存コードとの互換性を維持する関数"""
    analyzer = EnhancedTrendAnalyzer(data, price_column)
    trend, confidence = analyzer.detect_trend_with_confidence(method="advanced", **kwargs)
    return trend

def detect_trend_with_confidence(data: pd.DataFrame, price_column: str = "Adj Close", 
                               **kwargs) -> Tuple[str, float]:
    """信頼度付きトレンド判定"""
    analyzer = EnhancedTrendAnalyzer(data, price_column)
    return analyzer.detect_trend_with_confidence(method="advanced", **kwargs)

# VWAPとゴールデンクロス戦略向けの特化関数
def detect_vwap_trend(data: pd.DataFrame, price_column: str = "Adj Close", 
                     vwap_column: str = "VWAP") -> Tuple[str, float]:
    """VWAP戦略向けの特化されたトレンド判定"""
    if len(data) < 20 or vwap_column not in data.columns:
        return "unknown", 0.0
    
    analyzer = EnhancedTrendAnalyzer(data, price_column)
    base_trend, base_conf = analyzer.detect_trend_with_confidence(method="advanced")
    
    # VWAP関係の追加チェック
    current_price = float(data[price_column].iloc[-1])
    current_vwap = float(data[vwap_column].iloc[-1])
    
    # 価格とVWAPの関係による信頼度調整
    vwap_ratio = current_price / current_vwap
    if base_trend == "uptrend" and vwap_ratio > 1.01:  # 価格がVWAPより1%以上高い
        enhanced_conf = min(1.0, base_conf * 1.2)
    elif base_trend == "downtrend" and vwap_ratio < 0.99:  # 価格がVWAPより1%以上低い
        enhanced_conf = min(1.0, base_conf * 1.2)
    else:
        enhanced_conf = base_conf * 0.8  # 信頼度をやや下げる
    
    return base_trend, enhanced_conf

def detect_golden_cross_trend(data: pd.DataFrame, price_column: str = "Adj Close",
                             short_ma_col: str = "SMA_25", long_ma_col: str = "SMA_75") -> Tuple[str, float]:
    """ゴールデンクロス戦略向けの特化されたトレンド判定"""
    if len(data) < 75:  # 長期MAに必要な期間
        return "unknown", 0.0
    
    # MAがない場合は計算
    if short_ma_col not in data.columns:
        data[short_ma_col] = calculate_sma(data, price_column, 25)
    if long_ma_col not in data.columns:
        data[long_ma_col] = calculate_sma(data, price_column, 75)
    
    analyzer = EnhancedTrendAnalyzer(data, price_column)
    base_trend, base_conf = analyzer.detect_trend_with_confidence(method="sma", 
                                                                short_period=25, 
                                                                medium_period=50, 
                                                                long_period=75)
    
    # ゴールデンクロス/デッドクロスの確認
    short_ma_current = float(data[short_ma_col].iloc[-1])
    long_ma_current = float(data[long_ma_col].iloc[-1])
    short_ma_prev = float(data[short_ma_col].iloc[-2]) if len(data) > 1 else short_ma_current
    long_ma_prev = float(data[long_ma_col].iloc[-2]) if len(data) > 1 else long_ma_current
    
    # クロスオーバーの検出
    golden_cross = (short_ma_prev <= long_ma_prev) and (short_ma_current > long_ma_current)
    dead_cross = (short_ma_prev >= long_ma_prev) and (short_ma_current < long_ma_current)
    
    if golden_cross and base_trend == "uptrend":
        enhanced_conf = min(1.0, base_conf * 1.5)  # ゴールデンクロス発生時は信頼度大幅向上
    elif dead_cross and base_trend == "downtrend":
        enhanced_conf = min(1.0, base_conf * 1.5)  # デッドクロス発生時は信頼度大幅向上
    else:
        enhanced_conf = base_conf
    
    return base_trend, enhanced_conf

# テストコード
if __name__ == "__main__":
    # ダミーのデータ作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'High': [i + 5 for i in range(100)],
        'Low': [i for i in range(100)],
        'Adj Close': [i + (i % 5) * 2 for i in range(100)]  # ダミー価格データ
    }, index=dates)

    trend = detect_trend(df, price_column='Adj Close')
    print(f"トレンド判定: {trend}")

    volatility = detect_high_volatility(df, price_column='Adj Close', atr_threshold=10)
    print(f"ボラティリティ判定: {volatility}")
