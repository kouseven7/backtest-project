"""
Module: Unified Trend Detector
File: unified_trend_detector.py
Description: 
  統一されたトレンド判定インターフェースを提供するモジュールです。
  プロジェクト全体で一貫したトレンド判定を可能にし、各戦略から
  同じインターフェースで高精度なトレンド判定を利用できるようにします。

Author: imega
Created: 2025-07-03
Modified: 2025-07-03

Dependencies:
  - pandas
  - numpy
  - config.trend_params
  - indicators.trend_analysis
  - indicators.momentum_indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional, Literal, Union, Callable
import logging
import sys

sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")  # プロジェクトのルートを追加

# 既存の分析モジュールをインポート
from config.trend_params import get_trend_params, get_confidence_level, is_strategy_compatible_with_trend
from indicators.trend_analysis import EnhancedTrendAnalyzer
from indicators.basic_indicators import calculate_sma, calculate_rsi
from indicators.momentum_indicators import calculate_macd

# ロガーの設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class UnifiedTrendDetector:
    """
    統一されたトレンド判定を提供するクラス
    
    特徴:
    1. 複数の戦略から同じインターフェースで利用可能
    2. 高度なトレンド判定アルゴリズム
    3. 信頼度スコア付きの判定結果
    4. トレンド判定精度の検証機能
    5. VWAP・ゴールデンクロス等の戦略に最適化された判定方法
    """
    
    def __init__(self, data: pd.DataFrame, price_column: str = "Adj Close",
                 strategy_name: str = "default", method: str = "auto",
                 vwap_column: Optional[str] = None):
        """
        初期化
        
        Parameters:
            data (pd.DataFrame): 株価データ
            price_column (str): 価格データのカラム名
            strategy_name (str): 戦略名（パラメータ最適化のため）
            method (str): トレンド判定手法（"auto", "sma", "macd", "combined", "advanced"）
            vwap_column (str): VWAP値を含むカラム名（VWAPベースの戦略用）
        """
        self.data = data.copy()
        self.price_column = price_column
        self.strategy_name = strategy_name
        self.vwap_column = vwap_column
        
        # 設定するトレンド判定メソッド
        self.method = method if method != "auto" else self._get_default_method()
        
        # パラメータの取得
        self.params = get_trend_params(self.method, self.strategy_name)
        
        # 内部使用のアナライザーを作成
        self._analyzer = EnhancedTrendAnalyzer(self.data, self.price_column)
        
        # キャッシュの初期化
        self._trend_cache = {}
        self._last_update_time = None
        
        # ログ出力
        logger.debug(f"UnifiedTrendDetector initialized: strategy={strategy_name}, method={self.method}")
        
    def _get_default_method(self) -> str:
        """戦略に最適なデフォルトのトレンド判定手法を取得"""
        from config.trend_params import STRATEGY_TREND_METHODS
        return STRATEGY_TREND_METHODS.get(self.strategy_name, "advanced")
    
    def detect_trend(self, lookback: int = 0, use_cache: bool = True) -> str:
        """
        現在のトレンドを判定
        
        Parameters:
            lookback (int): 遡って判定する日数（0=最新）
            use_cache (bool): キャッシュを使用するかどうか
            
        Returns:
            str: トレンド判定結果 ("uptrend", "downtrend", "range-bound")
        """
        # キャッシュチェック
        cache_key = f"trend_{lookback}"
        if use_cache and cache_key in self._trend_cache:
            return self._trend_cache[cache_key][0]
            
        # 判定と信頼度を取得
        trend, _ = self.detect_trend_with_confidence(lookback)
        return trend
    
    def detect_trend_with_confidence(self, lookback: int = 0) -> Tuple[str, float]:
        """
        信頼度付きでトレンドを判定
        
        Parameters:
            lookback (int): 遡って判定する日数（0=最新）
            
        Returns:
            Tuple[str, float]: (トレンド判定結果, 信頼度スコア 0-1)
        """
        # データのチェック
        if len(self.data) <= lookback:
            return "unknown", 0.0
            
        # キャッシュチェック
        cache_key = f"trend_{lookback}"
        if cache_key in self._trend_cache:
            return self._trend_cache[cache_key]
            
        # 判定対象のデータ
        target_data = self.data.iloc[:-(lookback) if lookback > 0 else len(self.data)]
        
        # VWAP戦略の場合はVWAP特化判定
        if self.strategy_name.startswith("VWAP_") and self.vwap_column:
            if self.vwap_column in target_data.columns:
                trend, confidence = self._detect_vwap_trend(target_data)
                self._trend_cache[cache_key] = (trend, confidence)
                return trend, confidence
                
        # ゴールデンクロス戦略の場合はGC特化判定
        if self.strategy_name.startswith("Golden_Cross"):
            short_ma = self.params.get("short_ma", 25)
            long_ma = self.params.get("long_ma", 75)
            short_ma_col = f"SMA_{short_ma}"
            long_ma_col = f"SMA_{long_ma}"
            trend, confidence = self._detect_golden_cross_trend(target_data, short_ma_col, long_ma_col)
            self._trend_cache[cache_key] = (trend, confidence)
            return trend, confidence
            
        # 標準的なトレンド判定
        # 方法に基づいて適切なメソッドをLiteralとして直接指定
        method_literal = self.method
        
        # メソッドに適したパラメータだけを作成
        trend = "unknown"
        confidence = 0.0
        
        try:
            if method_literal == "sma":
                # SMAメソッド用パラメータ
                sma_params = {
                    "short_period": self.params.get("short_period", 10),
                    "medium_period": self.params.get("medium_period", 20),
                    "long_period": self.params.get("long_period", 50)
                }
                trend, confidence = self._analyzer.detect_trend_with_confidence(method="sma", **sma_params)
                
            elif method_literal == "macd":
                # MACDメソッド用パラメータ
                macd_params = {
                    "short_window": self.params.get("short_window", 12),
                    "long_window": self.params.get("long_window", 26),
                    "signal_window": self.params.get("signal_window", 9)
                }
                trend, confidence = self._analyzer.detect_trend_with_confidence(method="macd", **macd_params)
                
            elif method_literal == "combined":
                # combinedメソッド - 内部で適切に分割
                trend, confidence = self._analyzer.detect_trend_with_confidence(method="combined", 
                    short_period=self.params.get("short_period", 10),
                    medium_period=self.params.get("medium_period", 20),
                    long_period=self.params.get("long_period", 50),
                    short_window=self.params.get("short_window", 12),
                    long_window=self.params.get("long_window", 26),
                    signal_window=self.params.get("signal_window", 9)
                )
                
            elif method_literal == "advanced":
                # advancedメソッド - 内部で適切に分割
                trend, confidence = self._analyzer.detect_trend_with_confidence(method="advanced", 
                    short_period=self.params.get("short_period", 10),
                    medium_period=self.params.get("medium_period", 20),
                    long_period=self.params.get("long_period", 50),
                    short_window=self.params.get("short_window", 12),
                    long_window=self.params.get("long_window", 26),
                    signal_window=self.params.get("signal_window", 9)
                )
                
            else:
                # 不明なメソッド
                trend, confidence = "unknown", 0.0
                
        except Exception as e:
            # エラー処理（パラメータ不一致など）
            logger.error(f"トレンド判定エラー: {e}, method={method_literal}")
            trend, confidence = "unknown", 0.0
        
        # キャッシュに保存
        self._trend_cache[cache_key] = (trend, confidence)
        self._last_update_time = datetime.now()
        
        return trend, confidence
    
    def _detect_vwap_trend(self, data: pd.DataFrame) -> Tuple[str, float]:
        """VWAP戦略向けの特化トレンド判定"""
        # 基本的なトレンド判定
        method_literal = self.method
        
        # デフォルトの戻り値
        base_trend = "unknown"
        base_conf = 0.0
        
        try:
            if method_literal == "sma":
                # SMAメソッド用パラメータ
                sma_params = {
                    "short_period": self.params.get("short_period", 10),
                    "medium_period": self.params.get("medium_period", 20),
                    "long_period": self.params.get("long_period", 50)
                }
                base_trend, base_conf = self._analyzer.detect_trend_with_confidence(method="sma", **sma_params)
                
            elif method_literal == "macd":
                # MACDメソッド用パラメータ
                macd_params = {
                    "short_window": self.params.get("short_window", 12),
                    "long_window": self.params.get("long_window", 26),
                    "signal_window": self.params.get("signal_window", 9)
                }
                base_trend, base_conf = self._analyzer.detect_trend_with_confidence(method="macd", **macd_params)
                
            elif method_literal == "combined":
                # combinedメソッド - パラメータを適切に渡す
                base_trend, base_conf = self._analyzer.detect_trend_with_confidence(method="combined", 
                    short_period=self.params.get("short_period", 10),
                    medium_period=self.params.get("medium_period", 20),
                    long_period=self.params.get("long_period", 50),
                    short_window=self.params.get("short_window", 12),
                    long_window=self.params.get("long_window", 26),
                    signal_window=self.params.get("signal_window", 9)
                )
                
            elif method_literal == "advanced":
                # advancedメソッド - パラメータを適切に渡す
                base_trend, base_conf = self._analyzer.detect_trend_with_confidence(method="advanced", 
                    short_period=self.params.get("short_period", 10),
                    medium_period=self.params.get("medium_period", 20),
                    long_period=self.params.get("long_period", 50),
                    short_window=self.params.get("short_window", 12),
                    long_window=self.params.get("long_window", 26),
                    signal_window=self.params.get("signal_window", 9)
                )
                
            else:
                # 不明なメソッド
                logger.warning(f"未知のトレンド判定メソッド: {method_literal}、デフォルトのadvancedを使用")
                # advancedメソッドをデフォルトとして使用
                base_trend, base_conf = self._analyzer.detect_trend_with_confidence(method="advanced", 
                    short_period=self.params.get("short_period", 10),
                    medium_period=self.params.get("medium_period", 20),
                    long_period=self.params.get("long_period", 50),
                    short_window=self.params.get("short_window", 12),
                    long_window=self.params.get("long_window", 26),
                    signal_window=self.params.get("signal_window", 9)
                )
                
        except Exception as e:
            # エラー処理（パラメータ不一致など）
            logger.error(f"VWAPトレンド判定エラー: {e}, method={method_literal}")
            base_trend = "unknown"
            base_conf = 0.0
        
        # VWAP関係の追加チェック
        vwap_col = self.vwap_column
        if vwap_col not in data.columns:
            return base_trend, base_conf
        
        current_price = float(data[self.price_column].iloc[-1])
        current_vwap = float(data[vwap_col].iloc[-1])
        
        # 価格とVWAPの関係による信頼度調整
        threshold = self.params.get("price_vwap_ratio_threshold", 0.01)
        boost = self.params.get("confidence_boost", 1.2)
        penalty = self.params.get("confidence_penalty", 0.8)
        
        vwap_ratio = current_price / current_vwap
        if base_trend == "uptrend" and vwap_ratio > 1 + threshold:
            # 価格がVWAPより上で上昇トレンド → 信頼度アップ
            enhanced_conf = min(1.0, base_conf * boost)
        elif base_trend == "downtrend" and vwap_ratio < 1 - threshold:
            # 価格がVWAPより下で下降トレンド → 信頼度アップ
            enhanced_conf = min(1.0, base_conf * boost)
        else:
            # 不一致 → 信頼度ダウン
            enhanced_conf = base_conf * penalty
        
        return base_trend, enhanced_conf
    
    def _detect_golden_cross_trend(self, data: pd.DataFrame, 
                                 short_ma_col: str, long_ma_col: str) -> Tuple[str, float]:
        """ゴールデンクロス戦略向けの特化トレンド判定"""
        # データのコピーを作成して確実にコピーで作業する
        data_copy = data.copy()
        
        # MAがない場合は計算
        if short_ma_col not in data_copy.columns:
            short_ma = int(short_ma_col.split('_')[1])
            data_copy[short_ma_col] = calculate_sma(data_copy, self.price_column, short_ma)
        if long_ma_col not in data_copy.columns:
            long_ma = int(long_ma_col.split('_')[1])
            data_copy[long_ma_col] = calculate_sma(data_copy, self.price_column, long_ma)
        
        # 基本的なトレンド判定（明示的にパラメータを渡してエラーを避ける）
        base_trend = "unknown"
        base_conf = 0.0
        
        try:
            # SMA特化のパラメータセット
            short_period = int(short_ma_col.split('_')[1])
            long_period = int(long_ma_col.split('_')[1])
            
            # SMA判定を使用
            base_trend, base_conf = self._analyzer.detect_trend_with_confidence(
                method="sma",
                short_period=short_period,
                medium_period=50,
                long_period=long_period
            )
        except Exception as e:
            logger.error(f"ゴールデンクロストレンド判定エラー: {e}")
            return "unknown", 0.0
        
        # クロスオーバー確認
        if len(data_copy) < 2:
            return base_trend, base_conf
            
        short_ma_current = float(data_copy[short_ma_col].iloc[-1])
        long_ma_current = float(data_copy[long_ma_col].iloc[-1])
        short_ma_prev = float(data_copy[short_ma_col].iloc[-2])
        long_ma_prev = float(data_copy[long_ma_col].iloc[-2])
        
        # ゴールデンクロス/デッドクロスの検出
        golden_cross = (short_ma_prev <= long_ma_prev) and (short_ma_current > long_ma_current)
        dead_cross = (short_ma_prev >= long_ma_prev) and (short_ma_current < long_ma_current)
        
        # クロス発生時の信頼度ブースト
        boost = self.params.get("cross_confidence_boost", 1.5)
        
        if golden_cross and base_trend == "uptrend":
            enhanced_conf = min(1.0, base_conf * boost)
        elif dead_cross and base_trend == "downtrend":
            enhanced_conf = min(1.0, base_conf * boost)
        else:
            enhanced_conf = base_conf
        
        return base_trend, enhanced_conf
    
    def get_trend_confidence_level(self, lookback: int = 0) -> str:
        """
        トレンド信頼度のレベル（低・中・高）を取得
        
        Parameters:
            lookback (int): 遡って判定する日数（0=最新）
            
        Returns:
            str: 信頼度レベル ("low", "medium", "high")
        """
        _, confidence = self.detect_trend_with_confidence(lookback)
        return get_confidence_level(confidence)
    
    def is_trend_reliable(self, lookback: int = 0, min_level: str = "medium") -> bool:
        """
        トレンド判定が信頼できるかどうか
        
        Parameters:
            lookback (int): 遡って判定する日数
            min_level (str): 最小信頼度レベル
            
        Returns:
            bool: 信頼できる場合はTrue
        """
        level = self.get_trend_confidence_level(lookback)
        
        if min_level == "low":
            return level in ["low", "medium", "high"]
        elif min_level == "medium":
            return level in ["medium", "high"]
        else:  # high
            return level == "high"
    
    def is_strategy_compatible(self) -> bool:
        """
        現在の戦略が現在のトレンドに適合するか判定
        
        Returns:
            bool: 適合する場合はTrue
        """
        trend = self.detect_trend()
        return is_strategy_compatible_with_trend(self.strategy_name, trend)
    
    def get_trend_description(self, lookback: int = 0, with_confidence: bool = True) -> str:
        """
        トレンドの説明文を取得
        
        Parameters:
            lookback (int): 遡って判定する日数
            with_confidence (bool): 信頼度情報を含めるかどうか
            
        Returns:
            str: トレンドの説明文
        """
        trend, confidence = self.detect_trend_with_confidence(lookback)
        conf_level = get_confidence_level(confidence)
        
        trend_jp = {
            "uptrend": "上昇トレンド",
            "downtrend": "下降トレンド",
            "range-bound": "レンジ相場",
            "unknown": "不明"
        }.get(trend, "不明")
        
        conf_jp = {
            "high": "高",
            "medium": "中",
            "low": "低"
        }.get(conf_level, "不明")
        
        if with_confidence:
            return f"{trend_jp}（信頼度: {conf_jp}）"
        else:
            return trend_jp
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._trend_cache = {}
        self._last_update_time = None

# 関数ベースの互換インターフェース
def detect_unified_trend(data: pd.DataFrame, price_column: str = "Adj Close", 
                        strategy: str = "default", method: str = "auto",
                        vwap_column: Optional[str] = None) -> str:
    """
    統一トレンド判定インターフェース（関数版）
    
    Parameters:
        data (pd.DataFrame): 株価データ
        price_column (str): 価格カラム名
        strategy (str): 戦略名
        method (str): トレンド判定手法
        vwap_column (str): VWAPカラム名
    
    Returns:
        str: トレンド判定結果 ("uptrend", "downtrend", "range-bound")
    """
    detector = UnifiedTrendDetector(data, price_column, strategy, method, vwap_column)
    return detector.detect_trend()

def detect_unified_trend_with_confidence(data: pd.DataFrame, price_column: str = "Adj Close", 
                                      strategy: str = "default", method: str = "auto",
                                      vwap_column: Optional[str] = None) -> Tuple[str, float]:
    """
    信頼度付き統一トレンド判定インターフェース（関数版）
    
    Parameters:
        data (pd.DataFrame): 株価データ
        price_column (str): 価格カラム名
        strategy (str): 戦略名
        method (str): トレンド判定手法
        vwap_column (str): VWAPカラム名
    
    Returns:
        Tuple[str, float]: (トレンド判定結果, 信頼度スコア)
    """
    detector = UnifiedTrendDetector(data, price_column, strategy, method, vwap_column)
    return detector.detect_trend_with_confidence()

# テスト用コード
if __name__ == "__main__":
    # サンプルデータの作成
    dates = pd.date_range(start="2022-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'High': np.random.random(100) * 100 + 100,
        'Low': np.random.random(100) * 100,
        'Adj Close': np.random.random(100) * 100 + 50,
        'Volume': np.random.randint(100, 1000, 100),
        'VWAP': np.random.random(100) * 100 + 45
    }, index=dates)
    
    # 上昇トレンドにする
    df['Adj Close'] = [50 + i*0.5 + np.random.random()*5 for i in range(100)]
    df['VWAP'] = [48 + i*0.48 + np.random.random()*3 for i in range(100)]
    
    # 統一トレンド判定器のテスト
    detector = UnifiedTrendDetector(df, "Adj Close", "VWAP_Bounce", vwap_column="VWAP")
    trend, confidence = detector.detect_trend_with_confidence()
    
    print(f"トレンド判定: {trend}")
    print(f"信頼度スコア: {confidence:.2f}")
    print(f"信頼度レベル: {detector.get_trend_confidence_level()}")
    print(f"トレンド説明: {detector.get_trend_description()}")
    print(f"戦略適合性: {'適合' if detector.is_strategy_compatible() else '不適合'}")
