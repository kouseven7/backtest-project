"""
Module: Trend Parameters Configuration
File: trend_params.py
Description: 
  トレンド判定のための共通パラメータ設定ファイルです。
  システム全体で統一したパラメータ体系を提供し、戦略間の一貫性を確保します。
  各戦略はこのファイルのパラメータを参照することでトレンド判定の一貫性を維持します。

Author: imega
Created: 2025-07-03
Modified: 2025-07-03

Dependencies:
  - None
"""

from typing import Dict, Any, List, Tuple

# 基本トレンド判定パラメータ
DEFAULT_TREND_PARAMS = {
    # SMAベーストレンド判定用
    "sma": {
        "short_period": 10,     # 短期SMA期間
        "medium_period": 20,    # 中期SMA期間
        "long_period": 50,      # 長期SMA期間
        "lookback_period": 10,  # 傾き計算用の遡り期間
        "volatility_threshold": 0.02,  # ボラティリティ閾値（レンジ判定用）
        "slope_threshold": 0.01,  # 傾きの閾値
        "uptrend_score": 4,     # 上昇トレンド判定スコア
    },
    
    # MACDベーストレンド判定用
    "macd": {
        "short_window": 12,     # 短期EMA期間
        "long_window": 26,      # 長期EMA期間
        "signal_window": 9,     # シグナルライン期間
        "macd_threshold": 0.0001,  # MACDとシグナルラインの差の閾値
    },
    
    # モメンタム判定用
    "momentum": {
        "lookback_period": 10,  # 価格比較期間
        "threshold": 0.02,      # トレンド判定閾値（2%）
    },
    
    # 予測精度向上用の高度な設定
    "advanced": {
        "voting_weights": {
            "sma": 1.0,         # SMA判定の重み
            "macd": 0.8,        # MACD判定の重み
            "momentum": 0.6     # モメンタム判定の重み
        },
        "confidence_threshold": 0.6,  # 高信頼度判定の閾値
    },
    
    # VWAP特化パラメータ
    "vwap": {
        "price_vwap_ratio_threshold": 0.01,  # 価格とVWAPの比率閾値
        "confidence_boost": 1.2,  # 一致時の信頼度ブースト係数
        "confidence_penalty": 0.8  # 不一致時の信頼度ペナルティ係数
    },
    
    # ゴールデンクロス特化パラメータ
    "golden_cross": {
        "short_ma": 25,         # 短期MA期間
        "long_ma": 75,          # 長期MA期間
        "cross_confidence_boost": 1.5  # クロス発生時の信頼度ブースト係数
    }
}

# 戦略ごとの推奨トレンド判定手法
STRATEGY_TREND_METHODS = {
    "VWAP_Bounce": "advanced",    # 強化トレンド判定（複数指標）
    "VWAP_Breakout": "combined",  # SMA+MACD複合判定
    "Golden_Cross": "sma",        # SMAベース判定
    "Moving_Average": "sma",      # SMAベース判定
    "Momentum": "macd",           # MACDベース判定
    "default": "advanced"         # デフォルトは強化判定
}

# トレンド別のデフォルト許容戦略
TREND_COMPATIBLE_STRATEGIES = {
    "uptrend": ["VWAP_Breakout", "Golden_Cross", "Moving_Average", "Momentum"],
    "downtrend": ["Golden_Cross", "Moving_Average", "Momentum"],
    "range-bound": ["VWAP_Bounce", "Mean_Reversion"]
}

# トレンド信頼度スコアの重要度しきい値
CONFIDENCE_THRESHOLDS = {
    "low": 0.3,      # 低信頼度
    "medium": 0.6,   # 中信頼度
    "high": 0.8      # 高信頼度
}

def get_trend_params(method: str = "advanced", strategy: str = "default") -> Dict[str, Any]:
    """
    指定された戦略とメソッドに基づいてトレンド判定パラメータを取得する
    
    Parameters:
        method (str): 使用するトレンド判定メソッド (sma, macd, combined, advanced)
        strategy (str): 戦略名
        
    Returns:
        Dict[str, Any]: トレンド判定パラメータ
    """
    # 戦略に適した判定メソッドを取得
    if method == "auto" and strategy in STRATEGY_TREND_METHODS:
        method = STRATEGY_TREND_METHODS[strategy]
    elif method == "auto":
        method = STRATEGY_TREND_METHODS["default"]
        
    # メソッド別のパラメータを返す
    params = {}
    if method == "sma":
        params = DEFAULT_TREND_PARAMS["sma"]
    elif method == "macd":
        params = DEFAULT_TREND_PARAMS["macd"]
    elif method == "combined":
        # SMAとMACDのパラメータを統合
        params = {**DEFAULT_TREND_PARAMS["sma"], **DEFAULT_TREND_PARAMS["macd"]}
    elif method == "advanced":
        # 全てのパラメータを統合
        params = {**DEFAULT_TREND_PARAMS["sma"], 
                 **DEFAULT_TREND_PARAMS["macd"],
                 **DEFAULT_TREND_PARAMS["momentum"],
                 **DEFAULT_TREND_PARAMS["advanced"]}
        
    # 戦略特化パラメータを追加
    if strategy.startswith("VWAP_"):
        params = {**params, **DEFAULT_TREND_PARAMS["vwap"]}
    elif strategy.startswith("Golden_Cross"):
        params = {**params, **DEFAULT_TREND_PARAMS["golden_cross"]}
        
    return params

def is_strategy_compatible_with_trend(strategy: str, trend: str) -> bool:
    """
    指定された戦略が現在のトレンド状況に適合するか判定
    
    Parameters:
        strategy (str): 戦略名
        trend (str): 現在のトレンド (uptrend, downtrend, range-bound)
        
    Returns:
        bool: 戦略がトレンドに適合するならTrue
    """
    if trend not in TREND_COMPATIBLE_STRATEGIES:
        return True  # 不明なトレンドの場合は制限しない
        
    return strategy in TREND_COMPATIBLE_STRATEGIES[trend]

def get_confidence_level(confidence_score: float) -> str:
    """
    信頼度スコアからレベル（低、中、高）を取得
    
    Parameters:
        confidence_score (float): 0-1の信頼度スコア
        
    Returns:
        str: 信頼度レベル ("low", "medium", "high")
    """
    if confidence_score >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif confidence_score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"
