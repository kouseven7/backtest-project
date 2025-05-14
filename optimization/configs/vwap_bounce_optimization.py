# optimization/configs/vwap_bounce_optimization.py
"""
VWAP反発戦略の最適化設定ファイル
"""

# 最適化パラメータグリッド
PARAM_GRID = {
    # レンジ環境
    "vwap_lower_threshold": [0.98, 0.99, 0.995],
    "vwap_upper_threshold": [1.01, 1.02, 1.03],
    
    # 出来高と陽線
    "volume_increase_threshold": [1.1, 1.2, 1.5],
    "bullish_candle_min_pct": [0.002, 0.005, 0.01],
    
    # トレンドとレンジ
    "trend_filter_enabled": [True, False],
    "allowed_trends": [["range-bound"], ["range-bound", "uptrend"]],
    
    # イグジット条件
    "stop_loss": [0.01, 0.02, 0.03],
    "take_profit": [0.03, 0.05, 0.07],
    "trailing_stop_pct": [0.01, 0.015, 0.02],
    "max_hold_days": [5, 10, 15],
    
    # 取引制限
    "cool_down_period": [0, 3, 5],
    "partial_exit_enabled": [True, False],
    "partial_exit_portion": [0.3, 0.5],
}

# 最適化目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.7},
    {"name": "expectancy", "weight": 0.5}
]