# optimization/configs/vwap_bounce_optimization.py
"""
VWAP反発戦略の最適化設定ファイル
"""

# 最適化パラメータグリッド
PARAM_GRID = {
    # 核心パラメータ - 複数値を維持
    "vwap_lower_threshold": [0.97, 0.98, 0.99, 0.995],  # 4値に増やす
    "vwap_upper_threshold": [1.005, 1.01, 1.02, 1.03],  # 4値に増やす
    "volume_increase_threshold": [1.1, 1.2, 1.3, 1.5],  # 4値
    
    # 重要な損益管理パラメータ
    "stop_loss": [0.01, 0.015, 0.02, 0.025, 0.03],      # 5値に増やす
    "take_profit": [0.03, 0.04, 0.05, 0.06, 0.07],      # 5値に増やす
    "trailing_stop_pct": [0.01, 0.015, 0.02],           # 3値を維持
    
    # 固定値に変更するパラメータ
    "bullish_candle_min_pct": [0.005],                  # 中間値に固定
    "trend_filter_enabled": [True],                     # 固定
    "allowed_trends": [["range-bound", "uptrend"]],     # 固定
    "max_hold_days": [10],                              # 中間値に固定
    "cool_down_period": [3],                            # 中間値に固定
    "partial_exit_enabled": [False],                    # 固定
    "partial_exit_portion": [0.5]                       # 固定
}

# 最適化目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.7},
    {"name": "expectancy", "weight": 0.5}
]