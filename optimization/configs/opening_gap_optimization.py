# optimization/configs/opening_gap_optimization.py
"""
OpeningGap戦略の最適化設定ファイル
"""

# 最適化パラメータグリッド
PARAM_GRID = {
    # エントリー関連
    "gap_threshold": [0.005, 0.01, 0.015, 0.02],
    "gap_direction": ["both"],  # 固定
    "dow_filter_enabled": [True],  # 固定
    "volatility_filter": [True],  # 固定
    "entry_delay": [0],  # 固定
    "min_vol_ratio": [1.0],  # 固定
    
    # イグジット関連
    "stop_loss": [0.01, 0.015, 0.02, 0.03],
    "take_profit": [0.03, 0.05, 0.07, 0.1],
    "max_hold_days": [2, 3, 5, 7],
    "trailing_stop_pct": [0.01, 0.02, 0.03],
    "atr_stop_multiple": [1.0, 1.5, 2.0],
    "consecutive_down_days": [1],  # 固定
    
    # 一部利確
    "partial_exit_enabled": [False],
}

# 最適化目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.8},
    {"name": "expectancy", "weight": 0.6}
]