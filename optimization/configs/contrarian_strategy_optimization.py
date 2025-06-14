"""
Contrarian戦略（逆張り戦略）の最適化設定ファイル
"""

# 最適化対象パラメータ（損益に影響しやすいもののみグリッド化）
PARAM_GRID = {
    "rsi_period": [10, 14, 20],            # RSI計算期間
    "rsi_oversold": [25, 30, 35],          # RSI過売り閾値
    "gap_threshold": [0.01, 0.03],         # ギャップ判定閾値
    "stop_loss": [0.02, 0.04],             # 損切り幅
    "take_profit": [0.03, 0.05, 0.08],     # 利食い幅
    # 以下は固定値（strategyクラスのデフォルトparamsで指定）
    # "pin_bar_ratio": 2.0,
    # "max_hold_days": 5,
    # "rsi_exit_level": 50,
    # "trailing_stop_pct": 0.02,
}

# 固定値パラメータ（参考：strategyクラスのデフォルトparamsで指定）
FIXED_PARAMS = {
    "pin_bar_ratio": 2.0,
    "max_hold_days": 5,
    "rsi_exit_level": 50,
    "trailing_stop_pct": 0.02,
}

# 最適化の目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.7},
    {"name": "risk_adjusted_return", "weight": 0.6}
]