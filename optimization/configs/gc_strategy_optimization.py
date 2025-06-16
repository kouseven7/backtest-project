"""
GC戦略の最適化設定ファイル
"""

# GC戦略の最適化パラメータ
PARAM_GRID = {
    "short_window": [5, 10, 15, 20],
    "long_window": [50, 100, 200],
    "take_profit": [0.03, 0.05, 0.08],
    "stop_loss": [0.02, 0.03, 0.05],
    "trailing_stop_pct": [0.03],
    "max_hold_days": [15],
    "exit_on_death_cross": [True],
    "confirmation_days": [1],
    "ma_type": ["EMA"],
    # トレンド判定用パラメータを追加
    "trend_lookback_period": [3, 5, 10],
    "trend_short_period": [3, 5, 7],
    "trend_medium_period": [15, 25, 50],
    "trend_long_period": [50, 75, 100],
    "trend_up_score": [4, 5, 6],  # uptrend判定のスコア閾値
}

# 最適化の目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.8},
    {"name": "win_rate", "weight": 0.6},
    {"name": "risk_adjusted_return", "weight": 0.7}
]