"""
Breakout戦略の最適化設定ファイル
"""

# Breakout戦略の最適化パラメータ
PARAM_GRID = {
    "volume_threshold": [1.1, 1.2, 1.3, 1.4, 1.5],  # ボリューム閾値
    "take_profit": [0.02, 0.03, 0.04, 0.05],       # 利確レベル
    "stop_loss": [0.01, 0.015, 0.02, 0.025],       # 損切りレベル
    "look_back": [1, 2, 3],                        # 過去の期間数
    "trailing_stop_pct": [0.01, 0.02, 0.03],       # トレーリングストップ
    "max_hold_days": [5, 10, 15]                   # 最大保有期間
}

# 最適化の目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.6},
    {"name": "win_rate", "weight": 0.4},
    {"name": "risk_adjusted_return", "weight": 0.8}
]