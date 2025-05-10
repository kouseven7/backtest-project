"""
GC戦略の最適化設定ファイル
"""

# GC戦略の最適化パラメータ
PARAM_GRID = {
    "short_window": [5, 10, 15, 20],           # 短期移動平均期間
    "long_window": [50, 100, 200],             # 長期移動平均期間
    "take_profit": [0.03, 0.05, 0.08],         # 利益確定レベル
    "stop_loss": [0.02, 0.03, 0.05],           # ストップロスレベル
    "trailing_stop_pct": [0.02, 0.03, 0.05],   # トレーリングストップの割合
    "max_hold_days": [10, 15, 20],             # 最大保有期間
    "exit_on_death_cross": [True, False],      # デッドクロスでイグジット
    "confirmation_days": [1, 2],               # クロス確認日数
    "ma_type": ["SMA", "EMA"],                 # 移動平均の種類
}

# 最適化の目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.8},
    {"name": "win_rate", "weight": 0.6},
    {"name": "risk_adjusted_return", "weight": 0.7}
]