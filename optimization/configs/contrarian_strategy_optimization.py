"""
Contrarian戦略（逆張り戦略）の最適化設定ファイル
"""

# Contrarian戦略の最適化パラメータ
PARAM_GRID = {
    "rsi_period": [10, 14, 20],            # RSI計算期間
    "rsi_oversold": [25, 30, 35],          # RSI過売り閾値
    "gap_threshold": [0.01, 0.02, 0.03],   # ギャップ閾値
    "stop_loss": [0.02, 0.03, 0.05],       # 損切りレベル
    "take_profit": [0.03, 0.05, 0.08],     # 利益確定レベル
    "pin_bar_ratio": [1.5, 2.0, 2.5],      # ピンバー比率
    "max_hold_days": [3, 5, 7, 10],        # 最大保有期間
    "rsi_exit_level": [45, 50, 55],        # RSIイグジットレベル
    "trailing_stop_pct": [0.01, 0.02, 0.03] # トレーリングストップの割合
}

# 最適化の目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.7},
    {"name": "risk_adjusted_return", "weight": 0.6}
]