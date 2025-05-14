"""
VWAPブレイクアウト戦略の最適化設定ファイル
"""

# 最適化パラメータグリッド
PARAM_GRID = {
    # エントリー関連
    "sma_short": [10, 20, 30],
    "sma_long": [50, 100, 200],
    "volume_threshold": [1.0, 1.2, 1.5],
    "confirmation_bars": [0, 1, 2],
    "breakout_min_percent": [0, 0.005, 0.01],
    "volume_increase_mode": ["simple", "average"],
    
    # 指標・フィルター関連
    "rsi_period": [9, 14, 21],
    "atr_filter_enabled": [True, False],
    "rsi_filter_enabled": [True, False],
    "market_filter_method": ["sma", "macd"],
    
    # イグジット関連
    "stop_loss": [0.03, 0.05, 0.07],
    "take_profit": [0.07, 0.10, 0.15],
    "trailing_stop": [0.02, 0.03, 0.05],
    "trailing_start_threshold": [0, 0.03, 0.05],
    "max_holding_period": [5, 10, 15],
    "partial_exit_enabled": [True, False],
}

# 最適化目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.8},
    {"name": "expectancy", "weight": 0.6},
    {"name": "profit_factor", "weight": 0.5}
]