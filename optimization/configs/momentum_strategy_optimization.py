"""
モメンタム戦略の最適化設定ファイル
"""

# モメンタム戦略の最適化パラメータ
PARAM_GRID = {
    # 移動平均線関連
    "ma_type": ["SMA", "EMA"],                 # 移動平均の種類
    "sma_short": [10, 15, 20, 25],             # 短期移動平均期間
    "sma_long": [50, 100, 200],                # 長期移動平均期間
    
    # RSI関連
    "rsi_period": [14],                         # RSI計算期間
    "rsi_lower": [40, 45, 50],                 # RSI下限閾値
    "rsi_upper": [65, 68, 70],                 # RSI上限閾値
    
    # 出来高関連
    "volume_threshold": [1.18],                 # 出来高増加閾値
    
    # 利確・損切り関連
    "take_profit": [0.08, 0.12, 0.15],         # 利確レベル
    "stop_loss": [0.04, 0.06, 0.08],           # 損切りレベル
    "trailing_stop": [0.04],                   # トレーリングストップ
    
    # 新規パラメータ
    "max_hold_days": [15],                     # 最大保有期間
    "atr_multiple": [2.0],                     # ATRストップロス倍率
    "partial_exit_pct": [0.0],                 # 一部利確率 (0=無効)
    "partial_exit_threshold": [0.08],           # 一部利確の発動閾値
    "momentum_exit_threshold": [-0.02],         # モメンタム失速閾値
    "volume_exit_threshold": [0.7],             # 出来高減少イグジット閾値
    "trend_filter": [True]                     # トレンドフィルターの使用
}

# 最適化の目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.8},
    {"name": "risk_adjusted_return", "weight": 0.7},
    {"name": "expectancy", "weight": 0.6}
]