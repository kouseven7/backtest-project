"""
VWAPブレイクアウト戦略の最適化設定ファイル
"""

# 最適化パラメータグリッド
PARAM_GRID = {
    # === 最も重要なパラメータ (多くの値を割り当てる) ===
    "sma_short": [10, 20, 30],  # 短期移動平均 (3値)
    "sma_long": [5, 10, 15, 20, 25, 30],  # 長期移動平均 (6値・データ長より十分小さく)
    "stop_loss": [0.03, 0.05, 0.07],  # ストップロス (3値)
    "take_profit": [0.07, 0.10, 0.15],  # テイクプロフィット (3値)

    # === 中程度重要なパラメータ (複数値を保持) ===
    "volume_threshold": [1.2, 1.5],  # 出来高閾値 (2値)
    "rsi_period": [14, 21],  # RSI期間 (2値)
    "market_filter_method": ["sma", "macd"],  # 市場フィルター方法 (2値)
    "trailing_stop": [0.03, 0.05], # トレイリングストップ (2値)

    # === 応用の必要がない/影響の小さいパラメータ (固定値にする) ===
    "confirmation_bars": [1],  # 確認バー (1値) - ノイズを減らすため0よりは1
    "breakout_min_percent": [0.005],  # ブレイクアウト最小パーセント (1値) - 中間的な値
    "volume_increase_mode": ["simple"],  # 出来高増加モード (1値) - まずはシンプルに
    "atr_filter_enabled": [True],  # ATRフィルター (1値) - 有効にしておく
    "rsi_filter_enabled": [True],  # RSIフィルター (1値) - 有効にしておく
    "trailing_start_threshold": [0.03],  # トレイリング開始閾値 (1値) - ある程度利益が出てから
    "max_holding_period": [10],  # 最大保有期間 (1値) - 短期～中期
    "partial_exit_enabled": [False],  # 部分決済 (1値) - まずはシンプルな全決済
}

# 最適化目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.8},
    {"name": "expectancy", "weight": 0.6},
    {"name": "profit_factor", "weight": 0.5}
]