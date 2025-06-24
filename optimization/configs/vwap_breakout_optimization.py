"""
VWAPブレイクアウト戦略の最適化設定ファイル
"""

# 最適化パラメータグリッド
PARAM_GRID = {
    # --- リスクリワード重視・シンプル化 ---
    "stop_loss": [0.03, 0.05],  # ストップロス（浅め～標準）
    "take_profit": [0.10, 0.15],  # 利益確定（広め）
    "sma_short": [10],  # 短期移動平均（固定）
    "sma_long": [30],   # 長期移動平均（固定）
    "volume_threshold": [1.2],  # 出来高増加（やや緩め）
    "confirmation_bars": [1],  # ブレイク確認バー数
    "breakout_min_percent": [0.003],  # 最小ブレイク率
    "trailing_stop": [0.05],  # トレーリングストップ
    "trailing_start_threshold": [0.03],  # トレーリング開始閾値
    "max_holding_period": [10],  # 最大保有期間
    # --- フィルター・特殊機能は無効化 ---
    "market_filter_method": ["none"],  # 市場フィルター方式
    "rsi_filter_enabled": [False],  # RSIフィルター無効
    "atr_filter_enabled": [False],  # ATRフィルター無効
    "partial_exit_enabled": [False],  # 部分利確無効
    # --- その他（将来拡張用・固定値） ---
    "rsi_period": [14],  # RSI計算期間
    "volume_increase_mode": ["simple"],  # 出来高増加判定方式
}

# 最適化目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.8},
    {"name": "expectancy", "weight": 0.6}
]