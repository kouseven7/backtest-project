"""
VWAPブレイクアウト戦略の最適化設定ファイル
組み合わせ数を7,000程度に削減し、重要なパラメータを優先
"""

# 最適化パラメータグリッド
PARAM_GRID = {
    # --- 最重要パラメータ（取引の核心部分） ---
    "stop_loss": [0.02, 0.025, 0.03],  # ストップロス（最も重要なリスク管理パラメータ）
    "take_profit": [0.08, 0.10, 0.12],  # 利益確定（最適範囲に集中）
    "breakout_min_percent": [0.002, 0.003, 0.004],  # 最小ブレイク率（エントリー条件として重要）
    
    # --- トレンドとエントリー判定（取引機会に直結） ---
    "sma_short": [5, 10, 15],  # 短期移動平均
    "sma_long": [20, 30, 40],   # 長期移動平均
    "volume_threshold": [1.1, 1.2, 1.3],  # 出来高増加（感度調整）
    "confirmation_bars": [0, 1],  # ブレイク確認バー数
    
    # --- イグジット戦略（利益確保に重要） ---
    "trailing_stop": [0.03, 0.05],  # トレーリングストップ
    "trailing_start_threshold": [0.02, 0.04],  # トレーリング開始閾値
    "max_holding_period": [5, 15],  # 最大保有期間（両端値のみ）
    
    # --- フィルターと追加機能（革新的な部分を残す） ---
    "market_filter_method": ["sma", "rsi_plus"],  # 市場フィルター（最も効果的な2つ）
    "rsi_filter_enabled": [True, False],  # RSIフィルター
    "atr_filter_enabled": [True],  # ATRフィルターは有効化（前回のテストで効果的）
    "partial_exit_enabled": [True, False],  # 部分利確（革新的な機能として検証）
    
    # --- 補助パラメータ（影響度は低めだが差別化できる） ---
    "rsi_period": [14],  # RSI計算期間（標準値に固定）
    "volume_increase_mode": ["simple"],  # 出来高増加判定方式（シンプルモードに固定）
}

# 最適化目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},  # シャープレシオは重要
    {"name": "sortino_ratio", "weight": 0.8},  # ソルティノレシオもリスク調整で重要
    {"name": "win_rate", "weight": 0.7},       # 勝率も重視
    {"name": "expectancy", "weight": 0.6},     # 期待値
    {"name": "profit_factor", "weight": 0.5}   # 利益率/損失率の比率
]