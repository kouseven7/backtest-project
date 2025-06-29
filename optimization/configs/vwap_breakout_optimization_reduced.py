"""
VWAPブレイクアウト戦略の最適化設定ファイル
組み合わせ数を7,000程度に削減し、重要なパラメータを優先
"""

# 最適化パラメータグリッド - 組み合わせ数を約7,000通りに削減
PARAM_GRID = {
    # --- グループ1: リスクリワード設定（最重要パラメータ） ---
    "stop_loss": [0.02, 0.025, 0.03],  # ストップロス（詳細に探索）
    "take_profit": [0.08, 0.10, 0.12],  # 利益確定（詳細に探索）
    "breakout_min_percent": [0.002, 0.004],  # 最小ブレイク率（両端値）
    
    # --- グループ2: トレンド判定（トレード機会の中核） ---
    "sma_short": [5, 10, 15],  # 短期移動平均（重要なため詳細に探索）
    "sma_long": [20, 40],   # 長期移動平均（両端値）
    "volume_threshold": [1.1, 1.3],  # 出来高増加（両端値）
    "confirmation_bars": [0, 1],  # ブレイク確認バー数
    
    # --- グループ3: イグジット戦略（利益確保に重要） ---
    "trailing_stop": [0.04],  # トレーリングストップ（4%に固定）
    "trailing_start_threshold": [0.03],  # トレーリング開始閾値（3%に固定）
    "max_holding_period": [5, 15],  # 最大保有期間（両端値）
    
    # --- グループ4: フィルターと追加機能（革新的な部分を残す） ---
    "market_filter_method": ["sma", "rsi_plus"],  # 市場フィルター（最も効果的な2つ）
    "rsi_filter_enabled": [True, False],  # RSIフィルター
    "atr_filter_enabled": [True],  # ATRフィルター（有効のみ）
    "partial_exit_enabled": [True, False],  # 部分利確
    
    # --- グループ5: 補助パラメータ（固定値） ---
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
