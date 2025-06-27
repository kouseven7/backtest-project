"""
VWAPブレイクアウト戦略の最適化設定ファイル
組み合わせ総数を7,000以下に削減し、重要パラメータに効率的な値を割り当てる
"""

# 最適化パラメータグリッド - 重要度別に分類（組み合わせ数：約6,912通り）
PARAM_GRID = {
    # === 高重要度パラメータ (利益に大きく影響) ===
    
    # --- リスク/リワード調整（より収益を高める） ---
    "stop_loss": [0.03, 0.05, 0.07],  # ストップロス（5から3値に削減）
    "take_profit": [0.08, 0.12, 0.15, 0.18],  # 利益確定（5から4値に削減）
    
    # --- 重要なエントリー条件 ---
    "sma_short": [8, 10, 14],  # 短期移動平均（4から3値に削減）
    "sma_long": [20, 25, 35],  # 長期移動平均（4から3値に削減）
    "volume_threshold": [1.2, 1.5, 1.8],  # 出来高増加（維持）
    "breakout_min_percent": [0.003, 0.005, 0.007],  # 最小ブレイク率（維持）
    
    # --- 重要なイグジットロジック ---
    "trailing_stop": [0.04, 0.05, 0.07],  # トレーリングストップ（4から3値に削減）
    
    # === 中重要度パラメータ (組み合わせ数を減らす) ===
    
    "confirmation_bars": [1],  # ブレイク確認バー数（1に固定）
    "trailing_start_threshold": [0.04],  # トレーリング開始閾値（2から1値に削減）
    "max_holding_period": [15],  # 最大保有期間（2から1値に削減）
    "market_filter_method": ["none", "macd"],  # 市場フィルター方式（維持）
    
    # === 低重要度パラメータ (固定値または省略) ===
    
    # --- フィルター設定（固定値） ---
    "rsi_filter_enabled": [False],  # RSIフィルター（オフに固定）
    "atr_filter_enabled": [True],  # ATRフィルター（オンに固定）
    "partial_exit_enabled": [True],  # 部分利確（オンに固定）
    
    # --- 固定パラメータ（変更しない） ---
    "rsi_period": [14],  # RSI計算期間（標準値）
    "rsi_lower": [30],  # RSI下限値（標準値）
    "rsi_upper": [70],  # RSI上限値（標準値）
    "volume_increase_mode": ["average"],  # 出来高増加判定方式（averageに固定）
    "partial_exit_threshold": [0.07],  # 部分利確閾値（7%に固定）
    "partial_exit_portion": [0.5],  # 部分利確割合（50%に固定）
}

# 最適化目的関数設定（収益性とリスク調整後リターンを重視）
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},      # シャープレシオ（リスク調整後リターン）
    {"name": "sortino_ratio", "weight": 0.9},     # ソルティノレシオ（下方リスク調整後リターン）
    {"name": "expectancy", "weight": 0.8},        # 期待値（1取引あたりの期待収益）
    {"name": "win_rate_expectancy", "weight": 0.7}  # 勝率と期待値の組み合わせ
]