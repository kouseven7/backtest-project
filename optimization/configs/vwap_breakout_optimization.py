"""
VWAPブレイクアウト戦略の最適化設定ファイル
目標パラメータ組み合わせ数: 4,000〜8,000通り
"""

# 最適化パラメータグリッド
PARAM_GRID = {
    # --- リスクリワード設定（最重要パラメータ）---
    # stop_loss: 5値 - 損益に直結する最重要パラメータ
    "stop_loss": [0.015, 0.02, 0.025, 0.03, 0.04],
    
    # take_profit: 5値 - 利益確定の閾値（最重要パラメータ）
    "take_profit": [0.05, 0.08, 0.10, 0.12, 0.15],
    
    # trailing_stop: 3値 - 利益を守るトレール幅（重要パラメータ）
    "trailing_stop": [0.02, 0.03, 0.04],
    
    # --- エントリー条件（重要パラメータ）---
    # volume_threshold: 4値 - 出来高増加検知の精度向上（重要）
    "volume_threshold": [1.2, 1.3, 1.5, 1.8],
    
    # confirmation_bars: 2値 - 偽ブレイク除外に重要
    "confirmation_bars": [1, 2],
    
    # --- トレンドパラメータ ---
    # sma_short: 固定値に簡略化
    "sma_short": [10],
    
    # sma_long: 2値 - 長期トレンドの感度調整
    "sma_long": [30, 50],
    
    # --- 基本設定 ---
    # breakout_min_percent: 2値 - ブレイク強度の判定
    "breakout_min_percent": [0.002, 0.004],
    
    # trailing_start_threshold: 固定値に簡略化
    "trailing_start_threshold": [0.03],
    
    # max_holding_period: 2値 - 長期保有リスク制御
    "max_holding_period": [7, 15],
    
    # --- フィルター設定 ---
    # market_filter_method: 1値 - 市場トレンドフィルタの簡略化
    "market_filter_method": ["sma"],
    
    # --- 他のパラメータは固定値 ---
    "rsi_filter_enabled": [False],
    "atr_filter_enabled": [False],
    "partial_exit_enabled": [False],
    "rsi_period": [14],
    "volume_increase_mode": ["simple"],         # 出来高増加判定方式（固定）
}

# 組み合わせ数計算:
# 5(stop_loss) x 5(take_profit) x 3(trailing_stop) x 
# 4(volume_threshold) x 2(confirmation_bars) x 
# 1(sma_short) x 2(sma_long) x 
# 2(breakout_min_percent) x 1(trailing_start_threshold) x 2(max_holding_period) x 
# 1(market_filter_method) x 1 x 1 x 1 x 1 x 1
# = 5 x 5 x 3 x 4 x 2 x 1 x 2 x 2 x 1 x 2 x 1 x 1 x 1 x 1 x 1 x 1
# = 4,800通り ※目標範囲内（4,000～8,000通り）

# 最適化目的関数設定
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.8},
    {"name": "expectancy", "weight": 0.6}
]