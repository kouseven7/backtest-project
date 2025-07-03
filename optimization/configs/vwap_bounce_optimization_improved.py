# optimization/configs/vwap_bounce_optimization_improved.py
"""
VWAP反発戦略の改善された最適化設定ファイル
レンジ相場に特化し、トレンドフィルターを強化
"""

# 改善された最適化パラメータグリッド（レンジ相場特化、648通り）
PARAM_GRID = {
    # 核心パラメータ - VWAP閾値（レンジ相場に最適化）
    "vwap_lower_threshold": [0.995, 0.998, 0.999],       # VWAP-0.1%～0.5%（レンジ相場用）
    "vwap_upper_threshold": [1.001, 1.002, 1.005],       # VWAP+0.1%～0.5%（レンジ相場用）
    
    # 出来高・エントリー条件（レンジ相場に適応）
    "volume_increase_threshold": [1.0, 1.1, 1.2],        # 出来高増加（緩和）
    "bullish_candle_min_pct": [0.001, 0.002],            # 陽線条件（緩和）
    
    # 損益管理パラメータ（レンジ相場に最適化）
    "stop_loss": [0.015, 0.025],                         # 1.5%～2.5%（レンジ相場用）
    "take_profit": [0.02, 0.03, 0.04],                   # 2%～4%（短期利確）
    "trailing_stop_pct": [0.015],                        # トレーリングストップ固定
    
    # トレンド・保有期間設定（レンジ相場のみ）
    "trend_filter_enabled": [True],                      # トレンドフィルター有効
    "allowed_trends": [["range-bound"]],                 # レンジ相場のみ許可
    "max_hold_days": [3, 5, 8],                         # 短期保有（レンジ相場用）
    "cool_down_period": [1, 2],                         # 短いクールダウン
    
    # 固定値
    "partial_exit_enabled": [False],                     # 部分利確無効
    "partial_exit_portion": [0.5],                       # 固定
    "volatility_filter_enabled": [True]                 # ボラティリティフィルター有効
}

# パラメータ組み合わせ数: 3×3×3×2×2×3×1×1×1×3×2×1×1×1 = 648通り
# レンジ相場に特化することで質の高い最適化を実現

# 最適化目的関数設定（レンジ相場用に調整）
OBJECTIVES_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "sortino_ratio", "weight": 0.8},
    {"name": "expectancy", "weight": 0.6},
    {"name": "win_rate", "weight": 0.4}  # レンジ相場では勝率も重要
]
