"""
BreakoutStrategy の最適化設定
"""

# 最適化対象パラメータとその探索範囲
PARAM_GRID = {
    "volume_threshold": [1.1, 1.2, 1.3, 1.4, 1.5],
    "breakout_buffer": [0.005, 0.01, 0.015, 0.02],  # ブレイクアウト判定閾値
    "take_profit": [0.02, 0.03, 0.04, 0.05, 0.07],
    "trailing_stop": [0.01, 0.02, 0.03, 0.04],      # トレーリングストップ
    "look_back": [1, 2, 3]
}

# パラメータの説明
PARAM_DESCRIPTIONS = {
    "volume_threshold": "出来高増加率の閾値 - 前日比でこの値以上の出来高があると判断",
    "breakout_buffer": "ブレイクアウト判定閾値 - 前日高値からこの割合上昇したらブレイクアウトと判断",
    "take_profit": "利益確定率 - エントリー価格からこの割合上昇したら利確",
    "trailing_stop": "トレーリングストップ - 高値からこの割合下落したら損切り",
    "look_back": "ブレイクアウト判定に使用する過去の日数"
}

# 最適化の目的関数設定
OBJECTIVE_CONFIG = [
    {"name": "sharpe_ratio", "weight": 1.0},
    {"name": "risk_adjusted_return", "weight": 0.5}
]

# 交差検証設定
CV_SETTINGS = {
    "train_size": 252,  # 1年
    "test_size": 63     # 3ヶ月
}