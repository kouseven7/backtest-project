#!/usr/bin/env python3
"""
極限緩和パラメータでのVWAPBounceStrategy動作テスト
エントリー条件を事実上無効化し、取引回数の増加を確認
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.VWAP_Bounce import VWAPBounceStrategy
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_extreme_relaxed_params():
    """極限緩和パラメータでの動作テスト"""
    print("=== 極限緩和パラメータでのVWAPBounceStrategy動作テスト ===")
    
    # テスト用データ取得（1年分）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    try:
        data = yf.download("SPY", start=start_date, end=end_date)
        print(f"データ取得完了: {len(data)}日分")
    except Exception as e:
        print(f"データ取得エラー: {e}")
        return
    
    # 極限緩和パラメータセット
    extreme_params = {
        "vwap_lower_threshold": 0.98,           # VWAP-2%まで許可
        "vwap_upper_threshold": 1.02,           # VWAP+2%まで許可
        "volume_increase_threshold": 0.95,      # 出来高減少でもOK
        "bullish_candle_min_pct": 0.0,          # 陽線条件なし
        "stop_loss": 0.015,                     # 1.5%損切り
        "take_profit": 0.025,                   # 2.5%利確
        "trailing_stop_pct": 0.01,              # 1%トレーリング
        "trend_filter_enabled": False,          # トレンドフィルター無効
        "allowed_trends": ["range-bound", "uptrend", "downtrend"],
        "max_hold_days": 10,                    # 10日保有
        "cool_down_period": 0,                  # クールダウンなし
        "partial_exit_enabled": False,          # 部分利確無効
        "partial_exit_portion": 0.5,
        "volatility_filter_enabled": False      # ボラティリティフィルター無効
    }
    
    print("\n極限緩和パラメータ:")
    for key, value in extreme_params.items():
        print(f"  {key}: {value}")
    
    # 戦略実行
    try:
        strategy = VWAPBounceStrategy(data.copy(), extreme_params)
        results = strategy.backtest()
        
        print(f"\n=== 極限緩和パラメータ実行結果 ===")
        print(f"エントリー回数: {results.get('entry_count', 0)}")
        print(f"イグジット回数: {results.get('exit_count', 0)}")
        print(f"総損益: {results.get('total_pnl', 0):.4f}")
        print(f"勝率: {results.get('win_rate', 0):.2%}")
        print(f"期待値: {results.get('expectancy', 0):.4f}")
        print(f"最大ドローダウン: {results.get('max_drawdown', 0):.2%}")
        
    except Exception as e:
        print(f"戦略実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extreme_relaxed_params()
