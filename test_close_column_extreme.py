#!/usr/bin/env python3
"""
Close カラムを使った極限緩和パラメータでのVWAPBounceStrategy動作テスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.VWAP_Bounce import VWAPBounceStrategy
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_with_close_column():
    """Closeカラムを使った極限緩和パラメータでの動作テスト"""
    print("=== Closeカラムを使った極限緩和パラメータテスト ===")
    
    # キャッシュデータの読み込み
    cache_file = "data_cache/^N225_2023-01-01_2025-01-01_1d.csv"
    
    try:
        # データ読み込み（最初の2行をスキップして、カラム構造を修正）
        data = pd.read_csv(cache_file, skiprows=2, index_col=0, parse_dates=True)
        print(f"データ読み込み完了: {cache_file}")
        print(f"データ形状: {data.shape}")
        print(f"カラム: {data.columns.tolist()}")
        print(f"期間: {data.index[0]} ～ {data.index[-1]}")
        
        # データの最初の数行を確認
        print("\nデータサンプル:")
        print(data.head())
        
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
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
    
    # 戦略実行（price_column="Close"を指定）
    try:
        strategy = VWAPBounceStrategy(
            data.copy(), 
            extreme_params, 
            price_column="Close",  # Adj CloseではなくCloseを使用
            volume_column="Volume"
        )
        results = strategy.backtest()
        
        print(f"\n=== 極限緩和パラメータ実行結果 ===")
        print(f"エントリー回数: {results.get('entry_count', 0)}")
        print(f"イグジット回数: {results.get('exit_count', 0)}")
        print(f"総損益: {results.get('total_pnl', 0):.4f}")
        print(f"勝率: {results.get('win_rate', 0):.2%}")
        print(f"期待値: {results.get('expectancy', 0):.4f}")
        print(f"最大ドローダウン: {results.get('max_drawdown', 0):.2%}")
        print(f"シャープレシオ: {results.get('sharpe_ratio', 0):.4f}")
        print(f"ソルティノレシオ: {results.get('sortino_ratio', 0):.4f}")
        
        # 取引詳細があれば表示
        if 'trades' in results and hasattr(results['trades'], '__len__') and len(results['trades']) > 0:
            trades = results['trades']
            print(f"\n取引詳細（最初の3件）:")
            for i, trade in enumerate(trades[:3]):
                print(f"  取引{i+1}: {trade}")
        
    except Exception as e:
        print(f"戦略実行エラー: {e}")
        import traceback
        traceback.print_exc()

def test_multiple_relaxed_params_with_close():
    """Closeカラムを使った段階的緩和パラメータでのテスト"""
    print("\n=== Closeカラムを使った段階的緩和パラメータテスト ===")
    
    # キャッシュデータの読み込み
    cache_file = "data_cache/^N225_2023-01-01_2025-01-01_1d.csv"
    
    try:
        data = pd.read_csv(cache_file, skiprows=2, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # 段階的パラメータセット
    param_sets = [
        {
            "name": "軽度緩和",
            "params": {
                "vwap_lower_threshold": 0.995,
                "vwap_upper_threshold": 1.005,
                "volume_increase_threshold": 1.0,
                "bullish_candle_min_pct": 0.001,
                "stop_loss": 0.01,
                "take_profit": 0.02,
            }
        },
        {
            "name": "中度緩和",
            "params": {
                "vwap_lower_threshold": 0.99,
                "vwap_upper_threshold": 1.01,
                "volume_increase_threshold": 1.0,
                "bullish_candle_min_pct": 0.0,
                "stop_loss": 0.012,
                "take_profit": 0.025,
            }
        },
        {
            "name": "極限緩和",
            "params": {
                "vwap_lower_threshold": 0.98,
                "vwap_upper_threshold": 1.02,
                "volume_increase_threshold": 0.95,
                "bullish_candle_min_pct": 0.0,
                "stop_loss": 0.015,
                "take_profit": 0.025,
            }
        }
    ]
    
    for param_set in param_sets:
        print(f"\n--- {param_set['name']} ---")
        
        # 共通パラメータを追加
        params = param_set['params'].copy()
        params.update({
            "trailing_stop_pct": 0.01,
            "trend_filter_enabled": False,
            "allowed_trends": ["range-bound", "uptrend", "downtrend"],
            "max_hold_days": 10,
            "cool_down_period": 0,
            "partial_exit_enabled": False,
            "partial_exit_portion": 0.5,
            "volatility_filter_enabled": False
        })
        
        try:
            strategy = VWAPBounceStrategy(
                data.copy(), 
                params,
                price_column="Close",  # Closeカラムを使用
                volume_column="Volume"
            )
            results = strategy.backtest()
            
            print(f"エントリー回数: {results.get('entry_count', 0)}")
            print(f"総損益: {results.get('total_pnl', 0):.4f}")
            print(f"勝率: {results.get('win_rate', 0):.2%}")
            print(f"期待値: {results.get('expectancy', 0):.4f}")
            
        except Exception as e:
            print(f"エラー: {e}")

if __name__ == "__main__":
    test_with_close_column()
    test_multiple_relaxed_params_with_close()
