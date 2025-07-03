#!/usr/bin/env python3
"""
正しいカラム名で極限緩和パラメータでのVWAPBounceStrategy動作テスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.VWAP_Bounce import VWAPBounceStrategy
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_with_corrected_columns():
    """正しいカラム名で極限緩和パラメータテスト"""
    print("=== 正しいカラム名で極限緩和パラメータテスト ===")
    
    # キャッシュデータの読み込み（正しいカラム名を設定）
    cache_file = "data_cache/^N225_2023-01-01_2025-01-01_1d.csv"
    
    try:
        # データ読み込み（skiprows=2でヘッダー行をスキップ、カラム名を手動設定）
        data = pd.read_csv(cache_file, skiprows=2)
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
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
            price_column="Close",  # Closeカラムを使用
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
        
        if results.get('entry_count', 0) > 0:
            print(f"\n取引が発生しました！")
            if 'signals' in results:
                signals = results['signals']
                if hasattr(signals, 'sum') and callable(signals.sum):
                    entry_signals = signals[signals > 0].sum() if hasattr(signals, '__getitem__') else 0
                    print(f"エントリーシグナル合計: {entry_signals}")
        else:
            print(f"\n取引が発生しませんでした。条件をさらに緩和する必要があります。")
        
    except Exception as e:
        print(f"戦略実行エラー: {e}")
        import traceback
        traceback.print_exc()

def test_step_by_step_relaxation():
    """段階的に条件を緩和してテスト"""
    print("\n=== 段階的条件緩和テスト ===")
    
    cache_file = "data_cache/^N225_2023-01-01_2025-01-01_1d.csv"
    
    try:
        data = pd.read_csv(cache_file, skiprows=2)
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # 段階的緩和パラメータ
    relaxation_steps = [
        {
            "name": "Step1: 超極限緩和",
            "params": {
                "vwap_lower_threshold": 0.95,        # VWAP-5%まで
                "vwap_upper_threshold": 1.05,        # VWAP+5%まで
                "volume_increase_threshold": 0.5,    # 出来高半減でもOK
                "bullish_candle_min_pct": 0.0,       # 陽線条件完全無効
                "stop_loss": 0.05,                   # 5%損切り
                "take_profit": 0.02,                 # 2%利確
                "trend_filter_enabled": False,
                "max_hold_days": 20,
                "cool_down_period": 0
            }
        },
        {
            "name": "Step2: さらに極限緩和",
            "params": {
                "vwap_lower_threshold": 0.9,         # VWAP-10%まで
                "vwap_upper_threshold": 1.1,         # VWAP+10%まで
                "volume_increase_threshold": 0.1,    # 出来高90%減でもOK
                "bullish_candle_min_pct": 0.0,       # 陽線条件完全無効
                "stop_loss": 0.1,                    # 10%損切り
                "take_profit": 0.01,                 # 1%利確
                "trend_filter_enabled": False,
                "max_hold_days": 30,
                "cool_down_period": 0
            }
        }
    ]
    
    for step in relaxation_steps:
        print(f"\n--- {step['name']} ---")
        
        params = step['params'].copy()
        params.update({
            "trailing_stop_pct": 0.02,
            "allowed_trends": ["range-bound", "uptrend", "downtrend"],
            "partial_exit_enabled": False,
            "partial_exit_portion": 0.5,
            "volatility_filter_enabled": False
        })
        
        try:
            strategy = VWAPBounceStrategy(
                data.copy(), 
                params,
                price_column="Close",
                volume_column="Volume"
            )
            results = strategy.backtest()
            
            entry_count = results.get('entry_count', 0)
            print(f"エントリー回数: {entry_count}")
            
            if entry_count > 0:
                print(f"総損益: {results.get('total_pnl', 0):.4f}")
                print(f"勝率: {results.get('win_rate', 0):.2%}")
                print(f"期待値: {results.get('expectancy', 0):.4f}")
                print("★ 取引発生！")
                break  # 取引が発生した段階で終了
            else:
                print("取引なし。さらに緩和が必要。")
                
        except Exception as e:
            print(f"エラー: {e}")

if __name__ == "__main__":
    test_with_corrected_columns()
    test_step_by_step_relaxation()
