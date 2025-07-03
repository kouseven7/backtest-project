"""
Module: Test All Strategies Trend Integration
File: test_all_strategies_trend.py
Description: 
  全戦略に対して、統一トレンド判定インターフェースの実装を確認するテストスクリプト
  各戦略のinitialize_strategyメソッドとgenerate_entry_signalメソッドを実行し、
  統一トレンド判定が正常に動作することを確認します。

Author: imega
Created: 2025-07-03
Modified: 2025-07-03

Dependencies:
  - pandas
  - numpy
  - strategies.*
  - indicators.unified_trend_detector
  - data_fetcher
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# プロジェクトのルートパスを追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# 各戦略をインポート
from strategies.VWAP_Bounce import VWAPBounceStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy
from strategies.Breakout import BreakoutStrategy
from strategies.Opening_Gap import OpeningGapStrategy
from strategies.contrarian_strategy import ContrarianStrategy
from strategies.gc_strategy_signal import GCStrategy

# データ取得用
from data_fetcher import fetch_stock_data

import logging

# ロガー設定
logging.basicConfig(level=logging.INFO, 
                   format='[%(levelname)s] %(asctime)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data():
    """異なるトレンド状況のテストデータを生成"""
    # 上昇トレンドデータ
    dates1 = pd.date_range(start="2022-01-01", periods=50, freq='B')
    uptrend_data = pd.DataFrame({
        'High': [100 + i + np.random.random() * 2 for i in range(50)],
        'Low': [95 + i - np.random.random() * 2 for i in range(50)],
        'Open': [97 + i for i in range(50)],
        'Close': [98 + i for i in range(50)],
        'Adj Close': [98 + i for i in range(50)],
        'Volume': np.random.randint(1000, 5000, 50)
    }, index=dates1)
    uptrend_data['VWAP'] = (uptrend_data['High'] + uptrend_data['Low'] + uptrend_data['Close']) / 3
    
    # 下降トレンドデータ
    dates2 = pd.date_range(start="2022-03-01", periods=50, freq='B')
    downtrend_data = pd.DataFrame({
        'High': [150 - i + np.random.random() * 2 for i in range(50)],
        'Low': [145 - i - np.random.random() * 2 for i in range(50)],
        'Open': [148 - i for i in range(50)],
        'Close': [147 - i for i in range(50)],
        'Adj Close': [147 - i for i in range(50)],
        'Volume': np.random.randint(1000, 5000, 50)
    }, index=dates2)
    downtrend_data['VWAP'] = (downtrend_data['High'] + downtrend_data['Low'] + downtrend_data['Close']) / 3
    
    # レンジ相場データ
    dates3 = pd.date_range(start="2022-05-01", periods=50, freq='B')
    range_data = pd.DataFrame({
        'High': [100 + np.sin(i/5) * 5 + np.random.random() * 2 for i in range(50)],
        'Low': [95 + np.sin(i/5) * 5 - np.random.random() * 2 for i in range(50)],
        'Open': [97 + np.sin(i/5) * 5 for i in range(50)],
        'Close': [98 + np.sin(i/5) * 5 for i in range(50)],
        'Adj Close': [98 + np.sin(i/5) * 5 for i in range(50)],
        'Volume': np.random.randint(1000, 5000, 50)
    }, index=dates3)
    range_data['VWAP'] = (range_data['High'] + range_data['Low'] + range_data['Close']) / 3
    
    return {
        "uptrend": uptrend_data,
        "downtrend": downtrend_data,
        "range-bound": range_data
    }

def test_strategy(strategy_class, data_dict, strategy_name):
    """指定された戦略クラスでテストを実行"""
    logger.info(f"======= {strategy_name} のテスト開始 =======")
    
    results = {}
    
    for trend_type, data in data_dict.items():
        logger.info(f"\n--- {trend_type}データでのテスト ---")
        
        # 戦略インスタンス作成
        strategy = strategy_class(data, params={
            "trend_filter_enabled": True  # トレンドフィルタを有効化
        })
        
        # 初期化
        try:
            strategy.initialize_strategy()
            logger.info("✓ initialize_strategy() が正常に実行されました")
        except Exception as e:
            logger.error(f"✗ initialize_strategy() でエラー発生: {e}")
        
        # エントリーシグナル生成テスト
        entry_signals = []
        for idx in range(min(30, len(data))):
            try:
                signal = strategy.generate_entry_signal(idx)
                entry_signals.append(signal)
            except Exception as e:
                logger.error(f"✗ idx={idx}のエントリーシグナル生成でエラー発生: {e}")
                break
        
        # エントリーシグナルの統計
        if entry_signals:
            signal_count = sum(1 for s in entry_signals if s != 0)
            logger.info(f"エントリーシグナル発生数: {signal_count}/{len(entry_signals)}")
        
        # 結果を保存
        results[trend_type] = {
            "signals": entry_signals,
            "signal_count": sum(1 for s in entry_signals if s != 0) if entry_signals else 0
        }
    
    logger.info(f"======= {strategy_name} のテスト終了 =======\n")
    return results

def main():
    """全戦略のテスト実行"""
    logger.info("=== 全戦略の統一トレンド判定テスト ===")
    
    # テストデータ生成
    test_data = generate_test_data()
    
    # 各戦略クラスとその名前のマッピング
    strategy_classes = [
        (VWAPBounceStrategy, "VWAP Bounce"),
        (MomentumInvestingStrategy, "Momentum Investing"),
        (BreakoutStrategy, "Breakout"),
        (OpeningGapStrategy, "Opening Gap"),
        (ContrarianStrategy, "Contrarian"),
        (GCStrategy, "Golden Cross")
    ]
    
    all_results = {}
    
    # 各戦略をテスト
    for strategy_class, name in strategy_classes:
        try:
            results = test_strategy(strategy_class, test_data, name)
            all_results[name] = results
        except Exception as e:
            logger.error(f"{name} 戦略のテスト中に未処理の例外が発生: {e}")
    
    # 結果の要約
    logger.info("\n=== 結果の要約 ===")
    for strategy_name, results in all_results.items():
        logger.info(f"\n{strategy_name}:")
        for trend_type, data in results.items():
            logger.info(f"  {trend_type}: {data['signal_count']} シグナル")

if __name__ == "__main__":
    main()
