"""
GC戦略デバッグテスト - 最小構成

目的:
- 1エントリー/1イグジットのシンプルなケースで検証
- BaseStrategy.backtest()のロジック検証
- イグジットシグナルが1回だけ記録されるか確認

主な機能:
- 最小限のテストデータ生成（50日間）
- GCStrategy実行
- Entry/Exit シグナル数の検証
- デバッグログ出力

Author: Backtest Project Team
Created: 2025-10-24
Last Modified: 2025-10-24
"""

import sys
import os
from pathlib import Path

# デバッグログを有効化
os.environ['DEBUG_BACKTEST'] = '1'

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging
from strategies.gc_strategy_signal import GCStrategy

# ログ設定（全てのロガーをDEBUGレベルに）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# GCStrategyとBaseStrategyのロガーを明示的にDEBUGに設定
logging.getLogger('GCStrategy').setLevel(logging.DEBUG)
logging.getLogger('BaseStrategy').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def create_simple_test_data():
    """
    単純なテストデータを生成
    - 50日間のデータ
    - 明確な1回のゴールデンクロスとデッドクロス
    """
    dates = pd.date_range(start="2024-01-01", periods=50, freq='D')
    
    # 価格データ: 下降 → 上昇 → 下降でクロス発生
    prices = []
    for i in range(50):
        if i < 15:
            # 下降トレンド（デッドクロス状態）
            price = 1100.0 - i * 10
        elif i < 35:
            # 急上昇（ゴールデンクロス発生）
            price = 950.0 + (i - 15) * 15
        else:
            # 急下降（デッドクロス発生）
            price = 1250.0 - (i - 35) * 20
            
        prices.append(price)
    
    df = pd.DataFrame({
        'Adj Close': prices,
        'Close': prices,
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Volume': [1000000] * 50
    }, index=dates)
    
    return df


def main():
    """メイン関数"""
    logger.info("=" * 80)
    logger.info("GC戦略 デバッグテスト - 最小構成")
    logger.info("=" * 80)
    
    # テストデータ生成
    logger.info("\n[STEP 1] テストデータ生成")
    data = create_simple_test_data()
    logger.info(f"  データ行数: {len(data)}")
    logger.info(f"  期間: {data.index[0]} to {data.index[-1]}")
    
    # SMAを計算して確認
    data['SMA_5'] = data['Adj Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Adj Close'].rolling(window=10).mean()
    
    logger.info("\n  価格とSMAの推移:")
    for i in range(15, 35):  # 重要な期間のみ表示
        price = data['Adj Close'].iloc[i]
        sma5 = data['SMA_5'].iloc[i]
        sma10 = data['SMA_10'].iloc[i]
        date = data.index[i]
        
        cross = ""
        if i > 0:
            prev_sma5 = data['SMA_5'].iloc[i-1]
            prev_sma10 = data['SMA_10'].iloc[i-1]
            if prev_sma5 <= prev_sma10 and sma5 > sma10:
                cross = " <-- GOLDEN CROSS"
            elif prev_sma5 >= prev_sma10 and sma5 < sma10:
                cross = " <-- DEATH CROSS"
        
        logger.info(f"  {date.date()}: Price={price:.2f}, SMA5={sma5:.2f}, SMA10={sma10:.2f}{cross}")
    
    # GC戦略初期化
    logger.info("\n[STEP 2] GC戦略初期化")
    params = {
        "short_window": 5,
        "long_window": 10,
        "take_profit_pct": 0.10,  # 10%利益確定
        "stop_loss_pct": 0.05,    # 5%損切り
        "trailing_stop_pct": 0.05,
        "max_hold_days": 30,
        "exit_on_death_cross": True,
        "trend_filter_enabled": False  # トレンドフィルター無効
    }
    
    strategy = GCStrategy(data, params)
    logger.info(f"  パラメータ: {params}")
    
    # バックテスト実行
    logger.info("\n[STEP 3] バックテスト実行")
    logger.info("=" * 80)
    result = strategy.backtest()
    logger.info("=" * 80)
    
    # 結果検証
    logger.info("\n[STEP 4] 結果検証")
    entry_count = (result['Entry_Signal'] == 1).sum()
    exit_count = (result['Exit_Signal'] == -1).sum()
    
    logger.info(f"  Entry_Signal == 1: {entry_count} 回")
    logger.info(f"  Exit_Signal == -1: {exit_count} 回")
    
    # Entry_Signalの日付を表示
    entry_dates = result[result['Entry_Signal'] == 1].index.tolist()
    logger.info(f"  エントリー日: {entry_dates}")
    
    # Exit_Signalの日付を表示
    exit_dates = result[result['Exit_Signal'] == -1].index.tolist()
    logger.info(f"  イグジット日: {exit_dates}")
    
    # テスト判定
    logger.info("\n[TEST RESULT]")
    if entry_count == exit_count:
        logger.info(f"  PASS: エントリー({entry_count}) == イグジット({exit_count})")
        return True
    else:
        logger.error(f"  FAIL: エントリー({entry_count}) != イグジット({exit_count})")
        logger.error(f"  差分: {abs(entry_count - exit_count)} 回")
        
        # 詳細情報出力
        logger.info("\n[詳細情報]")
        logger.info(f"  Entry_Signalカラム:")
        logger.info(f"{result['Entry_Signal'].value_counts()}")
        logger.info(f"  Exit_Signalカラム:")
        logger.info(f"{result['Exit_Signal'].value_counts()}")
        logger.info(f"  Positionカラム:")
        logger.info(f"{result['Position'].value_counts()}")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
