"""
debug_open_entry_issue.py - Open価格エントリーでエントリーが0件になる原因調査

調査目的:
Open価格エントリーに変更後、エントリーが0件になった原因を特定する

調査項目:
1. 2024-07-26のデータを確認
2. ギャップダウン判定がOpen価格で正しく動作しているか
3. RSI条件を満たしているか
4. トレンドフィルターの影響

Author: Backtest Project Team
Created: 2025-10-28
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# プロジェクトパス設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from indicators.basic_indicators import calculate_rsi
from indicators.unified_trend_detector import detect_unified_trend

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """メイン実行"""
    logger.info("=" * 80)
    logger.info("Open価格エントリー問題の調査")
    logger.info("=" * 80)
    
    # データ取得
    data_feed = YFinanceDataFeed()
    stock_data = data_feed.get_stock_data(
        ticker="8306.T",
        start_date="2024-07-01",
        end_date="2024-08-10"
    )
    
    # RSI計算
    stock_data['RSI'] = calculate_rsi(stock_data['Adj Close'], period=14)
    
    logger.info(f"データ期間: {stock_data.index[0]} ~ {stock_data.index[-1]}")
    logger.info("")
    
    # 2024-07-26前後のデータを確認
    target_date = pd.Timestamp('2024-07-26', tz='Asia/Tokyo')
    
    # 前後3日間のデータを表示
    start_idx = stock_data.index.get_loc(target_date) - 3
    end_idx = stock_data.index.get_loc(target_date) + 3
    period_data = stock_data.iloc[start_idx:end_idx+1]
    
    logger.info("2024-07-26前後のデータ:")
    logger.info("-" * 80)
    logger.info(f"{'日付':<12} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Adj Close':<10} {'RSI':<10}")
    logger.info("-" * 80)
    
    for date, row in period_data.iterrows():
        logger.info(f"{date.strftime('%Y-%m-%d'):<12} "
                   f"{row['Open']:<10.2f} "
                   f"{row['High']:<10.2f} "
                   f"{row['Low']:<10.2f} "
                   f"{row['Close']:<10.2f} "
                   f"{row['Adj Close']:<10.2f} "
                   f"{row['RSI']:<10.2f}")
    
    logger.info("")
    
    # 2024-07-26のギャップダウン判定を確認
    logger.info("2024-07-26のエントリー条件チェック:")
    logger.info("-" * 80)
    
    target_idx = stock_data.index.get_loc(target_date)
    target_row = stock_data.iloc[target_idx]
    prev_row = stock_data.iloc[target_idx - 1]
    
    # 各条件をチェック
    rsi = target_row['RSI']
    rsi_oversold = 30
    
    # Open価格でのギャップダウン判定
    open_price = target_row['Open']
    prev_close = prev_row['Adj Close']
    gap_threshold = 0.02
    gap_down_open = open_price < prev_close * (1.0 - gap_threshold)
    gap_pct_open = ((open_price - prev_close) / prev_close) * 100
    
    # 終値でのギャップダウン判定（参考）
    current_close = target_row['Adj Close']
    gap_down_close = current_close < prev_close * (1.0 - gap_threshold)
    gap_pct_close = ((current_close - prev_close) / prev_close) * 100
    
    logger.info(f"RSI: {rsi:.2f} (閾値: {rsi_oversold})")
    logger.info(f"RSI条件: {'OK' if rsi <= rsi_oversold else 'NG'}")
    logger.info("")
    
    logger.info("ギャップダウン判定（Open価格ベース）:")
    logger.info(f"  前日終値: {prev_close:.2f} JPY")
    logger.info(f"  当日Open: {open_price:.2f} JPY")
    logger.info(f"  ギャップ: {gap_pct_open:+.2f}%")
    logger.info(f"  閾値: -{gap_threshold * 100}%")
    logger.info(f"  判定: {'OK（ギャップダウン）' if gap_down_open else 'NG'}")
    logger.info("")
    
    logger.info("ギャップダウン判定（終値ベース・参考）:")
    logger.info(f"  前日終値: {prev_close:.2f} JPY")
    logger.info(f"  当日終値: {current_close:.2f} JPY")
    logger.info(f"  ギャップ: {gap_pct_close:+.2f}%")
    logger.info(f"  閾値: -{gap_threshold * 100}%")
    logger.info(f"  判定: {'OK（ギャップダウン）' if gap_down_close else 'NG'}")
    logger.info("")
    
    # トレンド判定
    trend = detect_unified_trend(
        stock_data.iloc[:target_idx + 1],
        price_column="Adj Close",
        strategy="contrarian_strategy",
        method="combined"
    )
    
    logger.info(f"トレンド判定: {trend}")
    logger.info(f"トレンドフィルター条件: {'OK（range-bound）' if trend == 'range-bound' else 'NG'}")
    logger.info("")
    
    # 総合判定
    logger.info("=" * 80)
    logger.info("総合判定:")
    logger.info("=" * 80)
    
    entry_ok = (rsi <= rsi_oversold) and gap_down_open and (trend == "range-bound")
    
    logger.info(f"RSI条件: {'OK' if rsi <= rsi_oversold else 'NG'}")
    logger.info(f"ギャップダウン条件（Open）: {'OK' if gap_down_open else 'NG'}")
    logger.info(f"トレンド条件: {'OK' if trend == 'range-bound' else 'NG'}")
    logger.info("")
    logger.info(f"エントリー判定: {'OK - エントリー可能' if entry_ok else 'NG - エントリー不可'}")
    logger.info("")
    
    # 問題の特定
    if not entry_ok:
        logger.info("エントリーできない理由:")
        if rsi > rsi_oversold:
            logger.info(f"  - RSIが過売り閾値を超えています（{rsi:.2f} > {rsi_oversold}）")
        if not gap_down_open:
            logger.info(f"  - Open価格でのギャップダウンが検出されませんでした（{gap_pct_open:+.2f}% > -{gap_threshold * 100}%）")
        if trend != "range-bound":
            logger.info(f"  - トレンドがレンジ相場ではありません（{trend}）")


if __name__ == "__main__":
    main()
