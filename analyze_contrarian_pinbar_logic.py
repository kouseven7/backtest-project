"""
analyze_contrarian_pinbar_logic.py - ContrarianStrategy ピンバー検出ロジック詳細分析

調査目的:
1. ピンバー判定が52回中52回発動している異常を調査
2. pin_bar_ratio=2.0 の条件が適切かを検証
3. 実データでのHigh/Low/Closeの関係を数値で確認

調査方法:
- 全エントリー日のローソク足データを取得
- ピンバー判定条件を段階的に検証
- 上ヒゲ、下ヒゲ、実体の比率を計算

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
from strategies.contrarian_strategy import ContrarianStrategy

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_pinbar_detection():
    """ピンバー検出ロジックの詳細分析"""
    
    logger.info("=" * 80)
    logger.info("ピンバー検出ロジック詳細分析 開始")
    logger.info("=" * 80)
    
    # データ取得
    logger.info("\n[STEP 1] データ取得")
    data_feed = YFinanceDataFeed()
    stock_data = data_feed.get_stock_data(
        ticker="8306.T",
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    logger.info(f"取得データ: {len(stock_data)} 行")
    
    # 戦略初期化とバックテスト実行
    logger.info("\n[STEP 2] 戦略実行")
    strategy = ContrarianStrategy(data=stock_data, price_column="Close")
    result = strategy.backtest()
    
    # エントリー日を取得
    entry_dates = result[result['Entry_Signal'] == 1].index
    logger.info(f"総エントリー数: {len(entry_dates)}")
    
    # ピンバー検出ロジックの詳細分析
    logger.info("\n[STEP 3] ピンバー判定条件の詳細分析")
    logger.info("=" * 80)
    logger.info("ピンバー判定式: (High - Close) > pin_bar_ratio * (Close - Low)")
    logger.info(f"pin_bar_ratio: {strategy.params['pin_bar_ratio']}")
    logger.info("=" * 80)
    
    pinbar_count = 0
    non_pinbar_count = 0
    pinbar_details = []
    
    for i, date in enumerate(entry_dates, 1):
        idx = result.index.get_loc(date)
        
        if idx > 0:
            high = result['High'].iloc[idx]
            low = result['Low'].iloc[idx]
            close = result['Close'].iloc[idx]
            prev_close = result['Close'].iloc[idx - 1]
            rsi = result['RSI'].iloc[idx] if 'RSI' in result.columns else None
            
            # ピンバー判定計算
            upper_shadow = high - close
            lower_shadow = close - low
            body = abs(close - prev_close)
            
            # 実際の判定条件
            pin_bar_ratio = strategy.params["pin_bar_ratio"]
            is_pinbar = upper_shadow > pin_bar_ratio * lower_shadow
            
            # 比率計算
            upper_to_lower_ratio = upper_shadow / lower_shadow if lower_shadow > 0 else float('inf')
            upper_to_body_ratio = upper_shadow / body if body > 0 else float('inf')
            
            # ギャップダウン判定
            gap_threshold = strategy.params["gap_threshold"]
            is_gap_down = close < prev_close * (1.0 - gap_threshold)
            
            # RSI過売り判定
            is_rsi_oversold = rsi <= strategy.params["rsi_oversold"] if rsi is not None else False
            
            detail = {
                "no": i,
                "date": date.strftime('%Y-%m-%d'),
                "high": high,
                "low": low,
                "close": close,
                "prev_close": prev_close,
                "upper_shadow": upper_shadow,
                "lower_shadow": lower_shadow,
                "body": body,
                "upper_to_lower_ratio": upper_to_lower_ratio,
                "upper_to_body_ratio": upper_to_body_ratio,
                "is_pinbar": is_pinbar,
                "is_gap_down": is_gap_down,
                "is_rsi_oversold": is_rsi_oversold,
                "rsi": rsi
            }
            pinbar_details.append(detail)
            
            if is_pinbar:
                pinbar_count += 1
            else:
                non_pinbar_count += 1
            
            # 詳細ログ出力（最初の10件と非ピンバーの全件）
            if i <= 10 or not is_pinbar:
                logger.info(f"\nエントリー {i}: {date.strftime('%Y-%m-%d')}")
                logger.info(f"  High: {high:.2f}, Low: {low:.2f}, Close: {close:.2f}")
                logger.info(f"  上ヒゲ: {upper_shadow:.2f} 円")
                logger.info(f"  下ヒゲ: {lower_shadow:.2f} 円")
                logger.info(f"  実体: {body:.2f} 円")
                logger.info(f"  上ヒゲ/下ヒゲ比: {upper_to_lower_ratio:.2f}")
                logger.info(f"  上ヒゲ/実体比: {upper_to_body_ratio:.2f}")
                logger.info(f"  ピンバー判定: {'はい' if is_pinbar else 'いいえ'}")
                logger.info(f"  ギャップダウン: {'はい' if is_gap_down else 'いいえ'}")
                logger.info(f"  RSI過売り: {'はい' if is_rsi_oversold else 'いいえ'} (RSI={rsi:.2f})" if rsi is not None else "  RSI過売り: N/A")
    
    # 統計サマリー
    logger.info("\n" + "=" * 80)
    logger.info("統計サマリー")
    logger.info("=" * 80)
    logger.info(f"総エントリー数: {len(entry_dates)}")
    logger.info(f"ピンバー検出: {pinbar_count} 回 ({pinbar_count/len(entry_dates)*100:.1f}%)")
    logger.info(f"非ピンバー: {non_pinbar_count} 回 ({non_pinbar_count/len(entry_dates)*100:.1f}%)")
    
    # 比率の統計
    df = pd.DataFrame(pinbar_details)
    logger.info("\n上ヒゲ/下ヒゲ比の統計:")
    logger.info(f"  平均: {df['upper_to_lower_ratio'].mean():.2f}")
    logger.info(f"  中央値: {df['upper_to_lower_ratio'].median():.2f}")
    logger.info(f"  最小: {df['upper_to_lower_ratio'].min():.2f}")
    logger.info(f"  最大: {df['upper_to_lower_ratio'].max():.2f}")
    logger.info(f"  pin_bar_ratio閾値: {pin_bar_ratio}")
    logger.info(f"  閾値以上: {(df['upper_to_lower_ratio'] > pin_bar_ratio).sum()} 件")
    
    # エントリー条件の組み合わせ分析
    logger.info("\nエントリー条件の組み合わせ分析:")
    logger.info(f"  ピンバーのみ: {df['is_pinbar'].sum()} 件")
    logger.info(f"  RSI過売りのみ: {df['is_rsi_oversold'].sum()} 件")
    logger.info(f"  ギャップダウンのみ: {df['is_gap_down'].sum()} 件")
    logger.info(f"  RSI過売り+ギャップダウン: {((df['is_rsi_oversold']) & (df['is_gap_down'])).sum()} 件")
    
    # CSV出力
    output_path = Path("tests/results/contrarian_pinbar_analysis.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n詳細データCSV出力: {output_path}")
    
    # 問題の特定
    logger.info("\n" + "=" * 80)
    logger.info("問題の特定")
    logger.info("=" * 80)
    
    # 下ヒゲがゼロまたは極小のケースをチェック
    zero_lower_shadow = (df['lower_shadow'] == 0).sum()
    tiny_lower_shadow = (df['lower_shadow'] < 1.0).sum()
    
    logger.info(f"下ヒゲが0円: {zero_lower_shadow} 件")
    logger.info(f"下ヒゲが1円未満: {tiny_lower_shadow} 件")
    
    if zero_lower_shadow > 0 or tiny_lower_shadow > len(entry_dates) * 0.5:
        logger.info("\n[重大な問題発見]")
        logger.info("下ヒゲが極端に小さいケースが多数存在")
        logger.info("これにより上ヒゲ/下ヒゲ比が異常に高くなり、")
        logger.info("ほぼ全てのローソク足がピンバーと判定されている可能性があります")
    
    # ピンバー判定条件の問題点
    logger.info("\n[ピンバー判定条件の問題点]")
    logger.info("現在の条件: (High - Close) > 2.0 * (Close - Low)")
    logger.info("\n問題:")
    logger.info("1. Close が Low に近い場合（下ヒゲが小さい）、")
    logger.info("   わずかな上ヒゲでも条件を満たしてしまう")
    logger.info("2. 実体の大きさが考慮されていない")
    logger.info("3. 陰線と陽線の区別がない")
    
    logger.info("\n推奨される改善策:")
    logger.info("1. 実体に対する上ヒゲの比率も条件に加える")
    logger.info("2. 下ヒゲの最小値を設定する")
    logger.info("3. 全体のレンジ（High - Low）に対する比率で判定する")
    
    logger.info("\n" + "=" * 80)
    logger.info("分析完了")
    logger.info("=" * 80)
    
    return df


if __name__ == "__main__":
    result_df = analyze_pinbar_detection()
