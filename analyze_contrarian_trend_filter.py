"""
analyze_contrarian_trend_filter.py - ContrarianStrategy トレンドフィルター動作検証

調査目的:
1. allowed_trends=["range-bound"] の制約が機能しているか
2. 全エントリー時のトレンド判定結果を確認
3. トレンド判定の信頼度を検証

調査方法:
- 各エントリー日のトレンド判定結果を取得
- range-bound以外のトレンドでエントリーしているケースを特定
- トレンド判定の信頼度を数値化

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
from indicators.unified_trend_detector import UnifiedTrendDetector, detect_unified_trend

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_trend_filter():
    """トレンドフィルターの動作検証"""
    
    logger.info("=" * 80)
    logger.info("トレンドフィルター動作検証 開始")
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
    
    # トレンドフィルター設定確認
    logger.info("\n[STEP 3] トレンドフィルター設定")
    logger.info("=" * 80)
    logger.info(f"trend_filter_enabled: {strategy.params['trend_filter_enabled']}")
    logger.info(f"allowed_trends: {strategy.params['allowed_trends']}")
    logger.info("=" * 80)
    
    # 各エントリー日のトレンド判定を確認
    logger.info("\n[STEP 4] エントリー日のトレンド判定詳細")
    
    trend_details = []
    trend_counts = {}
    
    for i, date in enumerate(entry_dates, 1):
        idx = result.index.get_loc(date)
        
        if idx > 20:  # トレンド判定に必要な最小データ数
            # トレンド判定実行
            window_data = result.iloc[:idx + 1]
            
            try:
                # 統一トレンド判定
                trend = detect_unified_trend(
                    window_data,
                    price_column="Close",
                    strategy="contrarian_strategy",
                    method="combined"
                )
                
                # 信頼度付き判定
                detector = UnifiedTrendDetector(
                    window_data,
                    price_column="Close",
                    strategy_name="contrarian_strategy",
                    method="combined"
                )
                _, confidence = detector.detect_trend_with_confidence()
                
                # カウント
                trend_counts[trend] = trend_counts.get(trend, 0) + 1
                
                detail = {
                    "no": i,
                    "date": date.strftime('%Y-%m-%d'),
                    "trend": trend,
                    "confidence": confidence,
                    "is_allowed": trend in strategy.params['allowed_trends'],
                    "close": result['Close'].iloc[idx],
                    "rsi": result['RSI'].iloc[idx] if 'RSI' in result.columns else None
                }
                trend_details.append(detail)
                
                # 最初の10件とallowed外の全件を詳細ログ
                if i <= 10 or not detail["is_allowed"]:
                    logger.info(f"\nエントリー {i}: {date.strftime('%Y-%m-%d')}")
                    logger.info(f"  トレンド: {trend}")
                    logger.info(f"  信頼度: {confidence:.2f}")
                    logger.info(f"  許可範囲: {'はい' if detail['is_allowed'] else 'いいえ'}")
                    logger.info(f"  価格: {detail['close']:.2f} 円")
                    logger.info(f"  RSI: {detail['rsi']:.2f}" if detail['rsi'] is not None else "  RSI: N/A")
                    
            except Exception as e:
                logger.warning(f"エントリー {i}: トレンド判定エラー - {e}")
                detail = {
                    "no": i,
                    "date": date.strftime('%Y-%m-%d'),
                    "trend": "ERROR",
                    "confidence": 0.0,
                    "is_allowed": False,
                    "close": result['Close'].iloc[idx],
                    "rsi": result['RSI'].iloc[idx] if 'RSI' in result.columns else None
                }
                trend_details.append(detail)
    
    # 統計サマリー
    logger.info("\n" + "=" * 80)
    logger.info("統計サマリー")
    logger.info("=" * 80)
    logger.info(f"総エントリー数: {len(entry_dates)}")
    logger.info(f"トレンド判定成功: {len(trend_details)}")
    
    logger.info("\nトレンド別エントリー数:")
    for trend, count in sorted(trend_counts.items()):
        percentage = count / len(trend_details) * 100 if trend_details else 0
        is_allowed = trend in strategy.params['allowed_trends']
        status = "[許可]" if is_allowed else "[禁止]"
        logger.info(f"  {status} {trend}: {count} 回 ({percentage:.1f}%)")
    
    # DataFrame作成
    df = pd.DataFrame(trend_details)
    
    if not df.empty:
        # 許可外トレンドでのエントリー数
        not_allowed_count = (~df['is_allowed']).sum()
        logger.info(f"\n許可外トレンドでのエントリー: {not_allowed_count} 回")
        
        # 信頼度統計
        logger.info("\nトレンド判定信頼度の統計:")
        logger.info(f"  平均: {df['confidence'].mean():.2f}")
        logger.info(f"  中央値: {df['confidence'].median():.2f}")
        logger.info(f"  最小: {df['confidence'].min():.2f}")
        logger.info(f"  最大: {df['confidence'].max():.2f}")
        
        # CSV出力
        output_path = Path("tests/results/contrarian_trend_analysis.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"\n詳細データCSV出力: {output_path}")
    
    # 問題の特定
    logger.info("\n" + "=" * 80)
    logger.info("問題の特定")
    logger.info("=" * 80)
    
    if not df.empty:
        not_allowed_count = (~df['is_allowed']).sum()
        
        if not_allowed_count > 0:
            logger.info(f"\n[重大な問題発見]")
            logger.info(f"allowed_trends=['range-bound'] と設定されているにも関わらず、")
            logger.info(f"{not_allowed_count} 件のエントリーが許可外トレンドで発生しています")
            logger.info("\n該当エントリー:")
            not_allowed_entries = df[~df['is_allowed']]
            for _, row in not_allowed_entries.iterrows():
                logger.info(f"  {row['date']}: {row['trend']} (信頼度: {row['confidence']:.2f})")
        else:
            logger.info("\n[正常]")
            logger.info("全エントリーが許可されたトレンド（range-bound）で発生しています")
            logger.info("トレンドフィルターは正常に機能しています")
    
    # range-boundのみである場合の追加検証
    if "range-bound" in trend_counts and len(trend_counts) == 1:
        logger.info("\n[追加検証]")
        logger.info("全エントリーが range-bound と判定されています")
        logger.info("これは以下の可能性を示唆:")
        logger.info("1. 2023-2024年の8306.Tは実際にレンジ相場だった")
        logger.info("2. トレンド判定が range-bound に偏っている可能性")
        logger.info("3. トレンド判定の閾値が適切でない可能性")
        
        # 価格範囲の確認
        price_range = result['Close'].max() - result['Close'].min()
        price_volatility = (price_range / result['Close'].mean()) * 100
        logger.info(f"\n価格統計:")
        logger.info(f"  最小: {result['Close'].min():.2f} 円")
        logger.info(f"  最大: {result['Close'].max():.2f} 円")
        logger.info(f"  レンジ: {price_range:.2f} 円")
        logger.info(f"  変動率: {price_volatility:.2f}%")
    
    logger.info("\n" + "=" * 80)
    logger.info("分析完了")
    logger.info("=" * 80)
    
    return df


if __name__ == "__main__":
    result_df = analyze_trend_filter()
