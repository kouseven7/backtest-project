"""
パーフェクトオーダー問題の根本分析
検出ロジックの抜本的見直し
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from config.error_handling import fetch_stock_data

logger = setup_logger("perfect_order_analysis")

def analyze_toyota_trend_patterns(data: pd.DataFrame):
    """
    トヨタ株のトレンドパターンを詳細分析
    なぜパーフェクトオーダーが検出されないかを解明
    """
    logger.info("=" * 60)
    logger.info("トヨタ株 2023年トレンドパターン詳細分析")
    logger.info("=" * 60)
    
    # データ正規化
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    close_prices = data['Close'].dropna()
    
    # 基本統計
    logger.info(f"データ期間: {close_prices.index[0]} ～ {close_prices.index[-1]}")
    logger.info(f"データ日数: {len(close_prices)}日")
    logger.info(f"年間リターン: {((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100:.2f}%")
    
    # SMA計算
    sma5 = close_prices.rolling(window=5).mean()
    sma25 = close_prices.rolling(window=25).mean()
    sma75 = close_prices.rolling(window=75).mean()
    
    # 完全なデータが揃った期間（75日後から）
    valid_start_idx = 74  # 75日目から
    
    logger.info(f"\n有効分析期間: {close_prices.index[valid_start_idx]} ～ {close_prices.index[-1]}")
    logger.info(f"有効分析日数: {len(close_prices) - valid_start_idx}日")
    
    # 日次のパーフェクトオーダー状況をチェック
    perfect_order_days = 0
    semi_perfect_days = 0
    uptrend_days = 0
    
    trend_changes = []
    
    for i in range(valid_start_idx, len(close_prices)):
        price = close_prices.iloc[i]
        sma5_val = sma5.iloc[i]
        sma25_val = sma25.iloc[i]
        sma75_val = sma75.iloc[i]
        
        # 各種トレンド判定
        perfect = price > sma5_val > sma25_val > sma75_val
        semi_perfect = price > sma5_val > sma25_val
        uptrend = price > sma5_val
        
        if perfect:
            perfect_order_days += 1
        if semi_perfect:
            semi_perfect_days += 1
        if uptrend:
            uptrend_days += 1
        
        # トレンド変化点記録（週1回）
        if i % 5 == 0:  # 5日おき
            trend_changes.append({
                'date': close_prices.index[i],
                'price': price,
                'sma5': sma5_val,
                'sma25': sma25_val,
                'sma75': sma75_val,
                'perfect': perfect,
                'semi_perfect': semi_perfect,
                'uptrend': uptrend
            })
    
    valid_days = len(close_prices) - valid_start_idx
    
    logger.info(f"\n日次トレンド統計:")
    logger.info(f"  厳密Perfect Order: {perfect_order_days}日 ({perfect_order_days/valid_days*100:.1f}%)")
    logger.info(f"  準Perfect Order: {semi_perfect_days}日 ({semi_perfect_days/valid_days*100:.1f}%)")
    logger.info(f"  単純上昇トレンド: {uptrend_days}日 ({uptrend_days/valid_days*100:.1f}%)")
    
    # 週次トレンド変化
    logger.info(f"\n週次トレンド変化（抜粋）:")
    for i, change in enumerate(trend_changes[::4]):  # 月1回表示
        status = ""
        if change['perfect']:
            status = "🟢 Perfect"
        elif change['semi_perfect']:
            status = "🟡 Semi-Perfect"
        elif change['uptrend']:
            status = "🔵 Uptrend"
        else:
            status = "🔴 Downtrend"
        
        logger.info(f"  {change['date'].strftime('%Y-%m-%d')}: {status} 価格={change['price']:.0f}")
    
    # 最も良い期間を特定
    logger.info(f"\n最良パフォーマンス期間の特定:")
    
    # 四半期別分析
    q1_mask = (close_prices.index >= '2023-01-01') & (close_prices.index < '2023-04-01')
    q2_mask = (close_prices.index >= '2023-04-01') & (close_prices.index < '2023-07-01')
    q3_mask = (close_prices.index >= '2023-07-01') & (close_prices.index < '2023-10-01')
    q4_mask = (close_prices.index >= '2023-10-01') & (close_prices.index <= '2023-12-31')
    
    for q, mask in enumerate([q1_mask, q2_mask, q3_mask, q4_mask], 1):
        q_data = close_prices[mask]
        if len(q_data) == 0:
            continue
        
        q_return = (q_data.iloc[-1] / q_data.iloc[0] - 1) * 100
        logger.info(f"  Q{q}: {q_return:+.1f}% ({q_data.index[0].strftime('%m/%d')} - {q_data.index[-1].strftime('%m/%d')})")
    
    # 推奨戦略
    logger.info(f"\n[IDEA] 分析結果と推奨戦略:")
    
    if perfect_order_days == 0:
        logger.warning("[ERROR] 厳密なPerfect Orderは2023年中に一度も発生していません")
        logger.info("[UP] これは一般的で、多くの銘柄で同様の現象が見られます")
    
    if semi_perfect_days > 0:
        logger.info(f"[OK] 準Perfect Order ({semi_perfect_days}日) を基準にする戦略を推奨")
    elif uptrend_days > valid_days * 0.3:  # 30%以上
        logger.info(f"[OK] 単純上昇トレンド ({uptrend_days}日) を基準にする戦略を推奨")
    else:
        logger.warning("[WARNING]  2023年のトヨタはトレンド戦略に適さない可能性があります")
        logger.info("[CHART] 別の指標（RSI、MACD等）やレンジ戦略を検討してください")
    
    return {
        'perfect_order_days': perfect_order_days,
        'semi_perfect_days': semi_perfect_days,
        'uptrend_days': uptrend_days,
        'valid_days': valid_days,
        'trend_changes': trend_changes
    }

def suggest_alternative_strategies():
    """
    パーフェクトオーダー以外の代替戦略を提案
    """
    logger.info("=" * 60)
    logger.info("代替戦略の提案")
    logger.info("=" * 60)
    
    strategies = [
        {
            'name': '移動平均乖離率戦略',
            'description': '価格がSMA25から±5%乖離した時点で逆張り',
            'pros': '頻繁なシグナル、レンジ相場に強い',
            'cons': 'トレンド相場で損失拡大リスク'
        },
        {
            'name': 'ゴールデンクロス戦略',
            'description': 'SMA5がSMA25を上抜けで買い、下抜けで売り',
            'pros': 'シンプル、中期トレンドをキャッチ',
            'cons': 'ダマシが多い、遅行性'
        },
        {
            'name': 'RSI逆張り戦略',
            'description': 'RSI30以下で買い、70以上で売り',
            'pros': '過熱感を数値化、逆張りタイミング明確',
            'cons': '強いトレンドで機能しない'
        },
        {
            'name': 'ボリンジャーバンド戦略',
            'description': '下限タッチで買い、上限タッチで売り',
            'pros': 'ボラティリティ考慮、適応性高い',
            'cons': 'パラメータ調整が複雑'
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        logger.info(f"{i}. {strategy['name']}")
        logger.info(f"   概要: {strategy['description']}")
        logger.info(f"   メリット: {strategy['pros']}")
        logger.info(f"   デメリット: {strategy['cons']}")
        logger.info("")

def main():
    """メイン実行"""
    logger.info("[SEARCH] パーフェクトオーダー問題の根本分析開始")
    
    # データ取得
    data = fetch_stock_data("7203", "2023-01-01", "2023-12-31")
    
    # 詳細分析実行
    result = analyze_toyota_trend_patterns(data)
    
    # 代替戦略提案
    suggest_alternative_strategies()
    
    # 結論
    logger.info("=" * 60)
    logger.info("[TARGET] 結論と次のアクション")
    logger.info("=" * 60)
    
    if result['semi_perfect_days'] > 0:
        logger.info("[OK] 準Perfect Order戦略への変更を推奨")
        logger.info("   修正: 価格 > SMA5 > SMA25 の条件に緩和")
    elif result['uptrend_days'] > result['valid_days'] * 0.3:
        logger.info("[OK] 単純上昇トレンド戦略への変更を推奨")
        logger.info("   修正: 価格 > SMA5 の条件のみに簡略化")
    else:
        logger.info("[ERROR] パーフェクトオーダー系戦略は2023年トヨタに不適")
        logger.info("   推奨: 全く異なる戦略（RSI、ボリバン等）に変更")
    
    logger.info("\n[LIST] 実装優先順位:")
    logger.info("1. 準Perfect Order検出器の実装")
    logger.info("2. DSSMSバックテスター integration")
    logger.info("3. 代替戦略システムの検討")

if __name__ == "__main__":
    main()
