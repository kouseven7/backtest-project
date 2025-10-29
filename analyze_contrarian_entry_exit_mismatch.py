"""
analyze_contrarian_entry_exit_mismatch.py - エントリー/エグジット不一致の原因調査

調査目的:
1. エントリー52回、エグジット51回の不一致原因を特定
2. 未決済ポジションの詳細を確認
3. 戦略ロジックの整合性を検証

調査方法:
- 全エントリー/エグジットのペアリング確認
- 未決済ポジションの特定
- 最終日の状態確認

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


def analyze_entry_exit_mismatch():
    """エントリー/エグジット不一致の原因調査"""
    
    logger.info("=" * 80)
    logger.info("エントリー/エグジット不一致の原因調査 開始")
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
    logger.info(f"データ期間: {stock_data.index[0]} ~ {stock_data.index[-1]}")
    
    # 戦略初期化とバックテスト実行
    logger.info("\n[STEP 2] 戦略実行")
    strategy = ContrarianStrategy(data=stock_data, price_column="Close")
    result = strategy.backtest()
    
    # エントリー/エグジット数の確認
    entry_count = (result['Entry_Signal'] == 1).sum()
    exit_count = (result['Exit_Signal'] == -1).sum()
    
    logger.info(f"総エントリー数: {entry_count}")
    logger.info(f"総エグジット数: {exit_count}")
    logger.info(f"差分: {entry_count - exit_count}")
    
    # エントリー/エグジット日を取得
    entry_dates = result[result['Entry_Signal'] == 1].index
    exit_dates = result[result['Exit_Signal'] == -1].index
    
    logger.info("\n[STEP 3] エントリー/エグジットのペアリング分析")
    logger.info("=" * 80)
    
    # ペアリング分析
    paired_trades = []
    unpaired_entries = []
    
    entry_idx = 0
    exit_idx = 0
    
    while entry_idx < len(entry_dates):
        entry_date = entry_dates[entry_idx]
        
        # この entry に対応する exit を探す
        matched_exit = None
        for i in range(exit_idx, len(exit_dates)):
            if exit_dates[i] > entry_date:
                matched_exit = exit_dates[i]
                exit_idx = i + 1
                break
        
        if matched_exit:
            entry_loc = result.index.get_loc(entry_date)
            exit_loc = result.index.get_loc(matched_exit)
            days_held = exit_loc - entry_loc
            
            paired_trades.append({
                "entry_no": entry_idx + 1,
                "entry_date": entry_date,
                "exit_date": matched_exit,
                "days_held": days_held,
                "entry_price": result['Close'].iloc[entry_loc],
                "exit_price": result['Close'].iloc[exit_loc],
                "pnl": result['Close'].iloc[exit_loc] - result['Close'].iloc[entry_loc]
            })
        else:
            # 対応する exit が見つからない
            entry_loc = result.index.get_loc(entry_date)
            unpaired_entries.append({
                "entry_no": entry_idx + 1,
                "entry_date": entry_date,
                "entry_price": result['Close'].iloc[entry_loc],
                "rsi": result['RSI'].iloc[entry_loc] if 'RSI' in result.columns else None,
                "days_since_entry": len(result) - 1 - entry_loc
            })
        
        entry_idx += 1
    
    logger.info(f"ペアリング成功: {len(paired_trades)} 件")
    logger.info(f"未決済エントリー: {len(unpaired_entries)} 件")
    
    # 未決済エントリーの詳細
    if unpaired_entries:
        logger.info("\n[STEP 4] 未決済エントリーの詳細")
        logger.info("=" * 80)
        
        for entry in unpaired_entries:
            logger.info(f"\nエントリー {entry['entry_no']}: {entry['entry_date'].strftime('%Y-%m-%d')}")
            logger.info(f"  エントリー価格: {entry['entry_price']:.2f} 円")
            logger.info(f"  RSI: {entry['rsi']:.2f}" if entry['rsi'] is not None else "  RSI: N/A")
            logger.info(f"  エントリーからの経過日数: {entry['days_since_entry']} 日")
            logger.info(f"  最大保有日数: {strategy.params['max_hold_days']} 日")
            
            # エントリー日以降のデータを確認
            entry_loc = result.index.get_loc(entry['entry_date'])
            remaining_days = len(result) - entry_loc - 1
            
            logger.info(f"  エントリー後の残り営業日数: {remaining_days} 日")
            
            if remaining_days < strategy.params['max_hold_days']:
                logger.info(f"  [原因] データ期間終了までに最大保有日数に達していない")
                logger.info(f"         （残り{remaining_days}日 < 最大{strategy.params['max_hold_days']}日）")
            
            # 最終日の状態確認
            last_idx = len(result) - 1
            last_price = result['Close'].iloc[last_idx]
            last_rsi = result['RSI'].iloc[last_idx] if 'RSI' in result.columns else None
            
            logger.info(f"\n  最終日の状態（{result.index[-1].strftime('%Y-%m-%d')}）:")
            logger.info(f"  価格: {last_price:.2f} 円")
            logger.info(f"  RSI: {last_rsi:.2f}" if last_rsi is not None else "  RSI: N/A")
            logger.info(f"  PnL（未実現）: {last_price - entry['entry_price']:.2f} 円")
            
            # エグジット条件のチェック
            logger.info(f"\n  エグジット条件チェック:")
            
            # 利益確定条件
            take_profit_price = entry['entry_price'] * (1.0 + strategy.params['take_profit'])
            logger.info(f"    利益確定価格: {take_profit_price:.2f} 円 (閾値: +{strategy.params['take_profit']*100:.0f}%)")
            logger.info(f"      現在価格: {last_price:.2f} 円")
            logger.info(f"      達成: {'はい' if last_price >= take_profit_price else 'いいえ'}")
            
            # ストップロス条件
            stop_loss_price = entry['entry_price'] * (1.0 - strategy.params['stop_loss'])
            logger.info(f"    ストップロス価格: {stop_loss_price:.2f} 円 (閾値: -{strategy.params['stop_loss']*100:.0f}%)")
            logger.info(f"      現在価格: {last_price:.2f} 円")
            logger.info(f"      達成: {'はい' if last_price <= stop_loss_price else 'いいえ'}")
            
            # RSIエグジット条件
            logger.info(f"    RSI条件: RSI >= {strategy.params['rsi_exit_level']}")
            if last_rsi is not None:
                logger.info(f"      現在RSI: {last_rsi:.2f}")
                logger.info(f"      達成: {'はい' if last_rsi >= strategy.params['rsi_exit_level'] else 'いいえ'}")
            else:
                logger.info(f"      現在RSI: N/A")
            
            # 最大保有日数条件
            logger.info(f"    最大保有日数: {strategy.params['max_hold_days']} 日")
            logger.info(f"      経過日数: {entry['days_since_entry']} 日")
            logger.info(f"      達成: {'はい' if entry['days_since_entry'] >= strategy.params['max_hold_days'] else 'いいえ'}")
    
    # 統計サマリー
    logger.info("\n" + "=" * 80)
    logger.info("統計サマリー")
    logger.info("=" * 80)
    logger.info(f"総エントリー数: {entry_count}")
    logger.info(f"ペアリング成功: {len(paired_trades)}")
    logger.info(f"未決済エントリー: {len(unpaired_entries)}")
    
    if len(unpaired_entries) > 0:
        logger.info(f"\n不一致率: {len(unpaired_entries)/entry_count*100:.1f}%")
    
    # 結論
    logger.info("\n" + "=" * 80)
    logger.info("結論")
    logger.info("=" * 80)
    
    if len(unpaired_entries) == 1 and unpaired_entries[0]['entry_no'] == entry_count:
        logger.info("\n[正常]")
        logger.info("最後のエントリーがデータ期間終了により未決済となっています")
        logger.info("これはバックテストの仕様上、正常な動作です")
        logger.info("\n理由:")
        logger.info("  最終エントリー日以降の営業日数が不足しており、")
        logger.info("  いずれのエグジット条件も満たす前にデータが終了しています")
    else:
        logger.info("\n[異常]")
        logger.info("データ期間終了以外の理由で未決済エントリーが存在します")
        logger.info("戦略ロジックに問題がある可能性があります")
    
    logger.info("\n" + "=" * 80)
    logger.info("分析完了")
    logger.info("=" * 80)
    
    return paired_trades, unpaired_entries


if __name__ == "__main__":
    paired, unpaired = analyze_entry_exit_mismatch()
