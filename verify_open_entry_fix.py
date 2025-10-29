"""
verify_open_entry_fix.py - Open価格エントリー修正の検証

検証目的:
2024-07-26のトレードでOpen価格エントリーと同日エグジット判定により、
ストップロスが正常に機能することを確認する。

検証項目:
1. エントリー価格がOpen価格になっているか
2. 同日エグジット判定が実行されているか
3. ストップロスが4%設定通りに機能しているか
4. 2024-07-26のトレード結果が改善されているか

Author: Backtest Project Team
Created: 2025-10-28
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# プロジェクトパス設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from strategies.contrarian_strategy import ContrarianStrategy
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpenEntryVerifier:
    """Open価格エントリー修正の検証クラス"""
    
    def __init__(self):
        self.ticker = "8306.T"
        self.start_date = "2023-01-01"
        self.end_date = "2024-12-31"
        
        # Option 4パラメータ
        self.params = {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "gap_threshold": 0.02,
            "stop_loss": 0.04,
            "take_profit": 0.05,
            "pin_bar_ratio": 2.0,
            "max_hold_days": 5,
            "rsi_exit_level": 50,
            "trailing_stop_pct": 0.02,
            "trend_filter_enabled": True,
            "allowed_trends": ["range-bound"]
        }
        
    def run_backtest(self):
        """バックテスト実行"""
        logger.info("=" * 80)
        logger.info("Open価格エントリー修正の検証")
        logger.info("=" * 80)
        logger.info(f"銘柄: {self.ticker}")
        logger.info(f"期間: {self.start_date} ~ {self.end_date}")
        logger.info("")
        
        # データ取得
        logger.info("[STEP 1] データ取得")
        logger.info("-" * 80)
        data_feed = YFinanceDataFeed()
        stock_data = data_feed.get_stock_data(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date
        )
        logger.info(f"データ取得完了: {len(stock_data)} 行")
        logger.info("")
        
        # 戦略実行
        logger.info("[STEP 2] バックテスト実行")
        logger.info("-" * 80)
        strategy = ContrarianStrategy(stock_data, params=self.params)
        result = strategy.backtest()
        logger.info("バックテスト完了")
        logger.info("")
        
        return result, stock_data
    
    def analyze_trades(self, result, stock_data):
        """トレード分析"""
        logger.info("[STEP 3] トレード分析")
        logger.info("-" * 80)
        
        # エントリー/エグジットポイントを抽出
        entries = result[result['Entry_Signal'] == 1].copy()
        exits = result[result['Exit_Signal'] == -1].copy()
        
        num_entries = len(entries)
        num_exits = len(exits)
        
        logger.info(f"エントリー数: {num_entries}")
        logger.info(f"エグジット数: {num_exits}")
        logger.info("")
        
        if num_entries == 0:
            logger.warning("エントリーが0件です")
            return
        
        # トレード詳細分析
        trades = []
        for i, (entry_date, entry_row) in enumerate(entries.iterrows(), 1):
            # 対応するエグジットを検索
            exit_candidates = exits[exits.index > entry_date]
            if len(exit_candidates) == 0:
                continue
            
            exit_date = exit_candidates.index[0]
            exit_row = exits.loc[exit_date]
            
            # エントリー価格（Open価格）
            entry_price = stock_data.loc[entry_date, 'Open']
            # エグジット価格の計算
            days_held = (exit_date - entry_date).days
            
            if days_held == 0:
                # 同日エグジット: 安値をチェック
                day_low = stock_data.loc[exit_date, 'Low']
                stop_loss_price = entry_price * (1.0 - self.params["stop_loss"])
                
                if day_low <= stop_loss_price:
                    # ストップロス発動
                    exit_price = stop_loss_price
                    exit_reason = "ストップロス（同日）"
                else:
                    # 利益確定の可能性
                    exit_price = stock_data.loc[exit_date, 'Adj Close']
                    exit_reason = "同日エグジット"
            else:
                # 翌日以降のエグジット
                exit_price = stock_data.loc[exit_date, 'Adj Close']
                exit_reason = "通常エグジット"
            
            pnl = exit_price - entry_price
            pnl_pct = (pnl / entry_price) * 100
            
            trades.append({
                'trade_no': i,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'days_held': days_held,
                'exit_reason': exit_reason
            })
        
        trades_df = pd.DataFrame(trades)
        
        # トレード詳細表示
        logger.info("トレード詳細:")
        logger.info("")
        for _, trade in trades_df.iterrows():
            logger.info(f"Trade #{trade['trade_no']}")
            logger.info(f"  エントリー: {trade['entry_date'].strftime('%Y-%m-%d')} "
                       f"@ {trade['entry_price']} JPY (Open価格)")
            logger.info(f"  エグジット: {trade['exit_date'].strftime('%Y-%m-%d')} "
                       f"@ {trade['exit_price']} JPY")
            logger.info(f"  保有日数: {trade['days_held']} 日")
            logger.info(f"  損益: {trade['pnl']:+.2f} JPY ({trade['pnl_pct']:+.2f}%)")
            logger.info(f"  エグジット理由: {trade['exit_reason']}")
            logger.info("")
        
        # 統計サマリー
        logger.info("=" * 80)
        logger.info("統計サマリー")
        logger.info("=" * 80)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        avg_days = trades_df['days_held'].mean()
        
        logger.info(f"総トレード数: {total_trades}")
        logger.info(f"勝ちトレード: {winning_trades}")
        logger.info(f"負けトレード: {losing_trades}")
        logger.info(f"勝率: {win_rate:.2f}%")
        logger.info(f"総損益: {total_pnl:+.2f} JPY")
        logger.info(f"平均損益: {avg_pnl:+.2f} JPY")
        logger.info(f"平均保有日数: {avg_days:.2f} 日")
        logger.info("")
        
        # 2024-07-26のトレード検証
        logger.info("=" * 80)
        logger.info("2024-07-26トレードの検証")
        logger.info("=" * 80)
        
        target_date = pd.Timestamp('2024-07-26', tz='Asia/Tokyo')
        target_trade = trades_df[trades_df['entry_date'] == target_date]
        
        if len(target_trade) > 0:
            trade = target_trade.iloc[0]
            logger.info("修正後の結果:")
            logger.info(f"  エントリー価格: {trade['entry_price']} JPY (Open価格)")
            logger.info(f"  エグジット価格: {trade['exit_price']} JPY")
            logger.info(f"  損益: {trade['pnl']:+.2f} JPY ({trade['pnl_pct']:+.2f}%)")
            logger.info(f"  保有日数: {trade['days_held']} 日")
            logger.info(f"  エグジット理由: {trade['exit_reason']}")
            logger.info("")
            
            # ストップロス設定との比較
            logger.info("ストップロス検証:")
            stop_loss_threshold = self.params["stop_loss"] * 100
            logger.info(f"  設定: -{stop_loss_threshold}%")
            logger.info(f"  実際: {trade['pnl_pct']:+.2f}%")
            
            if trade['pnl_pct'] < -stop_loss_threshold - 0.5:
                logger.warning(f"  警告: ストップロス設定を超過しています（{abs(trade['pnl_pct']) - stop_loss_threshold:.2f}%超過）")
            else:
                logger.info("  確認: ストップロス設定内に収まっています")
            logger.info("")
            
            # 修正前との比較
            logger.info("修正前との比較:")
            logger.info(f"  修正前: 終値エントリー 1603.61 JPY → -169.50 JPY (-10.06%)")
            logger.info(f"  修正後: Open価格エントリー {trade['entry_price']} JPY → "
                       f"{trade['pnl']:+.2f} JPY ({trade['pnl_pct']:+.2f}%)")
            
            improvement = abs(-10.06) - abs(trade['pnl_pct'])
            logger.info(f"  改善度: {improvement:+.2f}%")
            
        else:
            logger.warning("2024-07-26のトレードが見つかりません")
        
        logger.info("")
        
        # CSV出力
        output_path = Path("tests/results/open_entry_verification_trades.csv")
        trades_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"トレード詳細をCSVに保存: {output_path}")
        logger.info("")
        
        return trades_df


def main():
    """メイン実行"""
    verifier = OpenEntryVerifier()
    result, stock_data = verifier.run_backtest()
    trades_df = verifier.analyze_trades(result, stock_data)
    
    logger.info("検証完了")


if __name__ == "__main__":
    main()
