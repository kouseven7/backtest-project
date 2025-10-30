"""
Phase B-2-2: DOW Filter Test for Opening Gap Strategy

目的:
- DOWフィルター（米国市場トレンド連動）の効果を検証
- Phase B-1（全フィルターOFF）との比較分析
- 米国市場との相関性を確認

設定:
- dow_filter_enabled: TRUE（DOWトレンドフィルター有効化）
- dow_trend_days: 5日間のトレンド判定
- trend_filter_enabled: FALSE
- volatility_filter: FALSE
- その他パラメータはPhase B-1と同一

期待される結果:
- 米国市場との連動性が高い場合、勝率向上
- グローバル要因によるギャップの精度向上
- 取引機会は減少する可能性あり

Author: Backtest Project Team
Created: 2025-10-30
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import json
from datetime import datetime
from strategies.Opening_Gap import OpeningGapStrategy
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


class DOWFilterTester:
    """DOWフィルターテスター"""
    
    def __init__(self):
        self.ticker = "7203.T"  # トヨタ自動車
        self.dow_ticker = "^DJI"  # ダウ・ジョーンズ工業株価平均
        self.start_date = "2022-01-01"
        self.end_date = "2024-12-31"
        
        # Phase B-1ベースライン（比較用）
        self.baseline = {
            "total_trades": 188,
            "win_rate": 35.6,
            "total_pnl": 38.43,
            "avg_pnl": 0.20,
            "avg_win": 3.14,
            "avg_loss": -1.42,
            "avg_hold_days": 4.1
        }
        
    def _fetch_data(self):
        """7203.Tと^DJIデータを取得"""
        logger.info("")
        logger.info("[PHASE 1] データ取得")
        logger.info("=" * 80)
        
        # 7203.T データ
        datafeed = YFinanceDataFeed()
        self.stock_data = datafeed.get_stock_data(self.ticker, self.start_date, self.end_date)
        logger.info(f"7203.T データ: {len(self.stock_data)}件取得")
        
        # ^DJI データ
        self.dow_data = datafeed.get_stock_data(self.dow_ticker, self.start_date, self.end_date)
        logger.info(f"^DJI データ: {len(self.dow_data)}件取得")
        logger.info("")
        
    def _run_backtest(self):
        """DOWフィルター有効でバックテスト実行"""
        logger.info("[PHASE 2] バックテスト実行（DOWフィルター有効）")
        logger.info("=" * 80)
        
        # Phase B-2-2設定: DOWフィルターのみ有効
        params = {
            "gap_threshold": 0.02,           # 2% ギャップ
            "stop_loss": 0.10,               # 10% ストップロス（緩和）
            "take_profit": 0.10,             # 10% 利益確定
            "max_hold_days": 20,             # 最大保有20日
            "trailing_stop_pct": 0.20,       # トレーリングストップ20%
            
            # DOWフィルター設定
            "dow_filter_enabled": True,      # DOWフィルター有効（KEY）
            "dow_trend_days": 5,             # 5日間トレンド判定
            "gap_direction": "up",           # ギャップアップのみ
            
            # 他のフィルター無効化
            "trend_filter_enabled": False,   # トレンドフィルターOFF
            "volatility_filter": False,      # ボラティリティフィルターOFF
        }
        
        logger.info(f"DOWフィルター設定:")
        logger.info(f"  - dow_filter_enabled: {params['dow_filter_enabled']}")
        logger.info(f"  - dow_trend_days: {params['dow_trend_days']}日")
        logger.info(f"  - gap_direction: {params['gap_direction']}")
        logger.info(f"  - trend_filter_enabled: {params['trend_filter_enabled']}")
        logger.info(f"  - volatility_filter: {params['volatility_filter']}")
        logger.info("")
        
        # 戦略実行
        strategy = OpeningGapStrategy(
            data=self.stock_data,
            dow_data=self.dow_data,
            params=params,
            price_column="Adj Close"
        )
        
        self.result_data = strategy.backtest()
        
        # Entry/Exit信号カウント
        entry_count = (self.result_data['Entry_Signal'] == 1).sum()
        exit_count = (self.result_data['Exit_Signal'] == -1).sum()
        
        logger.info("")
        logger.info("[SUCCESS] バックテスト完了")
        logger.info(f"  Entry Signal: {entry_count}回")
        logger.info(f"  Exit Signal: {exit_count}回")
        logger.info("")
        
    def _analyze_results(self):
        """Phase B-1との比較分析"""
        logger.info("[PHASE 3] 結果分析")
        logger.info("=" * 80)
        
        # トレード抽出
        entries = self.result_data[self.result_data['Entry_Signal'] == 1].copy()
        exits = self.result_data[self.result_data['Exit_Signal'] == -1].copy()
        
        trades = []
        entry_idx = 0
        
        for exit_idx in range(len(exits)):
            if entry_idx >= len(entries):
                break
                
            entry_date = entries.index[entry_idx]
            exit_date = exits.index[exit_idx]
            
            if exit_date <= entry_date:
                continue
                
            entry_price = entries['Open'].iloc[entry_idx]
            exit_price = exits['Adj Close'].iloc[exit_idx]
            pnl = (exit_price / entry_price - 1) * 100
            
            hold_days = (exit_date - entry_date).days
            
            trades.append({
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl": float(pnl),
                "hold_days": hold_days,
                "result": "win" if pnl > 0 else "loss"
            })
            
            entry_idx += 1
        
        # 統計計算
        if len(trades) == 0:
            logger.error("[ERROR] 取引が0件です")
            return
            
        total_trades = len(trades)
        wins = [t for t in trades if t['result'] == 'win']
        losses = [t for t in trades if t['result'] == 'loss']
        
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl = total_pnl / total_trades
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        avg_hold = sum(t['hold_days'] for t in trades) / total_trades
        
        # Phase B-1との比較
        win_rate_change = win_rate - self.baseline['win_rate']
        pnl_change = total_pnl - self.baseline['total_pnl']
        trade_change = total_trades - self.baseline['total_trades']
        
        # コンソール出力
        logger.info(f"総取引数: {total_trades}件")
        logger.info(f"勝率: {win_rate:.1f}% ({len(wins)}勝 / {len(losses)}敗)")
        logger.info(f"総損益: {total_pnl:.2f}%")
        logger.info(f"平均損益: {avg_pnl:.2f}%")
        logger.info(f"平均利益: {avg_win:.2f}%")
        logger.info(f"平均損失: {avg_loss:.2f}%")
        logger.info(f"平均保有日数: {avg_hold:.1f}日")
        logger.info("")
        
        logger.info("--- Phase B-1（最小構成）との比較 ---")
        logger.info(f"Phase B-1: 勝率{self.baseline['win_rate']}%, 総損益+{self.baseline['total_pnl']}%, 取引数{self.baseline['total_trades']}件")
        logger.info(f"Phase B-2-2: 勝率{win_rate:.1f}%, 総損益{total_pnl:.2f}%, 取引数{total_trades}件")
        logger.info("")
        logger.info(f"勝率変化: {win_rate_change:+.1f}%ポイント")
        logger.info(f"総損益変化: {pnl_change:+.2f}%")
        logger.info(f"取引数変化: {trade_change:+d}件")
        logger.info("")
        
        # 最初の5件表示
        logger.info("最初の5件のトレード:")
        for i, trade in enumerate(trades[:5], 1):
            logger.info(f"  #{i}: {trade['entry_date']} → {trade['exit_date']} "
                       f"({trade['hold_days']}日) P&L: {trade['pnl']:+.2f}%")
        logger.info("")
        
        # 最大利益・損失
        if trades:
            max_win_trade = max(trades, key=lambda x: x['pnl'])
            max_loss_trade = min(trades, key=lambda x: x['pnl'])
            logger.info(f"最大利益: {max_win_trade['pnl']:+.2f}% "
                       f"({max_win_trade['entry_date']} → {max_win_trade['exit_date']})")
            logger.info(f"最大損失: {max_loss_trade['pnl']:+.2f}% "
                       f"({max_loss_trade['entry_date']} → {max_loss_trade['exit_date']})")
            logger.info("")
        
        # JSON保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"tests/dow_filter_report_{timestamp}.json"
        
        report = {
            "test_info": {
                "phase": "B-2-2",
                "filter": "DOW Filter",
                "ticker": self.ticker,
                "period": f"{self.start_date} to {self.end_date}",
                "timestamp": timestamp
            },
            "configuration": {
                "dow_filter_enabled": True,
                "dow_trend_days": 5,
                "gap_direction": "up",
                "trend_filter_enabled": False,
                "volatility_filter": False,
                "gap_threshold": 0.02,
                "stop_loss": 0.10,
                "max_hold_days": 20
            },
            "results": {
                "total_trades": total_trades,
                "win_rate": round(win_rate, 1),
                "wins": len(wins),
                "losses": len(losses),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(avg_pnl, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "avg_hold_days": round(avg_hold, 1)
            },
            "comparison_with_baseline": {
                "baseline": self.baseline,
                "changes": {
                    "win_rate_change": round(win_rate_change, 1),
                    "pnl_change": round(pnl_change, 2),
                    "trade_change": trade_change
                }
            },
            "trades": trades
        }
        
        import os
        os.makedirs("tests", exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        abs_path = os.path.abspath(report_path)
        logger.info(f"詳細レポート保存: {abs_path}")
        logger.info("=" * 80)
        
    def run(self):
        """テスト実行"""
        try:
            self._fetch_data()
            self._run_backtest()
            self._analyze_results()
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("テスト完了: ステータス = SUCCESS")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"[ERROR] テスト失敗: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    tester = DOWFilterTester()
    tester.run()
