"""
Opening Gap戦略 最小構成テスト - Phase B

全フィルターを無効化して基本ロジックのみをテスト
株式分割の影響を避けるため2022-2024年のデータを使用

検証項目:
1. 基本的なギャップ検出ロジックの動作確認
2. エントリー/イグジットの実行確認
3. フィルター無しでのパフォーマンス測定
4. 株式分割の影響を排除した結果の確認

Author: Backtest Project Team
Created: 2025-10-30
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from strategies.Opening_Gap import OpeningGapStrategy

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)

class MinimalConfigTester:
    """最小構成テスター"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ticker = "7203.T"
        self.index_ticker = "^DJI"
        # 株式分割後の期間を使用（2022年以降）
        self.start_date = "2022-01-01"
        self.end_date = "2024-12-31"
        self.data_feed = None
        self.stock_data = None
        self.dow_data = None
        self.strategy = None
        self.result_df = None
        
        # 最小構成パラメータ（全フィルター無効）
        self.minimal_params = {
            "gap_threshold": 0.02,        # 2%のギャップ
            "stop_loss": 0.10,            # 10%ストップロス（緩く設定）
            "take_profit": 0.10,          # 10%利確
            "atr_threshold": 2.0,
            "entry_delay": 0,
            "gap_direction": "both",      # 上下両方のギャップ
            
            # 全フィルター無効化
            "dow_filter_enabled": False,
            "volatility_filter": False,
            "trend_filter_enabled": False,
            
            # イグジット条件も緩く
            "max_hold_days": 20,          # 20営業日保有可能
            "consecutive_down_days": 10,  # 10日連続下落でイグジット
            "trailing_stop_pct": 0.20,    # 20%トレーリングストップ
            "atr_stop_multiple": 5.0,     # ATR×5のストップ
            "partial_exit_enabled": False
        }
        
    def run(self):
        """テスト実行"""
        self.logger.info("=" * 80)
        self.logger.info("Phase B: 最小構成テスト開始")
        self.logger.info(f"銘柄: {self.ticker}, 期間: {self.start_date} ~ {self.end_date}")
        self.logger.info("設定: 全フィルター無効 / 緩いストップロス")
        self.logger.info("=" * 80)
        
        # Phase 1: データ取得
        if not self._fetch_data():
            return "FAILED_DATA_FETCH"
        
        # Phase 2: バックテスト実行
        if not self._run_backtest():
            return "FAILED_BACKTEST"
        
        # Phase 3: 結果分析
        self._analyze_results()
        
        return "SUCCESS"
    
    def _fetch_data(self) -> bool:
        """データ取得"""
        self.logger.info("")
        self.logger.info("[PHASE 1] データ取得")
        self.logger.info("-" * 80)
        
        try:
            self.data_feed = YFinanceDataFeed()
            
            # 株価データ取得
            self.logger.info(f"株価データ取得: {self.ticker}")
            self.stock_data = self.data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.stock_data is None or self.stock_data.empty:
                self.logger.error("[ERROR] 株価データ取得失敗")
                return False
            
            self.logger.info(f"[SUCCESS] 株価データ: {len(self.stock_data)} rows")
            
            # DOWデータ取得
            self.logger.info(f"DOWデータ取得: {self.index_ticker}")
            self.dow_data = self.data_feed.get_stock_data(
                ticker=self.index_ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.dow_data is None or self.dow_data.empty:
                self.logger.error("[ERROR] DOWデータ取得失敗")
                return False
            
            self.logger.info(f"[SUCCESS] DOWデータ: {len(self.dow_data)} rows")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] データ取得エラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _run_backtest(self) -> bool:
        """バックテスト実行"""
        self.logger.info("")
        self.logger.info("[PHASE 2] バックテスト実行")
        self.logger.info("-" * 80)
        self.logger.info("パラメータ設定:")
        for key, value in self.minimal_params.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("-" * 80)
        
        try:
            # 戦略初期化
            self.strategy = OpeningGapStrategy(
                data=self.stock_data,
                dow_data=self.dow_data,
                params=self.minimal_params,
                price_column="Adj Close"
            )
            
            # バックテスト実行
            self.result_df = self.strategy.backtest()
            
            if self.result_df is None or self.result_df.empty:
                self.logger.error("[ERROR] バックテスト結果が空")
                return False
            
            # シグナル統計
            entry_count = (self.result_df["Entry_Signal"] == 1).sum()
            exit_count = (self.result_df["Exit_Signal"] == -1).sum()
            
            self.logger.info("")
            self.logger.info("[SUCCESS] バックテスト完了")
            self.logger.info(f"  Entry Signal: {entry_count}回")
            self.logger.info(f"  Exit Signal: {exit_count}回")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] バックテスト実行エラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _analyze_results(self):
        """結果分析"""
        self.logger.info("")
        self.logger.info("[PHASE 3] 結果分析")
        self.logger.info("=" * 80)
        
        # トレード抽出
        trades = []
        entry_date = None
        entry_price = None
        
        for idx, row in self.result_df.iterrows():
            if row["Entry_Signal"] == 1 and entry_date is None:
                entry_date = idx
                entry_price = row["Adj Close"]
            elif row["Exit_Signal"] == -1 and entry_date is not None:
                exit_date = idx
                exit_price = row["Adj Close"]
                
                pnl_pct = ((exit_price / entry_price) - 1) * 100
                
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": (exit_date - entry_date).days
                })
                
                entry_date = None
                entry_price = None
        
        # 統計計算
        total_trades = len(trades)
        
        if total_trades == 0:
            self.logger.warning("[WARNING] トレードが0件")
            self.logger.info("=" * 80)
            return
        
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(t["pnl_pct"] for t in trades)
        avg_pnl = total_pnl / total_trades
        avg_win = sum(w["pnl_pct"] for w in wins) / len(wins) if wins else 0
        avg_loss = sum(l["pnl_pct"] for l in losses) / len(losses) if losses else 0
        avg_hold_days = sum(t["hold_days"] for t in trades) / total_trades
        
        # 結果表示
        self.logger.info(f"総取引数: {total_trades}件")
        self.logger.info(f"勝率: {win_rate:.1f}% ({len(wins)}勝 / {len(losses)}敗)")
        self.logger.info(f"総損益: {total_pnl:.2f}%")
        self.logger.info(f"平均損益: {avg_pnl:.2f}%")
        self.logger.info(f"平均利益: {avg_win:.2f}%")
        self.logger.info(f"平均損失: {avg_loss:.2f}%")
        self.logger.info(f"平均保有日数: {avg_hold_days:.1f}日")
        
        # トレード例表示
        if total_trades > 0:
            self.logger.info("")
            self.logger.info("最初の5件のトレード:")
            for i, trade in enumerate(trades[:5], 1):
                self.logger.info(
                    f"  #{i}: {trade['entry_date'].date()} → {trade['exit_date'].date()} "
                    f"({trade['hold_days']}日) P&L: {trade['pnl_pct']:.2f}%"
                )
            
            # 最大利益・最大損失
            max_win_trade = max(trades, key=lambda t: t["pnl_pct"])
            max_loss_trade = min(trades, key=lambda t: t["pnl_pct"])
            
            self.logger.info("")
            self.logger.info(f"最大利益: {max_win_trade['pnl_pct']:.2f}% "
                           f"({max_win_trade['entry_date'].date()} → {max_win_trade['exit_date'].date()})")
            self.logger.info(f"最大損失: {max_loss_trade['pnl_pct']:.2f}% "
                           f"({max_loss_trade['entry_date'].date()} → {max_loss_trade['exit_date'].date()})")
        
        # JSONレポート保存
        report_path = project_root / "tests" / f"minimal_config_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            "test_type": "minimal_configuration",
            "ticker": self.ticker,
            "period": f"{self.start_date} ~ {self.end_date}",
            "parameters": self.minimal_params,
            "results": {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "total_pnl_pct": total_pnl,
                "avg_pnl_pct": avg_pnl,
                "avg_win_pct": avg_win,
                "avg_loss_pct": avg_loss,
                "avg_hold_days": avg_hold_days,
                "wins": len(wins),
                "losses": len(losses)
            },
            "trades": [
                {
                    "entry_date": str(t["entry_date"].date()),
                    "exit_date": str(t["exit_date"].date()),
                    "pnl_pct": float(t["pnl_pct"]),
                    "hold_days": int(t["hold_days"])
                }
                for t in trades
            ]
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info("")
        self.logger.info(f"詳細レポート保存: {report_path}")
        self.logger.info("=" * 80)


def main():
    """メイン実行"""
    print("\n" + "=" * 80)
    print("Phase B: Opening Gap戦略 最小構成テスト")
    print("=" * 80)
    
    tester = MinimalConfigTester()
    status = tester.run()
    
    print("\n" + "=" * 80)
    print(f"テスト完了: ステータス = {status}")
    print("=" * 80)
    
    return status


if __name__ == "__main__":
    status = main()
    sys.exit(0 if status == "SUCCESS" else 1)
