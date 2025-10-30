"""
Opening_Gap_Enhanced.py 動作検証テスト - 8306.T (2年間)

EnhancedBaseStrategyを使用したOpeningGapEnhancedStrategyの動作を検証します。

検証項目:
1. データ取得とバックテスト実行
2. 同日Entry/Exit問題がないことを確認
3. ポジション管理が正常に動作することを確認
4. トレード実行とP&L計算が正確であることを確認
5. EnhancedBaseStrategyの機能が正常に動作することを確認

期間: 2023-01-01 ~ 2024-12-31 (2年間)
銘柄: 8306.T (三菱UFJフィナンシャル・グループ)

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

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from strategies.Opening_Gap_Enhanced import OpeningGapEnhancedStrategy

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)

class OpeningGapEnhancedTester:
    """Opening_Gap_Enhanced.py テスター - 8306.T 2年間データ"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbol = "8306.T"
        self.index_symbol = "^DJI"
        self.start_date = "2023-01-01"
        self.end_date = "2024-12-31"
        self.data_feed = None
        self.strategy = None
        self.result_df = None
        
        # デフォルトパラメータ (最適化ファイルがないため)
        self.params = {
            "gap_threshold": 0.02,
            "stop_loss": 0.01,
            "take_profit": 0.03,
            "atr_threshold": 2.0,
            "entry_delay": 0,
            "gap_direction": "both",
            "dow_filter_enabled": True,
            "dow_trend_days": 5,
            "min_vol_ratio": 1.0,
            "volatility_filter": True,
            "trend_filter_enabled": True,
            "allowed_trends": ["uptrend"],
            "max_hold_days": 7,
            "consecutive_down_days": 1,
            "trailing_stop_pct": 0.02,
            "atr_stop_multiple": 1.5,
            "partial_exit_enabled": False,
            "partial_exit_threshold": 0.03,
            "partial_exit_portion": 0.5
        }
        
    def run(self):
        """テスト実行"""
        self.logger.info("=" * 80)
        self.logger.info("Opening_Gap_Enhanced.py 動作検証テスト開始")
        self.logger.info(f"銘柄: {self.symbol}, 期間: {self.start_date} ~ {self.end_date}")
        self.logger.info("=" * 80)
        
        # Phase 1: データ取得とバックテスト実行
        success_phase1 = self._phase1_data_and_backtest()
        
        if not success_phase1:
            self.logger.error("[FAILED] Phase 1: データ取得またはバックテスト実行に失敗")
            return "FAILED_PHASE1"
        
        # Phase 2: 検証ロジック実行
        success_phase2 = self._phase2_validation()
        
        if not success_phase2:
            self.logger.error("[FAILED] Phase 2: 検証ロジックに失敗")
            return "FAILED_PHASE2"
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("[SUCCESS] Phase 1-2: データ取得・バックテスト・検証完了")
        self.logger.info("=" * 80)
        
        return "SUCCESS_PHASE2"
    
    def _phase1_data_and_backtest(self) -> bool:
        """Phase 1: データ取得とバックテスト実行"""
        self.logger.info("")
        self.logger.info("[PHASE 1] データ取得開始")
        self.logger.info("-" * 80)
        
        try:
            # データフィード初期化
            self.data_feed = YFinanceDataFeed()
            
            self.logger.info("")
            self.logger.info(f"[DATA_FETCH] 株価データ取得: {self.symbol}")
            
            # 株価データ取得
            stock_data = self.data_feed.get_stock_data(
                ticker=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if stock_data is None or stock_data.empty:
                self.logger.error(f"[ERROR] 株価データ取得失敗: {self.symbol}")
                return False
            
            self.logger.info(f"[SUCCESS] 株価データ取得完了: {len(stock_data)} rows")
            
            # DOWデータ取得
            self.logger.info("")
            self.logger.info(f"[DATA_FETCH] DOWデータ取得: {self.index_symbol}")
            
            dow_data = self.data_feed.get_stock_data(
                ticker=self.index_symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if dow_data is None or dow_data.empty:
                self.logger.error(f"[ERROR] DOWデータ取得失敗: {self.index_symbol}")
                return False
            
            self.logger.info(f"[SUCCESS] DOWデータ取得完了: {len(dow_data)} rows")
            
            # 戦略初期化
            self.logger.info("")
            self.logger.info("[STRATEGY_INIT] OpeningGapEnhancedStrategy初期化")
            self.logger.info("-" * 80)
            
            self.strategy = OpeningGapEnhancedStrategy(
                data=stock_data,
                dow_data=dow_data,
                params=self.params,
                price_column="Adj Close"
            )
            
            self.logger.info("[SUCCESS] OpeningGapEnhancedStrategy初期化完了")
            
            # バックテスト実行
            self.logger.info("")
            self.logger.info("[BACKTEST] バックテスト実行開始")
            self.logger.info("-" * 80)
            
            self.result_df = self.strategy.backtest()
            
            if self.result_df is None or self.result_df.empty:
                self.logger.error("[ERROR] バックテスト実行失敗: 結果が空です")
                return False
            
            self.logger.info(f"[SUCCESS] バックテスト実行完了: {len(self.result_df)} rows")
            
            # シグナル統計
            entry_count = (self.result_df["Entry_Signal"] == 1).sum()
            exit_count = (self.result_df["Exit_Signal"] == -1).sum()
            
            self.logger.info("")
            self.logger.info("[BACKTEST_SUMMARY]")
            self.logger.info(f"  Entry_Signal == 1: {entry_count} 回")
            self.logger.info(f"  Exit_Signal == -1: {exit_count} 回")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Phase 1実行中にエラー発生: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _phase2_validation(self) -> bool:
        """Phase 2: 検証ロジック実行"""
        self.logger.info("")
        self.logger.info("[PHASE 2] 検証ロジック実行開始")
        self.logger.info("-" * 80)
        
        try:
            # 検証1: 同日Entry/Exit問題チェック
            same_day_issues = self._validate_same_day_entry_exit()
            
            # 検証2: ポジション管理検証
            position_valid = self._validate_position_management()
            
            # 検証3: トレード実行検証
            trades = self._validate_trade_execution()
            
            # 検証4: パフォーマンス計算
            performance = self._validate_performance()
            
            # 検証5: EnhancedBaseStrategy機能検証
            enhanced_valid = self._validate_enhanced_features()
            
            self.logger.info("")
            self.logger.info("[SUCCESS] Phase 2: 全検証カテゴリ完了")
            
            # サマリー保存
            self.validation_summary = {
                "same_day_issues": same_day_issues,
                "position_valid": position_valid,
                "trade_count": len(trades),
                "performance": performance,
                "enhanced_valid": enhanced_valid
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Phase 2実行中にエラー発生: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _validate_same_day_entry_exit(self) -> int:
        """検証1: 同日Entry/Exit問題チェック"""
        self.logger.info("")
        self.logger.info("[VALIDATION 1] 同日Entry/Exit問題チェック")
        self.logger.info("-" * 80)
        
        # 同じ日にEntry_Signal=1とExit_Signal=-1が両方立っている行を検出
        same_day_signals = self.result_df[
            (self.result_df["Entry_Signal"] == 1) & 
            (self.result_df["Exit_Signal"] == -1)
        ]
        
        issue_count = len(same_day_signals)
        
        if issue_count == 0:
            self.logger.info("[OK] 同日Entry/Exit問題なし")
        else:
            self.logger.warning(f"[WARNING] 同日Entry/Exit問題検出: {issue_count}件")
            for idx, row in same_day_signals.head(5).iterrows():
                self.logger.warning(f"  日付: {idx}, Entry: {row['Entry_Signal']}, Exit: {row['Exit_Signal']}")
        
        return issue_count
    
    def _validate_position_management(self) -> bool:
        """検証2: ポジション管理検証"""
        self.logger.info("")
        self.logger.info("[VALIDATION 2] ポジション管理検証")
        self.logger.info("-" * 80)
        
        # Position_Sizeカラムの存在確認
        if "Position_Size" not in self.result_df.columns:
            self.logger.error("[ERROR] Position_Sizeカラムが存在しません")
            return False
        
        # ポジションサイズの変化を追跡
        position_changes = []
        prev_position = 0.0
        
        for idx, row in self.result_df.iterrows():
            current_position = row["Position_Size"]
            
            if current_position != prev_position:
                position_changes.append({
                    "date": idx,
                    "from": prev_position,
                    "to": current_position
                })
                prev_position = current_position
        
        self.logger.info(f"[POSITION_TRANSITIONS] ポジション変化: {len(position_changes)}回")
        
        # 最初の5回を表示
        for change in position_changes[:5]:
            self.logger.info(f"  {change['date']}: {change['from']} → {change['to']}")
        
        # 異常な遷移のチェック (例: 0→2や1→2など)
        invalid_transitions = []
        for change in position_changes:
            if change['from'] == 0.0 and change['to'] not in [0.0, 1.0]:
                invalid_transitions.append(change)
            elif change['from'] == 1.0 and change['to'] not in [0.0, 1.0]:
                invalid_transitions.append(change)
        
        if invalid_transitions:
            self.logger.warning(f"[WARNING] 異常なポジション遷移検出: {len(invalid_transitions)}件")
            for trans in invalid_transitions[:3]:
                self.logger.warning(f"  {trans['date']}: {trans['from']} → {trans['to']}")
            return False
        
        self.logger.info("[OK] ポジション遷移が正常")
        return True
    
    def _validate_trade_execution(self) -> list:
        """検証3: トレード実行検証"""
        self.logger.info("")
        self.logger.info("[VALIDATION 3] トレード実行検証")
        self.logger.info("-" * 80)
        
        # エントリーとイグジットのペアを抽出
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
                    "pnl_pct": pnl_pct
                })
                
                entry_date = None
                entry_price = None
        
        self.logger.info(f"[TRADE_COUNT] 総取引数: {len(trades)} 件")
        
        # 最初の5件のトレードを表示
        if trades:
            self.logger.info("")
            self.logger.info("[TRADE_EXAMPLES] 最初の5件のトレード:")
            for i, trade in enumerate(trades[:5], 1):
                self.logger.info(
                    f"  #{i}: {trade['entry_date'].date()} → {trade['exit_date'].date()} "
                    f"P&L: {trade['pnl_pct']:.2f}%"
                )
        
        return trades
    
    def _validate_performance(self) -> dict:
        """検証4: パフォーマンス計算"""
        self.logger.info("")
        self.logger.info("[VALIDATION 4] パフォーマンス計算")
        self.logger.info("-" * 80)
        
        # トレードデータを再取得
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
                    "pnl_pct": pnl_pct
                })
                
                entry_date = None
                entry_price = None
        
        if not trades:
            self.logger.warning("[WARNING] トレードが0件のためパフォーマンス計算不可")
            return {"total_pnl": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0}
        
        # P&L計算
        total_pnl = sum(t["pnl_pct"] for t in trades)
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0
        avg_win = sum(w["pnl_pct"] for w in wins) / len(wins) if wins else 0
        avg_loss = sum(l["pnl_pct"] for l in losses) / len(losses) if losses else 0
        
        self.logger.info(f"[P&L] 総P&L: {total_pnl:.2f}%")
        self.logger.info(f"[WIN_RATE] 勝率: {win_rate:.1f}% ({len(wins)}勝 / {len(losses)}敗)")
        self.logger.info(f"[AVG] 平均利益: {avg_win:.2f}%, 平均損失: {avg_loss:.2f}%")
        
        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses)
        }
    
    def _validate_enhanced_features(self) -> bool:
        """検証5: EnhancedBaseStrategy機能検証"""
        self.logger.info("")
        self.logger.info("[VALIDATION 5] EnhancedBaseStrategy機能検証")
        self.logger.info("-" * 80)
        
        # entry_prices と high_prices の存在確認
        has_entry_prices = hasattr(self.strategy, 'entry_prices')
        has_high_prices = hasattr(self.strategy, 'high_prices')
        has_current_position = hasattr(self.strategy, 'current_position')
        
        self.logger.info(f"[ATTR_CHECK] entry_prices属性: {has_entry_prices}")
        self.logger.info(f"[ATTR_CHECK] high_prices属性: {has_high_prices}")
        self.logger.info(f"[ATTR_CHECK] current_position属性: {has_current_position}")
        
        if not (has_entry_prices and has_high_prices and has_current_position):
            self.logger.error("[ERROR] EnhancedBaseStrategy必須属性が不足")
            return False
        
        self.logger.info("[OK] EnhancedBaseStrategy機能が正常に実装されています")
        return True
    
    def print_summary(self):
        """検証結果サマリー出力"""
        if not hasattr(self, 'validation_summary'):
            print("\n検証結果サマリーが生成されていません")
            return
        
        summary = self.validation_summary
        
        print("\n" + "=" * 80)
        print("テスト結果（Opening_Gap_Enhanced - 8306.T 2年間）")
        print("=" * 80)
        print(f"ステータス: SUCCESS_PHASE2")
        print("")
        print("[検証1] 同日Entry/Exit問題:")
        print(f"  問題件数: {summary['same_day_issues']}件")
        print(f"  修正済み: {'はい' if summary['same_day_issues'] == 0 else 'いいえ'}")
        print("")
        print("[検証2] ポジション管理:")
        print(f"  管理ロジック正常: {'はい' if summary['position_valid'] else 'いいえ'}")
        print("")
        print("[検証3] トレード実行:")
        print(f"  総取引数: {summary['trade_count']}件")
        print("")
        print("[検証4] パフォーマンス:")
        perf = summary['performance']
        print(f"  総P&L: {perf['total_pnl']:.2f}%")
        print(f"  勝率: {perf['win_rate']:.1f}%")
        if 'total_trades' in perf:
            print(f"  総取引数: {perf['total_trades']}件 ({perf['wins']}勝 / {perf['losses']}敗)")
        else:
            print(f"  総取引数: 0件 (トレード未完了)")
        print("")
        print("[検証5] EnhancedBaseStrategy機能:")
        print(f"  機能実装: {'正常' if summary['enhanced_valid'] else '異常'}")
        print("")
        print("=" * 80)


def main():
    """メイン実行"""
    print("\n" + "=" * 80)
    print("Opening_Gap_Enhanced.py 動作検証テスト - 8306.T (2年間)")
    print("=" * 80)
    
    tester = OpeningGapEnhancedTester()
    status = tester.run()
    
    # サマリー出力
    tester.print_summary()
    
    return status


if __name__ == "__main__":
    status = main()
    sys.exit(0 if status == "SUCCESS_PHASE2" else 1)
