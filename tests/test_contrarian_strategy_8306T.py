"""
test_contrarian_strategy_8306T.py - ContrarianStrategy 単体テスト

テスト対象: strategies/contrarian_strategy.py
テスト銘柄: 8306.T（三菱UFJフィナンシャル・グループ）
テスト期間: 2023/01/01 ~ 2024/12/31（2年間）

テスト目的:
- ContrarianStrategyが正常に動作しているかの確認
- 作成者の意図通りに動作しているかを確認
- マルチ戦略システムのバグ特定のため、戦略に問題がないかの確認

検証項目:
1. RSIシグナル生成の確認（RSI <= 30 でエントリー）
2. ギャップダウン検出の確認
3. ピンバー検出の確認
4. トレンドフィルターの確認（レンジ相場のみでエントリー）
5. 最大保有日数の確認（5日以内にエグジット）
6. トレーリングストップの確認
7. 総損益の計算

Author: Backtest Project Team
Created: 2025-10-23
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import logging

# プロジェクトパス設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# テスト対象モジュール
from strategies.contrarian_strategy import ContrarianStrategy
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed

# ロガー設定（出力ディレクトリを事前作成）
log_dir = Path("tests/results")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'test_contrarian_8306T.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class ContrarianStrategyTester:
    """ContrarianStrategy テストクラス"""
    
    def __init__(self, ticker: str, start_date: str, end_date: str):
        """
        初期化
        
        Args:
            ticker: ティッカーシンボル
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = None
        self.strategy = None
        self.result = None
        
        logger.info("=" * 80)
        logger.info("ContrarianStrategy 単体テスト開始")
        logger.info("=" * 80)
        logger.info(f"テスト銘柄: {ticker}")
        logger.info(f"テスト期間: {start_date} ~ {end_date}")
        logger.info("")
    
    def fetch_data(self) -> bool:
        """
        yfinanceからデータ取得
        
        Returns:
            bool: 成功時True
        """
        logger.info("[STEP 1] データ取得")
        logger.info("-" * 80)
        
        try:
            data_feed = YFinanceDataFeed()
            self.stock_data = data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            logger.info(f"[SUCCESS] データ取得完了: {len(self.stock_data)} 行")
            logger.info(f"  カラム: {self.stock_data.columns.tolist()}")
            logger.info(f"  期間: {self.stock_data.index[0]} ~ {self.stock_data.index[-1]}")
            logger.info(f"  最初の価格: {self.stock_data['Close'].iloc[0]:.2f} 円")
            logger.info(f"  最後の価格: {self.stock_data['Close'].iloc[-1]:.2f} 円")
            logger.info("")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] データ取得失敗: {e}")
            return False
    
    def initialize_strategy(self) -> bool:
        """
        戦略初期化
        
        Returns:
            bool: 成功時True
        """
        logger.info("[STEP 2] 戦略初期化")
        logger.info("-" * 80)
        
        try:
            # デフォルトパラメータで初期化
            self.strategy = ContrarianStrategy(
                data=self.stock_data,
                price_column="Close"
            )
            
            logger.info("[SUCCESS] ContrarianStrategy 初期化完了")
            logger.info(f"  戦略パラメータ:")
            for key, value in self.strategy.params.items():
                logger.info(f"    {key}: {value}")
            logger.info("")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] 戦略初期化失敗: {e}")
            return False
    
    def run_backtest(self) -> bool:
        """
        バックテスト実行
        
        Returns:
            bool: 成功時True
        """
        logger.info("[STEP 3] バックテスト実行")
        logger.info("-" * 80)
        
        try:
            self.result = self.strategy.backtest()
            
            # シグナルカウント
            entry_count = (self.result['Entry_Signal'] == 1).sum()
            exit_count = abs((self.result['Exit_Signal'] == -1).sum())
            
            logger.info(f"[SUCCESS] バックテスト完了")
            logger.info(f"  エントリーシグナル: {entry_count} 回")
            logger.info(f"  エグジットシグナル: {exit_count} 回")
            
            # RSI統計
            if 'RSI' in self.result.columns:
                rsi_min = self.result['RSI'].min()
                rsi_max = self.result['RSI'].max()
                rsi_mean = self.result['RSI'].mean()
                logger.info(f"  RSI統計: 最小={rsi_min:.2f}, 最大={rsi_max:.2f}, 平均={rsi_mean:.2f}")
            
            logger.info("")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] バックテスト実行失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def analyze_signals(self) -> Dict[str, Any]:
        """
        シグナル詳細分析
        
        Returns:
            Dict[str, Any]: 分析結果
        """
        logger.info("[STEP 4] シグナル詳細分析")
        logger.info("-" * 80)
        
        analysis = {
            "entry_signals": [],
            "exit_signals": [],
            "gap_down_count": 0,
            "pin_bar_count": 0,
            "rsi_oversold_count": 0,
            "range_bound_entries": 0
        }
        
        try:
            # エントリーシグナルの詳細
            entry_dates = self.result[self.result['Entry_Signal'] == 1].index
            for date in entry_dates:
                idx = self.result.index.get_loc(date)
                
                if idx > 0:
                    current_price = self.result['Close'].iloc[idx]
                    previous_price = self.result['Close'].iloc[idx - 1]
                    rsi = self.result['RSI'].iloc[idx] if 'RSI' in self.result.columns else None
                    
                    # ギャップダウン判定
                    gap_threshold = self.strategy.params["gap_threshold"]
                    is_gap_down = current_price < previous_price * (1.0 - gap_threshold)
                    if is_gap_down:
                        analysis["gap_down_count"] += 1
                    
                    # ピンバー判定（簡易）
                    if 'High' in self.result.columns and 'Low' in self.result.columns:
                        high = self.result['High'].iloc[idx]
                        low = self.result['Low'].iloc[idx]
                        pin_bar_ratio = self.strategy.params["pin_bar_ratio"]
                        is_pin_bar = (high - current_price) > pin_bar_ratio * (current_price - low)
                        if is_pin_bar:
                            analysis["pin_bar_count"] += 1
                    
                    # RSI判定
                    if rsi is not None and rsi <= self.strategy.params["rsi_oversold"]:
                        analysis["rsi_oversold_count"] += 1
                    
                    signal_info = {
                        "date": date,
                        "price": current_price,
                        "rsi": rsi,
                        "is_gap_down": is_gap_down,
                        "gap_pct": ((current_price - previous_price) / previous_price * 100) if previous_price > 0 else 0
                    }
                    analysis["entry_signals"].append(signal_info)
                    
                    logger.info(f"  エントリー {len(analysis['entry_signals'])}: {date.strftime('%Y-%m-%d')}")
                    logger.info(f"    価格: {current_price:.2f} 円")
                    logger.info(f"    RSI: {rsi:.2f}" if rsi is not None else "    RSI: N/A")
                    logger.info(f"    ギャップダウン: {'はい' if is_gap_down else 'いいえ'} ({signal_info['gap_pct']:.2f}%)")
            
            # エグジットシグナルの詳細
            exit_dates = self.result[self.result['Exit_Signal'] == -1].index
            for date in exit_dates:
                idx = self.result.index.get_loc(date)
                current_price = self.result['Close'].iloc[idx]
                rsi = self.result['RSI'].iloc[idx] if 'RSI' in self.result.columns else None
                
                # 対応するエントリーを探す
                entry_idx = None
                for entry_date in entry_dates:
                    entry_loc = self.result.index.get_loc(entry_date)
                    if entry_loc < idx:
                        entry_idx = entry_loc
                
                days_held = idx - entry_idx if entry_idx is not None else None
                
                signal_info = {
                    "date": date,
                    "price": current_price,
                    "rsi": rsi,
                    "days_held": days_held
                }
                analysis["exit_signals"].append(signal_info)
                
                logger.info(f"  エグジット {len(analysis['exit_signals'])}: {date.strftime('%Y-%m-%d')}")
                logger.info(f"    価格: {current_price:.2f} 円")
                logger.info(f"    RSI: {rsi:.2f}" if rsi is not None else "    RSI: N/A")
                logger.info(f"    保有日数: {days_held} 日" if days_held is not None else "    保有日数: N/A")
            
            # サマリー
            logger.info("")
            logger.info("  === シグナル統計 ===")
            logger.info(f"  総エントリー数: {len(analysis['entry_signals'])}")
            logger.info(f"  総エグジット数: {len(analysis['exit_signals'])}")
            logger.info(f"  ギャップダウン検出: {analysis['gap_down_count']} 回")
            logger.info(f"  ピンバー検出: {analysis['pin_bar_count']} 回")
            logger.info(f"  RSI過売り検出: {analysis['rsi_oversold_count']} 回")
            logger.info("")
            
            return analysis
            
        except Exception as e:
            logger.error(f"[ERROR] シグナル分析失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return analysis
    
    def calculate_performance(self, signal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        パフォーマンス計算
        
        Args:
            signal_analysis: シグナル分析結果
            
        Returns:
            Dict[str, Any]: パフォーマンス結果
        """
        logger.info("[STEP 5] パフォーマンス計算")
        logger.info("-" * 80)
        
        performance = {
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "trades": [],
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_hold_days": 0.0
        }
        
        try:
            entry_signals = signal_analysis["entry_signals"]
            exit_signals = signal_analysis["exit_signals"]
            
            # 各取引のPnL計算
            for i, (entry, exit) in enumerate(zip(entry_signals, exit_signals), 1):
                entry_price = entry["price"]
                exit_price = exit["price"]
                pnl = exit_price - entry_price
                pnl_pct = (pnl / entry_price * 100) if entry_price > 0 else 0
                
                trade = {
                    "trade_no": i,
                    "entry_date": entry["date"],
                    "exit_date": exit["date"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "days_held": exit.get("days_held", 0)
                }
                performance["trades"].append(trade)
                
                performance["total_pnl"] += pnl
                
                if pnl > 0:
                    performance["winning_trades"] += 1
                else:
                    performance["losing_trades"] += 1
                
                logger.info(f"  取引 {i}:")
                logger.info(f"    エントリー: {entry['date'].strftime('%Y-%m-%d')} @ {entry_price:.2f} 円")
                logger.info(f"    エグジット: {exit['date'].strftime('%Y-%m-%d')} @ {exit_price:.2f} 円")
                logger.info(f"    損益: {pnl:.2f} 円 ({pnl_pct:+.2f}%)")
                logger.info(f"    保有日数: {trade['days_held']} 日")
            
            # 統計計算
            total_trades = len(performance["trades"])
            if total_trades > 0:
                performance["win_rate"] = (performance["winning_trades"] / total_trades) * 100
                performance["avg_hold_days"] = sum(t["days_held"] for t in performance["trades"]) / total_trades
                
                initial_price = self.stock_data['Close'].iloc[0]
                performance["total_pnl_pct"] = (performance["total_pnl"] / initial_price * 100) if initial_price > 0 else 0
            
            # サマリー
            logger.info("")
            logger.info("  === パフォーマンスサマリー ===")
            logger.info(f"  総取引数: {total_trades}")
            logger.info(f"  勝ちトレード: {performance['winning_trades']}")
            logger.info(f"  負けトレード: {performance['losing_trades']}")
            logger.info(f"  勝率: {performance['win_rate']:.2f}%")
            logger.info(f"  平均保有日数: {performance['avg_hold_days']:.2f} 日")
            logger.info(f"  総損益: {performance['total_pnl']:.2f} 円")
            logger.info(f"  総損益率: {performance['total_pnl_pct']:+.2f}%")
            logger.info("")
            
            return performance
            
        except Exception as e:
            logger.error(f"[ERROR] パフォーマンス計算失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return performance
    
    def export_results(self, signal_analysis: Dict[str, Any], performance: Dict[str, Any]) -> bool:
        """
        結果をCSVとテキストレポートに出力
        
        Args:
            signal_analysis: シグナル分析結果
            performance: パフォーマンス結果
            
        Returns:
            bool: 成功時True
        """
        logger.info("[STEP 6] 結果出力")
        logger.info("-" * 80)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 取引履歴CSV
            if performance["trades"]:
                trades_df = pd.DataFrame(performance["trades"])
                trades_csv = log_dir / f"contrarian_trades_{self.ticker}_{timestamp}.csv"
                trades_df.to_csv(trades_csv, index=False, encoding='utf-8-sig')
                logger.info(f"[SUCCESS] 取引履歴CSV出力: {trades_csv}")
            
            # サマリーCSV
            summary_data = {
                "ticker": [self.ticker],
                "start_date": [self.start_date],
                "end_date": [self.end_date],
                "total_trades": [len(performance["trades"])],
                "winning_trades": [performance["winning_trades"]],
                "losing_trades": [performance["losing_trades"]],
                "win_rate_pct": [performance["win_rate"]],
                "avg_hold_days": [performance["avg_hold_days"]],
                "total_pnl": [performance["total_pnl"]],
                "total_pnl_pct": [performance["total_pnl_pct"]],
                "gap_down_count": [signal_analysis["gap_down_count"]],
                "pin_bar_count": [signal_analysis["pin_bar_count"]],
                "rsi_oversold_count": [signal_analysis["rsi_oversold_count"]]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = log_dir / f"contrarian_summary_{self.ticker}_{timestamp}.csv"
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
            logger.info(f"[SUCCESS] サマリーCSV出力: {summary_csv}")
            
            # テキストレポート
            report_path = log_dir / f"contrarian_report_{self.ticker}_{timestamp}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ContrarianStrategy テスト結果レポート\n")
                f.write("=" * 80 + "\n")
                f.write(f"テスト日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"テスト銘柄: {self.ticker}\n")
                f.write(f"テスト期間: {self.start_date} ~ {self.end_date}\n")
                f.write("\n")
                
                f.write("=== 戦略パラメータ ===\n")
                for key, value in self.strategy.params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                f.write("=== シグナル統計 ===\n")
                f.write(f"  総エントリー数: {len(signal_analysis['entry_signals'])}\n")
                f.write(f"  総エグジット数: {len(signal_analysis['exit_signals'])}\n")
                f.write(f"  ギャップダウン検出: {signal_analysis['gap_down_count']} 回\n")
                f.write(f"  ピンバー検出: {signal_analysis['pin_bar_count']} 回\n")
                f.write(f"  RSI過売り検出: {signal_analysis['rsi_oversold_count']} 回\n")
                f.write("\n")
                
                f.write("=== パフォーマンス ===\n")
                f.write(f"  総取引数: {len(performance['trades'])}\n")
                f.write(f"  勝ちトレード: {performance['winning_trades']}\n")
                f.write(f"  負けトレード: {performance['losing_trades']}\n")
                f.write(f"  勝率: {performance['win_rate']:.2f}%\n")
                f.write(f"  平均保有日数: {performance['avg_hold_days']:.2f} 日\n")
                f.write(f"  総損益: {performance['total_pnl']:.2f} 円\n")
                f.write(f"  総損益率: {performance['total_pnl_pct']:+.2f}%\n")
                f.write("\n")
                
                f.write("=== 取引詳細 ===\n")
                for trade in performance["trades"]:
                    f.write(f"  取引 {trade['trade_no']}:\n")
                    f.write(f"    エントリー: {trade['entry_date'].strftime('%Y-%m-%d')} @ {trade['entry_price']:.2f} 円\n")
                    f.write(f"    エグジット: {trade['exit_date'].strftime('%Y-%m-%d')} @ {trade['exit_price']:.2f} 円\n")
                    f.write(f"    損益: {trade['pnl']:.2f} 円 ({trade['pnl_pct']:+.2f}%)\n")
                    f.write(f"    保有日数: {trade['days_held']} 日\n")
                    f.write("\n")
            
            logger.info(f"[SUCCESS] テキストレポート出力: {report_path}")
            logger.info("")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] 結果出力失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_full_test(self) -> bool:
        """
        全テストステップ実行
        
        Returns:
            bool: 全ステップ成功時True
        """
        # STEP 1: データ取得
        if not self.fetch_data():
            logger.error("テスト中断: データ取得失敗")
            return False
        
        # STEP 2: 戦略初期化
        if not self.initialize_strategy():
            logger.error("テスト中断: 戦略初期化失敗")
            return False
        
        # STEP 3: バックテスト実行
        if not self.run_backtest():
            logger.error("テスト中断: バックテスト実行失敗")
            return False
        
        # STEP 4: シグナル分析
        signal_analysis = self.analyze_signals()
        
        # STEP 5: パフォーマンス計算
        performance = self.calculate_performance(signal_analysis)
        
        # STEP 6: 結果出力
        if not self.export_results(signal_analysis, performance):
            logger.error("テスト中断: 結果出力失敗")
            return False
        
        # 最終レポート
        logger.info("=" * 80)
        logger.info("テスト完了")
        logger.info("=" * 80)
        logger.info(f"総エントリー数: {len(signal_analysis['entry_signals'])}")
        logger.info(f"総エグジット数: {len(signal_analysis['exit_signals'])}")
        logger.info(f"総取引数: {len(performance['trades'])}")
        logger.info(f"勝率: {performance['win_rate']:.2f}%")
        logger.info(f"平均保有日数: {performance['avg_hold_days']:.2f} 日")
        logger.info(f"総損益: {performance['total_pnl']:.2f} 円 ({performance['total_pnl_pct']:+.2f}%)")
        logger.info("")
        
        # 検証チェック
        logger.info("=== 検証項目チェック ===")
        checks = {
            "データ取得成功": len(self.stock_data) > 0,
            "戦略初期化成功": self.strategy is not None,
            "バックテスト実行成功": self.result is not None,
            "エントリーシグナル生成": len(signal_analysis['entry_signals']) > 0,
            "エグジットシグナル生成": len(signal_analysis['exit_signals']) > 0,
            "取引実行確認": len(performance['trades']) > 0,
            "RSI計算完了": 'RSI' in self.result.columns,
            "最大保有日数遵守": all(t['days_held'] <= self.strategy.params['max_hold_days'] for t in performance['trades'])
        }
        
        for check_name, check_result in checks.items():
            status = "[OK]" if check_result else "[NG]"
            logger.info(f"{status} {check_name}")
        
        all_passed = all(checks.values())
        logger.info("")
        logger.info("=" * 80)
        if all_passed:
            logger.info("[SUCCESS] 全検証項目クリア")
        else:
            logger.info("[WARNING] 一部検証項目が失敗")
        logger.info("=" * 80)
        
        return all_passed


def main():
    """メイン実行"""
    tester = ContrarianStrategyTester(
        ticker="8306.T",
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    
    success = tester.run_full_test()
    
    if success:
        logger.info("テスト正常終了")
        return 0
    else:
        logger.error("テスト異常終了")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
