"""
test_vwap_breakout_8306T.py - VWAPBreakoutStrategy 単体テスト

テスト対象: strategies/VWAP_Breakout.py
テスト銘柄: 8306.T（三菱UFJフィナンシャル・グループ）
テスト期間: 2024/01/01 ~ 2024/12/31

テスト目的:
- VWAPBreakoutStrategyが正常に動作しているかの確認
- 作成者の意図通りに動作しているかを確認
- マルチ戦略システムのバグ特定のため、戦略に問題がないかの確認

検証項目:
1. データ品質の確認（欠損値、Volume、Adj Close）
2. VWAP計算の確認
3. シグナル生成の確認（Entry_Signal, Exit_Signal）
4. 取引実行の確認（エントリー日、エグジット日）
5. パフォーマンス計算（総損益、総損益率）※配当・株式分割調整済み
6. データ整合性（エントリー/エグジット数の一致）

注意事項:
- strategies/VWAP_Breakout.pyは修正しない
- yfinanceからのデータ取得失敗時はエラー
- Adj Close使用により配当・株式分割は自動調整される
- copilot-instructions.md準拠（モックデータ使用禁止）

Author: Backtest Project Team
Created: 2025-10-30
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# プロジェクトパス設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# テスト対象モジュール
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed

# ロガー設定（出力ディレクトリを事前作成）
log_dir = Path("tests/results")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'test_vwap_8306T.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class VWAPBreakoutTester:
    """VWAPBreakoutStrategy テストクラス"""
    
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
        self.index_data = None
        self.strategy = None
        self.result = None
        
        logger.info("=" * 80)
        logger.info("VWAPBreakoutStrategy 単体テスト開始")
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
            
            # 株価データ取得
            self.stock_data = data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            logger.info(f"[SUCCESS] 株価データ取得完了: {len(self.stock_data)} 行")
            logger.info(f"  カラム: {self.stock_data.columns.tolist()}")
            logger.info(f"  期間: {self.stock_data.index[0]} ~ {self.stock_data.index[-1]}")
            
            # インデックスデータ取得（日経225）
            # VWAP戦略は市場トレンドフィルターを使用する場合がある
            self.index_data = data_feed.get_index_data(
                index_symbol="^N225",  # 日経225
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            logger.info(f"[SUCCESS] インデックスデータ取得完了: {len(self.index_data)} 行")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] データ取得失敗: {e}")
            return False
    
    def verify_data_quality(self) -> Dict[str, Any]:
        """
        データ品質の確認
        
        Returns:
            dict: 検証結果
        """
        logger.info("[STEP 2] データ品質の確認")
        logger.info("-" * 80)
        
        # 欠損値チェック
        missing_values = self.stock_data.isnull().sum()
        has_missing = missing_values.sum() > 0
        
        logger.info(f"[DATA QUALITY CHECK]")
        logger.info(f"  欠損値: {'あり' if has_missing else 'なし'}")
        if has_missing:
            for col, count in missing_values[missing_values > 0].items():
                logger.info(f"    {col}: {count} 件")
        
        # Volume列の確認（VWAP計算に必須）
        has_volume = 'Volume' in self.stock_data.columns
        volume_valid = False
        if has_volume:
            volume_nonzero = (self.stock_data['Volume'] > 0).sum()
            volume_valid = volume_nonzero > 0
            logger.info(f"  Volume列: 存在")
            logger.info(f"    非ゼロデータ: {volume_nonzero} / {len(self.stock_data)} 行")
        else:
            logger.info(f"  Volume列: 存在しない（エラー）")
        
        # Adj Close列の確認（配当・株式分割調整）
        has_adj_close = 'Adj Close' in self.stock_data.columns
        logger.info(f"  Adj Close列: {'存在' if has_adj_close else '存在しない'}")
        
        # データ期間の確認
        expected_days = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days
        actual_days = len(self.stock_data)
        logger.info(f"  期待データ日数: 約{expected_days}日")
        logger.info(f"  実際のデータ行数: {actual_days}行")
        
        logger.info("")
        
        return {
            'has_missing': has_missing,
            'has_volume': has_volume,
            'volume_valid': volume_valid,
            'has_adj_close': has_adj_close,
            'data_sufficient': actual_days > 50  # 最低限のデータ数
        }
    
    def initialize_strategy(self) -> bool:
        """
        戦略初期化
        
        Returns:
            bool: 成功時True
        """
        logger.info("[STEP 3] 戦略初期化")
        logger.info("-" * 80)
        
        try:
            # VWAP戦略初期化（デフォルトパラメータ使用）
            self.strategy = VWAPBreakoutStrategy(
                data=self.stock_data,
                index_data=self.index_data,
                params=None,  # デフォルトパラメータ使用
                price_column="Adj Close",
                volume_column="Volume"
            )
            
            logger.info("[SUCCESS] 戦略初期化完了")
            logger.info(f"  戦略クラス: {self.strategy.__class__.__name__}")
            logger.info(f"  価格カラム: Adj Close")
            logger.info(f"  出来高カラム: Volume")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] 戦略初期化失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def verify_vwap_calculation(self) -> Dict[str, Any]:
        """
        VWAP計算の確認
        
        Returns:
            dict: 検証結果
        """
        logger.info("[STEP 4] VWAP計算の確認")
        logger.info("-" * 80)
        
        # VWAP列の存在確認
        has_vwap = 'VWAP' in self.strategy.data.columns
        
        if has_vwap:
            vwap_values = self.strategy.data['VWAP']
            vwap_valid = (~vwap_values.isnull()).sum()
            vwap_mean = vwap_values.mean()
            
            logger.info(f"[VWAP CHECK]")
            logger.info(f"  VWAP列: 存在")
            logger.info(f"  有効なVWAP値: {vwap_valid} / {len(vwap_values)} 行")
            logger.info(f"  VWAP平均値: {vwap_mean:.2f}")
            
            # サンプルデータ表示
            logger.info(f"  最新5行のVWAP値:")
            for idx in self.strategy.data.index[-5:]:
                date = idx.strftime('%Y-%m-%d')
                close = self.strategy.data.loc[idx, 'Adj Close']
                vwap = self.strategy.data.loc[idx, 'VWAP']
                logger.info(f"    {date}: Close={close:.2f}, VWAP={vwap:.2f}")
        else:
            logger.error(f"[VWAP CHECK] VWAP列が存在しません")
            vwap_valid = 0
        
        logger.info("")
        
        return {
            'has_vwap': has_vwap,
            'vwap_valid_count': vwap_valid if has_vwap else 0
        }
    
    def run_backtest(self) -> bool:
        """
        バックテスト実行
        
        Returns:
            bool: 成功時True
        """
        logger.info("[STEP 5] バックテスト実行")
        logger.info("-" * 80)
        
        try:
            # バックテスト実行（copilot-instructions.md準拠：必ず実行）
            self.result = self.strategy.backtest()
            
            logger.info("[SUCCESS] バックテスト実行完了")
            logger.info(f"  結果データ行数: {len(self.result)}")
            logger.info(f"  カラム: {self.result.columns.tolist()}")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAILED] バックテスト実行失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def verify_signals(self) -> Dict[str, Any]:
        """
        シグナル生成の検証
        
        Returns:
            dict: 検証結果
        """
        logger.info("[STEP 6] シグナル生成の検証")
        logger.info("-" * 80)
        
        # エントリーシグナルのカウント
        entry_signals = self.result[self.result['Entry_Signal'] == 1]
        entry_count = len(entry_signals)
        
        # エグジットシグナルのカウント
        exit_signals = self.result[self.result['Exit_Signal'] == -1]
        exit_count = len(exit_signals)
        
        # 整合性チェック
        signals_match = (entry_count == exit_count)
        
        logger.info(f"[SIGNAL CHECK]")
        logger.info(f"  Entry_Signal == 1: {entry_count} 回")
        logger.info(f"  Exit_Signal == -1: {exit_count} 回")
        logger.info(f"  整合性: {'OK' if signals_match else 'NG - 不一致!'}")
        
        if not signals_match:
            logger.warning(f"  エントリーとエグジットの数が一致しません")
        
        # エントリーシグナルの詳細
        if entry_count > 0:
            logger.info(f"  エントリー日一覧:")
            for idx in entry_signals.index[:10]:  # 最初の10件
                date = idx.strftime('%Y-%m-%d')
                price = self.result.loc[idx, 'Adj Close']
                vwap = self.result.loc[idx, 'VWAP']
                logger.info(f"    {date}: Price={price:.2f}, VWAP={vwap:.2f}")
            if entry_count > 10:
                logger.info(f"    ... 他 {entry_count - 10} 件")
        else:
            logger.warning(f"  エントリーシグナルが発生しませんでした")
        
        logger.info("")
        
        return {
            'entry_count': entry_count,
            'exit_count': exit_count,
            'signals_match': signals_match
        }
    
    def extract_trades(self) -> List[Dict[str, Any]]:
        """
        取引詳細の抽出
        
        Returns:
            list: 取引リスト
        """
        logger.info("[STEP 7] 取引詳細の抽出")
        logger.info("-" * 80)
        
        trades = []
        entry_signals = self.result[self.result['Entry_Signal'] == 1]
        exit_signals = self.result[self.result['Exit_Signal'] == -1]
        
        # エントリーとエグジットをペアリング
        for entry_idx, exit_idx in zip(entry_signals.index, exit_signals.index):
            entry_date = entry_idx.strftime('%Y-%m-%d')
            exit_date = exit_idx.strftime('%Y-%m-%d')
            
            entry_price = self.result.loc[entry_idx, 'Adj Close']
            exit_price = self.result.loc[exit_idx, 'Adj Close']
            
            entry_vwap = self.result.loc[entry_idx, 'VWAP']
            exit_vwap = self.result.loc[exit_idx, 'VWAP']
            
            # 損益計算（Adj Close使用により配当・株式分割調整済み）
            pnl = exit_price - entry_price
            return_pct = (pnl / entry_price) * 100
            
            # 保有期間
            holding_days = (exit_idx - entry_idx).days
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_vwap': entry_vwap,
                'exit_vwap': exit_vwap,
                'pnl': pnl,
                'return_pct': return_pct,
                'holding_days': holding_days
            })
        
        logger.info(f"[TRADES EXTRACTED] {len(trades)} 件の取引を抽出")
        logger.info("")
        
        return trades
    
    def calculate_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        パフォーマンス計算
        
        Args:
            trades: 取引リスト
        
        Returns:
            dict: パフォーマンス指標
        """
        logger.info("[STEP 8] パフォーマンス計算")
        logger.info("-" * 80)
        
        if not trades:
            logger.warning("[PERFORMANCE] 取引がないためパフォーマンス計算をスキップ")
            logger.info("")
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'avg_holding_days': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0
            }
        
        # 総損益計算
        total_pnl = sum(trade['pnl'] for trade in trades)
        
        # 総損益率計算（累積リターン）
        total_return_pct = sum(trade['return_pct'] for trade in trades)
        
        # 平均保有期間
        avg_holding_days = sum(trade['holding_days'] for trade in trades) / len(trades)
        
        # 勝率計算
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0.0
        
        # 最大ドローダウン計算
        cumulative_pnl = 0
        max_cumulative = 0
        max_drawdown = 0
        for trade in trades:
            cumulative_pnl += trade['pnl']
            max_cumulative = max(max_cumulative, cumulative_pnl)
            drawdown = max_cumulative - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        logger.info(f"[PERFORMANCE]")
        logger.info(f"  総取引回数: {len(trades)} 回")
        logger.info(f"  総損益: {total_pnl:,.2f} 円 (Adj Close使用・配当/分割調整済み)")
        logger.info(f"  総損益率: {total_return_pct:.2f} %")
        logger.info(f"  平均保有期間: {avg_holding_days:.1f} 日")
        logger.info(f"  勝率: {win_rate:.1f} % ({len(winning_trades)}/{len(trades)})")
        logger.info(f"  最大ドローダウン: {max_drawdown:,.2f} 円")
        logger.info("")
        
        return {
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'avg_holding_days': avg_holding_days,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }
    
    def verify_position_continuity(self) -> Dict[str, Any]:
        """
        Position列の連続性検証
        
        Returns:
            dict: 検証結果
        """
        logger.info("[STEP 9] Position列の連続性検証")
        logger.info("-" * 80)
        
        if 'Position' not in self.result.columns:
            logger.warning("[POSITION CHECK] Position列が存在しません")
            logger.info("")
            return {'has_position': False}
        
        # ポジション保有日数
        position_days = (self.result['Position'] == 1).sum()
        
        # 未決済ポジション確認（最終日のPosition）
        last_position = self.result['Position'].iloc[-1]
        has_open_position = (last_position == 1)
        
        logger.info(f"[POSITION CHECK]")
        logger.info(f"  ポジション保有日数: {position_days} 日")
        logger.info(f"  最終日のポジション: {last_position}")
        logger.info(f"  未決済ポジション: {'あり（要強制決済）' if has_open_position else 'なし'}")
        logger.info("")
        
        return {
            'has_position': True,
            'position_days': position_days,
            'has_open_position': has_open_position
        }
    
    def save_results(self, trades: List[Dict[str, Any]], performance: Dict[str, Any]) -> None:
        """
        結果をCSVに保存
        
        Args:
            trades: 取引リスト
            performance: パフォーマンス指標
        """
        logger.info("[STEP 10] 結果保存")
        logger.info("-" * 80)
        
        try:
            output_dir = Path("tests/results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 取引一覧CSV
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_csv = output_dir / "vwap_8306T_trades.csv"
                trades_df.to_csv(trades_csv, index=False, encoding='utf-8-sig')
                logger.info(f"[SUCCESS] 取引一覧を保存: {trades_csv}")
            
            # サマリーCSV
            summary_df = pd.DataFrame([performance])
            summary_csv = output_dir / "vwap_8306T_summary.csv"
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
            logger.info(f"[SUCCESS] サマリーを保存: {summary_csv}")
            
            # 全シグナルとVWAP値CSV
            signals_df = self.result[[
                'Adj Close', 'VWAP', 'Entry_Signal', 'Exit_Signal', 'Position'
            ]].copy()
            signals_csv = output_dir / "vwap_8306T_signals.csv"
            signals_df.to_csv(signals_csv, encoding='utf-8-sig')
            logger.info(f"[SUCCESS] 全シグナルを保存: {signals_csv}")
            
            logger.info("")
            
        except Exception as e:
            logger.error(f"[FAILED] 結果保存失敗: {e}")
            logger.info("")
    
    def print_trade_details(self, trades: List[Dict[str, Any]]) -> None:
        """
        取引詳細をログ出力
        
        Args:
            trades: 取引リスト
        """
        logger.info("[取引詳細]")
        logger.info("-" * 80)
        
        if not trades:
            logger.info("取引がありません")
            logger.info("")
            return
        
        for i, trade in enumerate(trades, 1):
            logger.info(f"[取引 {i}]")
            logger.info(f"  エントリー日: {trade['entry_date']}")
            logger.info(f"  エントリー価格: {trade['entry_price']:.2f} 円")
            logger.info(f"  エントリー時VWAP: {trade['entry_vwap']:.2f} 円")
            logger.info(f"  エグジット日: {trade['exit_date']}")
            logger.info(f"  エグジット価格: {trade['exit_price']:.2f} 円")
            logger.info(f"  エグジット時VWAP: {trade['exit_vwap']:.2f} 円")
            logger.info(f"  損益: {trade['pnl']:,.2f} 円 ({trade['return_pct']:+.2f}%)")
            logger.info(f"  保有期間: {trade['holding_days']} 日")
            logger.info("")
    
    def run_full_test(self) -> bool:
        """
        フルテスト実行
        
        Returns:
            bool: テスト成功時True
        """
        # Step 1: データ取得
        if not self.fetch_data():
            logger.error("[ABORT] データ取得失敗のためテスト中止")
            return False
        
        # Step 2: データ品質確認
        quality_result = self.verify_data_quality()
        if not quality_result['has_volume'] or not quality_result['volume_valid']:
            logger.error("[ABORT] Volume列が不正のためテスト中止")
            return False
        
        # Step 3: 戦略初期化
        if not self.initialize_strategy():
            logger.error("[ABORT] 戦略初期化失敗のためテスト中止")
            return False
        
        # Step 4: VWAP計算確認
        vwap_result = self.verify_vwap_calculation()
        if not vwap_result['has_vwap']:
            logger.error("[ABORT] VWAP計算失敗のためテスト中止")
            return False
        
        # Step 5: バックテスト実行（copilot-instructions.md準拠：必須）
        if not self.run_backtest():
            logger.error("[ABORT] バックテスト実行失敗のためテスト中止")
            return False
        
        # Step 6: シグナル検証
        signal_result = self.verify_signals()
        
        # Step 7: 取引抽出
        trades = self.extract_trades()
        
        # Step 8: パフォーマンス計算
        performance = self.calculate_performance(trades)
        
        # Step 9: Position連続性検証
        position_result = self.verify_position_continuity()
        
        # Step 10: 結果保存
        self.save_results(trades, performance)
        
        # Step 11: 取引詳細出力
        self.print_trade_details(trades)
        
        # テスト成功条件チェック
        logger.info("[TEST RESULT]")
        logger.info("=" * 80)
        
        # 条件1: エントリー回数 >= 0（市場条件により0も許容）
        condition1 = signal_result['entry_count'] >= 0
        logger.info(f"  条件1（エントリー回数 >= 0）: {'PASS' if condition1 else 'FAIL'}")
        
        # 条件2: エントリー回数 == エグジット回数
        condition2 = signal_result['signals_match']
        logger.info(f"  条件2（エントリー == エグジット）: {'PASS' if condition2 else 'FAIL'}")
        
        # 条件3: VWAP計算正常
        condition3 = vwap_result['has_vwap'] and vwap_result['vwap_valid_count'] > 0
        logger.info(f"  条件3（VWAP計算正常）: {'PASS' if condition3 else 'FAIL'}")
        
        # 条件4: データ不整合なし
        condition4 = not quality_result['has_missing'] and quality_result['volume_valid']
        logger.info(f"  条件4（データ不整合なし）: {'PASS' if condition4 else 'FAIL'}")
        
        # 総合判定
        all_passed = condition1 and condition2 and condition3 and condition4
        logger.info("")
        logger.info(f"  総合結果: {'TEST PASSED' if all_passed else 'TEST FAILED'}")
        logger.info("=" * 80)
        
        return all_passed


def main():
    """メインエントリーポイント"""
    # テスト設定
    ticker = "8306.T"  # 三菱UFJフィナンシャル・グループ
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # テスト実行
    tester = VWAPBreakoutTester(ticker, start_date, end_date)
    success = tester.run_full_test()
    
    # 終了コード
    exit_code = 0 if success else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
