"""
Phase B-3-1: 複数銘柄バックテストテスト

Phase B-1設定（全フィルターOFF）で複数の日本株銘柄をテストし、
Opening Gap戦略の汎用性を検証します。

テスト対象:
- 6758.T (ソニーグループ)
- 9984.T (ソフトバンクグループ) 
- 8306.T (三菱UFJフィナンシャル・グループ)

比較基準:
- 7203.T (トヨタ自動車) Phase B-1結果: 35.6%勝率, +38.43% P&L

主な機能:
- 複数銘柄の並列テスト実行
- Phase B-1設定の適用
- 銘柄間パフォーマンス比較
- セクター別分析
- 汎用性評価レポート生成

統合コンポーネント:
- strategies/Opening_Gap.py: テスト対象戦略
- main_system/data_acquisition/yfinance_data_feed.py: データ取得
- Phase B-1設定: gap_threshold=0.02, stop_loss=0.10, 全フィルター無効

セーフティ機能/注意事項:
- copilot-instructions.md準拠
- 実データ使用必須（モックデータ禁止）
- 実際のbacktest()呼び出し必須
- 取引件数 > 0 を検証

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from strategies.Opening_Gap import OpeningGapStrategy


class MultiTickerTester:
    """
    複数銘柄テストクラス（Phase B-3-1）
    """
    
    def __init__(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31"):
        """
        初期化
        
        Args:
            start_date: テスト期間開始日（株式分割後の期間）
            end_date: テスト期間終了日
        """
        self.start_date = start_date
        self.end_date = end_date
        
        # ロガー設定
        self.logger = setup_logger(
            "MultiTickerTester",
            log_file="logs/test_multi_ticker_phase_b3.log"
        )
        
        self.logger.info("=" * 80)
        self.logger.info("Phase B-3-1: 複数銘柄バックテストテスト開始")
        self.logger.info(f"期間: {start_date} ~ {end_date}")
        self.logger.info("=" * 80)
        
        # データフィード初期化
        self.data_feed = YFinanceDataFeed()
        
        # テスト対象銘柄
        self.tickers = {
            "7203.T": "トヨタ自動車（ベースライン）",
            "6758.T": "ソニーグループ",
            "9984.T": "ソフトバンクグループ",
            "8306.T": "三菱UFJフィナンシャル・グループ"
        }
        
        # Phase B-1設定（全フィルターOFF）
        self.phase_b1_params = {
            "gap_threshold": 0.02,
            "stop_loss": 0.10,
            "take_profit": 0.10,
            "max_hold_days": 20,
            "trailing_stop_pct": 0.20,
            "trend_filter_enabled": False,
            "dow_filter_enabled": False,
            "volatility_filter": False,
            "gap_direction": "up"
        }
        
        # 結果格納
        self.results = {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        全銘柄テスト実行
        
        Returns:
            全テスト結果
        """
        self.logger.info("\n[START] 全銘柄テスト開始")
        self.logger.info("-" * 80)
        
        for ticker, name in self.tickers.items():
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"銘柄テスト開始: {ticker} ({name})")
            self.logger.info(f"{'=' * 80}")
            
            try:
                result = self._test_single_ticker(ticker, name)
                self.results[ticker] = result
                
                if result['status'] == 'SUCCESS':
                    self.logger.info(f"[SUCCESS] {ticker} テスト完了")
                    self.logger.info(f"  勝率: {result['win_rate']:.1f}%")
                    self.logger.info(f"  総損益: {result['total_pnl']:.2f}%")
                    self.logger.info(f"  取引数: {result['trade_count']}件")
                else:
                    self.logger.error(f"[FAILED] {ticker} テスト失敗: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                self.logger.error(f"[ERROR] {ticker} テスト中にエラー: {e}")
                self.results[ticker] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'ticker': ticker,
                    'name': name
                }
        
        # 比較レポート生成
        self._generate_comparison_report()
        
        return self.results
    
    def _test_single_ticker(self, ticker: str, name: str) -> Dict[str, Any]:
        """
        単一銘柄テスト
        
        Args:
            ticker: 銘柄コード
            name: 銘柄名
            
        Returns:
            テスト結果
        """
        self.logger.info(f"\n[PHASE 1] データ取得: {ticker}")
        
        try:
            # 株価データ取得
            stock_data = self.data_feed.get_stock_data(
                ticker=ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if stock_data is None or len(stock_data) == 0:
                raise ValueError(f"データ取得失敗: {ticker}")
            
            self.logger.info(f"[SUCCESS] 株価データ取得完了: {len(stock_data)}行")
            
            # DOWデータ取得
            dow_data = self.data_feed.get_index_data(
                index_symbol="^DJI",
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if dow_data is None or len(dow_data) == 0:
                raise ValueError("DOWデータ取得失敗")
            
            self.logger.info(f"[SUCCESS] DOWデータ取得完了: {len(dow_data)}行")
            
        except Exception as e:
            self.logger.error(f"[ERROR] データ取得エラー: {e}")
            return {
                'status': 'DATA_ERROR',
                'error': str(e),
                'ticker': ticker,
                'name': name
            }
        
        # 戦略初期化
        self.logger.info(f"\n[PHASE 2] 戦略初期化: {ticker}")
        
        try:
            strategy = OpeningGapStrategy(
                data=stock_data,
                dow_data=dow_data,
                params=self.phase_b1_params.copy(),
                price_column="Adj Close"
            )
            
            self.logger.info(f"[SUCCESS] OpeningGapStrategy初期化完了")
            self.logger.info(f"[CONFIG] Phase B-1設定適用:")
            for key, value in self.phase_b1_params.items():
                self.logger.info(f"  {key}: {value}")
                
        except Exception as e:
            self.logger.error(f"[ERROR] 戦略初期化エラー: {e}")
            return {
                'status': 'INIT_ERROR',
                'error': str(e),
                'ticker': ticker,
                'name': name
            }
        
        # バックテスト実行
        self.logger.info(f"\n[PHASE 3] バックテスト実行: {ticker}")
        
        try:
            backtest_result = strategy.backtest()
            
            if backtest_result is None:
                raise ValueError("backtest()がNoneを返しました")
            
            self.logger.info(f"[SUCCESS] バックテスト実行完了")
            
        except Exception as e:
            self.logger.error(f"[ERROR] バックテスト実行エラー: {e}")
            return {
                'status': 'BACKTEST_ERROR',
                'error': str(e),
                'ticker': ticker,
                'name': name
            }
        
        # 結果分析
        self.logger.info(f"\n[PHASE 4] 結果分析: {ticker}")
        
        try:
            analysis = self._analyze_results(backtest_result, ticker, name)
            
            self.logger.info(f"[SUCCESS] 結果分析完了")
            self.logger.info(f"  取引数: {analysis['trade_count']}件")
            self.logger.info(f"  勝率: {analysis['win_rate']:.1f}%")
            self.logger.info(f"  総損益: {analysis['total_pnl']:.2f}%")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"[ERROR] 結果分析エラー: {e}")
            return {
                'status': 'ANALYSIS_ERROR',
                'error': str(e),
                'ticker': ticker,
                'name': name
            }
    
    def _analyze_results(self, df: pd.DataFrame, ticker: str, name: str) -> Dict[str, Any]:
        """
        バックテスト結果分析
        
        Args:
            df: バックテスト結果DataFrame
            ticker: 銘柄コード
            name: 銘柄名
            
        Returns:
            分析結果
        """
        # トレード抽出
        trades = []
        entry_date = None
        entry_price = None
        
        for idx, row in df.iterrows():
            if row['Entry_Signal'] == 1:
                entry_date = idx
                entry_price = row['Adj Close']
            elif row['Exit_Signal'] == -1 and entry_date is not None:
                exit_date = idx
                exit_price = row['Adj Close']
                
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                hold_days = (exit_date - entry_date).days
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'hold_days': hold_days
                })
                
                entry_date = None
                entry_price = None
        
        # 統計計算
        if len(trades) == 0:
            return {
                'status': 'NO_TRADES',
                'ticker': ticker,
                'name': name,
                'trade_count': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0
            }
        
        pnl_list = [t['pnl_pct'] for t in trades]
        total_pnl = sum(pnl_list)
        avg_pnl = np.mean(pnl_list)
        
        winning_trades = [p for p in pnl_list if p > 0]
        losing_trades = [p for p in pnl_list if p < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / len(trades)) * 100
        
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        hold_days_list = [t['hold_days'] for t in trades]
        avg_hold = np.mean(hold_days_list)
        
        return {
            'status': 'SUCCESS',
            'ticker': ticker,
            'name': name,
            'trade_count': len(trades),
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': float(win_rate),
            'total_pnl': float(total_pnl),
            'avg_pnl': float(avg_pnl),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'avg_hold': float(avg_hold),
            'trades': trades
        }
    
    def _generate_comparison_report(self):
        """
        比較レポート生成
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Phase B-3-1: 複数銘柄比較レポート")
        self.logger.info("=" * 80)
        
        # 成功した銘柄のみ抽出
        successful_results = {
            ticker: result for ticker, result in self.results.items()
            if result.get('status') == 'SUCCESS'
        }
        
        if not successful_results:
            self.logger.error("[ERROR] 成功した銘柄がありません")
            return
        
        # 比較テーブル
        self.logger.info("\n[COMPARISON TABLE]")
        self.logger.info("-" * 80)
        self.logger.info(f"{'銘柄':<12} {'勝率':>8} {'総損益':>10} {'取引数':>8} {'平均保有':>10}")
        self.logger.info("-" * 80)
        
        for ticker, result in successful_results.items():
            name = result['name']
            win_rate = result['win_rate']
            total_pnl = result['total_pnl']
            trade_count = result['trade_count']
            avg_hold = result['avg_hold']
            
            self.logger.info(
                f"{ticker:<12} {win_rate:>7.1f}% {total_pnl:>9.2f}% {trade_count:>8} {avg_hold:>9.1f}日"
            )
        
        # レポートファイル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = project_root / "tests" / f"multi_ticker_report_{timestamp}.json"
        
        report_data = {
            'test_type': 'Phase B-3-1: Multi-Ticker Test',
            'test_date': datetime.now().isoformat(),
            'period': f"{self.start_date} ~ {self.end_date}",
            'configuration': self.phase_b1_params,
            'results': self.results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"\n[REPORT] レポートファイル保存: {report_file.name}")


def main():
    """
    メインエントリーポイント
    """
    print("\n" + "=" * 80)
    print("Phase B-3-1: 複数銘柄バックテストテスト")
    print("=" * 80 + "\n")
    
    # テスター初期化
    tester = MultiTickerTester(
        start_date="2022-01-01",  # 株式分割後の期間
        end_date="2024-12-31"
    )
    
    # 全銘柄テスト実行
    results = tester.run_all_tests()
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("Phase B-3-1: テスト結果サマリー")
    print("=" * 80)
    
    successful_count = sum(1 for r in results.values() if r.get('status') == 'SUCCESS')
    total_count = len(results)
    
    print(f"\n成功: {successful_count}/{total_count} 銘柄")
    
    for ticker, result in results.items():
        if result.get('status') == 'SUCCESS':
            print(f"\n[OK] {ticker} ({result['name']})")
            print(f"  勝率: {result['win_rate']:.1f}%")
            print(f"  総損益: {result['total_pnl']:.2f}%")
            print(f"  取引数: {result['trade_count']}件")
        else:
            print(f"\n[FAILED] {ticker} ({result.get('name', 'Unknown')})")
            print(f"  エラー: {result.get('error', 'Unknown')}")
    
    print("\n" + "=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    main()
