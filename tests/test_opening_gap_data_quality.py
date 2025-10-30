"""
Opening Gap戦略 データ品質検証テスト

Phase A: データ品質検証
- 7203.T (トヨタ)の価格データの妥当性を検証
- 異常な価格変動、株式分割、ギャップの検出
- 2019年の-917%損失の原因究明

検証項目:
1. 極端な価格変動の検出（日次10%以上）
2. 株式分割・配当調整の確認
3. ギャップアップ/ダウンの妥当性
4. Open-Close-Adj Close の整合性
5. 欠損値・異常値の検出

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

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)

class DataQualityValidator:
    """データ品質検証クラス"""
    
    def __init__(self, ticker: str = "7203.T", start_date: str = "2019-01-01", end_date: str = "2024-12-31"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data_feed = None
        self.stock_data = None
        self.issues = []  # 検出された問題のリスト
        
    def run(self):
        """データ品質検証実行"""
        self.logger.info("=" * 80)
        self.logger.info(f"データ品質検証テスト開始: {self.ticker}")
        self.logger.info(f"期間: {self.start_date} ~ {self.end_date}")
        self.logger.info("=" * 80)
        
        # データ取得
        if not self._fetch_data():
            return "FAILED_DATA_FETCH"
        
        # 検証実行
        self._validate_extreme_moves()
        self._validate_splits_and_dividends()
        self._validate_gaps()
        self._validate_price_consistency()
        self._validate_missing_values()
        
        # レポート生成
        self._generate_report()
        
        return "SUCCESS" if len(self.issues) == 0 else "ISSUES_FOUND"
    
    def _fetch_data(self) -> bool:
        """データ取得"""
        self.logger.info("")
        self.logger.info("[PHASE 1] データ取得")
        self.logger.info("-" * 80)
        
        try:
            self.data_feed = YFinanceDataFeed()
            
            self.stock_data = self.data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.stock_data is None or self.stock_data.empty:
                self.logger.error(f"[ERROR] データ取得失敗: {self.ticker}")
                return False
            
            self.logger.info(f"[SUCCESS] データ取得完了: {len(self.stock_data)} rows")
            self.logger.info(f"  期間: {self.stock_data.index[0]} ~ {self.stock_data.index[-1]}")
            self.logger.info(f"  カラム: {list(self.stock_data.columns)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] データ取得中にエラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _validate_extreme_moves(self):
        """極端な価格変動の検出"""
        self.logger.info("")
        self.logger.info("[検証1] 極端な価格変動の検出")
        self.logger.info("-" * 80)
        
        # 日次リターン計算
        daily_returns = self.stock_data['Close'].pct_change()
        
        # 10%以上の変動を検出
        extreme_threshold = 0.10
        extreme_moves = daily_returns[abs(daily_returns) > extreme_threshold]
        
        self.logger.info(f"  10%以上の変動: {len(extreme_moves)}件")
        
        if len(extreme_moves) > 0:
            self.logger.warning("[WARNING] 極端な価格変動を検出:")
            for date, ret in extreme_moves.head(10).items():
                price_before = self.stock_data.loc[self.stock_data.index < date, 'Close'].iloc[-1] if len(self.stock_data.loc[self.stock_data.index < date]) > 0 else None
                price_after = self.stock_data.loc[date, 'Close']
                self.logger.warning(f"    {date.date()}: {ret*100:.2f}% (価格: {price_before:.2f} → {price_after:.2f})")
                
                self.issues.append({
                    "type": "extreme_move",
                    "date": str(date.date()),
                    "return_pct": float(ret * 100),
                    "price_before": float(price_before) if price_before else None,
                    "price_after": float(price_after)
                })
        else:
            self.logger.info("[OK] 極端な価格変動なし")
    
    def _validate_splits_and_dividends(self):
        """株式分割・配当調整の確認"""
        self.logger.info("")
        self.logger.info("[検証2] 株式分割・配当調整の確認")
        self.logger.info("-" * 80)
        
        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(self.ticker)
            
            # 株式分割情報
            splits = ticker_obj.splits
            if len(splits) > 0:
                splits_in_period = splits[(splits.index >= self.start_date) & (splits.index <= self.end_date)]
                if len(splits_in_period) > 0:
                    self.logger.warning(f"[WARNING] 株式分割イベント: {len(splits_in_period)}件")
                    for date, ratio in splits_in_period.items():
                        self.logger.warning(f"    {date.date()}: {ratio}対1の分割")
                        self.issues.append({
                            "type": "stock_split",
                            "date": str(date.date()),
                            "ratio": float(ratio)
                        })
                else:
                    self.logger.info("[OK] 期間内に株式分割なし")
            else:
                self.logger.info("[OK] 株式分割情報なし")
            
            # 配当情報
            dividends = ticker_obj.dividends
            if len(dividends) > 0:
                divs_in_period = dividends[(dividends.index >= self.start_date) & (dividends.index <= self.end_date)]
                self.logger.info(f"  配当支払い: {len(divs_in_period)}件")
                if len(divs_in_period) > 10:
                    self.logger.info(f"    最初の3件: {divs_in_period.head(3).to_dict()}")
                else:
                    self.logger.info(f"    全件: {divs_in_period.to_dict()}")
                    
        except Exception as e:
            self.logger.error(f"[ERROR] 株式分割・配当情報取得エラー: {str(e)}")
    
    def _validate_gaps(self):
        """ギャップアップ/ダウンの妥当性検証"""
        self.logger.info("")
        self.logger.info("[検証3] ギャップの妥当性検証")
        self.logger.info("-" * 80)
        
        # ギャップ計算（Open vs 前日Close）
        gap_ratios = (self.stock_data['Open'] / self.stock_data['Close'].shift(1) - 1) * 100
        
        # 2%以上のギャップを検出
        gap_threshold = 2.0
        large_gaps = gap_ratios[abs(gap_ratios) > gap_threshold]
        
        self.logger.info(f"  2%以上のギャップ: {len(large_gaps)}件")
        
        if len(large_gaps) > 0:
            self.logger.info("[INFO] 大きなギャップを検出:")
            for date, gap_pct in large_gaps.head(10).items():
                open_price = self.stock_data.loc[date, 'Open']
                prev_close = self.stock_data.loc[self.stock_data.index < date, 'Close'].iloc[-1] if len(self.stock_data.loc[self.stock_data.index < date]) > 0 else None
                close_price = self.stock_data.loc[date, 'Close']
                
                direction = "ギャップアップ" if gap_pct > 0 else "ギャップダウン"
                self.logger.info(f"    {date.date()}: {direction} {gap_pct:.2f}% (前日終値: {prev_close:.2f} → 始値: {open_price:.2f} → 終値: {close_price:.2f})")
        else:
            self.logger.info("[OK] 大きなギャップなし")
    
    def _validate_price_consistency(self):
        """Open-Close-Adj Close の整合性検証"""
        self.logger.info("")
        self.logger.info("[検証4] 価格データの整合性検証")
        self.logger.info("-" * 80)
        
        # CloseとAdj Closeの差異
        if 'Adj Close' in self.stock_data.columns:
            adj_ratio = (self.stock_data['Adj Close'] / self.stock_data['Close']).dropna()
            
            # 調整率が大きく変動している日を検出
            adj_ratio_change = adj_ratio.pct_change().abs()
            large_adj_changes = adj_ratio_change[adj_ratio_change > 0.01]  # 1%以上の変動
            
            if len(large_adj_changes) > 0:
                self.logger.warning(f"[WARNING] Adj Close調整率の大きな変動: {len(large_adj_changes)}件")
                for date, change in large_adj_changes.head(5).items():
                    close = self.stock_data.loc[date, 'Close']
                    adj_close = self.stock_data.loc[date, 'Adj Close']
                    self.logger.warning(f"    {date.date()}: Close={close:.2f}, Adj Close={adj_close:.2f}, 調整率変動={change*100:.2f}%")
                    
                    self.issues.append({
                        "type": "adj_close_anomaly",
                        "date": str(date.date()),
                        "close": float(close),
                        "adj_close": float(adj_close),
                        "ratio_change_pct": float(change * 100)
                    })
            else:
                self.logger.info("[OK] Adj Close調整率は安定")
        
        # High >= Low の確認
        invalid_hl = self.stock_data[self.stock_data['High'] < self.stock_data['Low']]
        if len(invalid_hl) > 0:
            self.logger.error(f"[ERROR] High < Low の異常データ: {len(invalid_hl)}件")
            for date, row in invalid_hl.iterrows():
                self.logger.error(f"    {date.date()}: High={row['High']:.2f} < Low={row['Low']:.2f}")
                self.issues.append({
                    "type": "invalid_high_low",
                    "date": str(date.date()),
                    "high": float(row['High']),
                    "low": float(row['Low'])
                })
        else:
            self.logger.info("[OK] High >= Low の整合性確認")
    
    def _validate_missing_values(self):
        """欠損値・異常値の検出"""
        self.logger.info("")
        self.logger.info("[検証5] 欠損値・異常値の検出")
        self.logger.info("-" * 80)
        
        # 欠損値チェック
        missing_counts = self.stock_data.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            self.logger.warning(f"[WARNING] 欠損値検出: {total_missing}個")
            for col, count in missing_counts[missing_counts > 0].items():
                self.logger.warning(f"    {col}: {count}個")
                self.issues.append({
                    "type": "missing_value",
                    "column": col,
                    "count": int(count)
                })
        else:
            self.logger.info("[OK] 欠損値なし")
        
        # ゼロ値・負値のチェック
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_cols:
            if col in self.stock_data.columns:
                zero_or_negative = self.stock_data[self.stock_data[col] <= 0]
                if len(zero_or_negative) > 0:
                    self.logger.error(f"[ERROR] {col}にゼロまたは負値: {len(zero_or_negative)}件")
                    for date, row in zero_or_negative.iterrows():
                        self.logger.error(f"    {date.date()}: {col}={row[col]:.2f}")
                        self.issues.append({
                            "type": "invalid_price",
                            "date": str(date.date()),
                            "column": col,
                            "value": float(row[col])
                        })
        
        if len([i for i in self.issues if i['type'] == 'invalid_price']) == 0:
            self.logger.info("[OK] 価格データは全て正の値")
    
    def _generate_report(self):
        """検証レポート生成"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("データ品質検証レポート")
        self.logger.info("=" * 80)
        
        self.logger.info(f"銘柄: {self.ticker}")
        self.logger.info(f"期間: {self.start_date} ~ {self.end_date}")
        self.logger.info(f"データ件数: {len(self.stock_data)}")
        self.logger.info(f"検出された問題: {len(self.issues)}件")
        
        if len(self.issues) > 0:
            self.logger.info("")
            self.logger.info("問題の内訳:")
            
            issue_types = {}
            for issue in self.issues:
                issue_type = issue['type']
                if issue_type not in issue_types:
                    issue_types[issue_type] = 0
                issue_types[issue_type] += 1
            
            for issue_type, count in issue_types.items():
                self.logger.info(f"  {issue_type}: {count}件")
            
            # JSONレポート保存
            report_path = project_root / "tests" / f"data_quality_report_{self.ticker.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_data = {
                "ticker": self.ticker,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "data_count": len(self.stock_data),
                "issue_count": len(self.issues),
                "issues": self.issues
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("")
            self.logger.info(f"詳細レポート保存: {report_path}")
        else:
            self.logger.info("")
            self.logger.info("[SUCCESS] データ品質に問題は検出されませんでした")
        
        self.logger.info("=" * 80)


def main():
    """メイン実行"""
    print("\n" + "=" * 80)
    print("Opening Gap戦略 - データ品質検証テスト")
    print("=" * 80)
    
    # 7203.T（トヨタ）のデータ品質検証
    validator = DataQualityValidator(
        ticker="7203.T",
        start_date="2019-01-01",
        end_date="2024-12-31"
    )
    
    status = validator.run()
    
    print("\n" + "=" * 80)
    print(f"検証完了: ステータス = {status}")
    print("=" * 80)
    
    return status


if __name__ == "__main__":
    status = main()
    sys.exit(0 if status in ["SUCCESS", "ISSUES_FOUND"] else 1)
