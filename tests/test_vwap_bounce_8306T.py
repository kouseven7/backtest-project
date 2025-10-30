"""
VWAP_Bounce戦略 動作検証テスト（8306.T 2024年データ）

Phase B-3検証: strategies/VWAP_Bounce.pyの動作確認

主な機能:
- yfinanceからリアルデータ取得（8306.T, ^N225）
- エントリー/イグジットシグナル生成検証
- VWAP固有ロジック検証（VWAP計算、エントリー条件、イグジット条件）
- トレンド判定検証（range-bound判定）
- 取引実行検証（Position管理、entry_prices辞書）
- パフォーマンス計算（総損益、勝率、平均保有期間）

統合コンポーネント:
- strategies.VWAP_Bounce: テスト対象戦略
- main_system.data_acquisition.yfinance_data_feed: データ取得
- indicators.basic_indicators: VWAP計算検証
- indicators.unified_trend_detector: トレンド判定検証

セーフティ機能/注意事項:
- copilot-instructions.md準拠（モックデータ禁止）
- strategies/VWAP_Bounce.pyは修正禁止
- yfinanceデータ取得失敗時はエラーとして処理
- デフォルトパラメータ使用（volatility_filter_enabled=False）

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# プロジェクトルート設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger

# テスト対象
from strategies.VWAP_Bounce import VWAPBounceStrategy

# データ取得
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed

# 検証用
from indicators.basic_indicators import calculate_vwap
from indicators.unified_trend_detector import detect_unified_trend

# ロガー設定
logger = setup_logger(__name__)


class VWAPBounceTestRunner:
    """VWAP_Bounce戦略テストランナー"""
    
    def __init__(self, ticker: str = "8306.T", index_ticker: str = "^N225"):
        """
        初期化
        
        Parameters:
            ticker: テスト対象銘柄（デフォルト: 8306.T）
            index_ticker: インデックス銘柄（デフォルト: ^N225）
        """
        self.ticker = ticker
        self.index_ticker = index_ticker
        self.logger = setup_logger(f"{__name__}.{ticker}")
        
        # データ格納用
        self.stock_data: Optional[pd.DataFrame] = None
        self.index_data: Optional[pd.DataFrame] = None
        self.strategy: Optional[VWAPBounceStrategy] = None
        self.result_data: Optional[pd.DataFrame] = None
        
        # 検証結果格納用
        self.validation_results: Dict[str, Any] = {}
        
    def fetch_data(self, start_date: str = "2024-01-01", end_date: str = "2024-12-31") -> bool:
        """
        yfinanceからデータ取得
        
        Parameters:
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
            
        Returns:
            bool: 取得成功=True
        """
        self.logger.info("=" * 80)
        self.logger.info(f"データ取得開始: {self.ticker}, 期間: {start_date} ~ {end_date}")
        self.logger.info("=" * 80)
        
        try:
            data_feed = YFinanceDataFeed()
            
            # 株価データ取得
            self.logger.info(f"株価データ取得中: {self.ticker}")
            self.stock_data = data_feed.get_stock_data(
                ticker=self.ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if self.stock_data is None or len(self.stock_data) == 0:
                self.logger.error(f"株価データ取得失敗: {self.ticker}")
                return False
            
            self.logger.info(f"株価データ取得成功: {len(self.stock_data)}行")
            self.logger.info(f"データ範囲: {self.stock_data.index[0]} ~ {self.stock_data.index[-1]}")
            
            # インデックスデータ取得（トレンド判定用）
            self.logger.info(f"インデックスデータ取得中: {self.index_ticker}")
            self.index_data = data_feed.get_stock_data(
                ticker=self.index_ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if self.index_data is None or len(self.index_data) == 0:
                self.logger.warning(f"インデックスデータ取得失敗: {self.index_ticker}（継続）")
            else:
                self.logger.info(f"インデックスデータ取得成功: {len(self.index_data)}行")
            
            return True
            
        except Exception as e:
            self.logger.error(f"データ取得中にエラー発生: {e}", exc_info=True)
            return False
    
    def run_backtest(self) -> bool:
        """
        バックテスト実行
        
        Returns:
            bool: 実行成功=True
        """
        if self.stock_data is None:
            self.logger.error("株価データが未取得です")
            return False
        
        self.logger.info("=" * 80)
        self.logger.info("バックテスト実行開始")
        self.logger.info("=" * 80)
        
        try:
            # 戦略初期化（デフォルトパラメータ使用）
            self.logger.info("VWAP_Bounce戦略初期化（デフォルトパラメータ）")
            self.strategy = VWAPBounceStrategy(
                data=self.stock_data.copy(),
                params=None,  # デフォルトパラメータ使用
                price_column="Adj Close",
                volume_column="Volume"
            )
            
            # 戦略初期化
            self.strategy.initialize_strategy()
            self.logger.info("戦略初期化完了")
            
            # バックテスト実行
            self.logger.info("バックテスト実行中...")
            self.result_data = self.strategy.backtest()
            
            if self.result_data is None:
                self.logger.error("バックテスト実行失敗: 結果がNone")
                return False
            
            self.logger.info("バックテスト実行完了")
            self.logger.info(f"結果データ: {len(self.result_data)}行")
            
            return True
            
        except Exception as e:
            self.logger.error(f"バックテスト実行中にエラー発生: {e}", exc_info=True)
            return False
    
    def validate_signals(self) -> Dict[str, Any]:
        """
        シグナル生成検証
        
        Returns:
            Dict[str, Any]: 検証結果
        """
        if self.result_data is None:
            self.logger.error("結果データが未取得です")
            return {}
        
        self.logger.info("=" * 80)
        self.logger.info("シグナル生成検証開始")
        self.logger.info("=" * 80)
        
        try:
            # エントリーシグナル検証
            entry_signals = self.result_data[self.result_data['Entry_Signal'] == 1]
            entry_count = len(entry_signals)
            
            self.logger.info(f"エントリーシグナル数: {entry_count}")
            
            # イグジットシグナル検証
            exit_signals = self.result_data[self.result_data['Exit_Signal'] == -1]
            exit_count = len(exit_signals)
            
            self.logger.info(f"イグジットシグナル数: {exit_count}")
            
            # エントリー/イグジット一致確認
            match_status = "一致" if entry_count == exit_count else "不一致"
            self.logger.info(f"エントリー/イグジット数: {match_status}")
            
            if entry_count != exit_count:
                self.logger.warning(f"エントリー/イグジット数が不一致: Entry={entry_count}, Exit={exit_count}")
            
            # エントリー詳細ログ
            if entry_count > 0:
                self.logger.info("\n--- エントリー詳細 ---")
                for idx, row in entry_signals.iterrows():
                    price = row['Adj Close']
                    vwap = row['VWAP']
                    volume = row['Volume']
                    self.logger.info(
                        f"日付: {idx.strftime('%Y-%m-%d')}, "
                        f"価格: {price:.2f}, "
                        f"VWAP: {vwap:.2f}, "
                        f"出来高: {volume:,.0f}"
                    )
            
            # イグジット詳細ログ
            if exit_count > 0:
                self.logger.info("\n--- イグジット詳細 ---")
                for idx, row in exit_signals.iterrows():
                    price = row['Adj Close']
                    vwap = row['VWAP']
                    self.logger.info(
                        f"日付: {idx.strftime('%Y-%m-%d')}, "
                        f"価格: {price:.2f}, "
                        f"VWAP: {vwap:.2f}"
                    )
            
            result = {
                "entry_count": entry_count,
                "exit_count": exit_count,
                "match_status": match_status,
                "entry_dates": [idx.strftime('%Y-%m-%d') for idx in entry_signals.index],
                "exit_dates": [idx.strftime('%Y-%m-%d') for idx in exit_signals.index],
                "validation_passed": entry_count > 0 and entry_count == exit_count
            }
            
            self.validation_results['signals'] = result
            return result
            
        except Exception as e:
            self.logger.error(f"シグナル検証中にエラー発生: {e}", exc_info=True)
            return {}
    
    def validate_vwap_logic(self) -> Dict[str, Any]:
        """
        VWAP固有ロジック検証
        
        Returns:
            Dict[str, Any]: 検証結果
        """
        if self.result_data is None or self.strategy is None:
            self.logger.error("結果データまたは戦略が未取得です")
            return {}
        
        self.logger.info("=" * 80)
        self.logger.info("VWAP固有ロジック検証開始")
        self.logger.info("=" * 80)
        
        try:
            params = self.strategy.params
            
            # VWAP計算検証
            self.logger.info("1. VWAP計算検証")
            vwap_exists = 'VWAP' in self.result_data.columns
            vwap_not_null = self.result_data['VWAP'].notna().sum() > 0
            
            self.logger.info(f"   VWAP列存在: {vwap_exists}")
            self.logger.info(f"   VWAP非NULL数: {self.result_data['VWAP'].notna().sum()}/{len(self.result_data)}")
            
            # エントリー条件検証
            self.logger.info("\n2. エントリー条件検証")
            entry_signals = self.result_data[self.result_data['Entry_Signal'] == 1]
            
            vwap_condition_violations = []
            volume_condition_violations = []
            bullish_condition_violations = []
            
            for idx, row in entry_signals.iterrows():
                row_idx = self.result_data.index.get_loc(idx)
                
                # VWAP条件（-1%以内）
                price = row['Adj Close']
                vwap = row['VWAP']
                vwap_lower = vwap * params['vwap_lower_threshold']
                
                if not (vwap_lower <= price <= vwap):
                    vwap_condition_violations.append({
                        'date': idx.strftime('%Y-%m-%d'),
                        'price': price,
                        'vwap': vwap,
                        'vwap_lower': vwap_lower,
                        'ratio': price / vwap
                    })
                
                # 出来高増加条件（前日比1.2倍以上）
                if row_idx > 0:
                    current_volume = row['Volume']
                    prev_volume = self.result_data['Volume'].iloc[row_idx - 1]
                    volume_ratio = current_volume / prev_volume
                    
                    if volume_ratio < params['volume_increase_threshold']:
                        volume_condition_violations.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'volume_ratio': volume_ratio,
                            'threshold': params['volume_increase_threshold']
                        })
                
                # 陽線判定（0.5%以上上昇）
                if row_idx > 0:
                    current_price = row['Adj Close']
                    prev_price = self.result_data['Adj Close'].iloc[row_idx - 1]
                    price_change_pct = (current_price - prev_price) / prev_price
                    
                    if price_change_pct <= params['bullish_candle_min_pct']:
                        bullish_condition_violations.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'price_change_pct': price_change_pct * 100,
                            'threshold_pct': params['bullish_candle_min_pct'] * 100
                        })
            
            self.logger.info(f"   VWAP条件違反: {len(vwap_condition_violations)}件")
            if len(vwap_condition_violations) > 0:
                for v in vwap_condition_violations:
                    self.logger.warning(
                        f"     日付: {v['date']}, 価格: {v['price']:.2f}, "
                        f"VWAP: {v['vwap']:.2f}, 下限: {v['vwap_lower']:.2f}, "
                        f"比率: {v['ratio']:.4f}"
                    )
            
            self.logger.info(f"   出来高条件違反: {len(volume_condition_violations)}件")
            if len(volume_condition_violations) > 0:
                for v in volume_condition_violations:
                    self.logger.warning(
                        f"     日付: {v['date']}, 出来高比率: {v['volume_ratio']:.2f}, "
                        f"閾値: {v['threshold']:.2f}"
                    )
            
            self.logger.info(f"   陽線条件違反: {len(bullish_condition_violations)}件")
            if len(bullish_condition_violations) > 0:
                for v in bullish_condition_violations:
                    self.logger.warning(
                        f"     日付: {v['date']}, 変化率: {v['price_change_pct']:.2f}%, "
                        f"閾値: {v['threshold_pct']:.2f}%"
                    )
            
            # イグジット条件検証
            self.logger.info("\n3. イグジット条件検証")
            exit_signals = self.result_data[self.result_data['Exit_Signal'] == -1]
            
            exit_reasons = {
                'vwap_upper': 0,
                'vwap_lower': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'trailing_stop': 0,
                'unknown': 0
            }
            
            # 簡易的なイグジット理由判定（ログから推測）
            for idx, row in exit_signals.iterrows():
                price = row['Adj Close']
                vwap = row['VWAP']
                vwap_upper = vwap * params['vwap_upper_threshold']
                vwap_lower = vwap * params['vwap_lower_threshold']
                
                if price >= vwap_upper:
                    exit_reasons['vwap_upper'] += 1
                elif price <= vwap_lower:
                    exit_reasons['vwap_lower'] += 1
                else:
                    exit_reasons['unknown'] += 1
            
            self.logger.info("   イグジット理由内訳:")
            for reason, count in exit_reasons.items():
                if count > 0:
                    self.logger.info(f"     {reason}: {count}件")
            
            result = {
                "vwap_calculated": vwap_exists and vwap_not_null,
                "entry_vwap_violations": len(vwap_condition_violations),
                "entry_volume_violations": len(volume_condition_violations),
                "entry_bullish_violations": len(bullish_condition_violations),
                "exit_reasons": exit_reasons,
                "validation_passed": (
                    vwap_exists and
                    vwap_not_null and
                    len(vwap_condition_violations) == 0
                )
            }
            
            self.validation_results['vwap_logic'] = result
            return result
            
        except Exception as e:
            self.logger.error(f"VWAPロジック検証中にエラー発生: {e}", exc_info=True)
            return {}
    
    def validate_trend_detection(self) -> Dict[str, Any]:
        """
        トレンド判定検証
        
        Returns:
            Dict[str, Any]: 検証結果
        """
        if self.result_data is None:
            self.logger.error("結果データが未取得です")
            return {}
        
        self.logger.info("=" * 80)
        self.logger.info("トレンド判定検証開始")
        self.logger.info("=" * 80)
        
        try:
            entry_signals = self.result_data[self.result_data['Entry_Signal'] == 1]
            
            trend_violations = []
            
            for idx, row in entry_signals.iterrows():
                row_idx = self.result_data.index.get_loc(idx)
                
                # エントリー時点のトレンド判定
                data_slice = self.result_data.iloc[:row_idx + 1]
                trend = detect_unified_trend(
                    data=data_slice,
                    price_column='Adj Close',
                    strategy='VWAP_Bounce'
                )
                
                if trend != "range-bound":
                    trend_violations.append({
                        'date': idx.strftime('%Y-%m-%d'),
                        'trend': trend
                    })
            
            self.logger.info(f"トレンド条件違反: {len(trend_violations)}件")
            if len(trend_violations) > 0:
                for v in trend_violations:
                    self.logger.warning(
                        f"  日付: {v['date']}, トレンド: {v['trend']} (期待: range-bound)"
                    )
            
            result = {
                "trend_violations": len(trend_violations),
                "validation_passed": len(trend_violations) == 0
            }
            
            self.validation_results['trend'] = result
            return result
            
        except Exception as e:
            self.logger.error(f"トレンド判定検証中にエラー発生: {e}", exc_info=True)
            return {}
    
    def analyze_trend_distribution(self) -> Dict[str, Any]:
        """
        全期間のトレンド分布を分析
        
        Returns:
            Dict[str, Any]: トレンド分布結果
        """
        if self.result_data is None:
            self.logger.error("結果データが未取得です")
            return {}
        
        self.logger.info("=" * 80)
        self.logger.info("トレンド分布分析開始")
        self.logger.info("=" * 80)
        
        try:
            # トレンドカウント
            trend_counts = {
                "uptrend": 0,
                "downtrend": 0,
                "range-bound": 0,
                "unknown": 0
            }
            
            # 日別トレンド記録
            daily_trends = []
            
            # VWAP条件を満たす日のトレンド
            vwap_candidate_trends = []
            
            params = self.strategy.params if self.strategy else {}
            vwap_lower_threshold = params.get('vwap_lower_threshold', 0.99)
            volume_threshold = params.get('volume_increase_threshold', 1.2)
            bullish_threshold = params.get('bullish_candle_min_pct', 0.005)
            
            self.logger.info("全期間のトレンド判定を実行中...")
            
            for idx in range(1, len(self.result_data)):
                current_date = self.result_data.index[idx]
                
                # トレンド判定
                data_slice = self.result_data.iloc[:idx + 1]
                trend = detect_unified_trend(
                    data=data_slice,
                    price_column='Adj Close',
                    strategy='VWAP_Bounce'
                )
                
                trend_counts[trend] = trend_counts.get(trend, 0) + 1
                
                daily_trends.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'trend': trend
                })
                
                # VWAP条件チェック
                current_price = self.result_data['Adj Close'].iloc[idx]
                vwap = self.result_data['VWAP'].iloc[idx]
                vwap_lower = vwap * vwap_lower_threshold
                
                prev_price = self.result_data['Adj Close'].iloc[idx - 1]
                price_change_pct = (current_price - prev_price) / prev_price
                
                current_volume = self.result_data['Volume'].iloc[idx]
                prev_volume = self.result_data['Volume'].iloc[idx - 1]
                volume_ratio = current_volume / prev_volume
                
                # VWAP条件を満たすか
                meets_vwap = (vwap_lower <= current_price <= vwap)
                meets_volume = (volume_ratio > volume_threshold)
                meets_bullish = (price_change_pct > bullish_threshold)
                
                if meets_vwap and meets_volume and meets_bullish:
                    vwap_candidate_trends.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'trend': trend,
                        'price': current_price,
                        'vwap': vwap,
                        'volume_ratio': volume_ratio,
                        'price_change_pct': price_change_pct * 100
                    })
            
            # 結果ログ出力
            self.logger.info("\n--- トレンド分布サマリー ---")
            total_days = sum(trend_counts.values())
            for trend, count in trend_counts.items():
                percentage = (count / total_days * 100) if total_days > 0 else 0
                self.logger.info(f"  {trend}: {count}日 ({percentage:.1f}%)")
            
            self.logger.info(f"\n--- VWAP条件を満たした日 ---")
            self.logger.info(f"  合計: {len(vwap_candidate_trends)}日")
            
            if len(vwap_candidate_trends) > 0:
                # トレンド別集計
                candidate_trend_counts = {}
                for candidate in vwap_candidate_trends:
                    t = candidate['trend']
                    candidate_trend_counts[t] = candidate_trend_counts.get(t, 0) + 1
                
                self.logger.info("\n  トレンド別内訳:")
                for trend, count in candidate_trend_counts.items():
                    self.logger.info(f"    {trend}: {count}日")
                
                # 詳細リスト（最大10件）
                self.logger.info("\n  詳細リスト（最大10件）:")
                for i, candidate in enumerate(vwap_candidate_trends[:10], 1):
                    self.logger.info(
                        f"    {i}. 日付: {candidate['date']}, "
                        f"トレンド: {candidate['trend']}, "
                        f"価格: {candidate['price']:.2f}, "
                        f"VWAP: {candidate['vwap']:.2f}, "
                        f"出来高比: {candidate['volume_ratio']:.2f}x, "
                        f"変化率: {candidate['price_change_pct']:.2f}%"
                    )
                
                if len(vwap_candidate_trends) > 10:
                    self.logger.info(f"    ... 他{len(vwap_candidate_trends) - 10}件")
            
            # range-bound日のリスト（最大20件）
            range_bound_days = [d for d in daily_trends if d['trend'] == 'range-bound']
            if len(range_bound_days) > 0:
                self.logger.info(f"\n--- range-bound日のリスト（最大20件） ---")
                for i, day in enumerate(range_bound_days[:20], 1):
                    self.logger.info(f"  {i}. {day['date']}")
                if len(range_bound_days) > 20:
                    self.logger.info(f"  ... 他{len(range_bound_days) - 20}件")
            
            result = {
                "trend_counts": trend_counts,
                "vwap_candidates": len(vwap_candidate_trends),
                "vwap_candidate_details": vwap_candidate_trends,
                "range_bound_days": len(range_bound_days),
                "daily_trends": daily_trends
            }
            
            self.validation_results['trend_distribution'] = result
            return result
            
        except Exception as e:
            self.logger.error(f"トレンド分布分析中にエラー発生: {e}", exc_info=True)
            return {}
    
    def calculate_performance(self) -> Dict[str, Any]:
        """
        パフォーマンス計算
        
        Returns:
            Dict[str, Any]: パフォーマンス指標
        """
        if self.result_data is None or self.strategy is None:
            self.logger.error("結果データまたは戦略が未取得です")
            return {}
        
        self.logger.info("=" * 80)
        self.logger.info("パフォーマンス計算開始")
        self.logger.info("=" * 80)
        
        try:
            # エントリー/イグジットペアの抽出
            entry_signals = self.result_data[self.result_data['Entry_Signal'] == 1]
            exit_signals = self.result_data[self.result_data['Exit_Signal'] == -1]
            
            total_trades = min(len(entry_signals), len(exit_signals))
            
            if total_trades == 0:
                self.logger.warning("取引が発生していません")
                return {
                    "total_trades": 0,
                    "total_pnl": 0.0,
                    "total_pnl_pct": 0.0,
                    "win_rate": 0.0,
                    "avg_hold_days": 0.0
                }
            
            # 取引ペアの構築
            trades = []
            initial_capital = 1_000_000  # 初期資本100万円
            
            for i in range(total_trades):
                entry_date = entry_signals.index[i]
                exit_date = exit_signals.index[i]
                
                entry_price = self.result_data.loc[entry_date, 'Adj Close']
                exit_price = self.result_data.loc[exit_date, 'Adj Close']
                
                # 保有期間
                hold_days = (exit_date - entry_date).days
                
                # 損益計算（100%投資と仮定）
                pnl = (exit_price - entry_price) / entry_price
                pnl_yen = initial_capital * pnl
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'hold_days': hold_days,
                    'pnl_pct': pnl * 100,
                    'pnl_yen': pnl_yen
                })
            
            # パフォーマンス指標計算
            total_pnl_yen = sum(t['pnl_yen'] for t in trades)
            total_pnl_pct = (total_pnl_yen / initial_capital) * 100
            
            wins = [t for t in trades if t['pnl_yen'] > 0]
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0.0
            
            avg_hold_days = sum(t['hold_days'] for t in trades) / total_trades if total_trades > 0 else 0.0
            
            # ログ出力
            self.logger.info(f"総取引回数: {total_trades}")
            self.logger.info(f"総損益: {total_pnl_yen:,.0f}円")
            self.logger.info(f"総損益率: {total_pnl_pct:.2f}%")
            self.logger.info(f"勝率: {win_rate:.2f}%")
            self.logger.info(f"平均保有期間: {avg_hold_days:.1f}日")
            
            # 取引詳細ログ
            self.logger.info("\n--- 取引詳細 ---")
            for i, trade in enumerate(trades, 1):
                self.logger.info(
                    f"取引{i}: "
                    f"エントリー: {trade['entry_date'].strftime('%Y-%m-%d')} ({trade['entry_price']:.2f}), "
                    f"イグジット: {trade['exit_date'].strftime('%Y-%m-%d')} ({trade['exit_price']:.2f}), "
                    f"保有: {trade['hold_days']}日, "
                    f"損益: {trade['pnl_yen']:,.0f}円 ({trade['pnl_pct']:.2f}%)"
                )
            
            result = {
                "total_trades": total_trades,
                "total_pnl": total_pnl_yen,
                "total_pnl_pct": total_pnl_pct,
                "win_rate": win_rate,
                "avg_hold_days": avg_hold_days,
                "trades": trades
            }
            
            self.validation_results['performance'] = result
            return result
            
        except Exception as e:
            self.logger.error(f"パフォーマンス計算中にエラー発生: {e}", exc_info=True)
            return {}
    
    def generate_summary_report(self) -> bool:
        """
        総合レポート生成
        
        Returns:
            bool: 生成成功=True
        """
        self.logger.info("=" * 80)
        self.logger.info("総合レポート生成")
        self.logger.info("=" * 80)
        
        try:
            # 検証結果の確認
            if not self.validation_results:
                self.logger.error("検証結果が空です")
                return False
            
            # テスト成功条件チェック
            signals = self.validation_results.get('signals', {})
            vwap_logic = self.validation_results.get('vwap_logic', {})
            performance = self.validation_results.get('performance', {})
            
            entry_count = signals.get('entry_count', 0)
            entry_exit_match = signals.get('match_status') == '一致'
            vwap_violations = vwap_logic.get('entry_vwap_violations', 0)
            total_trades = performance.get('total_trades', 0)
            
            # 成功判定
            test_passed = (
                entry_count > 0 and
                entry_exit_match and
                vwap_violations == 0 and
                total_trades > 0
            )
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("テスト結果サマリー")
            self.logger.info("=" * 80)
            self.logger.info(f"銘柄: {self.ticker}")
            self.logger.info(f"期間: 2024-01-01 ~ 2024-12-31")
            self.logger.info(f"\n【シグナル生成】")
            self.logger.info(f"  エントリー回数: {entry_count}")
            self.logger.info(f"  イグジット回数: {signals.get('exit_count', 0)}")
            self.logger.info(f"  エントリー/イグジット一致: {signals.get('match_status', 'N/A')}")
            
            self.logger.info(f"\n【VWAPロジック】")
            self.logger.info(f"  VWAP計算: {'OK' if vwap_logic.get('vwap_calculated') else 'NG'}")
            self.logger.info(f"  VWAP条件違反: {vwap_violations}件")
            self.logger.info(f"  出来高条件違反: {vwap_logic.get('entry_volume_violations', 0)}件")
            self.logger.info(f"  陽線条件違反: {vwap_logic.get('entry_bullish_violations', 0)}件")
            
            self.logger.info(f"\n【パフォーマンス】")
            self.logger.info(f"  総取引回数: {total_trades}")
            self.logger.info(f"  総損益: {performance.get('total_pnl', 0):,.0f}円")
            self.logger.info(f"  総損益率: {performance.get('total_pnl_pct', 0):.2f}%")
            self.logger.info(f"  勝率: {performance.get('win_rate', 0):.2f}%")
            self.logger.info(f"  平均保有期間: {performance.get('avg_hold_days', 0):.1f}日")
            
            self.logger.info(f"\n【テスト判定】")
            self.logger.info(f"  結果: {'合格' if test_passed else '不合格'}")
            
            if not test_passed:
                self.logger.warning("\nテスト不合格の理由:")
                if entry_count == 0:
                    self.logger.warning("  - エントリー回数が0")
                if not entry_exit_match:
                    self.logger.warning("  - エントリー/イグジット数が不一致")
                if vwap_violations > 0:
                    self.logger.warning(f"  - VWAP条件違反あり ({vwap_violations}件)")
                if total_trades == 0:
                    self.logger.warning("  - 取引が発生していない")
            
            self.logger.info("=" * 80)
            
            return test_passed
            
        except Exception as e:
            self.logger.error(f"レポート生成中にエラー発生: {e}", exc_info=True)
            return False
    
    def run_full_test(self) -> bool:
        """
        フルテスト実行
        
        Returns:
            bool: テスト成功=True
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("VWAP_Bounce戦略 動作検証テスト開始")
        self.logger.info(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80 + "\n")
        
        try:
            # Phase 1: データ取得
            if not self.fetch_data():
                self.logger.error("データ取得失敗")
                return False
            
            # Phase 2: バックテスト実行
            if not self.run_backtest():
                self.logger.error("バックテスト実行失敗")
                return False
            
            # Phase 3: 検証実行
            self.validate_signals()
            self.validate_vwap_logic()
            self.validate_trend_detection()
            
            # Phase 3.5: トレンド分布分析（詳細調査）
            self.analyze_trend_distribution()
            
            self.calculate_performance()
            
            # Phase 4: レポート生成
            test_passed = self.generate_summary_report()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("VWAP_Bounce戦略 動作検証テスト完了")
            self.logger.info("=" * 80 + "\n")
            
            return test_passed
            
        except Exception as e:
            self.logger.error(f"テスト実行中にエラー発生: {e}", exc_info=True)
            return False


def main():
    """メイン実行関数"""
    # テストランナー作成
    test_runner = VWAPBounceTestRunner(
        ticker="8306.T",
        index_ticker="^N225"
    )
    
    # フルテスト実行
    success = test_runner.run_full_test()
    
    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
