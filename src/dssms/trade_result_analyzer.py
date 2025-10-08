"""
DSSMS Phase 2 Task 2.2: 取引結果分析システム
Trade Result Analyzer - 高度な取引分析と統計処理

主要機能:
1. 取引結果の詳細分析
2. パフォーマンス指標の計算
3. 勝率・プロフィットファクターの算出
4. リスク調整済みリターンの評価
5. 取引パターンの統計分析

Author: GitHub Copilot Agent
Created: 2025-01-22
Task: Phase 2 Task 2.2 - パフォーマンス計算エンジン修正
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
import json
from scipy import stats
import traceback

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

# 警告を抑制
warnings.filterwarnings('ignore')

class TradeType(Enum):
    """取引タイプ"""
    LONG = "long"
    SHORT = "short"
    SWING = "swing"
    DAY = "day"

class TradeStatus(Enum):
    """取引ステータス"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"

class AnalysisLevel(Enum):
    """分析レベル"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"

@dataclass
class TradeRecord:
    """取引記録"""
    trade_id: str
    symbol: str
    trade_type: TradeType
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: Optional[float]
    commission: float
    status: TradeStatus
    strategy: str = "unknown"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradeStatistics:
    """取引統計"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    total_pnl: float
    gross_profit: float
    gross_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    average_trade_duration: timedelta
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float

@dataclass
class PerformanceBreakdown:
    """パフォーマンス内訳"""
    by_strategy: Dict[str, TradeStatistics]
    by_symbol: Dict[str, TradeStatistics]
    by_trade_type: Dict[str, TradeStatistics]
    by_time_period: Dict[str, TradeStatistics]
    monthly_returns: Dict[str, float]
    daily_returns: pd.Series

@dataclass
class RiskMetrics:
    """リスク指標"""
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var_95: float
    maximum_drawdown: float
    drawdown_duration: timedelta
    downside_deviation: float
    sortino_ratio: float
    beta: float
    alpha: float
    information_ratio: float

class TradeResultAnalyzer:
    """
    取引結果分析システム
    高度な取引分析と統計処理機能
    """
    
    def __init__(self, analysis_level: AnalysisLevel = AnalysisLevel.COMPREHENSIVE):
        """
        Args:
            analysis_level: 分析レベル
        """
        self.logger = setup_logger(__name__)
        self.analysis_level = analysis_level
        
        # 取引データの管理
        self.trade_records: List[TradeRecord] = []
        self.cached_statistics: Optional[TradeStatistics] = None
        self.cached_breakdown: Optional[PerformanceBreakdown] = None
        self.cache_timestamp: Optional[datetime] = None
        
        # 設定
        self.risk_free_rate = 0.001  # リスクフリーレート
        self.trading_days_per_year = 252
        self.cache_expiry_minutes = 30
        
        self.logger.info(f"TradeResultAnalyzer初期化完了 (レベル: {analysis_level.value})")
    
    def add_trade_record(self, trade_record: TradeRecord):
        """取引記録の追加"""
        try:
            self.trade_records.append(trade_record)
            self._invalidate_cache()
            self.logger.debug(f"取引記録追加: {trade_record.trade_id}")
        except Exception as e:
            self.logger.error(f"取引記録追加エラー: {e}")
    
    def add_trades_from_dataframe(self, trades_df: pd.DataFrame) -> int:
        """
        DataFrameから取引記録を一括追加
        
        Args:
            trades_df: 取引データのDataFrame
            
        Returns:
            追加された取引数
        """
        try:
            added_count = 0
            
            for index, row in trades_df.iterrows():
                try:
                    # データの型変換と検証
                    trade_record = self._dataframe_row_to_trade_record(row, index)
                    if trade_record:
                        self.add_trade_record(trade_record)
                        added_count += 1
                except Exception as e:
                    self.logger.warning(f"行{index}の取引記録変換エラー: {e}")
                    continue
            
            self.logger.info(f"取引記録一括追加完了: {added_count}件")
            return added_count
            
        except Exception as e:
            self.logger.error(f"取引記録一括追加エラー: {e}")
            return 0
    
    def _dataframe_row_to_trade_record(self, row: pd.Series, index: Any) -> Optional[TradeRecord]:
        """DataFrameの行を取引記録に変換"""
        try:
            # 必須フィールドの確認
            required_fields = ['symbol', 'entry_price', 'quantity']
            missing_fields = [field for field in required_fields if field not in row or pd.isna(row[field])]
            
            if missing_fields:
                self.logger.warning(f"必須フィールド不足: {missing_fields}")
                return None
            
            # 取引IDの生成
            trade_id = row.get('trade_id', f"trade_{index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # 取引タイプの判定
            trade_type_str = row.get('trade_type', 'long').lower()
            try:
                trade_type = TradeType(trade_type_str)
            except ValueError:
                trade_type = TradeType.LONG
            
            # 時刻データの処理
            entry_time = row.get('entry_time', datetime.now())
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            
            exit_time = row.get('exit_time')
            if exit_time and isinstance(exit_time, str):
                exit_time = pd.to_datetime(exit_time)
            
            # P&Lの計算
            pnl = row.get('pnl')
            if pnl is None and 'exit_price' in row and not pd.isna(row['exit_price']):
                entry_price = float(row['entry_price'])
                exit_price = float(row['exit_price'])
                quantity = float(row['quantity'])
                
                if trade_type == TradeType.LONG:
                    pnl = (exit_price - entry_price) * quantity
                else:  # SHORT
                    pnl = (entry_price - exit_price) * quantity
            
            # ステータスの判定
            status_str = row.get('status', 'closed' if exit_time else 'open').lower()
            try:
                status = TradeStatus(status_str)
            except ValueError:
                status = TradeStatus.CLOSED if exit_time else TradeStatus.OPEN
            
            return TradeRecord(
                trade_id=str(trade_id),
                symbol=str(row['symbol']),
                trade_type=trade_type,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=float(row['entry_price']),
                exit_price=float(row.get('exit_price', 0)) if 'exit_price' in row and not pd.isna(row['exit_price']) else None,
                quantity=float(row['quantity']),
                pnl=float(pnl) if pnl is not None else None,
                commission=float(row.get('commission', 0)),
                status=status,
                strategy=str(row.get('strategy', 'unknown')),
                confidence=float(row.get('confidence', 1.0)),
                metadata={k: v for k, v in row.items() if k not in [
                    'trade_id', 'symbol', 'trade_type', 'entry_time', 'exit_time',
                    'entry_price', 'exit_price', 'quantity', 'pnl', 'commission',
                    'status', 'strategy', 'confidence'
                ]}
            )
            
        except Exception as e:
            self.logger.error(f"取引記録変換エラー: {e}")
            return None
    
    def calculate_comprehensive_statistics(self) -> TradeStatistics:
        """包括的統計の計算"""
        if self._is_cache_valid() and self.cached_statistics:
            return self.cached_statistics
        
        try:
            # クローズ済み取引のフィルタリング
            closed_trades = [trade for trade in self.trade_records 
                           if trade.status == TradeStatus.CLOSED and trade.pnl is not None]
            
            if not closed_trades:
                return self._get_empty_statistics()
            
            # 基本統計の計算
            total_trades = len(closed_trades)
            pnls = [trade.pnl for trade in closed_trades]
            winning_trades = len([pnl for pnl in pnls if pnl > 0])
            losing_trades = len([pnl for pnl in pnls if pnl < 0])
            
            # 勝率の計算
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # 平均勝ち・負けの計算
            wins = [pnl for pnl in pnls if pnl > 0]
            losses = [pnl for pnl in pnls if pnl < 0]
            
            average_win = np.mean(wins) if wins else 0.0
            average_loss = np.mean(losses) if losses else 0.0
            
            # プロフィットファクターの計算
            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # 総P&Lの計算
            total_pnl = sum(pnls)
            
            # 連続勝ち・負けの計算
            max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_streaks(pnls)
            
            # 平均取引期間の計算
            average_trade_duration = self._calculate_average_trade_duration(closed_trades)
            
            # リスク調整指標の計算
            if self.analysis_level in [AnalysisLevel.ADVANCED, AnalysisLevel.COMPREHENSIVE]:
                sharpe_ratio = self._calculate_sharpe_ratio(pnls)
                calmar_ratio = self._calculate_calmar_ratio(pnls)
                max_drawdown = self._calculate_max_drawdown(pnls)
            else:
                sharpe_ratio = 0.0
                calmar_ratio = 0.0
                max_drawdown = 0.0
            
            statistics = TradeStatistics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                average_win=average_win,
                average_loss=average_loss,
                profit_factor=profit_factor,
                total_pnl=total_pnl,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                average_trade_duration=average_trade_duration,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown
            )
            
            # キャッシュの更新
            self.cached_statistics = statistics
            self.cache_timestamp = datetime.now()
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"統計計算エラー: {e}")
            self.logger.error(traceback.format_exc())
            return self._get_empty_statistics()
    
    def generate_performance_breakdown(self) -> PerformanceBreakdown:
        """パフォーマンス内訳の生成"""
        if self._is_cache_valid() and self.cached_breakdown:
            return self.cached_breakdown
        
        try:
            # クローズ済み取引のフィルタリング
            closed_trades = [trade for trade in self.trade_records 
                           if trade.status == TradeStatus.CLOSED and trade.pnl is not None]
            
            if not closed_trades:
                return self._get_empty_breakdown()
            
            # 戦略別分析
            by_strategy = self._analyze_by_category(closed_trades, 'strategy')
            
            # 銘柄別分析
            by_symbol = self._analyze_by_category(closed_trades, 'symbol')
            
            # 取引タイプ別分析
            by_trade_type = self._analyze_by_category(closed_trades, 'trade_type')
            
            # 時期別分析
            by_time_period = self._analyze_by_time_period(closed_trades)
            
            # 月次リターンの計算
            monthly_returns = self._calculate_monthly_returns(closed_trades)
            
            # 日次リターンの計算
            daily_returns = self._calculate_daily_returns(closed_trades)
            
            breakdown = PerformanceBreakdown(
                by_strategy=by_strategy,
                by_symbol=by_symbol,
                by_trade_type=by_trade_type,
                by_time_period=by_time_period,
                monthly_returns=monthly_returns,
                daily_returns=daily_returns
            )
            
            # キャッシュの更新
            self.cached_breakdown = breakdown
            self.cache_timestamp = datetime.now()
            
            return breakdown
            
        except Exception as e:
            self.logger.error(f"パフォーマンス内訳生成エラー: {e}")
            return self._get_empty_breakdown()
    
    def calculate_risk_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """リスク指標の計算"""
        try:
            # クローズ済み取引のフィルタリング
            closed_trades = [trade for trade in self.trade_records 
                           if trade.status == TradeStatus.CLOSED and trade.pnl is not None]
            
            if not closed_trades:
                return self._get_empty_risk_metrics()
            
            pnls = [trade.pnl for trade in closed_trades]
            
            # VaR (Value at Risk) の計算
            var_95 = np.percentile(pnls, 5)  # 95% VaR
            var_99 = np.percentile(pnls, 1)  # 99% VaR
            
            # CVaR (Conditional VaR) の計算
            cvar_95 = np.mean([pnl for pnl in pnls if pnl <= var_95])
            
            # ドローダウンの計算
            max_drawdown = self._calculate_max_drawdown(pnls)
            drawdown_duration = self._calculate_drawdown_duration(closed_trades)
            
            # 下方偏差の計算
            downside_returns = [pnl for pnl in pnls if pnl < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0.0
            
            # ソルティノレシオの計算
            mean_return = np.mean(pnls)
            sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
            
            # ベータとアルファの計算（ベンチマークがある場合）
            if benchmark_returns is not None:
                beta, alpha = self._calculate_beta_alpha(pnls, benchmark_returns)
                information_ratio = self._calculate_information_ratio(pnls, benchmark_returns)
            else:
                beta = 0.0
                alpha = 0.0
                information_ratio = 0.0
            
            return RiskMetrics(
                value_at_risk_95=var_95,
                value_at_risk_99=var_99,
                conditional_var_95=cvar_95,
                maximum_drawdown=max_drawdown,
                drawdown_duration=drawdown_duration,
                downside_deviation=downside_deviation,
                sortino_ratio=sortino_ratio,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio
            )
            
        except Exception as e:
            self.logger.error(f"リスク指標計算エラー: {e}")
            return self._get_empty_risk_metrics()
    
    def _analyze_by_category(self, trades: List[TradeRecord], category: str) -> Dict[str, TradeStatistics]:
        """カテゴリ別分析"""
        try:
            categories = {}
            
            # カテゴリ別にグループ化
            for trade in trades:
                if category == 'strategy':
                    key = trade.strategy
                elif category == 'symbol':
                    key = trade.symbol
                elif category == 'trade_type':
                    key = trade.trade_type.value
                else:
                    continue
                
                if key not in categories:
                    categories[key] = []
                categories[key].append(trade)
            
            # 各カテゴリの統計を計算
            result = {}
            for key, category_trades in categories.items():
                pnls = [trade.pnl for trade in category_trades]
                result[key] = self._calculate_statistics_for_trades(category_trades, pnls)
            
            return result
            
        except Exception as e:
            self.logger.error(f"カテゴリ別分析エラー: {e}")
            return {}
    
    def _analyze_by_time_period(self, trades: List[TradeRecord]) -> Dict[str, TradeStatistics]:
        """時期別分析"""
        try:
            periods = {}
            
            for trade in trades:
                if trade.exit_time:
                    # 月単位でグループ化
                    period_key = trade.exit_time.strftime('%Y-%m')
                    
                    if period_key not in periods:
                        periods[period_key] = []
                    periods[period_key].append(trade)
            
            # 各期間の統計を計算
            result = {}
            for period, period_trades in periods.items():
                pnls = [trade.pnl for trade in period_trades]
                result[period] = self._calculate_statistics_for_trades(period_trades, pnls)
            
            return result
            
        except Exception as e:
            self.logger.error(f"時期別分析エラー: {e}")
            return {}
    
    def _calculate_statistics_for_trades(self, trades: List[TradeRecord], pnls: List[float]) -> TradeStatistics:
        """取引リストの統計計算"""
        if not trades or not pnls:
            return self._get_empty_statistics()
        
        total_trades = len(pnls)
        winning_trades = len([pnl for pnl in pnls if pnl > 0])
        losing_trades = len([pnl for pnl in pnls if pnl < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        average_win = np.mean(wins) if wins else 0.0
        average_loss = np.mean(losses) if losses else 0.0
        
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        total_pnl = sum(pnls)
        
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_streaks(pnls)
        average_trade_duration = self._calculate_average_trade_duration(trades)
        
        return TradeStatistics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            average_trade_duration=average_trade_duration,
            sharpe_ratio=self._calculate_sharpe_ratio(pnls),
            calmar_ratio=self._calculate_calmar_ratio(pnls),
            max_drawdown=self._calculate_max_drawdown(pnls)
        )
    
    def _calculate_consecutive_streaks(self, pnls: List[float]) -> Tuple[int, int]:
        """連続勝ち・負けの計算"""
        if not pnls:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:  # pnl == 0
                current_wins = 0
                current_losses = 0
        
        return max_wins, max_losses
    
    def _calculate_average_trade_duration(self, trades: List[TradeRecord]) -> timedelta:
        """平均取引期間の計算"""
        durations = []
        
        for trade in trades:
            if trade.exit_time and trade.entry_time:
                duration = trade.exit_time - trade.entry_time
                durations.append(duration)
        
        if durations:
            avg_seconds = sum(d.total_seconds() for d in durations) / len(durations)
            return timedelta(seconds=avg_seconds)
        else:
            return timedelta(0)
    
    def _calculate_sharpe_ratio(self, pnls: List[float]) -> float:
        """シャープレシオの計算"""
        if not pnls or len(pnls) < 2:
            return 0.0
        
        mean_return = np.mean(pnls)
        std_return = np.std(pnls, ddof=1)
        
        if std_return > 0:
            return (mean_return - self.risk_free_rate) / std_return
        else:
            return 0.0
    
    def _calculate_calmar_ratio(self, pnls: List[float]) -> float:
        """カルマーレシオの計算"""
        if not pnls:
            return 0.0
        
        annual_return = np.mean(pnls) * self.trading_days_per_year
        max_drawdown = self._calculate_max_drawdown(pnls)
        
        if max_drawdown > 0:
            return annual_return / max_drawdown
        else:
            return 0.0
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """最大ドローダウンの計算"""
        if not pnls:
            return 0.0
        
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max)
        
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    def _calculate_drawdown_duration(self, trades: List[TradeRecord]) -> timedelta:
        """ドローダウン期間の計算"""
        if not trades:
            return timedelta(0)
        
        # 簡略化した実装 - 実際はより複雑な計算が必要
        total_duration = timedelta(0)
        if len(trades) > 1:
            start_time = min(trade.entry_time for trade in trades)
            end_time = max(trade.exit_time for trade in trades if trade.exit_time)
            if end_time:
                total_duration = end_time - start_time
        
        return total_duration
    
    def _calculate_beta_alpha(self, returns: List[float], benchmark_returns: pd.Series) -> Tuple[float, float]:
        """ベータとアルファの計算"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0, 0.0
        
        try:
            # 線形回帰でベータを計算
            slope, intercept, r_value, p_value, std_err = stats.linregress(benchmark_returns, returns)
            beta = slope
            alpha = intercept
            
            return beta, alpha
            
        except Exception as e:
            self.logger.error(f"ベータ・アルファ計算エラー: {e}")
            return 0.0, 0.0
    
    def _calculate_information_ratio(self, returns: List[float], benchmark_returns: pd.Series) -> float:
        """インフォメーションレシオの計算"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        try:
            excess_returns = np.array(returns) - np.array(benchmark_returns)
            tracking_error = np.std(excess_returns, ddof=1)
            
            if tracking_error > 0:
                return np.mean(excess_returns) / tracking_error
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"インフォメーションレシオ計算エラー: {e}")
            return 0.0
    
    def _calculate_monthly_returns(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """月次リターンの計算"""
        monthly_pnl = {}
        
        for trade in trades:
            if trade.exit_time and trade.pnl is not None:
                month_key = trade.exit_time.strftime('%Y-%m')
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0.0
                monthly_pnl[month_key] += trade.pnl
        
        return monthly_pnl
    
    def _calculate_daily_returns(self, trades: List[TradeRecord]) -> pd.Series:
        """日次リターンの計算"""
        daily_pnl = {}
        
        for trade in trades:
            if trade.exit_time and trade.pnl is not None:
                date_key = trade.exit_time.date()
                if date_key not in daily_pnl:
                    daily_pnl[date_key] = 0.0
                daily_pnl[date_key] += trade.pnl
        
        if daily_pnl:
            return pd.Series(daily_pnl, name='daily_returns')
        else:
            return pd.Series([], name='daily_returns')
    
    def _is_cache_valid(self) -> bool:
        """キャッシュの有効性確認"""
        if not self.cache_timestamp:
            return False
        
        elapsed = datetime.now() - self.cache_timestamp
        return elapsed.total_seconds() < (self.cache_expiry_minutes * 60)
    
    def _invalidate_cache(self):
        """キャッシュの無効化"""
        self.cached_statistics = None
        self.cached_breakdown = None
        self.cache_timestamp = None
    
    def _get_empty_statistics(self) -> TradeStatistics:
        """空の統計データ"""
        return TradeStatistics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=1.0,
            total_pnl=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            average_trade_duration=timedelta(0),
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0
        )
    
    def _get_empty_breakdown(self) -> PerformanceBreakdown:
        """空のパフォーマンス内訳"""
        return PerformanceBreakdown(
            by_strategy={},
            by_symbol={},
            by_trade_type={},
            by_time_period={},
            monthly_returns={},
            daily_returns=pd.Series([], name='daily_returns')
        )
    
    def _get_empty_risk_metrics(self) -> RiskMetrics:
        """空のリスク指標"""
        return RiskMetrics(
            value_at_risk_95=0.0,
            value_at_risk_99=0.0,
            conditional_var_95=0.0,
            maximum_drawdown=0.0,
            drawdown_duration=timedelta(0),
            downside_deviation=0.0,
            sortino_ratio=0.0,
            beta=0.0,
            alpha=0.0,
            information_ratio=0.0
        )
    
    def export_analysis_report(self) -> Dict[str, Any]:
        """分析レポートの出力"""
        try:
            statistics = self.calculate_comprehensive_statistics()
            breakdown = self.generate_performance_breakdown()
            risk_metrics = self.calculate_risk_metrics()
            
            return {
                'summary': {
                    'total_trades': statistics.total_trades,
                    'win_rate': statistics.win_rate,
                    'profit_factor': statistics.profit_factor,
                    'total_pnl': statistics.total_pnl,
                    'sharpe_ratio': statistics.sharpe_ratio,
                    'max_drawdown': statistics.max_drawdown
                },
                'detailed_statistics': statistics,
                'performance_breakdown': breakdown,
                'risk_metrics': risk_metrics,
                'analysis_level': self.analysis_level.value,
                'report_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"分析レポート出力エラー: {e}")
            return {'error': str(e)}

def main():
    """メイン実行関数"""
    print("DSSMS Task 2.2: 取引結果分析システム")
    print("=" * 45)
    
    try:
        # 取引結果分析システムの初期化
        analyzer = TradeResultAnalyzer(AnalysisLevel.COMPREHENSIVE)
        
        # サンプル取引データの作成
        sample_trades = pd.DataFrame({
            'trade_id': [f'T{i:03d}' for i in range(1, 21)],
            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'] * 4,
            'trade_type': ['long'] * 15 + ['short'] * 5,
            'entry_time': pd.date_range(start='2024-01-01', periods=20, freq='D'),
            'exit_time': pd.date_range(start='2024-01-02', periods=20, freq='D'),
            'entry_price': np.random.uniform(100, 200, 20),
            'exit_price': np.random.uniform(95, 210, 20),
            'quantity': np.random.randint(10, 100, 20),
            'commission': np.random.uniform(1, 5, 20),
            'strategy': ['VWAP', 'RSI', 'MACD'] * 6 + ['Momentum', 'MeanReversion']
        })
        
        # P&Lの計算
        sample_trades['pnl'] = (sample_trades['exit_price'] - sample_trades['entry_price']) * sample_trades['quantity']
        sample_trades['status'] = 'closed'
        
        # 取引データの追加
        added_count = analyzer.add_trades_from_dataframe(sample_trades)
        print(f"\n[CHART] サンプル取引データ追加: {added_count}件")
        
        # 統計の計算
        statistics = analyzer.calculate_comprehensive_statistics()
        print(f"\n[UP] 基本統計:")
        print(f"  総取引数: {statistics.total_trades}")
        print(f"  勝率: {statistics.win_rate:.2%}")
        print(f"  プロフィットファクター: {statistics.profit_factor:.2f}")
        print(f"  総P&L: ¥{statistics.total_pnl:,.0f}")
        print(f"  シャープレシオ: {statistics.sharpe_ratio:.3f}")
        print(f"  最大ドローダウン: ¥{statistics.max_drawdown:,.0f}")
        
        # パフォーマンス内訳の生成
        breakdown = analyzer.generate_performance_breakdown()
        print(f"\n[TARGET] 戦略別パフォーマンス:")
        for strategy, stats in breakdown.by_strategy.items():
            print(f"  {strategy}: 勝率 {stats.win_rate:.2%}, P&L ¥{stats.total_pnl:,.0f}")
        
        # リスク指標の計算
        risk_metrics = analyzer.calculate_risk_metrics()
        print(f"\n[WARNING]  リスク指標:")
        print(f"  VaR(95%): ¥{risk_metrics.value_at_risk_95:,.0f}")
        print(f"  最大ドローダウン: ¥{risk_metrics.maximum_drawdown:,.0f}")
        print(f"  ソルティノレシオ: {risk_metrics.sortino_ratio:.3f}")
        
        # 分析レポートの出力
        report = analyzer.export_analysis_report()
        print(f"\n[LIST] 分析レポート生成完了")
        print(f"  分析レベル: {report['analysis_level']}")
        print(f"  レポート作成時刻: {report['report_timestamp']}")
        
        print(f"\n[OK] 取引結果分析システム: 正常動作確認")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] エラー: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
