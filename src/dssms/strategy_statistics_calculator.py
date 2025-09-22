"""
Module: Strategy Statistics Calculator
File: strategy_statistics_calculator.py
Description: 
  Problem 戦略統計 - Strategy Statistics Sheet Quality Improvement
  戦略別統計シートの品質改善のための計算モジュール
  Problem 10準拠の8項目統計を提供し、フォーマット統一を実現

Author: DSSMS Team
Created: 2025-01-25
Version: 1.0.0

Dependencies:
  - pandas
  - numpy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from dataclasses import dataclass, field

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class StrategyStatistics:
    """戦略統計データクラス（Problem 10準拠）"""
    strategy_name: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    avg_holding_period: float = 0.0
    total_fees: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # メタデータ
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 1.0
    calculation_method: str = "Problem 10 Compliant"


class StrategyStatisticsCalculator:
    """戦略統計計算器（Problem 10準拠）"""
    
    def __init__(self, risk_free_rate: float = 0.0, trading_days: int = 252):
        """
        初期化
        
        Args:
            risk_free_rate: リスクフリーレート (デフォルト: 0.0)
            trading_days: 年間取引日数 (デフォルト: 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        logger.info(f"🔧 StrategyStatisticsCalculator初期化 - リスクフリーレート: {risk_free_rate:.3f}")
    
    def calculate_comprehensive_statistics(self, 
                                         strategy_name: str,
                                         trades_df: pd.DataFrame,
                                         daily_pnl: Optional[pd.DataFrame] = None) -> StrategyStatistics:
        """
        包括的戦略統計計算（Problem 10準拠 8項目統計）
        
        Args:
            strategy_name: 戦略名
            trades_df: 取引データフレーム
            daily_pnl: 日次損益データ（オプション）
            
        Returns:
            StrategyStatistics: 計算された統計
        """
        try:
            logger.info(f"📊 戦略統計計算開始: {strategy_name}")
            
            if trades_df.empty:
                logger.warning(f"⚠️ 空の取引データ: {strategy_name}")
                return StrategyStatistics(strategy_name=strategy_name)
            
            # 基本統計計算
            basic_stats = self._calculate_basic_statistics(trades_df)
            
            # リスク調整指標計算
            risk_metrics = self._calculate_risk_metrics(trades_df, daily_pnl)
            
            # 保有期間計算
            holding_period = self._calculate_holding_period(trades_df)
            
            # 手数料計算
            total_fees = self._calculate_total_fees(trades_df)
            
            # 統計オブジェクト作成
            statistics = StrategyStatistics(
                strategy_name=strategy_name,
                **basic_stats,
                **risk_metrics,
                avg_holding_period=holding_period,
                total_fees=total_fees,
                calculation_timestamp=datetime.now(),
                data_quality_score=self._assess_data_quality(trades_df),
                calculation_method="Problem 10 Compliant v1.0"
            )
            
            logger.info(f"✅ 戦略統計計算完了: {strategy_name}")
            return statistics
            
        except Exception as e:
            logger.error(f"❌ 戦略統計計算エラー [{strategy_name}]: {e}")
            return StrategyStatistics(
                strategy_name=strategy_name,
                data_quality_score=0.0,
                calculation_method=f"Error: {str(e)[:50]}"
            )
    
    def _calculate_basic_statistics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """基本統計計算"""
        try:
            if 'pnl' not in trades_df.columns:
                logger.warning("⚠️ PNL列が見つかりません")
                return self._get_default_basic_stats()
            
            pnl_series = trades_df['pnl']
            profit_trades = pnl_series[pnl_series > 0]
            loss_trades = pnl_series[pnl_series < 0]
            
            # 基本統計
            trade_count = len(trades_df)
            win_rate = len(profit_trades) / trade_count if trade_count > 0 else 0.0
            
            avg_profit = profit_trades.mean() if len(profit_trades) > 0 else 0.0
            avg_loss = loss_trades.mean() if len(loss_trades) > 0 else 0.0
            
            max_profit = pnl_series.max() if trade_count > 0 else 0.0
            max_loss = pnl_series.min() if trade_count > 0 else 0.0
            
            total_pnl = pnl_series.sum() if trade_count > 0 else 0.0
            
            # プロフィットファクター（Problem 10準拠）
            gross_profit = profit_trades.sum() if len(profit_trades) > 0 else 0.0
            gross_loss = abs(loss_trades.sum()) if len(loss_trades) > 0 else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            
            return {
                'trade_count': trade_count,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'total_pnl': total_pnl,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"❌ 基本統計計算エラー: {e}")
            return self._get_default_basic_stats()
    
    def _calculate_risk_metrics(self, trades_df: pd.DataFrame, daily_pnl: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """リスク調整指標計算（Problem 10準拠）"""
        try:
            # 日次リターン計算
            if daily_pnl is not None and 'daily_return' in daily_pnl.columns:
                returns = daily_pnl['daily_return'].dropna()
            elif 'pnl' in trades_df.columns:
                # 取引ベースリターン計算（近似）
                returns = trades_df['pnl'] / trades_df['pnl'].abs().shift(1).fillna(1.0)
                returns = returns.dropna()
            else:
                logger.warning("⚠️ リターン計算用データが不足")
                return self._get_default_risk_metrics()
            
            if len(returns) == 0:
                return self._get_default_risk_metrics()
            
            # ボラティリティ
            volatility = returns.std() * np.sqrt(self.trading_days)
            
            # シャープレシオ（Problem 10準拠）
            mean_return = returns.mean() * self.trading_days
            sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0.0
            
            # 最大ドローダウン
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # ソルティノレシオ（Problem 10準拠）
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(self.trading_days) if len(negative_returns) > 0 else 0.0
            sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
            
            # カルマーレシオ（Problem 10準拠）
            calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio
            }
            
        except Exception as e:
            logger.error(f"❌ リスク指標計算エラー: {e}")
            return self._get_default_risk_metrics()
    
    def _calculate_holding_period(self, trades_df: pd.DataFrame) -> float:
        """平均保有期間計算"""
        try:
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                entry_times = pd.to_datetime(trades_df['entry_time'])
                exit_times = pd.to_datetime(trades_df['exit_time'])
                holding_periods = (exit_times - entry_times).dt.days
                return holding_periods.mean()
            
            # フォールバック: データに基づく推定
            elif len(trades_df) > 1:
                # 取引間隔から推定
                return 1.0  # デフォルト1日
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ 保有期間計算エラー: {e}")
            return 0.0
    
    def _calculate_total_fees(self, trades_df: pd.DataFrame) -> float:
        """総手数料計算"""
        try:
            if 'fees' in trades_df.columns:
                return trades_df['fees'].sum()
            elif 'commission' in trades_df.columns:
                return trades_df['commission'].sum()
            
            # フォールバック: 取引量に基づく推定
            if 'volume' in trades_df.columns and 'price' in trades_df.columns:
                estimated_fees = (trades_df['volume'] * trades_df['price'] * 0.001).sum()  # 0.1%と仮定
                return estimated_fees
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ 手数料計算エラー: {e}")
            return 0.0
    
    def _assess_data_quality(self, trades_df: pd.DataFrame) -> float:
        """データ品質評価"""
        try:
            quality_score = 1.0
            
            # 必須列の存在確認
            required_columns = ['pnl']
            missing_columns = [col for col in required_columns if col not in trades_df.columns]
            if missing_columns:
                quality_score *= 0.7
            
            # データの完全性確認
            if trades_df.isnull().sum().sum() > 0:
                null_ratio = trades_df.isnull().sum().sum() / (len(trades_df) * len(trades_df.columns))
                quality_score *= (1.0 - null_ratio)
            
            # データサイズ確認
            if len(trades_df) < 10:
                quality_score *= 0.8
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"❌ データ品質評価エラー: {e}")
            return 0.5
    
    def _get_default_basic_stats(self) -> Dict[str, float]:
        """デフォルト基本統計"""
        return {
            'trade_count': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'total_pnl': 0.0,
            'profit_factor': 0.0
        }
    
    def _get_default_risk_metrics(self) -> Dict[str, float]:
        """デフォルトリスク指標"""
        return {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0
        }
    
    def format_statistics_for_export(self, statistics: StrategyStatistics) -> Dict[str, Any]:
        """統計をエクスポート用にフォーマット（フォーマット統一対応）"""
        try:
            formatted = {
                '戦略名': statistics.strategy_name,
                '取引回数': statistics.trade_count,
                '勝率(%)': round(statistics.win_rate * 100, 2),
                '平均利益': round(statistics.avg_profit, 2),
                '平均損失': round(statistics.avg_loss, 2),
                '最大利益': round(statistics.max_profit, 2),
                '最大損失': round(statistics.max_loss, 2),
                '総損益': round(statistics.total_pnl, 2),
                'プロフィットファクター': round(statistics.profit_factor, 3),
                'シャープレシオ': round(statistics.sharpe_ratio, 3),
                '最大ドローダウン(%)': round(statistics.max_drawdown * 100, 2),
                'ボラティリティ': round(statistics.volatility, 3),
                '平均保有期間(日)': round(statistics.avg_holding_period, 1),
                '総手数料': round(statistics.total_fees, 2),
                'ソルティノレシオ': round(statistics.sortino_ratio, 3),
                'カルマーレシオ': round(statistics.calmar_ratio, 3),
                'データ品質': round(statistics.data_quality_score, 3),
                '計算日時': statistics.calculation_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                '計算手法': statistics.calculation_method
            }
            
            return formatted
            
        except Exception as e:
            logger.error(f"❌ 統計フォーマットエラー: {e}")
            return {'戦略名': statistics.strategy_name, 'エラー': str(e)}


def demo_strategy_statistics_calculator():
    """戦略統計計算器デモ"""
    try:
        print("🔧 戦略統計計算器デモ開始")
        
        # サンプルデータ作成
        np.random.seed(42)
        sample_trades = pd.DataFrame({
            'pnl': np.random.normal(50, 200, 100),
            'volume': np.random.uniform(100, 1000, 100),
            'price': np.random.uniform(100, 200, 100),
            'fees': np.random.uniform(1, 10, 100)
        })
        
        # 計算器初期化
        calculator = StrategyStatisticsCalculator(risk_free_rate=0.02)
        
        # 統計計算
        stats = calculator.calculate_comprehensive_statistics(
            strategy_name="TestStrategy",
            trades_df=sample_trades
        )
        
        # フォーマット出力
        formatted = calculator.format_statistics_for_export(stats)
        
        print("\n📊 計算結果:")
        for key, value in formatted.items():
            print(f"  {key}: {value}")
        
        print("\n✅ デモ完了")
        return True
        
    except Exception as e:
        print(f"❌ デモエラー: {e}")
        return False


if __name__ == "__main__":
    demo_strategy_statistics_calculator()