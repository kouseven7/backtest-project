"""
Module: Trend Performance Calculator
File: trend_performance_calculator.py
Description: 
  トレンド期間別のパフォーマンス指標を計算し、詳細な分析を提供するモジュールです。
  既存のperformance_metrics.pyと連携して、Sharpe ratio、最大ドローダウン、
  リスク調整後リターンなどの指標をトレンド別に計算・比較・保存します。

Author: imega
Created: 2025-07-08

Dependencies:
  - pandas
  - numpy
  - metrics.performance_metrics
  - config.logger_config
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# 相対インポート
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.performance_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_expectancy,
    calculate_max_consecutive_losses,
    calculate_max_consecutive_wins,
    calculate_max_drawdown,
    calculate_max_drawdown_amount,
    calculate_win_rate,
    calculate_total_profit,
    calculate_average_profit,
    calculate_max_profit,
    calculate_max_loss,
    calculate_risk_return_ratio
)

from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__)


class TrendPerformanceCalculator:
    """
    トレンド別パフォーマンス指標計算クラス
    
    バックテスト結果をトレンド期間別に分析し、各種パフォーマンス指標を計算します。
    トレンド環境別の比較分析や、リスク調整後リターンの評価を提供します。
    """
    
    def __init__(self, 
                 output_dir: str = "logs",
                 risk_free_rate: float = 0.0,
                 trading_days: int = 252):
        """
        初期化
        
        Parameters:
            output_dir (str): 結果保存ディレクトリ
            risk_free_rate (float): 無リスク利子率
            trading_days (int): 年間取引日数
        """
        self.output_dir = output_dir
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        
        # 出力ディレクトリの作成
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 計算結果を保存する辞書
        self.performance_results: Dict[str, Dict[str, Any]] = {}
        
    def calculate_trend_performance_metrics(self, 
                                          backtest_results: Dict[str, Any],
                                          strategy_name: str = "strategy") -> Dict[str, Any]:
        """
        トレンド別パフォーマンス指標を計算
        
        Parameters:
            backtest_results (Dict): バックテスト結果（トレンド別）
            strategy_name (str): 戦略名
            
        Returns:
            Dict: トレンド別パフォーマンス指標
        """
        logger.info(f"トレンド別パフォーマンス指標計算開始: {strategy_name}")
        
        # 結果を保存するための辞書
        trend_performance = {
            "strategy_name": strategy_name,
            "calculation_timestamp": datetime.now().isoformat(),
            "risk_free_rate": self.risk_free_rate,
            "trading_days": self.trading_days,
            "trend_metrics": {},
            "comparative_analysis": {},
            "overall_summary": {}
        }
        
        # 各トレンドタイプ別に指標を計算
        trend_types = ["uptrend", "downtrend", "sideways"]
        
        for trend_type in trend_types:
            if trend_type in backtest_results:
                logger.info(f"{trend_type}のパフォーマンス指標を計算中...")
                trend_metrics = self._calculate_single_trend_metrics(
                    backtest_results[trend_type], trend_type
                )
                trend_performance["trend_metrics"][trend_type] = trend_metrics
        
        # 比較分析の実行
        trend_performance["comparative_analysis"] = self._perform_comparative_analysis(
            trend_performance["trend_metrics"]
        )
        
        # 全体サマリーの計算
        trend_performance["overall_summary"] = self._calculate_overall_summary(
            trend_performance["trend_metrics"]
        )
        
        # 結果を保存
        self.performance_results[strategy_name] = trend_performance
        
        logger.info(f"トレンド別パフォーマンス指標計算完了: {strategy_name}")
        return trend_performance
    
    def _calculate_single_trend_metrics(self, 
                                      trend_data: Dict[str, Any], 
                                      trend_type: str) -> Dict[str, Any]:
        """
        単一トレンドタイプのパフォーマンス指標を計算
        
        Parameters:
            trend_data (Dict): 単一トレンドのバックテスト結果
            trend_type (str): トレンドタイプ
            
        Returns:
            Dict: パフォーマンス指標
        """
        try:
            metrics = {
                "trend_type": trend_type,
                "period_count": 0,
                "total_trading_days": 0,
                "basic_metrics": {},
                "risk_metrics": {},
                "profitability_metrics": {},
                "consistency_metrics": {}
            }
            
            # データが存在するかチェック
            if "trades" not in trend_data or not trend_data["trades"]:
                logger.warning(f"{trend_type}にトレードデータがありません")
                return metrics
            
            # トレードデータの準備
            trades_df = pd.DataFrame(trend_data["trades"])
            if "取引結果" not in trades_df.columns:
                if "profit" in trades_df.columns:
                    trades_df["取引結果"] = trades_df["profit"]
                else:
                    logger.error(f"{trend_type}のトレードデータに利益カラムがありません")
                    return metrics
            
            # 期間情報の設定
            if "periods" in trend_data:
                periods_data = trend_data["periods"]
                if isinstance(periods_data, (list, tuple)):
                    metrics["period_count"] = len(periods_data)
                elif isinstance(periods_data, int):
                    metrics["period_count"] = periods_data
                else:
                    metrics["period_count"] = 0
            
            if "total_days" in trend_data:
                metrics["total_trading_days"] = trend_data["total_days"]
            
            # 基本指標の計算
            metrics["basic_metrics"] = self._calculate_basic_metrics(trades_df)
            
            # リスク指標の計算
            metrics["risk_metrics"] = self._calculate_risk_metrics(trades_df)
            
            # 収益性指標の計算
            metrics["profitability_metrics"] = self._calculate_profitability_metrics(trades_df)
            
            # 一貫性指標の計算
            metrics["consistency_metrics"] = self._calculate_consistency_metrics(trades_df)
            
            return metrics
            
        except Exception as e:
            logger.error(f"{trend_type}の指標計算中にエラー: {e}")
            return {
                "trend_type": trend_type,
                "error": str(e),
                "period_count": 0,
                "total_trading_days": 0
            }
    
    def _calculate_basic_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """基本指標の計算"""
        returns = trades_df["取引結果"]
        
        # 累積損益の計算
        cumulative_pnl = returns.cumsum()
        
        basic_metrics = {
            "total_trades": len(trades_df),
            "total_profit": calculate_total_profit(trades_df),
            "average_profit": calculate_average_profit(trades_df),
            "win_rate": calculate_win_rate(trades_df),
            "max_profit": calculate_max_profit(trades_df),
            "max_loss": calculate_max_loss(trades_df),
            "expectancy": calculate_expectancy(trades_df)
        }
        
        return basic_metrics
    
    def _calculate_risk_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """リスク指標の計算"""
        returns = trades_df["取引結果"]
        cumulative_pnl = returns.cumsum()
        
        risk_metrics = {
            "sharpe_ratio": calculate_sharpe_ratio(returns, self.risk_free_rate, self.trading_days),
            "sortino_ratio": calculate_sortino_ratio(returns, self.risk_free_rate, self.trading_days),
            "max_drawdown_percent": calculate_max_drawdown(cumulative_pnl),
            "max_drawdown_amount": calculate_max_drawdown_amount(cumulative_pnl),
            "volatility": returns.std() * np.sqrt(self.trading_days),
            "downside_volatility": self._calculate_downside_volatility(returns)
        }
        
        # リスクリターン比率の計算
        total_profit = calculate_total_profit(trades_df)
        max_dd = risk_metrics["max_drawdown_amount"]
        if isinstance(max_dd, (int, float)) and not np.isnan(max_dd):
            risk_metrics["risk_return_ratio"] = calculate_risk_return_ratio(total_profit, abs(max_dd))
        else:
            risk_metrics["risk_return_ratio"] = 0.0
        
        return risk_metrics
    
    def _calculate_profitability_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """収益性指標の計算"""
        winning_trades = trades_df[trades_df["取引結果"] > 0]
        losing_trades = trades_df[trades_df["取引結果"] < 0]
        
        profitability_metrics = {
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "win_loss_ratio": 0.0,
            "largest_win": calculate_max_profit(trades_df),
            "largest_loss": calculate_max_loss(trades_df)
        }
        
        # 勝率関連の計算
        if len(winning_trades) > 0:
            profitability_metrics["average_win"] = winning_trades["取引結果"].mean()
        
        if len(losing_trades) > 0:
            profitability_metrics["average_loss"] = losing_trades["取引結果"].mean()
        
        # プロフィットファクター
        total_wins = winning_trades["取引結果"].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades["取引結果"].sum()) if len(losing_trades) > 0 else 1
        profitability_metrics["profit_factor"] = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # 勝敗比率
        if profitability_metrics["average_loss"] < 0:
            profitability_metrics["win_loss_ratio"] = (
                profitability_metrics["average_win"] / abs(profitability_metrics["average_loss"])
            )
        
        return profitability_metrics
    
    def _calculate_consistency_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """一貫性指標の計算"""
        consistency_metrics = {
            "max_consecutive_wins": calculate_max_consecutive_wins(trades_df),
            "max_consecutive_losses": calculate_max_consecutive_losses(trades_df),
            "winning_streak_avg": 0.0,
            "losing_streak_avg": 0.0,
            "consistency_score": 0.0
        }
        
        # 連勝・連敗の平均の計算
        returns = trades_df["取引結果"]
        wins = (returns > 0).astype(int)
        losses = (returns < 0).astype(int)
        
        # 連勝平均の計算
        win_streaks = wins * (wins.groupby((wins != wins.shift()).cumsum()).cumcount() + 1)
        win_streak_values = win_streaks[win_streaks > 0]
        if len(win_streak_values) > 0:
            consistency_metrics["winning_streak_avg"] = win_streak_values.mean()
        
        # 連敗平均の計算
        loss_streaks = losses * (losses.groupby((losses != losses.shift()).cumsum()).cumcount() + 1)
        loss_streak_values = loss_streaks[loss_streaks > 0]
        if len(loss_streak_values) > 0:
            consistency_metrics["losing_streak_avg"] = loss_streak_values.mean()
        
        # 一貫性スコア（勝率と平均利益の積）
        win_rate = calculate_win_rate(trades_df) / 100  # パーセンテージを割合に変換
        avg_profit = calculate_average_profit(trades_df)
        consistency_metrics["consistency_score"] = win_rate * max(0, avg_profit)
        
        return consistency_metrics
    
    def _calculate_downside_volatility(self, returns: pd.Series) -> float:
        """下方ボラティリティの計算"""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        return negative_returns.std() * np.sqrt(self.trading_days)
    
    def _perform_comparative_analysis(self, trend_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """トレンド間の比較分析"""
        comparative_analysis = {
            "best_trend_by_metric": {},
            "worst_trend_by_metric": {},
            "trend_ranking": {},
            "performance_spread": {}
        }
        
        # 比較する主要指標
        key_metrics = [
            "total_profit", "win_rate", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_percent", "profit_factor", "consistency_score"
        ]
        
        for metric in key_metrics:
            metric_values = {}
            
            # 各トレンドから指標値を取得
            for trend_type, metrics in trend_metrics.items():
                if trend_type in ["uptrend", "downtrend", "sideways"]:
                    value = self._extract_metric_value(metrics, metric)
                    if value is not None:
                        metric_values[trend_type] = value
            
            if metric_values:
                # 最高・最低の判定
                if metric in ["max_drawdown_percent"]:  # 小さい方が良い指標
                    best_trend = min(metric_values, key=metric_values.get)
                    worst_trend = max(metric_values, key=metric_values.get)
                else:  # 大きい方が良い指標
                    best_trend = max(metric_values, key=metric_values.get)
                    worst_trend = min(metric_values, key=metric_values.get)
                
                comparative_analysis["best_trend_by_metric"][metric] = {
                    "trend": best_trend,
                    "value": metric_values[best_trend]
                }
                comparative_analysis["worst_trend_by_metric"][metric] = {
                    "trend": worst_trend,
                    "value": metric_values[worst_trend]
                }
                
                # パフォーマンススプレッドの計算
                max_val = max(metric_values.values())
                min_val = min(metric_values.values())
                comparative_analysis["performance_spread"][metric] = {
                    "range": max_val - min_val,
                    "coefficient_of_variation": np.std(list(metric_values.values())) / np.mean(list(metric_values.values()))
                    if np.mean(list(metric_values.values())) != 0 else 0
                }
        
        # 総合ランキングの計算
        comparative_analysis["trend_ranking"] = self._calculate_trend_ranking(trend_metrics)
        
        return comparative_analysis
    
    def _extract_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """指標値を階層構造から抽出"""
        # 基本指標から探す
        if "basic_metrics" in metrics and metric_name in metrics["basic_metrics"]:
            return metrics["basic_metrics"][metric_name]
        
        # リスク指標から探す
        if "risk_metrics" in metrics and metric_name in metrics["risk_metrics"]:
            return metrics["risk_metrics"][metric_name]
        
        # 収益性指標から探す
        if "profitability_metrics" in metrics and metric_name in metrics["profitability_metrics"]:
            return metrics["profitability_metrics"][metric_name]
        
        # 一貫性指標から探す
        if "consistency_metrics" in metrics and metric_name in metrics["consistency_metrics"]:
            return metrics["consistency_metrics"][metric_name]
        
        return None
    
    def _calculate_trend_ranking(self, trend_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """トレンドの総合ランキングを計算"""
        # 重み付けスコア計算用の指標と重み
        scoring_metrics = {
            "total_profit": 0.25,
            "sharpe_ratio": 0.20,
            "win_rate": 0.15,
            "consistency_score": 0.15,
            "sortino_ratio": 0.15,
            "profit_factor": 0.10
        }
        
        trend_scores = {}
        
        for trend_type in ["uptrend", "downtrend", "sideways"]:
            if trend_type in trend_metrics:
                score = 0.0
                total_weight = 0.0
                
                for metric, weight in scoring_metrics.items():
                    value = self._extract_metric_value(trend_metrics[trend_type], metric)
                    if value is not None and not np.isnan(value) and not np.isinf(value):
                        # 正規化スコアの計算（0-100の範囲）
                        normalized_score = self._normalize_metric_score(metric, value)
                        score += normalized_score * weight
                        total_weight += weight
                
                if total_weight > 0:
                    trend_scores[trend_type] = score / total_weight
        
        # ランキングの作成
        sorted_trends = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)
        
        ranking = {
            "scores": trend_scores,
            "ranking_order": [trend for trend, _ in sorted_trends],
            "best_trend": sorted_trends[0][0] if sorted_trends else None,
            "worst_trend": sorted_trends[-1][0] if sorted_trends else None
        }
        
        return ranking
    
    def _normalize_metric_score(self, metric_name: str, value: float) -> float:
        """指標値を0-100の範囲に正規化"""
        # 指標別の正規化ロジック
        if metric_name == "total_profit":
            return max(0, min(100, (value + 1000) / 20))  # 粗い正規化
        elif metric_name in ["sharpe_ratio", "sortino_ratio"]:
            return max(0, min(100, (value + 2) * 25))  # -2から2の範囲を0-100に
        elif metric_name == "win_rate":
            return min(100, max(0, value))  # 既に0-100の範囲
        elif metric_name == "consistency_score":
            return max(0, min(100, value * 10))  # 粗い正規化
        elif metric_name == "profit_factor":
            return max(0, min(100, (value - 0.5) * 20))  # 0.5-5の範囲を0-100に
        else:
            return 50  # デフォルト値
    
    def _calculate_overall_summary(self, trend_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """全体サマリーの計算"""
        summary = {
            "total_periods": 0,
            "total_trading_days": 0,
            "trend_distribution": {},
            "aggregated_metrics": {},
            "key_insights": []
        }
        
        # 期間数と取引日数の集計
        for trend_type, metrics in trend_metrics.items():
            if trend_type in ["uptrend", "downtrend", "sideways"]:
                summary["total_periods"] += metrics.get("period_count", 0)
                summary["total_trading_days"] += metrics.get("total_trading_days", 0)
                
                summary["trend_distribution"][trend_type] = {
                    "periods": metrics.get("period_count", 0),
                    "days": metrics.get("total_trading_days", 0),
                    "percentage": 0  # 後で計算
                }
        
        # パーセンテージの計算
        total_days = summary["total_trading_days"]
        if total_days > 0:
            for trend_type in summary["trend_distribution"]:
                days = summary["trend_distribution"][trend_type]["days"]
                summary["trend_distribution"][trend_type]["percentage"] = (days / total_days) * 100
        
        # 集約指標の計算
        summary["aggregated_metrics"] = self._calculate_aggregated_metrics(trend_metrics)
        
        # 重要な洞察の生成
        summary["key_insights"] = self._generate_key_insights(trend_metrics)
        
        return summary
    
    def _calculate_aggregated_metrics(self, trend_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """集約パフォーマンス指標の計算"""
        all_profits = []
        all_trades = 0
        
        # 全トレンドからデータを集約
        for trend_type, metrics in trend_metrics.items():
            if trend_type in ["uptrend", "downtrend", "sideways"]:
                total_profit = self._extract_metric_value(metrics, "total_profit")
                total_trades = self._extract_metric_value(metrics, "total_trades")
                
                if total_profit is not None:
                    all_profits.append(total_profit)
                if total_trades is not None:
                    all_trades += int(total_trades)
        
        aggregated = {
            "combined_total_profit": sum(all_profits),
            "combined_total_trades": all_trades,
            "combined_average_profit": sum(all_profits) / len(all_profits) if all_profits else 0,
            "profit_variance_across_trends": np.var(all_profits) if all_profits else 0
        }
        
        return aggregated
    
    def _generate_key_insights(self, trend_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
        """重要な洞察を自動生成"""
        insights = []
        
        # 最も収益性の高いトレンド環境の特定
        best_profit_trend = None
        best_profit = float('-inf')
        
        for trend_type, metrics in trend_metrics.items():
            if trend_type in ["uptrend", "downtrend", "sideways"]:
                total_profit = self._extract_metric_value(metrics, "total_profit")
                if total_profit is not None and total_profit > best_profit:
                    best_profit = total_profit
                    best_profit_trend = trend_type
        
        if best_profit_trend:
            insights.append(f"最も収益性が高いトレンド環境: {best_profit_trend} (利益: {best_profit:.2f})")
        
        # Sharpe ratioの分析
        sharpe_ratios = {}
        for trend_type, metrics in trend_metrics.items():
            if trend_type in ["uptrend", "downtrend", "sideways"]:
                sharpe = self._extract_metric_value(metrics, "sharpe_ratio")
                if sharpe is not None:
                    sharpe_ratios[trend_type] = sharpe
        
        if sharpe_ratios:
            best_sharpe_trend = max(sharpe_ratios, key=sharpe_ratios.get)
            insights.append(f"最もリスク調整後リターンが優秀: {best_sharpe_trend} (Sharpe: {sharpe_ratios[best_sharpe_trend]:.3f})")
        
        # 勝率の分析
        win_rates = {}
        for trend_type, metrics in trend_metrics.items():
            if trend_type in ["uptrend", "downtrend", "sideways"]:
                win_rate = self._extract_metric_value(metrics, "win_rate")
                if win_rate is not None:
                    win_rates[trend_type] = win_rate
        
        if win_rates:
            best_win_rate_trend = max(win_rates, key=win_rates.get)
            insights.append(f"最も勝率が高いトレンド環境: {best_win_rate_trend} (勝率: {win_rates[best_win_rate_trend]:.1f}%)")
        
        return insights
    
    def save_performance_analysis(self, 
                                strategy_name: str,
                                filename: Optional[str] = None) -> str:
        """
        パフォーマンス分析結果をJSONファイルに保存
        
        Parameters:
            strategy_name (str): 戦略名
            filename (str, optional): ファイル名（Noneの場合は自動生成）
            
        Returns:
            str: 保存されたファイルパス
        """
        if strategy_name not in self.performance_results:
            raise ValueError(f"戦略 '{strategy_name}' の分析結果が見つかりません")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_performance_analysis_{strategy_name}_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # JSON serializable形式に変換
            serializable_results = self._make_json_serializable(
                self.performance_results[strategy_name]
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"トレンド別パフォーマンス分析結果を保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"分析結果の保存中にエラー: {e}")
            raise
    
    def _make_json_serializable(self, data: Any) -> Any:
        """
        データをJSON serializable形式に変換
        """
        if isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (int, float)) and (pd.isna(data) or np.isnan(data)):
            return None
        elif isinstance(data, (int, float)) and np.isinf(data):
            return "infinity" if data > 0 else "-infinity"
        elif isinstance(data, str) and data in ["inf", "-inf", "nan"]:
            return data
        else:
            return data
    
    def generate_performance_report(self, strategy_name: str) -> str:
        """
        パフォーマンス分析のテキストレポートを生成
        
        Parameters:
            strategy_name (str): 戦略名
            
        Returns:
            str: レポートテキスト
        """
        if strategy_name not in self.performance_results:
            return f"戦略 '{strategy_name}' の分析結果が見つかりません"
        
        results = self.performance_results[strategy_name]
        
        report_lines = [
            f"=== トレンド別パフォーマンス分析レポート ===",
            f"戦略名: {strategy_name}",
            f"分析時刻: {results['calculation_timestamp']}",
            f"無リスク利子率: {results['risk_free_rate']:.2%}",
            f"年間取引日数: {results['trading_days']}日",
            "",
            "=== トレンド別指標 ==="
        ]
        
        # 各トレンドの詳細
        for trend_type, metrics in results["trend_metrics"].items():
            if trend_type in ["uptrend", "downtrend", "sideways"]:
                report_lines.extend([
                    f"\n[{trend_type.upper()}]",
                    f"  期間数: {metrics.get('period_count', 0)}",
                    f"  取引日数: {metrics.get('total_trading_days', 0)}日"
                ])
                
                # 基本指標
                if "basic_metrics" in metrics:
                    basic = metrics["basic_metrics"]
                    report_lines.extend([
                        f"  総取引数: {basic.get('total_trades', 0)}",
                        f"  総利益: {basic.get('total_profit', 0):.2f}",
                        f"  勝率: {basic.get('win_rate', 0):.1f}%",
                        f"  期待値: {basic.get('expectancy', 0):.2f}"
                    ])
                
                # リスク指標
                if "risk_metrics" in metrics:
                    risk = metrics["risk_metrics"]
                    report_lines.extend([
                        f"  Sharpe比: {risk.get('sharpe_ratio', 0):.3f}",
                        f"  Sortino比: {risk.get('sortino_ratio', 0):.3f}",
                        f"  最大DD: {risk.get('max_drawdown_percent', 0):.2f}%"
                    ])
        
        # 比較分析
        if "comparative_analysis" in results:
            comp = results["comparative_analysis"]
            report_lines.extend([
                "\n=== 比較分析 ===",
                f"総合最優秀トレンド: {comp.get('trend_ranking', {}).get('best_trend', 'N/A')}",
                f"総利益最優秀: {comp.get('best_trend_by_metric', {}).get('total_profit', {}).get('trend', 'N/A')}",
                f"Sharpe比最優秀: {comp.get('best_trend_by_metric', {}).get('sharpe_ratio', {}).get('trend', 'N/A')}"
            ])
        
        # 重要な洞察
        if "overall_summary" in results and "key_insights" in results["overall_summary"]:
            insights = results["overall_summary"]["key_insights"]
            if insights:
                report_lines.extend([
                    "\n=== 重要な洞察 ===",
                    *[f"• {insight}" for insight in insights]
                ])
        
        return "\n".join(report_lines)


def run_trend_performance_analysis(backtest_results: Dict[str, Any],
                                 strategy_name: str = "test_strategy",
                                 output_dir: str = "logs",
                                 save_results: bool = True) -> Dict[str, Any]:
    """
    トレンド別パフォーマンス分析を実行する便利関数
    
    Parameters:
        backtest_results (Dict): バックテスト結果
        strategy_name (str): 戦略名
        output_dir (str): 出力ディレクトリ
        save_results (bool): 結果を保存するかどうか
        
    Returns:
        Dict: 分析結果
    """
    # 計算器の初期化
    calculator = TrendPerformanceCalculator(output_dir=output_dir)
    
    # 分析の実行
    performance_analysis = calculator.calculate_trend_performance_metrics(
        backtest_results, strategy_name
    )
    
    # 結果の保存
    if save_results:
        filepath = calculator.save_performance_analysis(strategy_name)
        print(f"分析結果を保存しました: {filepath}")
        
        # レポートの生成と表示
        report = calculator.generate_performance_report(strategy_name)
        print(f"\n{report}")
    
    return performance_analysis


if __name__ == "__main__":
    # テスト用のダミーデータ
    test_backtest_results = {
        "uptrend": {
            "periods": 3,
            "total_days": 150,
            "trades": [
                {"profit": 100, "取引結果": 100},
                {"profit": -30, "取引結果": -30},
                {"profit": 200, "取引結果": 200},
                {"profit": -50, "取引結果": -50},
                {"profit": 150, "取引結果": 150}
            ]
        },
        "downtrend": {
            "periods": 2,
            "total_days": 80,
            "trades": [
                {"profit": -100, "取引結果": -100},
                {"profit": 50, "取引結果": 50},
                {"profit": -200, "取引結果": -200},
                {"profit": 100, "取引結果": 100}
            ]
        },
        "sideways": {
            "periods": 4,
            "total_days": 120,
            "trades": [
                {"profit": 20, "取引結果": 20},
                {"profit": -10, "取引結果": -10},
                {"profit": 30, "取引結果": 30},
                {"profit": -5, "取引結果": -5},
                {"profit": 15, "取引結果": 15}
            ]
        }
    }
    
    # テスト実行
    print("=== トレンド別パフォーマンス分析テスト ===")
    results = run_trend_performance_analysis(
        test_backtest_results,
        strategy_name="test_vwap_strategy",
        output_dir="logs"
    )
