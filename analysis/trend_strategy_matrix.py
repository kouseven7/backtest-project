"""
Module: Trend Strategy Performance Matrix
File: trend_strategy_matrix.py
Description: 
  トレンド環境と戦略のパフォーマンスマトリクスを生成し、
  トレンド別のリスク調整後リターン、適合度スコア、戦略ランキング等の
  高度な分析指標を提供します。Excel、JSON、可視化レポートの出力機能付き。

Author: imega
Created: 2025-07-08
Modified: 2025-07-08

Dependencies:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - openpyxl
  - analysis.trend_limited_backtest
  - analysis.trend_performance_calculator
  - metrics.performance_metrics
"""

import pandas as pd
import numpy as np
import os
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Type, Union, Literal
import logging
import warnings
warnings.filterwarnings('ignore')

# 可視化ライブラリ
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']

# プロジェクトのルートパスを追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# 既存モジュールのインポート
from analysis.trend_limited_backtest import TrendLimitedBacktester
from analysis.trend_performance_calculator import TrendPerformanceCalculator
from metrics.performance_metrics import (
    calculate_sharpe_ratio, calculate_maximum_drawdown, calculate_win_rate,
    calculate_profit_factor, calculate_calmar_ratio, calculate_sortino_ratio
)
from strategies.base_strategy import BaseStrategy
from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__)

class TrendStrategyMatrix:
    """
    トレンド×戦略のパフォーマンスマトリクス生成器
    
    複数の戦略を複数のトレンド環境でテストし、
    - トレンド別リスク調整後リターン
    - トレンド適合度スコア
    - 戦略ランキング
    - 包括的な分析レポート
    を生成します。
    """
    
    def __init__(self, 
                 stock_data: pd.DataFrame,
                 labeled_data: Optional[pd.DataFrame] = None,
                 price_column: str = "Adj Close"):
        """
        初期化
        
        Parameters:
            stock_data (pd.DataFrame): 元の株価データ
            labeled_data (pd.DataFrame, optional): ラベリング済みデータ
            price_column (str): 価格カラム名
        """
        self.stock_data = stock_data.copy()
        self.labeled_data = labeled_data
        self.price_column = price_column
        
        # 内部コンポーネント
        self.trend_backtester = TrendLimitedBacktester(
            stock_data, labeled_data, price_column
        )
        self.trend_calculator = TrendPerformanceCalculator(
            output_dir="logs",
            risk_free_rate=0.0,
            trading_days=252
        )
        
        # 結果保存用
        self.matrix_results: Dict[str, Dict[str, Any]] = {}
        self.strategy_rankings: Dict[str, List[Dict[str, Any]]] = {}
        self.adaptation_metrics: Dict[str, Dict[str, float]] = {}
        
        # 設定
        self.trend_types = ["uptrend", "downtrend", "range-bound"]
        self.reports_dir = "reports"
        self.logs_dir = "logs"
        
        # ディレクトリ作成
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        logger.info(f"TrendStrategyMatrix初期化完了: {len(stock_data)}日間のデータ")
    
    def generate_matrix(self, 
                       strategies: List[Tuple[Type[BaseStrategy], Dict[str, Any]]],
                       min_period_length: int = 10,
                       min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        トレンド×戦略のパフォーマンスマトリクスを生成
        
        Parameters:
            strategies: [(戦略クラス, パラメータ辞書), ...] のリスト
            min_period_length: 最小期間長
            min_confidence: 最小信頼度
            
        Returns:
            Dict: 包括的なマトリクス結果
        """
        logger.info(f"マトリクス生成開始: {len(strategies)}戦略 × {len(self.trend_types)}トレンド")
        
        matrix_data = {}
        detailed_results = {}
        
        # 各戦略・各トレンドでバックテストを実行
        for strategy_class, strategy_params in strategies:
            strategy_name = strategy_class.__name__
            logger.info(f"戦略 {strategy_name} の処理中...")
            
            matrix_data[strategy_name] = {}
            detailed_results[strategy_name] = {}
            
            for trend_type in self.trend_types:
                logger.info(f"  トレンド {trend_type} での実行中...")
                
                try:
                    # トレンド限定バックテスト実行
                    backtest_result = self.trend_backtester.run_strategy_on_trend_periods(
                        strategy_class=strategy_class,
                        strategy_params=strategy_params,
                        trend_type=trend_type,
                        min_period_length=min_period_length,
                        min_confidence=min_confidence
                    )
                    
                    # パフォーマンス指標計算
                    performance_metrics = self._calculate_enhanced_metrics(
                        backtest_result, trend_type, strategy_name
                    )
                    
                    # マトリクス用データ保存
                    matrix_data[strategy_name][trend_type] = performance_metrics
                    detailed_results[strategy_name][trend_type] = backtest_result
                    
                    logger.info(f"    完了: {performance_metrics.get('total_return', 0):.2%} リターン")
                    
                except Exception as e:
                    logger.error(f"  エラー発生 ({trend_type}): {e}")
                    matrix_data[strategy_name][trend_type] = self._create_error_metrics(str(e))
                    detailed_results[strategy_name][trend_type] = {"error": str(e)}
        
        # 高度な分析指標の計算
        self._calculate_trend_adaptation_scores(matrix_data)
        self._generate_strategy_rankings(matrix_data)
        self._calculate_risk_adjusted_performance(matrix_data)
        
        # 結果をまとめる
        comprehensive_results = {
            "matrix_data": matrix_data,
            "detailed_results": detailed_results,
            "strategy_rankings": self.strategy_rankings,
            "adaptation_metrics": self.adaptation_metrics,
            "generation_timestamp": datetime.now().isoformat(),
            "parameters": {
                "min_period_length": min_period_length,
                "min_confidence": min_confidence,
                "strategies_count": len(strategies),
                "trend_types": self.trend_types
            }
        }
        
        self.matrix_results = comprehensive_results
        logger.info("マトリクス生成完了")
        
        return comprehensive_results
    
    def _calculate_enhanced_metrics(self, 
                                   backtest_result: Dict[str, Any], 
                                   trend_type: str, 
                                   strategy_name: str) -> Dict[str, float]:
        """
        拡張されたパフォーマンス指標を計算
        """
        if "error" in backtest_result:
            return self._create_error_metrics(backtest_result["error"])
        
        try:
            # 基本指標
            total_trades = backtest_result.get("total_trades", 0)
            avg_return = backtest_result.get("avg_return_per_trade", 0.0)
            win_rate = backtest_result.get("win_rate", 0.0)
            total_days = backtest_result.get("total_days", 0)
            
            # リターン系列の構築（簡単な推定）
            if total_trades > 0:
                # 個別トレードリターンから日次リターンを推定
                trade_returns = []
                for period in backtest_result.get("period_results", []):
                    if "trades" in period:
                        for trade in period["trades"]:
                            trade_returns.append(trade.get("return_pct", 0.0) / 100.0)
                
                if trade_returns:
                    returns_series = pd.Series(trade_returns)
                    total_return = (1 + returns_series).prod() - 1
                else:
                    total_return = 0.0
            else:
                returns_series = pd.Series([0.0])
                total_return = 0.0
            
            # リスク調整後指標
            if len(returns_series) > 1 and returns_series.std() > 0:
                sharpe_ratio = calculate_sharpe_ratio(returns_series)
                sortino_ratio = calculate_sortino_ratio(returns_series)
                max_drawdown = calculate_maximum_drawdown(returns_series.cumsum())
                
                # Calmar比率（概算）
                if abs(max_drawdown) > 0.001:
                    calmar_ratio = total_return / abs(max_drawdown)
                else:
                    calmar_ratio = 0.0
            else:
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                max_drawdown = 0.0
                calmar_ratio = 0.0
            
            # プロフィットファクター
            if total_trades > 0:
                profit_factor = calculate_profit_factor(returns_series)
            else:
                profit_factor = 0.0
            
            # トレンド適合度スコア（独自指標）
            trend_adaptation_score = self._calculate_trend_adaptation_score(
                win_rate, total_return, sharpe_ratio, trend_type
            )
            
            return {
                "total_return": total_return,
                "avg_return_per_trade": avg_return,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "total_days": total_days,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "max_drawdown": max_drawdown,
                "profit_factor": profit_factor,
                "trend_adaptation_score": trend_adaptation_score,
                "risk_adjusted_return": total_return / max(abs(max_drawdown), 0.01),
                "periods_tested": backtest_result.get("periods_tested", 0)
            }
            
        except Exception as e:
            logger.error(f"指標計算エラー ({strategy_name}-{trend_type}): {e}")
            return self._create_error_metrics(str(e))
    
    def _calculate_trend_adaptation_score(self, 
                                        win_rate: float, 
                                        total_return: float, 
                                        sharpe_ratio: float, 
                                        trend_type: str) -> float:
        """
        トレンド適合度スコアを計算（独自指標）
        
        勝率、リターン、シャープ比率を組み合わせて、
        その戦略がそのトレンド環境にどれだけ適合しているかを評価
        """
        try:
            # トレンドタイプ別の重み
            trend_weights = {
                "uptrend": {"return": 0.4, "winrate": 0.3, "sharpe": 0.3},
                "downtrend": {"return": 0.3, "winrate": 0.4, "sharpe": 0.3},
                "range-bound": {"return": 0.3, "winrate": 0.3, "sharpe": 0.4}
            }
            
            weights = trend_weights.get(trend_type, {"return": 0.33, "winrate": 0.33, "sharpe": 0.34})
            
            # 正規化された値（0-1の範囲）
            normalized_return = min(max(total_return + 0.5, 0), 1)  # -50%から+50%を0-1に
            normalized_winrate = min(max(win_rate, 0), 1)
            normalized_sharpe = min(max((sharpe_ratio + 2) / 4, 0), 1)  # -2から+2を0-1に
            
            # 重み付きスコア
            adaptation_score = (
                normalized_return * weights["return"] +
                normalized_winrate * weights["winrate"] +
                normalized_sharpe * weights["sharpe"]
            )
            
            return min(max(adaptation_score, 0), 1)
            
        except Exception as e:
            logger.warning(f"適合度スコア計算エラー: {e}")
            return 0.0
    
    def _create_error_metrics(self, error_msg: str) -> Dict[str, float]:
        """エラー時のデフォルト指標を作成"""
        return {
            "total_return": 0.0,
            "avg_return_per_trade": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "total_days": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "trend_adaptation_score": 0.0,
            "risk_adjusted_return": 0.0,
            "periods_tested": 0,
            "error": error_msg
        }
    
    def _calculate_trend_adaptation_scores(self, matrix_data: Dict[str, Dict[str, Any]]) -> None:
        """各戦略のトレンド適応性を計算"""
        logger.info("トレンド適応性メトリクス計算中...")
        
        for strategy_name, trend_results in matrix_data.items():
            adaptation_scores = {}
            
            # 各トレンドでの適応スコア
            for trend_type in self.trend_types:
                if trend_type in trend_results and "error" not in trend_results[trend_type]:
                    adaptation_scores[f"{trend_type}_adaptation"] = trend_results[trend_type].get("trend_adaptation_score", 0.0)
                else:
                    adaptation_scores[f"{trend_type}_adaptation"] = 0.0
            
            # 総合適応性（全トレンドの平均）
            valid_scores = [score for score in adaptation_scores.values() if score > 0]
            adaptation_scores["overall_adaptation"] = np.mean(valid_scores) if valid_scores else 0.0
            
            # 適応性の分散（安定性指標）
            adaptation_scores["adaptation_stability"] = 1.0 - np.std(valid_scores) if len(valid_scores) > 1 else 0.0
            
            # 最適トレンド
            best_trend = max(adaptation_scores.items(), key=lambda x: x[1] if "_adaptation" in x[0] else 0)[0].replace("_adaptation", "")
            adaptation_scores["best_trend"] = best_trend
            
            self.adaptation_metrics[strategy_name] = adaptation_scores
    
    def _generate_strategy_rankings(self, matrix_data: Dict[str, Dict[str, Any]]) -> None:
        """戦略ランキングを生成"""
        logger.info("戦略ランキング生成中...")
        
        # トレンド別ランキング
        for trend_type in self.trend_types:
            trend_performances = []
            
            for strategy_name, trend_results in matrix_data.items():
                if trend_type in trend_results and "error" not in trend_results[trend_type]:
                    metrics = trend_results[trend_type]
                    
                    # 複合スコア計算（リターン、勝率、シャープ比率、適応スコアの重み付き平均）
                    composite_score = (
                        metrics.get("total_return", 0) * 0.3 +
                        metrics.get("win_rate", 0) * 0.2 +
                        metrics.get("sharpe_ratio", 0) * 0.25 +
                        metrics.get("trend_adaptation_score", 0) * 0.25
                    )
                    
                    trend_performances.append({
                        "strategy": strategy_name,
                        "composite_score": composite_score,
                        "total_return": metrics.get("total_return", 0),
                        "win_rate": metrics.get("win_rate", 0),
                        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                        "adaptation_score": metrics.get("trend_adaptation_score", 0),
                        "total_trades": metrics.get("total_trades", 0)
                    })
            
            # スコア順にソート
            trend_performances.sort(key=lambda x: x["composite_score"], reverse=True)
            
            # ランクを追加
            for i, perf in enumerate(trend_performances):
                perf["rank"] = i + 1
            
            self.strategy_rankings[trend_type] = trend_performances
        
        # 総合ランキング
        overall_performances = []
        for strategy_name in matrix_data.keys():
            if strategy_name in self.adaptation_metrics:
                overall_score = self.adaptation_metrics[strategy_name].get("overall_adaptation", 0)
                
                # 全トレンドでの平均パフォーマンス
                total_returns = []
                win_rates = []
                sharpe_ratios = []
                
                for trend_type in self.trend_types:
                    if trend_type in matrix_data[strategy_name] and "error" not in matrix_data[strategy_name][trend_type]:
                        metrics = matrix_data[strategy_name][trend_type]
                        total_returns.append(metrics.get("total_return", 0))
                        win_rates.append(metrics.get("win_rate", 0))
                        sharpe_ratios.append(metrics.get("sharpe_ratio", 0))
                
                avg_return = np.mean(total_returns) if total_returns else 0
                avg_winrate = np.mean(win_rates) if win_rates else 0
                avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
                
                overall_performances.append({
                    "strategy": strategy_name,
                    "overall_score": overall_score,
                    "avg_return": avg_return,
                    "avg_winrate": avg_winrate,
                    "avg_sharpe": avg_sharpe,
                    "adaptation_stability": self.adaptation_metrics[strategy_name].get("adaptation_stability", 0),
                    "best_trend": self.adaptation_metrics[strategy_name].get("best_trend", "unknown")
                })
        
        overall_performances.sort(key=lambda x: x["overall_score"], reverse=True)
        for i, perf in enumerate(overall_performances):
            perf["rank"] = i + 1
        
        self.strategy_rankings["overall"] = overall_performances
    
    def _calculate_risk_adjusted_performance(self, matrix_data: Dict[str, Dict[str, Any]]) -> None:
        """リスク調整後パフォーマンスの追加計算"""
        logger.info("リスク調整後パフォーマンス計算中...")
        
        for strategy_name, trend_results in matrix_data.items():
            for trend_type, metrics in trend_results.items():
                if "error" not in metrics:
                    # 情報比率（Information Ratio）の簡易版
                    total_return = metrics.get("total_return", 0)
                    max_drawdown = metrics.get("max_drawdown", 0)
                    
                    if abs(max_drawdown) > 0.001:
                        information_ratio = total_return / abs(max_drawdown)
                    else:
                        information_ratio = 0.0
                    
                    # リスク効率性スコア
                    win_rate = metrics.get("win_rate", 0)
                    sharpe = metrics.get("sharpe_ratio", 0)
                    risk_efficiency = (win_rate * 0.4 + (sharpe + 2) / 4 * 0.6) if sharpe != 0 else win_rate * 0.4
                    
                    # メトリクスを更新
                    metrics["information_ratio"] = information_ratio
                    metrics["risk_efficiency_score"] = risk_efficiency
    
    def save_results(self, filename_prefix: str = "trend_strategy_matrix") -> Dict[str, str]:
        """
        結果をJSON、Excel、可視化ファイルとして保存
        
        Returns:
            Dict: 保存されたファイルパスの辞書
        """
        if not self.matrix_results:
            raise ValueError("マトリクス結果が生成されていません。先にgenerate_matrix()を実行してください。")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        try:
            # 1. JSON保存
            json_filename = f"{filename_prefix}_{timestamp}.json"
            json_path = os.path.join(self.logs_dir, json_filename)
            
            # JSON用にデータを準備（numpy型をPython標準型に変換）
            json_data = self._prepare_json_data(self.matrix_results)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            
            saved_files["json"] = json_path
            logger.info(f"JSONファイル保存完了: {json_path}")
            
            # 2. Excel保存
            excel_filename = f"{filename_prefix}_{timestamp}.xlsx"
            excel_path = os.path.join(self.reports_dir, excel_filename)
            
            self._save_excel_report(excel_path)
            saved_files["excel"] = excel_path
            logger.info(f"Excelファイル保存完了: {excel_path}")
            
            # 3. 可視化保存
            viz_filename = f"{filename_prefix}_{timestamp}_visualization.png"
            viz_path = os.path.join(self.reports_dir, viz_filename)
            
            self._create_visualization(viz_path)
            saved_files["visualization"] = viz_path
            logger.info(f"可視化ファイル保存完了: {viz_path}")
            
            # 4. サマリーレポート
            summary_filename = f"{filename_prefix}_{timestamp}_summary.txt"
            summary_path = os.path.join(self.reports_dir, summary_filename)
            
            self._generate_text_summary(summary_path)
            saved_files["summary"] = summary_path
            logger.info(f"サマリーレポート保存完了: {summary_path}")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"結果保存中にエラー: {e}")
            raise
    
    def _prepare_json_data(self, data: Any) -> Any:
        """JSON保存用にデータを準備（numpy型の変換など）"""
        if isinstance(data, dict):
            return {k: self._prepare_json_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_json_data(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif pd.isna(data):
            return None
        else:
            return data
    
    def _save_excel_report(self, excel_path: str) -> None:
        """Excel形式でレポートを保存"""
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                
                # 1. マトリクスサマリー
                matrix_df = self._create_matrix_dataframe()
                matrix_df.to_excel(writer, sheet_name='Matrix_Summary', index=True)
                
                # 2. 戦略ランキング（トレンド別）
                for trend_type in self.trend_types:
                    if trend_type in self.strategy_rankings:
                        ranking_df = pd.DataFrame(self.strategy_rankings[trend_type])
                        ranking_df.to_excel(writer, sheet_name=f'Ranking_{trend_type}', index=False)
                
                # 3. 総合ランキング
                if "overall" in self.strategy_rankings:
                    overall_df = pd.DataFrame(self.strategy_rankings["overall"])
                    overall_df.to_excel(writer, sheet_name='Overall_Ranking', index=False)
                
                # 4. 適応性メトリクス
                adaptation_df = pd.DataFrame.from_dict(self.adaptation_metrics, orient='index')
                adaptation_df.to_excel(writer, sheet_name='Adaptation_Metrics', index=True)
                
                # 5. 詳細データ
                detailed_df = self._create_detailed_dataframe()
                detailed_df.to_excel(writer, sheet_name='Detailed_Metrics', index=False)
                
        except Exception as e:
            logger.error(f"Excel保存エラー: {e}")
            raise
    
    def _create_matrix_dataframe(self) -> pd.DataFrame:
        """マトリクス用のDataFrameを作成"""
        matrix_data = self.matrix_results.get("matrix_data", {})
        
        # 各戦略・トレンドの主要指標を抽出
        matrix_rows = []
        for strategy_name, trend_results in matrix_data.items():
            row = {"Strategy": strategy_name}
            
            for trend_type in self.trend_types:
                if trend_type in trend_results and "error" not in trend_results[trend_type]:
                    metrics = trend_results[trend_type]
                    
                    # 主要指標のみ選択
                    row[f"{trend_type}_Return"] = metrics.get("total_return", 0)
                    row[f"{trend_type}_WinRate"] = metrics.get("win_rate", 0)
                    row[f"{trend_type}_Sharpe"] = metrics.get("sharpe_ratio", 0)
                    row[f"{trend_type}_Adaptation"] = metrics.get("trend_adaptation_score", 0)
                else:
                    row[f"{trend_type}_Return"] = 0
                    row[f"{trend_type}_WinRate"] = 0
                    row[f"{trend_type}_Sharpe"] = 0
                    row[f"{trend_type}_Adaptation"] = 0
            
            matrix_rows.append(row)
        
        return pd.DataFrame(matrix_rows).set_index("Strategy")
    
    def _create_detailed_dataframe(self) -> pd.DataFrame:
        """詳細指標用のDataFrameを作成"""
        matrix_data = self.matrix_results.get("matrix_data", {})
        
        detailed_rows = []
        for strategy_name, trend_results in matrix_data.items():
            for trend_type in self.trend_types:
                if trend_type in trend_results:
                    metrics = trend_results[trend_type]
                    
                    row = {
                        "Strategy": strategy_name,
                        "Trend_Type": trend_type,
                        **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                    }
                    detailed_rows.append(row)
        
        return pd.DataFrame(detailed_rows)
    
    def _create_visualization(self, viz_path: str) -> None:
        """可視化チャートを作成・保存"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Trend-Strategy Performance Matrix Analysis', fontsize=16, fontweight='bold')
            
            # 1. リターンヒートマップ
            self._plot_return_heatmap(axes[0, 0])
            
            # 2. 適応スコアレーダーチャート
            self._plot_adaptation_radar(axes[0, 1])
            
            # 3. リスク・リターン散布図
            self._plot_risk_return_scatter(axes[1, 0])
            
            # 4. 戦略ランキング棒グラフ
            self._plot_strategy_ranking(axes[1, 1])
            
            plt.tight_layout()
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"可視化作成エラー: {e}")
            # エラー時でも空の図を保存
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Visualization Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_return_heatmap(self, ax) -> None:
        """リターンのヒートマップを描画"""
        matrix_df = self._create_matrix_dataframe()
        
        # リターンカラムのみ抽出
        return_cols = [col for col in matrix_df.columns if "_Return" in col]
        return_data = matrix_df[return_cols]
        
        # カラム名を整理
        return_data.columns = [col.replace("_Return", "") for col in return_data.columns]
        
        # ヒートマップ描画
        sns.heatmap(return_data, annot=True, fmt='.2%', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Total Return'})
        ax.set_title('Total Return by Strategy and Trend', fontweight='bold')
        ax.set_xlabel('Trend Type')
        ax.set_ylabel('Strategy')
    
    def _plot_adaptation_radar(self, ax) -> None:
        """適応スコアのレーダーチャートを描画"""
        if not self.strategy_rankings.get("overall"):
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return
        
        # 上位3戦略を選択
        top_strategies = self.strategy_rankings["overall"][:3]
        
        angles = np.linspace(0, 2 * np.pi, len(self.trend_types), endpoint=False).tolist()
        angles += angles[:1]  # 円を閉じる
        
        for i, strategy_data in enumerate(top_strategies):
            strategy_name = strategy_data["strategy"]
            
            if strategy_name in self.adaptation_metrics:
                values = []
                for trend_type in self.trend_types:
                    score = self.adaptation_metrics[strategy_name].get(f"{trend_type}_adaptation", 0)
                    values.append(score)
                
                values += values[:1]  # 円を閉じる
                
                ax.plot(angles, values, 'o-', linewidth=2, label=strategy_name)
                ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.trend_types)
        ax.set_ylim(0, 1)
        ax.set_title('Trend Adaptation Scores (Top 3 Strategies)', fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        ax.grid(True)
    
    def _plot_risk_return_scatter(self, ax) -> None:
        """リスク・リターン散布図を描画"""
        matrix_data = self.matrix_results.get("matrix_data", {})
        
        for trend_type in self.trend_types:
            returns = []
            risks = []
            labels = []
            
            for strategy_name, trend_results in matrix_data.items():
                if trend_type in trend_results and "error" not in trend_results[trend_type]:
                    metrics = trend_results[trend_type]
                    returns.append(metrics.get("total_return", 0))
                    risks.append(abs(metrics.get("max_drawdown", 0)))
                    labels.append(strategy_name)
            
            if returns and risks:
                ax.scatter(risks, returns, label=trend_type, alpha=0.7, s=60)
                
                # 戦略名を表示（上位のみ）
                for i, (x, y, label) in enumerate(zip(risks, returns, labels)):
                    if i < 3:  # 上位3つのみ
                        ax.annotate(label, (x, y), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Max Drawdown (Risk)')
        ax.set_ylabel('Total Return')
        ax.set_title('Risk-Return Profile by Trend Type', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_strategy_ranking(self, ax) -> None:
        """戦略ランキングの棒グラフを描画"""
        if not self.strategy_rankings.get("overall"):
            ax.text(0.5, 0.5, "No ranking data available", ha='center', va='center')
            return
        
        # 上位10戦略
        top_strategies = self.strategy_rankings["overall"][:10]
        
        strategies = [s["strategy"] for s in top_strategies]
        scores = [s["overall_score"] for s in top_strategies]
        
        bars = ax.barh(strategies, scores, color='skyblue', alpha=0.8)
        
        # スコアを表示
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Overall Adaptation Score')
        ax.set_title('Strategy Ranking (Overall Performance)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Y軸のラベルを短縮
        ax.set_yticklabels([s[:20] + "..." if len(s) > 20 else s for s in strategies])
    
    def _generate_text_summary(self, summary_path: str) -> None:
        """テキスト形式のサマリーレポートを生成"""
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("TREND-STRATEGY PERFORMANCE MATRIX ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 1. 概要
                f.write("1. ANALYSIS OVERVIEW\n")
                f.write("-" * 40 + "\n")
                params = self.matrix_results.get("parameters", {})
                f.write(f"Strategies Analyzed: {params.get('strategies_count', 'N/A')}\n")
                f.write(f"Trend Types: {', '.join(params.get('trend_types', []))}\n")
                f.write(f"Minimum Period Length: {params.get('min_period_length', 'N/A')} days\n")
                f.write(f"Minimum Confidence: {params.get('min_confidence', 'N/A')}\n\n")
                
                # 2. 総合ランキング
                f.write("2. OVERALL STRATEGY RANKING\n")
                f.write("-" * 40 + "\n")
                if "overall" in self.strategy_rankings:
                    for i, strategy in enumerate(self.strategy_rankings["overall"][:10]):
                        f.write(f"{i+1:2d}. {strategy['strategy']:<30} Score: {strategy['overall_score']:.3f}\n")
                        f.write(f"    Best Trend: {strategy['best_trend']:<15} Avg Return: {strategy['avg_return']:6.2%}\n")
                f.write("\n")
                
                # 3. トレンド別ベスト戦略
                f.write("3. BEST STRATEGY BY TREND TYPE\n")
                f.write("-" * 40 + "\n")
                for trend_type in self.trend_types:
                    if trend_type in self.strategy_rankings and self.strategy_rankings[trend_type]:
                        best = self.strategy_rankings[trend_type][0]
                        f.write(f"{trend_type.upper()}:\n")
                        f.write(f"  Strategy: {best['strategy']}\n")
                        f.write(f"  Return: {best['total_return']:6.2%}\n")
                        f.write(f"  Win Rate: {best['win_rate']:6.1%}\n")
                        f.write(f"  Sharpe Ratio: {best['sharpe_ratio']:6.2f}\n")
                        f.write(f"  Adaptation Score: {best['adaptation_score']:6.3f}\n\n")
                
                # 4. 重要な洞察
                f.write("4. KEY INSIGHTS\n")
                f.write("-" * 40 + "\n")
                self._write_insights(f)
                
        except Exception as e:
            logger.error(f"サマリーレポート作成エラー: {e}")
    
    def _write_insights(self, f) -> None:
        """重要な洞察をファイルに書き出し"""
        try:
            # 最も適応性の高い戦略
            if "overall" in self.strategy_rankings and self.strategy_rankings["overall"]:
                best_overall = self.strategy_rankings["overall"][0]
                f.write(f"• Most Adaptable Strategy: {best_overall['strategy']}\n")
                f.write(f"  Overall adaptation score: {best_overall['overall_score']:.3f}\n")
            
            # トレンド毎の平均パフォーマンス
            matrix_data = self.matrix_results.get("matrix_data", {})
            for trend_type in self.trend_types:
                returns = []
                for strategy_name, trend_results in matrix_data.items():
                    if trend_type in trend_results and "error" not in trend_results[trend_type]:
                        returns.append(trend_results[trend_type].get("total_return", 0))
                
                if returns:
                    avg_return = np.mean(returns)
                    f.write(f"• Average {trend_type} return: {avg_return:.2%}\n")
            
            # パフォーマンスの分散
            all_scores = []
            for strategy_name in self.adaptation_metrics:
                score = self.adaptation_metrics[strategy_name].get("overall_adaptation", 0)
                if score > 0:
                    all_scores.append(score)
            
            if all_scores:
                score_std = np.std(all_scores)
                f.write(f"• Strategy performance variability: {score_std:.3f}\n")
                if score_std > 0.2:
                    f.write("  → High variability suggests diverse strategy effectiveness\n")
                else:
                    f.write("  → Low variability suggests similar strategy effectiveness\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
            
        except Exception as e:
            f.write(f"Error generating insights: {e}\n")
    
    def get_best_strategy_for_trend(self, trend_type: str) -> Optional[Dict[str, Any]]:
        """指定されたトレンドタイプに最も適した戦略を返す"""
        if trend_type not in self.strategy_rankings:
            return None
        
        rankings = self.strategy_rankings[trend_type]
        return rankings[0] if rankings else None
    
    def get_strategy_recommendation(self, 
                                   current_trend: str, 
                                   risk_tolerance: str = "medium") -> Dict[str, Any]:
        """
        現在のトレンドとリスク許容度に基づく戦略推奨
        
        Parameters:
            current_trend: 現在のトレンドタイプ
            risk_tolerance: リスク許容度 ("low", "medium", "high")
            
        Returns:
            推奨戦略の詳細情報
        """
        if current_trend not in self.strategy_rankings:
            return {"error": f"Trend type '{current_trend}' not found in rankings"}
        
        candidates = self.strategy_rankings[current_trend]
        if not candidates:
            return {"error": "No suitable strategies found"}
        
        # リスク許容度に基づくフィルタリング
        risk_filters = {
            "low": lambda x: x.get("sharpe_ratio", 0) > 0.5 and abs(x.get("max_drawdown", 0)) < 0.1,
            "medium": lambda x: x.get("sharpe_ratio", 0) > 0.2 and abs(x.get("max_drawdown", 0)) < 0.2,
            "high": lambda x: x.get("total_return", 0) > 0.05  # リターン重視
        }
        
        risk_filter = risk_filters.get(risk_tolerance, risk_filters["medium"])
        
        # 条件に合う戦略を検索
        suitable_strategies = []
        matrix_data = self.matrix_results.get("matrix_data", {})
        
        for candidate in candidates[:5]:  # 上位5つから選択
            strategy_name = candidate["strategy"]
            if strategy_name in matrix_data and current_trend in matrix_data[strategy_name]:
                metrics = matrix_data[strategy_name][current_trend]
                if "error" not in metrics and risk_filter(metrics):
                    suitable_strategies.append({
                        **candidate,
                        "recommendation_reason": f"Suitable for {risk_tolerance} risk tolerance in {current_trend} market",
                        "detailed_metrics": metrics
                    })
        
        if suitable_strategies:
            return {
                "recommended_strategy": suitable_strategies[0],
                "alternatives": suitable_strategies[1:3],
                "trend_type": current_trend,
                "risk_tolerance": risk_tolerance
            }
        else:
            # 条件に合わない場合は最上位を推奨
            return {
                "recommended_strategy": {
                    **candidates[0],
                    "recommendation_reason": f"Best available option for {current_trend} (risk tolerance not perfectly matched)",
                    "detailed_metrics": matrix_data.get(candidates[0]["strategy"], {}).get(current_trend, {})
                },
                "alternatives": [],
                "trend_type": current_trend,
                "risk_tolerance": risk_tolerance,
                "warning": "No strategies perfectly match the risk tolerance criteria"
            }
