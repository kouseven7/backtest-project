"""
DSSMS統合システム - Performance Metrics Calculator
高度な投資指標計算モジュール

Author: AI Assistant
Created: 2025-09-30
Phase: Phase 4 - パフォーマンス指標計算
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import statistics
from collections import defaultdict

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.logger_config import setup_logger


class PerformanceMetricsCalculator:
    """
    高度な投資パフォーマンス指標計算クラス
    
    Responsibilities:
    - Sharpe比率の計算
    - 最大ドローダウンの算出
    - リスク調整後リターンの計算
    - 高度投資指標の提供
    """
    
    def __init__(self, risk_free_rate: float = 0.001, config: Optional[Dict[str, Any]] = None):
        """
        Performance Metrics Calculator初期化
        
        Args:
            risk_free_rate: リスクフリーレート（年率、デフォルト0.1%）
            config: 設定オプション
        """
        self.logger = setup_logger(__name__, level=logging.DEBUG)
        self.risk_free_rate = risk_free_rate
        self.config = config or {}
        
        # パフォーマンス計算設定
        self.calculation_settings: Dict[str, Union[int, float]] = {
            'days_per_year': 252,  # 営業日ベース
            'confidence_level': 0.95,  # VaR信頼水準
            'benchmark_return': 0.03,  # ベンチマークリターン（年率3%）
        }
        performance_settings = self.config.get('performance_settings', {})
        if isinstance(performance_settings, dict):
            for key, value in performance_settings.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    self.calculation_settings[key] = value
        
        self.logger.info("Performance Metrics Calculator initialized")
    
    def calculate_sharpe_ratio(self, returns: List[float], period: str = 'daily') -> Dict[str, float]:
        """
        Sharpe比率を計算
        
        Args:
            returns: リターン系列
            period: 期間種別 ('daily', 'monthly', 'yearly')
            
        Returns:
            Sharpe比率計算結果
        """
        try:
            if not returns or len(returns) < 2:
                return {
                    'sharpe_ratio': 0.0,
                    'excess_return': 0.0,
                    'volatility': 0.0,
                    'annualized_return': 0.0,
                    'annualized_volatility': 0.0
                }
            
            returns_array = np.array(returns)
            
            # 平均リターンの計算
            mean_return = np.mean(returns_array)
            
            # ボラティリティ（標準偏差）の計算
            volatility = np.std(returns_array, ddof=1) if len(returns) > 1 else 0.0
            
            # 期間に応じた年率化係数
            if period == 'daily':
                annualization_factor = np.sqrt(self.calculation_settings['days_per_year'])
                return_annualization = self.calculation_settings['days_per_year']
            elif period == 'monthly':
                annualization_factor = np.sqrt(12)
                return_annualization = 12
            elif period == 'yearly':
                annualization_factor = 1
                return_annualization = 1
            else:
                annualization_factor = np.sqrt(self.calculation_settings['days_per_year'])
                return_annualization = self.calculation_settings['days_per_year']
            
            # 年率化リターンとボラティリティ
            annualized_return = mean_return * return_annualization
            annualized_volatility = volatility * annualization_factor
            
            # 超過リターン（リスクフリーレートとの差）
            excess_return = annualized_return - self.risk_free_rate
            
            # Sharpe比率の計算
            sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0.0
            
            return {
                'sharpe_ratio': float(round(sharpe_ratio, 4)),
                'excess_return': float(round(excess_return, 4)),
                'volatility': float(round(volatility, 4)),
                'annualized_return': float(round(annualized_return, 4)),
                'annualized_volatility': float(round(annualized_volatility, 4))
            }
            
        except Exception as e:
            self.logger.error(f"Sharpe比率計算エラー: {e}")
            return {
                'sharpe_ratio': 0.0,
                'excess_return': 0.0,
                'volatility': 0.0,
                'annualized_return': 0.0,
                'annualized_volatility': 0.0
            }
    
    def calculate_maximum_drawdown(self, portfolio_values: List[float]) -> Dict[str, Any]:
        """
        最大ドローダウンを計算
        
        Args:
            portfolio_values: ポートフォリオ価値の時系列
            
        Returns:
            最大ドローダウン計算結果
        """
        try:
            if not portfolio_values or len(portfolio_values) < 2:
                return {
                    'max_drawdown': 0.0,
                    'max_drawdown_percent': 0.0,
                    'drawdown_duration': 0,
                    'recovery_period': 0,
                    'drawdown_start_date': None,
                    'drawdown_end_date': None,
                    'recovery_date': None
                }
            
            values_array = np.array(portfolio_values)
            
            # 累積最大値を計算
            cumulative_max = np.maximum.accumulate(values_array)
            
            # ドローダウンを計算
            drawdowns = values_array - cumulative_max
            drawdown_percentages = drawdowns / cumulative_max * 100
            
            # 最大ドローダウンを特定
            max_drawdown_idx = np.argmin(drawdowns)
            max_drawdown = abs(drawdowns[max_drawdown_idx])
            max_drawdown_percent = abs(drawdown_percentages[max_drawdown_idx])
            
            # ドローダウン期間の特定
            # 最大ドローダウンのピーク（開始点）を見つける
            drawdown_start_idx = 0
            for i in range(max_drawdown_idx, -1, -1):
                if values_array[i] >= cumulative_max[max_drawdown_idx]:
                    drawdown_start_idx = i
                    break
            
            # 回復点を見つける
            recovery_idx = len(values_array) - 1
            target_recovery_value = cumulative_max[max_drawdown_idx]
            
            for i in range(max_drawdown_idx + 1, len(values_array)):
                if values_array[i] >= target_recovery_value:
                    recovery_idx = i
                    break
            
            # 期間計算
            drawdown_duration = max_drawdown_idx - drawdown_start_idx
            recovery_period = recovery_idx - max_drawdown_idx if recovery_idx < len(values_array) - 1 else -1
            
            return {
                'max_drawdown': round(max_drawdown, 2),
                'max_drawdown_percent': round(max_drawdown_percent, 2),
                'drawdown_duration': drawdown_duration,
                'recovery_period': recovery_period if recovery_period >= 0 else None,
                'drawdown_start_index': drawdown_start_idx,
                'drawdown_end_index': max_drawdown_idx,
                'recovery_index': recovery_idx if recovery_period >= 0 else None
            }
            
        except Exception as e:
            self.logger.error(f"最大ドローダウン計算エラー: {e}")
            return {
                'max_drawdown': 0.0,
                'max_drawdown_percent': 0.0,
                'drawdown_duration': 0,
                'recovery_period': None,
                'drawdown_start_index': None,
                'drawdown_end_index': None,
                'recovery_index': None
            }
    
    def calculate_risk_adjusted_returns(self, returns: List[float], 
                                      downside_returns: Optional[List[float]] = None) -> Dict[str, float]:
        """
        リスク調整後リターンを計算
        
        Args:
            returns: リターン系列
            downside_returns: 下方偏差計算用のリターン（省略時はreturnsを使用）
            
        Returns:
            リスク調整後リターン指標
        """
        try:
            if not returns or len(returns) < 2:
                return {
                    'calmar_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'var_95': 0.0,
                    'cvar_95': 0.0,
                    'downside_deviation': 0.0
                }
            
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            
            # 年率化リターン
            annualized_return = mean_return * self.calculation_settings['days_per_year']
            
            # 下方偏差の計算（Sortino比率用）
            if downside_returns is None:
                downside_returns = returns
            
            downside_array = np.array(downside_returns)
            negative_returns = downside_array[downside_array < 0]
            downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
            annualized_downside_deviation = downside_deviation * np.sqrt(self.calculation_settings['days_per_year'])
            
            # Sortino比率
            sortino_ratio = (annualized_return - self.risk_free_rate) / annualized_downside_deviation if annualized_downside_deviation > 0 else 0.0
            
            # VaR (Value at Risk) 95%
            var_95 = np.percentile(returns_array, (1 - self.calculation_settings['confidence_level']) * 100)
            
            # CVaR (Conditional Value at Risk) 95%
            var_threshold = var_95
            tail_losses = returns_array[returns_array <= var_threshold]
            cvar_95 = np.mean(tail_losses) if len(tail_losses) > 0 else var_95
            
            # Calmar比率の計算にはMax Drawdownが必要
            # 簡易版として、最悪月間リターンを使用
            worst_return = np.min(returns_array)
            calmar_ratio = annualized_return / abs(worst_return) if worst_return < 0 else annualized_return
            
            return {
                'calmar_ratio': float(round(calmar_ratio, 4)),
                'sortino_ratio': float(round(sortino_ratio, 4)),
                'var_95': float(round(var_95, 4)),
                'cvar_95': float(round(cvar_95, 4)),
                'downside_deviation': float(round(downside_deviation, 4))
            }
            
        except Exception as e:
            self.logger.error(f"リスク調整後リターン計算エラー: {e}")
            return {
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'downside_deviation': 0.0
            }
    
    def calculate_advanced_metrics(self, returns: List[float], 
                                 benchmark_returns: Optional[List[float]] = None) -> Dict[str, float]:
        """
        高度投資指標を計算
        
        Args:
            returns: ポートフォリオリターン系列
            benchmark_returns: ベンチマークリターン系列
            
        Returns:
            高度投資指標
        """
        try:
            if not returns or len(returns) < 2:
                return {
                    'information_ratio': 0.0,
                    'treynor_ratio': 0.0,
                    'beta': 0.0,
                    'alpha': 0.0,
                    'correlation_with_benchmark': 0.0,
                    'tracking_error': 0.0
                }
            
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            annualized_return = mean_return * self.calculation_settings['days_per_year']
            
            # ベンチマークデータがない場合は仮想ベンチマークを使用
            if benchmark_returns is None or len(benchmark_returns) != len(returns):
                # 年率3%のベンチマークリターンを仮定
                daily_benchmark_return = self.calculation_settings['benchmark_return'] / self.calculation_settings['days_per_year']
                benchmark_returns = [daily_benchmark_return] * len(returns)
            
            benchmark_array = np.array(benchmark_returns)
            benchmark_mean = np.mean(benchmark_array)
            
            # 超過リターン
            excess_returns = returns_array - benchmark_array
            excess_return_mean = np.mean(excess_returns)
            
            # トラッキングエラー
            tracking_error = np.std(excess_returns, ddof=1) if len(excess_returns) > 1 else 0.0
            annualized_tracking_error = tracking_error * np.sqrt(self.calculation_settings['days_per_year'])
            
            # [DEBUG] Information Ratio計算の詳細ログ
            self.logger.debug(f"[IR_DEBUG] excess_return_mean: {excess_return_mean}")
            self.logger.debug(f"[IR_DEBUG] tracking_error (daily): {tracking_error}")
            self.logger.debug(f"[IR_DEBUG] annualized_tracking_error: {annualized_tracking_error}")
            self.logger.debug(f"[IR_DEBUG] excess_returns count: {len(excess_returns)}")
            self.logger.debug(f"[IR_DEBUG] excess_returns sample: {excess_returns[:5] if len(excess_returns) >= 5 else excess_returns}")
            
            # 情報比率 (Information Ratio)
            # 極小値チェックを1e-10に変更（浮動小数点誤差対策）
            information_ratio = excess_return_mean / tracking_error if tracking_error > 1e-10 else 0.0
            
            self.logger.debug(f"[IR_DEBUG] information_ratio (BEFORE rounding): {information_ratio}")
            
            # ベータとアルファの計算
            correlation = np.corrcoef(returns_array, benchmark_array)[0, 1] if len(returns) > 1 else 0.0
            
            if not np.isnan(correlation):
                returns_std = np.std(returns_array, ddof=1)
                benchmark_std = np.std(benchmark_array, ddof=1)
                beta = correlation * (returns_std / benchmark_std) if benchmark_std > 0 else 0.0
                
                # アルファ (Jensen's Alpha)
                alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_mean * self.calculation_settings['days_per_year'] - self.risk_free_rate))
                
                # Treynor比率
                treynor_ratio = (annualized_return - self.risk_free_rate) / beta if beta > 0 else 0.0
            else:
                beta = 0.0
                alpha = 0.0
                treynor_ratio = 0.0
                correlation = 0.0
            
            return {
                'information_ratio': float(round(information_ratio, 4)),
                'treynor_ratio': float(round(treynor_ratio, 4)),
                'beta': float(round(beta, 4)),
                'alpha': float(round(alpha, 4)),
                'correlation_with_benchmark': float(round(correlation, 4)),
                'tracking_error': float(round(annualized_tracking_error, 4))
            }
            
        except Exception as e:
            self.logger.error(f"高度投資指標計算エラー: {e}")
            return {
                'information_ratio': 0.0,
                'treynor_ratio': 0.0,
                'beta': 0.0,
                'alpha': 0.0,
                'correlation_with_benchmark': 0.0,
                'tracking_error': 0.0
            }
    
    def generate_comprehensive_metrics(self, portfolio_values: List[float], 
                                     returns: Optional[List[float]] = None,
                                     benchmark_returns: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        包括的なパフォーマンス指標を生成
        
        Args:
            portfolio_values: ポートフォリオ価値の時系列
            returns: リターン系列（省略時はportfolio_valuesから計算）
            benchmark_returns: ベンチマークリターン系列
            
        Returns:
            包括的パフォーマンス指標
        """
        try:
            if not portfolio_values or len(portfolio_values) < 2:
                return {
                    'status': 'insufficient_data',
                    'message': 'データが不足しています',
                    'metrics': {}
                }
            
            # リターン系列の計算（必要な場合）
            if returns is None:
                returns = []
                for i in range(1, len(portfolio_values)):
                    if portfolio_values[i-1] > 0:
                        daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                        returns.append(daily_return)
                    else:
                        returns.append(0.0)
            
            if len(returns) < 2:
                return {
                    'status': 'calculation_error',
                    'message': 'リターン計算に失敗しました',
                    'metrics': {}
                }
            
            # 各指標の計算
            sharpe_metrics = self.calculate_sharpe_ratio(returns, 'daily')
            drawdown_metrics = self.calculate_maximum_drawdown(portfolio_values)
            risk_adjusted_metrics = self.calculate_risk_adjusted_returns(returns)
            advanced_metrics = self.calculate_advanced_metrics(returns, benchmark_returns)
            
            # 総合スコアの計算
            performance_score = self._calculate_performance_score(
                sharpe_metrics, drawdown_metrics, risk_adjusted_metrics, advanced_metrics
            )
            
            return {
                'status': 'success',
                'message': '全指標の計算が完了しました',
                'calculation_date': datetime.now().isoformat(),
                'metrics': {
                    'sharpe_analysis': sharpe_metrics,
                    'drawdown_analysis': drawdown_metrics,
                    'risk_adjusted_metrics': risk_adjusted_metrics,
                    'advanced_metrics': advanced_metrics,
                    'performance_score': performance_score
                },
                'summary': {
                    'total_return_percent': round(((portfolio_values[-1] / portfolio_values[0]) - 1) * 100, 2),
                    'sharpe_ratio': sharpe_metrics['sharpe_ratio'],
                    'max_drawdown_percent': drawdown_metrics['max_drawdown_percent'],
                    'information_ratio': advanced_metrics['information_ratio'],
                    'overall_score': performance_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"包括的指標生成エラー: {e}")
            return {
                'status': 'error',
                'message': f'計算エラー: {str(e)}',
                'metrics': {}
            }
    
    def _calculate_performance_score(self, sharpe_metrics: Dict[str, float], drawdown_metrics: Dict[str, Any],
                                   risk_adjusted_metrics: Dict[str, float], advanced_metrics: Dict[str, float]) -> float:
        """
        総合パフォーマンススコアを計算
        
        Args:
            sharpe_metrics: Sharpe比率関連指標
            drawdown_metrics: ドローダウン関連指標
            risk_adjusted_metrics: リスク調整指標
            advanced_metrics: 高度指標
            
        Returns:
            総合スコア（0-100）
        """
        try:
            # 各指標の重み
            weights = {
                'sharpe_ratio': 0.3,
                'max_drawdown': 0.2,
                'sortino_ratio': 0.2,
                'information_ratio': 0.2,
                'alpha': 0.1
            }
            
            # 各指標の正規化スコア計算
            sharpe_score = min(100.0, max(0.0, (float(sharpe_metrics['sharpe_ratio']) + 2) * 25))  # -2~2を0~100に正規化
            drawdown_score = max(0.0, 100.0 - float(drawdown_metrics.get('max_drawdown_percent', 0)))  # ドローダウンは小さいほど良い
            sortino_score = min(100.0, max(0.0, (float(risk_adjusted_metrics['sortino_ratio']) + 2) * 25))
            info_ratio_score = min(100.0, max(0.0, (float(advanced_metrics['information_ratio']) + 1) * 50))
            alpha_score = min(100.0, max(0.0, (float(advanced_metrics['alpha']) + 0.1) * 500))  # アルファは小さい値なので拡大
            
            # 重み付き平均の計算
            total_score = (
                sharpe_score * weights['sharpe_ratio'] +
                drawdown_score * weights['max_drawdown'] +
                sortino_score * weights['sortino_ratio'] +
                info_ratio_score * weights['information_ratio'] +
                alpha_score * weights['alpha']
            )
            
            return round(total_score, 2)
            
        except Exception as e:
            self.logger.warning(f"パフォーマンススコア計算エラー: {e}")
            return 50.0  # デフォルトスコア


def main():
    """テスト用メイン関数"""
    calculator = PerformanceMetricsCalculator()
    
    # サンプルデータでテスト
    sample_values = [1000000.0, 1020000.0, 1015000.0, 1030000.0, 1025000.0, 1040000.0, 1050000.0]
    result = calculator.generate_comprehensive_metrics(sample_values)
    
    print("Performance Metrics Test Result:")
    print(f"Status: {result['status']}")
    print(f"Total Return: {result['summary']['total_return_percent']}%")
    print(f"Sharpe Ratio: {result['summary']['sharpe_ratio']}")
    print(f"Overall Score: {result['summary']['overall_score']}")


if __name__ == "__main__":
    main()