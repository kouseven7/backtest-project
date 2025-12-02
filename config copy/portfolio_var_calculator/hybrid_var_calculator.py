"""
ハイブリッドVaR計算システム

複数のVaR計算手法を統合して精度を向上させるハイブリッド計算機
"""

import os
import sys
import logging
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .advanced_var_engine import VaRCalculationConfig

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridVaRCalculator:
    """ハイブリッドVaR計算機"""
    
    def __init__(self, config: VaRCalculationConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # 手法重み（動的調整用）
        self.dynamic_weights = config.method_weights.copy()
        
        # 精度履歴（手法選択の改善用）
        self.accuracy_history: Dict[str, List[float]] = {
            "parametric": [],
            "historical": [],
            "monte_carlo": []
        }
        
        self.logger.info("HybridVaRCalculator initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.HybridVaRCalculator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def calculate_hybrid_var(self,
                            returns_data: pd.DataFrame,
                            weights: Dict[str, float],
                            confidence_level: float = 0.95) -> Dict[str, float]:
        """ハイブリッドVaR計算"""
        try:
            self.logger.info(f"Calculating hybrid VaR at {confidence_level*100}% confidence level")
            
            # 各手法でVaR計算
            method_results = {}
            
            # パラメトリック手法
            try:
                param_var, param_cvar = self._calculate_parametric_var(
                    returns_data, weights, confidence_level
                )
                method_results['parametric'] = {'var': param_var, 'cvar': param_cvar}
            except Exception as e:
                self.logger.warning(f"Parametric VaR calculation failed: {e}")
                method_results['parametric'] = {'var': 0.05, 'cvar': 0.08}
            
            # ヒストリカル手法
            try:
                hist_var, hist_cvar = self._calculate_historical_var(
                    returns_data, weights, confidence_level
                )
                method_results['historical'] = {'var': hist_var, 'cvar': hist_cvar}
            except Exception as e:
                self.logger.warning(f"Historical VaR calculation failed: {e}")
                method_results['historical'] = {'var': 0.05, 'cvar': 0.08}
            
            # モンテカルロ手法
            try:
                mc_var, mc_cvar = self._calculate_monte_carlo_var(
                    returns_data, weights, confidence_level
                )
                method_results['monte_carlo'] = {'var': mc_var, 'cvar': mc_cvar}
            except Exception as e:
                self.logger.warning(f"Monte Carlo VaR calculation failed: {e}")
                method_results['monte_carlo'] = {'var': 0.05, 'cvar': 0.08}
            
            # 動的重み調整
            adjusted_weights = self._adjust_method_weights(method_results, returns_data)
            
            # 重み付き平均でハイブリッドVaR計算
            hybrid_var = sum(
                method_results[method]['var'] * adjusted_weights.get(method, 0.0)
                for method in method_results.keys()
            )
            
            hybrid_cvar = sum(
                method_results[method]['cvar'] * adjusted_weights.get(method, 0.0)
                for method in method_results.keys()
            )
            
            # 結果の妥当性チェック
            hybrid_var = max(0.001, min(0.5, hybrid_var))  # 0.1%-50%の範囲
            hybrid_cvar = max(hybrid_var, min(0.8, hybrid_cvar))  # VaR以上、80%以下
            
            result = {
                'hybrid_var': hybrid_var,
                'hybrid_cvar': hybrid_cvar,
                'method_results': method_results,
                'method_weights': adjusted_weights,
                'confidence_level': confidence_level,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Hybrid VaR calculation completed: {hybrid_var:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Hybrid VaR calculation error: {e}")
            return {
                'hybrid_var': 0.05,
                'hybrid_cvar': 0.08,
                'method_results': {},
                'method_weights': self.config.method_weights,
                'confidence_level': confidence_level,
                'timestamp': datetime.now().isoformat()
            }
    
    def _adjust_method_weights(self,
                              method_results: Dict[str, Dict[str, float]],
                              returns_data: pd.DataFrame) -> Dict[str, float]:
        """手法重みの動的調整"""
        try:
            # データ品質に基づく調整
            data_quality_score = self._assess_data_quality(returns_data)
            
            # 基本重みから開始
            adjusted_weights = self.dynamic_weights.copy()
            
            # データ品質が低い場合はヒストリカル手法の重みを下げる
            if data_quality_score < 0.5:
                adjusted_weights['historical'] *= 0.7
                adjusted_weights['parametric'] *= 1.2
                adjusted_weights['monte_carlo'] *= 1.1
            
            # データが十分ある場合はヒストリカル手法の重みを上げる
            elif len(returns_data) > 500:
                adjusted_weights['historical'] *= 1.3
                adjusted_weights['parametric'] *= 0.8
            
            # 正規化
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.error(f"Method weights adjustment error: {e}")
            return self.dynamic_weights.copy()
    
    def _assess_data_quality(self, returns_data: pd.DataFrame) -> float:
        """データ品質評価"""
        try:
            if returns_data.empty:
                return 0.0
            
            quality_factors = []
            
            # データ量
            data_length_score = min(1.0, len(returns_data) / 252)  # 1年分を100%とする
            quality_factors.append(data_length_score)
            
            # 欠損値率
            missing_ratio = returns_data.isnull().sum().sum() / (len(returns_data) * len(returns_data.columns))
            missing_score = max(0.0, 1.0 - missing_ratio * 2)  # 50%以上欠損で0点
            quality_factors.append(missing_score)
            
            # 異常値率
            outlier_ratios = []
            for column in returns_data.columns:
                series = returns_data[column].dropna()
                if len(series) > 10:
                    q1, q3 = series.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    if iqr > 0:
                        outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
                        outlier_ratio = len(outliers) / len(series)
                        outlier_ratios.append(outlier_ratio)
            
            if outlier_ratios:
                avg_outlier_ratio = np.mean(outlier_ratios)
                outlier_score = max(0.0, 1.0 - avg_outlier_ratio * 5)  # 20%以上異常値で0点
                quality_factors.append(outlier_score)
            
            # データの安定性（分散の安定性）
            if len(returns_data) > 60:
                rolling_vol = returns_data.rolling(window=20).std().mean(axis=1)
                vol_stability = 1.0 / (1.0 + rolling_vol.std())
                quality_factors.append(vol_stability)
            
            # 総合品質スコア
            overall_quality = np.mean(quality_factors)
            
            return max(0.0, min(1.0, overall_quality))
            
        except Exception as e:
            self.logger.error(f"Data quality assessment error: {e}")
            return 0.5
    
    def _calculate_parametric_var(self,
                                 returns_data: pd.DataFrame,
                                 weights: Dict[str, float],
                                 confidence_level: float) -> Tuple[float, float]:
        """パラメトリックVaR計算"""
        try:
            portfolio_returns = self._calculate_portfolio_returns(returns_data, weights)
            
            if len(portfolio_returns) == 0:
                return 0.05, 0.08
            
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            
            if std_return == 0:
                return 0.0, 0.0
            
            # 正規分布仮定でのVaR
            from scipy import stats
            var_quantile = stats.norm.ppf(1 - confidence_level)
            var_value = abs(mean_return + std_return * var_quantile)
            
            # CVaR（期待ショートフォール）
            phi = stats.norm.pdf(var_quantile)
            cvar_value = abs(mean_return + std_return * phi / (1 - confidence_level))
            
            return var_value, cvar_value
            
        except Exception as e:
            self.logger.error(f"Parametric VaR calculation error: {e}")
            return 0.05, 0.08
    
    def _calculate_historical_var(self,
                                 returns_data: pd.DataFrame,
                                 weights: Dict[str, float],
                                 confidence_level: float) -> Tuple[float, float]:
        """ヒストリカルVaR計算"""
        try:
            portfolio_returns = self._calculate_portfolio_returns(returns_data, weights)
            
            if len(portfolio_returns) == 0:
                return 0.05, 0.08
            
            # VaR計算
            var_value = abs(np.percentile(portfolio_returns, (1 - confidence_level) * 100))
            
            # CVaR計算
            tail_losses = portfolio_returns[portfolio_returns <= -var_value]
            if len(tail_losses) > 0:
                cvar_value = abs(np.mean(tail_losses))
            else:
                cvar_value = var_value * 1.3
            
            return var_value, cvar_value
            
        except Exception as e:
            self.logger.error(f"Historical VaR calculation error: {e}")
            return 0.05, 0.08
    
    def _calculate_monte_carlo_var(self,
                                  returns_data: pd.DataFrame,
                                  weights: Dict[str, float],
                                  confidence_level: float) -> Tuple[float, float]:
        """モンテカルロVaR計算"""
        try:
            portfolio_returns = self._calculate_portfolio_returns(returns_data, weights)
            
            if len(portfolio_returns) == 0:
                return 0.05, 0.08
            
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            
            if std_return == 0:
                return 0.0, 0.0
            
            # モンテカルロシミュレーション
            np.random.seed(self.config.random_seed)
            simulated_returns = np.random.normal(
                mean_return, std_return, self.config.monte_carlo_simulations
            )
            
            # VaR計算
            var_value = abs(np.percentile(simulated_returns, (1 - confidence_level) * 100))
            
            # CVaR計算
            tail_losses = simulated_returns[simulated_returns <= -var_value]
            if len(tail_losses) > 0:
                cvar_value = abs(np.mean(tail_losses))
            else:
                cvar_value = var_value * 1.3
            
            return var_value, cvar_value
            
        except Exception as e:
            self.logger.error(f"Monte Carlo VaR calculation error: {e}")
            return 0.05, 0.08
    
    def _calculate_portfolio_returns(self,
                                   returns_data: pd.DataFrame,
                                   weights: Dict[str, float]) -> np.ndarray:
        """ポートフォリオリターン計算"""
        try:
            if returns_data.empty or not weights:
                return np.array([])
            
            # 重みの正規化
            total_weight = sum(abs(w) for w in weights.values())
            if total_weight == 0:
                return np.array([])
            
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            
            # ポートフォリオリターン計算
            portfolio_returns = np.zeros(len(returns_data))
            
            for strategy, weight in normalized_weights.items():
                if strategy in returns_data.columns:
                    strategy_returns = returns_data[strategy].fillna(0.0).values
                    portfolio_returns += strategy_returns * weight
            
            return portfolio_returns
            
        except Exception as e:
            self.logger.error(f"Portfolio returns calculation error: {e}")
            return np.array([])
    
    def update_method_performance(self,
                                 method: str,
                                 actual_loss: float,
                                 predicted_var: float) -> None:
        """手法のパフォーマンス更新"""
        try:
            if method not in self.accuracy_history:
                return
            
            # 精度計算（予測VaRと実際の損失の比較）
            if predicted_var > 0:
                accuracy = 1.0 - abs(actual_loss - predicted_var) / predicted_var
                accuracy = max(0.0, min(1.0, accuracy))
                
                self.accuracy_history[method].append(accuracy)
                
                # 履歴を最新100件に制限
                if len(self.accuracy_history[method]) > 100:
                    self.accuracy_history[method] = self.accuracy_history[method][-100:]
                
                # 動的重みを更新
                self._update_dynamic_weights()
                
        except Exception as e:
            self.logger.error(f"Method performance update error: {e}")
    
    def _update_dynamic_weights(self) -> None:
        """動的重みの更新"""
        try:
            # 各手法の平均精度を計算
            method_accuracies = {}
            for method, accuracies in self.accuracy_history.items():
                if accuracies:
                    # 最近の精度により大きな重みを付ける
                    weights = np.linspace(0.5, 1.0, len(accuracies))
                    weighted_accuracy = np.average(accuracies, weights=weights)
                    method_accuracies[method] = weighted_accuracy
            
            if not method_accuracies:
                return
            
            # 精度に基づいて動的重みを調整
            total_accuracy = sum(method_accuracies.values())
            if total_accuracy > 0:
                for method in self.dynamic_weights.keys():
                    if method in method_accuracies:
                        # 精度の比例配分
                        accuracy_weight = method_accuracies[method] / total_accuracy
                        # 基本重みと精度重みの加重平均
                        base_weight = self.config.method_weights.get(method, 0.0)
                        self.dynamic_weights[method] = 0.7 * base_weight + 0.3 * accuracy_weight
                
                # 正規化
                total_dynamic = sum(self.dynamic_weights.values())
                if total_dynamic > 0:
                    self.dynamic_weights = {k: v / total_dynamic for k, v in self.dynamic_weights.items()}
            
            self.logger.info(f"Dynamic weights updated: {self.dynamic_weights}")
            
        except Exception as e:
            self.logger.error(f"Dynamic weights update error: {e}")
    
    def get_method_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """手法パフォーマンスサマリー取得"""
        try:
            summary = {}
            
            for method, accuracies in self.accuracy_history.items():
                if accuracies:
                    summary[method] = {
                        'avg_accuracy': np.mean(accuracies),
                        'recent_accuracy': np.mean(accuracies[-10:]) if len(accuracies) >= 10 else np.mean(accuracies),
                        'accuracy_trend': np.mean(accuracies[-5:]) - np.mean(accuracies[-15:-5]) if len(accuracies) >= 15 else 0.0,
                        'sample_count': len(accuracies),
                        'current_weight': self.dynamic_weights.get(method, 0.0)
                    }
                else:
                    summary[method] = {
                        'avg_accuracy': 0.0,
                        'recent_accuracy': 0.0,
                        'accuracy_trend': 0.0,
                        'sample_count': 0,
                        'current_weight': self.dynamic_weights.get(method, 0.0)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Performance summary error: {e}")
            return {}
    
    def reset_performance_history(self) -> None:
        """パフォーマンス履歴のリセット"""
        try:
            self.accuracy_history = {method: [] for method in self.accuracy_history.keys()}
            self.dynamic_weights = self.config.method_weights.copy()
            self.logger.info("Performance history reset")
        except Exception as e:
            self.logger.error(f"Performance history reset error: {e}")
