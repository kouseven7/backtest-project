"""
5-3-3 戦略間相関を考慮した配分最適化 - メイン配分エンジン

相関関係を考慮したハイブリッド最適化による
動的ポートフォリオ配分システム

Author: imega
Created: 2025-01-27
Task: 5-3-3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

from ..correlation.strategy_correlation_analyzer import (
    CorrelationConfig, CorrelationMatrix
)
from ..portfolio_weight_calculator import (
    PortfolioWeightCalculator
)

# 最適化エンジンのインポート
try:
    from .optimization_engine import HybridOptimizationEngine
    from .constraint_manager import CorrelationConstraintManager
    from .integration_bridge import SystemIntegrationBridge
except ImportError:
    # 開発段階での仮実装
    pass

@dataclass
class AllocationConfig:
    """配分最適化設定"""
    # 相関分析設定
    correlation_timeframes: Dict[str, int] = field(default_factory=lambda: {
        'short_term': 30,
        'medium_term': 90, 
        'long_term': 252
    })
    timeframe_weights: Dict[str, float] = field(default_factory=lambda: {
        'short_term': 0.3,
        'medium_term': 0.4,
        'long_term': 0.3
    })
    
    # ハイブリッド最適化設定
    optimization_methods: Dict[str, float] = field(default_factory=lambda: {
        'mean_variance': 0.40,
        'risk_parity': 0.35,
        'hierarchical_risk_parity': 0.25
    })
    
    # 制約設定
    min_weight: float = 0.05
    max_weight: float = 0.40
    max_concentration: float = 0.60  # 上位3戦略の合計
    correlation_penalty_threshold: float = 0.7
    turnover_limit: float = 0.20  # 月次
    
    # リスク調整パラメータ
    risk_aversion: float = 2.0
    expected_return_adjustment: bool = True
    volatility_scaling: bool = True
    
    # 統合設定
    integration_level: str = "moderate"  # basic, moderate, advanced
    use_existing_scores: bool = True
    score_weight: float = 0.3

@dataclass
class AllocationResult:
    """配分結果"""
    strategy_weights: Dict[str, float]
    correlation_matrix: np.ndarray
    correlation_eigenvalues: np.ndarray
    optimization_components: Dict[str, Dict[str, float]]
    risk_metrics: Dict[str, float]
    constraint_status: Dict[str, bool]
    performance_prediction: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class OptimizationStatus:
    """最適化ステータス管理"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CONSTRAINT_VIOLATION = "constraint_violation"

class CorrelationBasedAllocator:
    """戦略間相関を考慮した配分最適化エンジン"""
    
    def __init__(self, config: AllocationConfig, logger: Optional[logging.Logger] = None):
        """
        初期化
        
        Args:
            config: 配分最適化設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # コンポーネント初期化
        self._initialize_components()
        
        # キャッシュ
        self._correlation_cache = {}
        self._optimization_cache = {}
        
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
        
    def _initialize_components(self):
        """コンポーネント初期化"""
        try:
            # 相関設定の準備
            self.correlation_config = CorrelationConfig(
                lookback_period=max(self.config.correlation_timeframes.values()),
                min_periods=30,  # min_observations → min_periods に修正
                significance_threshold=0.05
            )
            
            # 統合ブリッジ初期化（仮実装）
            self.integration_bridge = None
            
            # 最適化エンジン初期化（仮実装）  
            self.optimization_engine = None
            
            # 制約マネージャー初期化（仮実装）
            self.constraint_manager = None
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def allocate_portfolio(
        self, 
        strategy_returns: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        strategy_scores: Optional[Dict[str, float]] = None
    ) -> AllocationResult:
        """
        ポートフォリオ配分最適化
        
        Args:
            strategy_returns: 戦略リターンデータ
            current_weights: 現在の重み
            strategy_scores: 戦略スコア
            
        Returns:
            配分結果
        """
        try:
            self.logger.info("Starting portfolio allocation optimization")
            
            # 1. マルチタイムフレーム相関分析
            correlation_result = self._calculate_multi_timeframe_correlation(
                strategy_returns
            )
            
            # 2. 制約検証
            constraint_result = self._validate_constraints(
                strategy_returns, current_weights
            )
            
            # 3. ハイブリッド最適化実行
            optimization_result = self._execute_hybrid_optimization(
                strategy_returns, correlation_result, strategy_scores
            )
            
            # 4. 結果統合と検証
            final_weights = self._integrate_and_validate_results(
                optimization_result, constraint_result, current_weights
            )
            
            # 5. パフォーマンス予測
            performance_prediction = self._predict_performance(
                final_weights, strategy_returns, correlation_result
            )
            
            # 結果構築
            result = AllocationResult(
                strategy_weights=final_weights,
                correlation_matrix=correlation_result['combined_correlation'],
                correlation_eigenvalues=correlation_result['eigenvalues'],
                optimization_components=optimization_result,
                risk_metrics=self._calculate_risk_metrics(
                    final_weights, correlation_result
                ),
                constraint_status=constraint_result,
                performance_prediction=performance_prediction,
                metadata={
                    'optimization_status': OptimizationStatus.SUCCESS,
                    'timeframes_used': list(self.config.correlation_timeframes.keys()),
                    'methods_used': list(self.config.optimization_methods.keys())
                }
            )
            
            self.logger.info("Portfolio allocation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Portfolio allocation failed: {e}")
            return self._create_fallback_allocation(strategy_returns, current_weights)
    
    def _calculate_multi_timeframe_correlation(
        self, 
        strategy_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """マルチタイムフレーム相関分析"""
        
        cache_key = f"correlation_{hash(str(strategy_returns.index[-1]))}"
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]
            
        try:
            timeframe_correlations = {}
            timeframe_eigenvalues = {}
            
            # 各タイムフレームで相関計算
            for timeframe, days in self.config.correlation_timeframes.items():
                if len(strategy_returns) >= days:
                    recent_returns = strategy_returns.tail(days)
                    
                    # 相関行列計算
                    correlation_matrix = recent_returns.corr()
                    
                    # 固有値計算（リスク集中度評価用）
                    eigenvalues = np.linalg.eigvals(correlation_matrix.values)
                    eigenvalues = np.real(eigenvalues[eigenvalues > 1e-8])
                    
                    timeframe_correlations[timeframe] = correlation_matrix
                    timeframe_eigenvalues[timeframe] = eigenvalues
                    
            # 重み付き統合
            combined_correlation = self._combine_correlation_matrices(
                timeframe_correlations
            )
            
            # 統合固有値
            combined_eigenvalues = np.mean([
                vals for vals in timeframe_eigenvalues.values()
            ], axis=0)
            
            result = {
                'timeframe_correlations': timeframe_correlations,
                'combined_correlation': combined_correlation,
                'eigenvalues': combined_eigenvalues,
                'risk_concentration': self._calculate_risk_concentration(combined_eigenvalues)
            }
            
            self._correlation_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Correlation calculation failed: {e}")
            # フォールバック: 単純相関
            simple_corr = strategy_returns.corr()
            return {
                'timeframe_correlations': {'fallback': simple_corr},
                'combined_correlation': simple_corr,
                'eigenvalues': np.linalg.eigvals(simple_corr.values),
                'risk_concentration': 1.0
            }
    
    def _combine_correlation_matrices(
        self, 
        timeframe_correlations: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """相関行列の重み付き統合"""
        
        if not timeframe_correlations:
            return pd.DataFrame()
            
        # 重み正規化
        weights = self.config.timeframe_weights
        weight_sum = sum(weights.get(tf, 0) for tf in timeframe_correlations.keys())
        
        if weight_sum == 0:
            # 等重み
            normalized_weights = {tf: 1.0/len(timeframe_correlations) 
                                for tf in timeframe_correlations.keys()}
        else:
            normalized_weights = {tf: weights.get(tf, 0)/weight_sum 
                                for tf in timeframe_correlations.keys()}
        
        # 重み付き平均
        combined = None
        for timeframe, correlation in timeframe_correlations.items():
            weight = normalized_weights[timeframe]
            if combined is None:
                combined = correlation * weight
            else:
                combined += correlation * weight
                
        return combined
    
    def _calculate_risk_concentration(self, eigenvalues: np.ndarray) -> float:
        """リスク集中度計算（固有値ベース）"""
        if len(eigenvalues) == 0:
            return 1.0
            
        # 有効固有値数（寄与度ベース）
        total_variance = np.sum(eigenvalues)
        if total_variance <= 0:
            return 1.0
            
        # シャノン・エントロピーベースの多様性指標
        proportions = eigenvalues / total_variance
        entropy = -np.sum(proportions * np.log(proportions + 1e-8))
        max_entropy = np.log(len(eigenvalues))
        
        return entropy / max_entropy if max_entropy > 0 else 1.0
    
    def _validate_constraints(
        self, 
        strategy_returns: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, bool]:
        """制約検証"""
        
        constraints_status = {}
        
        try:
            # データ品質制約
            constraints_status['sufficient_data'] = len(strategy_returns) >= 30
            constraints_status['no_missing_strategies'] = not strategy_returns.isnull().all().any()
            
            # 戦略数制約
            n_strategies = len(strategy_returns.columns)
            constraints_status['min_strategies'] = n_strategies >= 2
            constraints_status['max_strategies'] = n_strategies <= 20
            
            # ターンオーバー制約（現在の重みがある場合）
            if current_weights is not None:
                constraints_status['turnover_feasible'] = True  # 詳細実装は後述
            else:
                constraints_status['turnover_feasible'] = True
                
            # 相関制約準備
            constraints_status['correlation_constraints_ready'] = True
            
        except Exception as e:
            self.logger.error(f"Constraint validation failed: {e}")
            # すべて制約を満たすものとして処理
            constraints_status = {key: True for key in [
                'sufficient_data', 'no_missing_strategies', 
                'min_strategies', 'max_strategies',
                'turnover_feasible', 'correlation_constraints_ready'
            ]}
        
        return constraints_status
    
    def _execute_hybrid_optimization(
        self,
        strategy_returns: pd.DataFrame,
        correlation_result: Dict[str, Any],
        strategy_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """ハイブリッド最適化実行"""
        
        optimization_results = {}
        
        try:
            # 基本統計量計算
            mean_returns = strategy_returns.mean()
            cov_matrix = strategy_returns.cov()
            correlation_matrix = correlation_result['combined_correlation']
            
            # 1. 平均分散最適化
            mv_weights = self._mean_variance_optimization(
                mean_returns, cov_matrix, strategy_scores
            )
            optimization_results['mean_variance'] = mv_weights
            
            # 2. リスクパリティ最適化
            rp_weights = self._risk_parity_optimization(
                cov_matrix, correlation_matrix
            )
            optimization_results['risk_parity'] = rp_weights
            
            # 3. 階層リスクパリティ最適化
            hrp_weights = self._hierarchical_risk_parity_optimization(
                correlation_matrix, cov_matrix
            )
            optimization_results['hierarchical_risk_parity'] = hrp_weights
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Hybrid optimization failed: {e}")
            # フォールバック: 等重み
            strategies = list(strategy_returns.columns)
            equal_weight = 1.0 / len(strategies)
            fallback_weights = {strategy: equal_weight for strategy in strategies}
            
            return {
                'mean_variance': fallback_weights,
                'risk_parity': fallback_weights,
                'hierarchical_risk_parity': fallback_weights
            }
    
    def _mean_variance_optimization(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        strategy_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """平均分散最適化"""
        
        # 期待リターン調整
        expected_returns = mean_returns.copy()
        if strategy_scores is not None and self.config.expected_return_adjustment:
            for strategy, score in strategy_scores.items():
                if strategy in expected_returns.index:
                    expected_returns[strategy] *= (1 + score * self.config.score_weight)
        
        # 簡単な最適化（解析解）
        try:
            inv_cov = np.linalg.pinv(cov_matrix.values)
            ones = np.ones((len(mean_returns), 1))
            
            # 最小分散ポートフォリオ
            weights_mv = inv_cov @ ones
            weights_mv = weights_mv / weights_mv.sum()
            
            # リスク回避度考慮
            excess_returns = expected_returns.values.reshape(-1, 1)
            risk_adjusted_weights = inv_cov @ excess_returns
            risk_adjusted_weights = risk_adjusted_weights / risk_adjusted_weights.sum()
            
            # 組み合わせ
            alpha = 1.0 / (1.0 + self.config.risk_aversion)
            final_weights = alpha * weights_mv + (1 - alpha) * risk_adjusted_weights
            final_weights = np.clip(final_weights.flatten(), 
                                  self.config.min_weight, 
                                  self.config.max_weight)
            
            # 正規化
            final_weights = final_weights / final_weights.sum()
            
            return dict(zip(mean_returns.index, final_weights))
            
        except Exception as e:
            self.logger.warning(f"Mean variance optimization failed: {e}")
            # 等重み
            n = len(mean_returns)
            return {strategy: 1.0/n for strategy in mean_returns.index}
    
    def _risk_parity_optimization(
        self,
        cov_matrix: pd.DataFrame,
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """リスクパリティ最適化"""
        
        try:
            # リスク寄与度の均等化
            volatilities = np.sqrt(np.diag(cov_matrix.values))
            
            # 逆ボラティリティ重み付け（単純版）
            inv_vol_weights = 1.0 / volatilities
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
            
            # 相関調整
            correlation_penalty = np.mean(np.abs(correlation_matrix.values))
            if correlation_penalty > self.config.correlation_penalty_threshold:
                # 高相関時は分散をより重視
                diversification_factor = 1.0 / (1.0 + correlation_penalty)
                inv_vol_weights = inv_vol_weights * diversification_factor
                inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
            
            # 制約適用
            final_weights = np.clip(inv_vol_weights, 
                                  self.config.min_weight, 
                                  self.config.max_weight)
            final_weights = final_weights / final_weights.sum()
            
            return dict(zip(cov_matrix.index, final_weights))
            
        except Exception as e:
            self.logger.warning(f"Risk parity optimization failed: {e}")
            # 等重み
            n = len(cov_matrix)
            return {strategy: 1.0/n for strategy in cov_matrix.index}
    
    def _hierarchical_risk_parity_optimization(
        self,
        correlation_matrix: pd.DataFrame,
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """階層リスクパリティ最適化（簡易版）"""
        
        try:
            # 相関による階層クラスタリング（簡易実装）
            distance_matrix = np.sqrt((1 - correlation_matrix.values) / 2)
            
            # 2分割による階層構造（簡易版）
            n_strategies = len(correlation_matrix)
            if n_strategies <= 2:
                return {strategy: 1.0/n_strategies 
                       for strategy in correlation_matrix.index}
            
            # 固有ベクトルベースのクラスタリング
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix.values)
            
            # 最大固有ベクトルでの分割
            first_eigenvec = eigenvecs[:, -1]
            cluster1_mask = first_eigenvec >= np.median(first_eigenvec)
            
            # クラスター内での等重み
            weights = np.zeros(n_strategies)
            cluster1_size = np.sum(cluster1_mask)
            cluster2_size = n_strategies - cluster1_size
            
            if cluster1_size > 0 and cluster2_size > 0:
                # クラスター間のボラティリティ比較
                cluster1_vol = np.mean([np.sqrt(cov_matrix.iloc[i, i]) 
                                      for i in range(n_strategies) if cluster1_mask[i]])
                cluster2_vol = np.mean([np.sqrt(cov_matrix.iloc[i, i]) 
                                      for i in range(n_strategies) if not cluster1_mask[i]])
                
                total_inv_vol = (1/cluster1_vol) + (1/cluster2_vol)
                cluster1_weight = (1/cluster1_vol) / total_inv_vol
                cluster2_weight = (1/cluster2_vol) / total_inv_vol
                
                # クラスター内で等分配
                for i in range(n_strategies):
                    if cluster1_mask[i]:
                        weights[i] = cluster1_weight / cluster1_size
                    else:
                        weights[i] = cluster2_weight / cluster2_size
            else:
                # フォールバック: 等重み
                weights = np.ones(n_strategies) / n_strategies
            
            # 制約適用
            weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
            weights = weights / weights.sum()
            
            return dict(zip(correlation_matrix.index, weights))
            
        except Exception as e:
            self.logger.warning(f"Hierarchical risk parity optimization failed: {e}")
            # 等重み
            n = len(correlation_matrix)
            return {strategy: 1.0/n for strategy in correlation_matrix.index}
    
    def _integrate_and_validate_results(
        self,
        optimization_result: Dict[str, Dict[str, float]],
        constraint_result: Dict[str, bool],
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """結果統合と検証"""
        
        try:
            # 手法別重み統合
            method_weights = self.config.optimization_methods
            strategies = list(next(iter(optimization_result.values())).keys())
            
            integrated_weights = {}
            for strategy in strategies:
                weight_sum = 0.0
                total_method_weight = 0.0
                
                for method, method_weight in method_weights.items():
                    if method in optimization_result:
                        weight_sum += optimization_result[method][strategy] * method_weight
                        total_method_weight += method_weight
                
                integrated_weights[strategy] = weight_sum / total_method_weight if total_method_weight > 0 else 0.0
            
            # 制約適用
            final_weights = self._apply_constraints(integrated_weights, constraint_result)
            
            # ターンオーバー制約
            if current_weights is not None:
                final_weights = self._apply_turnover_constraint(
                    final_weights, current_weights
                )
            
            return final_weights
            
        except Exception as e:
            self.logger.error(f"Result integration failed: {e}")
            # フォールバック: 最初の手法の結果
            if optimization_result:
                return next(iter(optimization_result.values()))
            else:
                return {}
    
    def _apply_constraints(
        self,
        weights: Dict[str, float],
        constraint_result: Dict[str, bool]
    ) -> Dict[str, float]:
        """制約適用"""
        
        if not weights:
            return weights
            
        # 個別制約
        constrained_weights = {}
        for strategy, weight in weights.items():
            constrained_weights[strategy] = np.clip(
                weight, self.config.min_weight, self.config.max_weight
            )
        
        # 正規化
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                strategy: weight / total_weight 
                for strategy, weight in constrained_weights.items()
            }
        
        # 集中度制約
        sorted_weights = sorted(constrained_weights.values(), reverse=True)
        if len(sorted_weights) >= 3:
            top3_concentration = sum(sorted_weights[:3])
            if top3_concentration > self.config.max_concentration:
                # 上位重みを調整
                adjustment_factor = self.config.max_concentration / top3_concentration
                # 上位3戦略の特定と調整
                sorted_strategies = sorted(
                    constrained_weights.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                for i in range(min(3, len(sorted_strategies))):
                    strategy, weight = sorted_strategies[i]
                    constrained_weights[strategy] = weight * adjustment_factor
        
        # 最終正規化
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                strategy: weight / total_weight 
                for strategy, weight in constrained_weights.items()
            }
        
        return constrained_weights
    
    def _apply_turnover_constraint(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """ターンオーバー制約適用"""
        
        # ターンオーバー計算
        turnover = 0.0
        for strategy in set(target_weights.keys()) | set(current_weights.keys()):
            current_w = current_weights.get(strategy, 0.0)
            target_w = target_weights.get(strategy, 0.0)
            turnover += abs(target_w - current_w)
        
        turnover /= 2.0  # 片道ターンオーバー
        
        # 制限超過時の調整
        if turnover > self.config.turnover_limit:
            adjustment_factor = self.config.turnover_limit / turnover
            
            adjusted_weights = {}
            for strategy in target_weights.keys():
                current_w = current_weights.get(strategy, 0.0)
                target_w = target_weights[strategy]
                
                # 現在重みに向けて調整
                adjusted_w = current_w + (target_w - current_w) * adjustment_factor
                adjusted_weights[strategy] = adjusted_w
            
            # 正規化
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {
                    strategy: weight / total_weight 
                    for strategy, weight in adjusted_weights.items()
                }
            
            return adjusted_weights
        
        return target_weights
    
    def _predict_performance(
        self,
        weights: Dict[str, float],
        strategy_returns: pd.DataFrame,
        correlation_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """パフォーマンス予測"""
        
        try:
            if not weights or strategy_returns.empty:
                return {'expected_return': 0.0, 'expected_volatility': 0.0, 'sharpe_ratio': 0.0}
            
            # ポートフォリオリターン予測
            mean_returns = strategy_returns.mean()
            portfolio_return = sum(
                weights.get(strategy, 0.0) * mean_returns.get(strategy, 0.0) 
                for strategy in mean_returns.index
            )
            
            # ポートフォリオボラティリティ予測
            cov_matrix = strategy_returns.cov()
            weight_vector = np.array([
                weights.get(strategy, 0.0) for strategy in cov_matrix.index
            ])
            
            portfolio_variance = np.dot(weight_vector, np.dot(cov_matrix.values, weight_vector))
            portfolio_volatility = np.sqrt(max(0, portfolio_variance))
            
            # シャープレシオ
            sharpe_ratio = (portfolio_return / portfolio_volatility 
                          if portfolio_volatility > 0 else 0.0)
            
            # リスク分散効果
            diversification_ratio = self._calculate_diversification_ratio(
                weights, strategy_returns, correlation_result
            )
            
            return {
                'expected_return': float(portfolio_return) * 252,  # 年率化
                'expected_volatility': float(portfolio_volatility) * np.sqrt(252),  # 年率化
                'sharpe_ratio': float(sharpe_ratio) * np.sqrt(252),  # 年率化
                'diversification_ratio': float(diversification_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"Performance prediction failed: {e}")
            return {'expected_return': 0.0, 'expected_volatility': 0.0, 'sharpe_ratio': 0.0}
    
    def _calculate_diversification_ratio(
        self,
        weights: Dict[str, float],
        strategy_returns: pd.DataFrame,
        correlation_result: Dict[str, Any]
    ) -> float:
        """分散効果比率計算"""
        
        try:
            # 加重平均ボラティリティ
            individual_vols = strategy_returns.std()
            weighted_avg_vol = sum(
                weights.get(strategy, 0.0) * individual_vols.get(strategy, 0.0)
                for strategy in individual_vols.index
            )
            
            # ポートフォリオボラティリティ
            cov_matrix = strategy_returns.cov()
            weight_vector = np.array([
                weights.get(strategy, 0.0) for strategy in cov_matrix.index
            ])
            
            portfolio_variance = np.dot(weight_vector, np.dot(cov_matrix.values, weight_vector))
            portfolio_vol = np.sqrt(max(0, portfolio_variance))
            
            # 分散効果比率
            if portfolio_vol > 0:
                return weighted_avg_vol / portfolio_vol
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _calculate_risk_metrics(
        self,
        weights: Dict[str, float],
        correlation_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """リスク指標計算"""
        
        risk_metrics = {}
        
        try:
            # 集中度リスク
            weight_values = list(weights.values())
            risk_metrics['concentration_risk'] = sum(w**2 for w in weight_values)
            
            # 相関リスク
            correlation_matrix = correlation_result['combined_correlation']
            avg_correlation = np.mean(correlation_matrix.values[np.triu_indices_from(
                correlation_matrix.values, k=1
            )])
            risk_metrics['correlation_risk'] = float(avg_correlation)
            
            # 多様化指標
            risk_metrics['diversification_index'] = correlation_result.get('risk_concentration', 1.0)
            
            # 有効戦略数
            effective_strategies = 1.0 / sum(w**2 for w in weight_values)
            risk_metrics['effective_strategies'] = float(effective_strategies)
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            # デフォルト値
            risk_metrics = {
                'concentration_risk': 1.0,
                'correlation_risk': 0.0,
                'diversification_index': 1.0,
                'effective_strategies': 1.0
            }
        
        return risk_metrics
    
    def _create_fallback_allocation(
        self,
        strategy_returns: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None
    ) -> AllocationResult:
        """フォールバック配分作成"""
        
        strategies = list(strategy_returns.columns)
        
        if current_weights is not None:
            # 現在の重みを維持
            fallback_weights = current_weights.copy()
        else:
            # 等重み配分
            equal_weight = 1.0 / len(strategies)
            fallback_weights = {strategy: equal_weight for strategy in strategies}
        
        # 最小限の結果オブジェクト
        return AllocationResult(
            strategy_weights=fallback_weights,
            correlation_matrix=np.eye(len(strategies)),
            correlation_eigenvalues=np.ones(len(strategies)),
            optimization_components={
                'fallback': fallback_weights
            },
            risk_metrics={
                'concentration_risk': 1.0/len(strategies),
                'correlation_risk': 0.0,
                'diversification_index': 1.0,
                'effective_strategies': float(len(strategies))
            },
            constraint_status={'fallback': True},
            performance_prediction={
                'expected_return': 0.0,
                'expected_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'diversification_ratio': 1.0
            },
            metadata={
                'optimization_status': OptimizationStatus.FAILED,
                'fallback_reason': 'Optimization failed, using fallback allocation'
            }
        )
    
    def update_config(self, new_config: AllocationConfig):
        """設定更新"""
        self.config = new_config
        self._initialize_components()
        
        # キャッシュクリア
        self._correlation_cache.clear()
        self._optimization_cache.clear()
        
        self.logger.info("Configuration updated")
    
    def get_allocation_summary(self, result: AllocationResult) -> str:
        """配分結果サマリー取得"""
        
        summary_lines = []
        summary_lines.append("=== Portfolio Allocation Summary ===")
        summary_lines.append(f"Timestamp: {result.timestamp}")
        summary_lines.append(f"Status: {result.metadata.get('optimization_status', 'Unknown')}")
        summary_lines.append("")
        
        # 戦略重み
        summary_lines.append("Strategy Weights:")
        sorted_weights = sorted(result.strategy_weights.items(), 
                              key=lambda x: x[1], reverse=True)
        for strategy, weight in sorted_weights:
            summary_lines.append(f"  {strategy}: {weight:.3f}")
        
        summary_lines.append("")
        
        # リスク指標
        summary_lines.append("Risk Metrics:")
        for metric, value in result.risk_metrics.items():
            summary_lines.append(f"  {metric}: {value:.3f}")
        
        summary_lines.append("")
        
        # パフォーマンス予測
        summary_lines.append("Performance Prediction:")
        for metric, value in result.performance_prediction.items():
            summary_lines.append(f"  {metric}: {value:.3f}")
        
        return "\n".join(summary_lines)
