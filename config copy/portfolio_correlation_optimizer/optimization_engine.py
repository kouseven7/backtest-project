"""
5-3-3 戦略間相関を考慮した配分最適化 - 最適化エンジン

ハイブリッド最適化手法による高度な配分計算システム

Author: imega
Created: 2025-01-27
Task: 5-3-3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from abc import ABC, abstractmethod

# 最適化ライブラリ
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    
try:
    from scipy.optimize import minimize
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.covariance import LedoitWolf
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')

class OptimizationMethod(Enum):
    """最適化手法"""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"

@dataclass
class OptimizationConfig:
    """最適化設定"""
    # 基本パラメータ
    risk_aversion: float = 2.0
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    
    # 制約設定
    min_weight: float = 0.01
    max_weight: float = 0.50
    max_turnover: float = 0.30
    max_concentration: float = 0.60
    
    # 最適化設定
    max_iterations: int = 1000
    tolerance: float = 1e-6
    regularization: float = 1e-8
    
    # 手法別設定
    shrinkage_target: float = 0.0
    clustering_method: str = "single"  # single, complete, average
    n_clusters: Optional[int] = None

@dataclass
class OptimizationResult:
    """最適化結果"""
    weights: Dict[str, float]
    objective_value: float
    risk_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]
    method_metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None

class BaseOptimizer(ABC):
    """基底最適化クラス"""
    
    def __init__(self, config: OptimizationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str],
        current_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """最適化実行"""
        pass
    
    def _validate_inputs(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str]
    ) -> bool:
        """入力検証"""
        if len(expected_returns) != len(strategy_names):
            return False
        if covariance_matrix.shape != (len(strategy_names), len(strategy_names)):
            return False
        if not np.allclose(covariance_matrix, covariance_matrix.T):
            return False
        return True
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """制約適用"""
        # 重み制約
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
        
        # 正規化
        weights = weights / np.sum(weights)
        
        return weights

class MeanVarianceOptimizer(BaseOptimizer):
    """平均分散最適化"""
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str],
        current_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """平均分散最適化実行"""
        
        if not self._validate_inputs(expected_returns, covariance_matrix, strategy_names):
            return OptimizationResult(
                weights={name: 0 for name in strategy_names},
                objective_value=np.inf,
                risk_metrics={},
                convergence_info={},
                method_metadata={},
                success=False,
                error_message="Invalid inputs"
            )
        
        try:
            n = len(strategy_names)
            
            if CVXPY_AVAILABLE:
                # CVXPY実装
                return self._optimize_cvxpy(expected_returns, covariance_matrix, strategy_names)
            else:
                # Scipy実装
                return self._optimize_scipy(expected_returns, covariance_matrix, strategy_names)
                
        except Exception as e:
            self.logger.error(f"Mean variance optimization failed: {e}")
            return self._create_fallback_result(strategy_names, str(e))
    
    def _optimize_cvxpy(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str]
    ) -> OptimizationResult:
        """CVXPY実装"""
        
        n = len(strategy_names)
        
        # 変数定義
        w = cp.Variable(n)
        
        # 目的関数（効用関数）
        portfolio_return = expected_returns.T @ w
        portfolio_risk = cp.quad_form(w, covariance_matrix)
        objective = portfolio_return - 0.5 * self.config.risk_aversion * portfolio_risk
        
        # 制約
        constraints = [
            cp.sum(w) == 1,  # 重み合計
            w >= self.config.min_weight,  # 最小重み
            w <= self.config.max_weight,  # 最大重み
        ]
        
        # 集中度制約
        if hasattr(self.config, 'max_concentration'):
            # 上位k戦略の重み制限（近似）
            constraints.append(cp.max(w) <= self.config.max_concentration / 3)
        
        # 問題定義と解法
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status in ["infeasible", "unbounded"]:
                raise ValueError(f"Optimization failed: {problem.status}")
            
            weights = w.value
            if weights is None:
                raise ValueError("No solution found")
                
            weights = self._apply_constraints(weights)
            
            # リスク指標計算
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            
            risk_metrics = {
                'portfolio_variance': float(portfolio_variance),
                'portfolio_volatility': float(np.sqrt(portfolio_variance)),
                'expected_return': float(np.dot(expected_returns, weights)),
                'sharpe_ratio': float(np.dot(expected_returns, weights) / np.sqrt(portfolio_variance)) if portfolio_variance > 0 else 0.0
            }
            
            return OptimizationResult(
                weights=dict(zip(strategy_names, weights)),
                objective_value=float(problem.value),
                risk_metrics=risk_metrics,
                convergence_info={'status': problem.status, 'solver_time': problem.solver_stats.solve_time},
                method_metadata={'method': 'mean_variance_cvxpy', 'risk_aversion': self.config.risk_aversion}
            )
            
        except Exception as e:
            raise ValueError(f"CVXPY optimization failed: {e}")
    
    def _optimize_scipy(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str]
    ) -> OptimizationResult:
        """Scipy実装"""
        
        n = len(strategy_names)
        
        # 目的関数
        def objective(weights):
            portfolio_return = np.dot(expected_returns, weights)
            portfolio_risk = np.dot(weights, np.dot(covariance_matrix, weights))
            return -(portfolio_return - 0.5 * self.config.risk_aversion * portfolio_risk)
        
        # 制約
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # 重み合計
        ]
        
        # 境界
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n)]
        
        # 初期値
        x0 = np.ones(n) / n
        
        # 最適化実行
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        if not result.success:
            raise ValueError(f"Scipy optimization failed: {result.message}")
        
        weights = self._apply_constraints(result.x)
        
        # リスク指標計算
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        
        risk_metrics = {
            'portfolio_variance': float(portfolio_variance),
            'portfolio_volatility': float(np.sqrt(portfolio_variance)),
            'expected_return': float(np.dot(expected_returns, weights)),
            'sharpe_ratio': float(np.dot(expected_returns, weights) / np.sqrt(portfolio_variance)) if portfolio_variance > 0 else 0.0
        }
        
        return OptimizationResult(
            weights=dict(zip(strategy_names, weights)),
            objective_value=float(-result.fun),
            risk_metrics=risk_metrics,
            convergence_info={'success': result.success, 'iterations': result.nit, 'message': result.message},
            method_metadata={'method': 'mean_variance_scipy', 'risk_aversion': self.config.risk_aversion}
        )
    
    def _create_fallback_result(self, strategy_names: List[str], error_message: str) -> OptimizationResult:
        """フォールバック結果作成"""
        
        n = len(strategy_names)
        equal_weight = 1.0 / n
        weights = {name: equal_weight for name in strategy_names}
        
        return OptimizationResult(
            weights=weights,
            objective_value=0.0,
            risk_metrics={'fallback': True},
            convergence_info={},
            method_metadata={'fallback_reason': error_message},
            success=False,
            error_message=error_message
        )

class RiskParityOptimizer(BaseOptimizer):
    """リスクパリティ最適化"""
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str],
        current_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """リスクパリティ最適化実行"""
        
        if not self._validate_inputs(expected_returns, covariance_matrix, strategy_names):
            return self._create_fallback_result(strategy_names, "Invalid inputs")
        
        try:
            if SCIPY_AVAILABLE:
                return self._optimize_scipy_rp(covariance_matrix, strategy_names)
            else:
                return self._optimize_inverse_volatility(covariance_matrix, strategy_names)
                
        except Exception as e:
            self.logger.error(f"Risk parity optimization failed: {e}")
            return self._create_fallback_result(strategy_names, str(e))
    
    def _optimize_scipy_rp(
        self,
        covariance_matrix: np.ndarray,
        strategy_names: List[str]
    ) -> OptimizationResult:
        """Scipy実装リスクパリティ"""
        
        n = len(strategy_names)
        
        def risk_contribution_objective(weights):
            """リスク寄与度の分散を最小化"""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            # 個別リスク寄与度
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # 目標リスク寄与度（等リスク）
            target_contrib = portfolio_vol / n
            
            # 寄与度差の二乗和
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # 制約
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # 境界
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n)]
        
        # 初期値（逆ボラティリティ）
        volatilities = np.sqrt(np.diag(covariance_matrix))
        x0 = (1.0 / volatilities) / np.sum(1.0 / volatilities)
        x0 = np.clip(x0, self.config.min_weight, self.config.max_weight)
        x0 = x0 / np.sum(x0)
        
        # 最適化実行
        result = minimize(
            risk_contribution_objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        if not result.success:
            self.logger.warning(f"Risk parity optimization did not converge: {result.message}")
            # フォールバック：逆ボラティリティ
            weights = x0
        else:
            weights = result.x
        
        weights = self._apply_constraints(weights)
        
        # リスク指標計算
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        risk_contrib = self._calculate_risk_contributions(weights, covariance_matrix)
        
        risk_metrics = {
            'portfolio_variance': float(portfolio_variance),
            'portfolio_volatility': float(np.sqrt(portfolio_variance)),
            'risk_contribution_std': float(np.std(risk_contrib)),
            'max_risk_contribution': float(np.max(risk_contrib)),
            'min_risk_contribution': float(np.min(risk_contrib))
        }
        
        return OptimizationResult(
            weights=dict(zip(strategy_names, weights)),
            objective_value=float(result.fun if result.success else np.std(risk_contrib)),
            risk_metrics=risk_metrics,
            convergence_info={'success': result.success, 'iterations': result.nit if result.success else 0},
            method_metadata={'method': 'risk_parity_scipy'}
        )
    
    def _optimize_inverse_volatility(
        self,
        covariance_matrix: np.ndarray,
        strategy_names: List[str]
    ) -> OptimizationResult:
        """逆ボラティリティ重み付け（フォールバック）"""
        
        volatilities = np.sqrt(np.diag(covariance_matrix))
        weights = 1.0 / volatilities
        weights = weights / np.sum(weights)
        weights = self._apply_constraints(weights)
        
        # リスク指標計算
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        
        risk_metrics = {
            'portfolio_variance': float(portfolio_variance),
            'portfolio_volatility': float(np.sqrt(portfolio_variance)),
            'method': 'inverse_volatility_fallback'
        }
        
        return OptimizationResult(
            weights=dict(zip(strategy_names, weights)),
            objective_value=0.0,
            risk_metrics=risk_metrics,
            convergence_info={'method': 'analytical'},
            method_metadata={'method': 'inverse_volatility'}
        )
    
    def _calculate_risk_contributions(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """リスク寄与度計算"""
        
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        
        if portfolio_vol > 0:
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            return risk_contrib
        else:
            return np.zeros_like(weights)
    
    def _create_fallback_result(self, strategy_names: List[str], error_message: str) -> OptimizationResult:
        """フォールバック結果作成"""
        
        n = len(strategy_names)
        equal_weight = 1.0 / n
        weights = {name: equal_weight for name in strategy_names}
        
        return OptimizationResult(
            weights=weights,
            objective_value=0.0,
            risk_metrics={'fallback': True},
            convergence_info={},
            method_metadata={'fallback_reason': error_message},
            success=False,
            error_message=error_message
        )

class HierarchicalRiskParityOptimizer(BaseOptimizer):
    """階層リスクパリティ最適化"""
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str],
        current_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """階層リスクパリティ最適化実行"""
        
        if not self._validate_inputs(expected_returns, covariance_matrix, strategy_names):
            return self._create_fallback_result(strategy_names, "Invalid inputs")
        
        try:
            # 相関行列計算
            correlation_matrix = self._covariance_to_correlation(covariance_matrix)
            
            if SCIPY_AVAILABLE:
                return self._optimize_hrp_scipy(
                    correlation_matrix, covariance_matrix, strategy_names
                )
            elif SKLEARN_AVAILABLE:
                return self._optimize_hrp_sklearn(
                    correlation_matrix, covariance_matrix, strategy_names
                )
            else:
                return self._optimize_hrp_simple(
                    correlation_matrix, covariance_matrix, strategy_names
                )
                
        except Exception as e:
            self.logger.error(f"Hierarchical risk parity optimization failed: {e}")
            return self._create_fallback_result(strategy_names, str(e))
    
    def _covariance_to_correlation(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """共分散行列から相関行列へ変換"""
        
        volatilities = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = covariance_matrix / np.outer(volatilities, volatilities)
        
        # 数値安定性のため
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return correlation_matrix
    
    def _optimize_hrp_scipy(
        self,
        correlation_matrix: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str]
    ) -> OptimizationResult:
        """Scipy階層クラスタリング実装"""
        
        n = len(strategy_names)
        
        # 距離行列計算
        distance_matrix = np.sqrt((1 - correlation_matrix) / 2)
        np.fill_diagonal(distance_matrix, 0.0)
        
        # 階層クラスタリング
        distance_vector = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(distance_vector, method=self.config.clustering_method)
        
        # クラスタリング結果からの重み計算
        weights = self._calculate_hrp_weights(
            linkage_matrix, covariance_matrix, n
        )
        
        weights = self._apply_constraints(weights)
        
        # リスク指標計算
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        
        # クラスタリング品質指標
        cluster_labels = fcluster(linkage_matrix, n//2, criterion='maxclust')
        silhouette_score = self._calculate_silhouette_score(
            distance_matrix, cluster_labels
        )
        
        risk_metrics = {
            'portfolio_variance': float(portfolio_variance),
            'portfolio_volatility': float(np.sqrt(portfolio_variance)),
            'silhouette_score': float(silhouette_score),
            'n_clusters_used': len(np.unique(cluster_labels))
        }
        
        return OptimizationResult(
            weights=dict(zip(strategy_names, weights)),
            objective_value=float(portfolio_variance),
            risk_metrics=risk_metrics,
            convergence_info={'clustering_method': self.config.clustering_method},
            method_metadata={'method': 'hrp_scipy', 'linkage_method': self.config.clustering_method}
        )
    
    def _optimize_hrp_sklearn(
        self,
        correlation_matrix: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str]
    ) -> OptimizationResult:
        """Sklearn実装"""
        
        n = len(strategy_names)
        n_clusters = self.config.n_clusters or max(2, n // 3)
        
        # K-meansクラスタリング
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # 特徴量として相関行列を使用
        features = correlation_matrix.copy()
        cluster_labels = kmeans.fit_predict(features)
        
        # クラスター内での重み計算
        weights = np.zeros(n)
        
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # クラスター内の共分散行列
            cluster_cov = covariance_matrix[np.ix_(cluster_indices, cluster_indices)]
            
            # クラスター内での逆ボラティリティ重み
            cluster_volatilities = np.sqrt(np.diag(cluster_cov))
            cluster_weights = 1.0 / cluster_volatilities
            cluster_weights = cluster_weights / np.sum(cluster_weights)
            
            # クラスター間での重み配分
            cluster_total_vol = np.sqrt(
                np.dot(cluster_weights, np.dot(cluster_cov, cluster_weights))
            )
            
            weights[cluster_indices] = cluster_weights / cluster_total_vol
        
        # クラスター間重み正規化
        cluster_total_weights = np.array([
            np.sum(weights[cluster_labels == i]) for i in range(n_clusters)
        ])
        cluster_volatilities = np.array([
            np.sqrt(np.sum(weights[cluster_labels == i]) * np.sum(
                weights[cluster_labels == i][:, np.newaxis] * 
                covariance_matrix[np.ix_(cluster_labels == i, cluster_labels == i)] * 
                weights[cluster_labels == i]
            )) if np.sum(cluster_labels == i) > 0 else 1.0
            for i in range(n_clusters)
        ])
        
        cluster_inv_vol_weights = 1.0 / (cluster_volatilities + 1e-8)
        cluster_inv_vol_weights = cluster_inv_vol_weights / np.sum(cluster_inv_vol_weights)
        
        # 最終重み計算
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            if np.sum(cluster_mask) > 0:
                weights[cluster_mask] *= cluster_inv_vol_weights[cluster_id] / cluster_total_weights[cluster_id]
        
        weights = self._apply_constraints(weights)
        
        # リスク指標計算
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        
        risk_metrics = {
            'portfolio_variance': float(portfolio_variance),
            'portfolio_volatility': float(np.sqrt(portfolio_variance)),
            'n_clusters_used': n_clusters,
            'inertia': float(kmeans.inertia_)
        }
        
        return OptimizationResult(
            weights=dict(zip(strategy_names, weights)),
            objective_value=float(portfolio_variance),
            risk_metrics=risk_metrics,
            convergence_info={'kmeans_iterations': kmeans.n_iter_},
            method_metadata={'method': 'hrp_sklearn', 'n_clusters': n_clusters}
        )
    
    def _optimize_hrp_simple(
        self,
        correlation_matrix: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str]
    ) -> OptimizationResult:
        """簡易HRP実装（フォールバック）"""
        
        n = len(strategy_names)
        
        # 固有ベクトルベースの2分割
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        
        # 最大固有ベクトルでの分割
        first_eigenvector = eigenvectors[:, -1]
        median_value = np.median(first_eigenvector)
        
        cluster1_mask = first_eigenvector >= median_value
        cluster2_mask = ~cluster1_mask
        
        # 各クラスター内での逆ボラティリティ重み
        weights = np.zeros(n)
        
        for cluster_mask in [cluster1_mask, cluster2_mask]:
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            cluster_cov = covariance_matrix[np.ix_(cluster_indices, cluster_indices)]
            cluster_volatilities = np.sqrt(np.diag(cluster_cov))
            
            cluster_weights = 1.0 / cluster_volatilities
            cluster_weights = cluster_weights / np.sum(cluster_weights)
            
            weights[cluster_indices] = cluster_weights
        
        # クラスター間重み調整
        cluster1_vol = np.sqrt(np.dot(weights[cluster1_mask], 
                                    np.dot(covariance_matrix[np.ix_(cluster1_mask, cluster1_mask)], 
                                          weights[cluster1_mask]))) if np.sum(cluster1_mask) > 0 else 1.0
        cluster2_vol = np.sqrt(np.dot(weights[cluster2_mask], 
                                    np.dot(covariance_matrix[np.ix_(cluster2_mask, cluster2_mask)], 
                                          weights[cluster2_mask]))) if np.sum(cluster2_mask) > 0 else 1.0
        
        total_inv_vol = (1.0 / cluster1_vol) + (1.0 / cluster2_vol)
        cluster1_weight = (1.0 / cluster1_vol) / total_inv_vol
        cluster2_weight = (1.0 / cluster2_vol) / total_inv_vol
        
        weights[cluster1_mask] *= cluster1_weight
        weights[cluster2_mask] *= cluster2_weight
        
        weights = self._apply_constraints(weights)
        
        # リスク指標計算
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        
        risk_metrics = {
            'portfolio_variance': float(portfolio_variance),
            'portfolio_volatility': float(np.sqrt(portfolio_variance)),
            'cluster1_size': int(np.sum(cluster1_mask)),
            'cluster2_size': int(np.sum(cluster2_mask))
        }
        
        return OptimizationResult(
            weights=dict(zip(strategy_names, weights)),
            objective_value=float(portfolio_variance),
            risk_metrics=risk_metrics,
            convergence_info={'method': 'eigenvalue_bisection'},
            method_metadata={'method': 'hrp_simple'}
        )
    
    def _calculate_hrp_weights(
        self,
        linkage_matrix: np.ndarray,
        covariance_matrix: np.ndarray,
        n: int
    ) -> np.ndarray:
        """HRP重み計算（再帰的）"""
        
        def _recursive_bisection(indices: List[int]) -> np.ndarray:
            """再帰的二分割"""
            
            if len(indices) == 1:
                return np.array([1.0])
            
            if len(indices) == 2:
                # 2資産の場合は逆ボラティリティ
                sub_cov = covariance_matrix[np.ix_(indices, indices)]
                vols = np.sqrt(np.diag(sub_cov))
                weights = 1.0 / vols
                return weights / np.sum(weights)
            
            # クラスター分割点を見つける
            mid_point = len(indices) // 2
            left_indices = indices[:mid_point]
            right_indices = indices[mid_point:]
            
            # 左右クラスターの重み計算
            left_weights = _recursive_bisection(left_indices)
            right_weights = _recursive_bisection(right_indices)
            
            # クラスター間の重み配分
            left_cov = covariance_matrix[np.ix_(left_indices, left_indices)]
            right_cov = covariance_matrix[np.ix_(right_indices, right_indices)]
            
            left_vol = np.sqrt(np.dot(left_weights, np.dot(left_cov, left_weights)))
            right_vol = np.sqrt(np.dot(right_weights, np.dot(right_cov, right_weights)))
            
            total_inv_vol = (1.0 / left_vol) + (1.0 / right_vol)
            left_allocation = (1.0 / left_vol) / total_inv_vol
            right_allocation = (1.0 / right_vol) / total_inv_vol
            
            # 結合
            combined_weights = np.zeros(len(indices))
            combined_weights[:mid_point] = left_weights * left_allocation
            combined_weights[mid_point:] = right_weights * right_allocation
            
            return combined_weights
        
        # 全インデックス
        all_indices = list(range(n))
        return _recursive_bisection(all_indices)
    
    def _calculate_silhouette_score(
        self,
        distance_matrix: np.ndarray,
        cluster_labels: np.ndarray
    ) -> float:
        """簡易シルエット係数計算"""
        
        try:
            n = len(cluster_labels)
            if len(np.unique(cluster_labels)) <= 1:
                return 0.0
            
            silhouette_scores = []
            
            for i in range(n):
                # 同一クラスター内平均距離
                same_cluster_mask = (cluster_labels == cluster_labels[i])
                same_cluster_indices = np.where(same_cluster_mask)[0]
                
                if len(same_cluster_indices) <= 1:
                    a_i = 0.0
                else:
                    a_i = np.mean([distance_matrix[i, j] for j in same_cluster_indices if j != i])
                
                # 他クラスターとの最小平均距離
                other_clusters = np.unique(cluster_labels[cluster_labels != cluster_labels[i]])
                if len(other_clusters) == 0:
                    b_i = 0.0
                else:
                    b_i = min([
                        np.mean([distance_matrix[i, j] for j in range(n) 
                                if cluster_labels[j] == other_cluster])
                        for other_cluster in other_clusters
                    ])
                
                # シルエット係数
                if max(a_i, b_i) > 0:
                    silhouette_scores.append((b_i - a_i) / max(a_i, b_i))
                else:
                    silhouette_scores.append(0.0)
            
            return np.mean(silhouette_scores)
            
        except Exception:
            return 0.0
    
    def _create_fallback_result(self, strategy_names: List[str], error_message: str) -> OptimizationResult:
        """フォールバック結果作成"""
        
        n = len(strategy_names)
        equal_weight = 1.0 / n
        weights = {name: equal_weight for name in strategy_names}
        
        return OptimizationResult(
            weights=weights,
            objective_value=0.0,
            risk_metrics={'fallback': True},
            convergence_info={},
            method_metadata={'fallback_reason': error_message},
            success=False,
            error_message=error_message
        )

class HybridOptimizationEngine:
    """ハイブリッド最適化エンジン"""
    
    def __init__(self, config: OptimizationConfig, logger: Optional[logging.Logger] = None):
        """
        初期化
        
        Args:
            config: 最適化設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 最適化器の初期化
        self.optimizers = {
            OptimizationMethod.MEAN_VARIANCE: MeanVarianceOptimizer(config, logger),
            OptimizationMethod.RISK_PARITY: RiskParityOptimizer(config, logger),
            OptimizationMethod.HIERARCHICAL_RISK_PARITY: HierarchicalRiskParityOptimizer(config, logger)
        }
    
    def optimize_hybrid(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        strategy_names: List[str],
        method_weights: Dict[OptimizationMethod, float],
        current_weights: Optional[np.ndarray] = None
    ) -> Dict[OptimizationMethod, OptimizationResult]:
        """
        ハイブリッド最適化実行
        
        Args:
            expected_returns: 期待リターン
            covariance_matrix: 共分散行列
            strategy_names: 戦略名リスト
            method_weights: 手法別重み
            current_weights: 現在重み（オプション）
            
        Returns:
            手法別最適化結果
        """
        
        results = {}
        
        for method, weight in method_weights.items():
            if weight <= 0:
                continue
                
            if method not in self.optimizers:
                self.logger.warning(f"Optimizer not available for method: {method}")
                continue
            
            try:
                self.logger.info(f"Running optimization for method: {method}")
                
                result = self.optimizers[method].optimize(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    strategy_names=strategy_names,
                    current_weights=current_weights
                )
                
                results[method] = result
                
                if result.success:
                    self.logger.info(f"Method {method} completed successfully")
                else:
                    self.logger.warning(f"Method {method} failed: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Optimization failed for method {method}: {e}")
                
                # フォールバック結果を作成
                results[method] = OptimizationResult(
                    weights={name: 1.0/len(strategy_names) for name in strategy_names},
                    objective_value=0.0,
                    risk_metrics={'fallback': True},
                    convergence_info={},
                    method_metadata={'error': str(e)},
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def combine_results(
        self,
        results: Dict[OptimizationMethod, OptimizationResult],
        method_weights: Dict[OptimizationMethod, float]
    ) -> OptimizationResult:
        """
        結果統合
        
        Args:
            results: 手法別結果
            method_weights: 手法別重み
            
        Returns:
            統合結果
        """
        
        if not results:
            strategy_names = ['strategy_1']  # フォールバック
            return OptimizationResult(
                weights={name: 1.0 for name in strategy_names},
                objective_value=0.0,
                risk_metrics={},
                convergence_info={},
                method_metadata={'error': 'No results to combine'},
                success=False,
                error_message='No results to combine'
            )
        
        try:
            # 重み正規化
            total_method_weight = sum(method_weights.get(method, 0) for method in results.keys())
            if total_method_weight == 0:
                normalized_weights = {method: 1.0/len(results) for method in results.keys()}
            else:
                normalized_weights = {
                    method: method_weights.get(method, 0) / total_method_weight 
                    for method in results.keys()
                }
            
            # 重み統合
            strategy_names = list(next(iter(results.values())).weights.keys())
            combined_weights = {name: 0.0 for name in strategy_names}
            
            combined_objective = 0.0
            combined_risk_metrics = {}
            all_successful = True
            
            for method, result in results.items():
                method_weight = normalized_weights.get(method, 0)
                
                if method_weight > 0:
                    # 戦略重み統合
                    for name in strategy_names:
                        combined_weights[name] += result.weights.get(name, 0) * method_weight
                    
                    # 目的関数値統合
                    combined_objective += result.objective_value * method_weight
                    
                    # 成功フラグ
                    all_successful = all_successful and result.success
            
            # リスク指標は最初の成功結果から取得（簡易版）
            for result in results.values():
                if result.success and result.risk_metrics:
                    combined_risk_metrics = result.risk_metrics.copy()
                    break
            
            return OptimizationResult(
                weights=combined_weights,
                objective_value=combined_objective,
                risk_metrics=combined_risk_metrics,
                convergence_info={
                    'methods_used': list(results.keys()),
                    'method_weights': normalized_weights
                },
                method_metadata={
                    'hybrid_combination': True,
                    'individual_results': {str(method): result.success for method, result in results.items()}
                },
                success=all_successful
            )
            
        except Exception as e:
            self.logger.error(f"Result combination failed: {e}")
            
            # フォールバック
            strategy_names = list(next(iter(results.values())).weights.keys()) if results else ['fallback']
            equal_weight = 1.0 / len(strategy_names)
            
            return OptimizationResult(
                weights={name: equal_weight for name in strategy_names},
                objective_value=0.0,
                risk_metrics={'fallback': True},
                convergence_info={},
                method_metadata={'combination_error': str(e)},
                success=False,
                error_message=f"Result combination failed: {e}"
            )
