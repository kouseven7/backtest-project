"""
Module: Optimization Algorithms
File: optimization_algorithms.py
Description: 
  5-1-3「リスク調整後リターンの最適化」
  最適化アルゴリズムの実装

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from scipy.optimize import differential_evolution, minimize
import warnings

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """最適化手法"""
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    SCIPY_MINIMIZE = "scipy_minimize"
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"

@dataclass
class OptimizationConfig:
    """最適化設定"""
    method: OptimizationMethod
    max_iterations: int = 1000
    population_size: int = 50
    tolerance: float = 1e-6
    bounds: Optional[List[Tuple[float, float]]] = None
    constraints: Optional[List[Dict]] = None
    seed: Optional[int] = None
    parallel: bool = False
    verbose: bool = False

@dataclass
class OptimizationResult:
    """最適化結果"""
    success: bool
    optimal_weights: Dict[str, float]
    optimal_value: float
    iterations: int
    function_evaluations: int
    convergence_message: str
    execution_time: float
    confidence_score: float
    constraint_violations: List[str] = field(default_factory=list)
    optimization_history: List[Dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class OptimizationAlgorithm(ABC):
    """最適化アルゴリズムの抽象基底クラス"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.optimization_history = []
        
    @abstractmethod
    def optimize(
        self, 
        objective_function: Callable,
        initial_weights: Dict[str, float],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        constraints: Optional[List] = None
    ) -> OptimizationResult:
        """最適化を実行"""
        pass
    
    def _weights_dict_to_array(self, weights_dict: Dict[str, float], strategy_names: List[str]) -> np.ndarray:
        """重み辞書を配列に変換"""
        return np.array([weights_dict.get(name, 0.0) for name in strategy_names])
    
    def _weights_array_to_dict(self, weights_array: np.ndarray, strategy_names: List[str]) -> Dict[str, float]:
        """重み配列を辞書に変換"""
        return {name: float(weight) for name, weight in zip(strategy_names, weights_array)}
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """重みを正規化（合計を1にする）"""
        total = weights.sum()
        return weights / total if total > 0 else weights

class DifferentialEvolutionOptimizer(OptimizationAlgorithm):
    """差分進化アルゴリズム最適化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.strategy_names = []
        
    def optimize(
        self,
        objective_function: Callable,
        initial_weights: Dict[str, float],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        constraints: Optional[List] = None
    ) -> OptimizationResult:
        """差分進化による最適化"""
        start_time = datetime.now()
        
        try:
            self.strategy_names = list(initial_weights.keys())
            n_strategies = len(self.strategy_names)
            
            # 境界条件の設定
            if bounds is None:
                bounds_list = [(0.0, 1.0) for _ in range(n_strategies)]
            else:
                bounds_list = [bounds.get(name, (0.0, 1.0)) for name in self.strategy_names]
            
            # 制約付き目的関数の定義
            def constrained_objective(weights_array):
                # 重みを正規化
                weights_array = self._normalize_weights(weights_array)
                weights_dict = self._weights_array_to_dict(weights_array, self.strategy_names)
                
                # 目的関数値を計算
                objective_value = objective_function(weights_dict)
                
                # 履歴に記録
                self.optimization_history.append({
                    'weights': weights_dict.copy(),
                    'objective_value': objective_value,
                    'timestamp': datetime.now()
                })
                
                return -objective_value  # 最小化問題として定式化
            
            # 初期重みを配列に変換
            initial_array = self._weights_dict_to_array(initial_weights, self.strategy_names)
            initial_array = self._normalize_weights(initial_array)
            
            # 差分進化の実行
            result = differential_evolution(
                constrained_objective,
                bounds_list,
                seed=self.config.seed,
                maxiter=self.config.max_iterations,
                popsize=self.config.population_size,
                tol=self.config.tolerance,
                disp=self.config.verbose,
                x0=initial_array
            )
            
            # 結果の処理
            optimal_array = self._normalize_weights(result.x)
            optimal_weights = self._weights_array_to_dict(optimal_array, self.strategy_names)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                success=result.success,
                optimal_weights=optimal_weights,
                optimal_value=-result.fun,  # 元の最大化問題の値に戻す
                iterations=result.nit,
                function_evaluations=result.nfev,
                convergence_message=result.message,
                execution_time=execution_time,
                confidence_score=self._calculate_confidence_score(result),
                optimization_history=self.optimization_history.copy()
            )
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                success=False,
                optimal_weights=initial_weights,
                optimal_value=0.0,
                iterations=0,
                function_evaluations=0,
                convergence_message=f"Optimization failed: {str(e)}",
                execution_time=execution_time,
                confidence_score=0.0
            )
    
    def _calculate_confidence_score(self, scipy_result) -> float:
        """最適化結果の信頼度スコアを計算"""
        if not scipy_result.success:
            return 0.0
        
        confidence = 1.0
        
        # 収束性に基づく信頼度
        if scipy_result.nit >= self.config.max_iterations * 0.9:
            confidence *= 0.8  # 最大反復に近い場合は信頼度低下
        
        # 目的関数値の安定性
        if len(self.optimization_history) > 10:
            recent_values = [h['objective_value'] for h in self.optimization_history[-10:]]
            stability = 1.0 - np.std(recent_values) / (abs(np.mean(recent_values)) + 1e-8)
            confidence *= max(0.5, stability)
        
        return min(1.0, max(0.0, confidence))

class ScipyMinimizeOptimizer(OptimizationAlgorithm):
    """Scipy minimize最適化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.strategy_names = []
        self.method = 'SLSQP'  # Sequential Least Squares Programming
        
    def optimize(
        self,
        objective_function: Callable,
        initial_weights: Dict[str, float],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        constraints: Optional[List] = None
    ) -> OptimizationResult:
        """scipy.optimize.minimizeによる最適化"""
        start_time = datetime.now()
        
        try:
            self.strategy_names = list(initial_weights.keys())
            n_strategies = len(self.strategy_names)
            
            # 境界条件の設定
            if bounds is None:
                bounds_list = [(0.0, 1.0) for _ in range(n_strategies)]
            else:
                bounds_list = [bounds.get(name, (0.0, 1.0)) for name in self.strategy_names]
            
            # 制約条件の設定
            constraint_list = []
            
            # 重みの合計が1になる制約
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: x.sum() - 1.0
            })
            
            # 制約付き目的関数の定義
            def constrained_objective(weights_array):
                weights_dict = self._weights_array_to_dict(weights_array, self.strategy_names)
                
                # 目的関数値を計算
                objective_value = objective_function(weights_dict)
                
                # 履歴に記録
                self.optimization_history.append({
                    'weights': weights_dict.copy(),
                    'objective_value': objective_value,
                    'timestamp': datetime.now()
                })
                
                return -objective_value  # 最小化問題として定式化
            
            # 初期重みを配列に変換
            initial_array = self._weights_dict_to_array(initial_weights, self.strategy_names)
            initial_array = self._normalize_weights(initial_array)
            
            # 最適化の実行
            result = minimize(
                constrained_objective,
                initial_array,
                method=self.method,
                bounds=bounds_list,
                constraints=constraint_list,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.tolerance,
                    'disp': self.config.verbose
                }
            )
            
            # 結果の処理
            optimal_array = self._normalize_weights(result.x)
            optimal_weights = self._weights_array_to_dict(optimal_array, self.strategy_names)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                success=result.success,
                optimal_weights=optimal_weights,
                optimal_value=-result.fun,
                iterations=result.nit if hasattr(result, 'nit') else 0,
                function_evaluations=result.nfev if hasattr(result, 'nfev') else 0,
                convergence_message=result.message if hasattr(result, 'message') else '',
                execution_time=execution_time,
                confidence_score=self._calculate_confidence_score(result),
                optimization_history=self.optimization_history.copy()
            )
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                success=False,
                optimal_weights=initial_weights,
                optimal_value=0.0,
                iterations=0,
                function_evaluations=0,
                convergence_message=f"Optimization failed: {str(e)}",
                execution_time=execution_time,
                confidence_score=0.0
            )
    
    def _calculate_confidence_score(self, scipy_result) -> float:
        """最適化結果の信頼度スコアを計算"""
        if not scipy_result.success:
            return 0.0
        
        confidence = 1.0
        
        # 最適性条件に基づく信頼度
        if hasattr(scipy_result, 'optimality') and scipy_result.optimality:
            confidence *= max(0.5, 1.0 - scipy_result.optimality * 1000)
        
        return min(1.0, max(0.0, confidence))

class GradientDescentOptimizer(OptimizationAlgorithm):
    """勾配降下法最適化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.learning_rate = 0.01
        self.momentum = 0.9
        
    def optimize(
        self,
        objective_function: Callable,
        initial_weights: Dict[str, float],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        constraints: Optional[List] = None
    ) -> OptimizationResult:
        """勾配降下法による最適化"""
        start_time = datetime.now()
        
        try:
            self.strategy_names = list(initial_weights.keys())
            n_strategies = len(self.strategy_names)
            
            # 初期重みを配列に変換
            current_weights = self._weights_dict_to_array(initial_weights, self.strategy_names)
            current_weights = self._normalize_weights(current_weights)
            
            # モメンタム項の初期化
            velocity = np.zeros_like(current_weights)
            
            best_weights = current_weights.copy()
            best_value = objective_function(self._weights_array_to_dict(current_weights, self.strategy_names))
            
            # 勾配降下法の実行
            for iteration in range(self.config.max_iterations):
                # 数値勾配の計算
                gradient = self._compute_numerical_gradient(objective_function, current_weights)
                
                # モメンタムを使った更新
                velocity = self.momentum * velocity + self.learning_rate * gradient
                current_weights = current_weights + velocity
                
                # 制約の適用
                current_weights = self._apply_constraints(current_weights, bounds)
                current_weights = self._normalize_weights(current_weights)
                
                # 目的関数値の計算
                weights_dict = self._weights_array_to_dict(current_weights, self.strategy_names)
                current_value = objective_function(weights_dict)
                
                # 履歴に記録
                self.optimization_history.append({
                    'weights': weights_dict.copy(),
                    'objective_value': current_value,
                    'timestamp': datetime.now(),
                    'iteration': iteration
                })
                
                # 最良解の更新
                if current_value > best_value:
                    best_value = current_value
                    best_weights = current_weights.copy()
                
                # 収束判定
                if iteration > 0 and len(self.optimization_history) >= 2:
                    prev_value = self.optimization_history[-2]['objective_value']
                    if abs(current_value - prev_value) < self.config.tolerance:
                        break
            
            # 結果の処理
            optimal_weights = self._weights_array_to_dict(best_weights, self.strategy_names)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                success=True,
                optimal_weights=optimal_weights,
                optimal_value=best_value,
                iterations=iteration + 1,
                function_evaluations=(iteration + 1) * (n_strategies + 1),
                convergence_message="Gradient descent completed",
                execution_time=execution_time,
                confidence_score=self._calculate_confidence_score_gd(),
                optimization_history=self.optimization_history.copy()
            )
            
        except Exception as e:
            self.logger.error(f"Gradient descent optimization failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                success=False,
                optimal_weights=initial_weights,
                optimal_value=0.0,
                iterations=0,
                function_evaluations=0,
                convergence_message=f"Optimization failed: {str(e)}",
                execution_time=execution_time,
                confidence_score=0.0
            )
    
    def _compute_numerical_gradient(self, objective_function: Callable, weights: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """数値勾配を計算"""
        gradient = np.zeros_like(weights)
        
        for i in range(len(weights)):
            weights_plus = weights.copy()
            weights_minus = weights.copy()
            
            weights_plus[i] += epsilon
            weights_minus[i] -= epsilon
            
            # 制約を適用
            weights_plus = self._normalize_weights(weights_plus)
            weights_minus = self._normalize_weights(weights_minus)
            
            # 目的関数値を計算
            f_plus = objective_function(self._weights_array_to_dict(weights_plus, self.strategy_names))
            f_minus = objective_function(self._weights_array_to_dict(weights_minus, self.strategy_names))
            
            gradient[i] = (f_plus - f_minus) / (2 * epsilon)
        
        return gradient
    
    def _apply_constraints(self, weights: np.ndarray, bounds: Optional[Dict[str, Tuple[float, float]]]) -> np.ndarray:
        """制約を適用"""
        if bounds is None:
            weights = np.clip(weights, 0.0, 1.0)
        else:
            for i, strategy_name in enumerate(self.strategy_names):
                if strategy_name in bounds:
                    min_bound, max_bound = bounds[strategy_name]
                    weights[i] = np.clip(weights[i], min_bound, max_bound)
        
        return weights
    
    def _calculate_confidence_score_gd(self) -> float:
        """勾配降下法の信頼度スコアを計算"""
        if len(self.optimization_history) < 10:
            return 0.5
        
        # 目的関数値の改善度合いに基づく信頼度
        values = [h['objective_value'] for h in self.optimization_history]
        initial_value = values[0]
        final_value = values[-1]
        
        if initial_value == 0:
            return 0.5
        
        improvement = (final_value - initial_value) / abs(initial_value)
        confidence = min(1.0, max(0.0, improvement + 0.5))
        
        return confidence

class OptimizationEngine:
    """最適化エンジン"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def create_optimizer(self, method: OptimizationMethod, config: OptimizationConfig) -> OptimizationAlgorithm:
        """最適化器を作成"""
        if method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            return DifferentialEvolutionOptimizer(config)
        elif method == OptimizationMethod.SCIPY_MINIMIZE:
            return ScipyMinimizeOptimizer(config)
        elif method == OptimizationMethod.GRADIENT_DESCENT:
            return GradientDescentOptimizer(config)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
    
    def run_optimization(
        self,
        objective_function: Callable,
        initial_weights: Dict[str, float],
        method: OptimizationMethod = OptimizationMethod.DIFFERENTIAL_EVOLUTION,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        constraints: Optional[List] = None,
        optimization_config: Optional[OptimizationConfig] = None
    ) -> OptimizationResult:
        """最適化を実行"""
        
        if optimization_config is None:
            optimization_config = OptimizationConfig(
                method=method,
                max_iterations=1000,
                population_size=50,
                tolerance=1e-6
            )
        
        optimizer = self.create_optimizer(method, optimization_config)
        result = optimizer.optimize(objective_function, initial_weights, bounds, constraints)
        
        self.logger.info(f"Optimization completed: success={result.success}, value={result.optimal_value:.4f}")
        
        return result


# テスト用のメイン関数
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    logger.info("Testing Optimization Algorithms...")
    
    # テスト目的関数（シンプルなシャープレシオ最大化）
    def test_objective_function(weights_dict: Dict[str, float]) -> float:
        # ダミーリターンデータ
        np.random.seed(42)
        returns_data = {
            'strategy1': np.random.normal(0.001, 0.02, 252),
            'strategy2': np.random.normal(0.0015, 0.025, 252),
            'strategy3': np.random.normal(0.0008, 0.018, 252)
        }
        
        # ポートフォリオリターンの計算
        portfolio_return = sum(weights_dict.get(s, 0) * np.mean(returns_data[s]) for s in returns_data.keys())
        portfolio_volatility = np.sqrt(sum((weights_dict.get(s, 0) ** 2) * np.var(returns_data[s]) for s in returns_data.keys()))
        
        if portfolio_volatility == 0:
            return 0.0
        
        # シャープレシオ
        sharpe_ratio = (portfolio_return * 252) / (portfolio_volatility * np.sqrt(252))
        return sharpe_ratio
    
    # 初期重み
    initial_weights = {
        'strategy1': 0.4,
        'strategy2': 0.3,
        'strategy3': 0.3
    }
    
    # 最適化エンジンのテスト
    engine = OptimizationEngine()
    
    # 差分進化による最適化
    config = OptimizationConfig(
        method=OptimizationMethod.DIFFERENTIAL_EVOLUTION,
        max_iterations=100,
        population_size=20,
        tolerance=1e-4,
        seed=42
    )
    
    result = engine.run_optimization(
        objective_function=test_objective_function,
        initial_weights=initial_weights,
        method=OptimizationMethod.DIFFERENTIAL_EVOLUTION,
        optimization_config=config
    )
    
    logger.info("Optimization Results:")
    logger.info(f"Success: {result.success}")
    logger.info(f"Optimal Value: {result.optimal_value:.4f}")
    logger.info(f"Optimal Weights: {result.optimal_weights}")
    logger.info(f"Iterations: {result.iterations}")
    logger.info(f"Execution Time: {result.execution_time:.2f} seconds")
    logger.info(f"Confidence Score: {result.confidence_score:.4f}")
    
    logger.info("Optimization Algorithms test completed successfully!")
