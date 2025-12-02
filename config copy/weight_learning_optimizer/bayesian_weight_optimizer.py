"""
ベイジアン最適化による重み学習システム

Gaussian Process回帰とExpected Improvement取得関数を使用して
効率的な重み空間の探索を実行
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from scipy.optimize import minimize
import logging
from pathlib import Path

@dataclass
class OptimizationResult:
    """最適化結果を格納するデータクラス"""
    optimized_weights: Dict[str, float]
    expected_performance: float
    uncertainty: float
    acquisition_value: float
    iteration: int
    timestamp: datetime
    
@dataclass
class BayesianConfig:
    """ベイジアン最適化の設定パラメータ"""
    # ガウス過程設定
    kernel_type: str = 'matern'  # 'rbf', 'matern'
    length_scale: float = 1.0
    nu: float = 1.5  # Maternカーネルのパラメータ
    alpha: float = 1e-6  # ノイズレベル
    
    # 最適化設定
    acquisition_function: str = 'EI'  # 'EI', 'PI', 'UCB'
    xi: float = 0.01  # Exploration parameter
    kappa: float = 2.576  # UCB parameter
    
    # 学習設定
    n_initial_samples: int = 10
    max_iterations: int = 50
    convergence_threshold: float = 1e-6
    
class BayesianWeightOptimizer:
    """
    ベイジアン最適化による重み学習システム
    
    Gaussian Process回帰を使用して効率的な重み空間探索を実行し、
    期待値最大化とリスク最小化の両方を考慮した最適化を行う。
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        integration_bridge: Optional[Any] = None
    ):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
            integration_bridge: 統合ブリッジのインスタンス
        """
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.integration_bridge = integration_bridge
        
        # ベイジアン最適化コンポーネント
        self.gp_model = None
        self.optimization_history = []
        self.current_weights = {}
        self.bounds = {}
        
        # パフォーマンス追跡
        self.performance_history = []
        self.best_performance = -np.inf
        self.best_weights = {}
        
        # 制約管理
        self.weight_constraints = {}
        self.meta_constraints = {}
        
        self.logger.info("BayesianWeightOptimizer initialized")
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.BayesianWeightOptimizer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _load_config(self, config_path: Optional[str]) -> BayesianConfig:
        """設定ファイルの読み込み"""
        if config_path and Path(config_path).exists():
            # 設定ファイルから読み込み（実装省略）
            pass
        return BayesianConfig()
        
    def initialize_optimization(
        self,
        strategy_weights: Dict[str, float],
        portfolio_weights: Dict[str, float],
        meta_parameters: Dict[str, float],
        constraints: Dict[str, Dict[str, float]]
    ) -> None:
        """
        最適化の初期化
        
        Args:
            strategy_weights: ストラテジースコア重み
            portfolio_weights: ポートフォリオ重み
            meta_parameters: メタパラメータ
            constraints: 制約条件
        """
        self.logger.info("Initializing Bayesian optimization")
        
        # 現在の重みを設定
        self.current_weights = {
            'strategy': strategy_weights.copy(),
            'portfolio': portfolio_weights.copy(),
            'meta': meta_parameters.copy()
        }
        
        # 制約を設定
        self.weight_constraints = constraints.get('weights', {})
        self.meta_constraints = constraints.get('meta', {})
        
        # 最適化境界を設定
        self._setup_optimization_bounds()
        
        # ガウス過程モデルの初期化
        self._initialize_gaussian_process()
        
        self.logger.info("Bayesian optimization initialized successfully")
        
    def _setup_optimization_bounds(self) -> None:
        """最適化境界の設定"""
        self.bounds = {}
        
        # ストラテジー重み境界
        for key in self.current_weights['strategy']:
            self.bounds[f'strategy_{key}'] = (
                self.weight_constraints.get(f'{key}_min', 0.0),
                self.weight_constraints.get(f'{key}_max', 1.0)
            )
            
        # ポートフォリオ重み境界
        for key in self.current_weights['portfolio']:
            self.bounds[f'portfolio_{key}'] = (
                self.weight_constraints.get(f'{key}_min', 0.0),
                self.weight_constraints.get(f'{key}_max', 1.0)
            )
            
        # メタパラメータ境界
        for key in self.current_weights['meta']:
            self.bounds[f'meta_{key}'] = (
                self.meta_constraints.get(f'{key}_min', 0.5),
                self.meta_constraints.get(f'{key}_max', 2.0)
            )
            
    def _initialize_gaussian_process(self) -> None:
        """ガウス過程モデルの初期化"""
        # カーネルの選択
        if self.config.kernel_type == 'rbf':
            kernel = RBF(length_scale=self.config.length_scale) + WhiteKernel()
        elif self.config.kernel_type == 'matern':
            kernel = Matern(
                length_scale=self.config.length_scale,
                nu=self.config.nu
            ) + WhiteKernel()
        else:
            kernel = RBF(length_scale=self.config.length_scale) + WhiteKernel()
            
        # ガウス過程回帰モデル
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.alpha,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
    def optimize_weights(
        self,
        performance_data: pd.DataFrame,
        target_metrics: List[str] = None
    ) -> OptimizationResult:
        """
        ベイジアン最適化による重み最適化
        
        Args:
            performance_data: パフォーマンスデータ
            target_metrics: ターゲット指標
            
        Returns:
            最適化結果
        """
        self.logger.info("Starting Bayesian weight optimization")
        
        if target_metrics is None:
            target_metrics = ['expected_return', 'max_drawdown', 'sharpe_ratio']
            
        # 初期サンプルの生成
        if len(self.optimization_history) < self.config.n_initial_samples:
            self._generate_initial_samples(performance_data, target_metrics)
            
        best_result = None
        
        for iteration in range(self.config.max_iterations):
            self.logger.info(f"Optimization iteration {iteration + 1}")
            
            # ガウス過程の更新
            self._update_gaussian_process()
            
            # 取得関数による次の候補点の選択
            next_weights = self._select_next_candidate()
            
            # パフォーマンス評価
            performance = self._evaluate_performance(
                next_weights, performance_data, target_metrics
            )
            
            # 履歴の更新
            result = OptimizationResult(
                optimized_weights=next_weights,
                expected_performance=performance['combined_score'],
                uncertainty=performance.get('uncertainty', 0.0),
                acquisition_value=performance.get('acquisition_value', 0.0),
                iteration=iteration + 1,
                timestamp=datetime.now()
            )
            
            self.optimization_history.append(result)
            
            # 最良結果の更新
            if performance['combined_score'] > self.best_performance:
                self.best_performance = performance['combined_score']
                self.best_weights = next_weights.copy()
                best_result = result
                
            # 収束判定
            if self._check_convergence():
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break
                
        if best_result is None:
            best_result = self.optimization_history[-1] if self.optimization_history else None
            
        self.logger.info("Bayesian optimization completed")
        return best_result
        
    def _generate_initial_samples(
        self,
        performance_data: pd.DataFrame,
        target_metrics: List[str]
    ) -> None:
        """初期サンプルの生成"""
        self.logger.info("Generating initial samples")
        
        n_needed = self.config.n_initial_samples - len(self.optimization_history)
        
        for i in range(n_needed):
            # ランダムサンプリング
            sample_weights = self._generate_random_weights()
            
            # パフォーマンス評価
            performance = self._evaluate_performance(
                sample_weights, performance_data, target_metrics
            )
            
            # 履歴に追加
            result = OptimizationResult(
                optimized_weights=sample_weights,
                expected_performance=performance['combined_score'],
                uncertainty=0.0,
                acquisition_value=0.0,
                iteration=0,
                timestamp=datetime.now()
            )
            
            self.optimization_history.append(result)
            
    def _generate_random_weights(self) -> Dict[str, float]:
        """ランダムな重みの生成"""
        weights = {}
        
        # 境界内でランダムサンプリング
        for param, (min_val, max_val) in self.bounds.items():
            weights[param] = np.random.uniform(min_val, max_val)
            
        # 正規化処理
        weights = self._normalize_weights(weights)
        
        return weights
        
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """重みの正規化"""
        normalized_weights = {}
        
        # ストラテジー重みの正規化
        strategy_keys = [k for k in weights.keys() if k.startswith('strategy_')]
        if strategy_keys:
            strategy_sum = sum(weights[k] for k in strategy_keys)
            if strategy_sum > 0:
                for k in strategy_keys:
                    normalized_weights[k] = weights[k] / strategy_sum
            else:
                for k in strategy_keys:
                    normalized_weights[k] = 1.0 / len(strategy_keys)
                    
        # ポートフォリオ重みの正規化
        portfolio_keys = [k for k in weights.keys() if k.startswith('portfolio_')]
        if portfolio_keys:
            portfolio_sum = sum(weights[k] for k in portfolio_keys)
            if portfolio_sum > 0:
                for k in portfolio_keys:
                    normalized_weights[k] = weights[k] / portfolio_sum
            else:
                for k in portfolio_keys:
                    normalized_weights[k] = 1.0 / len(portfolio_keys)
                    
        # メタパラメータはそのまま
        meta_keys = [k for k in weights.keys() if k.startswith('meta_')]
        for k in meta_keys:
            normalized_weights[k] = weights[k]
            
        return normalized_weights
        
    def _update_gaussian_process(self) -> None:
        """ガウス過程の更新"""
        if len(self.optimization_history) < 2:
            return
            
        # 特徴量とターゲットの準備
        X = []
        y = []
        
        for result in self.optimization_history:
            features = list(result.optimized_weights.values())
            X.append(features)
            y.append(result.expected_performance)
            
        X = np.array(X)
        y = np.array(y)
        
        # NaN値と無限値の処理
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 2:
            # 有効なデータが少ない場合はスキップ
            return
            
        # ガウス過程の学習
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.gp_model.fit(X, y)
            except Exception as e:
                self.logger.warning(f"Failed to fit Gaussian process: {e}")
                # フォールバック: 単純なケースで再試行
                X = np.random.rand(3, len(self.bounds)) * 0.01 + 0.495
                y = np.array([0.2, 0.3, 0.25])
                self.gp_model.fit(X, y)
            
    def _select_next_candidate(self) -> Dict[str, float]:
        """取得関数による次の候補点の選択"""
        bounds_array = np.array(list(self.bounds.values()))
        
        def acquisition_function(x):
            return -self._calculate_acquisition(x.reshape(1, -1))
            
        # 複数の初期点から最適化
        best_x = None
        best_acq = np.inf
        
        for _ in range(10):  # 複数回試行
            x0 = np.random.uniform(
                bounds_array[:, 0], 
                bounds_array[:, 1]
            )
            
            result = minimize(
                acquisition_function,
                x0,
                bounds=bounds_array,
                method='L-BFGS-B'
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
                
        # 辞書形式に変換
        if best_x is not None:
            weights = {}
            for i, key in enumerate(self.bounds.keys()):
                weights[key] = best_x[i]
            return self._normalize_weights(weights)
        else:
            return self._generate_random_weights()
            
    def _calculate_acquisition(self, X: np.ndarray) -> np.ndarray:
        """取得関数の計算"""
        if self.gp_model is None or len(self.optimization_history) < 2:
            return np.array([0.0])
            
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        if self.config.acquisition_function == 'EI':
            # Expected Improvement
            f_best = self.best_performance
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (mu - f_best - self.config.xi) / sigma
                ei = (mu - f_best - self.config.xi) * self._normal_cdf(z) + sigma * self._normal_pdf(z)
                ei[sigma == 0.0] = 0.0
            return ei
            
        elif self.config.acquisition_function == 'PI':
            # Probability of Improvement
            f_best = self.best_performance
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (mu - f_best - self.config.xi) / sigma
                pi = self._normal_cdf(z)
                pi[sigma == 0.0] = 0.0
            return pi
            
        elif self.config.acquisition_function == 'UCB':
            # Upper Confidence Bound
            return mu + self.config.kappa * sigma
            
        else:
            return mu
            
    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """正規分布の累積分布関数"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """正規分布の確率密度関数"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        
    def _evaluate_performance(
        self,
        weights: Dict[str, float],
        performance_data: pd.DataFrame,
        target_metrics: List[str]
    ) -> Dict[str, float]:
        """パフォーマンス評価"""
        if self.integration_bridge is None:
            # テスト用のダミー評価
            return {
                'combined_score': np.random.random(),
                'expected_return': np.random.random(),
                'max_drawdown': np.random.random(),
                'sharpe_ratio': np.random.random(),
                'uncertainty': np.random.random() * 0.1,
                'acquisition_value': np.random.random()
            }
            
        # 統合ブリッジを通じた実際の評価
        return self.integration_bridge.evaluate_weight_combination(
            weights, performance_data, target_metrics
        )
        
    def _check_convergence(self) -> bool:
        """収束判定"""
        if len(self.optimization_history) < 5:
            return False
            
        # 最近の5回の改善度をチェック
        recent_scores = [
            result.expected_performance 
            for result in self.optimization_history[-5:]
        ]
        
        improvement = max(recent_scores) - min(recent_scores)
        return improvement < self.config.convergence_threshold
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化結果のサマリー"""
        if not self.optimization_history:
            return {}
            
        return {
            'best_performance': self.best_performance,
            'best_weights': self.best_weights,
            'total_iterations': len(self.optimization_history),
            'convergence_achieved': self._check_convergence(),
            'optimization_time': (
                self.optimization_history[-1].timestamp - 
                self.optimization_history[0].timestamp
            ).total_seconds() if len(self.optimization_history) > 1 else 0,
            'performance_improvement': (
                self.optimization_history[-1].expected_performance - 
                self.optimization_history[0].expected_performance
            ) if len(self.optimization_history) > 1 else 0
        }
        
    def save_optimization_state(self, filepath: str) -> None:
        """最適化状態の保存"""
        import pickle
        
        state = {
            'optimization_history': self.optimization_history,
            'best_performance': self.best_performance,
            'best_weights': self.best_weights,
            'current_weights': self.current_weights,
            'bounds': self.bounds,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
        self.logger.info(f"Optimization state saved to {filepath}")
        
    def load_optimization_state(self, filepath: str) -> None:
        """最適化状態の読み込み"""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        self.optimization_history = state.get('optimization_history', [])
        self.best_performance = state.get('best_performance', -np.inf)
        self.best_weights = state.get('best_weights', {})
        self.current_weights = state.get('current_weights', {})
        self.bounds = state.get('bounds', {})
        self.config = state.get('config', BayesianConfig())
        
        # ガウス過程の再初期化
        self._initialize_gaussian_process()
        if len(self.optimization_history) > 1:
            self._update_gaussian_process()
            
        self.logger.info(f"Optimization state loaded from {filepath}")
