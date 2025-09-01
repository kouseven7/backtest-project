"""
DSSMS Phase 3 Task 3.1: Dynamic Weight Optimizer
動的重み最適化器

市場状況に応じて各分析要素の重みを動的に調整し、
最適化されたランキングを提供します。
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 設定とロガー
try:
    from config.logger_config import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

class OptimizationMethod(Enum):
    """最適化手法"""
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm" 
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    ENSEMBLE = "ensemble"

class MarketRegime(Enum):
    """市場レジーム定義"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"

class OptimizationMethod(Enum):
    """最適化手法定義"""
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    ENSEMBLE = "ensemble"

@dataclass
class WeightConstraints:
    """重み制約"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    sum_constraint: float = 1.0
    stability_factor: float = 0.1  # 重み変化の制限

@dataclass
class OptimizationConfig:
    """最適化設定"""
    method: OptimizationMethod = OptimizationMethod.GRADIENT_DESCENT
    constraints: WeightConstraints = field(default_factory=WeightConstraints)
    lookback_period: int = 252  # 学習期間
    rebalance_frequency: int = 20  # リバランス頻度（日）
    objective_function: str = "sharpe_ratio"  # "sharpe_ratio", "information_ratio", "accuracy"
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    enable_regime_detection: bool = True
    enable_adaptive_learning: bool = True
    rolling_window: int = 60

@dataclass
class WeightOptimizationResult:
    """重み最適化結果"""
    optimal_weights: Dict[str, float]
    regime: MarketRegime
    confidence_score: float
    optimization_score: float
    convergence_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: datetime

@dataclass
class OptimizationConfig:
    """最適化設定"""
    enable_gradient_descent: bool = True
    enable_genetic_algorithm: bool = True
    enable_bayesian_optimization: bool = True
    method: str = "gradient_descent"
    population_size: int = 50
    generations: int = 100
    learning_rate: float = 0.01
    convergence_threshold: float = 1e-6
    rolling_window: int = 252
    lookback_period: int = 252
    log_level: str = "INFO"

class DynamicWeightOptimizer:
    """
    動的重み最適化器
    
    機能:
    - 市場レジーム検出
    - 各レジームに最適な重み計算
    - 動的重み調整
    - パフォーマンス監視
    - 適応学習
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        初期化
        
        Args:
            config: 最適化設定
        """
        self.config = config or OptimizationConfig()
        self.logger = logger
        
        # 重み履歴
        self._weight_history = {}
        self._regime_history = []
        self._performance_history = []
        
        # 学習データ
        self._training_data = {}
        self._model_cache = {}
        
        # 現在の重み
        self._current_weights = self._initialize_default_weights()
        self._current_regime = MarketRegime.SIDEWAYS
        
        # 最適化履歴
        self._optimization_history = []
        
        self.logger.info(f"Dynamic Weight Optimizer initialized with method: {self.config.method}")
    
    def _initialize_default_weights(self) -> Dict[str, float]:
        """デフォルト重み初期化"""
        return {
            'momentum': 0.25,
            'volatility': 0.15,
            'volume': 0.15,
            'technical': 0.25,
            'fundamental': 0.10,
            'trend': 0.10
        }
    
    def optimize_weights(
        self, 
        historical_data: Dict[str, pd.DataFrame],
        performance_data: Optional[Dict[str, pd.DataFrame]] = None,
        current_scores: Optional[Dict[str, Dict[str, float]]] = None
    ) -> WeightOptimizationResult:
        """
        重み最適化実行
        
        Args:
            historical_data: 過去データ
            performance_data: パフォーマンスデータ
            current_scores: 現在のスコアデータ
            
        Returns:
            重み最適化結果
        """
        start_time = datetime.now()
        self.logger.info("Starting dynamic weight optimization")
        
        try:
            # 市場レジーム検出
            current_regime = self._detect_market_regime(historical_data)
            
            # 学習データ準備
            training_features, training_targets = self._prepare_training_data(
                historical_data, performance_data
            )
            
            # 最適化実行
            if self.config.method == OptimizationMethod.GRADIENT_DESCENT:
                optimization_result = self._optimize_gradient_descent(
                    training_features, training_targets, current_regime
                )
            elif self.config.method == OptimizationMethod.GENETIC_ALGORITHM:
                optimization_result = self._optimize_genetic_algorithm(
                    training_features, training_targets, current_regime
                )
            elif self.config.method == OptimizationMethod.ENSEMBLE:
                optimization_result = self._optimize_ensemble(
                    training_features, training_targets, current_regime
                )
            else:
                optimization_result = self._optimize_gradient_descent(
                    training_features, training_targets, current_regime
                )
            
            # 信頼度スコア計算
            confidence_score = self._calculate_confidence_score(
                optimization_result, training_features, training_targets
            )
            
            # パフォーマンスメトリクス計算
            performance_metrics = self._calculate_performance_metrics(
                optimization_result, current_scores
            )
            
            # 結果構築
            result = WeightOptimizationResult(
                optimal_weights=optimization_result['weights'],
                regime=current_regime,
                confidence_score=confidence_score,
                optimization_score=optimization_result['score'],
                convergence_info=optimization_result['convergence'],
                performance_metrics=performance_metrics,
                timestamp=start_time
            )
            
            # 履歴更新
            self._update_history(result)
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Weight optimization completed in {optimization_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Weight optimization failed: {e}")
            raise
    
    def _detect_market_regime(self, historical_data: Dict[str, pd.DataFrame]) -> MarketRegime:
        """市場レジーム検出"""
        try:
            # 市場データ集計
            market_returns = self._calculate_market_returns(historical_data)
            
            if market_returns.empty:
                return MarketRegime.SIDEWAYS
            
            # 統計指標計算
            recent_returns = market_returns.tail(self.config.rolling_window)
            mean_return = recent_returns.mean()
            volatility = recent_returns.std()
            skewness = recent_returns.skew()
            
            # 年率化
            annual_return = mean_return * 252
            annual_volatility = volatility * np.sqrt(252)
            
            # レジーム判定
            if annual_volatility > 0.30:  # 高ボラティリティ
                if annual_return < -0.20:
                    return MarketRegime.CRISIS
                else:
                    return MarketRegime.HIGH_VOLATILITY
            elif annual_volatility < 0.10:  # 低ボラティリティ
                return MarketRegime.LOW_VOLATILITY
            elif annual_return > 0.15:  # 強い上昇トレンド
                return MarketRegime.BULL_TRENDING
            elif annual_return < -0.15:  # 強い下降トレンド
                return MarketRegime.BEAR_TRENDING
            elif abs(annual_return) < 0.05:  # 横ばい
                return MarketRegime.SIDEWAYS
            elif annual_return > 0 and skewness > 0:  # 回復相場
                return MarketRegime.RECOVERY
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            self.logger.warning(f"Market regime detection failed: {e}")
            return MarketRegime.SIDEWAYS
    
    def _calculate_market_returns(self, historical_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """市場リターン計算"""
        try:
            # 複数銘柄の平均リターン
            all_returns = []
            
            for symbol, data in historical_data.items():
                if 'Close' in data.columns and len(data) > 1:
                    returns = data['Close'].pct_change().dropna()
                    all_returns.append(returns)
            
            if not all_returns:
                return pd.Series(dtype=float)
            
            # 日付で整列して平均
            market_returns = pd.concat(all_returns, axis=1).mean(axis=1, skipna=True)
            return market_returns.dropna()
            
        except Exception as e:
            self.logger.warning(f"Market returns calculation failed: {e}")
            return pd.Series(dtype=float)
    
    def _prepare_training_data(
        self, 
        historical_data: Dict[str, pd.DataFrame],
        performance_data: Optional[Dict[str, pd.DataFrame]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """学習データ準備"""
        try:
            features = []
            targets = []
            
            # 各銘柄のデータから特徴量とターゲット抽出
            for symbol, data in historical_data.items():
                if len(data) < self.config.lookback_period:
                    continue
                
                # 特徴量計算
                symbol_features = self._extract_features(data)
                
                # ターゲット計算（将来のパフォーマンス）
                symbol_targets = self._extract_targets(data, performance_data, symbol)
                
                if len(symbol_features) == len(symbol_targets):
                    features.extend(symbol_features)
                    targets.extend(symbol_targets)
            
            if not features:
                # フォールバック：ダミーデータ
                return np.array([[1.0] * 6]), np.array([1.0])
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            self.logger.warning(f"Training data preparation failed: {e}")
            return np.array([[1.0] * 6]), np.array([1.0])
    
    def _extract_features(self, data: pd.DataFrame) -> List[List[float]]:
        """特徴量抽出"""
        try:
            features = []
            
            # 移動窓で特徴量計算
            window_size = min(20, len(data) // 4)
            
            for i in range(window_size, len(data)):
                window_data = data.iloc[i-window_size:i]
                
                feature_vector = [
                    self._calculate_momentum_feature(window_data),
                    self._calculate_volatility_feature(window_data),
                    self._calculate_volume_feature(window_data),
                    self._calculate_technical_feature(window_data),
                    self._calculate_trend_feature(window_data),
                    self._calculate_relative_strength_feature(window_data)
                ]
                
                features.append(feature_vector)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return []
    
    def _extract_targets(
        self, 
        data: pd.DataFrame, 
        performance_data: Optional[Dict[str, pd.DataFrame]], 
        symbol: str
    ) -> List[float]:
        """ターゲット抽出"""
        try:
            # 将来リターンをターゲットとして使用
            returns = data['Close'].pct_change().shift(-5)  # 5日後のリターン
            
            targets = []
            window_size = min(20, len(data) // 4)
            
            for i in range(window_size, len(data)):
                if i < len(returns) and not pd.isna(returns.iloc[i]):
                    targets.append(returns.iloc[i])
                else:
                    targets.append(0.0)
            
            return targets
            
        except Exception as e:
            self.logger.warning(f"Target extraction failed: {e}")
            return []
    
    def _calculate_momentum_feature(self, data: pd.DataFrame) -> float:
        """モメンタム特徴量計算"""
        try:
            return data['Close'].pct_change().mean()
        except:
            return 0.0
    
    def _calculate_volatility_feature(self, data: pd.DataFrame) -> float:
        """ボラティリティ特徴量計算"""
        try:
            return data['Close'].pct_change().std()
        except:
            return 0.0
    
    def _calculate_volume_feature(self, data: pd.DataFrame) -> float:
        """出来高特徴量計算"""
        try:
            return data['Volume'].pct_change().mean()
        except:
            return 0.0
    
    def _calculate_technical_feature(self, data: pd.DataFrame) -> float:
        """テクニカル特徴量計算"""
        try:
            sma_short = data['Close'].rolling(5).mean()
            sma_long = data['Close'].rolling(10).mean()
            return (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        except:
            return 0.0
    
    def _calculate_trend_feature(self, data: pd.DataFrame) -> float:
        """トレンド特徴量計算"""
        try:
            prices = data['Close'].values
            return np.polyfit(range(len(prices)), prices, 1)[0] / prices.mean()
        except:
            return 0.0
    
    def _calculate_relative_strength_feature(self, data: pd.DataFrame) -> float:
        """相対強度特徴量計算"""
        try:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return (rsi.iloc[-1] - 50) / 50
        except:
            return 0.0
    
    def _optimize_gradient_descent(
        self, 
        features: np.ndarray, 
        targets: np.ndarray, 
        regime: MarketRegime
    ) -> Dict[str, Any]:
        """勾配降下法による最適化"""
        try:
            # 初期重み（レジーム別）
            initial_weights = self._get_regime_initial_weights(regime)
            x0 = np.array(list(initial_weights.values()))
            
            # 制約条件
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # 合計=1
            ]
            
            bounds = [(self.config.constraints.min_weight, self.config.constraints.max_weight) 
                     for _ in range(len(x0))]
            
            # 最適化実行
            result = minimize(
                fun=self._objective_function,
                x0=x0,
                args=(features, targets),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iterations}
            )
            
            # 結果整理
            weight_names = list(initial_weights.keys())
            optimal_weights = {name: weight for name, weight in zip(weight_names, result.x)}
            
            return {
                'weights': optimal_weights,
                'score': -result.fun,  # 目的関数は最小化なので符号反転
                'convergence': {
                    'success': result.success,
                    'iterations': result.nit,
                    'message': result.message
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Gradient descent optimization failed: {e}")
            return {
                'weights': self._get_regime_initial_weights(regime),
                'score': 0.0,
                'convergence': {'success': False, 'iterations': 0, 'message': str(e)}
            }
    
    def _optimize_genetic_algorithm(
        self, 
        features: np.ndarray, 
        targets: np.ndarray, 
        regime: MarketRegime
    ) -> Dict[str, Any]:
        """遺伝的アルゴリズムによる最適化（簡易版）"""
        try:
            # パラメータ
            population_size = 50
            generations = 20
            mutation_rate = 0.1
            
            # 初期集団生成
            population = []
            for _ in range(population_size):
                weights = np.random.dirichlet(np.ones(6))  # 合計=1になるランダム重み
                population.append(weights)
            
            best_weights = None
            best_score = float('-inf')
            
            # 世代ループ
            for generation in range(generations):
                # 評価
                scores = []
                for weights in population:
                    score = -self._objective_function(weights, features, targets)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_weights = weights.copy()
                
                # 選択・交叉・突然変異
                population = self._genetic_operations(population, scores, mutation_rate)
            
            # 結果整理
            weight_names = list(self._get_regime_initial_weights(regime).keys())
            optimal_weights = {name: weight for name, weight in zip(weight_names, best_weights)}
            
            return {
                'weights': optimal_weights,
                'score': best_score,
                'convergence': {
                    'success': True,
                    'iterations': generations,
                    'message': 'Genetic algorithm completed'
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Genetic algorithm optimization failed: {e}")
            return {
                'weights': self._get_regime_initial_weights(regime),
                'score': 0.0,
                'convergence': {'success': False, 'iterations': 0, 'message': str(e)}
            }
    
    def _optimize_ensemble(
        self, 
        features: np.ndarray, 
        targets: np.ndarray, 
        regime: MarketRegime
    ) -> Dict[str, Any]:
        """アンサンブル最適化"""
        try:
            # 複数手法で最適化
            results = []
            
            # 勾配降下法
            gd_result = self._optimize_gradient_descent(features, targets, regime)
            results.append(gd_result)
            
            # 遺伝的アルゴリズム
            ga_result = self._optimize_genetic_algorithm(features, targets, regime)
            results.append(ga_result)
            
            # 最高スコアの結果を選択
            best_result = max(results, key=lambda x: x['score'])
            
            return best_result
            
        except Exception as e:
            self.logger.warning(f"Ensemble optimization failed: {e}")
            return {
                'weights': self._get_regime_initial_weights(regime),
                'score': 0.0,
                'convergence': {'success': False, 'iterations': 0, 'message': str(e)}
            }
    
    def _genetic_operations(self, population: List[np.ndarray], scores: List[float], mutation_rate: float) -> List[np.ndarray]:
        """遺伝的操作"""
        try:
            new_population = []
            
            # エリート保存
            elite_indices = np.argsort(scores)[-10:]  # 上位10個体
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 交叉と突然変異
            while len(new_population) < len(population):
                # 親選択（トーナメント選択）
                parent1 = self._tournament_selection(population, scores)
                parent2 = self._tournament_selection(population, scores)
                
                # 交叉
                child = self._crossover(parent1, parent2)
                
                # 突然変異
                if np.random.random() < mutation_rate:
                    child = self._mutate(child)
                
                # 制約確認（合計=1）
                child = child / np.sum(child)
                
                new_population.append(child)
            
            return new_population[:len(population)]
            
        except Exception as e:
            self.logger.warning(f"Genetic operations failed: {e}")
            return population
    
    def _tournament_selection(self, population: List[np.ndarray], scores: List[float]) -> np.ndarray:
        """トーナメント選択"""
        tournament_size = 5
        indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = indices[np.argmax([scores[i] for i in indices])]
        return population[best_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """交叉"""
        alpha = np.random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child / np.sum(child)  # 正規化
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """突然変異"""
        mutation_strength = 0.1
        noise = np.random.normal(0, mutation_strength, len(individual))
        mutated = individual + noise
        mutated = np.clip(mutated, self.config.constraints.min_weight, self.config.constraints.max_weight)
        return mutated / np.sum(mutated)  # 正規化
    
    def _get_regime_initial_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """レジーム別初期重み"""
        
        if regime == MarketRegime.BULL_TRENDING:
            return {
                'momentum': 0.35,
                'volatility': 0.10,
                'volume': 0.20,
                'technical': 0.20,
                'fundamental': 0.10,
                'trend': 0.05
            }
        elif regime == MarketRegime.BEAR_TRENDING:
            return {
                'momentum': 0.15,
                'volatility': 0.30,
                'volume': 0.15,
                'technical': 0.25,
                'fundamental': 0.10,
                'trend': 0.05
            }
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return {
                'momentum': 0.20,
                'volatility': 0.35,
                'volume': 0.20,
                'technical': 0.15,
                'fundamental': 0.05,
                'trend': 0.05
            }
        elif regime == MarketRegime.LOW_VOLATILITY:
            return {
                'momentum': 0.15,
                'volatility': 0.10,
                'volume': 0.15,
                'technical': 0.30,
                'fundamental': 0.25,
                'trend': 0.05
            }
        elif regime == MarketRegime.CRISIS:
            return {
                'momentum': 0.10,
                'volatility': 0.40,
                'volume': 0.25,
                'technical': 0.15,
                'fundamental': 0.05,
                'trend': 0.05
            }
        elif regime == MarketRegime.RECOVERY:
            return {
                'momentum': 0.30,
                'volatility': 0.15,
                'volume': 0.25,
                'technical': 0.20,
                'fundamental': 0.05,
                'trend': 0.05
            }
        else:  # SIDEWAYS
            return {
                'momentum': 0.20,
                'volatility': 0.15,
                'volume': 0.20,
                'technical': 0.25,
                'fundamental': 0.15,
                'trend': 0.05
            }
    
    def _objective_function(self, weights: np.ndarray, features: np.ndarray, targets: np.ndarray) -> float:
        """目的関数"""
        try:
            if self.config.objective_function == "sharpe_ratio":
                return self._calculate_sharpe_ratio_objective(weights, features, targets)
            elif self.config.objective_function == "information_ratio":
                return self._calculate_information_ratio_objective(weights, features, targets)
            elif self.config.objective_function == "accuracy":
                return self._calculate_accuracy_objective(weights, features, targets)
            else:
                return self._calculate_sharpe_ratio_objective(weights, features, targets)
                
        except Exception as e:
            self.logger.warning(f"Objective function calculation failed: {e}")
            return -1000.0  # ペナルティ
    
    def _calculate_sharpe_ratio_objective(self, weights: np.ndarray, features: np.ndarray, targets: np.ndarray) -> float:
        """シャープレシオ目的関数"""
        try:
            # 重み付きスコア計算
            weighted_scores = np.dot(features, weights)
            
            # スコアと目標の相関
            correlation = np.corrcoef(weighted_scores, targets)[0, 1]
            
            if np.isnan(correlation):
                return -1000.0
            
            # シャープレシオ風の計算
            returns = weighted_scores * targets
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return -1000.0
            
            sharpe_ratio = mean_return / std_return
            
            # 最小化問題なので符号反転
            return -sharpe_ratio
            
        except Exception:
            return -1000.0
    
    def _calculate_information_ratio_objective(self, weights: np.ndarray, features: np.ndarray, targets: np.ndarray) -> float:
        """インフォメーションレシオ目的関数"""
        try:
            # 重み付きスコア計算
            weighted_scores = np.dot(features, weights)
            
            # ベンチマーク（等重み）
            benchmark_scores = np.mean(features, axis=1)
            
            # アクティブリターン
            active_returns = weighted_scores - benchmark_scores
            
            # インフォメーションレシオ
            mean_active = np.mean(active_returns)
            std_active = np.std(active_returns)
            
            if std_active == 0:
                return -1000.0
            
            info_ratio = mean_active / std_active
            
            return -info_ratio
            
        except Exception:
            return -1000.0
    
    def _calculate_accuracy_objective(self, weights: np.ndarray, features: np.ndarray, targets: np.ndarray) -> float:
        """精度目的関数"""
        try:
            # 重み付きスコア計算
            weighted_scores = np.dot(features, weights)
            
            # 予測（上位50%を買い推奨）
            threshold = np.median(weighted_scores)
            predictions = (weighted_scores > threshold).astype(int)
            
            # 正解（上位50%のリターン）
            target_threshold = np.median(targets)
            true_labels = (targets > target_threshold).astype(int)
            
            # 精度計算
            accuracy = accuracy_score(true_labels, predictions)
            
            return -(accuracy - 0.5)  # 0.5からの改善度
            
        except Exception:
            return -1000.0
    
    def _calculate_confidence_score(
        self, 
        optimization_result: Dict[str, Any], 
        features: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """信頼度スコア計算"""
        try:
            # 収束性
            convergence_score = 1.0 if optimization_result['convergence']['success'] else 0.0
            
            # スコア安定性
            score = optimization_result['score']
            stability_score = min(1.0, max(0.0, (score + 1) / 2))  # -1~1 を 0~1 に正規化
            
            # 重みの妥当性
            weights = list(optimization_result['weights'].values())
            weight_diversity = 1 - np.std(weights)  # 極端な偏りがないか
            
            # 総合信頼度
            confidence = (convergence_score + stability_score + weight_diversity) / 3
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    def _calculate_performance_metrics(
        self, 
        optimization_result: Dict[str, Any], 
        current_scores: Optional[Dict[str, Dict[str, float]]]
    ) -> Dict[str, float]:
        """パフォーマンスメトリクス計算"""
        try:
            metrics = {}
            
            # 最適化スコア
            metrics['optimization_score'] = optimization_result['score']
            
            # 重みの分散
            weights = list(optimization_result['weights'].values())
            metrics['weight_variance'] = np.var(weights)
            metrics['weight_entropy'] = -np.sum([w * np.log(w + 1e-10) for w in weights])
            
            # 前回からの変化
            if hasattr(self, '_current_weights'):
                prev_weights = list(self._current_weights.values())
                current_weights = list(optimization_result['weights'].values())
                weight_change = np.linalg.norm(np.array(current_weights) - np.array(prev_weights))
                metrics['weight_change'] = weight_change
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Performance metrics calculation failed: {e}")
            return {}
    
    def _update_history(self, result: WeightOptimizationResult):
        """履歴更新"""
        try:
            # 重み履歴
            timestamp_str = result.timestamp.strftime('%Y%m%d_%H%M%S')
            self._weight_history[timestamp_str] = result.optimal_weights
            
            # レジーム履歴
            self._regime_history.append({
                'timestamp': result.timestamp,
                'regime': result.regime,
                'confidence': result.confidence_score
            })
            
            # パフォーマンス履歴
            self._performance_history.append({
                'timestamp': result.timestamp,
                'score': result.optimization_score,
                'metrics': result.performance_metrics
            })
            
            # 現在の重みを更新
            self._current_weights = result.optimal_weights.copy()
            self._current_regime = result.regime
            
            # 履歴サイズ制限
            max_history = 1000
            if len(self._regime_history) > max_history:
                self._regime_history = self._regime_history[-max_history:]
            if len(self._performance_history) > max_history:
                self._performance_history = self._performance_history[-max_history:]
            
        except Exception as e:
            self.logger.warning(f"History update failed: {e}")
    
    def get_current_weights(self) -> Dict[str, float]:
        """現在の重み取得"""
        return self._current_weights.copy()
    
    def get_current_regime(self) -> MarketRegime:
        """現在のレジーム取得"""
        return self._current_regime
    
    def get_weight_history(self) -> Dict[str, Dict[str, float]]:
        """重み履歴取得"""
        return self._weight_history.copy()
    
    def get_regime_history(self) -> List[Dict[str, Any]]:
        """レジーム履歴取得"""
        return self._regime_history.copy()
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """パフォーマンス履歴取得"""
        return self._performance_history.copy()
    
    def apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """重み制約適用"""
        try:
            # 制約適用
            constrained_weights = {}
            total_weight = 0.0
            
            for key, weight in weights.items():
                constrained_weight = max(
                    self.config.constraints.min_weight,
                    min(self.config.constraints.max_weight, weight)
                )
                constrained_weights[key] = constrained_weight
                total_weight += constrained_weight
            
            # 正規化
            if total_weight > 0:
                for key in constrained_weights:
                    constrained_weights[key] /= total_weight
            
            return constrained_weights
            
        except Exception as e:
            self.logger.warning(f"Weight constraint application failed: {e}")
            return weights
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        return {
            'config': {
                'method': self.config.method.value,
                'objective_function': self.config.objective_function,
                'lookback_period': self.config.lookback_period,
                'rebalance_frequency': self.config.rebalance_frequency
            },
            'current_state': {
                'regime': self._current_regime.value,
                'weights': self._current_weights,
                'history_length': len(self._weight_history)
            },
            'performance': {
                'total_optimizations': len(self._performance_history),
                'recent_scores': [h['score'] for h in self._performance_history[-10:]]
            }
        }
