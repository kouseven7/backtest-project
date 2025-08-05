"""
Module: Parameter Adjuster
File: parameter_adjuster.py
Description: 
  5-2-2「トレンド判定精度の自動補正」
  パラメータ自動調整システム - トレンド判定パラメータの最適化

Author: imega
Created: 2025-07-22
Modified: 2025-07-22
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ParameterAdjuster:
    """トレンド判定パラメータの自動調整"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # 設定の読み込み
        self.optimization_window = config.get('optimization_window', 30)
        self.min_improvement_threshold = config.get('min_improvement_threshold', 0.02)
        self.max_correction_factor = config.get('max_correction_factor', 0.5)
        self.parameter_search_iterations = config.get('parameter_search_iterations', 50)
        self.enable_grid_search = config.get('enable_grid_search', True)
        self.optimization_timeout = config.get('optimization_timeout_seconds', 300)
        
        # パラメータ境界の読み込み
        self.parameter_bounds = self._load_parameter_bounds()
        
        # 最適化履歴
        self._optimization_history: Dict[str, List[Dict[str, Any]]] = {}
        
        self.logger.info("ParameterAdjuster initialized successfully")
    
    def _load_parameter_bounds(self) -> Dict[str, Any]:
        """パラメータ境界設定をロード"""
        try:
            bounds_path = Path(__file__).parent.parent / "trend_precision_config" / "parameter_bounds.json"
            
            if bounds_path.exists():
                with open(bounds_path, 'r', encoding='utf-8') as f:
                    bounds = json.load(f)
                self.logger.info("Parameter bounds loaded successfully")
                return bounds
            else:
                self.logger.warning("Parameter bounds file not found, using defaults")
                return self._get_default_parameter_bounds()
                
        except Exception as e:
            self.logger.error(f"Failed to load parameter bounds: {e}")
            return self._get_default_parameter_bounds()
    
    def _get_default_parameter_bounds(self) -> Dict[str, Any]:
        """デフォルトのパラメータ境界"""
        return {
            "parameter_bounds": {
                "sma": {
                    "short_period": [5, 20],
                    "medium_period": [10, 50],
                    "long_period": [20, 100]
                },
                "macd": {
                    "short_window": [8, 20],
                    "long_window": [20, 40],
                    "signal_window": [5, 15]
                }
            },
            "optimization_constraints": {
                "max_parameter_change_ratio": 0.3,
                "min_data_points_required": 30
            }
        }
    
    def get_optimized_parameters(self,
                               strategy_name: str,
                               method: str,
                               ticker: str,
                               precision_tracker: Any = None) -> Dict[str, Any]:
        """最適化されたパラメータを取得"""
        
        try:
            # 現在のパフォーマンス分析
            current_performance = self._analyze_current_performance(
                strategy_name, method, ticker, precision_tracker
            )
            
            self.logger.info(f"Current performance for {strategy_name}_{method}_{ticker}: {current_performance.get('accuracy', 0.0):.3f}")
            
            # パフォーマンスが良好な場合は最適化をスキップ
            if current_performance.get('accuracy', 0.0) >= 0.75:
                self.logger.info("Performance already good, skipping optimization")
                return {}
            
            # パラメータ探索空間の定義
            search_space = self._define_parameter_search_space(method, strategy_name)
            
            if not search_space:
                self.logger.warning(f"No search space defined for method: {method}")
                return {}
            
            # 最適化実行
            optimized_params = self._optimize_parameters(
                strategy_name, method, ticker, search_space, precision_tracker
            )
            
            if optimized_params:
                self.logger.info(f"Optimization completed for {strategy_name}_{method}_{ticker}")
                self._record_optimization_result(strategy_name, method, ticker, optimized_params, current_performance)
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"Failed to get optimized parameters: {e}")
            return {}
    
    def _analyze_current_performance(self,
                                   strategy_name: str,
                                   method: str,
                                   ticker: str,
                                   precision_tracker: Any) -> Dict[str, float]:
        """現在のパフォーマンスを分析"""
        
        try:
            if precision_tracker is None:
                return {'accuracy': 0.5, 'sample_size': 0}
            
            # 精度追跡器から性能データを取得
            performance = precision_tracker.calculate_method_accuracy(
                method=method,
                strategy_name=strategy_name,
                ticker=ticker,
                days=self.optimization_window
            )
            
            return {
                'accuracy': performance.get('average_accuracy', 0.5),
                'sample_size': performance.get('total_predictions', 0),
                'confidence_correlation': performance.get('confidence_correlation', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze current performance: {e}")
            return {'accuracy': 0.5, 'sample_size': 0}
    
    def _define_parameter_search_space(self, method: str, strategy_name: str) -> Dict[str, Tuple[Any, ...]]:
        """パラメータ探索空間を定義"""
        
        try:
            search_space = {}
            
            # メソッド別の基本パラメータ
            method_bounds = self.parameter_bounds.get('parameter_bounds', {}).get(method, {})
            
            for param_name, bounds in method_bounds.items():
                if isinstance(bounds, list) and len(bounds) == 2:
                    if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                        # 整数パラメータ
                        search_space[param_name] = tuple(range(bounds[0], bounds[1] + 1, max(1, (bounds[1] - bounds[0]) // 10)))
                    else:
                        # 浮動小数点パラメータ
                        step = (bounds[1] - bounds[0]) / 10
                        values = []
                        current = bounds[0]
                        while current <= bounds[1]:
                            values.append(round(current, 3))
                            current += step
                        search_space[param_name] = tuple(values)
            
            # 戦略固有のパラメータ
            strategy_bounds = self.parameter_bounds.get('strategy_specific_bounds', {}).get(strategy_name, {})
            
            for param_name, bounds in strategy_bounds.items():
                if isinstance(bounds, list) and len(bounds) == 2:
                    if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                        search_space[param_name] = tuple(range(bounds[0], bounds[1] + 1, max(1, (bounds[1] - bounds[0]) // 5)))
                    else:
                        step = (bounds[1] - bounds[0]) / 5
                        values = []
                        current = bounds[0]
                        while current <= bounds[1]:
                            values.append(round(current, 4))
                            current += step
                        search_space[param_name] = tuple(values)
            
            self.logger.info(f"Defined search space for {method}: {list(search_space.keys())}")
            return search_space
            
        except Exception as e:
            self.logger.error(f"Failed to define parameter search space: {e}")
            return {}
    
    def _optimize_parameters(self,
                           strategy_name: str,
                           method: str,
                           ticker: str,
                           search_space: Dict[str, Tuple[Any, ...]],
                           precision_tracker: Any) -> Dict[str, Any]:
        """パラメータ最適化の実行"""
        
        try:
            best_params = {}
            best_score = 0.0
            evaluated_combinations = 0
            start_time = time.time()
            
            if self.enable_grid_search:
                # グリッドサーチによる最適化
                param_names = list(search_space.keys())
                param_values = [search_space[name] for name in param_names]
                
                # 組み合わせ数を制限
                max_combinations = min(self.parameter_search_iterations, 1000)
                all_combinations = list(itertools.product(*param_values))
                
                if len(all_combinations) > max_combinations:
                    # ランダムサンプリング
                    np.random.seed(42)  # 再現性のため
                    sampled_indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
                    combinations = [all_combinations[i] for i in sampled_indices]
                else:
                    combinations = all_combinations
                
                self.logger.info(f"Evaluating {len(combinations)} parameter combinations")
                
                for combination in combinations:
                    # タイムアウトチェック
                    if time.time() - start_time > self.optimization_timeout:
                        self.logger.warning("Optimization timeout reached")
                        break
                    
                    param_dict = dict(zip(param_names, combination))
                    
                    score = self._evaluate_parameter_set(
                        strategy_name, method, ticker, param_dict, precision_tracker
                    )
                    
                    evaluated_combinations += 1
                    
                    if score > best_score:
                        best_score = score
                        best_params = param_dict.copy()
            
            else:
                # ランダムサーチによる最適化
                for _ in range(self.parameter_search_iterations):
                    # タイムアウトチェック
                    if time.time() - start_time > self.optimization_timeout:
                        self.logger.warning("Optimization timeout reached")
                        break
                    
                    random_params = {}
                    for param_name, values in search_space.items():
                        random_params[param_name] = np.random.choice(values)
                    
                    score = self._evaluate_parameter_set(
                        strategy_name, method, ticker, random_params, precision_tracker
                    )
                    
                    evaluated_combinations += 1
                    
                    if score > best_score:
                        best_score = score
                        best_params = random_params.copy()
            
            elapsed_time = time.time() - start_time
            improvement = best_score - 0.5  # ベースラインとの比較
            
            self.logger.info(f"Optimization completed: {evaluated_combinations} combinations, "
                           f"best score: {best_score:.3f}, improvement: {improvement:.3f}, "
                           f"time: {elapsed_time:.1f}s")
            
            # 改善がしきい値を超えた場合のみ返す
            if improvement > self.min_improvement_threshold:
                return best_params
            else:
                self.logger.info("No significant improvement found")
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to optimize parameters: {e}")
            return {}
    
    def _evaluate_parameter_set(self,
                              strategy_name: str,
                              method: str,
                              ticker: str,
                              params: Dict[str, Any],
                              precision_tracker: Any) -> float:
        """パラメータセットを評価"""
        
        try:
            # シミュレートされた精度スコア（実際の実装では、新しいパラメータでトレンド判定を実行し精度を測定）
            
            # パラメータの妥当性チェック
            if not self._validate_parameters(method, params):
                return 0.0
            
            # ベースラインスコア（現在の実装では簡易的な評価）
            base_score = 0.5
            
            # パラメータの組み合わせに基づくスコア調整
            score_adjustment = 0.0
            
            if method == "sma":
                # SMAパラメータの評価
                short_period = params.get('short_period', 10)
                medium_period = params.get('medium_period', 20)
                long_period = params.get('long_period', 50)
                
                # パラメータの関係性チェック
                if short_period < medium_period < long_period:
                    score_adjustment += 0.1
                
                # 期間の適度なばらつきを評価
                if medium_period / short_period > 1.5 and long_period / medium_period > 1.5:
                    score_adjustment += 0.1
                    
            elif method == "macd":
                # MACDパラメータの評価
                short_window = params.get('short_window', 12)
                long_window = params.get('long_window', 26)
                signal_window = params.get('signal_window', 9)
                
                # パラメータの関係性チェック
                if short_window < long_window:
                    score_adjustment += 0.1
                
                # 標準的な比率に近いかチェック
                ratio = long_window / short_window
                if 1.8 <= ratio <= 2.5:
                    score_adjustment += 0.05
            
            # ランダムノイズを追加（実際の性能のばらつきをシミュレート）
            np.random.seed(hash(f"{strategy_name}_{method}_{ticker}_{str(params)}") % (2**32))
            noise = np.random.normal(0, 0.05)
            
            final_score = base_score + score_adjustment + noise
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate parameter set: {e}")
            return 0.0
    
    def _validate_parameters(self, method: str, params: Dict[str, Any]) -> bool:
        """パラメータの妥当性をチェック"""
        
        try:
            constraints = self.parameter_bounds.get('optimization_constraints', {})
            
            if method == "sma":
                short = params.get('short_period', 0)
                medium = params.get('medium_period', 0)
                long = params.get('long_period', 0)
                
                # 期間の順序チェック
                if not (short < medium < long):
                    return False
                
                # 最小差分チェック
                if medium - short < 3 or long - medium < 5:
                    return False
                    
            elif method == "macd":
                short = params.get('short_window', 0)
                long = params.get('long_window', 0)
                signal = params.get('signal_window', 0)
                
                # ウィンドウサイズの順序チェック
                if short >= long:
                    return False
                
                # 最小差分チェック
                if long - short < 5:
                    return False
                
                # シグナル期間の妥当性
                if signal <= 0 or signal >= short:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}")
            return False
    
    def _record_optimization_result(self,
                                  strategy_name: str,
                                  method: str,
                                  ticker: str,
                                  optimized_params: Dict[str, Any],
                                  previous_performance: Dict[str, float]):
        """最適化結果を記録"""
        
        try:
            key = f"{strategy_name}_{method}_{ticker}"
            
            if key not in self._optimization_history:
                self._optimization_history[key] = []
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'optimized_parameters': optimized_params,
                'previous_performance': previous_performance,
                'optimization_trigger': 'automatic',
                'method': method,
                'strategy_name': strategy_name,
                'ticker': ticker
            }
            
            self._optimization_history[key].append(result)
            
            # 履歴の制限（最新20件まで）
            if len(self._optimization_history[key]) > 20:
                self._optimization_history[key] = self._optimization_history[key][-20:]
            
            self.logger.debug(f"Recorded optimization result for {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to record optimization result: {e}")
    
    def get_optimization_history(self, strategy_name: str = None, method: str = None) -> Dict[str, Any]:
        """最適化履歴を取得"""
        
        try:
            if strategy_name and method:
                filtered_history = {
                    key: history for key, history in self._optimization_history.items()
                    if key.startswith(f"{strategy_name}_{method}")
                }
            elif strategy_name:
                filtered_history = {
                    key: history for key, history in self._optimization_history.items()
                    if key.startswith(strategy_name)
                }
            else:
                filtered_history = self._optimization_history.copy()
            
            return {
                'total_optimizations': sum(len(history) for history in filtered_history.values()),
                'optimized_combinations': len(filtered_history),
                'recent_optimizations': {
                    key: history[-5:] if history else []
                    for key, history in filtered_history.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization history: {e}")
            return {}
    
    def suggest_parameter_adjustments(self,
                                    strategy_name: str,
                                    method: str,
                                    current_performance: Dict[str, float]) -> Dict[str, Any]:
        """パラメータ調整を提案"""
        
        try:
            suggestions = {
                'recommended_adjustments': {},
                'reasoning': [],
                'confidence_level': 'medium'
            }
            
            accuracy = current_performance.get('accuracy', 0.5)
            sample_size = current_performance.get('sample_size', 0)
            
            if sample_size < 10:
                suggestions['reasoning'].append("サンプルサイズが小さいため、調整を控えることを推奨")
                suggestions['confidence_level'] = 'low'
                return suggestions
            
            if accuracy < 0.4:
                suggestions['reasoning'].append("精度が低いため、積極的なパラメータ調整を推奨")
                suggestions['confidence_level'] = 'high'
                
                if method == "sma":
                    suggestions['recommended_adjustments'] = {
                        'short_period': 'reduce',
                        'long_period': 'increase'
                    }
                    suggestions['reasoning'].append("SMA期間の差を拡大して感度を向上")
                    
                elif method == "macd":
                    suggestions['recommended_adjustments'] = {
                        'signal_window': 'reduce'
                    }
                    suggestions['reasoning'].append("MACDシグナル期間を短縮して反応速度を向上")
            
            elif accuracy < 0.6:
                suggestions['reasoning'].append("精度改善の余地があるため、微調整を推奨")
                suggestions['confidence_level'] = 'medium'
                
                # より細かい調整提案
                if method == "sma":
                    suggestions['recommended_adjustments'] = {
                        'medium_period': 'fine_tune'
                    }
                    
            else:
                suggestions['reasoning'].append("精度が良好なため、現在のパラメータを維持")
                suggestions['confidence_level'] = 'low'
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to suggest parameter adjustments: {e}")
            return {'recommended_adjustments': {}, 'reasoning': ['エラーが発生しました'], 'confidence_level': 'low'}

if __name__ == "__main__":
    # テスト用コード
    print("ParameterAdjuster モジュールが正常にロードされました")
