"""
Module: Metric Weight Optimizer
File: metric_weight_optimizer.py
Description: 
  重み自動更新システム - 分析結果に基づく重み最適化
  重要指標分析の結果を基に戦略スコアリングシステムの重みを自動調整
  2-1-2「重要指標選定システム」の重み最適化コンポーネント

Author: imega
Created: 2025-07-10
Modified: 2025-07-10

Dependencies:
  - json
  - numpy
  - pandas
  - scipy
  - config.metric_importance_analyzer
  - config.strategy_scoring_model
"""

import json
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# 最適化ライブラリ
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Weight optimization will be limited.")

# 内部モジュール
try:
    from .metric_importance_analyzer import MetricImportanceAnalyzer
    from .metric_selection_config import MetricSelectionConfig
    from .strategy_scoring_model import ScoreWeights, StrategyScoreCalculator
except ImportError:
    # 直接実行時の対応
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.metric_importance_analyzer import MetricImportanceAnalyzer
    from config.metric_selection_config import MetricSelectionConfig
    from config.strategy_scoring_model import ScoreWeights, StrategyScoreCalculator

# ロガーの設定
logger = logging.getLogger(__name__)

@dataclass
class WeightOptimizationResult:
    """重み最適化結果"""
    original_weights: Dict[str, float]
    optimized_weights: Dict[str, float]
    improvement_score: float
    optimization_method: str
    validation_metrics: Dict[str, float]
    timestamp: str
    success: bool
    error_message: Optional[str] = None

class MetricWeightOptimizer:
    """
    重み自動更新システム
    
    重要指標分析の結果を基に、戦略スコアリングシステムで使用する
    指標の重みを自動的に最適化するシステム
    """
    
    def __init__(self, 
                 config: Optional[MetricSelectionConfig] = None,
                 base_dir: Optional[str] = None):
        """
        初期化
        
        Args:
            config: 設定インスタンス
            base_dir: 基底ディレクトリ
        """
        self.config = config if config is not None else MetricSelectionConfig()
        
        # パス設定
        if base_dir is None:
            project_root = Path(__file__).parent.parent
            base_dir = project_root / "logs" / "metric_weight_optimization"
        
        self.base_dir = Path(base_dir)
        self.weights_dir = self.base_dir / "weights"
        self.results_dir = self.base_dir / "results"
        
        # ディレクトリ作成
        for dir_path in [self.weights_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 分析エンジンの初期化
        self.analyzer = MetricImportanceAnalyzer(config)
        
        logger.info(f"MetricWeightOptimizer initialized: {self.base_dir}")
    
    def optimize_weights(self, 
                        importance_results: Optional[Dict[str, Any]] = None,
                        current_weights: Optional[Dict[str, float]] = None,
                        optimization_method: str = "importance_based") -> WeightOptimizationResult:
        """
        重みの最適化
        
        Args:
            importance_results: 重要度分析結果
            current_weights: 現在の重み設定
            optimization_method: 最適化手法
            
        Returns:
            WeightOptimizationResult: 最適化結果
        """
        logger.info(f"Starting weight optimization with method: {optimization_method}")
        
        try:
            # 重要度分析結果の取得
            if importance_results is None:
                importance_results = self.analyzer.analyze_metric_importance()
            
            if "error" in importance_results:
                return WeightOptimizationResult(
                    original_weights={},
                    optimized_weights={},
                    improvement_score=0.0,
                    optimization_method=optimization_method,
                    validation_metrics={},
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error_message=f"Analysis failed: {importance_results['error']}"
                )
            
            # 現在の重み設定の取得
            if current_weights is None:
                current_weights = self._load_current_weights()
            
            # 最適化手法に応じた処理
            if optimization_method == "importance_based":
                optimized_weights = self._optimize_importance_based(importance_results, current_weights)
            elif optimization_method == "correlation_weighted":
                optimized_weights = self._optimize_correlation_weighted(importance_results, current_weights)
            elif optimization_method == "balanced_approach":
                optimized_weights = self._optimize_balanced_approach(importance_results, current_weights)
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            # 最適化結果の評価
            improvement_score = self._evaluate_improvement(current_weights, optimized_weights, importance_results)
            
            # 検証指標の計算
            validation_metrics = self._calculate_validation_metrics(
                current_weights, optimized_weights, importance_results
            )
            
            # 結果の作成
            result = WeightOptimizationResult(
                original_weights=current_weights.copy(),
                optimized_weights=optimized_weights,
                improvement_score=improvement_score,
                optimization_method=optimization_method,
                validation_metrics=validation_metrics,
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
            # 結果の保存
            self._save_optimization_result(result)
            
            logger.info(f"Weight optimization completed. Improvement: {improvement_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            return WeightOptimizationResult(
                original_weights=current_weights or {},
                optimized_weights={},
                improvement_score=0.0,
                optimization_method=optimization_method,
                validation_metrics={},
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
    
    def _load_current_weights(self) -> Dict[str, float]:
        """現在の重み設定を読み込み"""
        try:
            # ScoreWeightsのデフォルト値を使用
            score_weights = ScoreWeights()
            return {
                "performance": score_weights.performance,
                "stability": score_weights.stability,
                "risk_adjusted": score_weights.risk_adjusted,
                "trend_adaptation": score_weights.trend_adaptation,
                "reliability": score_weights.reliability
            }
        except Exception as e:
            logger.warning(f"Failed to load current weights: {e}")
            # フォールバックのデフォルト値
            return {
                "performance": 0.35,
                "stability": 0.25,
                "risk_adjusted": 0.20,
                "trend_adaptation": 0.15,
                "reliability": 0.05
            }
    
    def _optimize_importance_based(self, 
                                 importance_results: Dict[str, Any],
                                 current_weights: Dict[str, float]) -> Dict[str, float]:
        """重要度ベースの重み最適化"""
        try:
            integrated_importance = importance_results.get("integrated_importance", {})
            
            if not integrated_importance:
                logger.warning("No integrated importance data available")
                return current_weights.copy()
            
            # 重要度スコアに基づく重み計算
            weight_adjustments = {}
            
            # 各カテゴリの重要指標を特定
            performance_metrics = ["sharpe_ratio", "total_return", "profit_factor"]
            stability_metrics = ["win_rate", "consistency_ratio", "max_consecutive_losses"]
            risk_metrics = ["max_drawdown", "volatility", "downside_deviation", "var_95"]
            trend_metrics = ["recovery_factor", "tail_ratio"]
            reliability_metrics = ["expectancy", "avg_holding_period"]
            
            # カテゴリ別重要度の計算
            category_importance = {
                "performance": self._calculate_category_importance(integrated_importance, performance_metrics),
                "stability": self._calculate_category_importance(integrated_importance, stability_metrics),
                "risk_adjusted": self._calculate_category_importance(integrated_importance, risk_metrics),
                "trend_adaptation": self._calculate_category_importance(integrated_importance, trend_metrics),
                "reliability": self._calculate_category_importance(integrated_importance, reliability_metrics)
            }
            
            # 重要度に基づく重みの調整
            total_importance = sum(category_importance.values())
            if total_importance > 0:
                # 重要度比率で重みを再配分
                optimized_weights = {}
                for category, importance in category_importance.items():
                    base_weight = current_weights.get(category, 0.2)
                    importance_ratio = importance / total_importance
                    
                    # 重要度による調整（現在の重みの±50%以内）
                    adjustment_factor = 0.5 + importance_ratio
                    new_weight = base_weight * adjustment_factor
                    optimized_weights[category] = new_weight
                
                # 重みの正規化
                optimized_weights = self._normalize_weights(optimized_weights)
            else:
                optimized_weights = current_weights.copy()
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Importance-based optimization error: {e}")
            return current_weights.copy()
    
    def _optimize_correlation_weighted(self, 
                                     importance_results: Dict[str, Any],
                                     current_weights: Dict[str, float]) -> Dict[str, float]:
        """相関重み付きの最適化"""
        try:
            correlation_results = importance_results.get("results", {}).get("correlation_analysis", {})
            
            if not correlation_results or "importance_scores" not in correlation_results:
                logger.warning("No correlation results available")
                return current_weights.copy()
            
            importance_scores = correlation_results["importance_scores"]
            
            # 相関スコアに基づく重み調整
            correlation_weights = {}
            for metric, score_data in importance_scores.items():
                correlation_strength = score_data.get("abs_correlation", 0.0)
                
                # メトリックをカテゴリにマッピング
                category = self._map_metric_to_category(metric)
                if category not in correlation_weights:
                    correlation_weights[category] = []
                correlation_weights[category].append(correlation_strength)
            
            # カテゴリ別平均相関の計算
            optimized_weights = {}
            for category in current_weights.keys():
                if category in correlation_weights:
                    avg_correlation = np.mean(correlation_weights[category])
                    base_weight = current_weights[category]
                    
                    # 相関強度による調整
                    adjustment = 1.0 + (avg_correlation - 0.5)  # 0.5を基準に調整
                    optimized_weights[category] = base_weight * max(0.1, adjustment)
                else:
                    optimized_weights[category] = current_weights[category]
            
            # 重みの正規化
            optimized_weights = self._normalize_weights(optimized_weights)
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Correlation-weighted optimization error: {e}")
            return current_weights.copy()
    
    def _optimize_balanced_approach(self, 
                                  importance_results: Dict[str, Any],
                                  current_weights: Dict[str, float]) -> Dict[str, float]:
        """バランス重視のアプローチ"""
        try:
            # 重要度ベースとコリレーションベースの結果を組み合わせ
            importance_weights = self._optimize_importance_based(importance_results, current_weights)
            correlation_weights = self._optimize_correlation_weighted(importance_results, current_weights)
            
            # 2つの結果を平均
            optimized_weights = {}
            for category in current_weights.keys():
                importance_weight = importance_weights.get(category, current_weights[category])
                correlation_weight = correlation_weights.get(category, current_weights[category])
                
                # 重み付き平均（重要度を重視）
                optimized_weights[category] = 0.6 * importance_weight + 0.4 * correlation_weight
            
            # 重みの正規化
            optimized_weights = self._normalize_weights(optimized_weights)
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Balanced approach optimization error: {e}")
            return current_weights.copy()
    
    def _calculate_category_importance(self, 
                                     integrated_importance: Dict[str, Dict[str, Any]],
                                     metrics: List[str]) -> float:
        """カテゴリの重要度を計算"""
        try:
            scores = []
            for metric in metrics:
                if metric in integrated_importance:
                    score = integrated_importance[metric].get("final_importance_score", 0.0)
                    scores.append(score)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception:
            return 0.0
    
    def _map_metric_to_category(self, metric: str) -> str:
        """メトリックをカテゴリにマッピング"""
        metric_lower = metric.lower()
        
        if any(term in metric_lower for term in ["sharpe", "return", "profit"]):
            return "performance"
        elif any(term in metric_lower for term in ["win_rate", "consistency", "consecutive"]):
            return "stability"
        elif any(term in metric_lower for term in ["drawdown", "volatility", "var", "deviation"]):
            return "risk_adjusted"
        elif any(term in metric_lower for term in ["recovery", "tail", "trend"]):
            return "trend_adaptation"
        else:
            return "reliability"
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """重みの正規化（合計を1.0にする）"""
        try:
            total = sum(weights.values())
            if total > 0:
                return {key: value / total for key, value in weights.items()}
            else:
                # すべて0の場合は均等分割
                num_categories = len(weights)
                return {key: 1.0 / num_categories for key in weights.keys()}
        except Exception:
            return weights
    
    def _evaluate_improvement(self, 
                            original_weights: Dict[str, float],
                            optimized_weights: Dict[str, float],
                            importance_results: Dict[str, Any]) -> float:
        """最適化の改善度を評価"""
        try:
            # 重要度との整合性を評価
            integrated_importance = importance_results.get("integrated_importance", {})
            
            if not integrated_importance:
                return 0.0
            
            # 各カテゴリの重要度と重みの整合性を計算
            original_alignment = self._calculate_weight_alignment(original_weights, integrated_importance)
            optimized_alignment = self._calculate_weight_alignment(optimized_weights, integrated_importance)
            
            # 改善スコア（-1.0 から 1.0）
            improvement = optimized_alignment - original_alignment
            
            return float(improvement)
            
        except Exception as e:
            logger.error(f"Improvement evaluation error: {e}")
            return 0.0
    
    def _calculate_weight_alignment(self, 
                                  weights: Dict[str, float],
                                  integrated_importance: Dict[str, Dict[str, Any]]) -> float:
        """重みと重要度の整合性を計算"""
        try:
            # カテゴリ別重要度の計算
            performance_metrics = ["sharpe_ratio", "total_return", "profit_factor"]
            stability_metrics = ["win_rate", "consistency_ratio", "max_consecutive_losses"]
            risk_metrics = ["max_drawdown", "volatility", "downside_deviation", "var_95"]
            trend_metrics = ["recovery_factor", "tail_ratio"]
            reliability_metrics = ["expectancy", "avg_holding_period"]
            
            category_importance = {
                "performance": self._calculate_category_importance(integrated_importance, performance_metrics),
                "stability": self._calculate_category_importance(integrated_importance, stability_metrics),
                "risk_adjusted": self._calculate_category_importance(integrated_importance, risk_metrics),
                "trend_adaptation": self._calculate_category_importance(integrated_importance, trend_metrics),
                "reliability": self._calculate_category_importance(integrated_importance, reliability_metrics)
            }
            
            # 重要度の正規化
            total_importance = sum(category_importance.values())
            if total_importance > 0:
                normalized_importance = {k: v / total_importance for k, v in category_importance.items()}
            else:
                normalized_importance = {k: 0.2 for k in category_importance.keys()}
            
            # 重みとの相関を計算
            alignment_scores = []
            for category in weights.keys():
                weight = weights[category]
                importance = normalized_importance.get(category, 0.0)
                # 重みと重要度の差の絶対値（小さいほど良い）
                alignment_scores.append(1.0 - abs(weight - importance))
            
            return np.mean(alignment_scores) if alignment_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_validation_metrics(self, 
                                    original_weights: Dict[str, float],
                                    optimized_weights: Dict[str, float],
                                    importance_results: Dict[str, Any]) -> Dict[str, float]:
        """検証指標の計算"""
        try:
            metrics = {}
            
            # 重みの変化量
            weight_changes = []
            for category in original_weights.keys():
                original = original_weights[category]
                optimized = optimized_weights.get(category, original)
                weight_changes.append(abs(optimized - original))
            
            metrics["avg_weight_change"] = float(np.mean(weight_changes))
            metrics["max_weight_change"] = float(np.max(weight_changes))
            
            # 重みの分散（バランス指標）
            metrics["weight_variance_original"] = float(np.var(list(original_weights.values())))
            metrics["weight_variance_optimized"] = float(np.var(list(optimized_weights.values())))
            
            # 正規化チェック
            metrics["weight_sum_original"] = float(sum(original_weights.values()))
            metrics["weight_sum_optimized"] = float(sum(optimized_weights.values()))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Validation metrics calculation error: {e}")
            return {}
    
    def _save_optimization_result(self, result: WeightOptimizationResult):
        """最適化結果の保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weight_optimization_{result.optimization_method}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # 結果をシリアライズ可能な形式に変換
            result_dict = {
                "original_weights": result.original_weights,
                "optimized_weights": result.optimized_weights,
                "improvement_score": result.improvement_score,
                "optimization_method": result.optimization_method,
                "validation_metrics": result.validation_metrics,
                "timestamp": result.timestamp,
                "success": result.success,
                "error_message": result.error_message
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            # 最新の重み設定も保存
            if result.success:
                weights_file = self.weights_dir / "latest_optimized_weights.json"
                with open(weights_file, 'w', encoding='utf-8') as f:
                    json.dump(result.optimized_weights, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Optimization result saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Result saving error: {e}")
    
    def apply_optimized_weights(self, result: WeightOptimizationResult) -> bool:
        """最適化された重みを適用"""
        try:
            if not result.success:
                logger.warning("Cannot apply weights from failed optimization")
                return False
            
            # ScoreWeightsファイルを更新
            weights_config_file = Path(__file__).parent / "scoring_weights.json"
            
            # スコアリングシステムの重み形式に変換
            scoring_weights = {
                "performance": result.optimized_weights.get("performance", 0.35),
                "stability": result.optimized_weights.get("stability", 0.25),
                "risk_adjusted": result.optimized_weights.get("risk_adjusted", 0.20),
                "trend_adaptation": result.optimized_weights.get("trend_adaptation", 0.15),
                "reliability": result.optimized_weights.get("reliability", 0.05)
            }
            
            # バックアップの作成
            if weights_config_file.exists():
                backup_file = weights_config_file.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                import shutil
                shutil.copy2(weights_config_file, backup_file)
            
            # 新しい重みを保存
            with open(weights_config_file, 'w', encoding='utf-8') as f:
                json.dump(scoring_weights, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Optimized weights applied to {weights_config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Weight application error: {e}")
            return False

# 使用例とテスト
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== 重み最適化システムテスト ===")
    
    try:
        # 最適化器の初期化
        optimizer = MetricWeightOptimizer()
        print("✓ 重み最適化器初期化完了")
        
        # 重み最適化の実行
        result = optimizer.optimize_weights(optimization_method="balanced_approach")
        
        if result.success:
            print("✓ 重み最適化完了")
            print(f"改善スコア: {result.improvement_score:.3f}")
            
            # 重みの変化を表示
            print("\n重みの変化:")
            for category in result.original_weights.keys():
                original = result.original_weights[category]
                optimized = result.optimized_weights[category]
                change = optimized - original
                print(f"  {category}: {original:.3f} → {optimized:.3f} ({change:+.3f})")
            
            # 重みの適用確認
            print("\n重み適用を実行しますか? (y/n): ", end="")
            response = input().lower().strip()
            
            if response == 'y':
                if optimizer.apply_optimized_weights(result):
                    print("✓ 重みが正常に適用されました")
                else:
                    print("✗ 重み適用に失敗しました")
            else:
                print("重み適用をスキップしました")
                
        else:
            print(f"✗ 最適化エラー: {result.error_message}")
            
    except Exception as e:
        print(f"✗ テストエラー: {e}")
    
    print("\nテスト完了")
