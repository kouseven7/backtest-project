"""
Module: Execution Result Aggregator
File: execution_result_aggregator.py
Description: 
  4-1-2「複合戦略実行フロー設計・実装」- Aggregation Component
  複数戦略の実行結果を統合・集約する
  重み付き統合、信頼度調整、外れ値処理

Author: imega
Created: 2025-01-28
Modified: 2025-01-28

Dependencies:
  - config.strategy_execution_coordinator
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import warnings

# 統計・数値計算ライブラリ
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some statistical functions will be limited.")

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ロガーの設定
logger = logging.getLogger(__name__)

class AggregationMethod(Enum):
    """集約手法"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    MAX_CONFIDENCE = "max_confidence"
    ENSEMBLE = "ensemble"

class OutlierHandling(Enum):
    """外れ値処理"""
    NONE = "none"
    CAP = "cap"
    REMOVE = "remove"
    WINSORIZE = "winsorize"

@dataclass
class StrategyResult:
    """戦略結果"""
    strategy_name: str
    signals: Dict[str, Any]
    weights: Dict[str, float]
    confidence: float
    execution_time: float
    success: bool
    metadata: Optional[Dict[str, Any]] = None
    
    def get_signal_strength(self) -> float:
        """シグナル強度を取得"""
        if not self.signals:
            return 0.0
            
        strength_values = []
        for signal_data in self.signals.values():
            if isinstance(signal_data, dict) and "strength" in signal_data:
                strength_values.append(signal_data["strength"])
            elif isinstance(signal_data, (int, float)):
                strength_values.append(abs(float(signal_data)))
                
        return np.mean(strength_values) if strength_values else 0.0

@dataclass
class AggregatedResult:
    """集約結果"""
    aggregation_id: str
    final_signals: Dict[str, Any]
    final_weights: Dict[str, float]
    overall_confidence: float
    contributing_strategies: List[str]
    aggregation_metadata: Dict[str, Any]
    execution_summary: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

class ConfidenceCalculator:
    """信頼度計算機"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ConfidenceCalculator")
        
    def calculate_strategy_confidence(self, result: StrategyResult) -> float:
        """戦略別信頼度計算"""
        base_confidence = result.confidence
        
        # 実行時間による調整
        time_factor = self._get_time_factor(result.execution_time)
        
        # シグナル強度による調整
        signal_strength = result.get_signal_strength()
        strength_factor = min(1.0, signal_strength)
        
        # 成功率による調整
        success_factor = 1.0 if result.success else 0.3
        
        adjusted_confidence = (
            base_confidence * 0.5 +
            time_factor * 0.2 +
            strength_factor * 0.2 +
            success_factor * 0.1
        )
        
        return max(0.0, min(1.0, adjusted_confidence))
        
    def _get_time_factor(self, execution_time: float, 
                        optimal_time: float = 30.0) -> float:
        """実行時間による信頼度ファクター"""
        if execution_time <= optimal_time:
            return 1.0
        elif execution_time <= optimal_time * 2:
            return 0.8
        elif execution_time <= optimal_time * 4:
            return 0.6
        else:
            return 0.4
            
    def calculate_consensus_confidence(self, strategies: List[StrategyResult]) -> float:
        """コンセンサス信頼度計算"""
        if not strategies:
            return 0.0
            
        # 戦略間の一致度を計算
        signal_agreements = self._calculate_signal_agreements(strategies)
        weight_consistency = self._calculate_weight_consistency(strategies)
        
        # 平均信頼度
        avg_confidence = np.mean([s.confidence for s in strategies])
        
        # コンセンサススコア
        consensus_score = (
            signal_agreements * 0.4 +
            weight_consistency * 0.3 +
            avg_confidence * 0.3
        )
        
        return max(0.0, min(1.0, consensus_score))
        
    def _calculate_signal_agreements(self, strategies: List[StrategyResult]) -> float:
        """シグナル一致度計算"""
        if len(strategies) < 2:
            return 1.0
            
        # 各戦略のシグナル方向を取得
        signal_directions = []
        for strategy in strategies:
            direction = 0  # neutral
            for signal_data in strategy.signals.values():
                if isinstance(signal_data, dict):
                    action = signal_data.get("action", "hold")
                    if action in ["buy", "entry_long"]:
                        direction = 1
                    elif action in ["sell", "entry_short"]:
                        direction = -1
                        
            signal_directions.append(direction)
            
        # 方向の一致度を計算
        if not signal_directions:
            return 0.5
            
        unique_directions = set(signal_directions)
        if len(unique_directions) == 1:
            return 1.0
        elif len(unique_directions) == 2 and 0 in unique_directions:
            return 0.7
        else:
            return 0.3
            
    def _calculate_weight_consistency(self, strategies: List[StrategyResult]) -> float:
        """重み一貫性計算"""
        if len(strategies) < 2:
            return 1.0
            
        # 各戦略の重みベクトルを収集
        all_instruments = set()
        for strategy in strategies:
            all_instruments.update(strategy.weights.keys())
            
        weight_vectors = []
        for strategy in strategies:
            vector = [strategy.weights.get(instrument, 0.0) 
                     for instrument in sorted(all_instruments)]
            weight_vectors.append(vector)
            
        if not weight_vectors or len(weight_vectors[0]) == 0:
            return 0.5
            
        # 重みベクトル間の相関計算
        if len(weight_vectors) >= 2:
            correlations = []
            for i in range(len(weight_vectors)):
                for j in range(i + 1, len(weight_vectors)):
                    corr = np.corrcoef(weight_vectors[i], weight_vectors[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                        
            return np.mean(correlations) if correlations else 0.5
        else:
            return 1.0

class OutlierDetector:
    """外れ値検出器"""
    
    def __init__(self, method: str = "iqr"):
        self.method = method
        self.logger = logging.getLogger(f"{__name__}.OutlierDetector")
        
    def detect_outliers(self, values: List[float], 
                       threshold: float = 1.5) -> List[bool]:
        """外れ値検出"""
        if len(values) < 3:
            return [False] * len(values)
            
        values_array = np.array(values)
        
        if self.method == "iqr":
            return self._detect_iqr_outliers(values_array, threshold)
        elif self.method == "zscore":
            return self._detect_zscore_outliers(values_array, threshold)
        elif self.method == "modified_zscore":
            return self._detect_modified_zscore_outliers(values_array, threshold)
        else:
            return [False] * len(values)
            
    def _detect_iqr_outliers(self, values: np.ndarray, 
                           threshold: float) -> List[bool]:
        """IQR法による外れ値検出"""
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        return [(v < lower_bound or v > upper_bound) for v in values]
        
    def _detect_zscore_outliers(self, values: np.ndarray, 
                              threshold: float) -> List[bool]:
        """Z-score法による外れ値検出"""
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return [False] * len(values)
            
        z_scores = np.abs((values - mean_val) / std_val)
        return [z > threshold for z in z_scores]
        
    def _detect_modified_zscore_outliers(self, values: np.ndarray, 
                                       threshold: float) -> List[bool]:
        """修正Z-score法による外れ値検出"""
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))
        
        if mad == 0:
            return [False] * len(values)
            
        modified_z_scores = 0.6745 * (values - median_val) / mad
        return [abs(z) > threshold for z in modified_z_scores]
        
    def handle_outliers(self, values: List[float], 
                       outlier_mask: List[bool],
                       handling: OutlierHandling) -> List[float]:
        """外れ値処理"""
        if handling == OutlierHandling.NONE:
            return values
            
        values_array = np.array(values)
        result = values_array.copy()
        
        if handling == OutlierHandling.REMOVE:
            # 外れ値を除去（NaNで置換）
            result[outlier_mask] = np.nan
        elif handling == OutlierHandling.CAP:
            # 外れ値を上下限でキャップ
            non_outlier_values = values_array[~np.array(outlier_mask)]
            if len(non_outlier_values) > 0:
                lower_cap = np.percentile(non_outlier_values, 5)
                upper_cap = np.percentile(non_outlier_values, 95)
                result = np.clip(result, lower_cap, upper_cap)
        elif handling == OutlierHandling.WINSORIZE:
            # ウィンザー化
            lower_percentile = np.percentile(values_array, 10)
            upper_percentile = np.percentile(values_array, 90)
            result = np.clip(result, lower_percentile, upper_percentile)
            
        return result.tolist()

class WeightAggregator:
    """重み集約器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.WeightAggregator")
        self.outlier_detector = OutlierDetector()
        
    def aggregate_weights(self, strategies: List[StrategyResult],
                         method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
                         outlier_handling: OutlierHandling = OutlierHandling.CAP) -> Dict[str, float]:
        """重みの集約"""
        if not strategies:
            return {}
            
        # 全楽器の収集
        all_instruments = set()
        for strategy in strategies:
            all_instruments.update(strategy.weights.keys())
            
        aggregated_weights = {}
        
        for instrument in all_instruments:
            instrument_weights = []
            strategy_confidences = []
            
            for strategy in strategies:
                weight = strategy.weights.get(instrument, 0.0)
                instrument_weights.append(weight)
                strategy_confidences.append(strategy.confidence)
                
            # 外れ値処理
            if outlier_handling != OutlierHandling.NONE:
                outliers = self.outlier_detector.detect_outliers(instrument_weights)
                instrument_weights = self.outlier_detector.handle_outliers(
                    instrument_weights, outliers, outlier_handling
                )
                
            # 集約
            if method == AggregationMethod.SIMPLE_AVERAGE:
                aggregated_weights[instrument] = np.nanmean(instrument_weights)
            elif method == AggregationMethod.WEIGHTED_AVERAGE:
                weights_array = np.array(instrument_weights)
                confidences_array = np.array(strategy_confidences)
                
                # NaN値の処理
                valid_mask = ~np.isnan(weights_array)
                if valid_mask.any():
                    valid_weights = weights_array[valid_mask]
                    valid_confidences = confidences_array[valid_mask]
                    
                    if np.sum(valid_confidences) > 0:
                        aggregated_weights[instrument] = np.average(
                            valid_weights, weights=valid_confidences
                        )
                    else:
                        aggregated_weights[instrument] = np.mean(valid_weights)
                else:
                    aggregated_weights[instrument] = 0.0
            elif method == AggregationMethod.MEDIAN:
                aggregated_weights[instrument] = np.nanmedian(instrument_weights)
            elif method == AggregationMethod.MAX_CONFIDENCE:
                max_conf_idx = np.nanargmax(strategy_confidences)
                aggregated_weights[instrument] = instrument_weights[max_conf_idx]
            else:
                # デフォルトは加重平均
                aggregated_weights[instrument] = np.nanmean(instrument_weights)
                
        # 重みの正規化
        total_weight = sum(abs(w) for w in aggregated_weights.values())
        if total_weight > 0:
            aggregated_weights = {
                k: v / total_weight for k, v in aggregated_weights.items()
            }
            
        return aggregated_weights

class SignalAggregator:
    """シグナル集約器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SignalAggregator")
        
    def aggregate_signals(self, strategies: List[StrategyResult],
                         confidence_weights: List[float]) -> Dict[str, Any]:
        """シグナルの集約"""
        if not strategies:
            return {}
            
        # シグナル種別ごとに集約
        signal_types = set()
        for strategy in strategies:
            if strategy.signals:
                signal_types.update(strategy.signals.keys())
                
        aggregated_signals = {}
        
        for signal_type in signal_types:
            signals_for_type = []
            weights_for_type = []
            
            for i, strategy in enumerate(strategies):
                if signal_type in strategy.signals:
                    signal_data = strategy.signals[signal_type]
                    signals_for_type.append(signal_data)
                    weights_for_type.append(confidence_weights[i])
                    
            if signals_for_type:
                aggregated_signals[signal_type] = self._aggregate_signal_type(
                    signals_for_type, weights_for_type
                )
                
        return aggregated_signals
        
    def _aggregate_signal_type(self, signals: List[Any], 
                              weights: List[float]) -> Dict[str, Any]:
        """特定シグナル種別の集約"""
        if not signals:
            return {}
            
        # アクション集約
        actions = []
        strengths = []
        confidences = []
        
        for signal in signals:
            if isinstance(signal, dict):
                actions.append(signal.get("action", "hold"))
                strengths.append(signal.get("strength", 0.0))
                confidences.append(signal.get("confidence", 0.0))
            else:
                # 単純な数値シグナル
                strength = float(signal) if signal is not None else 0.0
                strengths.append(abs(strength))
                if strength > 0.1:
                    actions.append("buy")
                elif strength < -0.1:
                    actions.append("sell")
                else:
                    actions.append("hold")
                confidences.append(abs(strength))
                
        # アクションの多数決
        action_counts = {}
        for i, action in enumerate(actions):
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += weights[i]
            
        final_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else "hold"
        
        # 強度と信頼度の加重平均
        total_weight = sum(weights)
        if total_weight > 0:
            final_strength = sum(s * w for s, w in zip(strengths, weights)) / total_weight
            final_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
        else:
            final_strength = 0.0
            final_confidence = 0.0
            
        return {
            "action": final_action,
            "strength": final_strength,
            "confidence": final_confidence,
            "consensus": len(set(actions)) == 1,  # 全戦略が同じアクション
            "contributing_strategies": len(signals)
        }

class ExecutionResultAggregator:
    """実行結果集約器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.confidence_calculator = ConfidenceCalculator()
        self.weight_aggregator = WeightAggregator()
        self.signal_aggregator = SignalAggregator()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "aggregation": {
                "method": "weighted",
                "confidence_weighting": True,
                "performance_adjustment": True,
                "outlier_handling": "cap"
            }
        }
        
    def aggregate_results(self, execution_results) -> AggregatedResult:
        """実行結果の集約"""
        # ExecutionResult からStrategyResult への変換
        strategy_results = []
        
        for exec_result in execution_results:
            if not exec_result.success or not exec_result.context:
                continue
                
            # パイプラインコンテキストから結果を抽出
            signals = {}
            weights = {}
            confidence = 0.0
            
            stage_results = exec_result.context.stage_results
            
            # シグナル統合結果
            if "signal_integration" in stage_results:
                signal_stage = stage_results["signal_integration"]
                if signal_stage.is_success() and signal_stage.data:
                    signals = signal_stage.data.get("integrated_signals", {})
                    confidence += signal_stage.data.get("signal_confidence", 0.0)
                    
            # 重み計算結果
            if "weight_calculation" in stage_results:
                weight_stage = stage_results["weight_calculation"]
                if weight_stage.is_success() and weight_stage.data:
                    weights = weight_stage.data.get("strategy_weights", {})
                    confidence += weight_stage.data.get("weight_confidence", 0.0)
                    
            # リスク調整結果
            if "risk_adjustment" in stage_results:
                risk_stage = stage_results["risk_adjustment"]
                if risk_stage.is_success() and risk_stage.data:
                    # 調整済み重みがある場合は使用
                    weights = risk_stage.data.get("adjusted_weights", weights)
                    confidence += risk_stage.data.get("risk_confidence", 0.0)
                    
            confidence = confidence / 3.0  # 平均
            
            strategy_result = StrategyResult(
                strategy_name=exec_result.strategy_name,
                signals=signals,
                weights=weights,
                confidence=confidence,
                execution_time=exec_result.execution_time,
                success=exec_result.success,
                metadata={
                    "task_id": exec_result.task_id,
                    "resource_usage": exec_result.resource_usage
                }
            )
            
            strategy_results.append(strategy_result)
            
        if not strategy_results:
            return self._create_empty_result()
            
        return self._perform_aggregation(strategy_results)
        
    def _perform_aggregation(self, strategy_results: List[StrategyResult]) -> AggregatedResult:
        """集約処理の実行"""
        aggregation_id = f"agg_{int(time.time() * 1000)}"
        
        # 信頼度計算
        strategy_confidences = []
        for result in strategy_results:
            adjusted_conf = self.confidence_calculator.calculate_strategy_confidence(result)
            strategy_confidences.append(adjusted_conf)
            
        # 全体コンセンサス信頼度
        overall_confidence = self.confidence_calculator.calculate_consensus_confidence(strategy_results)
        
        # 重み集約
        aggregation_method = AggregationMethod.WEIGHTED_AVERAGE
        if self.config.get("aggregation", {}).get("method") == "simple_average":
            aggregation_method = AggregationMethod.SIMPLE_AVERAGE
        elif self.config.get("aggregation", {}).get("method") == "median":
            aggregation_method = AggregationMethod.MEDIAN
            
        outlier_handling = OutlierHandling.CAP
        handling_str = self.config.get("aggregation", {}).get("outlier_handling", "cap")
        if handling_str == "remove":
            outlier_handling = OutlierHandling.REMOVE
        elif handling_str == "none":
            outlier_handling = OutlierHandling.NONE
        elif handling_str == "winsorize":
            outlier_handling = OutlierHandling.WINSORIZE
            
        final_weights = self.weight_aggregator.aggregate_weights(
            strategy_results, aggregation_method, outlier_handling
        )
        
        # シグナル集約
        final_signals = self.signal_aggregator.aggregate_signals(
            strategy_results, strategy_confidences
        )
        
        # 実行サマリー
        execution_summary = {
            "total_strategies": len(strategy_results),
            "successful_strategies": sum(1 for r in strategy_results if r.success),
            "average_execution_time": np.mean([r.execution_time for r in strategy_results]),
            "confidence_range": [min(strategy_confidences), max(strategy_confidences)],
            "aggregation_method": aggregation_method.value,
            "outlier_handling": outlier_handling.value
        }
        
        # メタデータ
        aggregation_metadata = {
            "strategy_confidences": dict(zip(
                [r.strategy_name for r in strategy_results], 
                strategy_confidences
            )),
            "weight_distribution": self._calculate_weight_distribution(final_weights),
            "signal_consensus": self._calculate_signal_consensus(final_signals),
            "performance_metrics": self._calculate_performance_metrics(strategy_results)
        }
        
        return AggregatedResult(
            aggregation_id=aggregation_id,
            final_signals=final_signals,
            final_weights=final_weights,
            overall_confidence=overall_confidence,
            contributing_strategies=[r.strategy_name for r in strategy_results],
            aggregation_metadata=aggregation_metadata,
            execution_summary=execution_summary
        )
        
    def _create_empty_result(self) -> AggregatedResult:
        """空の結果を作成"""
        return AggregatedResult(
            aggregation_id=f"empty_{int(time.time() * 1000)}",
            final_signals={},
            final_weights={},
            overall_confidence=0.0,
            contributing_strategies=[],
            aggregation_metadata={"error": "No successful strategy results to aggregate"},
            execution_summary={"total_strategies": 0, "successful_strategies": 0}
        )
        
    def _calculate_weight_distribution(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """重み分布の計算"""
        if not weights:
            return {}
            
        weight_values = list(weights.values())
        return {
            "total_instruments": len(weights),
            "non_zero_weights": sum(1 for w in weight_values if abs(w) > 1e-6),
            "max_weight": max(weight_values) if weight_values else 0.0,
            "min_weight": min(weight_values) if weight_values else 0.0,
            "weight_std": np.std(weight_values) if len(weight_values) > 1 else 0.0,
            "concentration": max(weight_values) / sum(abs(w) for w in weight_values) if weight_values else 0.0
        }
        
    def _calculate_signal_consensus(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """シグナルコンセンサスの計算"""
        if not signals:
            return {}
            
        consensus_metrics = {}
        for signal_type, signal_data in signals.items():
            if isinstance(signal_data, dict):
                consensus_metrics[signal_type] = {
                    "action": signal_data.get("action", "hold"),
                    "strength": signal_data.get("strength", 0.0),
                    "confidence": signal_data.get("confidence", 0.0),
                    "consensus": signal_data.get("consensus", False),
                    "contributing_strategies": signal_data.get("contributing_strategies", 0)
                }
                
        return consensus_metrics
        
    def _calculate_performance_metrics(self, strategy_results: List[StrategyResult]) -> Dict[str, Any]:
        """パフォーマンス指標の計算"""
        if not strategy_results:
            return {}
            
        execution_times = [r.execution_time for r in strategy_results]
        confidences = [r.confidence for r in strategy_results]
        signal_strengths = [r.get_signal_strength() for r in strategy_results]
        
        return {
            "execution_time_stats": {
                "mean": np.mean(execution_times),
                "std": np.std(execution_times),
                "min": np.min(execution_times),
                "max": np.max(execution_times)
            },
            "confidence_stats": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences)
            },
            "signal_strength_stats": {
                "mean": np.mean(signal_strengths),
                "std": np.std(signal_strengths),
                "min": np.min(signal_strengths),
                "max": np.max(signal_strengths)
            }
        }
        
    def get_aggregation_report(self, result: AggregatedResult) -> str:
        """集約レポートの生成"""
        report = []
        report.append(f"=== 集約結果レポート (ID: {result.aggregation_id}) ===")
        report.append(f"作成日時: {result.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"全体信頼度: {result.overall_confidence:.3f}")
        report.append(f"貢献戦略数: {len(result.contributing_strategies)}")
        report.append("")
        
        # 実行サマリー
        summary = result.execution_summary
        report.append("--- 実行サマリー ---")
        report.append(f"総戦略数: {summary.get('total_strategies', 0)}")
        report.append(f"成功戦略数: {summary.get('successful_strategies', 0)}")
        report.append(f"平均実行時間: {summary.get('average_execution_time', 0):.2f}秒")
        report.append("")
        
        # シグナルサマリー
        report.append("--- シグナルサマリー ---")
        for signal_type, signal_data in result.final_signals.items():
            if isinstance(signal_data, dict):
                action = signal_data.get("action", "unknown")
                strength = signal_data.get("strength", 0.0)
                confidence = signal_data.get("confidence", 0.0)
                report.append(f"{signal_type}: {action} (強度: {strength:.3f}, 信頼度: {confidence:.3f})")
        report.append("")
        
        # 重みサマリー
        report.append("--- 重みサマリー ---")
        sorted_weights = sorted(result.final_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for instrument, weight in sorted_weights[:10]:  # 上位10銘柄
            report.append(f"{instrument}: {weight:.4f}")
            
        if len(sorted_weights) > 10:
            report.append(f"... 他{len(sorted_weights) - 10}銘柄")
            
        return "\n".join(report)

if __name__ == "__main__":
    # テスト用のサンプル実行
    aggregator = ExecutionResultAggregator()
    
    # サンプル戦略結果作成
    sample_results = []
    for i in range(3):
        result = StrategyResult(
            strategy_name=f"strategy_{i}",
            signals={
                "main_signal": {
                    "action": "buy" if i % 2 == 0 else "sell",
                    "strength": 0.7 + i * 0.1,
                    "confidence": 0.8 + i * 0.05
                }
            },
            weights={f"asset_{j}": np.random.uniform(0, 0.3) for j in range(5)},
            confidence=0.7 + i * 0.1,
            execution_time=20 + i * 5,
            success=True
        )
        sample_results.append(result)
    
    # 集約実行（実際のExecutionResultがない場合のテスト用）
    # aggregated = aggregator.aggregate_results(sample_results)
    
    print("Aggregator initialized successfully")
