"""
Module: Metric Normalization Engine
File: metric_normalization_engine.py
Description: 
  指標正規化システムの処理エンジン
  5つの正規化手法（Min-Max, Z-Score, Robust, Rank, Custom）の実装
  統計分析と信頼度評価を提供
  2-1-3「指標の正規化手法の設計」のコア処理コンポーネント

Author: imega
Created: 2025-07-10
Modified: 2025-07-10

Dependencies:
  - numpy
  - pandas
  - scipy (optional)
  - sklearn (optional)
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime

# 内部モジュール
try:
    from .metric_normalization_config import (
        MetricNormalizationConfig, 
        NormalizationParameters, 
        NormalizationMethod
    )
except ImportError:
    # 直接実行時の対応
    import sys
    sys.path.append(str(Path(__file__).parent))
    from metric_normalization_config import (
        MetricNormalizationConfig, 
        NormalizationParameters, 
        NormalizationMethod
    )

# オプショナル依存関係
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scipy not available, some advanced features will be disabled")

try:
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sklearn not available, using numpy implementations")

# ロガーの設定
logger = logging.getLogger(__name__)

@dataclass
class NormalizationResult:
    """正規化結果の格納クラス"""
    normalized_data: Union[pd.Series, pd.DataFrame, np.ndarray]
    original_data: Union[pd.Series, pd.DataFrame, np.ndarray]
    method_used: str
    parameters: Dict[str, Any]
    statistics: Dict[str, float]
    confidence_score: float
    outliers_detected: int
    missing_values_handled: int
    transformation_log: List[str]
    success: bool
    error_message: Optional[str] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """結果の要約を取得"""
        return {
            "method": self.method_used,
            "success": self.success,
            "confidence_score": self.confidence_score,
            "data_shape": getattr(self.normalized_data, 'shape', 'scalar'),
            "outliers_detected": self.outliers_detected,
            "missing_values_handled": self.missing_values_handled,
            "statistics": self.statistics,
            "error": self.error_message
        }

class MetricNormalizationEngine:
    """
    指標正規化システムの処理エンジン
    
    複数の正規化手法を提供し、統計分析と信頼度評価を行う
    コア処理エンジンクラス
    """
    
    def __init__(self, config: Optional[MetricNormalizationConfig] = None):
        """
        初期化
        
        Args:
            config: 設定インスタンス
        """
        self.config = config if config is not None else MetricNormalizationConfig()
        
        # カスタム正規化関数の登録
        self.custom_functions: Dict[str, Callable] = {
            "profit_factor_transform": self._profit_factor_transform,
            "win_rate_transform": self._win_rate_transform,
            "drawdown_transform": self._drawdown_transform,
            "log_transform": self._log_transform
        }
        
        # 統計キャッシュ
        self.statistics_cache: Dict[str, Dict[str, float]] = {}
        
        logger.info("MetricNormalizationEngine initialized")
    
    def normalize_metric(self, 
                        data: Union[pd.Series, pd.DataFrame, np.ndarray, List], 
                        metric_name: str,
                        strategy_name: Optional[str] = None,
                        parameters: Optional[NormalizationParameters] = None) -> NormalizationResult:
        """
        指標の正規化実行
        
        Args:
            data: 正規化対象データ
            metric_name: 指標名
            strategy_name: 戦略名（オーバーライド用）
            parameters: 正規化パラメータ（指定時は設定を上書き）
            
        Returns:
            NormalizationResult: 正規化結果
        """
        logger.info(f"Normalizing metric: {metric_name}, strategy: {strategy_name}")
        
        try:
            # データの前処理
            processed_data = self._preprocess_data(data)
            if processed_data is None:
                return NormalizationResult(
                    normalized_data=data,
                    original_data=data,
                    method_used="none",
                    parameters={},
                    statistics={},
                    confidence_score=0.0,
                    outliers_detected=0,
                    missing_values_handled=0,
                    transformation_log=["Data preprocessing failed"],
                    success=False,
                    error_message="Invalid input data"
                )
            
            # パラメータの取得
            if parameters is None:
                parameters = self.config.get_normalization_parameters(metric_name, strategy_name)
            
            # 正規化の実行
            if parameters.method == NormalizationMethod.MIN_MAX.value:
                result = self._min_max_normalize(processed_data, parameters)
            elif parameters.method == NormalizationMethod.Z_SCORE.value:
                result = self._z_score_normalize(processed_data, parameters)
            elif parameters.method == NormalizationMethod.ROBUST.value:
                result = self._robust_normalize(processed_data, parameters)
            elif parameters.method == NormalizationMethod.RANK.value:
                result = self._rank_normalize(processed_data, parameters)
            elif parameters.method == NormalizationMethod.CUSTOM.value:
                result = self._custom_normalize(processed_data, parameters)
            else:
                raise ValueError(f"Unknown normalization method: {parameters.method}")
            
            logger.info(f"✓ Normalization completed: {metric_name}, confidence: {result.confidence_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Normalization failed for {metric_name}: {e}")
            return NormalizationResult(
                normalized_data=data,
                original_data=data,
                method_used="failed",
                parameters=parameters.__dict__ if parameters else {},
                statistics={},
                confidence_score=0.0,
                outliers_detected=0,
                missing_values_handled=0,
                transformation_log=[f"Error: {str(e)}"],
                success=False,
                error_message=str(e)
            )
    
    def batch_normalize(self, 
                       data_dict: Dict[str, Union[pd.Series, pd.DataFrame, np.ndarray]], 
                       strategy_name: Optional[str] = None) -> Dict[str, NormalizationResult]:
        """
        複数指標の一括正規化
        
        Args:
            data_dict: {指標名: データ} の辞書
            strategy_name: 戦略名
            
        Returns:
            Dict[str, NormalizationResult]: 正規化結果の辞書
        """
        logger.info(f"Starting batch normalization for {len(data_dict)} metrics")
        
        results = {}
        for metric_name, data in data_dict.items():
            try:
                result = self.normalize_metric(data, metric_name, strategy_name)
                results[metric_name] = result
                
                if result.success:
                    logger.debug(f"✓ {metric_name}: confidence {result.confidence_score:.3f}")
                else:
                    logger.warning(f"✗ {metric_name}: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Error normalizing {metric_name}: {e}")
                results[metric_name] = NormalizationResult(
                    normalized_data=data,
                    original_data=data,
                    method_used="failed",
                    parameters={},
                    statistics={},
                    confidence_score=0.0,
                    outliers_detected=0,
                    missing_values_handled=0,
                    transformation_log=[f"Batch error: {str(e)}"],
                    success=False,
                    error_message=str(e)
                )
        
        success_count = sum(1 for r in results.values() if r.success)
        logger.info(f"Batch normalization completed: {success_count}/{len(data_dict)} successful")
        
        return results
    
    def _preprocess_data(self, data: Union[pd.Series, pd.DataFrame, np.ndarray, List]) -> Optional[np.ndarray]:
        """データの前処理"""
        try:
            # データ型の統一
            if isinstance(data, list):
                data = np.array(data)
            elif isinstance(data, (pd.Series, pd.DataFrame)):
                data = data.values
            
            # 数値型チェック
            if not np.issubdtype(data.dtype, np.number):
                logger.warning("Non-numeric data detected, attempting conversion")
                data = pd.to_numeric(data.flatten(), errors='coerce')
            
            # 空データチェック
            if len(data) == 0:
                logger.error("Empty data provided")
                return None
            
            return data.flatten() if data.ndim > 1 else data
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return None
    
    def _min_max_normalize(self, data: np.ndarray, params: NormalizationParameters) -> NormalizationResult:
        """Min-Max正規化"""
        transformation_log = ["Starting Min-Max normalization"]
        original_data = data.copy()
        
        try:
            # 外れ値処理
            data, outliers_count = self._handle_outliers(data, params.outlier_handling)
            transformation_log.append(f"Outliers handled: {outliers_count}")
            
            # 欠損値処理
            data, missing_count = self._handle_missing_values(data, params.missing_value_strategy)
            transformation_log.append(f"Missing values handled: {missing_count}")
            
            # ログ変換（必要時）
            if params.apply_log_transform:
                data = self._apply_log_transform(data)
                transformation_log.append("Log transform applied")
            
            # Min-Max正規化
            if SKLEARN_AVAILABLE:
                scaler = MinMaxScaler(feature_range=params.target_range)
                normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            else:
                data_min, data_max = np.nanmin(data), np.nanmax(data)
                if data_max == data_min:
                    normalized_data = np.full_like(data, np.mean(params.target_range))
                else:
                    normalized_data = (data - data_min) / (data_max - data_min)
                    normalized_data = (normalized_data * (params.target_range[1] - params.target_range[0]) + 
                                     params.target_range[0])
            
            # 統計情報の計算
            statistics = self._calculate_statistics(original_data, normalized_data)
            confidence_score = self._calculate_confidence_score(original_data, normalized_data, "min_max")
            
            transformation_log.append("Min-Max normalization completed")
            
            return NormalizationResult(
                normalized_data=normalized_data,
                original_data=original_data,
                method_used="min_max",
                parameters=params.__dict__,
                statistics=statistics,
                confidence_score=confidence_score,
                outliers_detected=outliers_count,
                missing_values_handled=missing_count,
                transformation_log=transformation_log,
                success=True
            )
            
        except Exception as e:
            transformation_log.append(f"Error: {str(e)}")
            return NormalizationResult(
                normalized_data=original_data,
                original_data=original_data,
                method_used="min_max",
                parameters=params.__dict__,
                statistics={},
                confidence_score=0.0,
                outliers_detected=0,
                missing_values_handled=0,
                transformation_log=transformation_log,
                success=False,
                error_message=str(e)
            )
    
    def _z_score_normalize(self, data: np.ndarray, params: NormalizationParameters) -> NormalizationResult:
        """Z-Score正規化"""
        transformation_log = ["Starting Z-Score normalization"]
        original_data = data.copy()
        
        try:
            # 外れ値処理
            data, outliers_count = self._handle_outliers(data, params.outlier_handling)
            transformation_log.append(f"Outliers handled: {outliers_count}")
            
            # 欠損値処理
            data, missing_count = self._handle_missing_values(data, params.missing_value_strategy)
            transformation_log.append(f"Missing values handled: {missing_count}")
            
            # Z-Score正規化
            if SKLEARN_AVAILABLE:
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            else:
                mean_val = np.nanmean(data)
                std_val = np.nanstd(data)
                if std_val == 0:
                    normalized_data = np.zeros_like(data)
                else:
                    normalized_data = (data - mean_val) / std_val
            
            # 範囲クリッピング（必要時）
            if params.target_range != (-np.inf, np.inf):
                normalized_data = np.clip(normalized_data, params.target_range[0], params.target_range[1])
                transformation_log.append(f"Clipped to range: {params.target_range}")
            
            # 統計情報の計算
            statistics = self._calculate_statistics(original_data, normalized_data)
            confidence_score = self._calculate_confidence_score(original_data, normalized_data, "z_score")
            
            transformation_log.append("Z-Score normalization completed")
            
            return NormalizationResult(
                normalized_data=normalized_data,
                original_data=original_data,
                method_used="z_score",
                parameters=params.__dict__,
                statistics=statistics,
                confidence_score=confidence_score,
                outliers_detected=outliers_count,
                missing_values_handled=missing_count,
                transformation_log=transformation_log,
                success=True
            )
            
        except Exception as e:
            transformation_log.append(f"Error: {str(e)}")
            return NormalizationResult(
                normalized_data=original_data,
                original_data=original_data,
                method_used="z_score",
                parameters=params.__dict__,
                statistics={},
                confidence_score=0.0,
                outliers_detected=0,
                missing_values_handled=0,
                transformation_log=transformation_log,
                success=False,
                error_message=str(e)
            )
    
    def _robust_normalize(self, data: np.ndarray, params: NormalizationParameters) -> NormalizationResult:
        """Robust正規化（中央値とMADベース）"""
        transformation_log = ["Starting Robust normalization"]
        original_data = data.copy()
        
        try:
            # 欠損値処理（外れ値処理はRobust手法では通常スキップ）
            data, missing_count = self._handle_missing_values(data, params.missing_value_strategy)
            transformation_log.append(f"Missing values handled: {missing_count}")
            
            # Robust正規化
            if SKLEARN_AVAILABLE:
                scaler = RobustScaler(quantile_range=(25.0, 75.0))
                normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            else:
                median_val = np.nanmedian(data)
                q25, q75 = np.nanpercentile(data, [25, 75])
                iqr = q75 - q25
                if iqr == 0:
                    normalized_data = np.zeros_like(data)
                else:
                    normalized_data = (data - median_val) / iqr
            
            # 範囲調整
            if params.target_range != (-np.inf, np.inf):
                # スケーリングして目標範囲に調整
                current_min, current_max = np.nanmin(normalized_data), np.nanmax(normalized_data)
                if current_max != current_min:
                    normalized_data = (normalized_data - current_min) / (current_max - current_min)
                    normalized_data = (normalized_data * (params.target_range[1] - params.target_range[0]) + 
                                     params.target_range[0])
                transformation_log.append(f"Scaled to range: {params.target_range}")
            
            # 統計情報の計算
            statistics = self._calculate_statistics(original_data, normalized_data)
            confidence_score = self._calculate_confidence_score(original_data, normalized_data, "robust")
            
            transformation_log.append("Robust normalization completed")
            
            return NormalizationResult(
                normalized_data=normalized_data,
                original_data=original_data,
                method_used="robust",
                parameters=params.__dict__,
                statistics=statistics,
                confidence_score=confidence_score,
                outliers_detected=0,  # Robustでは外れ値処理をスキップ
                missing_values_handled=missing_count,
                transformation_log=transformation_log,
                success=True
            )
            
        except Exception as e:
            transformation_log.append(f"Error: {str(e)}")
            return NormalizationResult(
                normalized_data=original_data,
                original_data=original_data,
                method_used="robust",
                parameters=params.__dict__,
                statistics={},
                confidence_score=0.0,
                outliers_detected=0,
                missing_values_handled=0,
                transformation_log=transformation_log,
                success=False,
                error_message=str(e)
            )
    
    def _rank_normalize(self, data: np.ndarray, params: NormalizationParameters) -> NormalizationResult:
        """ランク正規化"""
        transformation_log = ["Starting Rank normalization"]
        original_data = data.copy()
        
        try:
            # 欠損値処理
            data, missing_count = self._handle_missing_values(data, params.missing_value_strategy)
            transformation_log.append(f"Missing values handled: {missing_count}")
            
            # ランク正規化
            if SCIPY_AVAILABLE:
                ranks = stats.rankdata(data, method='average')
            else:
                # numpy実装
                sorted_indices = np.argsort(data)
                ranks = np.empty_like(sorted_indices, dtype=float)
                ranks[sorted_indices] = np.arange(1, len(data) + 1)
            
            # 0-1スケールに正規化
            normalized_data = (ranks - 1) / (len(ranks) - 1)
            
            # 目標範囲への調整
            if params.target_range != (0, 1):
                normalized_data = (normalized_data * (params.target_range[1] - params.target_range[0]) + 
                                 params.target_range[0])
                transformation_log.append(f"Scaled to range: {params.target_range}")
            
            # 統計情報の計算
            statistics = self._calculate_statistics(original_data, normalized_data)
            confidence_score = self._calculate_confidence_score(original_data, normalized_data, "rank")
            
            transformation_log.append("Rank normalization completed")
            
            return NormalizationResult(
                normalized_data=normalized_data,
                original_data=original_data,
                method_used="rank",
                parameters=params.__dict__,
                statistics=statistics,
                confidence_score=confidence_score,
                outliers_detected=0,  # ランク正規化では外れ値影響が自動的に軽減
                missing_values_handled=missing_count,
                transformation_log=transformation_log,
                success=True
            )
            
        except Exception as e:
            transformation_log.append(f"Error: {str(e)}")
            return NormalizationResult(
                normalized_data=original_data,
                original_data=original_data,
                method_used="rank",
                parameters=params.__dict__,
                statistics={},
                confidence_score=0.0,
                outliers_detected=0,
                missing_values_handled=0,
                transformation_log=transformation_log,
                success=False,
                error_message=str(e)
            )
    
    def _custom_normalize(self, data: np.ndarray, params: NormalizationParameters) -> NormalizationResult:
        """カスタム正規化"""
        transformation_log = ["Starting Custom normalization"]
        original_data = data.copy()
        
        try:
            # カスタム関数の取得
            if not params.custom_function or params.custom_function not in self.custom_functions:
                raise ValueError(f"Unknown custom function: {params.custom_function}")
            
            custom_func = self.custom_functions[params.custom_function]
            transformation_log.append(f"Using custom function: {params.custom_function}")
            
            # 欠損値処理
            data, missing_count = self._handle_missing_values(data, params.missing_value_strategy)
            transformation_log.append(f"Missing values handled: {missing_count}")
            
            # カスタム正規化の実行
            normalized_data = custom_func(data, params)
            transformation_log.append("Custom transformation applied")
            
            # 統計情報の計算
            statistics = self._calculate_statistics(original_data, normalized_data)
            confidence_score = self._calculate_confidence_score(original_data, normalized_data, "custom")
            
            transformation_log.append("Custom normalization completed")
            
            return NormalizationResult(
                normalized_data=normalized_data,
                original_data=original_data,
                method_used="custom",
                parameters=params.__dict__,
                statistics=statistics,
                confidence_score=confidence_score,
                outliers_detected=0,
                missing_values_handled=missing_count,
                transformation_log=transformation_log,
                success=True
            )
            
        except Exception as e:
            transformation_log.append(f"Error: {str(e)}")
            return NormalizationResult(
                normalized_data=original_data,
                original_data=original_data,
                method_used="custom",
                parameters=params.__dict__,
                statistics={},
                confidence_score=0.0,
                outliers_detected=0,
                missing_values_handled=0,
                transformation_log=transformation_log,
                success=False,
                error_message=str(e)
            )
    
    def _handle_outliers(self, data: np.ndarray, method: str) -> Tuple[np.ndarray, int]:
        """外れ値処理"""
        if method == "none":
            return data, 0
        
        # IQRベースの外れ値検出
        q25, q75 = np.nanpercentile(data, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_count = np.sum(outlier_mask)
        
        if method == "clip":
            data = np.clip(data, lower_bound, upper_bound)
        elif method == "remove":
            data = data[~outlier_mask]
        elif method == "transform":
            # Winsorization
            data[data < lower_bound] = lower_bound
            data[data > upper_bound] = upper_bound
        
        return data, outlier_count
    
    def _handle_missing_values(self, data: np.ndarray, method: str) -> Tuple[np.ndarray, int]:
        """欠損値処理"""
        missing_mask = np.isnan(data)
        missing_count = np.sum(missing_mask)
        
        if missing_count == 0:
            return data, 0
        
        if method == "drop":
            data = data[~missing_mask]
        elif method == "mean":
            data[missing_mask] = np.nanmean(data)
        elif method == "median":
            data[missing_mask] = np.nanmedian(data)
        elif method == "interpolate":
            # 線形補間
            indices = np.arange(len(data))
            data[missing_mask] = np.interp(indices[missing_mask], indices[~missing_mask], data[~missing_mask])
        
        return data, missing_count
    
    def _apply_log_transform(self, data: np.ndarray) -> np.ndarray:
        """ログ変換"""
        # 負の値を0.001に置換してログ変換
        data = np.where(data <= 0, 0.001, data)
        return np.log(data)
    
    def _calculate_statistics(self, original: np.ndarray, normalized: np.ndarray) -> Dict[str, float]:
        """統計情報の計算"""
        try:
            stats = {
                "original_mean": float(np.nanmean(original)),
                "original_std": float(np.nanstd(original)),
                "original_min": float(np.nanmin(original)),
                "original_max": float(np.nanmax(original)),
                "normalized_mean": float(np.nanmean(normalized)),
                "normalized_std": float(np.nanstd(normalized)),
                "normalized_min": float(np.nanmin(normalized)),
                "normalized_max": float(np.nanmax(normalized))
            }
            return stats
        except Exception as e:
            logger.warning(f"Statistics calculation failed: {e}")
            return {}
    
    def _calculate_confidence_score(self, original: np.ndarray, normalized: np.ndarray, method: str) -> float:
        """信頼度スコアの計算"""
        try:
            # 基本スコア（データ完整性）
            base_score = 0.8 if len(normalized) == len(original) else 0.6
            
            # 分布保持スコア
            if SCIPY_AVAILABLE and len(original) > 10:
                # Spearman相関係数で順序関係の保持を評価
                correlation, _ = stats.spearmanr(original, normalized)
                if np.isnan(correlation):
                    correlation = 0.0
                distribution_score = abs(correlation) * 0.2
            else:
                distribution_score = 0.1
            
            # 手法固有スコア
            method_scores = {
                "min_max": 0.85,
                "z_score": 0.9,
                "robust": 0.95,
                "rank": 0.8,
                "custom": 0.75
            }
            method_score = method_scores.get(method, 0.7)
            
            # 総合スコア
            confidence = base_score + distribution_score
            confidence = min(confidence, method_score)  # 手法固有の上限適用
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    # カスタム正規化関数群
    def _profit_factor_transform(self, data: np.ndarray, params: NormalizationParameters) -> np.ndarray:
        """プロフィットファクター専用変換"""
        # 1.0を基準に対数変換
        transformed = np.where(data <= 0, 0.001, data)  # 負の値を除去
        transformed = np.log(transformed) / np.log(10)  # 常用対数
        
        # 範囲調整
        transformed = np.clip(transformed, -2, 2)  # -2から2の範囲に制限
        
        # 0-1スケールに正規化
        normalized = (transformed + 2) / 4
        
        return normalized
    
    def _win_rate_transform(self, data: np.ndarray, params: NormalizationParameters) -> np.ndarray:
        """勝率専用変換"""
        # パーセント値を0-1に正規化
        if np.max(data) > 1:
            data = data / 100.0
        
        # ロジット変換（0と1を避ける）
        data = np.clip(data, 0.001, 0.999)
        transformed = np.log(data / (1 - data))
        
        # Min-Max正規化
        data_min, data_max = np.min(transformed), np.max(transformed)
        if data_max == data_min:
            return np.full_like(data, 0.5)
        
        normalized = (transformed - data_min) / (data_max - data_min)
        return normalized
    
    def _drawdown_transform(self, data: np.ndarray, params: NormalizationParameters) -> np.ndarray:
        """ドローダウン専用変換"""
        # 絶対値を取り、ログ変換
        abs_data = np.abs(data)
        abs_data = np.where(abs_data == 0, 0.001, abs_data)
        transformed = -np.log(abs_data)  # 負の対数（小さいドローダウンほど高スコア）
        
        # Min-Max正規化
        data_min, data_max = np.min(transformed), np.max(transformed)
        if data_max == data_min:
            return np.full_like(data, 0.5)
        
        normalized = (transformed - data_min) / (data_max - data_min)
        return normalized
    
    def _log_transform(self, data: np.ndarray, params: NormalizationParameters) -> np.ndarray:
        """汎用ログ変換"""
        # 正の値に調整
        min_val = np.min(data)
        if min_val <= 0:
            data = data - min_val + 1
        
        # ログ変換
        transformed = np.log(data)
        
        # Min-Max正規化
        data_min, data_max = np.min(transformed), np.max(transformed)
        if data_max == data_min:
            return np.full_like(data, 0.5)
        
        normalized = (transformed - data_min) / (data_max - data_min)
        
        # 目標範囲への調整
        if params.target_range != (0, 1):
            normalized = (normalized * (params.target_range[1] - params.target_range[0]) + 
                         params.target_range[0])
        
        return normalized
    
    def register_custom_function(self, name: str, func: Callable) -> bool:
        """カスタム正規化関数の登録"""
        try:
            self.custom_functions[name] = func
            logger.info(f"Custom function registered: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register custom function {name}: {e}")
            return False
    
    def get_available_methods(self) -> List[str]:
        """利用可能な正規化手法の一覧"""
        methods = [method.value for method in NormalizationMethod]
        return methods
    
    def get_custom_functions(self) -> List[str]:
        """利用可能なカスタム関数の一覧"""
        return list(self.custom_functions.keys())

# 使用例とテスト用の関数
def create_sample_engine() -> MetricNormalizationEngine:
    """サンプルエンジンの作成"""
    config = MetricNormalizationConfig()
    engine = MetricNormalizationEngine(config)
    return engine

if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    # サンプルデータの作成
    np.random.seed(42)
    sample_data = {
        "sharpe_ratio": np.random.normal(1.0, 0.5, 100),
        "profit_factor": np.random.exponential(1.5, 100),
        "win_rate": np.random.beta(2, 2, 100)
    }
    
    # エンジンのテスト
    engine = create_sample_engine()
    results = engine.batch_normalize(sample_data)
    
    print("Normalization Results:")
    for metric, result in results.items():
        print(f"{metric}: {result.get_summary()}")
