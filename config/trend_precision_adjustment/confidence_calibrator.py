"""
Module: Confidence Calibrator
File: confidence_calibrator.py
Description: 
  5-2-2「トレンド判定精度の自動補正」
  信頼度較正システム - 予測信頼度の較正とキャリブレーション

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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ConfidenceCalibrator:
    """信頼度較正システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # 設定の読み込み
        self.calibration_method = config.get('method', 'platt_scaling')
        self.min_samples_for_calibration = config.get('min_samples_for_calibration', 50)
        self.calibration_window_days = config.get('calibration_window_days', 90)
        self.enable_isotonic_regression = config.get('enable_isotonic_regression', False)
        self.confidence_smoothing_factor = config.get('confidence_smoothing_factor', 0.1)
        
        # 較正モデル保存パス
        self.models_path = Path(config.get('models_path', 'logs/trend_precision/calibration_models'))
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # 較正モデルのキャッシュ
        self._calibration_models: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"ConfidenceCalibrator initialized with method: {self.calibration_method}")
    
    def calibrate_confidence(self,
                           raw_confidence: float,
                           strategy_name: str,
                           method: str,
                           ticker: str,
                           precision_tracker: Any = None) -> float:
        """信頼度を較正"""
        
        try:
            # 入力検証
            if not (0.0 <= raw_confidence <= 1.0):
                self.logger.warning(f"Invalid confidence value: {raw_confidence}, clipping to [0,1]")
                raw_confidence = max(0.0, min(1.0, raw_confidence))
            
            # 較正データの取得
            calibration_data = self._get_calibration_data(
                strategy_name, method, ticker, precision_tracker
            )
            
            if len(calibration_data) < self.min_samples_for_calibration:
                self.logger.debug(f"Insufficient calibration data ({len(calibration_data)} samples), "
                                f"returning smoothed raw confidence")
                return self._apply_smoothing(raw_confidence, strategy_name, method)
            
            # 較正モデルの適用
            calibrated = self._apply_calibration_model(
                raw_confidence, calibration_data, strategy_name, method, ticker
            )
            
            # 結果のクリッピング
            calibrated = max(0.0, min(1.0, calibrated))
            
            self.logger.debug(f"Calibrated confidence: {raw_confidence:.3f} -> {calibrated:.3f}")
            return calibrated
            
        except Exception as e:
            self.logger.error(f"Failed to calibrate confidence: {e}")
            return max(0.0, min(1.0, raw_confidence))
    
    def _get_calibration_data(self,
                            strategy_name: str,
                            method: str,
                            ticker: str,
                            precision_tracker: Any) -> List[Tuple[float, bool]]:
        """較正用データを取得"""
        
        try:
            calibration_data = []
            
            if precision_tracker is None:
                # サンプルデータを生成
                return self._generate_sample_calibration_data(strategy_name, method, ticker)
            
            # 実際のトラッカーからデータを取得
            cutoff_date = datetime.now() - timedelta(days=self.calibration_window_days)
            
            # トラッカーから検証済み記録を取得
            validated_records = [
                r for r in precision_tracker._prediction_records
                if (r.is_validated and 
                    r.timestamp >= cutoff_date and
                    r.strategy_name == strategy_name and
                    r.method == method and
                    r.ticker == ticker and
                    r.accuracy is not None)
            ]
            
            for record in validated_records:
                confidence = record.confidence_score
                accuracy = record.accuracy
                
                # 精度を二値に変換（成功/失敗）
                is_correct = accuracy > 0.6  # 閾値は調整可能
                calibration_data.append((confidence, is_correct))
            
            self.logger.debug(f"Retrieved {len(calibration_data)} calibration data points")
            return calibration_data
            
        except Exception as e:
            self.logger.error(f"Failed to get calibration data: {e}")
            return []
    
    def _generate_sample_calibration_data(self,
                                        strategy_name: str,
                                        method: str,
                                        ticker: str) -> List[Tuple[float, bool]]:
        """サンプル較正データを生成"""
        
        try:
            # 戦略・手法・ティッカーに基づくシード
            seed = hash(f"{strategy_name}_{method}_{ticker}") % (2**32)
            np.random.seed(seed)
            
            # 100個のサンプルデータポイント
            n_samples = 100
            calibration_data = []
            
            for _ in range(n_samples):
                # 信頼度をサンプリング（ベータ分布を使用）
                confidence = np.random.beta(2, 2)  # 0-1の範囲で中央寄り
                
                # 信頼度に基づいた成功確率（較正が必要なケースをシミュレート）
                # 完全に較正済みなら confidence == success_prob
                # 較正が必要な場合は deviation がある
                
                if method == "sma":
                    # SMAは一般的に較正が良好
                    success_prob = confidence * 0.9 + 0.05
                elif method == "macd":
                    # MACDは過信傾向（信頼度 > 実際の成功率）
                    success_prob = confidence * 0.7 + 0.15
                else:
                    # その他は中間的
                    success_prob = confidence * 0.8 + 0.1
                
                # ノイズを追加
                success_prob += np.random.normal(0, 0.1)
                success_prob = max(0.0, min(1.0, success_prob))
                
                # 二項サンプリングで成功/失敗を決定
                is_correct = np.random.random() < success_prob
                
                calibration_data.append((confidence, is_correct))
            
            self.logger.debug(f"Generated {len(calibration_data)} sample calibration data points")
            return calibration_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate sample calibration data: {e}")
            return []
    
    def _apply_calibration_model(self,
                               raw_confidence: float,
                               calibration_data: List[Tuple[float, bool]],
                               strategy_name: str,
                               method: str,
                               ticker: str) -> float:
        """較正モデルを適用"""
        
        try:
            model_key = f"{strategy_name}_{method}_{ticker}"
            
            # キャッシュされたモデルがあるかチェック
            if (model_key in self._calibration_models and 
                self._is_model_fresh(self._calibration_models[model_key])):
                model_info = self._calibration_models[model_key]
                self.logger.debug(f"Using cached calibration model for {model_key}")
            else:
                # 新しいモデルを訓練
                model_info = self._train_calibration_model(calibration_data, method)
                self._calibration_models[model_key] = model_info
                self.logger.debug(f"Trained new calibration model for {model_key}")
            
            # モデルを使用して較正
            if self.calibration_method == 'platt_scaling':
                calibrated = self._apply_platt_scaling(raw_confidence, model_info)
            elif self.calibration_method == 'isotonic_regression':
                calibrated = self._apply_isotonic_regression(raw_confidence, model_info)
            else:
                calibrated = self._apply_empirical_calibration(raw_confidence, calibration_data)
            
            return calibrated
            
        except Exception as e:
            self.logger.error(f"Failed to apply calibration model: {e}")
            return raw_confidence
    
    def _train_calibration_model(self,
                               calibration_data: List[Tuple[float, bool]],
                               method: str) -> Dict[str, Any]:
        """較正モデルを訓練"""
        
        try:
            if not calibration_data:
                return {'method': 'none', 'trained_at': datetime.now()}
            
            confidences = np.array([cd[0] for cd in calibration_data])
            accuracies = np.array([float(cd[1]) for cd in calibration_data])
            
            model_info = {
                'method': self.calibration_method,
                'trained_at': datetime.now(),
                'sample_size': len(calibration_data),
                'confidence_mean': float(np.mean(confidences)),
                'accuracy_mean': float(np.mean(accuracies))
            }
            
            if self.calibration_method == 'platt_scaling':
                # ロジスティック回帰でA, Bパラメータを学習
                A, B = self._fit_platt_scaling_params(confidences, accuracies)
                model_info['platt_A'] = A
                model_info['platt_B'] = B
                
            elif self.calibration_method == 'isotonic_regression':
                # Isotonic Regressionモデル
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                iso_reg.fit(confidences, accuracies)
                model_info['isotonic_model'] = iso_reg
                
            else:
                # 経験的較正（ビニング）
                bins = self._create_calibration_bins(confidences, accuracies)
                model_info['empirical_bins'] = bins
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to train calibration model: {e}")
            return {'method': 'none', 'trained_at': datetime.now()}
    
    def _fit_platt_scaling_params(self,
                                confidences: np.ndarray,
                                accuracies: np.ndarray) -> Tuple[float, float]:
        """Platt Scalingのパラメータを学習"""
        
        try:
            # ロジスティック回帰を使用してA, Bを学習
            # sigmoid(A * confidence + B) = calibrated_probability
            
            X = confidences.reshape(-1, 1)
            y = accuracies
            
            log_reg = LogisticRegression(fit_intercept=True, random_state=42)
            log_reg.fit(X, y)
            
            A = float(log_reg.coef_[0][0])
            B = float(log_reg.intercept_[0])
            
            self.logger.debug(f"Fitted Platt scaling parameters: A={A:.3f}, B={B:.3f}")
            return A, B
            
        except Exception as e:
            self.logger.error(f"Failed to fit Platt scaling parameters: {e}")
            # デフォルト値（恒等変換）
            return 1.0, 0.0
    
    def _create_calibration_bins(self,
                               confidences: np.ndarray,
                               accuracies: np.ndarray,
                               n_bins: int = 10) -> List[Dict[str, float]]:
        """経験的較正用のビンを作成"""
        
        try:
            bins = []
            bin_edges = np.linspace(0, 1, n_bins + 1)
            
            for i in range(n_bins):
                bin_start = bin_edges[i]
                bin_end = bin_edges[i + 1]
                
                # ビンに属するデータ
                mask = (confidences >= bin_start) & (confidences < bin_end)
                bin_confidences = confidences[mask]
                bin_accuracies = accuracies[mask]
                
                if len(bin_confidences) > 0:
                    bin_info = {
                        'bin_start': float(bin_start),
                        'bin_end': float(bin_end),
                        'count': len(bin_confidences),
                        'avg_confidence': float(np.mean(bin_confidences)),
                        'avg_accuracy': float(np.mean(bin_accuracies))
                    }
                else:
                    bin_info = {
                        'bin_start': float(bin_start),
                        'bin_end': float(bin_end),
                        'count': 0,
                        'avg_confidence': (bin_start + bin_end) / 2,
                        'avg_accuracy': (bin_start + bin_end) / 2  # フォールバック
                    }
                
                bins.append(bin_info)
            
            return bins
            
        except Exception as e:
            self.logger.error(f"Failed to create calibration bins: {e}")
            return []
    
    def _apply_platt_scaling(self,
                           raw_confidence: float,
                           model_info: Dict[str, Any]) -> float:
        """Platt Scalingによる較正"""
        
        try:
            A = model_info.get('platt_A', 1.0)
            B = model_info.get('platt_B', 0.0)
            
            # シグモイド関数を適用
            calibrated = 1.0 / (1.0 + np.exp(A * raw_confidence + B))
            return float(calibrated)
            
        except Exception as e:
            self.logger.error(f"Failed to apply Platt scaling: {e}")
            return raw_confidence
    
    def _apply_isotonic_regression(self,
                                 raw_confidence: float,
                                 model_info: Dict[str, Any]) -> float:
        """Isotonic Regressionによる較正"""
        
        try:
            iso_model = model_info.get('isotonic_model')
            if iso_model is None:
                return raw_confidence
            
            calibrated = iso_model.predict([raw_confidence])[0]
            return float(calibrated)
            
        except Exception as e:
            self.logger.error(f"Failed to apply isotonic regression: {e}")
            return raw_confidence
    
    def _apply_empirical_calibration(self,
                                   raw_confidence: float,
                                   calibration_data: List[Tuple[float, bool]]) -> float:
        """経験的較正による較正"""
        
        try:
            if not calibration_data:
                return raw_confidence
            
            # 単純なビニングによる較正
            n_bins = 10
            bin_size = 1.0 / n_bins
            bin_index = min(int(raw_confidence / bin_size), n_bins - 1)
            
            # 該当ビンのデータを取得
            bin_start = bin_index * bin_size
            bin_end = (bin_index + 1) * bin_size
            
            bin_data = [
                accuracy for conf, accuracy in calibration_data
                if bin_start <= conf < bin_end
            ]
            
            if bin_data:
                # ビンの平均精度を返す
                return float(np.mean(bin_data))
            else:
                # データがない場合は近隣ビンを使用
                all_accuracies = [accuracy for _, accuracy in calibration_data]
                return float(np.mean(all_accuracies))
                
        except Exception as e:
            self.logger.error(f"Failed to apply empirical calibration: {e}")
            return raw_confidence
    
    def _apply_smoothing(self,
                       raw_confidence: float,
                       strategy_name: str,
                       method: str) -> float:
        """信頼度の平滑化"""
        
        try:
            # 戦略・手法に基づく調整ファクター
            adjustment_factors = {
                'sma': 0.95,      # SMAは一般的に信頼できる
                'macd': 0.85,     # MACDは過信傾向
                'combined': 0.90, # 複合手法は中間的
                'advanced': 0.88  # 高度な手法も過信傾向
            }
            
            factor = adjustment_factors.get(method, 0.90)
            
            # 平滑化を適用
            smoothed = raw_confidence * factor + (1 - factor) * 0.5
            smoothed *= (1 - self.confidence_smoothing_factor) + self.confidence_smoothing_factor * 0.6
            
            return float(smoothed)
            
        except Exception as e:
            self.logger.error(f"Failed to apply smoothing: {e}")
            return raw_confidence
    
    def _is_model_fresh(self, model_info: Dict[str, Any], max_age_hours: float = 24) -> bool:
        """モデルが新しいかどうか判定"""
        
        try:
            trained_at = model_info.get('trained_at')
            if not isinstance(trained_at, datetime):
                return False
            
            age = datetime.now() - trained_at
            return age.total_seconds() < max_age_hours * 3600
            
        except Exception as e:
            self.logger.error(f"Failed to check model freshness: {e}")
            return False
    
    def get_calibration_statistics(self,
                                 strategy_name: str = None,
                                 method: str = None) -> Dict[str, Any]:
        """較正統計を取得"""
        
        try:
            stats = {
                'total_models': len(self._calibration_models),
                'calibration_method': self.calibration_method,
                'min_samples_threshold': self.min_samples_for_calibration
            }
            
            # フィルタリング
            filtered_models = self._calibration_models
            if strategy_name and method:
                key_prefix = f"{strategy_name}_{method}"
                filtered_models = {k: v for k, v in self._calibration_models.items() if k.startswith(key_prefix)}
            
            if filtered_models:
                sample_sizes = [info.get('sample_size', 0) for info in filtered_models.values()]
                stats.update({
                    'filtered_models': len(filtered_models),
                    'avg_sample_size': float(np.mean(sample_sizes)) if sample_sizes else 0.0,
                    'total_samples': sum(sample_sizes),
                    'model_keys': list(filtered_models.keys())[:10]  # 最初の10個のみ表示
                })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get calibration statistics: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # テスト用コード
    print("ConfidenceCalibrator モジュールが正常にロードされました")
