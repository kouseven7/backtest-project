"""
Module: Trend Error Detector
File: trend_error_detector.py
Description: 
  5-1-2「トレンド判定エラーの影響分析」
  トレンド判定エラーの検出エンジン

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 警告を抑制
warnings.filterwarnings('ignore')

# ロガーの設定
logger = logging.getLogger(__name__)

try:
    from .error_classification_engine import TrendErrorInstance, TrendErrorType, ErrorSeverity, TrendErrorClassificationEngine
except ImportError:
    from error_classification_engine import TrendErrorInstance, TrendErrorType, ErrorSeverity, TrendErrorClassificationEngine

@dataclass
class ErrorDetectionResult:
    """エラー検出結果"""
    detection_timestamp: datetime
    period_analyzed: Tuple[datetime, datetime]
    total_predictions: int
    errors_detected: int
    error_rate: float
    error_instances: List[TrendErrorInstance]
    detection_config: Dict[str, Any]

class TrendErrorDetector:
    """トレンド判定エラー検出エンジン"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Parameters:
            config_path: 設定ファイルのパス
        """
        self.logger = logger
        self.config_path = config_path or self._get_default_config_path()
        self.detection_config = self._load_detection_config()
        
        # エラー分類エンジンの初期化
        self.classifier = TrendErrorClassificationEngine()
        
        # 検出パラメータ
        self.detection_window = self.detection_config.get("detection_window", 30)
        self.minimum_confidence_threshold = self.detection_config.get("minimum_confidence_threshold", 0.3)
        self.error_rate_alert_threshold = self.detection_config.get("error_rate_alert_threshold", 0.3)
        
        # 既存システムとの統合
        self._initialize_trend_systems()
    
    def _get_default_config_path(self) -> str:
        """デフォルト設定ファイルパスを取得"""
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "trend_error_analysis", "error_analysis_config.json"
        )
    
    def _load_detection_config(self) -> Dict[str, Any]:
        """検出設定を読み込み"""
        try:
            import json
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return self._get_default_detection_config()
        except Exception as e:
            self.logger.error(f"Failed to load detection configuration: {e}")
            return self._get_default_detection_config()
    
    def _get_default_detection_config(self) -> Dict[str, Any]:
        """デフォルト検出設定を取得"""
        return {
            "detection_window": 30,
            "minimum_confidence_threshold": 0.3,
            "error_rate_alert_threshold": 0.3,
            "trend_validation_window": 10,
            "volatility_adjustment": True,
            "real_time_monitoring": True,
            "batch_processing": {
                "enabled": True,
                "batch_size": 100
            }
        }
    
    def _initialize_trend_systems(self):
        """トレンド判定システムとの統合を初期化"""
        try:
            # 既存のtrend_accuracy_validatorとの統合
            from indicators.trend_accuracy_validator import TrendAccuracyValidator
            self.trend_validator = TrendAccuracyValidator
            self.logger.info("Trend accuracy validator integration initialized")
        except ImportError:
            self.logger.warning("Trend accuracy validator not available")
            self.trend_validator = None
        
        try:
            # UnifiedTrendDetectorとの統合
            from indicators.unified_trend_detector import UnifiedTrendDetector
            self.trend_detector_class = UnifiedTrendDetector
            self.logger.info("Unified trend detector integration initialized")
        except ImportError:
            self.logger.warning("Unified trend detector not available")
            self.trend_detector_class = None
    
    def detect_trend_errors(self, 
                          market_data: pd.DataFrame,
                          trend_predictions: pd.DataFrame,
                          analysis_period: Optional[Tuple[datetime, datetime]] = None) -> ErrorDetectionResult:
        """
        トレンド判定エラーを検出
        
        Parameters:
            market_data: 市場データ
            trend_predictions: トレンド予測結果
            analysis_period: 分析期間
        
        Returns:
            ErrorDetectionResult: 検出結果
        """
        try:
            self.logger.info("Starting trend error detection")
            
            # 分析期間の設定
            if analysis_period is None:
                end_date = market_data.index[-1]
                start_date = end_date - timedelta(days=self.detection_window)
                analysis_period = (start_date, end_date)
            
            # データの準備とフィルタリング
            filtered_data, filtered_predictions = self._prepare_analysis_data(
                market_data, trend_predictions, analysis_period
            )
            
            # Ground Truthの生成
            ground_truth = self._generate_ground_truth(filtered_data)
            
            # エラー分類の実行
            classification_result = self.classifier.classify_trend_errors(
                filtered_predictions, ground_truth, filtered_data
            )
            
            # エラー率の計算
            total_predictions = len(filtered_predictions)
            error_rate = classification_result.total_errors / total_predictions if total_predictions > 0 else 0
            
            # 検出結果の作成
            result = ErrorDetectionResult(
                detection_timestamp=datetime.now(),
                period_analyzed=analysis_period,
                total_predictions=total_predictions,
                errors_detected=classification_result.total_errors,
                error_rate=error_rate,
                error_instances=classification_result.error_instances,
                detection_config=self.detection_config.copy()
            )
            
            # アラートの生成
            if error_rate > self.error_rate_alert_threshold:
                self.logger.warning(f"High error rate detected: {error_rate:.2%} (threshold: {self.error_rate_alert_threshold:.2%})")
            
            self.logger.info(f"Error detection completed: {classification_result.total_errors} errors in {total_predictions} predictions ({error_rate:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Trend error detection failed: {e}")
            raise
    
    def _prepare_analysis_data(self, 
                             market_data: pd.DataFrame,
                             trend_predictions: pd.DataFrame,
                             analysis_period: Tuple[datetime, datetime]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分析用データを準備"""
        
        start_date, end_date = analysis_period
        
        # 期間でフィルタリング
        market_mask = (market_data.index >= start_date) & (market_data.index <= end_date)
        pred_mask = (trend_predictions.index >= start_date) & (trend_predictions.index <= end_date)
        
        filtered_market = market_data[market_mask].copy()
        filtered_predictions = trend_predictions[pred_mask].copy()
        
        # データの整合性チェック
        if len(filtered_market) == 0:
            raise ValueError(f"No market data available for period {start_date} to {end_date}")
        if len(filtered_predictions) == 0:
            raise ValueError(f"No prediction data available for period {start_date} to {end_date}")
        
        # 必要なカラムの確認と追加
        required_market_columns = ['Adj Close', 'Volume']
        for col in required_market_columns:
            if col not in filtered_market.columns:
                if col == 'Adj Close' and 'Close' in filtered_market.columns:
                    filtered_market['Adj Close'] = filtered_market['Close']
                elif col == 'Volume' and col not in filtered_market.columns:
                    filtered_market['Volume'] = 1000000  # デフォルト値
        
        # 予測データの標準化
        if 'trend' in filtered_predictions.columns and 'predicted_trend' not in filtered_predictions.columns:
            filtered_predictions['predicted_trend'] = filtered_predictions['trend']
        
        if 'confidence' not in filtered_predictions.columns:
            filtered_predictions['confidence'] = 0.5  # デフォルト信頼度
        
        return filtered_market, filtered_predictions
    
    def _generate_ground_truth(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Ground Truthを生成"""
        
        try:
            if self.trend_validator:
                # 既存のTrendAccuracyValidatorを使用
                validator = self.trend_validator(market_data)
                ground_truth_series = validator.create_ground_truth_trends(
                    future_window=self.detection_config.get("trend_validation_window", 10),
                    trend_threshold=0.02
                )
                ground_truth = pd.DataFrame({
                    'actual_trend': ground_truth_series
                }, index=market_data.index)
            else:
                # フォールバック：簡易的なground truth生成
                ground_truth = self._create_simple_ground_truth(market_data)
            
            return ground_truth
            
        except Exception as e:
            self.logger.error(f"Ground truth generation failed: {e}")
            return self._create_simple_ground_truth(market_data)
    
    def _create_simple_ground_truth(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """簡易Ground Truth生成"""
        
        # 価格変動に基づく簡易トレンド判定
        price_col = 'Adj Close'
        if price_col not in market_data.columns:
            price_col = 'Close'
        
        # 5日移動平均との比較
        market_data_copy = market_data.copy()
        market_data_copy['sma_5'] = market_data_copy[price_col].rolling(window=5).mean()
        market_data_copy['price_change'] = market_data_copy[price_col].pct_change(5)
        
        # トレンド判定
        conditions = [
            (market_data_copy['price_change'] > 0.02) & (market_data_copy[price_col] > market_data_copy['sma_5']),
            (market_data_copy['price_change'] < -0.02) & (market_data_copy[price_col] < market_data_copy['sma_5']),
        ]
        choices = ['uptrend', 'downtrend']
        
        actual_trends = np.select(conditions, choices, default='range-bound')
        
        ground_truth = pd.DataFrame({
            'actual_trend': actual_trends
        }, index=market_data.index)
        
        return ground_truth
    
    def detect_realtime_errors(self, 
                             current_prediction: Dict[str, Any],
                             market_data: pd.DataFrame,
                             historical_predictions: pd.DataFrame) -> Optional[TrendErrorInstance]:
        """
        リアルタイムエラー検出
        
        Parameters:
            current_prediction: 現在の予測
            market_data: 市場データ
            historical_predictions: 過去の予測履歴
        
        Returns:
            Optional[TrendErrorInstance]: エラーが検出された場合のインスタンス
        """
        
        if not self.detection_config.get("real_time_monitoring", True):
            return None
        
        try:
            current_time = datetime.now()
            prediction_timestamp = current_prediction.get('timestamp', current_time)
            
            # 十分な履歴データがあるかチェック
            lookback_period = timedelta(days=self.detection_config.get("trend_validation_window", 10))
            validation_start = prediction_timestamp - lookback_period
            
            if validation_start not in market_data.index:
                return None  # 検証に必要なデータが不足
            
            # 過去の実際のトレンドを確認
            validation_data = market_data[market_data.index >= validation_start].copy()
            if len(validation_data) < 5:
                return None  # データ不足
            
            # 実際のトレンドを計算
            price_change = (validation_data['Adj Close'].iloc[-1] / validation_data['Adj Close'].iloc[0]) - 1
            
            if price_change > 0.02:
                actual_trend = "uptrend"
            elif price_change < -0.02:
                actual_trend = "downtrend"
            else:
                actual_trend = "range-bound"
            
            # エラー判定
            predicted_trend = current_prediction.get('trend', 'unknown')
            if predicted_trend != actual_trend:
                # エラーインスタンスの作成
                error_type = self._determine_realtime_error_type(predicted_trend, actual_trend)
                severity = self._estimate_realtime_severity(current_prediction, validation_data)
                
                error_instance = TrendErrorInstance(
                    timestamp=prediction_timestamp,
                    error_type=error_type,
                    severity=severity,
                    predicted_trend=predicted_trend,
                    actual_trend=actual_trend,
                    confidence_level=current_prediction.get('confidence', 0.5),
                    market_context=self._extract_realtime_market_context(validation_data)
                )
                
                self.logger.warning(f"Real-time error detected: {error_type.value} at {prediction_timestamp}")
                return error_instance
            
            return None
            
        except Exception as e:
            self.logger.error(f"Real-time error detection failed: {e}")
            return None
    
    def _determine_realtime_error_type(self, predicted: str, actual: str) -> TrendErrorType:
        """リアルタイムエラータイプを判定"""
        
        if actual == "range-bound" and predicted != "range-bound":
            return TrendErrorType.FALSE_POSITIVE
        elif actual != "range-bound" and predicted == "range-bound":
            return TrendErrorType.FALSE_NEGATIVE
        elif (predicted == "uptrend" and actual == "downtrend") or (predicted == "downtrend" and actual == "uptrend"):
            return TrendErrorType.DIRECTION_WRONG
        else:
            return TrendErrorType.REGIME_MISMATCH
    
    def _estimate_realtime_severity(self, 
                                   prediction: Dict[str, Any],
                                   market_data: pd.DataFrame) -> ErrorSeverity:
        """リアルタイム深刻度を推定"""
        
        # 信頼度による判定
        confidence = prediction.get('confidence', 0.5)
        
        # 市場ボラティリティ
        returns = market_data['Adj Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.2
        
        # 深刻度の判定
        if confidence > 0.8 and volatility > 0.3:
            return ErrorSeverity.CRITICAL
        elif confidence > 0.7 or volatility > 0.25:
            return ErrorSeverity.HIGH
        elif confidence > 0.5 or volatility > 0.15:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _extract_realtime_market_context(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """リアルタイム市場コンテキストを抽出"""
        
        try:
            latest_data = market_data.iloc[-1]
            
            # 基本情報
            context = {
                'price': latest_data['Adj Close'],
                'volume': latest_data.get('Volume', 0),
            }
            
            # ボラティリティ
            if len(market_data) >= 20:
                returns = market_data['Adj Close'].pct_change().dropna()
                context['volatility'] = returns.std() * np.sqrt(252)
            else:
                context['volatility'] = 0.2
            
            # ボリューム比率
            if len(market_data) >= 10:
                avg_volume = market_data['Volume'].rolling(10).mean().iloc[-1]
                context['volume_ratio'] = latest_data.get('Volume', 0) / avg_volume if avg_volume > 0 else 1.0
            else:
                context['volume_ratio'] = 1.0
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to extract real-time market context: {e}")
            return {'volatility': 0.2, 'volume_ratio': 1.0}
    
    def batch_detect_errors(self, 
                          market_data_list: List[pd.DataFrame],
                          prediction_list: List[pd.DataFrame]) -> List[ErrorDetectionResult]:
        """
        バッチでのエラー検出
        
        Parameters:
            market_data_list: 市場データのリスト
            prediction_list: 予測データのリスト
        
        Returns:
            List[ErrorDetectionResult]: 検出結果のリスト
        """
        
        if not self.detection_config.get("batch_processing", {}).get("enabled", True):
            raise ValueError("Batch processing is disabled")
        
        results = []
        batch_size = self.detection_config.get("batch_processing", {}).get("batch_size", 100)
        
        total_batches = len(market_data_list)
        self.logger.info(f"Starting batch error detection: {total_batches} batches")
        
        for i, (market_data, predictions) in enumerate(zip(market_data_list, prediction_list)):
            try:
                self.logger.info(f"Processing batch {i+1}/{total_batches}")
                result = self.detect_trend_errors(market_data, predictions)
                results.append(result)
                
                # Progress reporting
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i+1}/{total_batches} batches")
                    
            except Exception as e:
                self.logger.error(f"Failed to process batch {i+1}: {e}")
                continue
        
        self.logger.info(f"Batch processing completed: {len(results)}/{total_batches} batches processed")
        return results
    
    def generate_detection_summary(self, results: List[ErrorDetectionResult]) -> Dict[str, Any]:
        """検出結果のサマリーを生成"""
        
        if not results:
            return {"summary": "No detection results available"}
        
        # 統計計算
        total_predictions = sum(r.total_predictions for r in results)
        total_errors = sum(r.errors_detected for r in results)
        error_rates = [r.error_rate for r in results if r.total_predictions > 0]
        
        # エラータイプの集計
        error_type_counts = {}
        all_error_instances = []
        for result in results:
            all_error_instances.extend(result.error_instances)
        
        for error_instance in all_error_instances:
            error_type = error_instance.error_type.value
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        summary = {
            "detection_summary": {
                "total_batches_processed": len(results),
                "total_predictions_analyzed": total_predictions,
                "total_errors_detected": total_errors,
                "overall_error_rate": total_errors / total_predictions if total_predictions > 0 else 0,
                "average_error_rate": np.mean(error_rates) if error_rates else 0,
                "error_rate_std": np.std(error_rates) if len(error_rates) > 1 else 0
            },
            "error_breakdown": error_type_counts,
            "period_coverage": {
                "earliest_date": min(r.period_analyzed[0] for r in results).strftime('%Y-%m-%d'),
                "latest_date": max(r.period_analyzed[1] for r in results).strftime('%Y-%m-%d')
            },
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary

# テスト用のメイン関数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # サンプルデータでのテスト
    dates = pd.date_range('2023-01-01', periods=100)
    
    # サンプル市場データ
    market_data = pd.DataFrame({
        'Adj Close': 100 + np.cumsum(np.random.randn(100) * 2),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # サンプル予測データ
    trend_predictions = pd.DataFrame({
        'predicted_trend': np.random.choice(['uptrend', 'downtrend', 'range-bound'], 100),
        'confidence': np.random.uniform(0.3, 0.9, 100)
    }, index=dates)
    
    # エラー検出の実行
    detector = TrendErrorDetector()
    result = detector.detect_trend_errors(market_data, trend_predictions)
    
    print(f"Error detection completed:")
    print(f"Total Predictions: {result.total_predictions}")
    print(f"Errors Detected: {result.errors_detected}")
    print(f"Error Rate: {result.error_rate:.2%}")
    
    # リアルタイム検出のテスト
    current_prediction = {
        'timestamp': datetime.now(),
        'trend': 'uptrend',
        'confidence': 0.75
    }
    
    realtime_error = detector.detect_realtime_errors(
        current_prediction, market_data, trend_predictions
    )
    
    if realtime_error:
        print(f"Real-time error detected: {realtime_error.error_type.value}")
    else:
        print("No real-time error detected")
