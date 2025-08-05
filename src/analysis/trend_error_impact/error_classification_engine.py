"""
Module: Trend Error Classification Engine
File: error_classification_engine.py
Description: 
  5-1-2「トレンド判定エラーの影響分析」
  トレンド判定エラーの詳細分類システム

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
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

class TrendErrorType(Enum):
    """トレンド判定エラータイプ"""
    FALSE_POSITIVE = "false_positive"      # 偽陽性：トレンドなしをありと判定
    FALSE_NEGATIVE = "false_negative"      # 偽陰性：トレンドありをなしと判定
    TIMING_EARLY = "timing_early"          # タイミング早すぎ
    TIMING_LATE = "timing_late"            # タイミング遅すぎ
    DIRECTION_WRONG = "direction_wrong"    # 方向間違い
    CONFIDENCE_MISMATCH = "confidence_mismatch"  # 信頼度と実際の乖離
    REGIME_MISMATCH = "regime_mismatch"    # 市場レジーム誤判定

class ErrorSeverity(Enum):
    """エラー深刻度"""
    LOW = "low"           # 軽微な影響
    MEDIUM = "medium"     # 中程度の影響
    HIGH = "high"         # 深刻な影響
    CRITICAL = "critical" # 致命的な影響

@dataclass
class TrendErrorInstance:
    """トレンド判定エラーインスタンス"""
    timestamp: datetime
    error_type: TrendErrorType
    severity: ErrorSeverity
    predicted_trend: str
    actual_trend: str
    confidence_level: float
    market_context: Dict[str, Any]
    impact_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ErrorClassificationResult:
    """エラー分類結果"""
    total_errors: int
    error_breakdown: Dict[TrendErrorType, int]
    severity_distribution: Dict[ErrorSeverity, int]
    error_instances: List[TrendErrorInstance]
    classification_timestamp: datetime
    period_analyzed: Tuple[datetime, datetime]

class TrendErrorClassificationEngine:
    """トレンド判定エラー分類エンジン"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Parameters:
            config_path: 設定ファイルのパス
        """
        self.logger = logger
        self.config_path = config_path or self._get_default_config_path()
        self.classification_config = self._load_classification_config()
        
        # 分類パラメータ
        self.timing_threshold = self.classification_config.get("timing_threshold", 3)
        self.confidence_threshold = self.classification_config.get("confidence_threshold", 0.3)
        self.severity_thresholds = self.classification_config.get("severity_thresholds", {
            "low": 0.01,
            "medium": 0.03,
            "high": 0.05,
            "critical": 0.1
        })
    
    def _get_default_config_path(self) -> str:
        """デフォルト設定ファイルパスを取得"""
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "trend_error_analysis", "classification_rules.json"
        )
    
    def _load_classification_config(self) -> Dict[str, Any]:
        """分類設定を読み込み"""
        try:
            import json
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            "timing_threshold": 3,
            "confidence_threshold": 0.3,
            "severity_thresholds": {
                "low": 0.01,
                "medium": 0.03,
                "high": 0.05,
                "critical": 0.1
            },
            "trend_mapping": {
                "uptrend": 1,
                "downtrend": -1,
                "range-bound": 0,
                "unknown": None
            }
        }
    
    def classify_trend_errors(self, 
                            predictions: pd.DataFrame,
                            ground_truth: pd.DataFrame,
                            market_data: pd.DataFrame) -> ErrorClassificationResult:
        """
        トレンド判定エラーを分類
        
        Parameters:
            predictions: 予測結果 (timestamp, trend, confidence)
            ground_truth: 正解データ (timestamp, actual_trend)
            market_data: 市場データ (価格、ボリューム等)
        
        Returns:
            ErrorClassificationResult: 分類結果
        """
        try:
            self.logger.info("Starting trend error classification")
            
            # データの前処理と整合性チェック
            aligned_data = self._align_prediction_data(predictions, ground_truth, market_data)
            
            error_instances = []
            error_counts = {error_type: 0 for error_type in TrendErrorType}
            severity_counts = {severity: 0 for severity in ErrorSeverity}
            
            # 各時点でのエラー分析
            for idx, row in aligned_data.iterrows():
                if pd.isna(row['predicted_trend']) or pd.isna(row['actual_trend']):
                    continue
                
                # エラータイプの判定
                error_type = self._determine_error_type(
                    predicted=row['predicted_trend'],
                    actual=row['actual_trend'],
                    confidence=row.get('confidence', 0.5),
                    timestamp=idx
                )
                
                if error_type is not None:
                    # エラー深刻度の計算
                    severity = self._calculate_error_severity(
                        error_type=error_type,
                        market_context=self._extract_market_context(row, market_data)
                    )
                    
                    # エラーインスタンスの作成
                    error_instance = TrendErrorInstance(
                        timestamp=idx,
                        error_type=error_type,
                        severity=severity,
                        predicted_trend=row['predicted_trend'],
                        actual_trend=row['actual_trend'],
                        confidence_level=row.get('confidence', 0.5),
                        market_context=self._extract_market_context(row, market_data)
                    )
                    
                    error_instances.append(error_instance)
                    error_counts[error_type] += 1
                    severity_counts[severity] += 1
            
            # 分類結果の作成
            result = ErrorClassificationResult(
                total_errors=len(error_instances),
                error_breakdown=error_counts,
                severity_distribution=severity_counts,
                error_instances=error_instances,
                classification_timestamp=datetime.now(),
                period_analyzed=(aligned_data.index[0], aligned_data.index[-1])
            )
            
            self.logger.info(f"Error classification completed: {result.total_errors} errors found")
            return result
            
        except Exception as e:
            self.logger.error(f"Error classification failed: {e}")
            raise
    
    def _align_prediction_data(self, 
                              predictions: pd.DataFrame,
                              ground_truth: pd.DataFrame,
                              market_data: pd.DataFrame) -> pd.DataFrame:
        """予測データと正解データを整列"""
        try:
            # インデックスを日時に統一
            if not isinstance(predictions.index, pd.DatetimeIndex):
                predictions.index = pd.to_datetime(predictions.index)
            if not isinstance(ground_truth.index, pd.DatetimeIndex):
                ground_truth.index = pd.to_datetime(ground_truth.index)
            if not isinstance(market_data.index, pd.DatetimeIndex):
                market_data.index = pd.to_datetime(market_data.index)
            
            # データを結合
            aligned = predictions.join(ground_truth, how='inner', rsuffix='_actual')
            aligned = aligned.join(market_data[['Adj Close', 'Volume']], how='inner')
            
            return aligned
            
        except Exception as e:
            self.logger.error(f"Data alignment failed: {e}")
            raise
    
    def _determine_error_type(self, 
                            predicted: str, 
                            actual: str, 
                            confidence: float,
                            timestamp: datetime) -> Optional[TrendErrorType]:
        """エラータイプを判定"""
        
        if predicted == actual:
            return None  # エラーなし
        
        trend_mapping = self.classification_config.get("trend_mapping", {
            "uptrend": 1, "downtrend": -1, "range-bound": 0, "unknown": None
        })
        
        pred_value = trend_mapping.get(predicted)
        actual_value = trend_mapping.get(actual)
        
        if pred_value is None or actual_value is None:
            return TrendErrorType.REGIME_MISMATCH
        
        # False Positive/Negative の判定
        if actual_value == 0:  # 実際はrange-bound
            if pred_value != 0:
                return TrendErrorType.FALSE_POSITIVE
        elif pred_value == 0:  # 予測がrange-bound
            if actual_value != 0:
                return TrendErrorType.FALSE_NEGATIVE
        
        # 方向間違いの判定
        if (pred_value > 0 and actual_value < 0) or (pred_value < 0 and actual_value > 0):
            return TrendErrorType.DIRECTION_WRONG
        
        # 信頼度とのミスマッチ
        if confidence > 0.7 and predicted != actual:
            return TrendErrorType.CONFIDENCE_MISMATCH
        
        # デフォルト
        return TrendErrorType.REGIME_MISMATCH
    
    def _calculate_error_severity(self, 
                                error_type: TrendErrorType,
                                market_context: Dict[str, Any]) -> ErrorSeverity:
        """エラー深刻度を計算"""
        
        # 市場ボラティリティを考慮
        volatility = market_context.get('volatility', 0.02)
        volume_ratio = market_context.get('volume_ratio', 1.0)
        
        # ベース重要度
        base_severity = {
            TrendErrorType.FALSE_POSITIVE: 0.02,
            TrendErrorType.FALSE_NEGATIVE: 0.03,
            TrendErrorType.TIMING_EARLY: 0.01,
            TrendErrorType.TIMING_LATE: 0.025,
            TrendErrorType.DIRECTION_WRONG: 0.06,
            TrendErrorType.CONFIDENCE_MISMATCH: 0.04,
            TrendErrorType.REGIME_MISMATCH: 0.035
        }.get(error_type, 0.03)
        
        # 市場状況による調整
        adjusted_severity = base_severity * (1 + volatility) * (1 + (volume_ratio - 1) * 0.5)
        
        # 閾値による分類
        thresholds = self.severity_thresholds
        if adjusted_severity >= thresholds["critical"]:
            return ErrorSeverity.CRITICAL
        elif adjusted_severity >= thresholds["high"]:
            return ErrorSeverity.HIGH
        elif adjusted_severity >= thresholds["medium"]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _extract_market_context(self, 
                              row: pd.Series,
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """市場コンテキストを抽出"""
        try:
            # 基本的な市場情報
            price = row.get('Adj Close', 0)
            volume = row.get('Volume', 0)
            
            # ボラティリティ計算（過去20日）
            if len(market_data) >= 20:
                recent_prices = market_data['Adj Close'].tail(20)
                returns = recent_prices.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # 年率換算
            else:
                volatility = 0.2  # デフォルト値
            
            # ボリューム比率（過去20日平均との比較）
            if len(market_data) >= 20:
                avg_volume = market_data['Volume'].tail(20).mean()
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            return {
                'price': price,
                'volume': volume,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'market_cap_estimate': price * volume  # 簡易時価総額推定
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to extract market context: {e}")
            return {'volatility': 0.2, 'volume_ratio': 1.0}
    
    def generate_classification_report(self, 
                                     result: ErrorClassificationResult) -> Dict[str, Any]:
        """分類結果のレポートを生成"""
        
        if result.total_errors == 0:
            return {
                'summary': 'No errors detected',
                'error_rate': 0.0,
                'recommendations': ['Continue current trend detection approach']
            }
        
        # エラー率の計算
        error_rates = {
            error_type.value: count / result.total_errors 
            for error_type, count in result.error_breakdown.items()
            if count > 0
        }
        
        # 深刻度分布
        severity_rates = {
            severity.value: count / result.total_errors 
            for severity, count in result.severity_distribution.items()
            if count > 0
        }
        
        # 改善提案の生成
        recommendations = self._generate_recommendations(result)
        
        report = {
            'summary': f'Total {result.total_errors} errors analyzed',
            'analysis_period': {
                'start': result.period_analyzed[0].strftime('%Y-%m-%d'),
                'end': result.period_analyzed[1].strftime('%Y-%m-%d')
            },
            'error_breakdown': error_rates,
            'severity_distribution': severity_rates,
            'most_common_error': max(error_rates.items(), key=lambda x: x[1])[0] if error_rates else None,
            'critical_error_count': result.severity_distribution.get(ErrorSeverity.CRITICAL, 0),
            'recommendations': recommendations,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return report
    
    def _generate_recommendations(self, result: ErrorClassificationResult) -> List[str]:
        """改善提案を生成"""
        recommendations = []
        
        # 最も多いエラータイプに基づく提案
        if result.error_breakdown[TrendErrorType.FALSE_POSITIVE] > result.total_errors * 0.3:
            recommendations.append("Consider tightening trend detection thresholds to reduce false positives")
        
        if result.error_breakdown[TrendErrorType.FALSE_NEGATIVE] > result.total_errors * 0.3:
            recommendations.append("Consider loosening trend detection sensitivity to catch more trends")
        
        if result.error_breakdown[TrendErrorType.CONFIDENCE_MISMATCH] > result.total_errors * 0.2:
            recommendations.append("Review confidence calculation methodology for better calibration")
        
        if result.severity_distribution[ErrorSeverity.CRITICAL] > result.total_errors * 0.1:
            recommendations.append("Immediate attention required: High number of critical errors detected")
        
        # デフォルト提案
        if not recommendations:
            recommendations.append("Continue monitoring trend detection performance")
        
        return recommendations

# テスト用のメイン関数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # サンプルデータでのテスト
    dates = pd.date_range('2023-01-01', periods=100)
    
    # サンプル予測データ
    predictions = pd.DataFrame({
        'predicted_trend': np.random.choice(['uptrend', 'downtrend', 'range-bound'], 100),
        'confidence': np.random.uniform(0.3, 0.9, 100)
    }, index=dates)
    
    # サンプル正解データ
    ground_truth = pd.DataFrame({
        'actual_trend': np.random.choice(['uptrend', 'downtrend', 'range-bound'], 100)
    }, index=dates)
    
    # サンプル市場データ
    market_data = pd.DataFrame({
        'Adj Close': 100 + np.cumsum(np.random.randn(100) * 2),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # エラー分類の実行
    classifier = TrendErrorClassificationEngine()
    result = classifier.classify_trend_errors(predictions, ground_truth, market_data)
    
    print(f"Classification completed: {result.total_errors} errors found")
    
    # レポート生成
    report = classifier.generate_classification_report(result)
    print(f"Report: {report}")
