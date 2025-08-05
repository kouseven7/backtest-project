"""
Module: Trend Precision Tracker
File: precision_tracker.py
Description: 
  5-2-2「トレンド判定精度の自動補正」
  トレンド判定精度追跡システム - 予測記録と実績の検証を行う

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
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 必要なモジュールのインポート
try:
    from indicators.unified_trend_detector import UnifiedTrendDetector
    from indicators.trend_accuracy_validator import TrendAccuracyValidator
except ImportError as e:
    logging.warning(f"Import warning in precision_tracker: {e}")

@dataclass
class TrendPredictionRecord:
    """トレンド判定記録データクラス"""
    timestamp: datetime
    ticker: str
    strategy_name: str
    method: str
    predicted_trend: str
    confidence_score: float
    actual_trend: Optional[str] = None
    accuracy: Optional[float] = None
    market_context: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_date: Optional[datetime] = None
    is_validated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        # datetime オブジェクトを文字列に変換
        if self.timestamp:
            result['timestamp'] = self.timestamp.isoformat()
        if self.validation_date:
            result['validation_date'] = self.validation_date.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrendPredictionRecord':
        """辞書からインスタンスを作成"""
        # 文字列をdatetimeオブジェクトに変換
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'validation_date' in data and isinstance(data['validation_date'], str):
            data['validation_date'] = datetime.fromisoformat(data['validation_date'])
        
        return cls(**data)

class TrendPrecisionTracker:
    """トレンド判定精度追跡システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # 設定の読み込み
        self.tracking_window_days = config.get('tracking_window_days', 60)
        self.validation_delay_days = config.get('validation_delay_days', 5)
        self.min_records_for_correction = config.get('min_records', 20)
        self.max_records_to_keep = config.get('max_records_to_keep', 10000)
        
        # データ保存パス
        self.persistence_path = Path(config.get('data_persistence_path', 'logs/trend_precision/prediction_records'))
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # メモリ内記録
        self._prediction_records: List[TrendPredictionRecord] = []
        self._accuracy_cache: Dict[str, Dict[str, float]] = {}
        self._load_existing_records()
        
        self.logger.info(f"TrendPrecisionTracker initialized with {len(self._prediction_records)} existing records")
    
    def _load_existing_records(self):
        """既存の記録をロード"""
        try:
            for record_file in self.persistence_path.glob("*.json"):
                with open(record_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            record = TrendPredictionRecord.from_dict(item)
                            self._prediction_records.append(record)
                    else:
                        record = TrendPredictionRecord.from_dict(data)
                        self._prediction_records.append(record)
            
            # 古い記録の削除
            self._cleanup_old_records()
            self.logger.info(f"Loaded {len(self._prediction_records)} prediction records")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing records: {e}")
    
    def _cleanup_old_records(self):
        """古い記録のクリーンアップ"""
        if len(self._prediction_records) <= self.max_records_to_keep:
            return
        
        # タイムスタンプでソート
        self._prediction_records.sort(key=lambda x: x.timestamp, reverse=True)
        
        # 最大記録数まで切り詰め
        removed_count = len(self._prediction_records) - self.max_records_to_keep
        self._prediction_records = self._prediction_records[:self.max_records_to_keep]
        
        self.logger.info(f"Cleaned up {removed_count} old prediction records")
    
    def record_trend_prediction(self,
                              ticker: str,
                              strategy_name: str,
                              method: str,
                              predicted_trend: str,
                              confidence_score: float,
                              parameters: Dict[str, Any],
                              market_context: Dict[str, Any]) -> str:
        """トレンド判定を記録"""
        
        try:
            record = TrendPredictionRecord(
                timestamp=datetime.now(),
                ticker=ticker,
                strategy_name=strategy_name,
                method=method,
                predicted_trend=predicted_trend,
                confidence_score=confidence_score,
                parameters=parameters.copy(),
                market_context=market_context.copy()
            )
            
            self._prediction_records.append(record)
            
            # 定期的な保存
            if len(self._prediction_records) % 100 == 0:
                self._save_recent_records()
            
            record_id = f"{ticker}_{strategy_name}_{method}_{record.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.debug(f"Recorded trend prediction: {record_id}")
            return record_id
            
        except Exception as e:
            self.logger.error(f"Failed to record trend prediction: {e}")
            return ""
    
    def _save_recent_records(self, count: int = 100):
        """最近の記録を保存"""
        try:
            if not self._prediction_records:
                return
            
            # 最新の記録を保存
            recent_records = self._prediction_records[-count:]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            save_data = [record.to_dict() for record in recent_records]
            
            save_path = self.persistence_path / f"predictions_{timestamp}.json"
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
            self.logger.debug(f"Saved {len(recent_records)} recent records to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save recent records: {e}")
    
    def validate_predictions(self, validation_date: datetime) -> List[TrendPredictionRecord]:
        """過去の予測を検証（実際のトレンドと比較）"""
        
        validated_records = []
        cutoff_date = validation_date - timedelta(days=self.validation_delay_days)
        
        try:
            unvalidated_records = [
                r for r in self._prediction_records 
                if not r.is_validated and r.timestamp <= cutoff_date
            ]
            
            self.logger.info(f"Validating {len(unvalidated_records)} prediction records")
            
            # ティッカー別にグループ化
            ticker_groups = {}
            for record in unvalidated_records:
                if record.ticker not in ticker_groups:
                    ticker_groups[record.ticker] = []
                ticker_groups[record.ticker].append(record)
            
            # ティッカー別に検証
            for ticker, records in ticker_groups.items():
                try:
                    validated = self._validate_ticker_predictions(ticker, records, validation_date)
                    validated_records.extend(validated)
                except Exception as e:
                    self.logger.error(f"Failed to validate predictions for {ticker}: {e}")
            
            # 精度キャッシュを更新
            self._update_accuracy_cache(validated_records)
            
            self.logger.info(f"Validated {len(validated_records)} prediction records")
            return validated_records
            
        except Exception as e:
            self.logger.error(f"Failed to validate predictions: {e}")
            return []
    
    def _validate_ticker_predictions(self, 
                                   ticker: str, 
                                   records: List[TrendPredictionRecord],
                                   validation_date: datetime) -> List[TrendPredictionRecord]:
        """特定のティッカーの予測を検証"""
        
        validated_records = []
        
        try:
            # サンプルデータを作成（実際の実装では外部データソースから取得）
            sample_data = self._create_sample_market_data(ticker, validation_date)
            
            if sample_data is None or len(sample_data) < 10:
                self.logger.warning(f"Insufficient data for validation: {ticker}")
                return []
            
            # トレンド精度検証器を作成
            validator = TrendAccuracyValidator(sample_data)
            
            for record in records:
                try:
                    # 記録の日時に基づいて実際のトレンドを判定
                    actual_trend = self._determine_actual_trend(
                        sample_data, record.timestamp, validation_date
                    )
                    
                    # 精度を計算
                    accuracy = self._calculate_prediction_accuracy(
                        record.predicted_trend, actual_trend, record.confidence_score
                    )
                    
                    # 記録を更新
                    record.actual_trend = actual_trend
                    record.accuracy = accuracy
                    record.validation_date = validation_date
                    record.is_validated = True
                    
                    validated_records.append(record)
                    
                except Exception as e:
                    self.logger.error(f"Failed to validate individual record: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to validate ticker predictions for {ticker}: {e}")
        
        return validated_records
    
    def _create_sample_market_data(self, ticker: str, validation_date: datetime) -> Optional[pd.DataFrame]:
        """サンプル市場データを作成（実際の実装では外部データソースから取得）"""
        
        try:
            # 90日間のサンプルデータを生成
            days = 90
            dates = pd.date_range(
                start=validation_date - timedelta(days=days),
                end=validation_date,
                freq='D'
            )
            
            # ランダムウォークベースの価格データ
            np.random.seed(hash(ticker) % (2**32))  # ティッカーベースのシード
            
            initial_price = 100.0
            returns = np.random.normal(0.001, 0.02, len(dates))  # 平均0.1%、標準偏差2%
            
            prices = [initial_price]
            for i in range(1, len(dates)):
                price = prices[-1] * (1 + returns[i])
                prices.append(max(price, 1.0))  # 最低価格1.0
            
            # データフレームを作成
            data = pd.DataFrame({
                'Date': dates,
                'Adj Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            })
            
            data.set_index('Date', inplace=True)
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to create sample market data for {ticker}: {e}")
            return None
    
    def _determine_actual_trend(self,
                              data: pd.DataFrame,
                              prediction_date: datetime,
                              validation_date: datetime) -> str:
        """実際のトレンドを判定"""
        
        try:
            # 予測日から検証日までの価格変動を分析
            prediction_idx = None
            validation_idx = None
            
            # インデックスを探す
            for i, date in enumerate(data.index):
                if abs((date - prediction_date).days) < 1:
                    prediction_idx = i
                if abs((date - validation_date).days) < 1:
                    validation_idx = i
            
            if prediction_idx is None or validation_idx is None or prediction_idx >= validation_idx:
                return "unknown"
            
            # 期間のデータを取得
            period_data = data.iloc[prediction_idx:validation_idx+1]
            
            if len(period_data) < 2:
                return "unknown"
            
            start_price = period_data['Adj Close'].iloc[0]
            end_price = period_data['Adj Close'].iloc[-1]
            max_price = period_data['Adj Close'].max()
            min_price = period_data['Adj Close'].min()
            
            # トレンド判定ロジック
            price_change = (end_price - start_price) / start_price
            max_drawdown = (start_price - min_price) / start_price
            max_gain = (max_price - start_price) / start_price
            
            if price_change > 0.02 and max_gain > 0.03:
                return "uptrend"
            elif price_change < -0.02 and max_drawdown > 0.03:
                return "downtrend"
            else:
                return "range-bound"
                
        except Exception as e:
            self.logger.error(f"Failed to determine actual trend: {e}")
            return "unknown"
    
    def _calculate_prediction_accuracy(self,
                                     predicted_trend: str,
                                     actual_trend: str,
                                     confidence_score: float) -> float:
        """予測精度を計算"""
        
        try:
            if actual_trend == "unknown":
                return 0.5  # 不明な場合は中間値
            
            # 基本精度
            base_accuracy = 1.0 if predicted_trend == actual_trend else 0.0
            
            # 信頼度を考慮した精度調整
            if base_accuracy == 1.0:
                # 正解の場合、信頼度が高いほど高得点
                return min(1.0, base_accuracy + confidence_score * 0.2)
            else:
                # 不正解の場合、信頼度が高いほど低得点
                return max(0.0, base_accuracy - confidence_score * 0.3)
                
        except Exception as e:
            self.logger.error(f"Failed to calculate prediction accuracy: {e}")
            return 0.0
    
    def _update_accuracy_cache(self, validated_records: List[TrendPredictionRecord]):
        """精度キャッシュを更新"""
        
        try:
            for record in validated_records:
                key = f"{record.strategy_name}_{record.method}_{record.ticker}"
                
                if key not in self._accuracy_cache:
                    self._accuracy_cache[key] = {
                        'total_records': 0,
                        'total_accuracy': 0.0,
                        'average_accuracy': 0.0,
                        'last_updated': datetime.now().isoformat()
                    }
                
                cache_entry = self._accuracy_cache[key]
                cache_entry['total_records'] += 1
                cache_entry['total_accuracy'] += record.accuracy or 0.0
                cache_entry['average_accuracy'] = cache_entry['total_accuracy'] / cache_entry['total_records']
                cache_entry['last_updated'] = datetime.now().isoformat()
            
            self.logger.debug(f"Updated accuracy cache with {len(validated_records)} records")
            
        except Exception as e:
            self.logger.error(f"Failed to update accuracy cache: {e}")
    
    def calculate_method_accuracy(self,
                                method: str,
                                strategy_name: str = None,
                                ticker: str = None,
                                days: int = 30) -> Dict[str, float]:
        """手法別精度を計算"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # フィルタリング条件
            filtered_records = [
                r for r in self._prediction_records
                if r.is_validated and r.timestamp >= cutoff_date and r.method == method
            ]
            
            if strategy_name:
                filtered_records = [r for r in filtered_records if r.strategy_name == strategy_name]
            
            if ticker:
                filtered_records = [r for r in filtered_records if r.ticker == ticker]
            
            if not filtered_records:
                return {
                    'average_accuracy': 0.0,
                    'total_predictions': 0,
                    'correct_predictions': 0,
                    'confidence_correlation': 0.0
                }
            
            # 精度統計を計算
            accuracies = [r.accuracy for r in filtered_records if r.accuracy is not None]
            confidences = [r.confidence_score for r in filtered_records]
            
            average_accuracy = np.mean(accuracies) if accuracies else 0.0
            total_predictions = len(filtered_records)
            correct_predictions = sum(1 for acc in accuracies if acc > 0.5)
            
            # 信頼度と精度の相関
            confidence_correlation = 0.0
            if len(accuracies) == len(confidences) and len(accuracies) > 1:
                correlation_matrix = np.corrcoef(accuracies, confidences)
                confidence_correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
            
            return {
                'average_accuracy': float(average_accuracy),
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'confidence_correlation': float(confidence_correlation),
                'accuracy_std': float(np.std(accuracies)) if accuracies else 0.0,
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate method accuracy: {e}")
            return {
                'average_accuracy': 0.0,
                'total_predictions': 0,
                'correct_predictions': 0,
                'confidence_correlation': 0.0
            }
    
    def get_recent_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """最近のパフォーマンス要約を取得"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_records = [
                r for r in self._prediction_records
                if r.is_validated and r.timestamp >= cutoff_date
            ]
            
            if not recent_records:
                return {"message": "No recent validated records found"}
            
            # 統計を計算
            total_records = len(recent_records)
            accuracies = [r.accuracy for r in recent_records if r.accuracy is not None]
            
            # メソッド別統計
            method_stats = {}
            for record in recent_records:
                method = record.method
                if method not in method_stats:
                    method_stats[method] = []
                if record.accuracy is not None:
                    method_stats[method].append(record.accuracy)
            
            # 戦略別統計
            strategy_stats = {}
            for record in recent_records:
                strategy = record.strategy_name
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = []
                if record.accuracy is not None:
                    strategy_stats[strategy].append(record.accuracy)
            
            return {
                'period_days': days,
                'total_validated_records': total_records,
                'overall_accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
                'accuracy_std': float(np.std(accuracies)) if accuracies else 0.0,
                'method_accuracy': {
                    method: float(np.mean(accs)) for method, accs in method_stats.items()
                },
                'strategy_accuracy': {
                    strategy: float(np.mean(accs)) for strategy, accs in strategy_stats.items()
                },
                'best_performing_method': max(method_stats.items(), key=lambda x: np.mean(x[1]))[0] if method_stats else None,
                'records_by_trend': {
                    trend: len([r for r in recent_records if r.predicted_trend == trend])
                    for trend in ['uptrend', 'downtrend', 'range-bound']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get recent performance summary: {e}")
            return {"error": str(e)}
    
    def save_all_records(self):
        """すべての記録を保存"""
        try:
            if not self._prediction_records:
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_data = [record.to_dict() for record in self._prediction_records]
            
            save_path = self.persistence_path / f"all_predictions_{timestamp}.json"
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved all {len(self._prediction_records)} records to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save all records: {e}")
    
    def generate_sample_data(self, count: int = 10) -> List[TrendPredictionRecord]:
        """サンプル予測記録を生成"""
        try:
            sample_records = []
            
            for i in range(count):
                record = TrendPredictionRecord(
                    timestamp=datetime.now() - timedelta(minutes=i*30),
                    ticker=np.random.choice(["AAPL", "MSFT", "GOOGL"]),
                    strategy_name=np.random.choice(["sma", "macd", "rsi"]),
                    method=np.random.choice(["sma_cross", "macd_signal", "rsi_oversold"]),
                    predicted_trend=np.random.choice(["uptrend", "downtrend", "range-bound"]),
                    confidence_score=np.random.uniform(0.4, 0.9),
                    actual_trend=np.random.choice(["uptrend", "downtrend", "range-bound"]),
                    accuracy=np.random.uniform(0.3, 0.8),
                    market_context={
                        "volatility": np.random.uniform(0.1, 0.5),
                        "volume": np.random.randint(1000, 10000)
                    }
                )
                sample_records.append(record)
            
            self.logger.info(f"Generated {count} sample prediction records")
            return sample_records
            
        except Exception as e:
            self.logger.error(f"Failed to generate sample data: {e}")
            return []

if __name__ == "__main__":
    # テスト用コード
    print("TrendPrecisionTracker モジュールが正常にロードされました")
