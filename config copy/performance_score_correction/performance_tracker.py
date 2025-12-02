"""
Module: Performance Tracker
File: performance_tracker.py
Description: 
  5-2-1「戦略実績に基づくスコア補正機能」
  戦略実績追跡システム - 予測スコアと実際のパフォーマンスを記録・追跡

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
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from metrics.performance_metrics import calculate_sharpe_ratio, calculate_sortino_ratio
except ImportError:
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0, trading_days=252):
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return np.sqrt(trading_days) * (returns.mean() - risk_free_rate/trading_days) / returns.std()
    
    def calculate_sortino_ratio(returns, risk_free_rate=0.0, trading_days=252):
        if len(returns) == 0:
            return 0.0
        downside = returns[returns < 0].std()
        if downside == 0:
            return 0.0
        return np.sqrt(trading_days) * (returns.mean() - risk_free_rate/trading_days) / downside

# ロガー設定
logger = logging.getLogger(__name__)

@dataclass
class StrategyPerformanceRecord:
    """戦略パフォーマンス記録"""
    strategy_name: str
    ticker: str
    timestamp: datetime
    predicted_score: float          # 予測スコア
    actual_performance: float       # 実際のパフォーマンス
    market_context: Dict[str, Any]  # 市場コンテキスト
    prediction_accuracy: float      # 予測精度
    correction_factor: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'strategy_name': self.strategy_name,
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat(),
            'predicted_score': self.predicted_score,
            'actual_performance': self.actual_performance,
            'market_context': self.market_context,
            'prediction_accuracy': self.prediction_accuracy,
            'correction_factor': self.correction_factor,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyPerformanceRecord':
        """辞書から復元"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class PerformanceTracker:
    """戦略実績追跡システム"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.tracking_window_days = config.get('tracking_window_days', 30)
        self.min_records_for_correction = config.get('min_records', 10)
        self.performance_threshold = config.get('performance_threshold', 0.1)
        self.data_retention_days = config.get('data_retention_days', 180)
        
        # データ保存ディレクトリ
        self.data_dir = Path(__file__).parent.parent.parent / "logs" / "score_correction" / "performance_records"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 内部データストレージ
        self.performance_records: Dict[str, List[StrategyPerformanceRecord]] = {}
        self._load_existing_data()
        
        logger.info(f"PerformanceTracker initialized with config: {config}")
    
    def record_strategy_performance(self, 
                                  strategy_name: str,
                                  ticker: str,
                                  predicted_score: float,
                                  actual_performance: float,
                                  market_context: Dict[str, Any] = None) -> str:
        """
        戦略パフォーマンスを記録
        
        Args:
            strategy_name: 戦略名
            ticker: ティッカー
            predicted_score: 予測スコア
            actual_performance: 実際のパフォーマンス
            market_context: 市場コンテキスト
            
        Returns:
            str: 記録ID
        """
        try:
            # 予測精度を計算
            prediction_accuracy = self._calculate_prediction_accuracy_single(
                predicted_score, actual_performance
            )
            
            # レコード作成
            record = StrategyPerformanceRecord(
                strategy_name=strategy_name,
                ticker=ticker,
                timestamp=datetime.now(),
                predicted_score=predicted_score,
                actual_performance=actual_performance,
                market_context=market_context or {},
                prediction_accuracy=prediction_accuracy,
                metadata={'recorded_by': 'PerformanceTracker'}
            )
            
            # ストレージに追加
            key = f"{strategy_name}_{ticker}"
            if key not in self.performance_records:
                self.performance_records[key] = []
            
            self.performance_records[key].append(record)
            
            # データクリーンアップ（古いデータを削除）
            self._cleanup_old_records(key)
            
            # データ永続化
            self._save_record(record)
            
            record_id = f"{key}_{record.timestamp.strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Recorded performance for {strategy_name}/{ticker}: {actual_performance}")
            
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to record performance: {e}")
            raise
    
    def get_performance_history(self,
                              strategy_name: str,
                              ticker: str = None,
                              days: int = 30) -> List[StrategyPerformanceRecord]:
        """
        パフォーマンス履歴を取得
        
        Args:
            strategy_name: 戦略名
            ticker: ティッカー（指定がない場合は全ティッカー）
            days: 取得日数
            
        Returns:
            List[StrategyPerformanceRecord]: パフォーマンス履歴
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            history = []
            
            if ticker:
                key = f"{strategy_name}_{ticker}"
                if key in self.performance_records:
                    records = [r for r in self.performance_records[key] 
                             if r.timestamp >= cutoff_date]
                    history.extend(records)
            else:
                # 全ティッカー対象
                for key, records in self.performance_records.items():
                    if key.startswith(f"{strategy_name}_"):
                        filtered_records = [r for r in records 
                                          if r.timestamp >= cutoff_date]
                        history.extend(filtered_records)
            
            # 時系列順にソート
            history.sort(key=lambda x: x.timestamp)
            
            logger.debug(f"Retrieved {len(history)} records for {strategy_name}")
            return history
            
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []
    
    def calculate_prediction_accuracy(self,
                                    predicted_scores: List[float],
                                    actual_performances: List[float]) -> float:
        """
        予測精度を計算
        
        Args:
            predicted_scores: 予測スコアのリスト
            actual_performances: 実際のパフォーマンスのリスト
            
        Returns:
            float: 予測精度 (0.0-1.0)
        """
        if not predicted_scores or not actual_performances:
            return 0.0
        
        if len(predicted_scores) != len(actual_performances):
            logger.warning("Predicted scores and actual performances length mismatch")
            min_len = min(len(predicted_scores), len(actual_performances))
            predicted_scores = predicted_scores[:min_len]
            actual_performances = actual_performances[:min_len]
        
        try:
            # 平均絶対誤差ベースの精度計算
            mae = np.mean(np.abs(np.array(predicted_scores) - np.array(actual_performances)))
            
            # 精度に変換 (0.0-1.0)
            accuracy = max(0.0, 1.0 - mae)
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Failed to calculate prediction accuracy: {e}")
            return 0.0
    
    def get_strategy_statistics(self, strategy_name: str, days: int = 30) -> Dict[str, Any]:
        """
        戦略の統計情報を取得
        
        Args:
            strategy_name: 戦略名
            days: 分析日数
            
        Returns:
            Dict[str, Any]: 統計情報
        """
        try:
            history = self.get_performance_history(strategy_name, days=days)
            
            if not history:
                return {
                    'total_records': 0,
                    'avg_accuracy': 0.0,
                    'avg_performance': 0.0,
                    'performance_std': 0.0,
                    'accuracy_trend': 0.0
                }
            
            performances = [r.actual_performance for r in history]
            accuracies = [r.prediction_accuracy for r in history]
            
            stats = {
                'total_records': len(history),
                'avg_accuracy': np.mean(accuracies),
                'avg_performance': np.mean(performances),
                'performance_std': np.std(performances),
                'accuracy_trend': self._calculate_accuracy_trend(accuracies),
                'recent_records': len([r for r in history if r.timestamp >= datetime.now() - timedelta(days=7)]),
                'performance_range': {
                    'min': np.min(performances),
                    'max': np.max(performances),
                    'median': np.median(performances)
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get strategy statistics: {e}")
            return {}
    
    def _calculate_prediction_accuracy_single(self, predicted: float, actual: float) -> float:
        """単一の予測精度を計算"""
        try:
            error = abs(predicted - actual)
            accuracy = max(0.0, 1.0 - error)
            return accuracy
        except:
            return 0.0
    
    def _calculate_accuracy_trend(self, accuracies: List[float]) -> float:
        """精度のトレンドを計算"""
        if len(accuracies) < 2:
            return 0.0
        
        try:
            # 線形トレンド計算
            x = np.arange(len(accuracies))
            slope = np.polyfit(x, accuracies, 1)[0]
            return slope
        except:
            return 0.0
    
    def _cleanup_old_records(self, key: str):
        """古いレコードをクリーンアップ"""
        if key not in self.performance_records:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        self.performance_records[key] = [
            r for r in self.performance_records[key] 
            if r.timestamp >= cutoff_date
        ]
    
    def _save_record(self, record: StrategyPerformanceRecord):
        """レコードをファイルに保存"""
        try:
            filename = f"{record.strategy_name}_{record.ticker}_{record.timestamp.strftime('%Y%m')}.json"
            filepath = self.data_dir / filename
            
            # 月次ファイルに追記
            records = []
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            
            records.append(record.to_dict())
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save record: {e}")
    
    def _load_existing_data(self):
        """既存データを読み込み"""
        try:
            if not self.data_dir.exists():
                return
            
            for filepath in self.data_dir.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        records_data = json.load(f)
                    
                    for record_data in records_data:
                        record = StrategyPerformanceRecord.from_dict(record_data)
                        key = f"{record.strategy_name}_{record.ticker}"
                        
                        if key not in self.performance_records:
                            self.performance_records[key] = []
                        
                        self.performance_records[key].append(record)
                        
                except Exception as e:
                    logger.warning(f"Failed to load data from {filepath}: {e}")
            
            logger.info(f"Loaded existing performance data from {len(self.performance_records)} strategies")
            
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")

# エクスポート
__all__ = [
    "StrategyPerformanceRecord",
    "PerformanceTracker"
]
