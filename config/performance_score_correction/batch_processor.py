"""
Module: Batch Processor
File: batch_processor.py
Description: 
  5-2-1「戦略実績に基づくスコア補正機能」
  スコア補正のバッチ処理システム - 日次/週次更新とパフォーマンス分析

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
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import concurrent.futures
import time
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from .performance_tracker import PerformanceTracker
    from .score_corrector import PerformanceBasedScoreCorrector
    from .enhanced_score_calculator import EnhancedStrategyScoreCalculator
except ImportError:
    from performance_tracker import PerformanceTracker
    from score_corrector import PerformanceBasedScoreCorrector
    from enhanced_score_calculator import EnhancedStrategyScoreCalculator

# ロガー設定
logger = logging.getLogger(__name__)

@dataclass
class BatchUpdateResult:
    """バッチ更新結果"""
    update_type: str
    start_time: datetime
    end_time: datetime
    total_strategies: int
    successful_updates: int
    failed_updates: int
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_success_rate(self) -> float:
        """成功率を取得"""
        if self.total_strategies == 0:
            return 0.0
        return self.successful_updates / self.total_strategies
    
    def get_duration(self) -> float:
        """実行時間を秒で取得"""
        return (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'update_type': self.update_type,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.get_duration(),
            'total_strategies': self.total_strategies,
            'successful_updates': self.successful_updates,
            'failed_updates': self.failed_updates,
            'success_rate': self.get_success_rate(),
            'errors': self.errors,
            'performance_metrics': self.performance_metrics
        }

class ScoreCorrectionBatchProcessor:
    """スコア補正のバッチ処理システム"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        batch_config = config.get('batch_processing', {})
        
        self.update_schedule = batch_config.get('update_schedule', 'daily')
        self.batch_size = batch_config.get('batch_size', 50)
        self.max_concurrent_updates = batch_config.get('max_concurrent_updates', 5)
        self.timeout_minutes = batch_config.get('timeout_minutes', 30)
        self.retry_attempts = batch_config.get('retry_attempts', 3)
        
        # コンポーネントの初期化
        self.performance_tracker = PerformanceTracker(config.get('tracker', {}))
        self.score_corrector = PerformanceBasedScoreCorrector(config)
        self.enhanced_calculator = EnhancedStrategyScoreCalculator(
            score_corrector=self.score_corrector
        )
        
        # レポート保存ディレクトリ
        self.report_dir = Path(__file__).parent.parent.parent / "logs" / "score_correction"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # バッチ処理統計
        self.batch_stats = {
            'total_batches_run': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'avg_processing_time': 0.0,
            'last_run_time': None
        }
        
        logger.info("ScoreCorrectionBatchProcessor initialized")
    
    def run_daily_correction_update(self, strategy_list: Optional[List[str]] = None) -> BatchUpdateResult:
        """
        日次補正更新を実行
        
        Args:
            strategy_list: 更新対象の戦略リスト（指定がない場合は自動取得）
            
        Returns:
            BatchUpdateResult: 更新結果
        """
        start_time = datetime.now()
        logger.info("Starting daily score correction update")
        
        try:
            # 1. 更新対象の戦略・ティッカーを取得
            targets = self._get_update_targets(strategy_list)
            
            if not targets:
                logger.warning("No update targets found")
                return BatchUpdateResult(
                    update_type='daily',
                    start_time=start_time,
                    end_time=datetime.now(),
                    total_strategies=0,
                    successful_updates=0,
                    failed_updates=0,
                    errors=['No update targets found']
                )
            
            # 2. バッチでパフォーマンスデータを収集・更新
            update_result = self._process_batch_updates(targets, 'daily')
            
            # 3. 更新結果をレポート
            self._generate_update_report(update_result)
            
            # 4. バッチ統計を更新
            self._update_batch_stats(update_result)
            
            logger.info(f"Daily score correction update completed: "
                       f"{update_result.successful_updates}/{update_result.total_strategies} successful")
            
            return update_result
            
        except Exception as e:
            logger.error(f"Daily score correction update failed: {e}")
            
            return BatchUpdateResult(
                update_type='daily',
                start_time=start_time,
                end_time=datetime.now(),
                total_strategies=0,
                successful_updates=0,
                failed_updates=1,
                errors=[f"Critical error: {str(e)}"]
            )
    
    def run_weekly_analysis(self) -> BatchUpdateResult:
        """
        週次分析を実行
        
        Returns:
            BatchUpdateResult: 分析結果
        """
        start_time = datetime.now()
        logger.info("Starting weekly score correction analysis")
        
        try:
            # 1. 週次パフォーマンス分析
            weekly_stats = self._analyze_weekly_performance()
            
            # 2. 補正モデルの評価
            model_performance = self._evaluate_correction_model()
            
            # 3. パラメータ調整の提案
            adjustment_suggestions = self._suggest_parameter_adjustments(
                weekly_stats, model_performance
            )
            
            # 4. 週次レポート生成
            weekly_result = BatchUpdateResult(
                update_type='weekly_analysis',
                start_time=start_time,
                end_time=datetime.now(),
                total_strategies=len(weekly_stats),
                successful_updates=len(weekly_stats),
                failed_updates=0,
                performance_metrics={
                    'avg_correction_accuracy': model_performance.get('avg_accuracy', 0.0),
                    'total_corrections_applied': model_performance.get('total_corrections', 0),
                    'avg_improvement_ratio': model_performance.get('avg_improvement', 0.0)
                }
            )
            
            # レポート生成
            self._generate_weekly_report(weekly_result, weekly_stats, adjustment_suggestions)
            
            logger.info("Weekly score correction analysis completed")
            return weekly_result
            
        except Exception as e:
            logger.error(f"Weekly analysis failed: {e}")
            
            return BatchUpdateResult(
                update_type='weekly_analysis',
                start_time=start_time,
                end_time=datetime.now(),
                total_strategies=0,
                successful_updates=0,
                failed_updates=1,
                errors=[f"Analysis error: {str(e)}"]
            )
    
    def _get_update_targets(self, strategy_list: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """更新対象の戦略・ティッカーを取得"""
        try:
            targets = []
            
            if strategy_list:
                # 指定された戦略リストを使用
                for strategy_name in strategy_list:
                    # 各戦略のティッカーを取得（ダミー実装）
                    tickers = self._get_strategy_tickers(strategy_name)
                    for ticker in tickers:
                        targets.append({'strategy': strategy_name, 'ticker': ticker})
            else:
                # パフォーマンス記録から自動取得
                all_records = self.performance_tracker.performance_records
                for key in all_records.keys():
                    if '_' in key:
                        strategy_name, ticker = key.split('_', 1)
                        targets.append({'strategy': strategy_name, 'ticker': ticker})
            
            logger.info(f"Found {len(targets)} update targets")
            return targets
            
        except Exception as e:
            logger.error(f"Failed to get update targets: {e}")
            return []
    
    def _get_strategy_tickers(self, strategy_name: str) -> List[str]:
        """戦略のティッカーリストを取得（ダミー実装）"""
        # 実際の実装では、戦略設定から対応ティッカーを取得
        default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        return default_tickers[:2]  # デモ用に2つに制限
    
    def _process_batch_updates(self, targets: List[Dict[str, str]], update_type: str) -> BatchUpdateResult:
        """バッチ更新を処理"""
        start_time = datetime.now()
        successful_updates = 0
        failed_updates = 0
        errors = []
        
        try:
            # 並列処理で更新実行
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_updates) as executor:
                # Future オブジェクトのリスト
                future_to_target = {}
                
                for target in targets:
                    future = executor.submit(
                        self._update_single_target,
                        target['strategy'],
                        target['ticker'],
                        update_type
                    )
                    future_to_target[future] = target
                
                # 結果を収集
                for future in concurrent.futures.as_completed(future_to_target, timeout=self.timeout_minutes*60):
                    target = future_to_target[future]
                    try:
                        success = future.result()
                        if success:
                            successful_updates += 1
                        else:
                            failed_updates += 1
                            errors.append(f"Update failed for {target['strategy']}/{target['ticker']}")
                    except Exception as e:
                        failed_updates += 1
                        errors.append(f"Exception in {target['strategy']}/{target['ticker']}: {str(e)}")
            
        except concurrent.futures.TimeoutError:
            errors.append(f"Batch processing timeout after {self.timeout_minutes} minutes")
            failed_updates += len(targets) - successful_updates
        except Exception as e:
            errors.append(f"Batch processing error: {str(e)}")
            failed_updates = len(targets)
        
        return BatchUpdateResult(
            update_type=update_type,
            start_time=start_time,
            end_time=datetime.now(),
            total_strategies=len(targets),
            successful_updates=successful_updates,
            failed_updates=failed_updates,
            errors=errors
        )
    
    def _update_single_target(self, strategy_name: str, ticker: str, update_type: str) -> bool:
        """単一の戦略・ティッカー組み合わせを更新"""
        try:
            # 補正ファクターを再計算
            correction_result = self.score_corrector.calculate_correction_factor(
                strategy_name, ticker, 0.5  # ダミー現在スコア
            )
            
            # 成功の判定（簡単な例）
            return correction_result.confidence > 0.0
            
        except Exception as e:
            logger.error(f"Failed to update {strategy_name}/{ticker}: {e}")
            return False
    
    def _analyze_weekly_performance(self) -> Dict[str, Any]:
        """週次パフォーマンスを分析"""
        try:
            weekly_stats = {}
            
            # 全戦略のパフォーマンスを収集
            all_records = self.performance_tracker.performance_records
            
            for key, records in all_records.items():
                if not records:
                    continue
                
                strategy_name, ticker = key.split('_', 1)
                recent_records = [r for r in records 
                                if r.timestamp >= datetime.now() - timedelta(days=7)]
                
                if recent_records:
                    weekly_stats[key] = {
                        'strategy': strategy_name,
                        'ticker': ticker,
                        'record_count': len(recent_records),
                        'avg_accuracy': np.mean([r.prediction_accuracy for r in recent_records]),
                        'avg_performance': np.mean([r.actual_performance for r in recent_records]),
                        'performance_std': np.std([r.actual_performance for r in recent_records])
                    }
            
            return weekly_stats
            
        except Exception as e:
            logger.error(f"Failed to analyze weekly performance: {e}")
            return {}
    
    def _evaluate_correction_model(self) -> Dict[str, Any]:
        """補正モデルの評価"""
        try:
            performance = self.enhanced_calculator.get_correction_performance()
            
            model_eval = {
                'avg_accuracy': performance.get('avg_improvement', 0.0),
                'total_corrections': performance.get('corrections_applied', 0),
                'correction_rate': performance.get('correction_rate', 0.0),
                'high_confidence_rate': performance.get('high_confidence_rate', 0.0),
                'model_health': 'good' if performance.get('correction_rate', 0) > 0.1 else 'needs_attention'
            }
            
            return model_eval
            
        except Exception as e:
            logger.error(f"Failed to evaluate correction model: {e}")
            return {}
    
    def _suggest_parameter_adjustments(self, weekly_stats: Dict[str, Any], 
                                     model_performance: Dict[str, Any]) -> List[str]:
        """パラメータ調整を提案"""
        suggestions = []
        
        try:
            # 補正率が低い場合
            correction_rate = model_performance.get('correction_rate', 0.0)
            if correction_rate < 0.1:
                suggestions.append("Consider lowering min_confidence_threshold to increase correction rate")
            
            # 高信頼度補正率が低い場合
            high_conf_rate = model_performance.get('high_confidence_rate', 0.0)
            if high_conf_rate < 0.3:
                suggestions.append("Consider increasing lookback_periods for better confidence")
            
            # パフォーマンス分散が高い場合
            avg_std = np.mean([stats.get('performance_std', 0) for stats in weekly_stats.values()])
            if avg_std > 0.5:
                suggestions.append("High performance variance detected, consider adjusting ema_alpha")
            
            if not suggestions:
                suggestions.append("Current parameters appear to be well-tuned")
            
        except Exception as e:
            logger.error(f"Failed to suggest parameter adjustments: {e}")
            suggestions.append("Unable to generate parameter suggestions due to analysis error")
        
        return suggestions
    
    def _generate_update_report(self, result: BatchUpdateResult):
        """更新レポートを生成"""
        try:
            report_file = self.report_dir / f"daily_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"Update report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate update report: {e}")
    
    def _generate_weekly_report(self, result: BatchUpdateResult, 
                              weekly_stats: Dict[str, Any],
                              suggestions: List[str]):
        """週次レポートを生成"""
        try:
            report_data = {
                'result': result.to_dict(),
                'weekly_statistics': weekly_stats,
                'parameter_suggestions': suggestions,
                'batch_processor_stats': self.batch_stats,
                'generated_at': datetime.now().isoformat()
            }
            
            report_file = self.report_dir / f"weekly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Weekly report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate weekly report: {e}")
    
    def _update_batch_stats(self, result: BatchUpdateResult):
        """バッチ統計を更新"""
        try:
            self.batch_stats['total_batches_run'] += 1
            
            if result.failed_updates == 0:
                self.batch_stats['successful_batches'] += 1
            else:
                self.batch_stats['failed_batches'] += 1
            
            # 平均処理時間の更新
            current_avg = self.batch_stats['avg_processing_time']
            total_batches = self.batch_stats['total_batches_run']
            new_duration = result.get_duration()
            
            self.batch_stats['avg_processing_time'] = (
                (current_avg * (total_batches - 1) + new_duration) / total_batches
            )
            
            self.batch_stats['last_run_time'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Failed to update batch stats: {e}")

# エクスポート
__all__ = [
    "BatchUpdateResult",
    "ScoreCorrectionBatchProcessor"
]
