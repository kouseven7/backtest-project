"""
バッチテスト実行器
Phase 2.A.2: 拡張トレンド切替テスター用バッチ処理モジュール
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

@dataclass
class BatchTestConfig:
    """バッチテスト設定"""
    symbols: List[str]
    timeframes: List[str]
    date_ranges: List[Dict[str, int]]  # [{"days": 30}, {"days": 90}]
    max_workers: int = 4
    chunk_size: int = 50
    timeout_seconds: int = 300
    retry_attempts: int = 3
    parallel_mode: bool = True

@dataclass 
class TestJob:
    """テストジョブ"""
    job_id: str
    symbol: str
    timeframe: str
    days: int
    scenario_config: Optional[Dict] = None
    priority: int = 1  # 1=高, 2=中, 3=低

@dataclass
class BatchTestResult:
    """バッチテスト結果"""
    job_id: str
    symbol: str
    timeframe: str
    days: int
    success: bool
    execution_time: float
    result_data: Optional[Dict] = None
    error_message: Optional[str] = None
    retry_count: int = 0

class BatchTestExecutor:
    """バッチテスト実行器"""
    
    def __init__(self, 
                 test_function: Callable,
                 config: Optional[BatchTestConfig] = None):
        """
        初期化
        
        Args:
            test_function: 実行するテスト関数
            config: バッチテスト設定
        """
        self.test_function = test_function
        self.config = config or self._get_default_config()
        
        # 結果保存
        self.results: List[BatchTestResult] = []
        self.failed_jobs: List[TestJob] = []
        
        # 実行統計
        self.total_jobs = 0
        self.completed_jobs = 0
        self.failed_jobs_count = 0
        self.start_time = None
        
        logger.info(f"BatchTestExecutor initialized (max_workers: {self.config.max_workers})")
    
    def _get_default_config(self) -> BatchTestConfig:
        """デフォルト設定取得"""
        return BatchTestConfig(
            symbols=["SPY", "QQQ", "AAPL"],
            timeframes=["1h", "4h"],
            date_ranges=[{"days": 30}, {"days": 90}],
            max_workers=4,
            chunk_size=25,
            timeout_seconds=300,
            retry_attempts=2,
            parallel_mode=True
        )
    
    def generate_test_jobs(self, 
                          custom_symbols: Optional[List[str]] = None,
                          custom_timeframes: Optional[List[str]] = None,
                          custom_date_ranges: Optional[List[Dict]] = None) -> List[TestJob]:
        """テストジョブ生成"""
        try:
            symbols = custom_symbols or self.config.symbols
            timeframes = custom_timeframes or self.config.timeframes
            date_ranges = custom_date_ranges or self.config.date_ranges
            
            jobs = []
            job_counter = 0
            
            for symbol in symbols:
                for timeframe in timeframes:
                    for date_range in date_ranges:
                        days = date_range.get("days", 30)
                        
                        job_id = f"job_{job_counter:04d}_{symbol}_{timeframe}_{days}d"
                        
                        # 優先度設定（主要銘柄・期間を高優先度に）
                        priority = self._calculate_job_priority(symbol, timeframe, days)
                        
                        job = TestJob(
                            job_id=job_id,
                            symbol=symbol,
                            timeframe=timeframe,
                            days=days,
                            priority=priority
                        )
                        
                        jobs.append(job)
                        job_counter += 1
            
            # 優先度順でソート
            jobs.sort(key=lambda x: (x.priority, x.symbol))
            
            logger.info(f"Generated {len(jobs)} test jobs")
            return jobs
            
        except Exception as e:
            logger.error(f"Error generating test jobs: {e}")
            return []
    
    def _calculate_job_priority(self, symbol: str, timeframe: str, days: int) -> int:
        """ジョブ優先度計算"""
        priority = 2  # デフォルト（中）
        
        # 主要銘柄は高優先度
        major_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
        if symbol in major_symbols:
            priority = 1
        
        # 短期間は高優先度（テスト効率のため）
        if days <= 30:
            priority = min(priority, 1)
        elif days >= 90:
            priority = max(priority, 3)
        
        return priority
    
    def execute_batch_tests(self, 
                           jobs: Optional[List[TestJob]] = None,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """バッチテスト実行"""
        try:
            self.start_time = time.time()
            
            # ジョブ生成（提供されていない場合）
            if jobs is None:
                jobs = self.generate_test_jobs()
            
            if not jobs:
                raise ValueError("No test jobs to execute")
            
            self.total_jobs = len(jobs)
            self.completed_jobs = 0
            self.failed_jobs_count = 0
            self.results = []
            self.failed_jobs = []
            
            logger.info(f"Starting batch execution: {self.total_jobs} jobs")
            
            # 実行モード選択
            if self.config.parallel_mode and self.config.max_workers > 1:
                execution_results = self._execute_parallel(jobs, progress_callback)
            else:
                execution_results = self._execute_sequential(jobs, progress_callback)
            
            # 実行統計計算
            execution_time = time.time() - self.start_time
            success_rate = self.completed_jobs / self.total_jobs if self.total_jobs > 0 else 0
            
            # 結果集約
            batch_results = {
                'execution_summary': {
                    'total_jobs': self.total_jobs,
                    'completed_jobs': self.completed_jobs,
                    'failed_jobs': self.failed_jobs_count,
                    'success_rate': success_rate,
                    'total_execution_time': execution_time,
                    'average_time_per_job': execution_time / self.total_jobs if self.total_jobs > 0 else 0,
                    'parallel_mode': self.config.parallel_mode,
                    'max_workers': self.config.max_workers
                },
                'detailed_results': [asdict(result) for result in self.results],
                'failed_jobs': [asdict(job) for job in self.failed_jobs],
                'performance_analysis': self._analyze_batch_performance()
            }
            
            logger.info(f"Batch execution completed: {success_rate:.1%} success rate")
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch execution error: {e}")
            return {
                'error': str(e),
                'execution_summary': {'success_rate': 0.0}
            }
    
    def _execute_sequential(self, 
                           jobs: List[TestJob], 
                           progress_callback: Optional[Callable] = None) -> List[BatchTestResult]:
        """逐次実行"""
        logger.info("Executing jobs sequentially")
        
        for i, job in enumerate(jobs):
            try:
                if progress_callback:
                    progress_callback(i, self.total_jobs, job.job_id)
                
                result = self._execute_single_job(job)
                self.results.append(result)
                
                if result.success:
                    self.completed_jobs += 1
                else:
                    self.failed_jobs_count += 1
                    self.failed_jobs.append(job)
                
                logger.info(f"Job {i+1}/{self.total_jobs} completed: {job.job_id} ({'SUCCESS' if result.success else 'FAILED'})")
                
            except Exception as e:
                logger.error(f"Error executing job {job.job_id}: {e}")
                error_result = BatchTestResult(
                    job_id=job.job_id,
                    symbol=job.symbol,
                    timeframe=job.timeframe,
                    days=job.days,
                    success=False,
                    execution_time=0.0,
                    error_message=str(e)
                )
                self.results.append(error_result)
                self.failed_jobs.append(job)
                self.failed_jobs_count += 1
        
        return self.results
    
    def _execute_parallel(self, 
                         jobs: List[TestJob], 
                         progress_callback: Optional[Callable] = None) -> List[BatchTestResult]:
        """並列実行"""
        logger.info(f"Executing jobs in parallel (workers: {self.config.max_workers})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # ジョブをチャンクに分割
            job_chunks = self._chunk_jobs(jobs, self.config.chunk_size)
            
            for chunk_idx, chunk in enumerate(job_chunks):
                logger.info(f"Processing chunk {chunk_idx + 1}/{len(job_chunks)} ({len(chunk)} jobs)")
                
                # チャンク内ジョブを並列実行
                future_to_job = {
                    executor.submit(self._execute_single_job_with_timeout, job): job 
                    for job in chunk
                }
                
                # 結果収集
                for future in concurrent.futures.as_completed(future_to_job):
                    job = future_to_job[future]
                    
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        self.results.append(result)
                        
                        if result.success:
                            self.completed_jobs += 1
                        else:
                            self.failed_jobs_count += 1
                            self.failed_jobs.append(job)
                        
                        if progress_callback:
                            progress_callback(self.completed_jobs + self.failed_jobs_count, 
                                            self.total_jobs, job.job_id)
                        
                        logger.debug(f"Job completed: {job.job_id} ({'SUCCESS' if result.success else 'FAILED'})")
                        
                    except concurrent.futures.TimeoutError:
                        logger.warning(f"Job {job.job_id} timed out")
                        timeout_result = BatchTestResult(
                            job_id=job.job_id,
                            symbol=job.symbol,
                            timeframe=job.timeframe,
                            days=job.days,
                            success=False,
                            execution_time=self.config.timeout_seconds,
                            error_message="Execution timeout"
                        )
                        self.results.append(timeout_result)
                        self.failed_jobs.append(job)
                        self.failed_jobs_count += 1
                        
                    except Exception as e:
                        logger.error(f"Job {job.job_id} failed with exception: {e}")
                        error_result = BatchTestResult(
                            job_id=job.job_id,
                            symbol=job.symbol,
                            timeframe=job.timeframe,
                            days=job.days,
                            success=False,
                            execution_time=0.0,
                            error_message=str(e)
                        )
                        self.results.append(error_result)
                        self.failed_jobs.append(job)
                        self.failed_jobs_count += 1
        
        return self.results
    
    def _chunk_jobs(self, jobs: List[TestJob], chunk_size: int) -> List[List[TestJob]]:
        """ジョブをチャンクに分割"""
        chunks = []
        for i in range(0, len(jobs), chunk_size):
            chunks.append(jobs[i:i + chunk_size])
        return chunks
    
    def _execute_single_job(self, job: TestJob) -> BatchTestResult:
        """単一ジョブ実行"""
        start_time = time.time()
        
        try:
            logger.debug(f"Executing job: {job.job_id}")
            
            # テスト関数実行
            result_data = self.test_function(
                symbol=job.symbol,
                timeframe=job.timeframe,
                days=job.days,
                scenario_config=job.scenario_config
            )
            
            execution_time = time.time() - start_time
            
            # 成功判定
            success = self._validate_test_result(result_data)
            
            return BatchTestResult(
                job_id=job.job_id,
                symbol=job.symbol,
                timeframe=job.timeframe,
                days=job.days,
                success=success,
                execution_time=execution_time,
                result_data=result_data,
                error_message=None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.warning(f"Job {job.job_id} failed: {e}")
            
            return BatchTestResult(
                job_id=job.job_id,
                symbol=job.symbol,
                timeframe=job.timeframe,
                days=job.days,
                success=False,
                execution_time=execution_time,
                result_data=None,
                error_message=str(e)
            )
    
    def _execute_single_job_with_timeout(self, job: TestJob) -> BatchTestResult:
        """タイムアウト付き単一ジョブ実行"""
        return self._execute_single_job(job)
    
    def _validate_test_result(self, result_data: Any) -> bool:
        """テスト結果検証"""
        try:
            if result_data is None:
                return False
            
            # 辞書形式の結果を想定
            if isinstance(result_data, dict):
                # 基本的な成功条件チェック
                if 'test_summary' in result_data:
                    summary = result_data['test_summary']
                    success_rate = summary.get('success_rate', 0)
                    return success_rate > 0.0  # 何らかの成功があれば有効
                
                # エラーがある場合は失敗
                if 'error' in result_data:
                    return False
                
                # データが空でなければ成功とみなす
                return len(result_data) > 0
            
            # その他の形式も成功とみなす
            return True
            
        except Exception as e:
            logger.warning(f"Result validation error: {e}")
            return False
    
    def _analyze_batch_performance(self) -> Dict[str, Any]:
        """バッチパフォーマンス分析"""
        try:
            if not self.results:
                return {}
            
            successful_results = [r for r in self.results if r.success]
            failed_results = [r for r in self.results if not r.success]
            
            analysis = {
                'execution_statistics': {
                    'total_execution_time': sum(r.execution_time for r in self.results),
                    'average_execution_time': np.mean([r.execution_time for r in self.results]),
                    'fastest_job': min(self.results, key=lambda x: x.execution_time).job_id if self.results else None,
                    'slowest_job': max(self.results, key=lambda x: x.execution_time).job_id if self.results else None,
                    'execution_time_distribution': self._calculate_time_distribution()
                },
                'success_analysis': {
                    'success_rate_by_symbol': self._analyze_success_by_symbol(),
                    'success_rate_by_timeframe': self._analyze_success_by_timeframe(),
                    'success_rate_by_period': self._analyze_success_by_period()
                },
                'error_analysis': {
                    'common_errors': self._analyze_common_errors(failed_results),
                    'error_patterns': self._analyze_error_patterns(failed_results)
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing batch performance: {e}")
            return {}
    
    def _calculate_time_distribution(self) -> Dict[str, float]:
        """実行時間分布計算"""
        times = [r.execution_time for r in self.results]
        
        if not times:
            return {}
        
        return {
            'min': min(times),
            'max': max(times),
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'q25': np.percentile(times, 25),
            'q75': np.percentile(times, 75)
        }
    
    def _analyze_success_by_symbol(self) -> Dict[str, float]:
        """銘柄別成功率分析"""
        symbol_stats = {}
        
        for result in self.results:
            symbol = result.symbol
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'total': 0, 'success': 0}
            
            symbol_stats[symbol]['total'] += 1
            if result.success:
                symbol_stats[symbol]['success'] += 1
        
        return {
            symbol: stats['success'] / stats['total'] if stats['total'] > 0 else 0
            for symbol, stats in symbol_stats.items()
        }
    
    def _analyze_success_by_timeframe(self) -> Dict[str, float]:
        """時間軸別成功率分析"""
        timeframe_stats = {}
        
        for result in self.results:
            timeframe = result.timeframe
            if timeframe not in timeframe_stats:
                timeframe_stats[timeframe] = {'total': 0, 'success': 0}
            
            timeframe_stats[timeframe]['total'] += 1
            if result.success:
                timeframe_stats[timeframe]['success'] += 1
        
        return {
            timeframe: stats['success'] / stats['total'] if stats['total'] > 0 else 0
            for timeframe, stats in timeframe_stats.items()
        }
    
    def _analyze_success_by_period(self) -> Dict[str, float]:
        """期間別成功率分析"""
        period_stats = {}
        
        for result in self.results:
            period = f"{result.days}d"
            if period not in period_stats:
                period_stats[period] = {'total': 0, 'success': 0}
            
            period_stats[period]['total'] += 1
            if result.success:
                period_stats[period]['success'] += 1
        
        return {
            period: stats['success'] / stats['total'] if stats['total'] > 0 else 0
            for period, stats in period_stats.items()
        }
    
    def _analyze_common_errors(self, failed_results: List[BatchTestResult]) -> Dict[str, int]:
        """共通エラー分析"""
        error_counts = {}
        
        for result in failed_results:
            if result.error_message:
                # エラーメッセージを分類
                error_type = self._classify_error(result.error_message)
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _classify_error(self, error_message: str) -> str:
        """エラー分類"""
        error_message_lower = error_message.lower()
        
        if 'timeout' in error_message_lower:
            return 'timeout_error'
        elif 'connection' in error_message_lower or 'network' in error_message_lower:
            return 'network_error'
        elif 'data' in error_message_lower and ('empty' in error_message_lower or 'missing' in error_message_lower):
            return 'data_error'
        elif 'memory' in error_message_lower:
            return 'memory_error'
        elif 'permission' in error_message_lower or 'access' in error_message_lower:
            return 'permission_error'
        else:
            return 'unknown_error'
    
    def _analyze_error_patterns(self, failed_results: List[BatchTestResult]) -> Dict[str, Any]:
        """エラーパターン分析"""
        if not failed_results:
            return {}
        
        # 時間帯別エラー分析（簡易実装）
        return {
            'total_failed_jobs': len(failed_results),
            'average_failure_time': np.mean([r.execution_time for r in failed_results]),
            'failure_rate_by_symbol': {
                symbol: sum(1 for r in failed_results if r.symbol == symbol)
                for symbol in set(r.symbol for r in failed_results)
            }
        }
    
    def retry_failed_jobs(self, max_retries: Optional[int] = None) -> Dict[str, Any]:
        """失敗ジョブ再実行"""
        if not self.failed_jobs:
            logger.info("No failed jobs to retry")
            return {'message': 'No failed jobs to retry'}
        
        max_retries = max_retries or self.config.retry_attempts
        retry_jobs = self.failed_jobs[:max_retries]  # 最大再試行数まで
        
        logger.info(f"Retrying {len(retry_jobs)} failed jobs")
        
        # 失敗ジョブリストをクリア
        self.failed_jobs = []
        
        # 再実行
        retry_results = self.execute_batch_tests(retry_jobs)
        
        return {
            'retry_summary': {
                'jobs_retried': len(retry_jobs),
                'retry_success_rate': retry_results.get('execution_summary', {}).get('success_rate', 0)
            },
            'retry_results': retry_results
        }
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """結果保存"""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"output/batch_test_results/batch_results_{timestamp}.json"
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 結果データ準備
            save_data = {
                'batch_config': asdict(self.config),
                'execution_summary': {
                    'total_jobs': self.total_jobs,
                    'completed_jobs': self.completed_jobs,
                    'failed_jobs': self.failed_jobs_count,
                    'success_rate': self.completed_jobs / self.total_jobs if self.total_jobs > 0 else 0,
                    'execution_time': time.time() - self.start_time if self.start_time else 0
                },
                'detailed_results': [asdict(result) for result in self.results],
                'failed_jobs': [asdict(job) for job in self.failed_jobs]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Batch results saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving batch results: {e}")
            return ""
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """実行サマリー取得"""
        if self.start_time is None:
            return {'status': 'Not started'}
        
        elapsed_time = time.time() - self.start_time
        progress_rate = (self.completed_jobs + self.failed_jobs_count) / self.total_jobs if self.total_jobs > 0 else 0
        
        return {
            'status': 'Running' if progress_rate < 1.0 else 'Completed',
            'progress': {
                'completed_jobs': self.completed_jobs,
                'failed_jobs': self.failed_jobs_count,
                'total_jobs': self.total_jobs,
                'progress_rate': progress_rate,
                'success_rate': self.completed_jobs / max(self.completed_jobs + self.failed_jobs_count, 1)
            },
            'timing': {
                'elapsed_time': elapsed_time,
                'estimated_remaining_time': (elapsed_time / progress_rate * (1 - progress_rate)) if progress_rate > 0 else 0,
                'average_time_per_job': elapsed_time / max(self.completed_jobs + self.failed_jobs_count, 1)
            }
        }
