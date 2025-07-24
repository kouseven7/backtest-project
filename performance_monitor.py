"""
パフォーマンス監視とメトリクス収集
5-3-3 戦略間相関を考慮した配分最適化システム 負荷テスト用

Author: imega
Created: 2025-07-24
Task: Load Testing for 5-3-3
"""

import time
import threading
import gc
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime

# psutilの可用性チェック
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with: pip install psutil")

@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    execution_time: float
    peak_memory_mb: float
    avg_cpu_percent: float
    memory_growth_mb: float
    gc_collections: int
    start_memory_mb: float
    end_memory_mb: float
    thread_count: int
    process_count: int

class PerformanceMonitor:
    """パフォーマンス監視システム"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory = 0.0
        self.cpu_samples: List[float] = []
        self.gc_start_count = 0
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self) -> str:
        """監視開始"""
        if self.monitoring:
            return "already_running"
            
        self.monitoring = True
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage_mb()
        self.peak_memory = self.start_memory
        self.cpu_samples = []
        self.gc_start_count = len(gc.get_stats())
        
        # 監視スレッド開始
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        monitor_id = f"monitor_{int(time.time())}"
        self.logger.info(f"Performance monitoring started: {monitor_id}")
        return monitor_id
        
    def stop_monitoring(self) -> PerformanceMetrics:
        """監視終了とメトリクス取得"""
        if not self.monitoring:
            raise RuntimeError("Monitoring not started")
            
        self.monitoring = False
        
        # スレッド終了を待機
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        end_time = time.time()
        end_memory = self._get_memory_usage_mb()
        gc_end_count = len(gc.get_stats())
        
        # None チェック
        start_time = self.start_time or 0.0
        start_memory = self.start_memory or 0.0
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            peak_memory_mb=self.peak_memory,
            avg_cpu_percent=sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0,
            memory_growth_mb=end_memory - start_memory,
            gc_collections=gc_end_count - self.gc_start_count,
            start_memory_mb=start_memory,
            end_memory_mb=end_memory,
            thread_count=threading.active_count(),
            process_count=self._get_process_count()
        )
        
        self.metrics_history.append(metrics)
        self.logger.info(f"Performance monitoring stopped. Execution time: {metrics.execution_time:.2f}s")
        return metrics
        
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                # メモリ使用量更新
                current_memory = self._get_memory_usage_mb()
                self.peak_memory = max(self.peak_memory, current_memory)
                
                # CPU使用率サンプリング（psutil利用可能時のみ）
                if PSUTIL_AVAILABLE and 'psutil' in globals():
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.cpu_samples.append(cpu_percent)
                
                # サンプリング間隔
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                break
                
    def _get_memory_usage_mb(self) -> float:
        """メモリ使用量取得（MB）"""
        try:
            if PSUTIL_AVAILABLE and 'psutil' in globals():
                process = psutil.Process()
                memory_info = process.memory_info()
                return memory_info.rss / 1024 / 1024  # MB変換
            else:
                # フォールバック（大まかな推定）
                import tracemalloc
                if tracemalloc.is_tracing():
                    _, peak = tracemalloc.get_traced_memory()
                    return peak / 1024 / 1024
                return 0.0
        except Exception:
            return 0.0
            
    def _get_process_count(self) -> int:
        """プロセス数取得"""
        try:
            if PSUTIL_AVAILABLE and 'psutil' in globals():
                return len(psutil.pids())
            else:
                return 1  # フォールバック
        except Exception:
            return 1
            
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        try:
            if PSUTIL_AVAILABLE and 'psutil' in globals():
                cpu_count = psutil.cpu_count()
                memory = psutil.virtual_memory()
                # Windowsでは '/' の代わりに 'C:\' を使用
                disk_path = 'C:\\' if os.name == 'nt' else '/'
                disk = psutil.disk_usage(disk_path)
                
                return {
                    "cpu_count": cpu_count,
                    "total_memory_gb": memory.total / (1024**3),
                    "available_memory_gb": memory.available / (1024**3),
                    "memory_percent": memory.percent,
                    "disk_total_gb": disk.total / (1024**3),
                    "disk_free_gb": disk.free / (1024**3),
                    "python_version": sys.version,
                    "platform": os.name
                }
            else:
                # psutil未利用時のフォールバック
                return {
                    "cpu_count": "unknown",
                    "total_memory_gb": "unknown",
                    "available_memory_gb": "unknown", 
                    "memory_percent": "unknown",
                    "disk_total_gb": "unknown",
                    "disk_free_gb": "unknown",
                    "python_version": sys.version,
                    "platform": os.name,
                    "note": "psutil not available - limited system info"
                }
        except Exception as e:
            self.logger.error(f"System info error: {e}")
            return {"error": str(e)}
            
    def export_metrics(self, filepath: str):
        """メトリクス履歴をJSONエクスポート"""
        try:
            export_data = {
                "system_info": self.get_system_info(),
                "metrics_history": [
                    {
                        "execution_time": m.execution_time,
                        "peak_memory_mb": m.peak_memory_mb,
                        "avg_cpu_percent": m.avg_cpu_percent,
                        "memory_growth_mb": m.memory_growth_mb,
                        "gc_collections": m.gc_collections,
                        "start_memory_mb": m.start_memory_mb,
                        "end_memory_mb": m.end_memory_mb,
                        "thread_count": m.thread_count,
                        "process_count": m.process_count
                    }
                    for m in self.metrics_history
                ],
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Metrics exported to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            
    def clear_history(self):
        """履歴クリア"""
        self.metrics_history.clear()
        self.logger.info("Metrics history cleared")

# テスト用の簡単な実行例
if __name__ == "__main__":
    import time
    
    # ロギング設定
    logging.basicConfig(level=logging.INFO)
    
    monitor = PerformanceMonitor()
    
    # 簡単な負荷テスト
    monitor_id = monitor.start_monitoring()
    
    # 重い処理をシミュレート
    data = [i**2 for i in range(100000)]
    time.sleep(2)
    
    metrics = monitor.stop_monitoring()
    
    print(f"実行時間: {metrics.execution_time:.2f}秒")
    print(f"ピークメモリ: {metrics.peak_memory_mb:.2f}MB")
    print(f"平均CPU使用率: {metrics.avg_cpu_percent:.2f}%")
    print(f"メモリ増加: {metrics.memory_growth_mb:.2f}MB")
    
    # システム情報表示
    system_info = monitor.get_system_info()
    print("\nシステム情報:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
