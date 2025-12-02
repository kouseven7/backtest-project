"""
DSSMS Phase 3 Task 3.2: リアルタイム実行エンジン
Realtime Execution Engine - ハイブリッドアーキテクチャによるリアルタイム実行環境

主要機能:
1. ハイブリッドアーキテクチャ（イベント駆動 + ポーリング）
2. 非同期並行処理による高速実行
3. DSSMSシステム完全統合
4. 緊急事態検出・自動停止
5. 動的ポートフォリオ管理

Author: GitHub Copilot Agent
Created: 2025-01-22
Task: Phase 3 Task 3.2 - リアルタイム実行環境構築
"""

import asyncio
import logging
import threading
import time
import json
import queue
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set, Union, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
import sys
from contextlib import asynccontextmanager

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

# 既存DSSMSコンポーネント統合
try:
    # DEPRECATED: dssms_backtester_v2.py は動作不可のため無効化 (2025-12-01)
    # from src.dssms.dssms_backtester_v2 import DSSMSBacktesterV2
    from src.dssms.trade_result_analyzer import TradeResultAnalyzer
    from config.realtime_update_engine import RealtimeUpdateEngine
except ImportError as e:
    logging.warning(f"DSSMS コンポーネントインポート失敗: {e}")

# RealtimeConfigManagerは現在のファイル内で定義されているのでインポート
try:
    from src.dssms.realtime_config_manager import RealtimeConfigManager
except ImportError:
    # クラス定義が必要な場合の処理
    RealtimeConfigManager = None

class ExecutionMode(Enum):
    """実行モード"""
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    SIMULATION = "simulation"

class ExecutionStatus(Enum):
    """実行ステータス"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"

class EventType(Enum):
    """イベントタイプ"""
    MARKET_DATA = "market_data"
    TRADE_SIGNAL = "trade_signal"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_ALERT = "risk_alert"
    SYSTEM_EVENT = "system_event"
    USER_COMMAND = "user_command"

@dataclass
class ExecutionEvent:
    """実行イベント"""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 1
    source: str = "unknown"
    processed: bool = False
    
@dataclass
class ExecutionResult:
    """実行結果"""
    success: bool
    timestamp: datetime
    event_id: str
    execution_time_ms: float
    result_data: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class PortfolioState:
    """ポートフォリオ状態"""
    timestamp: datetime
    total_value: float
    positions: Dict[str, Dict[str, Any]]
    cash: float
    unrealized_pnl: float
    realized_pnl: float
    risk_metrics: Dict[str, float]

class RealtimeExecutionEngine:
    """
    リアルタイム実行エンジン
    ハイブリッドアーキテクチャによる高速実行環境
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.logger = setup_logger(__name__)
        
        # 設定管理の初期化
        if RealtimeConfigManager:
            self.config_manager = RealtimeConfigManager(config_path)
            self.config = self.config_manager.get_config()
        else:
            self.config = self._get_default_config()
            self.config_manager = None
        
        # コア状態
        self.status = ExecutionStatus.STOPPED
        self.mode = ExecutionMode.SIMULATION
        self.start_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None
        
        # イベント処理
        self.event_queue = asyncio.Queue(maxsize=self.config.get('event_queue_size', 10000))
        self.high_priority_queue = asyncio.Queue(maxsize=1000)
        self.processed_events = 0
        self.failed_events = 0
        
        # 並行処理制御
        self.event_loop = None
        self.worker_tasks: List[asyncio.Task] = []
        self.polling_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # コンポーネント
        self.market_time_manager = None
        self.emergency_detector = None
        self.portfolio_state: Optional[PortfolioState] = None
        
        # パフォーマンス監視
        self.performance_metrics = {
            'events_per_second': 0.0,
            'avg_processing_time_ms': 0.0,
            'queue_depth': 0,
            'error_rate': 0.0
        }
        
        # 統計情報
        self.execution_stats = {
            'total_events': 0,
            'successful_events': 0,
            'failed_events': 0,
            'start_time': None,
            'uptime_seconds': 0
        }
        
        self.logger.info("リアルタイム実行エンジン初期化完了")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定取得"""
        return {
            'event_queue_size': 10000,
            'event_worker_count': 4,
            'enable_market_polling': True,
            'market_data_poll_interval': 1.0,
            'performance_monitor_interval': 10.0,
            'emergency_monitor_interval': 5.0,
            'initial_portfolio_value': 1000000.0,
            'initial_cash': 1000000.0
        }
    
    async def initialize(self) -> bool:
        """
        非同期初期化
        
        Returns:
            bool: 初期化成功フラグ
        """
        try:
            self.logger.info("リアルタイム実行エンジン非同期初期化開始")
            
            # マーケット時間管理初期化（未実装: copilot-instructions.md違反のモックコンポーネント削除）
            # TODO: 実際のMarketTimeManagerコンポーネントの実装が必要
            self.logger.error("MarketTimeManagerが未実装です")
            raise NotImplementedError("MarketTimeManagerの実装が必要です")
            
        except Exception as e:
            self.logger.error(f"初期化エラー: {e}")
            self.logger.error(f"トレースバック: {traceback.format_exc()}")
            return False
    
    # _create_mock_market_time_manager() メソッドを削除
    # 理由: copilot-instructions.md違反（モックコンポーネント常時使用）
    # 削除日: 2025-12-02
    # 代替策: 実際のMarketTimeManagerコンポーネントの実装が必要
    
    # _create_mock_emergency_detector() メソッドを削除
    # 理由: copilot-instructions.md違反（モックコンポーネント常時使用）
    # 削除日: 2025-12-02
    # 代替策: 実際のEmergencyDetectorコンポーネントの実装が必要
    
    async def start_execution(self, mode: ExecutionMode = ExecutionMode.SIMULATION) -> bool:
        """
        実行開始
        
        Args:
            mode (ExecutionMode): 実行モード
            
        Returns:
            bool: 開始成功フラグ
        """
        try:
            if self.status != ExecutionStatus.STOPPED:
                self.logger.warning(f"既に実行中または停止中です: {self.status}")
                return False
            
            self.logger.info(f"リアルタイム実行開始: {mode.value}")
            self.status = ExecutionStatus.STARTING
            self.mode = mode
            self.start_time = datetime.now()
            self.execution_stats['start_time'] = self.start_time
            
            # 市場時間チェック
            if not await self.market_time_manager.is_market_open():
                self.logger.warning("市場時間外のため実行をスキップ")
                # テストモードでは継続
                if mode == ExecutionMode.SIMULATION:
                    self.logger.info("シミュレーションモードのため実行継続")
                else:
                    self.status = ExecutionStatus.STOPPED
                    return False
            
            # 初期化確認
            if not await self.initialize():
                self.status = ExecutionStatus.ERROR
                return False
            
            # 実行開始イベント送信
            await self.add_event(ExecutionEvent(
                event_type=EventType.SYSTEM_EVENT,
                timestamp=datetime.now(),
                data={
                    'action': 'execution_started',
                    'mode': mode.value,
                    'config': self.config
                },
                priority=0,
                source='execution_engine'
            ))
            
            self.status = ExecutionStatus.RUNNING
            self.logger.info("リアルタイム実行開始完了")
            return True
            
        except Exception as e:
            self.logger.error(f"実行開始エラー: {e}")
            self.status = ExecutionStatus.ERROR
            return False
    
    async def stop_execution(self, emergency: bool = False) -> bool:
        """
        実行停止
        
        Args:
            emergency (bool): 緊急停止フラグ
            
        Returns:
            bool: 停止成功フラグ
        """
        try:
            if emergency:
                self.logger.critical("緊急停止実行")
                self.status = ExecutionStatus.EMERGENCY_STOP
            else:
                self.logger.info("通常停止実行")
                self.status = ExecutionStatus.STOPPED
            
            # シャットダウンイベント設定
            self.shutdown_event.set()
            
            # ワーカータスク停止
            await self._stop_workers()
            
            # 最終状態保存
            await self._save_final_state()
            
            # 統計更新
            if self.start_time:
                self.execution_stats['uptime_seconds'] = (
                    datetime.now() - self.start_time
                ).total_seconds()
            
            self.logger.info("リアルタイム実行停止完了")
            return True
            
        except Exception as e:
            self.logger.error(f"停止エラー: {e}")
            return False
    
    async def add_event(self, event: ExecutionEvent) -> bool:
        """
        イベント追加
        
        Args:
            event (ExecutionEvent): 実行イベント
            
        Returns:
            bool: 追加成功フラグ
        """
        try:
            # 優先度判定
            if event.priority == 0:  # 高優先度
                await self.high_priority_queue.put(event)
            else:
                await self.event_queue.put(event)
            
            self.execution_stats['total_events'] += 1
            return True
            
        except asyncio.QueueFull:
            self.logger.error("イベントキューが満杯です")
            return False
        except Exception as e:
            self.logger.error(f"イベント追加エラー: {e}")
            return False
    
    async def _initialize_portfolio_state(self):
        """ポートフォリオ状態初期化"""
        try:
            self.portfolio_state = PortfolioState(
                timestamp=datetime.now(),
                total_value=self.config.get('initial_portfolio_value', 1000000.0),
                positions={},
                cash=self.config.get('initial_cash', 1000000.0),
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                risk_metrics={}
            )
            self.logger.info("ポートフォリオ状態初期化完了")
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ初期化エラー: {e}")
            raise
    
    async def _start_event_workers(self):
        """イベント処理ワーカー起動"""
        try:
            worker_count = self.config.get('event_worker_count', 4)
            
            for i in range(worker_count):
                # 通常優先度ワーカー
                task = asyncio.create_task(self._event_worker(f"worker_{i}"))
                self.worker_tasks.append(task)
                
                # 高優先度ワーカー
                task = asyncio.create_task(self._high_priority_worker(f"hp_worker_{i}"))
                self.worker_tasks.append(task)
            
            self.logger.info(f"イベントワーカー {len(self.worker_tasks)} 個起動")
            
        except Exception as e:
            self.logger.error(f"ワーカー起動エラー: {e}")
            raise
    
    async def _start_polling_tasks(self):
        """ポーリングタスク起動"""
        try:
            # 市場データポーリング
            if self.config.get('enable_market_polling', True):
                task = asyncio.create_task(self._market_data_poller())
                self.polling_tasks.append(task)
            
            # パフォーマンス監視ポーリング
            task = asyncio.create_task(self._performance_monitor())
            self.polling_tasks.append(task)
            
            # 緊急事態監視ポーリング
            task = asyncio.create_task(self._emergency_monitor())
            self.polling_tasks.append(task)
            
            self.logger.info(f"ポーリングタスク {len(self.polling_tasks)} 個起動")
            
        except Exception as e:
            self.logger.error(f"ポーリングタスク起動エラー: {e}")
            raise
    
    async def _event_worker(self, worker_id: str):
        """イベント処理ワーカー"""
        self.logger.info(f"イベントワーカー {worker_id} 開始")
        
        while not self.shutdown_event.is_set():
            try:
                # タイムアウト付きでイベント取得
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                # イベント処理
                start_time = time.time()
                result = await self._process_event(event)
                processing_time = (time.time() - start_time) * 1000
                
                # 結果記録
                if result.success:
                    self.execution_stats['successful_events'] += 1
                else:
                    self.execution_stats['failed_events'] += 1
                
                # パフォーマンス統計更新
                self._update_performance_metrics(processing_time)
                
                # イベント完了マーク
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"ワーカー {worker_id} エラー: {e}")
                
        self.logger.info(f"イベントワーカー {worker_id} 終了")
    
    async def _high_priority_worker(self, worker_id: str):
        """高優先度イベント処理ワーカー"""
        self.logger.info(f"高優先度ワーカー {worker_id} 開始")
        
        while not self.shutdown_event.is_set():
            try:
                event = await asyncio.wait_for(
                    self.high_priority_queue.get(),
                    timeout=1.0
                )
                
                start_time = time.time()
                result = await self._process_event(event)
                processing_time = (time.time() - start_time) * 1000
                
                if result.success:
                    self.execution_stats['successful_events'] += 1
                else:
                    self.execution_stats['failed_events'] += 1
                
                self._update_performance_metrics(processing_time)
                self.high_priority_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"高優先度ワーカー {worker_id} エラー: {e}")
                
        self.logger.info(f"高優先度ワーカー {worker_id} 終了")
    
    async def _process_event(self, event: ExecutionEvent) -> ExecutionResult:
        """
        イベント処理
        
        Args:
            event (ExecutionEvent): 処理対象イベント
            
        Returns:
            ExecutionResult: 処理結果
        """
        try:
            result_data = {}
            
            if event.event_type == EventType.MARKET_DATA:
                result_data = await self._process_market_data(event.data)
            elif event.event_type == EventType.TRADE_SIGNAL:
                result_data = await self._process_trade_signal(event.data)
            elif event.event_type == EventType.PORTFOLIO_UPDATE:
                result_data = await self._process_portfolio_update(event.data)
            elif event.event_type == EventType.RISK_ALERT:
                result_data = await self._process_risk_alert(event.data)
            elif event.event_type == EventType.SYSTEM_EVENT:
                result_data = await self._process_system_event(event.data)
            elif event.event_type == EventType.USER_COMMAND:
                result_data = await self._process_user_command(event.data)
            
            event.processed = True
            
            return ExecutionResult(
                success=True,
                timestamp=datetime.now(),
                event_id=str(id(event)),
                execution_time_ms=0.0,
                result_data=result_data
            )
            
        except Exception as e:
            self.logger.error(f"イベント処理エラー: {e}")
            return ExecutionResult(
                success=False,
                timestamp=datetime.now(),
                event_id=str(id(event)),
                execution_time_ms=0.0,
                result_data={},
                error_message=str(e)
            )
    
    async def _process_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """市場データ処理"""
        # 市場データの処理ロジック
        return {'processed': 'market_data', 'data': data}
    
    async def _process_trade_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """取引シグナル処理"""
        # 取引シグナルの処理ロジック
        return {'processed': 'trade_signal', 'data': data}
    
    async def _process_portfolio_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """ポートフォリオ更新処理"""
        # ポートフォリオ更新ロジック
        return {'processed': 'portfolio_update', 'data': data}
    
    async def _process_risk_alert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """リスクアラート処理"""
        # リスクアラートの処理ロジック
        return {'processed': 'risk_alert', 'data': data}
    
    async def _process_system_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """システムイベント処理"""
        # システムイベントの処理ロジック
        return {'processed': 'system_event', 'data': data}
    
    async def _process_user_command(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """ユーザーコマンド処理"""
        # ユーザーコマンドの処理ロジック
        return {'processed': 'user_command', 'data': data}
    
    async def _market_data_poller(self):
        """市場データポーリング"""
        self.logger.info("市場データポーリング開始")
        
        poll_interval = self.config.get('market_data_poll_interval', 1.0)
        
        while not self.shutdown_event.is_set():
            try:
                # 市場データ取得とイベント生成
                await self.add_event(ExecutionEvent(
                    event_type=EventType.MARKET_DATA,
                    timestamp=datetime.now(),
                    data={'source': 'polling', 'timestamp': datetime.now().isoformat()},
                    source='market_poller'
                ))
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                self.logger.error(f"市場データポーリングエラー: {e}")
                await asyncio.sleep(poll_interval)
    
    async def _performance_monitor(self):
        """パフォーマンス監視"""
        self.logger.info("パフォーマンス監視開始")
        
        monitor_interval = self.config.get('performance_monitor_interval', 10.0)
        
        while not self.shutdown_event.is_set():
            try:
                # パフォーマンス指標計算
                self.performance_metrics['queue_depth'] = self.event_queue.qsize()
                
                if self.execution_stats['total_events'] > 0:
                    self.performance_metrics['error_rate'] = (
                        self.execution_stats['failed_events'] / 
                        self.execution_stats['total_events']
                    ) * 100
                
                self.logger.debug(f"パフォーマンス指標: {self.performance_metrics}")
                
                await asyncio.sleep(monitor_interval)
                
            except Exception as e:
                self.logger.error(f"パフォーマンス監視エラー: {e}")
                await asyncio.sleep(monitor_interval)
    
    async def _emergency_monitor(self):
        """緊急事態監視"""
        self.logger.info("緊急事態監視開始")
        
        monitor_interval = self.config.get('emergency_monitor_interval', 5.0)
        
        while not self.shutdown_event.is_set():
            try:
                if self.emergency_detector:
                    emergency_status = await self.emergency_detector.check_emergency()
                    
                    if emergency_status.is_emergency:
                        self.logger.critical(f"緊急事態検出: {emergency_status.message}")
                        await self.stop_execution(emergency=True)
                        break
                
                await asyncio.sleep(monitor_interval)
                
            except Exception as e:
                self.logger.error(f"緊急事態監視エラー: {e}")
                await asyncio.sleep(monitor_interval)
    
    def _update_performance_metrics(self, processing_time_ms: float):
        """パフォーマンス指標更新"""
        # 移動平均で処理時間更新
        alpha = 0.1
        self.performance_metrics['avg_processing_time_ms'] = (
            alpha * processing_time_ms + 
            (1 - alpha) * self.performance_metrics['avg_processing_time_ms']
        )
    
    async def _stop_workers(self):
        """ワーカー停止"""
        try:
            # 全タスクのキャンセルを待機
            all_tasks = self.worker_tasks + self.polling_tasks
            
            for task in all_tasks:
                task.cancel()
            
            # タスク完了待機
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)
            
            self.worker_tasks.clear()
            self.polling_tasks.clear()
            
            self.logger.info("全ワーカー停止完了")
            
        except Exception as e:
            self.logger.error(f"ワーカー停止エラー: {e}")
    
    async def _save_final_state(self):
        """最終状態保存"""
        try:
            final_state = {
                'execution_stats': self.execution_stats,
                'performance_metrics': self.performance_metrics,
                'portfolio_state': self.portfolio_state.__dict__ if self.portfolio_state else {},
                'config': self.config,
                'stop_time': datetime.now().isoformat()
            }
            
            # 状態保存ファイル
            state_file = Path(project_root) / 'logs' / f'final_state_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            state_file.parent.mkdir(exist_ok=True)
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(final_state, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"最終状態保存完了: {state_file}")
            
        except Exception as e:
            self.logger.error(f"最終状態保存エラー: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        実行状態取得
        
        Returns:
            Dict[str, Any]: 状態情報
        """
        return {
            'status': self.status.value,
            'mode': self.mode.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'execution_stats': self.execution_stats,
            'performance_metrics': self.performance_metrics,
            'queue_sizes': {
                'normal': self.event_queue.qsize(),
                'high_priority': self.high_priority_queue.qsize()
            },
            'portfolio_value': self.portfolio_state.total_value if self.portfolio_state else 0
        }

# 使用例とテスト
async def demo_realtime_execution():
    """リアルタイム実行デモ"""
    logger = setup_logger(__name__)
    logger.info("リアルタイム実行エンジンデモ開始")
    
    try:
        # エンジン初期化
        engine = RealtimeExecutionEngine()
        
        # 実行開始
        success = await engine.start_execution(ExecutionMode.SIMULATION)
        if not success:
            logger.error("実行開始失敗")
            return
        
        # テストイベント送信
        for i in range(5):
            await engine.add_event(ExecutionEvent(
                event_type=EventType.MARKET_DATA,
                timestamp=datetime.now(),
                data={'test_data': f'test_{i}', 'value': i * 100},
                source='demo'
            ))
        
        # 少し実行
        await asyncio.sleep(5)
        
        # 状態確認
        status = engine.get_status()
        logger.info(f"実行状態: {status}")
        
        # 停止
        await engine.stop_execution()
        
        logger.info("リアルタイム実行エンジンデモ完了")
        
    except Exception as e:
        logger.error(f"デモエラー: {e}")
        logger.error(f"トレースバック: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(demo_realtime_execution())
