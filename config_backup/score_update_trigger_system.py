"""
Module: Score Update Trigger System
File: score_update_trigger_system.py
Description: 
  2-3-3「スコアアップデートトリガー設計」
  パフォーマンスデータの蓄積に応じたスコア更新トリガー管理システム

Author: GitHub Copilot
Created: 2025-07-13
Modified: 2025-07-13
"""
import asyncio
import logging
import threading
import time
import json
import queue
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd

# 既存モジュールのインポート
try:
    from .enhanced_score_history_manager import EnhancedScoreHistoryManager, EnhancedScoreHistoryEntry
    from .strategy_scoring_model import StrategyScore, StrategyScoreCalculator
    from .time_decay_factor import TimeDecayFactor
    from .score_history_manager import ScoreHistoryManager
except ImportError:
    # 直接実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from enhanced_score_history_manager import EnhancedScoreHistoryManager, EnhancedScoreHistoryEntry
        from strategy_scoring_model import StrategyScore, StrategyScoreCalculator
        from time_decay_factor import TimeDecayFactor
        from score_history_manager import ScoreHistoryManager
    except ImportError:
        # さらに上の階層から
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config.enhanced_score_history_manager import EnhancedScoreHistoryManager, EnhancedScoreHistoryEntry
        from config.strategy_scoring_model import StrategyScore, StrategyScoreCalculator
        from config.time_decay_factor import TimeDecayFactor
        from config.score_history_manager import ScoreHistoryManager

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """トリガータイプ定義"""
    TIME_BASED = "time_based"           # 時間ベース
    EVENT_BASED = "event_based"         # イベントベース  
    THRESHOLD_BASED = "threshold_based" # 閾値ベース
    QUALITY_BASED = "quality_based"     # 品質ベース
    MANUAL = "manual"                   # 手動

class TriggerPriority(Enum):
    """トリガー優先度"""
    CRITICAL = 1    # 緊急（即座に処理）
    HIGH = 2        # 高（優先処理）
    MEDIUM = 3      # 中（通常処理）
    LOW = 4         # 低（バッチ処理可）

@dataclass
class TriggerCondition:
    """トリガー条件定義"""
    condition_id: str
    trigger_type: TriggerType
    priority: TriggerPriority
    
    # 条件パラメータ
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 実行設定
    enabled: bool = True
    cooldown_seconds: int = 300  # 再実行制限時間
    max_retries: int = 3
    
    # フィルタ条件
    strategy_filter: Optional[List[str]] = None
    ticker_filter: Optional[List[str]] = None
    
    # メタデータ
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None

@dataclass
class TriggerEvent:
    """トリガーイベント"""
    event_id: str
    condition_id: str
    trigger_type: TriggerType
    priority: TriggerPriority
    
    # イベントデータ
    strategy_name: str
    ticker: str = "DEFAULT"
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    # 実行情報
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    
    # 結果
    execution_result: Optional[Dict[str, Any]] = None
    retry_count: int = 0

@dataclass
class UpdateRequest:
    """更新リクエスト"""
    request_id: str
    strategy_name: str
    ticker: str
    trigger_type: TriggerType
    priority: int = 1
    market_data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class UpdateResult:
    """更新結果"""
    request_id: str
    strategy_name: str
    ticker: str
    success: bool
    old_score: Optional[float] = None
    new_score: Optional[float] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    updated_at: datetime = field(default_factory=datetime.now)

class ScoreUpdateTriggerSystem:
    """
    スコアアップデートトリガーシステム
    
    複数のトリガー条件を管理し、条件が満たされた際に
    スコア更新を非同期で実行する統合システム
    """
    
    def __init__(self, 
                 enhanced_manager: Optional[EnhancedScoreHistoryManager] = None,
                 score_calculator: Optional[StrategyScoreCalculator] = None):
        """
        初期化
        
        Parameters:
            enhanced_manager: 拡張スコア履歴管理システム
            score_calculator: スコア計算器
        """
        # 既存システムとの統合
        self.enhanced_manager = enhanced_manager or self._create_enhanced_manager()
        self.score_calculator = score_calculator or StrategyScoreCalculator()
        
        # トリガー管理
        self.trigger_conditions: Dict[str, TriggerCondition] = {}
        self.active_triggers: Set[str] = set()
        self.trigger_queue = queue.PriorityQueue()
        self.event_history: List[TriggerEvent] = []
        
        # 実行制御
        self.is_running = False
        self.workers: List[threading.Thread] = []
        self.max_workers = 3
        
        # 統計情報
        self.stats = {
            "total_triggers": 0,
            "successful_triggers": 0,
            "failed_triggers": 0,
            "last_trigger_time": None,
            "average_execution_time": 0.0
        }
        
        # コールバック
        self.trigger_callbacks: List[Callable[[TriggerEvent], None]] = []
        
        # デフォルトトリガー条件を設定
        self._setup_default_triggers()
        
        logger.info("Score Update Trigger System initialized")
    
    def _create_enhanced_manager(self) -> EnhancedScoreHistoryManager:
        """拡張管理システム作成"""
        try:
            return EnhancedScoreHistoryManager()
        except Exception as e:
            logger.warning(f"Failed to create EnhancedScoreHistoryManager, using basic: {e}")
            # フォールバック：基本的な管理システム
            return None
    
    def _setup_default_triggers(self):
        """デフォルトトリガー条件設定"""
        
        # 1. 定期更新トリガー
        self.add_trigger_condition(TriggerCondition(
            condition_id="daily_update",
            trigger_type=TriggerType.TIME_BASED,
            priority=TriggerPriority.LOW,
            parameters={
                "interval_hours": 24,
                "execution_time": "09:00"  # 市場開始前
            },
            description="日次定期更新"
        ))
        
        # 2. スコア変化トリガー
        self.add_trigger_condition(TriggerCondition(
            condition_id="score_change_threshold",
            trigger_type=TriggerType.THRESHOLD_BASED,
            priority=TriggerPriority.HIGH,
            parameters={
                "score_change_threshold": 0.15,  # 15%変化
                "monitoring_window_hours": 6
            },
            description="スコア大幅変化検出"
        ))
        
        # 3. データ品質トリガー
        self.add_trigger_condition(TriggerCondition(
            condition_id="data_quality_degradation",
            trigger_type=TriggerType.QUALITY_BASED,
            priority=TriggerPriority.MEDIUM,
            parameters={
                "min_data_completeness": 0.95,  # 95%以上のデータ完全性
                "max_missing_days": 2
            },
            description="データ品質劣化検出"
        ))
    
    def add_trigger_condition(self, condition: TriggerCondition):
        """トリガー条件追加"""
        self.trigger_conditions[condition.condition_id] = condition
        if condition.enabled:
            self.active_triggers.add(condition.condition_id)
        
        logger.info(f"Added trigger condition: {condition.condition_id}")
    
    def remove_trigger_condition(self, condition_id: str):
        """トリガー条件削除"""
        if condition_id in self.trigger_conditions:
            del self.trigger_conditions[condition_id]
            self.active_triggers.discard(condition_id)
            logger.info(f"Removed trigger condition: {condition_id}")
    
    def enable_trigger(self, condition_id: str):
        """トリガー有効化"""
        if condition_id in self.trigger_conditions:
            self.trigger_conditions[condition_id].enabled = True
            self.active_triggers.add(condition_id)
            logger.info(f"Enabled trigger: {condition_id}")
    
    def disable_trigger(self, condition_id: str):
        """トリガー無効化"""
        if condition_id in self.trigger_conditions:
            self.trigger_conditions[condition_id].enabled = False
            self.active_triggers.discard(condition_id)
            logger.info(f"Disabled trigger: {condition_id}")
    
    def start(self):
        """トリガーシステム開始"""
        if self.is_running:
            logger.warning("Trigger system is already running")
            return
        
        self.is_running = True
        
        # ワーカースレッド開始
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._trigger_worker_loop,
                name=f"TriggerWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # 監視スレッド開始
        monitor = threading.Thread(
            target=self._monitoring_loop,
            name="TriggerMonitor",
            daemon=True
        )
        monitor.start()
        self.workers.append(monitor)
        
        logger.info(f"Trigger system started with {len(self.workers)} workers")
    
    def stop(self):
        """トリガーシステム停止"""
        if not self.is_running:
            return
        
        logger.info("Stopping trigger system...")
        self.is_running = False
        
        # ワーカー終了
        for _ in range(len(self.workers)):
            try:
                self.trigger_queue.put_nowait((0, None))  # 停止シグナル
            except queue.Full:
                pass
        
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        logger.info("Trigger system stopped")
    
    def _monitoring_loop(self):
        """監視ループ"""
        logger.debug("Trigger monitoring started")
        
        while self.is_running:
            try:
                # アクティブなトリガー条件をチェック
                for condition_id in list(self.active_triggers):
                    if condition_id in self.trigger_conditions:
                        condition = self.trigger_conditions[condition_id]
                        
                        if self._should_evaluate_condition(condition):
                            self._evaluate_trigger_condition(condition)
                
                # 監視間隔
                time.sleep(60)  # 1分間隔
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(300)  # エラー時は5分待機
        
        logger.debug("Trigger monitoring stopped")
    
    def _should_evaluate_condition(self, condition: TriggerCondition) -> bool:
        """条件評価が必要かチェック"""
        # クールダウン期間チェック
        if condition.last_triggered:
            cooldown_end = condition.last_triggered + timedelta(seconds=condition.cooldown_seconds)
            if datetime.now() < cooldown_end:
                return False
        
        # 時間ベーストリガーの特別処理
        if condition.trigger_type == TriggerType.TIME_BASED:
            return self._check_time_based_condition(condition)
        
        return True
    
    def _check_time_based_condition(self, condition: TriggerCondition) -> bool:
        """時間ベース条件チェック"""
        params = condition.parameters
        
        if "interval_hours" in params:
            interval = timedelta(hours=params["interval_hours"])
            if condition.last_triggered:
                next_trigger = condition.last_triggered + interval
                return datetime.now() >= next_trigger
            return True
        
        if "execution_time" in params:
            exec_time = params["execution_time"]  # "HH:MM"形式
            current_time = datetime.now().strftime("%H:%M")
            return current_time == exec_time
        
        return False
    
    def _evaluate_trigger_condition(self, condition: TriggerCondition):
        """トリガー条件評価"""
        try:
            if condition.trigger_type == TriggerType.TIME_BASED:
                self._evaluate_time_based_trigger(condition)
            elif condition.trigger_type == TriggerType.THRESHOLD_BASED:
                self._evaluate_threshold_trigger(condition)
            elif condition.trigger_type == TriggerType.QUALITY_BASED:
                self._evaluate_quality_trigger(condition)
            elif condition.trigger_type == TriggerType.EVENT_BASED:
                self._evaluate_event_trigger(condition)
                
        except Exception as e:
            logger.error(f"Failed to evaluate trigger {condition.condition_id}: {e}")
    
    def _evaluate_time_based_trigger(self, condition: TriggerCondition):
        """時間ベーストリガー評価"""
        # 全ての戦略に対して更新をトリガー
        strategies = self._get_active_strategies()
        
        for strategy_name in strategies:
            tickers = self._get_strategy_tickers(strategy_name)
            
            for ticker in tickers:
                if self._should_trigger_for_strategy(condition, strategy_name, ticker):
                    self._queue_trigger_event(condition, strategy_name, ticker, {
                        "trigger_reason": "scheduled_update",
                        "execution_time": datetime.now().isoformat()
                    })
    
    def _evaluate_threshold_trigger(self, condition: TriggerCondition):
        """閾値ベーストリガー評価"""
        params = condition.parameters
        threshold = params.get("score_change_threshold", 0.1)
        window_hours = params.get("monitoring_window_hours", 6)
        
        strategies = self._get_active_strategies()
        
        for strategy_name in strategies:
            # 最近のスコア履歴を取得
            recent_scores = self._get_recent_scores(strategy_name, window_hours)
            
            if len(recent_scores) >= 2:
                if recent_scores[-1] != 0:  # ゼロ除算回避
                    score_change = abs(recent_scores[0] - recent_scores[-1]) / recent_scores[-1]
                    
                    if score_change >= threshold:
                        tickers = self._get_strategy_tickers(strategy_name)
                        
                        for ticker in tickers:
                            self._queue_trigger_event(condition, strategy_name, ticker, {
                                "trigger_reason": "score_threshold_exceeded",
                                "score_change": score_change,
                                "threshold": threshold
                            })
    
    def _evaluate_quality_trigger(self, condition: TriggerCondition):
        """品質ベーストリガー評価"""
        params = condition.parameters
        min_completeness = params.get("min_data_completeness", 0.95)
        max_missing_days = params.get("max_missing_days", 2)
        
        strategies = self._get_active_strategies()
        
        for strategy_name in strategies:
            quality_metrics = self._assess_data_quality(strategy_name)
            
            if (quality_metrics["completeness"] < min_completeness or 
                quality_metrics["missing_days"] > max_missing_days):
                
                tickers = self._get_strategy_tickers(strategy_name)
                
                for ticker in tickers:
                    self._queue_trigger_event(condition, strategy_name, ticker, {
                        "trigger_reason": "data_quality_degradation",
                        "quality_metrics": quality_metrics
                    })
    
    def _evaluate_event_trigger(self, condition: TriggerCondition):
        """イベントベーストリガー評価"""
        # 市場データの異常値検出（プレースホルダー）
        market_events = self._detect_market_events(condition.parameters)
        
        for event in market_events:
            affected_strategies = self._get_strategies_affected_by_event(event)
            
            for strategy_name in affected_strategies:
                tickers = self._get_strategy_tickers(strategy_name)
                
                for ticker in tickers:
                    self._queue_trigger_event(condition, strategy_name, ticker, {
                        "trigger_reason": "market_event",
                        "event_type": event["type"],
                        "event_data": event
                    })
    
    def _queue_trigger_event(self, 
                           condition: TriggerCondition,
                           strategy_name: str,
                           ticker: str,
                           event_data: Dict[str, Any]):
        """トリガーイベントをキューに追加"""
        event = TriggerEvent(
            event_id=f"{condition.condition_id}_{strategy_name}_{ticker}_{int(time.time())}",
            condition_id=condition.condition_id,
            trigger_type=condition.trigger_type,
            priority=condition.priority,
            strategy_name=strategy_name,
            ticker=ticker,
            event_data=event_data
        )
        
        # 優先度付きキューに追加
        try:
            self.trigger_queue.put_nowait((condition.priority.value, event))
            
            # 最終トリガー時刻更新
            condition.last_triggered = datetime.now()
            
            logger.debug(f"Queued trigger event: {event.event_id}")
        except queue.Full:
            logger.warning(f"Trigger queue is full, dropping event: {event.event_id}")
    
    def _trigger_worker_loop(self):
        """トリガーワーカーループ"""
        worker_name = threading.current_thread().name
        logger.debug(f"Trigger worker {worker_name} started")
        
        while self.is_running:
            try:
                # キューからイベント取得
                priority, event = self.trigger_queue.get(timeout=1.0)
                
                # 停止シグナルチェック
                if event is None:
                    break
                
                # イベント実行
                self._execute_trigger_event(event)
                
                self.trigger_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Trigger worker {worker_name} error: {e}")
        
        logger.debug(f"Trigger worker {worker_name} stopped")
    
    def _execute_trigger_event(self, event: TriggerEvent):
        """トリガーイベント実行"""
        start_time = time.time()
        
        try:
            # スコア更新実行
            old_score = self._get_current_score(event.strategy_name, event.ticker)
            new_score = self._calculate_new_score(event.strategy_name, event.ticker)
            
            if new_score is not None:
                # スコア履歴に保存
                if self.enhanced_manager:
                    self.enhanced_manager.add_enhanced_entry(
                        strategy_name=event.strategy_name,
                        strategy_score=new_score,
                        metadata={
                            "trigger_type": event.trigger_type.value,
                            "trigger_condition": event.condition_id,
                            "event_data": event.event_data
                        }
                    )
                
                execution_time = time.time() - start_time
                event.executed_at = datetime.now()
                event.execution_result = {
                    "status": "success",
                    "execution_time": execution_time,
                    "old_score": old_score,
                    "new_score": new_score.total_score if hasattr(new_score, 'total_score') else None
                }
                
                # 統計更新
                self._update_trigger_stats(event, True, execution_time)
                
                logger.debug(f"Executed trigger event: {event.event_id}")
            else:
                raise ValueError("Failed to calculate new score")
            
        except Exception as e:
            execution_time = time.time() - start_time
            event.execution_result = {
                "status": "error",
                "error_message": str(e),
                "execution_time": execution_time
            }
            
            self._update_trigger_stats(event, False, execution_time)
            logger.error(f"Failed to execute trigger event {event.event_id}: {e}")
        
        # イベント履歴に追加
        self.event_history.append(event)
        
        # 履歴サイズ制限
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-500:]
        
        # コールバック実行
        for callback in self.trigger_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Trigger callback error: {e}")
    
    def _get_current_score(self, strategy_name: str, ticker: str) -> Optional[float]:
        """現在のスコア取得"""
        try:
            if self.enhanced_manager:
                entries = self.enhanced_manager.get_entries(strategy_name, limit=1)
                if entries and hasattr(entries[0].strategy_score, 'total_score'):
                    return entries[0].strategy_score.total_score
        except Exception as e:
            logger.debug(f"Failed to get current score: {e}")
        
        return None
    
    def _calculate_new_score(self, strategy_name: str, ticker: str) -> Optional[StrategyScore]:
        """新しいスコア計算"""
        try:
            # サンプルデータでスコア計算（実際の実装では市場データを使用）
            sample_score = StrategyScore(
                strategy_name=strategy_name,
                ticker=ticker,
                total_score=0.75,  # 実際の計算結果に置き換え
                component_scores={
                    "performance": 0.8,
                    "stability": 0.7,
                    "risk_adjusted": 0.75,
                    "reliability": 0.8
                },
                trend_fitness=0.7,
                confidence=0.85,
                metadata={"update_type": "trigger_based"},
                calculated_at=datetime.now()
            )
            
            return sample_score
            
        except Exception as e:
            logger.error(f"Failed to calculate new score for {strategy_name}_{ticker}: {e}")
            return None
    
    # =========================================================================
    # ヘルパーメソッド（実装依存）
    # =========================================================================
    
    def _get_active_strategies(self) -> List[str]:
        """アクティブ戦略リスト取得"""
        # プレースホルダー実装
        return ["vwap_bounce_strategy", "momentum_strategy", "mean_reversion_strategy"]
    
    def _get_strategy_tickers(self, strategy_name: str) -> List[str]:
        """戦略のティッカーリスト取得"""
        # プレースホルダー実装
        return ["AAPL", "MSFT", "GOOGL"]
    
    def _should_trigger_for_strategy(self, 
                                   condition: TriggerCondition,
                                   strategy_name: str,
                                   ticker: str) -> bool:
        """戦略に対してトリガーすべきかチェック"""
        # フィルタチェック
        if condition.strategy_filter and strategy_name not in condition.strategy_filter:
            return False
        
        if condition.ticker_filter and ticker not in condition.ticker_filter:
            return False
        
        return True
    
    def _get_recent_scores(self, strategy_name: str, hours: int) -> List[float]:
        """最近のスコア取得"""
        try:
            if self.enhanced_manager:
                since_time = datetime.now() - timedelta(hours=hours)
                entries = self.enhanced_manager.get_entries(
                    strategy_name=strategy_name,
                    since=since_time.isoformat()
                )
                
                scores = []
                for entry in entries:
                    if hasattr(entry.strategy_score, 'total_score'):
                        scores.append(entry.strategy_score.total_score)
                
                return scores
            
        except Exception as e:
            logger.error(f"Failed to get recent scores: {e}")
        
        return []
    
    def _assess_data_quality(self, strategy_name: str) -> Dict[str, float]:
        """データ品質評価"""
        # プレースホルダー実装
        return {
            "completeness": 0.98,  # 98%のデータ完全性
            "missing_days": 0,     # 欠損日数
            "last_update_hours": 2  # 最終更新からの時間
        }
    
    def _detect_market_events(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """市場イベント検出"""
        # プレースホルダー実装
        events = []
        
        # 現在時刻で条件チェック
        current_hour = datetime.now().hour
        if current_hour in [9, 15]:  # 市場開始・終了時
            events.append({
                "type": "market_hours_event",
                "severity": "medium",
                "detected_at": datetime.now().isoformat(),
                "description": f"Market {'open' if current_hour == 9 else 'close'} detected"
            })
        
        return events
    
    def _get_strategies_affected_by_event(self, event: Dict[str, Any]) -> List[str]:
        """イベントの影響を受ける戦略取得"""
        # イベントタイプに基づいて影響戦略を決定
        event_type = event.get("type", "")
        
        if "volatility" in event_type:
            return ["momentum_strategy"]
        elif "volume" in event_type:
            return ["vwap_bounce_strategy"]
        else:
            return self._get_active_strategies()  # 全戦略
    
    def _update_trigger_stats(self, event: TriggerEvent, success: bool, execution_time: float):
        """トリガー統計更新"""
        self.stats["total_triggers"] += 1
        
        if success:
            self.stats["successful_triggers"] += 1
        else:
            self.stats["failed_triggers"] += 1
        
        self.stats["last_trigger_time"] = event.executed_at
        
        # 平均実行時間更新
        total = self.stats["total_triggers"]
        current_avg = self.stats["average_execution_time"]
        self.stats["average_execution_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )
    
    # =========================================================================
    # 公開API
    # =========================================================================
    
    def manual_trigger(self, 
                      strategy_name: str,
                      ticker: str = "DEFAULT",
                      priority: TriggerPriority = TriggerPriority.HIGH,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """手動トリガー実行"""
        event_id = f"manual_{strategy_name}_{ticker}_{int(time.time())}"
        
        event = TriggerEvent(
            event_id=event_id,
            condition_id="manual",
            trigger_type=TriggerType.MANUAL,
            priority=priority,
            strategy_name=strategy_name,
            ticker=ticker,
            event_data=metadata or {"trigger_reason": "manual_request"}
        )
        
        try:
            self.trigger_queue.put_nowait((priority.value, event))
            logger.info(f"Manual trigger queued: {event_id}")
            return event_id
        except queue.Full:
            logger.error(f"Failed to queue manual trigger: queue full")
            return ""
    
    def get_trigger_statistics(self) -> Dict[str, Any]:
        """トリガー統計取得"""
        return {
            **self.stats,
            "active_conditions": len(self.active_triggers),
            "total_conditions": len(self.trigger_conditions),
            "queue_size": self.trigger_queue.qsize(),
            "recent_events": len([e for e in self.event_history 
                                if e.created_at > datetime.now() - timedelta(hours=24)])
        }
    
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """最近のトリガーイベント取得"""
        recent_events = sorted(
            self.event_history,
            key=lambda x: x.created_at,
            reverse=True
        )[:limit]
        
        return [
            {
                "event_id": event.event_id,
                "condition_id": event.condition_id,
                "trigger_type": event.trigger_type.value,
                "priority": event.priority.value,
                "strategy_name": event.strategy_name,
                "ticker": event.ticker,
                "created_at": event.created_at.isoformat(),
                "executed_at": event.executed_at.isoformat() if event.executed_at else None,
                "status": event.execution_result.get("status") if event.execution_result else "pending"
            }
            for event in recent_events
        ]
    
    def add_trigger_callback(self, callback: Callable[[TriggerEvent], None]):
        """トリガーコールバック追加"""
        self.trigger_callbacks.append(callback)


# =============================================================================
# ユーティリティ関数
# =============================================================================

def create_trigger_system(enhanced_manager: Optional[EnhancedScoreHistoryManager] = None,
                         score_calculator: Optional[StrategyScoreCalculator] = None) -> ScoreUpdateTriggerSystem:
    """トリガーシステム作成"""
    return ScoreUpdateTriggerSystem(enhanced_manager, score_calculator)


# =============================================================================
# エクスポート
# =============================================================================

__all__ = [
    "ScoreUpdateTriggerSystem",
    "TriggerType",
    "TriggerPriority", 
    "TriggerCondition",
    "TriggerEvent",
    "UpdateRequest",
    "UpdateResult",
    "create_trigger_system"
]


if __name__ == "__main__":
    # デバッグ用テスト
    logging.basicConfig(level=logging.INFO)
    
    trigger_system = create_trigger_system()
    
    try:
        trigger_system.start()
        
        # 手動トリガーテスト
        event_id = trigger_system.manual_trigger("test_strategy", "TEST")
        print(f"Manual trigger queued: {event_id}")
        
        # 統計情報表示
        time.sleep(2)
        stats = trigger_system.get_trigger_statistics()
        print(f"Trigger Stats: {stats}")
        
    finally:
        trigger_system.stop()
