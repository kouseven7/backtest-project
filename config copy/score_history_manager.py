"""
スコア履歴保存システム (2-3-1)
既存のStrategyScoreシステムと統合したスコア履歴管理機能

主な機能:
1. スコア履歴の保存と管理
2. イベント駆動型の履歴記録
3. 効率的な検索とフィルタリング
4. 統計分析とレポート生成
5. データ整合性とバックアップ管理

作成者: GitHub Copilot
作成日: 2024年
"""

import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, DefaultDict
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
from collections import defaultdict, deque

# 既存のStrategyScoreを使用
try:
    from .strategy_scoring_model import StrategyScore, StrategyScoreCalculator
except ImportError:
    from strategy_scoring_model import StrategyScore, StrategyScoreCalculator

# ロガー設定
logger = logging.getLogger(__name__)

@dataclass
class ScoreHistoryEntry:
    """スコア履歴エントリ - 既存のStrategyScoreを拡張"""
    
    # 基本情報
    strategy_score: StrategyScore
    entry_id: str = field(default_factory=lambda: f"entry_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    
    # イベント情報
    trigger_event: str = "manual"  # manual, scheduled, threshold_change, market_event
    event_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # バージョン管理
    version: str = "1.0"
    parent_entry_id: Optional[str] = None
    
    # アクセス情報
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'entry_id': self.entry_id,
            'strategy_score': self.strategy_score.to_dict(),
            'trigger_event': self.trigger_event,
            'event_metadata': self.event_metadata,
            'version': self.version,
            'parent_entry_id': self.parent_entry_id,
            'accessed_count': self.accessed_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoreHistoryEntry':
        """辞書から復元"""
        # StrategyScoreを復元
        strategy_score_data = data['strategy_score']
        strategy_score_data['calculated_at'] = datetime.fromisoformat(strategy_score_data['calculated_at'])
        strategy_score = StrategyScore(**strategy_score_data)
        
        # ScoreHistoryEntryを復元
        last_accessed = None
        if data.get('last_accessed'):
            last_accessed = datetime.fromisoformat(data['last_accessed'])
        
        return cls(
            strategy_score=strategy_score,
            entry_id=data['entry_id'],
            trigger_event=data.get('trigger_event', 'manual'),
            event_metadata=data.get('event_metadata', {}),
            version=data.get('version', '1.0'),
            parent_entry_id=data.get('parent_entry_id'),
            accessed_count=data.get('accessed_count', 0),
            last_accessed=last_accessed
        )

@dataclass
class ScoreHistoryConfig:
    """スコア履歴システム設定"""
    
    # ストレージ設定
    storage_directory: str = "score_history"
    max_entries_per_file: int = 1000
    compression_enabled: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    # 保持ポリシー
    max_history_days: int = 365
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 168  # 1週間
    
    # パフォーマンス設定
    cache_size: int = 500
    index_enabled: bool = True
    lazy_loading: bool = True
    
    # イベント設定
    event_listeners_enabled: bool = True
    notification_threshold: float = 0.1  # スコア変化10%で通知
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoreHistoryConfig':
        """辞書から設定を復元"""
        return cls(**data)

class ScoreHistoryIndex:
    """スコア履歴インデックス - 高速検索用"""
    
    def __init__(self):
        self.strategy_index: Dict[str, List[str]] = defaultdict(list)  # strategy_name -> [entry_ids]
        self.ticker_index: Dict[str, List[str]] = defaultdict(list)    # ticker -> [entry_ids]
        self.date_index: Dict[str, List[str]] = defaultdict(list)      # date_str -> [entry_ids]
        self.score_index: Dict[str, List[str]] = defaultdict(list)     # score_range -> [entry_ids]
        self._lock = threading.RLock()
    
    def add_entry(self, entry: ScoreHistoryEntry):
        """エントリをインデックスに追加"""
        with self._lock:
            entry_id = entry.entry_id
            score = entry.strategy_score
            
            # 戦略名インデックス
            self.strategy_index[score.strategy_name].append(entry_id)
            
            # ティッカーインデックス
            self.ticker_index[score.ticker].append(entry_id)
            
            # 日付インデックス
            date_str = score.calculated_at.strftime('%Y-%m-%d')
            self.date_index[date_str].append(entry_id)
            
            # スコア範囲インデックス (0.1刻み)
            score_range = f"{int(score.total_score * 10) / 10:.1f}"
            self.score_index[score_range].append(entry_id)
    
    def remove_entry(self, entry: ScoreHistoryEntry):
        """エントリをインデックスから削除"""
        with self._lock:
            entry_id = entry.entry_id
            score = entry.strategy_score
            
            # 各インデックスから削除
            self._safe_remove(self.strategy_index[score.strategy_name], entry_id)
            self._safe_remove(self.ticker_index[score.ticker], entry_id)
            
            date_str = score.calculated_at.strftime('%Y-%m-%d')
            self._safe_remove(self.date_index[date_str], entry_id)
            
            score_range = f"{int(score.total_score * 10) / 10:.1f}"
            self._safe_remove(self.score_index[score_range], entry_id)
    
    def _safe_remove(self, entry_list: List[str], entry_id: str):
        """安全にリストから要素を削除"""
        try:
            entry_list.remove(entry_id)
        except ValueError:
            pass
    
    def search(self, strategy_name: str = None, ticker: str = None, 
               date_range: tuple = None, score_range: tuple = None) -> List[str]:
        """条件に基づいてエントリIDを検索"""
        with self._lock:
            result_sets = []
            
            if strategy_name:
                result_sets.append(set(self.strategy_index.get(strategy_name, [])))
            
            if ticker:
                result_sets.append(set(self.ticker_index.get(ticker, [])))
            
            if date_range:
                start_date, end_date = date_range
                date_entries = set()
                current_date = start_date
                while current_date <= end_date:
                    date_str = current_date.strftime('%Y-%m-%d')
                    date_entries.update(self.date_index.get(date_str, []))
                    current_date += timedelta(days=1)
                result_sets.append(date_entries)
            
            if score_range:
                min_score, max_score = score_range
                score_entries = set()
                score = min_score
                while score <= max_score:
                    score_key = f"{score:.1f}"
                    score_entries.update(self.score_index.get(score_key, []))
                    score += 0.1
                result_sets.append(score_entries)
            
            if not result_sets:
                return []
            
            # 積集合を計算
            result = result_sets[0]
            for result_set in result_sets[1:]:
                result = result.intersection(result_set)
            
            return list(result)

class ScoreHistoryEventManager:
    """スコア履歴イベント管理"""
    
    def __init__(self):
        self.listeners: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def add_listener(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """イベントリスナーを追加"""
        with self._lock:
            self.listeners[event_type].append(callback)
    
    def remove_listener(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """イベントリスナーを削除"""
        with self._lock:
            try:
                self.listeners[event_type].remove(callback)
            except ValueError:
                pass
    
    def trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """イベントを発火"""
        with self._lock:
            for callback in self.listeners.get(event_type, []):
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event listener for {event_type}: {e}")

class ScoreHistoryManager:
    """スコア履歴管理メインクラス"""
    
    def __init__(self, config: ScoreHistoryConfig = None, base_dir: str = None):
        """
        初期化
        
        Args:
            config: 設定オブジェクト
            base_dir: ベースディレクトリ
        """
        self.config = config or ScoreHistoryConfig()
        self.base_dir = Path(base_dir or os.getcwd())
        self.storage_dir = self.base_dir / self.config.storage_directory
        
        # ストレージディレクトリを作成
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 内部状態
        self._history_cache: Dict[str, ScoreHistoryEntry] = {}
        self._recent_entries = deque(maxlen=self.config.cache_size)
        self._lock = threading.RLock()
        
        # インデックスとイベント管理
        self.index = ScoreHistoryIndex() if self.config.index_enabled else None
        self.event_manager = ScoreHistoryEventManager() if self.config.event_listeners_enabled else None
        
        # 初期化
        self._load_existing_history()
        self._setup_cleanup_scheduler()
        
        logger.info(f"ScoreHistoryManager initialized with {len(self._history_cache)} cached entries")
    
    def save_score(self, strategy_score: StrategyScore, 
                   trigger_event: str = "manual",
                   event_metadata: Dict[str, Any] = None) -> str:
        """
        スコアを履歴に保存
        
        Args:
            strategy_score: 保存するStrategyScore
            trigger_event: トリガーイベントタイプ
            event_metadata: イベントメタデータ
            
        Returns:
            str: 作成されたエントリID
        """
        try:
            with self._lock:
                # 履歴エントリを作成
                entry = ScoreHistoryEntry(
                    strategy_score=strategy_score,
                    trigger_event=trigger_event,
                    event_metadata=event_metadata or {}
                )
                
                # キャッシュに追加
                self._history_cache[entry.entry_id] = entry
                self._recent_entries.append(entry.entry_id)
                
                # インデックスに追加
                if self.index:
                    self.index.add_entry(entry)
                
                # ディスクに保存
                self._save_entry_to_disk(entry)
                
                # イベント発火
                if self.event_manager:
                    self.event_manager.trigger_event('score_saved', {
                        'entry_id': entry.entry_id,
                        'strategy_name': strategy_score.strategy_name,
                        'ticker': strategy_score.ticker,
                        'score': strategy_score.total_score,
                        'trigger_event': trigger_event
                    })
                
                logger.debug(f"Score saved with entry ID: {entry.entry_id}")
                return entry.entry_id
                
        except Exception as e:
            logger.error(f"Error saving score: {e}")
            raise
    
    def get_score_history(self, strategy_name: str = None, ticker: str = None,
                         date_range: tuple = None, score_range: tuple = None,
                         limit: int = None) -> List[ScoreHistoryEntry]:
        """
        スコア履歴を取得
        
        Args:
            strategy_name: 戦略名でフィルタ
            ticker: ティッカーでフィルタ
            date_range: 日付範囲でフィルタ (start_date, end_date)
            score_range: スコア範囲でフィルタ (min_score, max_score)
            limit: 取得件数制限
            
        Returns:
            List[ScoreHistoryEntry]: 履歴エントリリスト
        """
        try:
            with self._lock:
                # インデックス検索を使用
                if self.index:
                    entry_ids = self.index.search(
                        strategy_name=strategy_name,
                        ticker=ticker,
                        date_range=date_range,
                        score_range=score_range
                    )
                else:
                    # フルスキャン
                    entry_ids = list(self._history_cache.keys())
                
                # エントリを取得
                entries = []
                for entry_id in entry_ids:
                    entry = self._get_entry_by_id(entry_id)
                    if entry and self._matches_criteria(entry, strategy_name, ticker, date_range, score_range):
                        entries.append(entry)
                
                # 日付順でソート（新しい順）
                entries.sort(key=lambda e: e.strategy_score.calculated_at, reverse=True)
                
                # 制限を適用
                if limit:
                    entries = entries[:limit]
                
                logger.debug(f"Retrieved {len(entries)} history entries")
                return entries
                
        except Exception as e:
            logger.error(f"Error retrieving score history: {e}")
            return []
    
    def get_score_statistics(self, strategy_name: str = None, ticker: str = None,
                           days: int = 30) -> Dict[str, Any]:
        """
        スコア統計情報を取得
        
        Args:
            strategy_name: 戦略名でフィルタ
            ticker: ティッカーでフィルタ
            days: 統計対象日数
            
        Returns:
            Dict[str, Any]: 統計情報
        """
        try:
            # 期間指定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 履歴を取得
            entries = self.get_score_history(
                strategy_name=strategy_name,
                ticker=ticker,
                date_range=(start_date, end_date)
            )
            
            if not entries:
                return {
                    'count': 0,
                    'message': 'No data available for the specified criteria'
                }
            
            # 統計計算
            scores = [entry.strategy_score.total_score for entry in entries]
            
            statistics = {
                'count': len(entries),
                'period_days': days,
                'score_stats': {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'std': self._calculate_std(scores)
                },
                'latest_score': entries[0].strategy_score.total_score if entries else None,
                'score_trend': self._calculate_trend(scores),
                'component_averages': self._calculate_component_averages(entries)
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating score statistics: {e}")
            return {'error': str(e)}
    
    def _save_entry_to_disk(self, entry: ScoreHistoryEntry):
        """エントリをディスクに保存"""
        try:
            # ファイル名を生成 (日付ベース)
            date_str = entry.strategy_score.calculated_at.strftime('%Y-%m-%d')
            filename = f"score_history_{date_str}.json"
            filepath = self.storage_dir / filename
            
            # 既存ファイルを読み込みまたは新規作成
            entries_data = []
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    entries_data = json.load(f)
            
            # 新しいエントリを追加
            entries_data.append(entry.to_dict())
            
            # ファイルに保存
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(entries_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Entry saved to disk: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving entry to disk: {e}")
            raise
    
    def _load_existing_history(self):
        """既存の履歴をロード"""
        try:
            if not self.storage_dir.exists():
                return
            
            loaded_count = 0
            for filepath in self.storage_dir.glob("score_history_*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        entries_data = json.load(f)
                    
                    for entry_data in entries_data:
                        entry = ScoreHistoryEntry.from_dict(entry_data)
                        
                        # キャッシュサイズ制限を考慮してロード
                        if len(self._history_cache) < self.config.cache_size:
                            self._history_cache[entry.entry_id] = entry
                        
                        # インデックスに追加
                        if self.index:
                            self.index.add_entry(entry)
                        
                        loaded_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error loading history file {filepath}: {e}")
            
            logger.info(f"Loaded {loaded_count} existing history entries")
            
        except Exception as e:
            logger.error(f"Error loading existing history: {e}")
    
    def _get_entry_by_id(self, entry_id: str) -> Optional[ScoreHistoryEntry]:
        """エントリIDでエントリを取得"""
        # まずキャッシュから
        if entry_id in self._history_cache:
            entry = self._history_cache[entry_id]
            entry.accessed_count += 1
            entry.last_accessed = datetime.now()
            return entry
        
        # キャッシュにない場合はディスクから検索（遅延ロード）
        if self.config.lazy_loading:
            return self._load_entry_from_disk(entry_id)
        
        return None
    
    def _load_entry_from_disk(self, entry_id: str) -> Optional[ScoreHistoryEntry]:
        """ディスクからエントリをロード"""
        try:
            for filepath in self.storage_dir.glob("score_history_*.json"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    entries_data = json.load(f)
                
                for entry_data in entries_data:
                    if entry_data.get('entry_id') == entry_id:
                        entry = ScoreHistoryEntry.from_dict(entry_data)
                        
                        # キャッシュに追加
                        self._history_cache[entry_id] = entry
                        
                        return entry
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading entry {entry_id} from disk: {e}")
            return None
    
    def _matches_criteria(self, entry: ScoreHistoryEntry, strategy_name: str = None,
                         ticker: str = None, date_range: tuple = None,
                         score_range: tuple = None) -> bool:
        """エントリが検索条件に一致するかチェック"""
        score = entry.strategy_score
        
        if strategy_name and score.strategy_name != strategy_name:
            return False
        
        if ticker and score.ticker != ticker:
            return False
        
        if date_range:
            start_date, end_date = date_range
            # datetime同士で比較するように修正
            if hasattr(start_date, 'date'):
                start_date = start_date.date()
            if hasattr(end_date, 'date'):
                end_date = end_date.date()
            score_date = score.calculated_at.date()
            
            if not (start_date <= score_date <= end_date):
                return False
        
        if score_range:
            min_score, max_score = score_range
            if not (min_score <= score.total_score <= max_score):
                return False
        
        return True
    
    def _calculate_std(self, values: List[float]) -> float:
        """標準偏差を計算"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """スコアトレンドを計算"""
        if len(scores) < 2:
            return "insufficient_data"
        
        # 最近の半分と前半分を比較
        mid = len(scores) // 2
        recent_avg = sum(scores[:mid]) / mid if mid > 0 else 0
        older_avg = sum(scores[mid:]) / (len(scores) - mid) if len(scores) - mid > 0 else 0
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "declining"
        else:
            return "stable"
    
    def _calculate_component_averages(self, entries: List[ScoreHistoryEntry]) -> Dict[str, float]:
        """コンポーネントスコアの平均を計算"""
        if not entries:
            return {}
        
        component_sums = defaultdict(float)
        component_counts = defaultdict(int)
        
        for entry in entries:
            for component, score in entry.strategy_score.component_scores.items():
                component_sums[component] += score
                component_counts[component] += 1
        
        return {
            component: component_sums[component] / component_counts[component]
            for component in component_sums
        }
    
    def _setup_cleanup_scheduler(self):
        """クリーンアップスケジューラーを設定"""
        # 簡単な実装（実際の運用では適切なスケジューラーを使用）
        if self.config.auto_cleanup_enabled:
            logger.info("Auto cleanup is enabled")
            # TODO: 実際のスケジューラー実装
    
    def cleanup_old_entries(self, days_to_keep: int = None):
        """古いエントリをクリーンアップ"""
        days_to_keep = days_to_keep or self.config.max_history_days
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleaned_count = 0
        with self._lock:
            # キャッシュからクリーンアップ
            entries_to_remove = []
            for entry_id, entry in self._history_cache.items():
                if entry.strategy_score.calculated_at < cutoff_date:
                    entries_to_remove.append(entry_id)
            
            for entry_id in entries_to_remove:
                entry = self._history_cache.pop(entry_id)
                if self.index:
                    self.index.remove_entry(entry)
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old entries from cache")
        return cleaned_count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュ情報を取得"""
        return {
            'cached_entries': len(self._history_cache),
            'recent_entries': len(self._recent_entries),
            'cache_limit': self.config.cache_size,
            'storage_directory': str(self.storage_dir),
            'index_enabled': self.config.index_enabled,
            'lazy_loading': self.config.lazy_loading
        }
