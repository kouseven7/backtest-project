"""
リアルタイムフィードシステム
フェーズ3B: 適応的更新頻度によるリアルタイムデータ配信
"""

import sys
import asyncio
import threading
import time
import json
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

# プロジェクトルート追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.data_source_adapter import DataSourceManager
from src.data.realtime_cache import HybridRealtimeCache
from src.utils.exception_handler import UnifiedExceptionHandler, DataError
from src.utils.error_recovery import ErrorRecoveryManager
from config.logger_config import setup_logger


class MarketState(Enum):
    """市場状態"""
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    OPEN = "open"
    POST_MARKET = "post_market"
    HOLIDAY = "holiday"


class UpdateFrequency(Enum):
    """更新頻度"""
    REALTIME = 1      # 1秒
    HIGH = 5         # 5秒
    NORMAL = 15      # 15秒
    LOW = 60         # 60秒
    MINIMAL = 300    # 5分


@dataclass
class FeedSubscription:
    """フィード購読情報"""
    symbol: str
    subscriber_id: str
    callback: Callable[[str, Dict[str, Any]], None]
    frequency: UpdateFrequency
    last_update: Optional[datetime] = None
    active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MarketDataPoint:
    """市場データポイント"""
    symbol: str
    price: float
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    timestamp: datetime = None
    source: str = ""
    volatility: Optional[float] = None
    change_percent: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


class VolatilityAnalyzer:
    """ボラティリティ分析器"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.price_history: Dict[str, List[float]] = {}
        self.logger = setup_logger(f"{__name__}.VolatilityAnalyzer")
        
    def add_price(self, symbol: str, price: float):
        """価格データ追加"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append(price)
        
        # ウィンドウサイズ制限
        if len(self.price_history[symbol]) > self.window_size:
            self.price_history[symbol] = self.price_history[symbol][-self.window_size:]
            
    def calculate_volatility(self, symbol: str) -> Optional[float]:
        """ボラティリティ計算"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return None
            
        prices = np.array(self.price_history[symbol])
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(len(returns))
        
        return float(volatility)
        
    def get_adaptive_frequency(self, symbol: str) -> UpdateFrequency:
        """適応的更新頻度決定"""
        volatility = self.calculate_volatility(symbol)
        
        if volatility is None:
            return UpdateFrequency.NORMAL
            
        # ボラティリティベースの頻度決定
        if volatility > 0.05:      # 高ボラティリティ
            return UpdateFrequency.REALTIME
        elif volatility > 0.03:    # 中ボラティリティ
            return UpdateFrequency.HIGH
        elif volatility > 0.01:    # 低ボラティリティ
            return UpdateFrequency.NORMAL
        else:                      # 極低ボラティリティ
            return UpdateFrequency.LOW


class MarketScheduler:
    """市場スケジューラー"""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.MarketScheduler")
        # 米国市場時間 (UTC)
        self.market_hours = {
            'pre_market_start': (9, 0),    # 9:00 AM EST = 14:00 UTC
            'market_open': (14, 30),       # 9:30 AM EST = 14:30 UTC
            'market_close': (21, 0),       # 4:00 PM EST = 21:00 UTC
            'post_market_end': (24, 0)     # 8:00 PM EST = 1:00 UTC+1
        }
        
    def get_market_state(self, now: Optional[datetime] = None) -> MarketState:
        """現在の市場状態取得"""
        if now is None:
            now = datetime.utcnow()
            
        # 平日チェック
        if now.weekday() >= 5:  # 土曜日、日曜日
            return MarketState.CLOSED
            
        current_time = (now.hour, now.minute)
        
        # 市場時間チェック
        if current_time < self.market_hours['pre_market_start']:
            return MarketState.CLOSED
        elif current_time < self.market_hours['market_open']:
            return MarketState.PRE_MARKET
        elif current_time < self.market_hours['market_close']:
            return MarketState.OPEN
        elif current_time < self.market_hours['post_market_end']:
            return MarketState.POST_MARKET
        else:
            return MarketState.CLOSED
            
    def get_market_frequency(self, state: MarketState) -> UpdateFrequency:
        """市場状態に基づく更新頻度"""
        frequency_map = {
            MarketState.OPEN: UpdateFrequency.HIGH,
            MarketState.PRE_MARKET: UpdateFrequency.NORMAL,
            MarketState.POST_MARKET: UpdateFrequency.NORMAL,
            MarketState.CLOSED: UpdateFrequency.MINIMAL,
            MarketState.HOLIDAY: UpdateFrequency.MINIMAL
        }
        return frequency_map.get(state, UpdateFrequency.LOW)


class RealtimeFeedManager:
    """リアルタイムフィードマネージャー"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(__name__)
        self.exception_handler = UnifiedExceptionHandler()
        self.recovery_manager = ErrorRecoveryManager()
        
        # コンポーネント初期化
        self.data_manager = DataSourceManager(config.get('data_sources_config'))
        self.cache = HybridRealtimeCache(config.get('cache_config', {}))
        self.volatility_analyzer = VolatilityAnalyzer(
            window_size=config.get('volatility_window', 20)
        )
        self.market_scheduler = MarketScheduler()
        
        # 購読管理
        self.subscriptions: Dict[str, List[FeedSubscription]] = {}
        self.active_symbols: Set[str] = set()
        
        # スレッド管理
        self.update_threads: Dict[str, threading.Thread] = {}
        self.shutdown_event = threading.Event()
        self.thread_lock = threading.RLock()
        
        # パフォーマンス統計
        self.stats = {
            'updates_sent': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'active_subscriptions': 0,
            'last_update': None
        }
        
        # 初期化
        self._initialize()
        
    def _initialize(self):
        """初期化処理"""
        try:
            # データソース接続
            if not self.data_manager.connect_all():
                self.logger.warning("Some data sources failed to connect")
                
            self.logger.info("Realtime feed manager initialized")
            
        except Exception as e:
            self.exception_handler.handle_data_error(
                e, context={'operation': 'feed_initialization'}
            )
            
    def subscribe(self, symbol: str, subscriber_id: str, 
                 callback: Callable[[str, Dict[str, Any]], None],
                 frequency: Optional[UpdateFrequency] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """フィード購読"""
        try:
            # 頻度自動決定
            if frequency is None:
                market_state = self.market_scheduler.get_market_state()
                frequency = self.market_scheduler.get_market_frequency(market_state)
                
            subscription = FeedSubscription(
                symbol=symbol,
                subscriber_id=subscriber_id,
                callback=callback,
                frequency=frequency,
                metadata=metadata or {}
            )
            
            with self.thread_lock:
                if symbol not in self.subscriptions:
                    self.subscriptions[symbol] = []
                    
                # 既存購読チェック
                for i, sub in enumerate(self.subscriptions[symbol]):
                    if sub.subscriber_id == subscriber_id:
                        self.subscriptions[symbol][i] = subscription
                        self.logger.info(f"Updated subscription: {subscriber_id} -> {symbol}")
                        return True
                        
                # 新規購読追加
                self.subscriptions[symbol].append(subscription)
                self.active_symbols.add(symbol)
                self.stats['active_subscriptions'] += 1
                
                # 更新スレッド開始
                self._start_update_thread(symbol)
                
            self.logger.info(f"New subscription: {subscriber_id} -> {symbol} ({frequency.name})")
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            self.exception_handler.handle_data_error(
                e, context={'operation': 'subscribe', 'symbol': symbol}
            )
            return False
            
    def unsubscribe(self, symbol: str, subscriber_id: str) -> bool:
        """フィード購読解除"""
        try:
            with self.thread_lock:
                if symbol not in self.subscriptions:
                    return False
                    
                # 購読削除
                original_count = len(self.subscriptions[symbol])
                self.subscriptions[symbol] = [
                    sub for sub in self.subscriptions[symbol]
                    if sub.subscriber_id != subscriber_id
                ]
                
                removed = original_count - len(self.subscriptions[symbol])
                if removed > 0:
                    self.stats['active_subscriptions'] -= removed
                    
                # シンボル関連の購読がなくなった場合
                if not self.subscriptions[symbol]:
                    del self.subscriptions[symbol]
                    self.active_symbols.discard(symbol)
                    self._stop_update_thread(symbol)
                    
            self.logger.info(f"Unsubscribed: {subscriber_id} -> {symbol}")
            return removed > 0
            
        except Exception as e:
            self.stats['errors'] += 1
            self.exception_handler.handle_data_error(
                e, context={'operation': 'unsubscribe', 'symbol': symbol}
            )
            return False
            
    def _start_update_thread(self, symbol: str):
        """更新スレッド開始"""
        with self.thread_lock:
            if symbol in self.update_threads and self.update_threads[symbol].is_alive():
                return
                
            thread = threading.Thread(
                target=self._update_worker,
                args=(symbol,),
                daemon=True,
                name=f"Feed-{symbol}"
            )
            thread.start()
            self.update_threads[symbol] = thread
            
            self.logger.debug(f"Started update thread for {symbol}")
            
    def _stop_update_thread(self, symbol: str):
        """更新スレッド停止"""
        with self.thread_lock:
            if symbol in self.update_threads:
                # スレッドは shutdown_event で自動停止
                del self.update_threads[symbol]
                self.logger.debug(f"Stopped update thread for {symbol}")
                
    def _update_worker(self, symbol: str):
        """更新ワーカー"""
        self.logger.debug(f"Update worker started for {symbol}")
        
        while not self.shutdown_event.is_set():
            try:
                with self.thread_lock:
                    if symbol not in self.subscriptions:
                        break
                        
                    subscriptions = self.subscriptions[symbol].copy()
                    
                if not subscriptions:
                    break
                    
                # 最も高い頻度を取得
                min_interval = min(sub.frequency.value for sub in subscriptions if sub.active)
                
                # データ取得
                data_point = self._fetch_market_data(symbol)
                if data_point:
                    # ボラティリティ分析
                    self.volatility_analyzer.add_price(symbol, data_point.price)
                    data_point.volatility = self.volatility_analyzer.calculate_volatility(symbol)
                    
                    # 購読者に配信
                    self._distribute_data(symbol, data_point, subscriptions)
                    
                # 次の更新まで待機
                if self.shutdown_event.wait(min_interval):
                    break
                    
            except Exception as e:
                self.stats['errors'] += 1
                self.exception_handler.handle_data_error(
                    e, context={'operation': 'update_worker', 'symbol': symbol}
                )
                
                # エラー時は少し待機
                if self.shutdown_event.wait(5):
                    break
                    
        self.logger.debug(f"Update worker stopped for {symbol}")
        
    def _fetch_market_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """市場データ取得"""
        try:
            # キャッシュから確認
            cache_key = f"realtime_{symbol}_{int(time.time() // 30)}"  # 30秒キャッシュ
            cached_data = self.cache.get(cache_key)
            
            if cached_data:
                self.stats['cache_hits'] += 1
                return MarketDataPoint(**cached_data)
                
            # データソースから取得
            price, source = self.data_manager.get_current_price(symbol)
            if price is None:
                return None
                
            # データポイント作成
            data_point = MarketDataPoint(
                symbol=symbol,
                price=price,
                source=source,
                timestamp=datetime.now()
            )
            
            # 変化率計算
            prev_cache_key = f"realtime_{symbol}_{int((time.time() - 60) // 30)}"
            prev_data = self.cache.get(prev_cache_key)
            if prev_data and 'price' in prev_data:
                prev_price = prev_data['price']
                data_point.change_percent = ((price - prev_price) / prev_price) * 100
                
            # キャッシュに保存
            self.cache.put(cache_key, data_point.to_dict(), ttl=timedelta(seconds=60))
            self.stats['cache_misses'] += 1
            
            return data_point
            
        except Exception as e:
            self.exception_handler.handle_data_error(
                e, context={'operation': 'fetch_market_data', 'symbol': symbol}
            )
            return None
            
    def _distribute_data(self, symbol: str, data_point: MarketDataPoint, 
                        subscriptions: List[FeedSubscription]):
        """データ配信"""
        try:
            data_dict = data_point.to_dict()
            now = datetime.now()
            
            for subscription in subscriptions:
                if not subscription.active:
                    continue
                    
                # 更新頻度チェック
                if (subscription.last_update and 
                    now - subscription.last_update < timedelta(seconds=subscription.frequency.value)):
                    continue
                    
                try:
                    # コールバック実行
                    subscription.callback(symbol, data_dict)
                    subscription.last_update = now
                    self.stats['updates_sent'] += 1
                    
                except Exception as callback_error:
                    self.logger.error(f"Callback error for {subscription.subscriber_id}: {callback_error}")
                    # コールバックエラーは購読を無効化
                    subscription.active = False
                    
            self.stats['last_update'] = now.isoformat()
            
        except Exception as e:
            self.exception_handler.handle_data_error(
                e, context={'operation': 'distribute_data', 'symbol': symbol}
            )
            
    def get_subscription_info(self) -> Dict[str, Any]:
        """購読情報取得"""
        with self.thread_lock:
            info = {
                'active_symbols': list(self.active_symbols),
                'total_subscriptions': sum(len(subs) for subs in self.subscriptions.values()),
                'subscriptions_by_symbol': {}
            }
            
            for symbol, subs in self.subscriptions.items():
                info['subscriptions_by_symbol'][symbol] = [
                    {
                        'subscriber_id': sub.subscriber_id,
                        'frequency': sub.frequency.name,
                        'active': sub.active,
                        'last_update': sub.last_update.isoformat() if sub.last_update else None
                    }
                    for sub in subs
                ]
                
            return info
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        cache_stats = self.cache.get_stats()
        adapter_status = self.data_manager.get_adapter_status()
        
        return {
            'feed_stats': self.stats,
            'cache_stats': cache_stats,
            'adapter_status': adapter_status,
            'subscription_info': self.get_subscription_info()
        }
        
    def update_subscription_frequency(self, symbol: str, subscriber_id: str, 
                                    frequency: UpdateFrequency) -> bool:
        """購読頻度更新"""
        try:
            with self.thread_lock:
                if symbol not in self.subscriptions:
                    return False
                    
                for subscription in self.subscriptions[symbol]:
                    if subscription.subscriber_id == subscriber_id:
                        subscription.frequency = frequency
                        self.logger.info(f"Updated frequency for {subscriber_id}: {frequency.name}")
                        return True
                        
            return False
            
        except Exception as e:
            self.exception_handler.handle_data_error(
                e, context={'operation': 'update_frequency', 'symbol': symbol}
            )
            return False
            
    def optimize_frequencies(self):
        """頻度最適化"""
        try:
            market_state = self.market_scheduler.get_market_state()
            base_frequency = self.market_scheduler.get_market_frequency(market_state)
            
            with self.thread_lock:
                for symbol, subscriptions in self.subscriptions.items():
                    # ボラティリティベースの頻度
                    adaptive_freq = self.volatility_analyzer.get_adaptive_frequency(symbol)
                    
                    # より高い頻度を採用
                    optimal_freq = min(base_frequency, adaptive_freq, key=lambda x: x.value)
                    
                    for subscription in subscriptions:
                        if subscription.frequency != optimal_freq:
                            subscription.frequency = optimal_freq
                            self.logger.debug(f"Optimized frequency for {subscription.subscriber_id}: {optimal_freq.name}")
                            
        except Exception as e:
            self.exception_handler.handle_data_error(
                e, context={'operation': 'optimize_frequencies'}
            )
            
    def shutdown(self):
        """システムシャットダウン"""
        self.logger.info("Shutting down realtime feed manager...")
        
        # スレッド停止
        self.shutdown_event.set()
        
        # 全スレッド終了待機
        with self.thread_lock:
            for symbol, thread in self.update_threads.items():
                if thread.is_alive():
                    thread.join(timeout=5.0)
                    
        # コンポーネントクリーンアップ
        self.data_manager.disconnect_all()
        self.cache.shutdown()
        
        self.logger.info("Realtime feed manager shutdown complete")


if __name__ == "__main__":
    # デモ実行
    import signal
    
    # テスト用設定
    test_config = {
        'data_sources_config': None,  # デフォルト設定を使用
        'cache_config': {
            'memory_max_items': 100,
            'memory_max_mb': 64,
            'disk_cache_dir': 'cache/test_feed',
            'memory_ttl_seconds': 30,
            'disk_ttl_seconds': 300
        },
        'volatility_window': 10
    }
    
    # フィードマネージャー初期化
    feed_manager = RealtimeFeedManager(test_config)
    
    # テストコールバック
    def test_callback(symbol: str, data: Dict[str, Any]):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: ${data['price']:.2f} "
              f"({data.get('change_percent', 0):.2f}%) - {data['source']}")
    
    try:
        print("=== Realtime Feed Demo ===")
        
        # 購読開始
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            feed_manager.subscribe(
                symbol=symbol,
                subscriber_id=f"demo_{symbol.lower()}",
                callback=test_callback,
                frequency=UpdateFrequency.HIGH
            )
            
        print(f"Subscribed to: {symbols}")
        print("Receiving data... (Press Ctrl+C to stop)")
        
        # シグナルハンドラー
        def signal_handler(signum, frame):
            print("\nShutting down...")
            feed_manager.shutdown()
            exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        # 統計表示ループ
        while True:
            time.sleep(30)
            stats = feed_manager.get_performance_stats()
            print(f"\n=== Stats (Updates: {stats['feed_stats']['updates_sent']}, "
                  f"Errors: {stats['feed_stats']['errors']}) ===")
            
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    finally:
        feed_manager.shutdown()
        print("Demo completed")
