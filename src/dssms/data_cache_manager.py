"""
DSSMS統合システム - DataCacheManager
複数銘柄データの効率的キャッシュ管理を行うクラス

Author: AI Assistant
Created: 2025-09-27
Phase: Phase 3 Tier 2 実装
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from threading import Lock
import hashlib
import pickle
import psutil

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.logger_config import setup_logger


class CacheError(Exception):
    """キャッシュ関連エラー"""
    pass


class DataError(Exception):
    """データ関連エラー"""
    pass


class DataCacheManager:
    """
    複数銘柄データの効率的キャッシュ管理
    
    Responsibilities:
    - データキャッシュ戦略（LRU, 容量制限）
    - メモリ管理・最適化
    - アクセス最適化（人気銘柄事前ロード）
    - 統計機能（ヒット率、使用量監視）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        データキャッシュ管理の初期化
        
        Args:
            config: 設定辞書
        
        Raises:
            ConfigError: 設定値エラー
        """
        try:
            # 基本設定
            self.config = config
            cache_config = config.get('data_management', {})
            
            # キャッシュ設定
            self.cache_size_mb = cache_config.get('cache_size_mb', 100)
            self.cache_ttl_days = cache_config.get('cache_ttl_days', 30)
            self.max_symbols = cache_config.get('max_symbols', 50)
            self.enable_preload = cache_config.get('enable_preload', True)
            self.enable_compression = cache_config.get('enable_compression', True)
            
            # キャッシュストレージ
            self.stock_data_cache: Dict[str, pd.DataFrame] = {}
            self.index_data_cache: Dict[str, pd.DataFrame] = {}
            self.cache_metadata: Dict[str, Dict[str, Any]] = {}
            
            # 統計データ
            self.cache_stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'data_fetches': 0,
                'total_requests': 0,
                'memory_usage_mb': 0.0,
                'preload_count': 0
            }
            
            # スレッドセーフティ
            self.cache_lock = Lock()
            
            # 人気銘柄追跡
            self.symbol_access_count: Dict[str, int] = {}
            self.popular_symbols: List[str] = []
            
            # ログ設定
            self.logger = setup_logger(f"{self.__class__.__name__}")
            
            # 設定検証
            self._validate_config()
            
            # 初期化完了
            self.logger.info(f"DataCacheManager初期化完了 - キャッシュサイズ: {self.cache_size_mb}MB, "
                           f"TTL: {self.cache_ttl_days}日, 最大銘柄数: {self.max_symbols}")
            
        except Exception as e:
            self.logger.error(f"DataCacheManager初期化エラー: {e}")
            raise CacheError(f"DataCacheManager初期化失敗: {e}")
    
    def _validate_config(self) -> None:
        """設定値の検証"""
        try:
            if self.cache_size_mb <= 0:
                raise ValueError(f"キャッシュサイズが無効: {self.cache_size_mb}MB")
            
            if self.cache_ttl_days <= 0:
                raise ValueError(f"TTLが無効: {self.cache_ttl_days}日")
            
            if self.max_symbols <= 0:
                raise ValueError(f"最大銘柄数が無効: {self.max_symbols}")
            
            self.logger.debug("設定値検証完了")
            
        except Exception as e:
            raise CacheError(f"設定値検証失敗: {e}")
    
    def get_cached_data(self, symbol: str, start_date: datetime, 
                       end_date: datetime) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        キャッシュからデータを取得、なければ外部取得
        
        Args:
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (株価データ, インデックスデータ) or (None, None)
        
        Raises:
            DataError: データ取得失敗
        
        Example:
            stock_data, index_data = cache_mgr.get_cached_data(
                symbol='7203',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31)
            )
            if stock_data is not None:
                print(f"データ取得成功: {len(stock_data)}日分")
        """
        try:
            with self.cache_lock:
                self.cache_stats['total_requests'] += 1
                
                # キャッシュキー生成
                cache_key = self._generate_cache_key(symbol, start_date, end_date)
                
                # キャッシュヒットチェック
                if self._is_cache_hit(cache_key):
                    self.cache_stats['hits'] += 1
                    self._update_cache_access(cache_key)
                    
                    stock_data = self.stock_data_cache.get(cache_key)
                    index_cache_key = self._generate_index_cache_key(start_date, end_date)
                    index_data = self.index_data_cache.get(index_cache_key)
                    
                    self.logger.debug(f"キャッシュヒット: {symbol} @ {cache_key}")
                    return stock_data, index_data
                
                # キャッシュミス - 外部データ取得
                self.cache_stats['misses'] += 1
                self.logger.debug(f"キャッシュミス: {symbol} @ {cache_key}")
                
            # ロック外でデータ取得（時間がかかるため）
            stock_data, index_data = self._fetch_external_data(symbol, start_date, end_date)
            
            if stock_data is not None and index_data is not None:
                # キャッシュに保存
                self.store_cached_data(symbol, start_date, end_date, stock_data, index_data)
                
                # 人気銘柄追跡
                self._track_symbol_popularity(symbol)
                
                return stock_data, index_data
            
            return None, None
            
        except Exception as e:
            self.logger.error(f"キャッシュデータ取得エラー: {e}")
            raise DataError(f"データ取得失敗 ({symbol}): {e}")
    
    def store_cached_data(self, symbol: str, start_date: datetime, 
                         end_date: datetime, stock_data: pd.DataFrame, 
                         index_data: pd.DataFrame) -> bool:
        """
        データをキャッシュに保存
        
        Args:
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            stock_data: 株価データ
            index_data: インデックスデータ
        
        Returns:
            bool: 保存成功フラグ
        
        Raises:
            CacheError: キャッシュ保存失敗
        
        Example:
            success = cache_mgr.store_cached_data(
                symbol='7203',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                stock_data=stock_df,
                index_data=index_df
            )
        """
        try:
            with self.cache_lock:
                # 容量チェック
                if not self._check_cache_capacity():
                    self._evict_old_data()
                
                # キャッシュキー生成
                cache_key = self._generate_cache_key(symbol, start_date, end_date)
                index_cache_key = self._generate_index_cache_key(start_date, end_date)
                
                # データ圧縮（オプション）
                if self.enable_compression:
                    stock_data = self._compress_dataframe(stock_data)
                    index_data = self._compress_dataframe(index_data)
                
                # キャッシュに保存
                self.stock_data_cache[cache_key] = stock_data
                self.index_data_cache[index_cache_key] = index_data
                
                # メタデータ更新
                self._update_cache_metadata(cache_key, stock_data)
                self._update_cache_metadata(index_cache_key, index_data)
                
                # メモリ使用量更新
                self._update_memory_usage()
                
                self.logger.debug(f"キャッシュ保存完了: {symbol} @ {cache_key}")
                return True
                
        except Exception as e:
            self.logger.error(f"キャッシュ保存エラー: {e}")
            raise CacheError(f"キャッシュ保存失敗: {e}")
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        キャッシュクリア
        
        Args:
            older_than_days: 指定日数より古いデータのみクリア（Noneで全クリア）
        
        Returns:
            int: クリアしたアイテム数
        
        Example:
            cleared_count = cache_mgr.clear_cache(older_than_days=7)
            print(f"{cleared_count}件のキャッシュをクリアしました")
        """
        try:
            with self.cache_lock:
                cleared_count = 0
                current_time = datetime.now()
                
                if older_than_days is None:
                    # 全クリア
                    cleared_count = len(self.stock_data_cache) + len(self.index_data_cache)
                    self.stock_data_cache.clear()
                    self.index_data_cache.clear()
                    self.cache_metadata.clear()
                    self.logger.info(f"キャッシュ全クリア: {cleared_count}件")
                else:
                    # 期限切れのみクリア
                    cutoff_time = current_time - timedelta(days=older_than_days)
                    
                    # 株価データキャッシュクリア
                    keys_to_remove = []
                    for key, metadata in self.cache_metadata.items():
                        if metadata.get('created_at', current_time) < cutoff_time:
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        if key in self.stock_data_cache:
                            del self.stock_data_cache[key]
                            cleared_count += 1
                        if key in self.index_data_cache:
                            del self.index_data_cache[key]
                            cleared_count += 1
                        if key in self.cache_metadata:
                            del self.cache_metadata[key]
                    
                    self.logger.info(f"期限切れキャッシュクリア: {cleared_count}件 ({older_than_days}日以上)")
                
                # メモリ使用量更新
                self._update_memory_usage()
                
                return cleared_count
                
        except Exception as e:
            self.logger.error(f"キャッシュクリアエラー: {e}")
            return 0
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        キャッシュ統計情報の取得
        
        Returns:
            Dict[str, Any]: キャッシュ統計
        
        Example:
            stats = cache_mgr.get_cache_statistics()
            print(f"ヒット率: {stats['performance']['hit_rate']:.1%}")
            print(f"メモリ使用量: {stats['memory']['usage_mb']:.1f}MB")
        """
        try:
            with self.cache_lock:
                total_requests = self.cache_stats['total_requests']
                hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0.0
                
                # メモリ使用量更新
                self._update_memory_usage()
                
                statistics = {
                    'performance': {
                        'hit_rate': hit_rate,
                        'hits': self.cache_stats['hits'],
                        'misses': self.cache_stats['misses'],
                        'total_requests': total_requests,
                        'evictions': self.cache_stats['evictions']
                    },
                    'memory': {
                        'usage_mb': self.cache_stats['memory_usage_mb'],
                        'limit_mb': self.cache_size_mb,
                        'utilization': self.cache_stats['memory_usage_mb'] / self.cache_size_mb if self.cache_size_mb > 0 else 0.0
                    },
                    'cache_size': {
                        'stock_data_entries': len(self.stock_data_cache),
                        'index_data_entries': len(self.index_data_cache),
                        'total_entries': len(self.stock_data_cache) + len(self.index_data_cache),
                        'max_symbols': self.max_symbols
                    },
                    'popular_symbols': {
                        'top_symbols': self.popular_symbols[:10],  # 上位10銘柄
                        'access_counts': {symbol: count for symbol, count in sorted(self.symbol_access_count.items(), key=lambda x: x[1], reverse=True)[:10]}
                    },
                    'configuration': {
                        'cache_size_mb': self.cache_size_mb,
                        'cache_ttl_days': self.cache_ttl_days,
                        'max_symbols': self.max_symbols,
                        'enable_preload': self.enable_preload,
                        'enable_compression': self.enable_compression
                    },
                    'last_updated': datetime.now()
                }
                
                return statistics
                
        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {
                'performance': {},
                'error': str(e),
                'last_updated': datetime.now()
            }
    
    def preload_popular_symbols(self, symbols: List[str], 
                               date_range_days: int = 365) -> Dict[str, bool]:
        """
        人気銘柄の事前ロード
        
        Args:
            symbols: 事前ロード対象銘柄リスト
            date_range_days: データ取得日数
        
        Returns:
            Dict[str, bool]: 銘柄別成功フラグ
        """
        try:
            if not self.enable_preload:
                self.logger.info("事前ロード無効化されています")
                return {}
            
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=date_range_days)
            
            results = {}
            
            for symbol in symbols:
                try:
                    # 既にキャッシュにある場合はスキップ
                    cache_key = self._generate_cache_key(symbol, start_date, end_date)
                    if self._is_cache_hit(cache_key):
                        results[symbol] = True
                        continue
                    
                    # データ取得・キャッシュ
                    stock_data, index_data = self.get_cached_data(symbol, start_date, end_date)
                    results[symbol] = stock_data is not None
                    
                    if results[symbol]:
                        self.cache_stats['preload_count'] += 1
                        
                except Exception as e:
                    self.logger.warning(f"事前ロード失敗 ({symbol}): {e}")
                    results[symbol] = False
            
            successful_preloads = sum(results.values())
            self.logger.info(f"事前ロード完了: {successful_preloads}/{len(symbols)}銘柄成功")
            
            return results
            
        except Exception as e:
            self.logger.error(f"事前ロードエラー: {e}")
            return {}
    
    def _generate_cache_key(self, symbol: str, start_date: datetime, end_date: datetime) -> str:
        """キャッシュキー生成"""
        try:
            key_string = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            return hashlib.md5(key_string.encode()).hexdigest()[:16]  # 短縮ハッシュ
        except Exception:
            return f"{symbol}_{start_date}_{end_date}"
    
    def _generate_index_cache_key(self, start_date: datetime, end_date: datetime) -> str:
        """インデックスデータ用キャッシュキー生成"""
        try:
            key_string = f"INDEX_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            return hashlib.md5(key_string.encode()).hexdigest()[:16]
        except Exception:
            return f"INDEX_{start_date}_{end_date}"
    
    def _is_cache_hit(self, cache_key: str) -> bool:
        """キャッシュヒット判定"""
        try:
            # データ存在チェック
            if cache_key not in self.stock_data_cache:
                return False
            
            # TTLチェック
            if cache_key in self.cache_metadata:
                metadata = self.cache_metadata[cache_key]
                created_at = metadata.get('created_at', datetime.now())
                age_days = (datetime.now() - created_at).days
                
                if age_days > self.cache_ttl_days:
                    # 期限切れ - 削除
                    self._remove_expired_data(cache_key)
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"キャッシュヒット判定エラー: {e}")
            return False
    
    def _remove_expired_data(self, cache_key: str) -> None:
        """期限切れデータの削除"""
        try:
            if cache_key in self.stock_data_cache:
                del self.stock_data_cache[cache_key]
            if cache_key in self.index_data_cache:
                del self.index_data_cache[cache_key]
            if cache_key in self.cache_metadata:
                del self.cache_metadata[cache_key]
            
            self.logger.debug(f"期限切れデータ削除: {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"期限切れデータ削除エラー: {e}")
    
    def _fetch_external_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """外部データソースからデータ取得"""
        try:
            self.cache_stats['data_fetches'] += 1
            
            # yfinanceでデータ取得
            ticker = yf.Ticker(f"{symbol}.T")  # 東証銘柄
            stock_data = ticker.history(start=start_date, end=end_date)
            
            if stock_data.empty:
                self.logger.warning(f"株価データが空です: {symbol}")
                return None, None
            
            # 日経平均データ取得
            nikkei_ticker = yf.Ticker("^N225")
            index_data = nikkei_ticker.history(start=start_date, end=end_date)
            
            if index_data.empty:
                self.logger.warning("日経平均データが空です")
                # 株価データのみでも処理継続
                index_data = pd.DataFrame()
            
            self.logger.debug(f"外部データ取得成功: {symbol} ({len(stock_data)}日分)")
            return stock_data, index_data
            
        except Exception as e:
            self.logger.error(f"外部データ取得エラー ({symbol}): {e}")
            return None, None
    
    def _check_cache_capacity(self) -> bool:
        """キャッシュ容量チェック"""
        try:
            # メモリ使用量チェック
            self._update_memory_usage()
            if self.cache_stats['memory_usage_mb'] >= self.cache_size_mb:
                return False
            
            # 銘柄数チェック
            total_entries = len(self.stock_data_cache) + len(self.index_data_cache)
            if total_entries >= self.max_symbols * 2:  # 株価+インデックス
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"容量チェックエラー: {e}")
            return False
    
    def _evict_old_data(self) -> None:
        """古いデータの削除（LRU）"""
        try:
            # アクセス時刻順でソート
            sorted_metadata = sorted(
                self.cache_metadata.items(),
                key=lambda x: x[1].get('last_access', datetime.min)
            )
            
            # 古いデータから削除（1/4程度を削除）
            items_to_remove = len(sorted_metadata) // 4
            
            for i in range(items_to_remove):
                cache_key, _ = sorted_metadata[i]
                
                if cache_key in self.stock_data_cache:
                    del self.stock_data_cache[cache_key]
                if cache_key in self.index_data_cache:
                    del self.index_data_cache[cache_key]
                if cache_key in self.cache_metadata:
                    del self.cache_metadata[cache_key]
                
                self.cache_stats['evictions'] += 1
            
            self.logger.info(f"古いデータ削除完了: {items_to_remove}件")
            
        except Exception as e:
            self.logger.error(f"データ削除エラー: {e}")
    
    def _update_cache_access(self, cache_key: str) -> None:
        """キャッシュアクセス時刻更新"""
        try:
            if cache_key in self.cache_metadata:
                self.cache_metadata[cache_key]['last_access'] = datetime.now()
                self.cache_metadata[cache_key]['access_count'] = self.cache_metadata[cache_key].get('access_count', 0) + 1
        except Exception as e:
            self.logger.warning(f"アクセス時刻更新エラー: {e}")
    
    def _update_cache_metadata(self, cache_key: str, data: pd.DataFrame) -> None:
        """キャッシュメタデータ更新"""
        try:
            data_size = data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            
            self.cache_metadata[cache_key] = {
                'created_at': datetime.now(),
                'last_access': datetime.now(),
                'access_count': 1,
                'data_size_mb': data_size,
                'record_count': len(data)
            }
        except Exception as e:
            self.logger.warning(f"メタデータ更新エラー: {e}")
    
    def _update_memory_usage(self) -> None:
        """メモリ使用量更新"""
        try:
            total_size_mb = 0.0
            
            for metadata in self.cache_metadata.values():
                total_size_mb += metadata.get('data_size_mb', 0.0)
            
            self.cache_stats['memory_usage_mb'] = total_size_mb
            
        except Exception as e:
            self.logger.warning(f"メモリ使用量更新エラー: {e}")
    
    def _track_symbol_popularity(self, symbol: str) -> None:
        """銘柄アクセス回数追跡"""
        try:
            self.symbol_access_count[symbol] = self.symbol_access_count.get(symbol, 0) + 1
            
            # 人気銘柄リスト更新（上位銘柄）
            sorted_symbols = sorted(
                self.symbol_access_count.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            self.popular_symbols = [symbol for symbol, _ in sorted_symbols[:20]]
            
        except Exception as e:
            self.logger.warning(f"人気銘柄追跡エラー: {e}")
    
    def _compress_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """データフレーム圧縮（オプション）"""
        try:
            if not self.enable_compression:
                return data
            
            # 数値カラムを効率的な型に変換
            for col in data.select_dtypes(include=[np.number]).columns:
                if data[col].dtype == np.float64:
                    data[col] = pd.to_numeric(data[col], downcast='float')
                elif data[col].dtype == np.int64:
                    data[col] = pd.to_numeric(data[col], downcast='integer')
            
            return data
            
        except Exception as e:
            self.logger.warning(f"データフレーム圧縮エラー: {e}")
            return data


def main():
    """DataCacheManager 動作テスト"""
    print("DataCacheManager 動作テスト")
    print("=" * 50)
    
    try:
        # 1. 初期化テスト
        config = {
            'data_management': {
                'cache_size_mb': 50,
                'cache_ttl_days': 7,
                'max_symbols': 20,
                'enable_preload': True,
                'enable_compression': True
            }
        }
        
        dcm = DataCacheManager(config)
        print("✅ DataCacheManager初期化成功")
        
        # 2. データ取得テスト（キャッシュミス）
        test_date_start = datetime(2023, 1, 1)
        test_date_end = datetime(2023, 6, 30)
        
        print(f"\n📊 データ取得テスト:")
        stock_data, index_data = dcm.get_cached_data('7203', test_date_start, test_date_end)
        
        if stock_data is not None:
            print(f"✅ データ取得成功: 7203 ({len(stock_data)}日分)")
        else:
            print("⚠️  データ取得失敗 (yfinanceエラーの可能性)")
        
        # 3. キャッシュヒットテスト
        print(f"\n🎯 キャッシュヒットテスト:")
        stock_data2, index_data2 = dcm.get_cached_data('7203', test_date_start, test_date_end)
        print("✅ 2回目データ取得（キャッシュヒット予定）")
        
        # 4. 統計情報テスト
        stats = dcm.get_cache_statistics()
        print(f"\n📈 統計情報:")
        print(f"  - ヒット率: {stats['performance']['hit_rate']:.1%}")
        print(f"  - 総リクエスト数: {stats['performance']['total_requests']}")
        print(f"  - メモリ使用量: {stats['memory']['usage_mb']:.2f}MB")
        print(f"  - キャッシュエントリ数: {stats['cache_size']['total_entries']}")
        
        # 5. 複数銘柄テスト
        print(f"\n🏢 複数銘柄テスト:")
        test_symbols = ['6758', '9984', '8306']
        
        for symbol in test_symbols:
            try:
                data, _ = dcm.get_cached_data(symbol, test_date_start, test_date_end)
                if data is not None:
                    print(f"✅ {symbol}: {len(data)}日分取得")
                else:
                    print(f"⚠️  {symbol}: データ取得失敗")
            except Exception as e:
                print(f"❌ {symbol}: エラー - {e}")
        
        # 6. 事前ロードテスト
        print(f"\n🚀 事前ロードテスト:")
        preload_symbols = ['4503', '8411', '7974']
        preload_results = dcm.preload_popular_symbols(preload_symbols, date_range_days=180)
        
        for symbol, success in preload_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {symbol}: {'成功' if success else '失敗'}")
        
        # 7. キャッシュクリアテスト
        print(f"\n🧹 キャッシュ管理テスト:")
        
        # 期限切れクリア（存在しない場合）
        cleared_old = dcm.clear_cache(older_than_days=1)
        print(f"✅ 期限切れクリア: {cleared_old}件")
        
        # 部分クリア
        cleared_partial = dcm.clear_cache(older_than_days=0)  # 全て期限切れ扱い
        print(f"✅ 部分クリア: {cleared_partial}件")
        
        # 最終統計
        final_stats = dcm.get_cache_statistics()
        print(f"\n📊 最終統計:")
        print(f"  - 最終ヒット率: {final_stats['performance']['hit_rate']:.1%}")
        print(f"  - データ取得回数: {final_stats['performance']['total_requests']}")
        print(f"  - 人気銘柄: {final_stats['popular_symbols']['top_symbols'][:5]}")
        
        print(f"\n🎉 DataCacheManager テスト完了！")
        print(f"実装機能: データキャッシュ、LRU削除、統計監視、事前ロード")
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()