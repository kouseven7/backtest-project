#!/usr/bin/env python3
"""
Stage 3-1: SmartCache統合実装ヘルパー

SmartCacheシステムをnikkei225_screener.pyに統合するための軽量統合ヘルパー
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from src.utils.symbol_utils import to_yfinance

class ScreenerSmartCache:
    """Screener特化SmartCacheシステム - 軽量統合版"""
    
    def __init__(self, cache_dir: str = "cache/screener_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # メモリキャッシュ
        self.memory_cache = {}
        self.cache_expiry = {}
        
        # 統計
        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0
        }
        
    def get_cache_key(self, symbol: str, data_type: str, date: str = None) -> str:
        """キャッシュキー生成"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        return f"{symbol}_{date}_{data_type}"
    
    def get_cached_data(self, symbol: str, data_type: str) -> Optional[Dict[str, Any]]:
        """キャッシュデータ取得"""
        cache_key = self.get_cache_key(symbol, data_type)
        
        # メモリキャッシュチェック
        if cache_key in self.memory_cache:
            if self._is_cache_valid(cache_key):
                self.stats["hits"] += 1
                return self.memory_cache[cache_key]
            else:
                # 期限切れキャッシュ削除
                del self.memory_cache[cache_key]
                if cache_key in self.cache_expiry:
                    del self.cache_expiry[cache_key]
        
        # ディスクキャッシュチェック
        cache_file = self._get_cache_file_path(symbol, data_type)
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # 有効期限チェック
                if self._is_disk_cache_valid(cached_data):
                    # メモリキャッシュに登録
                    self.memory_cache[cache_key] = cached_data
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=24)
                    self.stats["hits"] += 1
                    return cached_data
                else:
                    # 期限切れファイル削除
                    cache_file.unlink()
                    
            except Exception:
                pass
        
        self.stats["misses"] += 1
        return None
    
    def cache_data(self, symbol: str, data_type: str, data: Dict[str, Any]):
        """データキャッシュ保存"""
        cache_key = self.get_cache_key(symbol, data_type)
        
        # タイムスタンプ追加
        cache_data = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        # メモリキャッシュ
        self.memory_cache[cache_key] = cache_data
        self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=24)
        
        # ディスクキャッシュ
        try:
            cache_file = self._get_cache_file_path(symbol, data_type)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.stats["writes"] += 1
            
        except Exception:
            pass
    
    def _get_cache_file_path(self, symbol: str, data_type: str) -> Path:
        """キャッシュファイルパス生成"""
        today = datetime.now()
        year_month = today.strftime("%Y/%m")
        date_str = today.strftime("%Y%m%d")
        
        return self.cache_dir / year_month / f"{symbol}_{data_type}_{date_str}.json"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """メモリキャッシュ有効性チェック"""
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]
    
    def _is_disk_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """ディスクキャッシュ有効性チェック"""
        try:
            expires_at = datetime.fromisoformat(cached_data["expires_at"])
            return datetime.now() < expires_at
        except:
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hit_rate_percentage": round(hit_rate, 1),
            "total_hits": self.stats["hits"],
            "total_misses": self.stats["misses"],
            "total_writes": self.stats["writes"],
            "memory_cache_size": len(self.memory_cache)
        }


class CachedMarketDataFetcher:
    """キャッシュ統合市場データ取得システム"""
    
    def __init__(self):
        self.cache = ScreenerSmartCache()
        self.api_call_count = 0
        self.api_lock = threading.Lock()
        
    def get_market_cap_data_cached(self, symbol: str, min_cap: Optional[float] = None) -> Optional[float]:
        """キャッシュ統合時価総額データ取得（単一値返却版）"""
        
        # キャッシュから取得試行
        cached_data = self.cache.get_cached_data(symbol, "market_cap")
        if cached_data:
            market_cap = cached_data["data"].get("market_cap")
            if market_cap is not None:
                return market_cap
        
        # キャッシュミス：APIから取得
        market_cap_data = self._fetch_market_cap_from_api(symbol)
        if market_cap_data:
            # キャッシュに保存
            self.cache.cache_data(symbol, "market_cap", market_cap_data)
            market_cap = market_cap_data.get("market_cap")
            return market_cap if market_cap else 0
        
        return None
    
    def check_market_cap_threshold(self, symbol: str, min_cap: float) -> bool:
        """キャッシュ統合時価総額データ取得"""
        
        # キャッシュから取得試行
        cached_data = self.cache.get_cached_data(symbol, "market_cap")
        if cached_data:
            market_cap = cached_data["data"].get("market_cap")
            if market_cap is not None:
                return market_cap >= min_cap
        
        # キャッシュミス：APIから取得
        market_cap_data = self._fetch_market_cap_from_api(symbol)
        if market_cap_data:
            # キャッシュに保存
            self.cache.cache_data(symbol, "market_cap", market_cap_data)
            market_cap = market_cap_data.get("market_cap", 0)
            return market_cap and market_cap >= min_cap
        
        return False
    
    def get_price_data_cached(self, symbol: str) -> Optional[float]:
        """キャッシュ統合価格データ取得"""
        
        # キャッシュから取得試行
        cached_data = self.cache.get_cached_data(symbol, "price")
        if cached_data:
            return cached_data["data"].get("current_price")
        
        # キャッシュミス：APIから取得
        price_data = self._fetch_price_from_api(symbol)
        if price_data:
            # キャッシュに保存
            self.cache.cache_data(symbol, "price", price_data)
            return price_data.get("current_price")
        
        return None
    
    def get_volume_data_cached(self, symbol: str) -> Optional[int]:
        """キャッシュ統合出来高データ取得"""
        
        # キャッシュから取得試行
        cached_data = self.cache.get_cached_data(symbol, "volume")
        if cached_data:
            return cached_data["data"].get("volume")
        
        # キャッシュミス：APIから取得
        volume_data = self._fetch_volume_from_api(symbol)
        if volume_data:
            # キャッシュに保存
            self.cache.cache_data(symbol, "volume", volume_data)
            return volume_data.get("volume")
        
        return None
    
    def _fetch_market_cap_from_api(self, symbol: str) -> Optional[Dict[str, Any]]:
        """API経由時価総額データ取得"""
        try:
            with self.api_lock:
                time.sleep(0.15)  # レート制限
                self.api_call_count += 1
            
            from src.utils.lazy_import_manager import get_yfinance
            yf = get_yfinance()
            ticker = yf.Ticker(symbol + ".T")
            info = ticker.info
            
            market_cap = info.get('marketCap')
            if market_cap is None:
                # 代替計算
                shares_outstanding = info.get('sharesOutstanding')
                current_price = info.get('currentPrice')
                
                if shares_outstanding and current_price:
                    market_cap = shares_outstanding * current_price
            
            return {
                "market_cap": market_cap,
                "shares_outstanding": info.get('sharesOutstanding'),
                "current_price": info.get('currentPrice'),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception:
            return None
    
    def _fetch_price_from_api(self, symbol: str) -> Optional[Dict[str, Any]]:
        """API経由価格データ取得"""
        try:
            with self.api_lock:
                time.sleep(0.15)  # レート制限
                self.api_call_count += 1
            
            from src.utils.lazy_import_manager import get_yfinance
            yf = get_yfinance()
            ticker = yf.Ticker(symbol + ".T")
            info = ticker.info
            
            return {
                "current_price": info.get('currentPrice'),
                "previous_close": info.get('previousClose'),
                "day_low": info.get('dayLow'),
                "day_high": info.get('dayHigh'),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception:
            return None
    
    def _fetch_volume_from_api(self, symbol: str) -> Optional[Dict[str, Any]]:
        """API経由出来高データ取得"""
        try:
            with self.api_lock:
                time.sleep(0.15)  # レート制限
                self.api_call_count += 1
            
            from src.utils.lazy_import_manager import get_yfinance
            yf = get_yfinance()
            ticker = yf.Ticker(symbol + ".T")
            info = ticker.info
            
            return {
                "volume": info.get('volume'),
                "average_volume": info.get('averageVolume'),
                "average_volume_10days": info.get('averageVolume10days'),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception:
            return None


def create_screener_cache_integration():
    """Screenerキャッシュ統合ファクトリー"""
    return CachedMarketDataFetcher()