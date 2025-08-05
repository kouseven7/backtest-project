"""
マーケットデータプロバイダー
Phase 2.A.2: 拡張トレンド切替テスター用データ取得モジュール
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import pickle
import json
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class CacheManager:
    """データキャッシュ管理"""
    
    def __init__(self, cache_dir: str = "cache/market_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_hours = 24
        
    def _get_cache_key(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """キャッシュキー生成"""
        key_string = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """キャッシュからデータ取得"""
        try:
            cache_key = self._get_cache_key(symbol, timeframe, start_date, end_date)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return None
                
            # ファイル更新時刻チェック
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.total_seconds() > self.max_age_hours * 3600:
                logger.info(f"Cache expired for {symbol} {timeframe}")
                cache_file.unlink()  # 古いキャッシュ削除
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                
            logger.info(f"Cache hit for {symbol} {timeframe} ({len(data)} rows)")
            return data
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def save_to_cache(self, data: pd.DataFrame, symbol: str, timeframe: str, start_date: str, end_date: str):
        """データをキャッシュに保存"""
        try:
            cache_key = self._get_cache_key(symbol, timeframe, start_date, end_date)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Data cached for {symbol} {timeframe} ({len(data)} rows)")
            
        except Exception as e:
            logger.warning(f"Cache save error: {e}")
    
    def clear_cache(self):
        """キャッシュクリア"""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

class MarketDataProvider:
    """マーケットデータプロバイダー"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.cache_manager = CacheManager()
        
        # 設定読み込み
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # デフォルト設定
            self.config = {
                "market_data": {
                    "symbols": ["SPY", "QQQ", "AAPL", "MSFT"],
                    "timeframes": ["1h", "4h", "1d"],
                    "cache_settings": {"enabled": True, "max_age_hours": 24}
                }
            }
        
        self.cache_enabled = self.config.get("market_data", {}).get("cache_settings", {}).get("enabled", True)
        self.cache_manager.max_age_hours = self.config.get("market_data", {}).get("cache_settings", {}).get("max_age_hours", 24)
        
        logger.info(f"MarketDataProvider initialized (cache: {'enabled' if self.cache_enabled else 'disabled'})")
    
    def get_data(self, 
                 symbol: str,
                 timeframe: str = "1h",
                 days: int = 30,
                 end_date: Optional[datetime] = None) -> pd.DataFrame:
        """マーケットデータ取得"""
        try:
            # 日付範囲設定
            if end_date is None:
                end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # キャッシュチェック
            if self.cache_enabled:
                cached_data = self.cache_manager.get_cached_data(symbol, timeframe, start_str, end_str)
                if cached_data is not None:
                    return cached_data
            
            # リアルタイムデータ取得
            logger.info(f"Fetching real-time data: {symbol} {timeframe} ({start_str} to {end_str})")
            data = self._fetch_real_time_data(symbol, timeframe, start_date, end_date)
            
            if data is None or data.empty:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # データ処理
            processed_data = self._process_data(data, timeframe)
            
            # キャッシュ保存
            if self.cache_enabled and not processed_data.empty:
                self.cache_manager.save_to_cache(processed_data, symbol, timeframe, start_str, end_str)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_real_time_data(self, 
                             symbol: str, 
                             timeframe: str, 
                             start_date: datetime, 
                             end_date: datetime) -> pd.DataFrame:
        """リアルタイムデータ取得"""
        try:
            # yfinanceでデータ取得
            ticker = yf.Ticker(symbol)
            
            # 期間設定
            period_map = {
                "1h": "1h",
                "4h": "1h",  # 1時間足から4時間足を作成
                "1d": "1d"
            }
            
            interval = period_map.get(timeframe, "1h")
            
            # データ取得
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                logger.warning(f"No data returned from yfinance for {symbol}")
                return pd.DataFrame()
            
            # カラム名を小文字に統一
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except Exception as e:
            logger.error(f"yfinance fetch error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _process_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """データ処理"""
        try:
            if data.empty:
                return data
            
            # 4時間足の場合はリサンプリング
            if timeframe == "4h":
                data = self._resample_to_4h(data)
            
            # 必要な列のチェック
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                # デフォルト値で補完
                for col in missing_columns:
                    if col == 'volume':
                        data[col] = 1000000  # デフォルトボリューム
                    else:
                        data[col] = data.get('close', 100.0)  # デフォルト価格
            
            # インデックスを確実にDatetimeIndexに
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # データクリーニング
            data = data.dropna()
            
            # 技術指標追加
            data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            return data
    
    def _resample_to_4h(self, data: pd.DataFrame) -> pd.DataFrame:
        """1時間足を4時間足にリサンプリング"""
        try:
            resampled = data.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled
            
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """技術指標追加"""
        try:
            # シンプルな技術指標を追加
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            data['rsi'] = self._calculate_rsi(data['close'])
            data['atr'] = self._calculate_atr(data)
            
            return data
            
        except Exception as e:
            logger.warning(f"Technical indicator calculation error: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI計算"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.warning(f"RSI calculation error: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """ATR計算"""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = tr.rolling(window=window).mean()
            
            return atr
            
        except Exception as e:
            logger.warning(f"ATR calculation error: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def get_multiple_symbols(self, 
                           symbols: List[str],
                           timeframe: str = "1h",
                           days: int = 30) -> Dict[str, pd.DataFrame]:
        """複数銘柄データ取得"""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_data(symbol, timeframe, days)
                if not data.empty:
                    results[symbol] = data
                    logger.info(f"Successfully fetched data for {symbol}")
                else:
                    logger.warning(f"No data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        logger.info(f"Fetched data for {len(results)} symbols out of {len(symbols)}")
        return results
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """データ品質検証"""
        quality_report = {
            'symbol': symbol,
            'total_rows': len(data),
            'date_range': {
                'start': str(data.index.min()) if not data.empty else None,
                'end': str(data.index.max()) if not data.empty else None
            },
            'missing_data': {},
            'data_gaps': 0,
            'price_anomalies': 0,
            'volume_anomalies': 0,
            'quality_score': 0.0
        }
        
        if data.empty:
            return quality_report
        
        try:
            # 欠損データチェック
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    missing_count = data[col].isna().sum()
                    quality_report['missing_data'][col] = missing_count
            
            # データギャップチェック
            time_diff = data.index.to_series().diff()
            expected_freq = pd.Timedelta(hours=1)  # 1時間足想定
            gaps = (time_diff > expected_freq * 2).sum()
            quality_report['data_gaps'] = gaps
            
            # 価格異常値チェック
            if 'close' in data.columns:
                price_changes = data['close'].pct_change().abs()
                anomalies = (price_changes > 0.2).sum()  # 20%以上の変動
                quality_report['price_anomalies'] = anomalies
            
            # ボリューム異常値チェック
            if 'volume' in data.columns:
                volume_changes = data['volume'].pct_change().abs()
                vol_anomalies = (volume_changes > 5.0).sum()  # 500%以上の変動
                quality_report['volume_anomalies'] = vol_anomalies
            
            # 品質スコア計算（0-1）
            total_issues = (
                sum(quality_report['missing_data'].values()) +
                quality_report['data_gaps'] +
                quality_report['price_anomalies'] +
                quality_report['volume_anomalies']
            )
            
            if len(data) > 0:
                quality_score = max(0.0, 1.0 - (total_issues / len(data)))
                quality_report['quality_score'] = quality_score
            
        except Exception as e:
            logger.warning(f"Data quality validation error: {e}")
        
        return quality_report
