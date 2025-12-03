"""
DSSMS Data Management System
ハイブリッド方式でのマルチタイムフレームデータ管理

既存data_fetcher.pyを活用しつつ、DSSMS専用の最適化を実装
"""

import sys
from pathlib import Path
import pandas as pd
# import yfinance as yf  # Phase 3最適化: 遅延インポートに変更
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存システムインポート
from config.logger_config import setup_logger
from src.utils.lazy_import_manager import get_yfinance  # Phase 3最適化: 遅延インポート

class DSSMSDataManager:
    """
    DSSMS専用データ管理システム
    
    特徴:
    - 既存DataFetcherとの統合
    - マルチタイムフレーム効率取得
    - インメモリキャッシュ最適化
    - 並列データ取得
    """
    
    def __init__(self):
        self.logger = setup_logger('dssms.data_manager')
        
        # 既存データフェッチャー（削除 - 直接Yahoo Financeを使用）
        self.data_fetcher = None
        
        # マルチタイムフレームキャッシュ
        self._timeframe_cache = {
            'daily': {},
            'weekly': {},
            'monthly': {}
        }
        
        # キャッシュ有効期限
        self._cache_expiry = {
            'daily': timedelta(minutes=5),
            'weekly': timedelta(hours=1),
            'monthly': timedelta(hours=4)
        }
        
        # キャッシュのタイムスタンプ
        self._cache_timestamps = {
            'daily': {},
            'weekly': {},
            'monthly': {}
        }
        
        self.logger.info("DSSMS Data Manager initialized")
    
    def get_multi_timeframe_data(self, symbol: str, days_back: int = 300) -> Dict[str, pd.DataFrame]:
        """
        単一銘柄の複数時間軸データ取得
        
        Args:
            symbol: 銘柄コード
            days_back: 取得日数
            
        Returns:
            Dict[str, pd.DataFrame]: {"daily": df, "weekly": df, "monthly": df}
        """
        try:
            result = {}
            
            # 日足データ取得（ベース）
            daily_data = self._get_cached_data(symbol, 'daily', days_back)
            if daily_data is None:
                daily_data = self._fetch_daily_data(symbol, days_back)
                self._cache_data(symbol, 'daily', daily_data)
            
            result['daily'] = daily_data
            
            # 週足・月足データ生成
            result['weekly'] = self._resample_to_weekly(daily_data)
            result['monthly'] = self._resample_to_monthly(daily_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get multi-timeframe data for {symbol}: {e}")
            raise
    
    def batch_get_multi_timeframe_data(self, symbols: List[str], 
                                     max_workers: int = 5) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        複数銘柄の並列データ取得
        
        Args:
            symbols: 銘柄リスト
            max_workers: 並列実行数
            
        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: {symbol: {"daily": df, "weekly": df, "monthly": df}}
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 並列タスク送信
            future_to_symbol = {
                executor.submit(self.get_multi_timeframe_data, symbol): symbol 
                for symbol in symbols
            }
            
            # 結果収集
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result(timeout=30)
                    results[symbol] = data
                    self.logger.debug(f"Successfully fetched data for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue
        
        self.logger.info(f"Batch data fetch completed: {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_daily_data(self, symbol: str, days_back: int = 300) -> Optional[pd.DataFrame]:
        """
        単一銘柄の日足データ取得
        
        Args:
            symbol: 銘柄コード
            days_back: 取得日数
            
        Returns:
            pd.DataFrame: 日足データ（OHLCV）またはNone
        """
        try:
            # キャッシュから取得を試行
            daily_data = self._get_cached_data(symbol, 'daily', days_back)
            if daily_data is None:
                # キャッシュにない場合は新規取得
                daily_data = self._fetch_daily_data(symbol, days_back)
                self._cache_data(symbol, 'daily', daily_data)
            
            return daily_data
            
        except Exception as e:
            self.logger.error(f"Failed to get daily data for {symbol}: {e}")
            return None
    
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        単一銘柄の最新価格取得
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            Dict[str, float]: 最新価格情報 {"Close": float, "Open": float, ...} またはNone
        """
        try:
            # 最新の日足データを取得（1日分で十分）
            yf = get_yfinance()  # Phase 3最適化: 遅延インポート
            ticker = yf.Ticker(symbol)  # symbolはすでに.T含む
            
            # 直近2日分取得（市場休場を考慮、copilot-instructions.md: auto_adjust=False必須）
            hist = ticker.history(period="2d", interval="1d", auto_adjust=False)
            
            if hist.empty:
                self.logger.warning(f"No recent data available for {symbol}")
                return None
            
            # 最新の価格データを取得
            latest = hist.iloc[-1]
            
            price_data = {
                'Open': float(latest['Open']),
                'High': float(latest['High']),
                'Low': float(latest['Low']),
                'Close': float(latest['Close']),
                'Volume': float(latest['Volume'])
            }
            
            self.logger.debug(f"Latest price for {symbol}: ¥{price_data['Close']:.2f}")
            return price_data
            
        except Exception as e:
            self.logger.error(f"Failed to get latest price for {symbol}: {e}")
            return None
    
    def _get_cached_data(self, symbol: str, timeframe: str, days_back: int) -> Optional[pd.DataFrame]:
        """キャッシュからデータ取得"""
        try:
            cache_key = f"{symbol}_{days_back}"
            
            if cache_key in self._timeframe_cache[timeframe]:
                # タイムスタンプチェック
                timestamp = self._cache_timestamps[timeframe].get(cache_key)
                if timestamp and datetime.now() - timestamp < self._cache_expiry[timeframe]:
                    self.logger.debug(f"Cache hit for {symbol} {timeframe}")
                    return self._timeframe_cache[timeframe][cache_key]
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Cache retrieval failed for {symbol}: {e}")
            return None
    
    def _cache_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> None:
        """データをキャッシュに保存"""
        try:
            cache_key = f"{symbol}_300"  # 固定期間でキャッシュ
            self._timeframe_cache[timeframe][cache_key] = data.copy()
            self._cache_timestamps[timeframe][cache_key] = datetime.now()
            
            # キャッシュサイズ制限（最大100エントリ）
            if len(self._timeframe_cache[timeframe]) > 100:
                # 古いエントリを削除
                oldest_key = min(
                    self._cache_timestamps[timeframe].keys(),
                    key=lambda k: self._cache_timestamps[timeframe][k]
                )
                del self._timeframe_cache[timeframe][oldest_key]
                del self._cache_timestamps[timeframe][oldest_key]
            
        except Exception as e:
            self.logger.debug(f"Cache storage failed for {symbol}: {e}")
    
    def _fetch_daily_data(self, symbol: str, days_back: int) -> pd.DataFrame:
        """日足データ取得"""
        try:
            # Yahoo Finance から直接取得
            yf = get_yfinance()  # Phase 3最適化: 遅延インポート
            ticker = yf.Ticker(symbol)  # symbolはすでに.T含む
            
            # 期間設定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # データ取得（copilot-instructions.md: auto_adjust=False必須）
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d',
                auto_adjust=False
            )
            
            if hist.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # カラム名を統一
            hist.index.name = 'Date'
            hist = hist.rename(columns={
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # 必要な列のみ保持
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            hist = hist[required_columns]
            
            self.logger.debug(f"Fetched {len(hist)} days of data for {symbol}")
            return hist
            
        except Exception as e:
            self.logger.error(f"Failed to fetch daily data for {symbol}: {e}")
            raise
    
    def _resample_to_weekly(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """日足データを週足にリサンプル"""
        try:
            if daily_data.empty:
                return daily_data
            
            # 週足リサンプル（月曜開始）
            weekly = daily_data.resample('W-MON').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return weekly
            
        except Exception as e:
            self.logger.error(f"Failed to resample to weekly: {e}")
            return pd.DataFrame()
    
    def _resample_to_monthly(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """日足データを月足にリサンプル"""
        try:
            if daily_data.empty:
                return daily_data
            
            # 月足リサンプル
            monthly = daily_data.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return monthly
            
        except Exception as e:
            self.logger.error(f"Failed to resample to monthly: {e}")
            return pd.DataFrame()
    
    def clear_cache(self, timeframe: Optional[str] = None) -> None:
        """キャッシュクリア"""
        try:
            if timeframe:
                self._timeframe_cache[timeframe].clear()
                self._cache_timestamps[timeframe].clear()
                self.logger.info(f"Cleared {timeframe} cache")
            else:
                for tf in self._timeframe_cache:
                    self._timeframe_cache[tf].clear()
                    self._cache_timestamps[tf].clear()
                self.logger.info("Cleared all caches")
                
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """キャッシュ統計取得"""
        try:
            stats = {}
            
            for timeframe in self._timeframe_cache:
                stats[timeframe] = {
                    'total_entries': len(self._timeframe_cache[timeframe]),
                    'active_entries': len([
                        k for k, timestamp in self._cache_timestamps[timeframe].items()
                        if datetime.now() - timestamp < self._cache_expiry[timeframe]
                    ])
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {}


if __name__ == "__main__":
    # テスト実行
    data_manager = DSSMSDataManager()
    
    try:
        test_symbols = ["7203", "9984"]  # トヨタ、ソフトバンクG
        
        print("=== DSSMS Data Manager Test ===")
        
        # 単一銘柄テスト
        print("\n1. Single symbol test:")
        symbol = test_symbols[0]
        data = data_manager.get_multi_timeframe_data(symbol)
        
        for timeframe, df in data.items():
            print(f"  {symbol} {timeframe}: {len(df)} records")
            if not df.empty:
                print(f"    Latest: {df.index[-1].strftime('%Y-%m-%d')} Close: {df['Close'].iloc[-1]:.2f}")
        
        # バッチ取得テスト
        print("\n2. Batch fetch test:")
        batch_data = data_manager.batch_get_multi_timeframe_data(test_symbols, max_workers=2)
        
        for symbol, timeframe_data in batch_data.items():
            print(f"  {symbol}:")
            for timeframe, df in timeframe_data.items():
                print(f"    {timeframe}: {len(df)} records")
        
        # キャッシュ統計
        print("\n3. Cache statistics:")
        cache_stats = data_manager.get_cache_stats()
        for timeframe, stats in cache_stats.items():
            print(f"  {timeframe}: {stats['active_entries']}/{stats['total_entries']} active")
            
    except Exception as e:
        print(f"Test failed: {e}")


# 拡張メソッドを追加
def _extend_dssms_data_manager():
    """DSSMSDataManagerに日経225対応メソッドを追加"""
    
    def get_nikkei225_data(self, period: str = "1y") -> pd.DataFrame:
        """
        日経225指数データ取得（市場時間考慮）
        
        Args:
            period: データ期間 ("1y", "6mo", "3mo", "1mo")
            
        Returns:
            日経225指数データ
        """
        try:
            # キャッシュチェック
            cache_key = f"^N225_{period}"
            timeframe = "daily"
            
            if (cache_key in self._timeframe_cache[timeframe] and 
                cache_key in self._cache_timestamps[timeframe] and 
                datetime.now() - self._cache_timestamps[timeframe][cache_key] < self._cache_expiry[timeframe]):
                return self._timeframe_cache[timeframe][cache_key]
            
            # Yahoo Financeから取得
            yf = get_yfinance()  # Phase 3最適化: 遅延インポート
            ticker = yf.Ticker("^N225")
            data = ticker.history(period=period)
            
            if data.empty:
                self.logger.warning("Failed to fetch Nikkei 225 data")
                return pd.DataFrame()
            
            # データクリーニング
            data = data.dropna()
            
            # キャッシュ保存
            self._timeframe_cache[timeframe][cache_key] = data
            self._cache_timestamps[timeframe][cache_key] = datetime.now()
            
            self.logger.info(f"Nikkei 225 data fetched: {len(data)} records for {period}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching Nikkei 225 data: {e}")
            return pd.DataFrame()
    
    def get_realtime_nikkei225_data(self) -> Dict[str, Any]:
        """
        リアルタイム日経225データ取得（市場時間考慮）
        
        Returns:
            リアルタイムデータの辞書
        """
        try:
            # 基本データ取得
            data = self.get_nikkei225_data(period="5d")  # 直近5日分
            
            if data.empty:
                return {"error": "No data available"}
            
            latest = data.iloc[-1]
            previous = data.iloc[-2] if len(data) > 1 else latest
            
            # 変化率計算
            price_change = latest['Close'] - previous['Close']
            price_change_pct = (price_change / previous['Close']) * 100
            
            # 市場時間判定
            now = datetime.now()
            is_market_hours = self._is_market_hours(now)
            
            result = {
                "symbol": "^N225",
                "current_price": float(latest['Close']),
                "previous_close": float(previous['Close']),
                "price_change": float(price_change),
                "price_change_pct": float(price_change_pct),
                "volume": int(latest['Volume']),
                "high": float(latest['High']),
                "low": float(latest['Low']),
                "is_market_hours": is_market_hours,
                "last_update": latest.name.strftime('%Y-%m-%d %H:%M:%S'),
                "data_points": len(data)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting realtime Nikkei 225 data: {e}")
            return {"error": str(e)}
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """
        市場時間判定
        
        Args:
            timestamp: 判定する時刻
            
        Returns:
            市場時間内かどうか
        """
        try:
            weekday = timestamp.weekday()
            if weekday >= 5:  # 土日
                return False
            
            time_str = timestamp.strftime('%H:%M')
            
            # 午前: 09:00-11:30, 午後: 12:30-15:00
            morning_start = "09:00"
            morning_end = "11:30"
            afternoon_start = "12:30" 
            afternoon_end = "15:00"
            
            return ((morning_start <= time_str <= morning_end) or 
                   (afternoon_start <= time_str <= afternoon_end))
            
        except Exception:
            return False
    
    def calculate_market_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        市場指標計算（ADX、RSI、ボラティリティ等）
        
        Args:
            data: 価格データ
            
        Returns:
            計算された指標の辞書
        """
        try:
            if len(data) < 20:
                return {"error": "Insufficient data"}
            
            close = data['Close'].astype(float)
            high = data['High'].astype(float)
            low = data['Low'].astype(float)
            volume = data['Volume'].astype(float)
            
            indicators = {}
            
            # RSI計算
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = float(rsi.iloc[-1])
            
            # ボラティリティ計算
            returns = close.pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
            indicators['volatility'] = float(volatility)
            
            # 出来高比率
            avg_volume = volume.rolling(window=20).mean()
            volume_ratio = volume.iloc[-1] / avg_volume.iloc[-1]
            indicators['volume_ratio'] = float(volume_ratio)
            
            # 価格変化率
            price_change_1d = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
            price_change_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100
            indicators['price_change_1d'] = float(price_change_1d)
            indicators['price_change_5d'] = float(price_change_5d)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating market indicators: {e}")
            return {"error": str(e)}
    
    # DSSMSDataManagerクラスにメソッドを追加
    DSSMSDataManager.get_nikkei225_data = get_nikkei225_data
    DSSMSDataManager.get_realtime_nikkei225_data = get_realtime_nikkei225_data
    DSSMSDataManager._is_market_hours = _is_market_hours
    DSSMSDataManager.calculate_market_indicators = calculate_market_indicators

# 拡張を実行
_extend_dssms_data_manager()
