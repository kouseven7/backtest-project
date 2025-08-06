"""
リアルタイムデータソースアダプター
フェーズ3B: 複数のデータソースに対応した統一インターフェース
"""

import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests
import time
from pathlib import Path

# プロジェクトルート追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.exception_handler import UnifiedExceptionHandler, DataError
from src.utils.error_recovery import ErrorRecoveryManager
from config.logger_config import setup_logger


class DataSourceAdapter(ABC):
    """データソースアダプターの基底クラス"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = setup_logger(f"DataAdapter_{name}")
        self.exception_handler = UnifiedExceptionHandler()
        self.recovery_manager = ErrorRecoveryManager()
        
        # 接続状態管理
        self.is_connected = False
        self.last_successful_request = None
        self.consecutive_failures = 0
        self.rate_limit_remaining = 100
        self.rate_limit_reset_time = None
        
    @abstractmethod
    def connect(self) -> bool:
        """データソースに接続"""
        pass
        
    @abstractmethod
    def disconnect(self) -> bool:
        """データソースから切断"""
        pass
        
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """現在価格取得"""
        pass
        
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, 
                          end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """履歴データ取得"""
        pass
        
    @abstractmethod
    def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """複数シンボルの市場データ取得"""
        pass
        
    def check_rate_limits(self) -> bool:
        """レート制限チェック"""
        if self.rate_limit_reset_time and datetime.now() < self.rate_limit_reset_time:
            if self.rate_limit_remaining <= 0:
                self.logger.warning(f"{self.name}: Rate limit exceeded")
                return False
        return True
        
    def update_rate_limit_info(self, remaining: int, reset_time: datetime):
        """レート制限情報更新"""
        self.rate_limit_remaining = remaining
        self.rate_limit_reset_time = reset_time
        
    def handle_error(self, error: Exception, operation: str) -> bool:
        """エラーハンドリング"""
        self.consecutive_failures += 1
        error_info = {
            'adapter': self.name,
            'operation': operation,
            'error': str(error),
            'consecutive_failures': self.consecutive_failures,
            'timestamp': datetime.now().isoformat()
        }
        
        # 統一例外処理システムを使用
        result = self.exception_handler.handle_data_error(
            error, 
            context={'data_source': self.name, 'operation': operation}
        )
        
        # 復旧試行
        if self.consecutive_failures >= 3:
            recovery_result = self.recovery_manager.attempt_recovery(
                'data_connection_failure', 
                {'adapter': self.name}
            )
            if recovery_result.get('success'):
                self.consecutive_failures = 0
                return True
                
        return result.get('should_continue', False)


class YahooFinanceAdapter(DataSourceAdapter):
    """Yahoo Finance データアダプター"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("YahooFinance", config)
        self.session = None
        
    def connect(self) -> bool:
        """Yahoo Finance に接続"""
        try:
            # セッション初期化
            self.session = requests.Session()
            
            # 接続テスト
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d")
            
            if not test_data.empty:
                self.is_connected = True
                self.consecutive_failures = 0
                self.logger.info("Yahoo Finance connected successfully")
                return True
            else:
                raise DataError("Failed to retrieve test data")
                
        except Exception as e:
            self.handle_error(e, "connect")
            self.is_connected = False
            return False
            
    def disconnect(self) -> bool:
        """Yahoo Finance から切断"""
        try:
            if self.session:
                self.session.close()
            self.is_connected = False
            self.logger.info("Yahoo Finance disconnected")
            return True
        except Exception as e:
            self.handle_error(e, "disconnect")
            return False
            
    def get_current_price(self, symbol: str) -> Optional[float]:
        """現在価格取得"""
        if not self.check_rate_limits():
            return None
            
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 複数の価格フィールドを試行
            price_fields = ['regularMarketPrice', 'currentPrice', 'previousClose']
            for field in price_fields:
                if field in info and info[field] is not None:
                    price = float(info[field])
                    self.last_successful_request = datetime.now()
                    self.consecutive_failures = 0
                    return price
                    
            # リアルタイムデータが取得できない場合は履歴データから
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                self.last_successful_request = datetime.now()
                self.consecutive_failures = 0
                return price
                
            raise DataError(f"No price data available for {symbol}")
            
        except Exception as e:
            self.handle_error(e, f"get_current_price_{symbol}")
            return None
            
    def get_historical_data(self, symbol: str, start_date: str, 
                          end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """履歴データ取得"""
        if not self.check_rate_limits():
            return None
            
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if not data.empty:
                self.last_successful_request = datetime.now()
                self.consecutive_failures = 0
                return data
            else:
                raise DataError(f"No historical data for {symbol}")
                
        except Exception as e:
            self.handle_error(e, f"get_historical_data_{symbol}")
            return None
            
    def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """複数シンボルの市場データ取得"""
        if not self.check_rate_limits():
            return {}
            
        try:
            results = {}
            for symbol in symbols:
                price = self.get_current_price(symbol)
                if price is not None:
                    results[symbol] = {
                        'price': price,
                        'timestamp': datetime.now().isoformat(),
                        'source': self.name
                    }
                    
            return results
            
        except Exception as e:
            self.handle_error(e, "get_market_data")
            return {}


class AlphaVantageAdapter(DataSourceAdapter):
    """Alpha Vantage データアダプター"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AlphaVantage", config)
        self.api_key = config.get('api_key', '')
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
        
    def connect(self) -> bool:
        """Alpha Vantage に接続"""
        try:
            if not self.api_key:
                raise DataError("Alpha Vantage API key not provided")
                
            # セッション初期化
            self.session = requests.Session()
            
            # 接続テスト
            test_params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'AAPL',
                'apikey': self.api_key
            }
            
            response = self.session.get(self.base_url, params=test_params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'Error Message' in data:
                raise DataError(f"API Error: {data['Error Message']}")
            if 'Note' in data:
                self.logger.warning(f"API Note: {data['Note']}")
                
            self.is_connected = True
            self.consecutive_failures = 0
            self.logger.info("Alpha Vantage connected successfully")
            return True
            
        except Exception as e:
            self.handle_error(e, "connect")
            self.is_connected = False
            return False
            
    def disconnect(self) -> bool:
        """Alpha Vantage から切断"""
        try:
            if self.session:
                self.session.close()
            self.is_connected = False
            self.logger.info("Alpha Vantage disconnected")
            return True
        except Exception as e:
            self.handle_error(e, "disconnect")
            return False
            
    def get_current_price(self, symbol: str) -> Optional[float]:
        """現在価格取得"""
        if not self.check_rate_limits():
            return None
            
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise DataError(f"API Error: {data['Error Message']}")
                
            if 'Global Quote' in data:
                quote = data['Global Quote']
                if '05. price' in quote:
                    price = float(quote['05. price'])
                    self.last_successful_request = datetime.now()
                    self.consecutive_failures = 0
                    return price
                    
            raise DataError(f"No price data available for {symbol}")
            
        except Exception as e:
            self.handle_error(e, f"get_current_price_{symbol}")
            return None
            
    def get_historical_data(self, symbol: str, start_date: str, 
                          end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """履歴データ取得"""
        if not self.check_rate_limits():
            return None
            
        try:
            # Alpha Vantage の間隔マッピング
            interval_map = {
                "1d": "TIME_SERIES_DAILY",
                "1h": "TIME_SERIES_INTRADAY"
            }
            
            function = interval_map.get(interval, "TIME_SERIES_DAILY")
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            if interval == "1h":
                params['interval'] = '60min'
                
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise DataError(f"API Error: {data['Error Message']}")
                
            # データ解析
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
                    
            if not time_series_key:
                raise DataError("Time series data not found")
                
            time_series = data[time_series_key]
            
            # DataFrameに変換
            df_data = []
            for date_str, values in time_series.items():
                date = pd.to_datetime(date_str)
                if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
                    df_data.append({
                        'Date': date,
                        'Open': float(values.get('1. open', 0)),
                        'High': float(values.get('2. high', 0)),
                        'Low': float(values.get('3. low', 0)),
                        'Close': float(values.get('4. close', 0)),
                        'Volume': int(values.get('5. volume', 0))
                    })
                    
            if df_data:
                df = pd.DataFrame(df_data)
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                self.last_successful_request = datetime.now()
                self.consecutive_failures = 0
                return df
            else:
                raise DataError(f"No data in specified date range for {symbol}")
                
        except Exception as e:
            self.handle_error(e, f"get_historical_data_{symbol}")
            return None
            
    def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """複数シンボルの市場データ取得"""
        if not self.check_rate_limits():
            return {}
            
        try:
            results = {}
            for symbol in symbols:
                # Alpha Vantage はレート制限が厳しいので間隔を空ける
                time.sleep(0.2)
                
                price = self.get_current_price(symbol)
                if price is not None:
                    results[symbol] = {
                        'price': price,
                        'timestamp': datetime.now().isoformat(),
                        'source': self.name
                    }
                    
            return results
            
        except Exception as e:
            self.handle_error(e, "get_market_data")
            return {}


class DataSourceManager:
    """データソース管理クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = setup_logger(__name__)
        self.adapters: Dict[str, DataSourceAdapter] = {}
        self.primary_adapter = None
        self.fallback_adapters: List[str] = []
        
        # 設定読み込み
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._get_default_config()
            
        self._initialize_adapters()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "data_sources": {
                "yahoo_finance": {
                    "enabled": True,
                    "priority": 1,
                    "config": {}
                },
                "alpha_vantage": {
                    "enabled": False,
                    "priority": 2,
                    "config": {
                        "api_key": ""
                    }
                }
            },
            "fallback_strategy": "priority_order",
            "connection_timeout": 10,
            "retry_attempts": 3
        }
        
    def _initialize_adapters(self):
        """アダプター初期化"""
        data_sources = self.config.get('data_sources', {})
        
        # 優先度順にソート
        sorted_sources = sorted(
            data_sources.items(),
            key=lambda x: x[1].get('priority', 999)
        )
        
        for source_name, source_config in sorted_sources:
            if not source_config.get('enabled', False):
                continue
                
            try:
                adapter = self._create_adapter(source_name, source_config['config'])
                if adapter:
                    self.adapters[source_name] = adapter
                    
                    # プライマリアダプター設定
                    if self.primary_adapter is None:
                        self.primary_adapter = source_name
                    else:
                        self.fallback_adapters.append(source_name)
                        
                    self.logger.info(f"Initialized adapter: {source_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize {source_name}: {e}")
                
    def _create_adapter(self, source_name: str, config: Dict[str, Any]) -> Optional[DataSourceAdapter]:
        """アダプター生成"""
        if source_name == "yahoo_finance":
            return YahooFinanceAdapter(config)
        elif source_name == "alpha_vantage":
            return AlphaVantageAdapter(config)
        else:
            self.logger.warning(f"Unknown data source: {source_name}")
            return None
            
    def connect_all(self) -> bool:
        """全アダプター接続"""
        success = True
        for name, adapter in self.adapters.items():
            try:
                if not adapter.connect():
                    self.logger.warning(f"Failed to connect {name}")
                    success = False
            except Exception as e:
                self.logger.error(f"Connection error for {name}: {e}")
                success = False
                
        return success
        
    def disconnect_all(self):
        """全アダプター切断"""
        for name, adapter in self.adapters.items():
            try:
                adapter.disconnect()
            except Exception as e:
                self.logger.error(f"Disconnection error for {name}: {e}")
                
    def get_current_price(self, symbol: str) -> Tuple[Optional[float], str]:
        """現在価格取得（フォールバック付き）"""
        # プライマリアダプターから試行
        if self.primary_adapter and self.primary_adapter in self.adapters:
            adapter = self.adapters[self.primary_adapter]
            price = adapter.get_current_price(symbol)
            if price is not None:
                return price, self.primary_adapter
                
        # フォールバックアダプターを試行
        for adapter_name in self.fallback_adapters:
            if adapter_name in self.adapters:
                adapter = self.adapters[adapter_name]
                price = adapter.get_current_price(symbol)
                if price is not None:
                    self.logger.info(f"Used fallback adapter {adapter_name} for {symbol}")
                    return price, adapter_name
                    
        return None, ""
        
    def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """市場データ取得（フォールバック付き）"""
        results = {}
        
        # プライマリアダプターから試行
        if self.primary_adapter and self.primary_adapter in self.adapters:
            adapter = self.adapters[self.primary_adapter]
            results = adapter.get_market_data(symbols)
            if results:
                return results
                
        # フォールバックアダプターを試行
        for adapter_name in self.fallback_adapters:
            if adapter_name in self.adapters:
                adapter = self.adapters[adapter_name]
                results = adapter.get_market_data(symbols)
                if results:
                    self.logger.info(f"Used fallback adapter {adapter_name}")
                    return results
                    
        return results
        
    def get_adapter_status(self) -> Dict[str, Any]:
        """アダプター状態取得"""
        status = {}
        for name, adapter in self.adapters.items():
            status[name] = {
                'connected': adapter.is_connected,
                'consecutive_failures': adapter.consecutive_failures,
                'last_successful_request': adapter.last_successful_request.isoformat() if adapter.last_successful_request else None,
                'rate_limit_remaining': adapter.rate_limit_remaining
            }
        return status


if __name__ == "__main__":
    # デモ実行
    import json
    
    # テスト用設定
    test_config = {
        "data_sources": {
            "yahoo_finance": {
                "enabled": True,
                "priority": 1,
                "config": {}
            }
        }
    }
    
    # マネージャー初期化
    manager = DataSourceManager()
    
    # 接続テスト
    print("Connecting to data sources...")
    if manager.connect_all():
        print("Connected successfully")
        
        # 価格取得テスト
        price, source = manager.get_current_price("AAPL")
        print(f"AAPL current price: {price} (from {source})")
        
        # 複数シンボルテスト
        market_data = manager.get_market_data(["AAPL", "GOOGL", "MSFT"])
        print(f"Market data: {json.dumps(market_data, indent=2)}")
        
        # 状態確認
        status = manager.get_adapter_status()
        print(f"Adapter status: {json.dumps(status, indent=2)}")
        
    else:
        print("Connection failed")
        
    # 切断
    manager.disconnect_all()
    print("Disconnected")
