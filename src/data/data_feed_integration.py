"""
データフィード統合システム
フェーズ3B: エラーハンドリング統合とデータ品質管理
"""

import sys
import json
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
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
from src.data.realtime_feed import RealtimeFeedManager, MarketDataPoint, UpdateFrequency
from src.utils.exception_handler import UnifiedExceptionHandler, DataError
from src.utils.error_recovery import ErrorRecoveryManager
from src.utils.monitoring_agent import MonitoringAgent
from config.logger_config import setup_logger


class DataQualityLevel(Enum):
    """データ品質レベル"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class DataQualityMetrics:
    """データ品質メトリクス"""
    symbol: str
    timestamp: datetime
    completeness_score: float      # 完全性 (0-1)
    accuracy_score: float          # 精度 (0-1)
    timeliness_score: float        # 適時性 (0-1)
    consistency_score: float       # 一貫性 (0-1)
    overall_score: float           # 総合スコア (0-1)
    quality_level: DataQualityLevel
    issues: List[str]              # 品質問題リスト
    recommendations: List[str]     # 改善推奨事項
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        data['quality_level'] = self.quality_level.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class DataQualityValidator:
    """データ品質検証器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(f"{__name__}.DataQualityValidator")
        
        # 品質閾値
        self.thresholds = {
            'price_change_max': config.get('max_price_change_percent', 20.0),
            'volume_change_max': config.get('max_volume_change_percent', 500.0),
            'data_age_max_seconds': config.get('max_data_age_seconds', 300),
            'missing_fields_max': config.get('max_missing_fields', 2),
            'outlier_threshold': config.get('outlier_threshold', 3.0)  # 標準偏差
        }
        
        # 履歴データ（異常値検出用）
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[int]] = {}
        self.history_window = config.get('history_window', 50)
        
    def validate_data_point(self, data_point: MarketDataPoint, 
                          previous_data: Optional[MarketDataPoint] = None) -> DataQualityMetrics:
        """データポイント品質検証"""
        issues = []
        recommendations = []
        scores = {}
        
        # 完全性チェック
        scores['completeness'] = self._check_completeness(data_point, issues, recommendations)
        
        # 精度チェック
        scores['accuracy'] = self._check_accuracy(data_point, previous_data, issues, recommendations)
        
        # 適時性チェック
        scores['timeliness'] = self._check_timeliness(data_point, issues, recommendations)
        
        # 一貫性チェック
        scores['consistency'] = self._check_consistency(data_point, previous_data, issues, recommendations)
        
        # 総合スコア計算
        overall_score = np.mean(list(scores.values()))
        quality_level = self._determine_quality_level(overall_score)
        
        # 履歴更新
        self._update_history(data_point)
        
        return DataQualityMetrics(
            symbol=data_point.symbol,
            timestamp=datetime.now(),
            completeness_score=scores['completeness'],
            accuracy_score=scores['accuracy'],
            timeliness_score=scores['timeliness'],
            consistency_score=scores['consistency'],
            overall_score=overall_score,
            quality_level=quality_level,
            issues=issues,
            recommendations=recommendations
        )
        
    def _check_completeness(self, data_point: MarketDataPoint, 
                          issues: List[str], recommendations: List[str]) -> float:
        """完全性チェック"""
        required_fields = ['symbol', 'price', 'timestamp', 'source']
        optional_fields = ['volume', 'bid', 'ask', 'volatility', 'change_percent']
        
        missing_required = []
        missing_optional = []
        
        for field in required_fields:
            if getattr(data_point, field) is None:
                missing_required.append(field)
                
        for field in optional_fields:
            if getattr(data_point, field) is None:
                missing_optional.append(field)
                
        if missing_required:
            issues.append(f"Missing required fields: {missing_required}")
            recommendations.append("Ensure all required fields are populated")
            return 0.0
            
        # オプションフィールドの完全性
        optional_score = 1.0 - (len(missing_optional) / len(optional_fields))
        
        if len(missing_optional) > self.thresholds['missing_fields_max']:
            issues.append(f"Too many missing optional fields: {missing_optional}")
            recommendations.append("Improve data source to provide more complete data")
            
        return max(0.8, optional_score)  # 必須フィールドがあれば最低0.8
        
    def _check_accuracy(self, data_point: MarketDataPoint,
                       previous_data: Optional[MarketDataPoint],
                       issues: List[str], recommendations: List[str]) -> float:
        """精度チェック"""
        score = 1.0
        
        # 価格妥当性
        if data_point.price <= 0:
            issues.append("Invalid price: price must be positive")
            recommendations.append("Verify data source price calculation")
            score -= 0.5
            
        # ボリューム妥当性
        if data_point.volume is not None and data_point.volume < 0:
            issues.append("Invalid volume: volume cannot be negative")
            recommendations.append("Check volume data integrity")
            score -= 0.2
            
        # 価格変化率チェック
        if previous_data and previous_data.price > 0:
            price_change = abs((data_point.price - previous_data.price) / previous_data.price) * 100
            if price_change > self.thresholds['price_change_max']:
                issues.append(f"Extreme price change: {price_change:.2f}%")
                recommendations.append("Verify price accuracy with multiple sources")
                score -= 0.3
                
        # 異常値検出
        if self._is_price_outlier(data_point.symbol, data_point.price):
            issues.append("Price appears to be an outlier")
            recommendations.append("Compare with historical data and other sources")
            score -= 0.2
            
        return max(0.0, score)
        
    def _check_timeliness(self, data_point: MarketDataPoint,
                         issues: List[str], recommendations: List[str]) -> float:
        """適時性チェック"""
        now = datetime.now()
        data_age = (now - data_point.timestamp).total_seconds()
        max_age = self.thresholds['data_age_max_seconds']
        
        if data_age > max_age:
            issues.append(f"Data is stale: {data_age:.0f} seconds old")
            recommendations.append("Improve data refresh frequency")
            return max(0.0, 1.0 - (data_age - max_age) / max_age)
            
        # 未来のタイムスタンプチェック
        if data_point.timestamp > now + timedelta(seconds=60):
            issues.append("Data timestamp is in the future")
            recommendations.append("Check system clock synchronization")
            return 0.5
            
        return 1.0
        
    def _check_consistency(self, data_point: MarketDataPoint,
                          previous_data: Optional[MarketDataPoint],
                          issues: List[str], recommendations: List[str]) -> float:
        """一貫性チェック"""
        score = 1.0
        
        # シンボル一貫性
        if not data_point.symbol or len(data_point.symbol.strip()) == 0:
            issues.append("Empty or invalid symbol")
            recommendations.append("Ensure symbol is properly formatted")
            score -= 0.3
            
        # Bid/Ask一貫性
        if (data_point.bid is not None and data_point.ask is not None and 
            data_point.bid > data_point.ask):
            issues.append("Bid price is higher than ask price")
            recommendations.append("Verify bid/ask data calculation")
            score -= 0.2
            
        # 価格とbid/ask一貫性
        if data_point.bid is not None and data_point.price < data_point.bid:
            issues.append("Price is below bid price")
            recommendations.append("Check price feed consistency")
            score -= 0.1
            
        if data_point.ask is not None and data_point.price > data_point.ask:
            issues.append("Price is above ask price")
            recommendations.append("Check price feed consistency")
            score -= 0.1
            
        # 変化率一貫性
        if (previous_data and data_point.change_percent is not None and 
            previous_data.price > 0):
            expected_change = ((data_point.price - previous_data.price) / previous_data.price) * 100
            actual_change = data_point.change_percent
            
            if abs(expected_change - actual_change) > 0.1:  # 0.1%の許容誤差
                issues.append("Change percent is inconsistent with price data")
                recommendations.append("Recalculate change percentage")
                score -= 0.1
                
        return max(0.0, score)
        
    def _update_history(self, data_point: MarketDataPoint):
        """履歴データ更新"""
        symbol = data_point.symbol
        
        # 価格履歴
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(data_point.price)
        if len(self.price_history[symbol]) > self.history_window:
            self.price_history[symbol] = self.price_history[symbol][-self.history_window:]
            
        # ボリューム履歴
        if data_point.volume is not None:
            if symbol not in self.volume_history:
                self.volume_history[symbol] = []
            self.volume_history[symbol].append(data_point.volume)
            if len(self.volume_history[symbol]) > self.history_window:
                self.volume_history[symbol] = self.volume_history[symbol][-self.history_window:]
                
    def _is_price_outlier(self, symbol: str, price: float) -> bool:
        """価格異常値判定"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return False
            
        prices = np.array(self.price_history[symbol])
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if std_price == 0:
            return False
            
        z_score = abs((price - mean_price) / std_price)
        return z_score > self.thresholds['outlier_threshold']
        
    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """品質レベル決定"""
        if score >= 0.9:
            return DataQualityLevel.EXCELLENT
        elif score >= 0.8:
            return DataQualityLevel.GOOD
        elif score >= 0.6:
            return DataQualityLevel.FAIR
        elif score >= 0.4:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.INVALID


class IntegratedDataFeedSystem:
    """統合データフィードシステム"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = setup_logger(__name__)
        self.exception_handler = UnifiedExceptionHandler()
        self.recovery_manager = ErrorRecoveryManager()
        
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # コンポーネント初期化
        self.data_manager = DataSourceManager(self.config.get('data_sources'))
        self.cache = HybridRealtimeCache(self.config.get('cache', {}))
        self.feed_manager = RealtimeFeedManager(self.config.get('realtime_feed', {}))
        self.quality_validator = DataQualityValidator(self.config.get('data_quality', {}))
        
        # モニタリング
        self.monitoring = MonitoringAgent()
        
        # データ品質履歴
        self.quality_history: Dict[str, List[DataQualityMetrics]] = {}
        self.quality_alerts: List[Dict[str, Any]] = []
        
        # 統計情報
        self.stats = {
            'total_data_points': 0,
            'quality_checks': 0,
            'quality_failures': 0,
            'data_corrections': 0,
            'system_errors': 0,
            'uptime_start': datetime.now()
        }
        
        # イベントハンドラー
        self.data_handlers: List[Callable[[str, MarketDataPoint, DataQualityMetrics], None]] = []
        self.error_handlers: List[Callable[[Exception, Dict[str, Any]], None]] = []
        
        self._initialize()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定読み込み"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
                
        return self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "data_sources": {
                "yahoo_finance": {
                    "enabled": True,
                    "priority": 1,
                    "config": {}
                }
            },
            "cache": {
                "memory_max_items": 200,
                "memory_max_mb": 256,
                "disk_cache_dir": "cache/integrated_feed",
                "memory_ttl_seconds": 300,
                "disk_ttl_seconds": 3600
            },
            "realtime_feed": {
                "volatility_window": 20
            },
            "data_quality": {
                "max_price_change_percent": 15.0,
                "max_volume_change_percent": 300.0,
                "max_data_age_seconds": 180,
                "max_missing_fields": 1,
                "outlier_threshold": 2.5,
                "history_window": 30
            },
            "error_handling": {
                "auto_correction": True,
                "quality_threshold": 0.7,
                "alert_threshold": 0.5
            }
        }
        
    def _initialize(self):
        """システム初期化"""
        try:
            # データソース接続
            if not self.data_manager.connect_all():
                self.logger.warning("Some data sources failed to connect")
                
            # モニタリング開始
            self.monitoring.start_monitoring()
            
            self.logger.info("Integrated data feed system initialized")
            
        except Exception as e:
            self.exception_handler.handle_data_error(
                e, context={'operation': 'system_initialization'}
            )
            
    def register_data_handler(self, handler: Callable[[str, MarketDataPoint, DataQualityMetrics], None]):
        """データハンドラー登録"""
        self.data_handlers.append(handler)
        self.logger.debug(f"Registered data handler: {handler.__name__}")
        
    def register_error_handler(self, handler: Callable[[Exception, Dict[str, Any]], None]):
        """エラーハンドラー登録"""
        self.error_handlers.append(handler)
        self.logger.debug(f"Registered error handler: {handler.__name__}")
        
    def subscribe_to_symbol(self, symbol: str, subscriber_id: str,
                          frequency: Optional[UpdateFrequency] = None,
                          quality_threshold: Optional[float] = None) -> bool:
        """シンボル購読"""
        try:
            # 品質閾値設定
            if quality_threshold is None:
                quality_threshold = self.config['error_handling']['quality_threshold']
                
            # データ処理コールバック
            def data_callback(symbol: str, data_dict: Dict[str, Any]):
                self._process_data_update(symbol, data_dict, quality_threshold)
                
            # フィード購読
            success = self.feed_manager.subscribe(
                symbol=symbol,
                subscriber_id=subscriber_id,
                callback=data_callback,
                frequency=frequency
            )
            
            if success:
                self.logger.info(f"Subscribed to {symbol} for {subscriber_id}")
            else:
                self.logger.error(f"Failed to subscribe to {symbol} for {subscriber_id}")
                
            return success
            
        except Exception as e:
            self.stats['system_errors'] += 1
            self.exception_handler.handle_data_error(
                e, context={'operation': 'subscribe', 'symbol': symbol}
            )
            return False
            
    def unsubscribe_from_symbol(self, symbol: str, subscriber_id: str) -> bool:
        """シンボル購読解除"""
        try:
            success = self.feed_manager.unsubscribe(symbol, subscriber_id)
            if success:
                self.logger.info(f"Unsubscribed from {symbol} for {subscriber_id}")
            return success
            
        except Exception as e:
            self.exception_handler.handle_data_error(
                e, context={'operation': 'unsubscribe', 'symbol': symbol}
            )
            return False
            
    def _process_data_update(self, symbol: str, data_dict: Dict[str, Any], 
                           quality_threshold: float):
        """データ更新処理"""
        try:
            self.stats['total_data_points'] += 1
            
            # MarketDataPoint作成
            data_point = MarketDataPoint(**data_dict)
            
            # 前回データ取得
            previous_data = self._get_previous_data(symbol)
            
            # 品質検証
            quality_metrics = self.quality_validator.validate_data_point(
                data_point, previous_data
            )
            self.stats['quality_checks'] += 1
            
            # 品質履歴保存
            self._store_quality_metrics(symbol, quality_metrics)
            
            # 品質チェック
            if quality_metrics.overall_score < quality_threshold:
                self.stats['quality_failures'] += 1
                self._handle_quality_failure(symbol, data_point, quality_metrics)
                
                # 品質アラート
                if quality_metrics.overall_score < self.config['error_handling']['alert_threshold']:
                    self._generate_quality_alert(symbol, quality_metrics)
                    
            else:
                # データ保存
                self._store_data_point(symbol, data_point)
                
                # ハンドラー実行
                self._execute_data_handlers(symbol, data_point, quality_metrics)
                
        except Exception as e:
            self.stats['system_errors'] += 1
            self.exception_handler.handle_data_error(
                e, context={'operation': 'process_data', 'symbol': symbol}
            )
            self._execute_error_handlers(e, {'symbol': symbol, 'data': data_dict})
            
    def _get_previous_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """前回データ取得"""
        cache_key = f"latest_{symbol}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return MarketDataPoint(**cached_data)
        return None
        
    def _store_data_point(self, symbol: str, data_point: MarketDataPoint):
        """データポイント保存"""
        cache_key = f"latest_{symbol}"
        self.cache.put(cache_key, data_point.to_dict(), ttl=timedelta(hours=1))
        
    def _store_quality_metrics(self, symbol: str, metrics: DataQualityMetrics):
        """品質メトリクス保存"""
        if symbol not in self.quality_history:
            self.quality_history[symbol] = []
            
        self.quality_history[symbol].append(metrics)
        
        # 履歴サイズ制限
        max_history = 100
        if len(self.quality_history[symbol]) > max_history:
            self.quality_history[symbol] = self.quality_history[symbol][-max_history:]
            
    def _handle_quality_failure(self, symbol: str, data_point: MarketDataPoint,
                              quality_metrics: DataQualityMetrics):
        """品質失敗処理"""
        if not self.config['error_handling']['auto_correction']:
            return
            
        try:
            # 自動修正試行
            corrected_data = self._attempt_data_correction(symbol, data_point, quality_metrics)
            
            if corrected_data:
                self.stats['data_corrections'] += 1
                self.logger.info(f"Auto-corrected data for {symbol}")
                
                # 修正データでハンドラー実行
                self._execute_data_handlers(symbol, corrected_data, quality_metrics)
                
        except Exception as e:
            self.logger.error(f"Data correction failed for {symbol}: {e}")
            
    def _attempt_data_correction(self, symbol: str, data_point: MarketDataPoint,
                               quality_metrics: DataQualityMetrics) -> Optional[MarketDataPoint]:
        """データ修正試行"""
        # 基本的な修正ロジック
        corrected_data = MarketDataPoint(**data_point.to_dict())
        
        # 価格修正
        if "Invalid price" in str(quality_metrics.issues):
            # 前回データから推定
            previous_data = self._get_previous_data(symbol)
            if previous_data:
                corrected_data.price = previous_data.price
                
        # ボリューム修正
        if "Invalid volume" in str(quality_metrics.issues):
            corrected_data.volume = 0
            
        # タイムスタンプ修正
        if "future" in str(quality_metrics.issues).lower():
            corrected_data.timestamp = datetime.now()
            
        return corrected_data
        
    def _generate_quality_alert(self, symbol: str, quality_metrics: DataQualityMetrics):
        """品質アラート生成"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'quality_score': quality_metrics.overall_score,
            'quality_level': quality_metrics.quality_level.value,
            'issues': quality_metrics.issues,
            'recommendations': quality_metrics.recommendations
        }
        
        self.quality_alerts.append(alert)
        
        # アラート履歴制限
        if len(self.quality_alerts) > 1000:
            self.quality_alerts = self.quality_alerts[-500:]
            
        self.logger.warning(f"Quality alert for {symbol}: {quality_metrics.quality_level.value}")
        
    def _execute_data_handlers(self, symbol: str, data_point: MarketDataPoint,
                             quality_metrics: DataQualityMetrics):
        """データハンドラー実行"""
        for handler in self.data_handlers:
            try:
                handler(symbol, data_point, quality_metrics)
            except Exception as e:
                self.logger.error(f"Data handler error: {e}")
                
    def _execute_error_handlers(self, error: Exception, context: Dict[str, Any]):
        """エラーハンドラー実行"""
        for handler in self.error_handlers:
            try:
                handler(error, context)
            except Exception as e:
                self.logger.error(f"Error handler error: {e}")
                
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        uptime = datetime.now() - self.stats['uptime_start']
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'stats': self.stats,
            'feed_performance': self.feed_manager.get_performance_stats(),
            'data_quality_summary': self._get_quality_summary(),
            'active_alerts': len(self.quality_alerts),
            'adapter_status': self.data_manager.get_adapter_status()
        }
        
    def _get_quality_summary(self) -> Dict[str, Any]:
        """品質サマリー取得"""
        if not self.quality_history:
            return {}
            
        all_scores = []
        level_counts = {}
        
        for symbol, metrics_list in self.quality_history.items():
            for metrics in metrics_list[-10:]:  # 最新10件
                all_scores.append(metrics.overall_score)
                level = metrics.quality_level.value
                level_counts[level] = level_counts.get(level, 0) + 1
                
        return {
            'average_quality_score': np.mean(all_scores) if all_scores else 0,
            'quality_distribution': level_counts,
            'symbols_monitored': len(self.quality_history)
        }
        
    def get_quality_report(self, symbol: Optional[str] = None,
                          hours: int = 24) -> Dict[str, Any]:
        """品質レポート取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if symbol:
            symbols = [symbol] if symbol in self.quality_history else []
        else:
            symbols = list(self.quality_history.keys())
            
        report = {}
        
        for sym in symbols:
            recent_metrics = [
                m for m in self.quality_history[sym]
                if m.timestamp > cutoff_time
            ]
            
            if recent_metrics:
                scores = [m.overall_score for m in recent_metrics]
                report[sym] = {
                    'sample_count': len(recent_metrics),
                    'average_score': np.mean(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'latest_quality': recent_metrics[-1].to_dict()
                }
                
        return report
        
    def shutdown(self):
        """システムシャットダウン"""
        self.logger.info("Shutting down integrated data feed system...")
        
        try:
            self.feed_manager.shutdown()
            self.data_manager.disconnect_all()
            self.cache.shutdown()
            self.monitoring.stop_monitoring()
            
            self.logger.info("Integrated data feed system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


if __name__ == "__main__":
    # デモ実行
    import signal
    
    # テストハンドラー
    def test_data_handler(symbol: str, data_point: MarketDataPoint, quality: DataQualityMetrics):
        print(f"[{data_point.timestamp.strftime('%H:%M:%S')}] {symbol}: "
              f"${data_point.price:.2f} (Quality: {quality.quality_level.value})")
        
    def test_error_handler(error: Exception, context: Dict[str, Any]):
        print(f"Error: {error} in {context.get('symbol', 'unknown')}")
        
    # システム初期化
    system = IntegratedDataFeedSystem()
    
    # ハンドラー登録
    system.register_data_handler(test_data_handler)
    system.register_error_handler(test_error_handler)
    
    try:
        print("=== Integrated Data Feed Demo ===")
        
        # シンボル購読
        symbols = ["AAPL", "GOOGL"]
        for symbol in symbols:
            system.subscribe_to_symbol(
                symbol=symbol,
                subscriber_id=f"demo_{symbol.lower()}",
                frequency=UpdateFrequency.HIGH,
                quality_threshold=0.7
            )
            
        print(f"Subscribed to: {symbols}")
        print("Monitoring data quality... (Press Ctrl+C to stop)")
        
        # シグナルハンドラー
        def signal_handler(signum, frame):
            print("\nGenerating final report...")
            
            # システム状態
            status = system.get_system_status()
            print(f"Total data points: {status['stats']['total_data_points']}")
            print(f"Quality checks: {status['stats']['quality_checks']}")
            print(f"Quality failures: {status['stats']['quality_failures']}")
            
            # 品質レポート
            quality_report = system.get_quality_report(hours=1)
            print("\nQuality Report:")
            for symbol, metrics in quality_report.items():
                print(f"  {symbol}: avg={metrics['average_score']:.3f}")
                
            system.shutdown()
            exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        # メインループ
        while True:
            time.sleep(60)
            status = system.get_system_status()
            print(f"\nStatus: {status['stats']['total_data_points']} data points, "
                  f"{status['active_alerts']} alerts")
            
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    finally:
        system.shutdown()
        print("Demo completed")
