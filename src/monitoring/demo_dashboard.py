"""
フェーズ3B 監視ダッシュボード デモンストレーション

このスクリプトは、監視ダッシュボードのデモンストレーションを実行し、
リアルタイムでの動作を確認できます。
"""

import asyncio
import time
import threading
import random
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# プロジェクト内インポート
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.logger_config import setup_logger
from src.monitoring.dashboard import MonitoringDashboard, DashboardConfig, create_dashboard
from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.alert_manager import AlertManager, NotificationChannel, AlertLevel
from src.data.data_feed_integration import IntegratedDataFeedSystem, DataQualityMetrics, DataQualityLevel


class DashboardDemo:
    """ダッシュボードデモンストレーション"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.is_running = False
        self.demo_thread = None
        
        # デモ用コンポーネント
        self.data_feed_system = None
        self.dashboard = None
        self.metrics_collector = None
        self.alert_manager = None
        
        # デモデータ生成器
        self.data_generators = []
        
    def setup_demo_environment(self):
        """デモ環境セットアップ"""
        self.logger.info("Setting up demo environment...")
        
        try:
            # データフィードシステム初期化
            self.data_feed_system = IntegratedDataFeedSystem()
            
            # メトリクス収集器初期化
            self.metrics_collector = MetricsCollector()
            
            # アラート管理器初期化
            self.alert_manager = AlertManager()
            
            # デモ用通知チャネル設定
            self._setup_demo_notifications()
            
            # ダッシュボード設定
            dashboard_config = DashboardConfig(
                host="localhost",
                port=8080,
                auto_refresh_interval=3,
                enable_real_time_updates=True,
                chart_update_interval=2
            )
            
            # ダッシュボード作成
            self.dashboard = create_dashboard(self.data_feed_system, dashboard_config)
            
            # メトリクスコールバック設定
            self.metrics_collector.add_metric_callback(self._on_metric_collected)
            
            self.logger.info("Demo environment setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up demo environment: {e}")
            raise
            
    def _setup_demo_notifications(self):
        """デモ用通知設定"""
        # コンソール通知チャネル（実際のメール送信の代わり）
        console_channel = NotificationChannel(
            channel_id="console_demo",
            name="コンソール通知（デモ用）",
            type="console",  # カスタムタイプ
            config={},
            alert_levels=[AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
        )
        
        self.alert_manager.notification_manager.add_channel(console_channel)
        
        # カスタム通知ハンドラー追加
        original_send = self.alert_manager.notification_manager.send_alert_notification
        
        def demo_send_notification(alert):
            # コンソールに通知内容表示
            self.logger.warning(f"[ALERT] ALERT: {alert.title} ({alert.level.value.upper()})")
            self.logger.warning(f"   Message: {alert.message}")
            self.logger.warning(f"   Source: {alert.source}")
            self.logger.warning(f"   Time: {alert.timestamp.strftime('%H:%M:%S')}")
            print("-" * 50)
            return True
            
        self.alert_manager.notification_manager.send_alert_notification = demo_send_notification
        
    def start_demo(self):
        """デモ開始"""
        if self.is_running:
            self.logger.warning("Demo is already running")
            return
            
        self.logger.info("Starting dashboard demo...")
        print("\n" + "=" * 60)
        print("フェーズ3B リアルタイムデータ監視ダッシュボード デモ")
        print("=" * 60)
        print(f"ダッシュボードURL: http://localhost:8080")
        print("デモデータが自動生成され、リアルタイムで更新されます")
        print("Ctrl+C で停止")
        print("=" * 60 + "\n")
        
        try:
            self.is_running = True
            
            # データ生成開始
            self._start_data_generators()
            
            # アラート監視開始
            self.alert_manager.start_monitoring()
            
            # ダッシュボード開始（バックグラウンド）
            dashboard_thread = threading.Thread(
                target=self._start_dashboard_background,
                daemon=True
            )
            dashboard_thread.start()
            
            # メインデモループ
            self._run_demo_loop()
            
        except KeyboardInterrupt:
            print("\n\nデモを停止しています...")
            
        finally:
            self.stop_demo()
            
    def _start_dashboard_background(self):
        """バックグラウンドでダッシュボード開始"""
        try:
            self.dashboard.start()
        except Exception as e:
            self.logger.error(f"Dashboard start error: {e}")
            
    def _start_data_generators(self):
        """データ生成器開始"""
        # 品質メトリクス生成器
        quality_generator = threading.Thread(
            target=self._generate_quality_data,
            daemon=True
        )
        quality_generator.start()
        self.data_generators.append(quality_generator)
        
        # パフォーマンスメトリクス生成器
        performance_generator = threading.Thread(
            target=self._generate_performance_data,
            daemon=True
        )
        performance_generator.start()
        self.data_generators.append(performance_generator)
        
        # ネットワークメトリクス生成器
        network_generator = threading.Thread(
            target=self._generate_network_data,
            daemon=True
        )
        network_generator.start()
        self.data_generators.append(network_generator)
        
        # キャッシュメトリクス生成器
        cache_generator = threading.Thread(
            target=self._generate_cache_data,
            daemon=True
        )
        cache_generator.start()
        self.data_generators.append(cache_generator)
        
    def _generate_quality_data(self):
        """品質データ生成"""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        while self.is_running:
            try:
                for symbol in symbols:
                    # ランダムな品質スコア生成（時々低品質でアラート発生）
                    base_quality = 0.85
                    noise = random.gauss(0, 0.1)
                    
                    # 時々意図的に品質を下げる
                    if random.random() < 0.1:  # 10%の確率
                        base_quality = 0.6
                        
                    overall_score = max(0.1, min(1.0, base_quality + noise))
                    
                    # 各軸スコア
                    completeness = max(0.1, min(1.0, overall_score + random.gauss(0, 0.05)))
                    accuracy = max(0.1, min(1.0, overall_score + random.gauss(0, 0.05)))
                    timeliness = max(0.1, min(1.0, overall_score + random.gauss(0, 0.05)))
                    consistency = max(0.1, min(1.0, overall_score + random.gauss(0, 0.05)))
                    
                    # 品質レベル決定
                    if overall_score >= 0.9:
                        level = DataQualityLevel.EXCELLENT
                    elif overall_score >= 0.8:
                        level = DataQualityLevel.GOOD
                    elif overall_score >= 0.6:
                        level = DataQualityLevel.FAIR
                    elif overall_score >= 0.4:
                        level = DataQualityLevel.POOR
                    else:
                        level = DataQualityLevel.INVALID
                        
                    # 問題とレコメンデーション
                    issues = []
                    recommendations = []
                    
                    if overall_score < 0.7:
                        issues.append("データ品質が低下しています")
                        recommendations.append("データソースを確認してください")
                        
                    if accuracy < 0.8:
                        issues.append("データ精度に問題があります")
                        recommendations.append("異常値検出を強化してください")
                        
                    # 品質メトリクス作成
                    quality_metrics = DataQualityMetrics(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        completeness_score=completeness,
                        accuracy_score=accuracy,
                        timeliness_score=timeliness,
                        consistency_score=consistency,
                        overall_score=overall_score,
                        quality_level=level,
                        issues=issues,
                        recommendations=recommendations
                    )
                    
                    # メトリクス記録
                    self.metrics_collector.record_quality_metrics(quality_metrics)
                    
                    # データフィードシステムにも追加
                    if symbol not in self.data_feed_system.quality_history:
                        self.data_feed_system.quality_history[symbol] = []
                    self.data_feed_system.quality_history[symbol].append(quality_metrics)
                    
                time.sleep(5)  # 5秒間隔
                
            except Exception as e:
                self.logger.error(f"Error generating quality data: {e}")
                time.sleep(1)
                
    def _generate_performance_data(self):
        """パフォーマンスデータ生成"""
        operations = ["data_fetch", "cache_get", "validation", "processing", "storage"]
        
        while self.is_running:
            try:
                for operation in operations:
                    # 基本応答時間（操作によって異なる）
                    base_times = {
                        "data_fetch": 150,
                        "cache_get": 5,
                        "validation": 30,
                        "processing": 100,
                        "storage": 80
                    }
                    
                    base_time = base_times.get(operation, 50)
                    
                    # ランダムな変動追加
                    duration = max(1, base_time + random.gauss(0, base_time * 0.3))
                    
                    # 時々遅延を発生
                    if random.random() < 0.05:  # 5%の確率
                        duration *= 5  # 5倍の遅延
                        
                    # 成功/失敗決定
                    success = random.random() > 0.02  # 2%のエラー率
                    error_message = None if success else f"{operation} failed with timeout"
                    
                    # パフォーマンスメトリクス記録
                    self.metrics_collector.record_performance_metrics(
                        operation, duration, success, error_message
                    )
                    
                time.sleep(2)  # 2秒間隔
                
            except Exception as e:
                self.logger.error(f"Error generating performance data: {e}")
                time.sleep(1)
                
    def _generate_network_data(self):
        """ネットワークデータ生成"""
        sources = ["yahoo_finance", "alpha_vantage", "internal_cache", "backup_source"]
        
        while self.is_running:
            try:
                for source in sources:
                    # リクエスト数
                    request_count = random.randint(5, 50)
                    
                    # 応答時間
                    base_response_time = {
                        "yahoo_finance": 120,
                        "alpha_vantage": 200,
                        "internal_cache": 10,
                        "backup_source": 300
                    }.get(source, 100)
                    
                    response_time = max(5, base_response_time + random.gauss(0, 30))
                    
                    # 成功率
                    base_success_rate = 0.98
                    if random.random() < 0.1:  # 時々成功率低下
                        base_success_rate = 0.85
                        
                    success_rate = max(0.5, min(1.0, base_success_rate + random.gauss(0, 0.05)))
                    
                    # エラー・タイムアウト数
                    error_count = int(request_count * (1 - success_rate))
                    timeout_count = random.randint(0, max(1, error_count // 2))
                    
                    # ネットワークメトリクス記録
                    self.metrics_collector.record_network_metrics(
                        source, request_count, response_time, success_rate,
                        error_count, timeout_count
                    )
                    
                time.sleep(10)  # 10秒間隔
                
            except Exception as e:
                self.logger.error(f"Error generating network data: {e}")
                time.sleep(1)
                
    def _generate_cache_data(self):
        """キャッシュデータ生成"""
        cache_types = ["memory", "disk"]
        
        while self.is_running:
            try:
                for cache_type in cache_types:
                    # キャッシュ性能
                    base_hit_rate = 0.85 if cache_type == "memory" else 0.65
                    hit_rate = max(0.3, min(0.98, base_hit_rate + random.gauss(0, 0.1)))
                    
                    # リクエスト数
                    total_requests = random.randint(100, 1000)
                    hit_count = int(total_requests * hit_rate)
                    miss_count = total_requests - hit_count
                    
                    # サイズ
                    if cache_type == "memory":
                        size_bytes = random.randint(50*1024*1024, 200*1024*1024)  # 50-200MB
                        item_count = random.randint(500, 2000)
                    else:
                        size_bytes = random.randint(500*1024*1024, 2*1024*1024*1024)  # 500MB-2GB
                        item_count = random.randint(5000, 20000)
                        
                    eviction_count = random.randint(0, 10)
                    
                    # キャッシュメトリクス記録
                    self.metrics_collector.record_cache_metrics(
                        cache_type, hit_count, miss_count, size_bytes,
                        item_count, eviction_count
                    )
                    
                time.sleep(8)  # 8秒間隔
                
            except Exception as e:
                self.logger.error(f"Error generating cache data: {e}")
                time.sleep(1)
                
    def _run_demo_loop(self):
        """デモメインループ"""
        start_time = time.time()
        
        while self.is_running:
            try:
                # 定期的なメトリクス評価（アラート生成）
                metrics_summary = self.metrics_collector.get_all_metrics_summary(5)
                if metrics_summary:
                    self.alert_manager.evaluate_metrics(metrics_summary)
                    
                # 統計情報表示（30秒ごと）
                if int(time.time() - start_time) % 30 == 0:
                    self._print_demo_status()
                    
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in demo loop: {e}")
                time.sleep(1)
                
    def _print_demo_status(self):
        """デモ状態表示"""
        try:
            # メトリクス統計
            collection_stats = self.metrics_collector.collection_stats
            
            # アラート統計
            alert_stats = self.alert_manager.get_alert_stats()
            
            # 現在のメトリクス
            current_summary = self.metrics_collector.get_all_metrics_summary(5)
            
            print(f"\n[CHART] デモ状態 [{datetime.now().strftime('%H:%M:%S')}]")
            print(f"   収集メトリクス: {collection_stats['total_collected']}")
            print(f"   アクティブアラート: {alert_stats['active_count']}")
            print(f"   総アラート数: {alert_stats['total_generated']}")
            
            if current_summary.get('quality'):
                avg_quality = current_summary['quality'].get('overall', {}).get('mean', 0)
                print(f"   平均品質スコア: {avg_quality:.2f}")
                
            if current_summary.get('performance'):
                avg_response = current_summary['performance'].get('response_time', {}).get('mean', 0)
                print(f"   平均応答時間: {avg_response:.1f}ms")
                
        except Exception as e:
            self.logger.error(f"Error printing demo status: {e}")
            
    def _on_metric_collected(self, metric_type: str, metric_data):
        """メトリクス収集コールバック"""
        # デバッグ用の詳細ログ（必要に応じて）
        if metric_type == "quality" and hasattr(metric_data, 'overall_score'):
            if metric_data.overall_score < 0.7:
                self.logger.warning(
                    f"[SEARCH] 低品質検出: {metric_data.symbol} = {metric_data.overall_score:.2f}"
                )
                
    def stop_demo(self):
        """デモ停止"""
        if not self.is_running:
            return
            
        self.logger.info("Stopping demo...")
        self.is_running = False
        
        try:
            # アラート監視停止
            if self.alert_manager:
                self.alert_manager.stop_monitoring()
                
            # ダッシュボードエージェント停止
            if self.dashboard and self.dashboard.agent:
                self.dashboard.agent.stop()
                
            self.logger.info("Demo stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping demo: {e}")


def main():
    """メイン実行関数"""
    demo = DashboardDemo()
    
    try:
        # デモ環境セットアップ
        demo.setup_demo_environment()
        
        # デモ開始
        demo.start_demo()
        
    except KeyboardInterrupt:
        print("\nデモが中断されました")
        
    except Exception as e:
        print(f"デモ実行中にエラーが発生しました: {e}")
        
    finally:
        demo.stop_demo()


if __name__ == "__main__":
    main()
