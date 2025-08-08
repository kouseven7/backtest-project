"""
フェーズ3B リアルタイムデータ監視ダッシュボード

このモジュールは、リアルタイムデータシステムの包括的な監視ダッシュボードを提供します。
Webベースのインターフェースで、データ品質、システム状態、パフォーマンス指標を
リアルタイムで表示・監視します。
"""

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import plotly.graph_objects as go
import plotly.utils
import numpy as np

# プロジェクト内インポート
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.logger_config import setup_logger
from src.data.data_feed_integration import IntegratedDataFeedSystem, DataQualityMetrics, DataQualityLevel
from src.error_handling.exception_handler import UnifiedExceptionHandler
from src.error_handling.error_recovery import ErrorRecoveryManager


@dataclass
class SystemMetrics:
    """システムメトリクス"""
    timestamp: datetime
    
    # データフィード統計
    total_data_points: int
    data_points_per_second: float
    active_symbols: int
    
    # データ品質統計
    avg_quality_score: float
    quality_distribution: Dict[str, int]  # level -> count
    quality_issues_count: int
    
    # システム性能
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    cache_size_mb: float
    
    # エラー統計
    error_count: int
    recovery_count: int
    alert_count: int
    
    # ネットワーク統計
    network_latency_ms: float
    data_source_status: Dict[str, str]  # source -> status
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class DashboardConfig:
    """ダッシュボード設定"""
    host: str = "localhost"
    port: int = 8080
    auto_refresh_interval: int = 5  # 秒
    max_history_points: int = 1000
    alert_retention_hours: int = 24
    enable_real_time_updates: bool = True
    log_level: str = "INFO"
    
    # グラフ設定
    chart_update_interval: int = 2  # 秒
    chart_max_points: int = 200
    chart_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.chart_colors is None:
            self.chart_colors = {
                'excellent': '#28a745',
                'good': '#17a2b8', 
                'fair': '#ffc107',
                'poor': '#fd7e14',
                'invalid': '#dc3545',
                'primary': '#007bff',
                'secondary': '#6c757d'
            }


class DashboardAgent:
    """ダッシュボード自動更新エージェント"""
    
    def __init__(self, dashboard: 'MonitoringDashboard'):
        self.dashboard = dashboard
        self.logger = setup_logger(f"{__name__}.DashboardAgent")
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        
    def start(self):
        """エージェント開始"""
        if self.is_running:
            self.logger.warning("Dashboard agent is already running")
            return
            
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        self.logger.info("Dashboard agent started")
        
    def stop(self):
        """エージェント停止"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.logger.info("Dashboard agent stopped")
        
    def _update_loop(self):
        """更新ループ"""
        while self.is_running:
            try:
                # システムメトリクス更新
                self.dashboard._update_system_metrics()
                
                # データ品質履歴更新
                self.dashboard._update_quality_history()
                
                # アラート処理
                self.dashboard._process_alerts()
                
                # WebSocketクライアントに更新通知
                if self.dashboard.config.enable_real_time_updates:
                    asyncio.run(self.dashboard._broadcast_updates())
                    
                time.sleep(self.dashboard.config.auto_refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Dashboard agent update error: {e}")
                time.sleep(1)  # エラー時は短い間隔で再試行


class MonitoringDashboard:
    """リアルタイムデータ監視ダッシュボード"""
    
    def __init__(self, data_feed_system: IntegratedDataFeedSystem, 
                 config: Optional[DashboardConfig] = None):
        self.data_feed_system = data_feed_system
        self.config = config or DashboardConfig()
        self.logger = setup_logger(__name__)
        
        # エラーハンドリング
        self.exception_handler = UnifiedExceptionHandler()
        self.recovery_manager = ErrorRecoveryManager()
        
        # FastAPIアプリ
        self.app = FastAPI(title="リアルタイムデータ監視ダッシュボード")
        
        # WebSocketクライアント管理
        self.websocket_clients: List[WebSocket] = []
        
        # データ履歴
        self.metrics_history: deque = deque(maxlen=self.config.max_history_points)
        self.quality_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.chart_max_points)
        )
        self.alert_history: deque = deque(maxlen=1000)
        
        # 現在のシステム状態
        self.current_metrics: Optional[SystemMetrics] = None
        self.system_status = "initializing"
        
        # ダッシュボードエージェント
        self.agent = DashboardAgent(self)
        
        # FastAPIルート設定
        self._setup_routes()
        
        # 静的ファイルとテンプレート設定
        self._setup_static_files()
        
        self.logger.info("Monitoring dashboard initialized")
        
    def _setup_routes(self):
        """APIルート設定"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """ダッシュボードホーム"""
            return self.templates.TemplateResponse(
                "dashboard.html", 
                {"request": request, "title": "リアルタイムデータ監視"}
            )
            
        @self.app.get("/api/metrics")
        async def get_current_metrics():
            """現在のメトリクス取得"""
            try:
                if self.current_metrics:
                    return self.current_metrics.to_dict()
                return {"error": "No metrics available"}
            except Exception as e:
                self.logger.error(f"Error getting metrics: {e}")
                return {"error": str(e)}
                
        @self.app.get("/api/metrics/history")
        async def get_metrics_history():
            """メトリクス履歴取得"""
            try:
                return [metrics.to_dict() for metrics in self.metrics_history]
            except Exception as e:
                self.logger.error(f"Error getting metrics history: {e}")
                return {"error": str(e)}
                
        @self.app.get("/api/quality/{symbol}")
        async def get_quality_history(symbol: str):
            """シンボル別品質履歴"""
            try:
                if symbol in self.quality_history:
                    return [
                        quality.to_dict() 
                        for quality in self.quality_history[symbol]
                    ]
                return []
            except Exception as e:
                self.logger.error(f"Error getting quality history for {symbol}: {e}")
                return {"error": str(e)}
                
        @self.app.get("/api/quality/chart/{symbol}")
        async def get_quality_chart(symbol: str):
            """品質チャート生成"""
            try:
                return await self._generate_quality_chart(symbol)
            except Exception as e:
                self.logger.error(f"Error generating quality chart for {symbol}: {e}")
                return {"error": str(e)}
                
        @self.app.get("/api/system/status")
        async def get_system_status():
            """システム状態取得"""
            try:
                return {
                    "status": self.system_status,
                    "uptime": self._get_uptime(),
                    "data_sources": self._get_data_source_status(),
                    "cache_status": self._get_cache_status(),
                    "error_summary": self._get_error_summary()
                }
            except Exception as e:
                self.logger.error(f"Error getting system status: {e}")
                return {"error": str(e)}
                
        @self.app.get("/api/alerts")
        async def get_alerts():
            """アラート履歴取得"""
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.config.alert_retention_hours)
                return [
                    alert for alert in self.alert_history 
                    if alert.get('timestamp', datetime.min) > cutoff_time
                ]
            except Exception as e:
                self.logger.error(f"Error getting alerts: {e}")
                return {"error": str(e)}
                
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket接続処理"""
            await self._handle_websocket(websocket)
            
    def _setup_static_files(self):
        """静的ファイル設定"""
        # テンプレートディレクトリ作成
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        
        # 静的ファイルディレクトリ作成
        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(exist_ok=True)
        
        # テンプレート設定
        self.templates = Jinja2Templates(directory=str(template_dir))
        
        # 静的ファイル設定
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # デフォルトHTMLテンプレート作成
        self._create_default_template()
        
    def _create_default_template(self):
        """デフォルトHTMLテンプレート作成"""
        template_path = Path(__file__).parent / "templates" / "dashboard.html"
        
        if not template_path.exists():
            html_content = '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        .metric-card { margin-bottom: 1rem; }
        .status-indicator { 
            width: 12px; height: 12px; 
            border-radius: 50%; 
            display: inline-block; 
            margin-right: 5px; 
        }
        .status-online { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        .chart-container { height: 400px; margin-bottom: 2rem; }
        .alert-item { 
            padding: 0.5rem; 
            margin-bottom: 0.5rem; 
            border-left: 4px solid #007bff; 
            background-color: #f8f9fa; 
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <nav class="navbar navbar-dark bg-dark">
            <div class="container-fluid">
                <span class="navbar-brand">リアルタイムデータ監視ダッシュボード</span>
                <span class="navbar-text">
                    <span id="system-status" class="status-indicator status-online"></span>
                    <span id="status-text">接続中</span>
                </span>
            </div>
        </nav>
        
        <div class="row mt-3">
            <!-- システム概要 -->
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-header">システム概要</div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6"><small>データポイント/秒</small></div>
                            <div class="col-6"><span id="data-rate">-</span></div>
                        </div>
                        <div class="row">
                            <div class="col-6"><small>アクティブシンボル</small></div>
                            <div class="col-6"><span id="active-symbols">-</span></div>
                        </div>
                        <div class="row">
                            <div class="col-6"><small>平均品質スコア</small></div>
                            <div class="col-6"><span id="avg-quality">-</span></div>
                        </div>
                        <div class="row">
                            <div class="col-6"><small>キャッシュヒット率</small></div>
                            <div class="col-6"><span id="cache-hit-rate">-</span></div>
                        </div>
                    </div>
                </div>
                
                <div class="card metric-card">
                    <div class="card-header">システム性能</div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6"><small>CPU使用率</small></div>
                            <div class="col-6"><span id="cpu-usage">-</span></div>
                        </div>
                        <div class="row">
                            <div class="col-6"><small>メモリ使用量</small></div>
                            <div class="col-6"><span id="memory-usage">-</span></div>
                        </div>
                        <div class="row">
                            <div class="col-6"><small>ネットワーク遅延</small></div>
                            <div class="col-6"><span id="network-latency">-</span></div>
                        </div>
                    </div>
                </div>
                
                <div class="card metric-card">
                    <div class="card-header">エラー統計</div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6"><small>エラー数</small></div>
                            <div class="col-6"><span id="error-count">-</span></div>
                        </div>
                        <div class="row">
                            <div class="col-6"><small>復旧数</small></div>
                            <div class="col-6"><span id="recovery-count">-</span></div>
                        </div>
                        <div class="row">
                            <div class="col-6"><small>アラート数</small></div>
                            <div class="col-6"><span id="alert-count">-</span></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- メインチャート -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">データ品質推移</div>
                    <div class="card-body">
                        <div id="quality-chart" class="chart-container"></div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">システムパフォーマンス</div>
                    <div class="card-body">
                        <div id="performance-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
            
            <!-- アラート・ログ -->
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">最新アラート</div>
                    <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                        <div id="alerts-list"></div>
                    </div>
                </div>
                
                <div class="card mt-3">
                    <div class="card-header">データソース状態</div>
                    <div class="card-body">
                        <div id="data-sources-list"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket接続
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = function(event) {
            console.log('WebSocket connected');
            updateStatus('online', '接続中');
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        ws.onclose = function(event) {
            console.log('WebSocket disconnected');
            updateStatus('error', '接続切断');
            // 再接続試行
            setTimeout(() => location.reload(), 5000);
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            updateStatus('error', 'エラー');
        };
        
        function updateStatus(status, text) {
            const indicator = document.getElementById('system-status');
            const statusText = document.getElementById('status-text');
            
            indicator.className = `status-indicator status-${status}`;
            statusText.textContent = text;
        }
        
        function updateDashboard(data) {
            if (data.type === 'metrics' && data.metrics) {
                updateMetrics(data.metrics);
            }
            if (data.type === 'alerts' && data.alerts) {
                updateAlerts(data.alerts);
            }
            if (data.type === 'quality_charts' && data.charts) {
                updateQualityCharts(data.charts);
            }
        }
        
        function updateMetrics(metrics) {
            document.getElementById('data-rate').textContent = 
                metrics.data_points_per_second.toFixed(1);
            document.getElementById('active-symbols').textContent = 
                metrics.active_symbols;
            document.getElementById('avg-quality').textContent = 
                metrics.avg_quality_score.toFixed(2);
            document.getElementById('cache-hit-rate').textContent = 
                (metrics.cache_hit_rate * 100).toFixed(1) + '%';
            document.getElementById('cpu-usage').textContent = 
                metrics.cpu_usage_percent.toFixed(1) + '%';
            document.getElementById('memory-usage').textContent = 
                metrics.memory_usage_mb.toFixed(0) + 'MB';
            document.getElementById('network-latency').textContent = 
                metrics.network_latency_ms.toFixed(0) + 'ms';
            document.getElementById('error-count').textContent = 
                metrics.error_count;
            document.getElementById('recovery-count').textContent = 
                metrics.recovery_count;
            document.getElementById('alert-count').textContent = 
                metrics.alert_count;
        }
        
        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alerts-list');
            alertsList.innerHTML = '';
            
            alerts.slice(0, 10).forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert-item';
                alertDiv.innerHTML = `
                    <small class="text-muted">${new Date(alert.timestamp).toLocaleTimeString()}</small><br>
                    <strong>${alert.level}</strong>: ${alert.message}
                `;
                alertsList.appendChild(alertDiv);
            });
        }
        
        function updateQualityCharts(charts) {
            // 品質チャート更新
            if (charts.quality_chart) {
                Plotly.react('quality-chart', charts.quality_chart.data, charts.quality_chart.layout);
            }
            
            // パフォーマンスチャート更新
            if (charts.performance_chart) {
                Plotly.react('performance-chart', charts.performance_chart.data, charts.performance_chart.layout);
            }
        }
        
        // 初期データ読み込み
        async function loadInitialData() {
            try {
                const metricsResponse = await fetch('/api/metrics');
                const metrics = await metricsResponse.json();
                if (metrics && !metrics.error) {
                    updateMetrics(metrics);
                }
                
                const alertsResponse = await fetch('/api/alerts');
                const alerts = await alertsResponse.json();
                if (alerts && !alerts.error) {
                    updateAlerts(alerts);
                }
            } catch (error) {
                console.error('Error loading initial data:', error);
            }
        }
        
        // ページ読み込み時の初期化
        document.addEventListener('DOMContentLoaded', function() {
            loadInitialData();
            
            // 定期的な手動更新（WebSocketバックアップ）
            setInterval(loadInitialData, 30000);
        });
    </script>
</body>
</html>
            '''
            
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(html_content.strip())
                
    async def _handle_websocket(self, websocket: WebSocket):
        """WebSocket接続処理"""
        await websocket.accept()
        self.websocket_clients.append(websocket)
        
        try:
            while True:
                # クライアントからのメッセージ待機
                message = await websocket.receive_text()
                
                # ping/pong for keep-alive
                if message == "ping":
                    await websocket.send_text("pong")
                    
        except WebSocketDisconnect:
            self.websocket_clients.remove(websocket)
            self.logger.info("WebSocket client disconnected")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            if websocket in self.websocket_clients:
                self.websocket_clients.remove(websocket)
                
    async def _broadcast_updates(self):
        """全クライアントに更新を配信"""
        if not self.websocket_clients:
            return
            
        try:
            # 更新データ準備
            update_data = {
                "type": "metrics",
                "metrics": self.current_metrics.to_dict() if self.current_metrics else None,
                "timestamp": datetime.now().isoformat()
            }
            
            # アラート追加
            recent_alerts = list(self.alert_history)[-10:]  # 最新10件
            if recent_alerts:
                update_data["alerts"] = recent_alerts
                update_data["type"] = "metrics_and_alerts"
                
            message = json.dumps(update_data)
            
            # 全クライアントに送信
            disconnected_clients = []
            for client in self.websocket_clients:
                try:
                    await client.send_text(message)
                except Exception as e:
                    self.logger.warning(f"Failed to send to WebSocket client: {e}")
                    disconnected_clients.append(client)
                    
            # 切断されたクライアントを削除
            for client in disconnected_clients:
                self.websocket_clients.remove(client)
                
        except Exception as e:
            self.logger.error(f"Error broadcasting updates: {e}")
            
    def _update_system_metrics(self):
        """システムメトリクス更新"""
        try:
            # データフィード統計
            stats = self.data_feed_system.stats
            
            # 品質統計計算
            quality_stats = self._calculate_quality_stats()
            
            # システム性能取得
            performance_stats = self._get_performance_stats()
            
            # メトリクス作成
            self.current_metrics = SystemMetrics(
                timestamp=datetime.now(),
                total_data_points=stats.get('total_data_points', 0),
                data_points_per_second=self._calculate_data_rate(),
                active_symbols=len(self.data_feed_system.quality_history),
                avg_quality_score=quality_stats['avg_score'],
                quality_distribution=quality_stats['distribution'],
                quality_issues_count=quality_stats['issues_count'],
                memory_usage_mb=performance_stats['memory_mb'],
                cpu_usage_percent=performance_stats['cpu_percent'],
                cache_hit_rate=self._get_cache_hit_rate(),
                cache_size_mb=performance_stats['cache_mb'],
                error_count=stats.get('system_errors', 0),
                recovery_count=stats.get('data_corrections', 0),
                alert_count=len(self.alert_history),
                network_latency_ms=performance_stats['network_latency'],
                data_source_status=self._get_data_source_status()
            )
            
            # 履歴に追加
            self.metrics_history.append(self.current_metrics)
            
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {e}")
            self.exception_handler.handle_system_error(
                e, context={'operation': 'metrics_update'}
            )
            
    def _calculate_quality_stats(self) -> Dict[str, Any]:
        """品質統計計算"""
        try:
            all_quality_metrics = []
            for symbol_metrics in self.data_feed_system.quality_history.values():
                all_quality_metrics.extend(symbol_metrics)
                
            if not all_quality_metrics:
                return {
                    'avg_score': 0.0,
                    'distribution': {},
                    'issues_count': 0
                }
                
            # 平均スコア
            avg_score = np.mean([m.overall_score for m in all_quality_metrics])
            
            # 品質レベル分布
            distribution = defaultdict(int)
            issues_count = 0
            
            for metrics in all_quality_metrics:
                distribution[metrics.quality_level.value] += 1
                issues_count += len(metrics.issues)
                
            return {
                'avg_score': float(avg_score),
                'distribution': dict(distribution),
                'issues_count': issues_count
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating quality stats: {e}")
            return {'avg_score': 0.0, 'distribution': {}, 'issues_count': 0}
            
    def _calculate_data_rate(self) -> float:
        """データレート計算"""
        try:
            if len(self.metrics_history) < 2:
                return 0.0
                
            current = self.metrics_history[-1]
            previous = self.metrics_history[-2]
            
            time_diff = (current.timestamp - previous.timestamp).total_seconds()
            if time_diff <= 0:
                return 0.0
                
            data_diff = current.total_data_points - previous.total_data_points
            return data_diff / time_diff
            
        except Exception as e:
            self.logger.error(f"Error calculating data rate: {e}")
            return 0.0
            
    def _get_performance_stats(self) -> Dict[str, float]:
        """システム性能統計取得"""
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # メモリ使用量
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            # キャッシュサイズ
            cache_mb = 0.0
            if hasattr(self.data_feed_system.cache, 'get_size_mb'):
                cache_mb = self.data_feed_system.cache.get_size_mb()
                
            # ネットワーク遅延（推定）
            network_latency = self._estimate_network_latency()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'cache_mb': cache_mb,
                'network_latency': network_latency
            }
            
        except ImportError:
            self.logger.warning("psutil not available, using mock performance stats")
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'cache_mb': 0.0,
                'network_latency': 0.0
            }
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'cache_mb': 0.0,
                'network_latency': 0.0
            }
            
    def _get_cache_hit_rate(self) -> float:
        """キャッシュヒット率取得"""
        try:
            if hasattr(self.data_feed_system.cache, 'get_hit_rate'):
                return self.data_feed_system.cache.get_hit_rate()
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting cache hit rate: {e}")
            return 0.0
            
    def _estimate_network_latency(self) -> float:
        """ネットワーク遅延推定"""
        try:
            # データソースの応答時間から推定
            total_latency = 0.0
            source_count = 0
            
            for source_name, adapter in self.data_feed_system.data_manager.adapters.items():
                if hasattr(adapter, 'last_response_time'):
                    total_latency += adapter.last_response_time
                    source_count += 1
                    
            if source_count > 0:
                return total_latency / source_count
                
            return 50.0  # デフォルト値
            
        except Exception as e:
            self.logger.error(f"Error estimating network latency: {e}")
            return 50.0
            
    def _get_data_source_status(self) -> Dict[str, str]:
        """データソース状態取得"""
        try:
            status = {}
            for source_name, adapter in self.data_feed_system.data_manager.adapters.items():
                if hasattr(adapter, 'is_connected') and adapter.is_connected():
                    status[source_name] = "connected"
                else:
                    status[source_name] = "disconnected"
            return status
        except Exception as e:
            self.logger.error(f"Error getting data source status: {e}")
            return {}
            
    def _update_quality_history(self):
        """品質履歴更新"""
        try:
            # データフィードシステムから最新の品質メトリクスを取得
            for symbol, metrics_list in self.data_feed_system.quality_history.items():
                if metrics_list:
                    latest_metrics = metrics_list[-1]
                    self.quality_history[symbol].append(latest_metrics)
                    
        except Exception as e:
            self.logger.error(f"Error updating quality history: {e}")
            
    def _process_alerts(self):
        """アラート処理"""
        try:
            # データフィードシステムからアラートを取得
            if hasattr(self.data_feed_system, 'quality_alerts'):
                for alert in self.data_feed_system.quality_alerts:
                    if alert not in self.alert_history:
                        alert['timestamp'] = datetime.now()
                        self.alert_history.append(alert)
                        
        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")
            
    async def _generate_quality_chart(self, symbol: str) -> Dict[str, Any]:
        """品質チャート生成"""
        try:
            if symbol not in self.quality_history:
                return {"error": f"No data for symbol {symbol}"}
                
            metrics_list = list(self.quality_history[symbol])
            if not metrics_list:
                return {"error": f"No quality data for symbol {symbol}"}
                
            # タイムスタンプとスコア
            timestamps = [m.timestamp for m in metrics_list]
            overall_scores = [m.overall_score for m in metrics_list]
            completeness_scores = [m.completeness_score for m in metrics_list]
            accuracy_scores = [m.accuracy_score for m in metrics_list]
            timeliness_scores = [m.timeliness_score for m in metrics_list]
            consistency_scores = [m.consistency_score for m in metrics_list]
            
            # Plotlyチャート作成
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=overall_scores,
                mode='lines', name='総合スコア',
                line=dict(color=self.config.chart_colors['primary'], width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=completeness_scores,
                mode='lines', name='完全性',
                line=dict(color=self.config.chart_colors['good'], width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=accuracy_scores,
                mode='lines', name='精度',
                line=dict(color=self.config.chart_colors['excellent'], width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=timeliness_scores,
                mode='lines', name='適時性',
                line=dict(color=self.config.chart_colors['fair'], width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=consistency_scores,
                mode='lines', name='一貫性',
                line=dict(color=self.config.chart_colors['poor'], width=2)
            ))
            
            fig.update_layout(
                title=f'{symbol} データ品質推移',
                xaxis_title='時刻',
                yaxis_title='品質スコア',
                yaxis=dict(range=[0, 1]),
                hovermode='x unified',
                height=400
            )
            
            return {
                "data": fig.data,
                "layout": fig.layout
            }
            
        except Exception as e:
            self.logger.error(f"Error generating quality chart for {symbol}: {e}")
            return {"error": str(e)}
            
    def _get_uptime(self) -> str:
        """システム稼働時間取得"""
        try:
            uptime = datetime.now() - self.data_feed_system.stats.get('uptime_start', datetime.now())
            days = uptime.days
            hours, remainder = divmod(uptime.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            
            if days > 0:
                return f"{days}日 {hours}時間 {minutes}分"
            elif hours > 0:
                return f"{hours}時間 {minutes}分"
            else:
                return f"{minutes}分"
                
        except Exception as e:
            self.logger.error(f"Error calculating uptime: {e}")
            return "不明"
            
    def _get_cache_status(self) -> Dict[str, Any]:
        """キャッシュ状態取得"""
        try:
            cache = self.data_feed_system.cache
            return {
                "hit_rate": self._get_cache_hit_rate(),
                "size_mb": cache.get_size_mb() if hasattr(cache, 'get_size_mb') else 0.0,
                "item_count": cache.get_item_count() if hasattr(cache, 'get_item_count') else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting cache status: {e}")
            return {"hit_rate": 0.0, "size_mb": 0.0, "item_count": 0}
            
    def _get_error_summary(self) -> Dict[str, Any]:
        """エラーサマリー取得"""
        try:
            stats = self.data_feed_system.stats
            return {
                "total_errors": stats.get('system_errors', 0),
                "recoveries": stats.get('data_corrections', 0),
                "quality_failures": stats.get('quality_failures', 0),
                "recent_alerts": len([
                    alert for alert in self.alert_history
                    if alert.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=1)
                ])
            }
        except Exception as e:
            self.logger.error(f"Error getting error summary: {e}")
            return {"total_errors": 0, "recoveries": 0, "quality_failures": 0, "recent_alerts": 0}
            
    def start(self):
        """ダッシュボード開始"""
        try:
            self.system_status = "starting"
            
            # エージェント開始
            self.agent.start()
            
            self.system_status = "running"
            self.logger.info(f"Dashboard starting on {self.config.host}:{self.config.port}")
            
            # uvicornサーバー開始
            uvicorn.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level=self.config.log_level.lower()
            )
            
        except Exception as e:
            self.system_status = "error"
            self.logger.error(f"Error starting dashboard: {e}")
            self.exception_handler.handle_system_error(
                e, context={'operation': 'dashboard_start'}
            )
            raise
            
    def stop(self):
        """ダッシュボード停止"""
        try:
            self.system_status = "stopping"
            
            # エージェント停止
            self.agent.stop()
            
            # WebSocketクライアント切断
            for client in self.websocket_clients:
                try:
                    client.close()
                except:
                    pass
            self.websocket_clients.clear()
            
            self.system_status = "stopped"
            self.logger.info("Dashboard stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping dashboard: {e}")


def create_dashboard(data_feed_system: IntegratedDataFeedSystem,
                    config: Optional[DashboardConfig] = None) -> MonitoringDashboard:
    """ダッシュボード作成ファクトリ関数"""
    return MonitoringDashboard(data_feed_system, config)


if __name__ == "__main__":
    # スタンドアロンテスト
    from src.data.data_feed_integration import IntegratedDataFeedSystem
    
    # テスト用データフィードシステム
    data_feed = IntegratedDataFeedSystem()
    
    # ダッシュボード作成・開始
    dashboard = create_dashboard(data_feed)
    
    try:
        dashboard.start()
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        dashboard.stop()
