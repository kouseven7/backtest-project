"""
Strategy Performance Dashboard for 4-3-2
戦略比率とパフォーマンスのリアルタイム表示メインエンジン

既存システム統合:
- 4-3-1: ChartConfigManager, TrendStrategyTimeSeriesVisualizer
- 3-1: StrategySelector
- 3-2: PortfolioWeightCalculator  
- 3-3: StrategyScoreManager
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import pandas as pd

# プロジェクトパス追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 内部モジュールインポート
try:
    from .performance_data_collector import PerformanceDataCollector, PerformanceSnapshot
    from .dashboard_chart_generator import DashboardChartGenerator
    from .chart_config import ChartConfigManager
    from .dashboard_config import DashboardConfig
except ImportError:
    from performance_data_collector import PerformanceDataCollector, PerformanceSnapshot
    from dashboard_chart_generator import DashboardChartGenerator
    from chart_config import ChartConfigManager
    from dashboard_config import DashboardConfig

# 外部ライブラリ
try:
    import schedule
except ImportError:
    logging.warning("schedule not available, using time.sleep for scheduling")
    schedule = None

logger = logging.getLogger(__name__)

class StrategyPerformanceDashboard:
    """戦略パフォーマンスダッシュボード メインクラス"""
    
    def __init__(self, 
                 ticker: str = "USDJPY",
                 config_file: Optional[str] = None,
                 output_dir: str = "logs/dashboard"):
        
        self.ticker = ticker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定読み込み
        if config_file:
            self.config = DashboardConfig.load_from_file(config_file)
        else:
            self.config = DashboardConfig()
        
        # コンポーネント初期化
        self.data_collector = PerformanceDataCollector()
        self.chart_generator = DashboardChartGenerator()
        
        # 実行状態管理
        self.is_running = False
        self.last_update = None
        self.update_thread = None
        self.scheduler_thread = None
        
        # データキャッシュ
        self._current_snapshot = None
        self._historical_snapshots = []
        
        logger.info(f"StrategyPerformanceDashboard initialized for {ticker}")
    
    def start_dashboard(self, initial_market_data: Optional[pd.DataFrame] = None) -> bool:
        """ダッシュボード開始"""
        try:
            logger.info("Starting dashboard...")
            self.is_running = True
            
            # 初期データ収集
            if initial_market_data is not None:
                self._perform_update(initial_market_data)
            else:
                self._perform_update()
            
            # スケジューラー開始
            if schedule:
                self._start_scheduler()
            else:
                self._start_simple_timer()
            
            # 初期レポート生成
            self.generate_dashboard_report()
            
            logger.info("Dashboard started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            self.is_running = False
            return False
    
    def stop_dashboard(self) -> bool:
        """ダッシュボード停止"""
        try:
            logger.info("Stopping dashboard...")
            self.is_running = False
            
            # スケジューラー停止
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=2)
            
            # クリーンアップ処理
            self._cleanup_resources()
            
            logger.info("Dashboard stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop dashboard: {e}")
            return False
    
    def _start_scheduler(self):
        """スケジューラーの開始（scheduleライブラリ使用）"""
        try:
            # スケジュール設定
            update_interval = self.config.update_interval_minutes
            
            schedule.clear()
            schedule.every(update_interval).minutes.do(self._scheduled_update)
            
            # データクリーンアップ（日次）
            schedule.every().day.at("02:00").do(self._daily_cleanup)
            
            # スケジューラー実行スレッド
            def run_scheduler():
                while self.is_running:
                    try:
                        schedule.run_pending()
                        time.sleep(30)  # 30秒間隔でチェック
                    except Exception as e:
                        logger.warning(f"Scheduler error: {e}")
                        time.sleep(60)
            
            self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            logger.info(f"Scheduler started with {update_interval}min interval")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
    
    def _start_simple_timer(self):
        """シンプルなタイマー開始（scheduleライブラリなし）"""
        try:
            update_interval = self.config.update_interval_minutes * 60  # 秒に変換
            
            def run_timer():
                while self.is_running:
                    try:
                        time.sleep(update_interval)
                        if self.is_running:
                            self._scheduled_update()
                    except Exception as e:
                        logger.warning(f"Timer error: {e}")
                        time.sleep(60)
            
            self.scheduler_thread = threading.Thread(target=run_timer, daemon=True)
            self.scheduler_thread.start()
            
            logger.info(f"Simple timer started with {update_interval}sec interval")
            
        except Exception as e:
            logger.error(f"Failed to start simple timer: {e}")
    
    def _scheduled_update(self):
        """スケジュール実行される更新処理"""
        try:
            logger.debug("Scheduled update triggered")
            
            # データ更新実行
            self._perform_update()
            
            # レポート更新
            if self._current_snapshot:
                self.generate_dashboard_report()
            
        except Exception as e:
            logger.error(f"Scheduled update failed: {e}")
    
    def _perform_update(self, market_data: Optional[pd.DataFrame] = None):
        """データ更新実行"""
        try:
            # 新しいスナップショット収集
            new_snapshot = self.data_collector.collect_current_snapshot(
                self.ticker, market_data
            )
            
            if new_snapshot:
                # キャッシュ更新
                self._current_snapshot = new_snapshot
                
                # 履歴データ管理（簡易版）
                self._historical_snapshots.append(new_snapshot)
                # 30日分のデータのみ保持
                cutoff_date = datetime.now() - timedelta(days=30)
                self._historical_snapshots = [
                    s for s in self._historical_snapshots 
                    if s.timestamp >= cutoff_date
                ]
                
                self.last_update = datetime.now()
                logger.debug(f"Data updated for {self.ticker}")
            
        except Exception as e:
            logger.error(f"Update execution failed: {e}")
    
    def generate_dashboard_report(self) -> Optional[str]:
        """ダッシュボードレポートの生成"""
        try:
            if not self._current_snapshot:
                logger.warning("No current snapshot available for report generation")
                return None
            
            # 1. チャート生成
            chart_path = self.chart_generator.generate_performance_dashboard(
                self._current_snapshot,
                self._historical_snapshots
            )
            
            # 2. HTML レポート生成
            html_path = self._generate_html_report(chart_path)
            
            # 3. サマリー生成
            summary = self.chart_generator.generate_simple_summary(self._current_snapshot)
            self._save_text_summary(summary)
            
            logger.info(f"Dashboard report generated: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None
    
    def _generate_html_report(self, chart_path: Optional[str]) -> str:
        """HTML レポートの生成（シンプル版）"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_filename = f"dashboard_report_{self.ticker}_{timestamp}.html"
            html_path = self.output_dir / "dashboard_reports" / html_filename
            html_path.parent.mkdir(parents=True, exist_ok=True)
            
            # シンプルHTML生成
            return self._generate_simple_html(html_path, chart_path)
            
        except Exception as e:
            logger.error(f"HTML generation failed: {e}")
            return str(html_path) if 'html_path' in locals() else ""
    
    def _generate_simple_html(self, html_path: Path, chart_path: Optional[str]) -> str:
        """シンプルHTML生成（テンプレートなし）"""
        try:
            snapshot = self._current_snapshot
            
            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>戦略パフォーマンスダッシュボード - {self.ticker}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .panel {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .metric-name {{ font-weight: bold; }}
        .metric-value {{ color: #333; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .alert {{ background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin: 5px 0; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .timestamp {{ font-size: 0.9em; color: #666; }}
        @media (max-width: 768px) {{
            .container {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>戦略パフォーマンスダッシュボード - {self.ticker}</h1>
        <div class="timestamp">{snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')} 現在</div>
    </div>
    
    <div class="container">
        <div class="panel">
            <h3>主要パフォーマンス指標</h3>
            <div class="metric">
                <span class="metric-name">ポートフォリオリターン:</span>
                <span class="metric-value {'positive' if snapshot.total_performance.get('portfolio_return', 0) > 0 else 'negative'}">
                    {snapshot.total_performance.get('portfolio_return', 0):.2f}%
                </span>
            </div>
            <div class="metric">
                <span class="metric-name">シャープレシオ:</span>
                <span class="metric-value">{snapshot.total_performance.get('sharpe_ratio', 0):.3f}</span>
            </div>
            <div class="metric">
                <span class="metric-name">リスクスコア:</span>
                <span class="metric-value">{snapshot.risk_metrics.get('risk_score', 0):.1f}/100</span>
            </div>
        </div>
        
        <div class="panel">
            <h3>戦略配分</h3>
            {self._format_strategy_allocations_html(snapshot.strategy_allocations)}
        </div>
    </div>
    
    {f'<div class="chart"><img src="{Path(chart_path).name}" alt="Performance Chart" style="max-width: 100%; height: auto;"></div>' if chart_path else ''}
    
    {self._format_alerts_html(snapshot.alerts)}
    
    <div class="panel">
        <h3>市場コンテキスト</h3>
        <div class="metric">
            <span class="metric-name">トレンド:</span>
            <span class="metric-value">{snapshot.market_context.get('trend', '不明')}</span>
        </div>
        <div class="metric">
            <span class="metric-name">信頼度:</span>
            <span class="metric-value">{snapshot.market_context.get('trend_confidence', 0):.2f}</span>
        </div>
    </div>
</body>
</html>
            """
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(html_path)
            
        except Exception as e:
            logger.error(f"Simple HTML generation failed: {e}")
            return str(html_path)
    
    def _format_strategy_allocations_html(self, allocations: Dict[str, float]) -> str:
        """戦略配分のHTML フォーマット"""
        if not allocations:
            return "<p>配分データなし</p>"
        
        html_parts = []
        for strategy, weight in allocations.items():
            percentage = weight * 100
            html_parts.append(f'''
                <div class="metric">
                    <span class="metric-name">{strategy}:</span>
                    <span class="metric-value">{percentage:.1f}%</span>
                </div>
            ''')
        
        return ''.join(html_parts)
    
    def _format_alerts_html(self, alerts: List[str]) -> str:
        """アラートのHTML フォーマット"""
        if not alerts:
            return ""
        
        alert_html = '<div class="panel"><h3>アラート</h3>'
        for alert in alerts:
            alert_html += f'<div class="alert">{alert}</div>'
        alert_html += '</div>'
        
        return alert_html
    
    def _save_text_summary(self, summary: str):
        """テキストサマリーの保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = f"dashboard_summary_{self.ticker}_{timestamp}.txt"
            summary_path = self.output_dir / "dashboard_reports" / summary_filename
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            logger.debug(f"Text summary saved: {summary_filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save text summary: {e}")
    
    def _daily_cleanup(self):
        """日次クリーンアップ処理"""
        try:
            logger.info("Running daily cleanup...")
            
            # 古いデータの削除
            self.data_collector.cleanup_old_data()
            self.chart_generator.cleanup_old_charts()
            
            logger.info("Daily cleanup completed")
            
        except Exception as e:
            logger.error(f"Daily cleanup failed: {e}")
    
    def _cleanup_resources(self):
        """リソースクリーンアップ"""
        try:
            # キャッシュクリア
            self._current_snapshot = None
            self._historical_snapshots = []
            
            # スケジューラークリア
            if schedule:
                schedule.clear()
            
            logger.debug("Resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Resource cleanup warning: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """ダッシュボード状態の取得"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'ticker': self.ticker,
            'current_snapshot_available': self._current_snapshot is not None,
            'historical_snapshots_count': len(self._historical_snapshots),
            'update_interval_minutes': self.config.update_interval_minutes
        }
    
    def manual_update(self, market_data: Optional[pd.DataFrame] = None) -> bool:
        """手動更新実行"""
        try:
            logger.info("Manual update triggered")
            self._perform_update(market_data)
            
            if self._current_snapshot:
                report_path = self.generate_dashboard_report()
                logger.info(f"Manual update completed, report: {report_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Manual update failed: {e}")
            return False

# ユーティリティ関数
def create_dashboard(ticker: str = "USDJPY", 
                    config_file: Optional[str] = None) -> StrategyPerformanceDashboard:
    """ダッシュボード作成ファクトリー関数"""
    return StrategyPerformanceDashboard(
        ticker=ticker,
        config_file=config_file
    )

if __name__ == "__main__":
    # テスト実行
    dashboard = create_dashboard("USDJPY")
    
    try:
        if dashboard.start_dashboard():
            print("Dashboard started successfully")
            print(f"Status: {dashboard.get_status()}")
            
            # 5秒後に手動更新テスト
            time.sleep(5)
            dashboard.manual_update()
            
            # 10秒後に停止
            time.sleep(10)
            dashboard.stop_dashboard()
            print("Dashboard test completed")
        
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        dashboard.stop_dashboard()
