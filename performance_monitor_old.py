"""
フェーズ4A2: パフォーマンス監視スクリプト
戦略別詳細監視とポートフォリオレベル統合監視
既存のpaper_trade_runnerと連携してリアルタイムパフォーマンス監視を実行
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 既存システム統合
try:
    from src.monitoring.dashboard import MonitoringDashboard
    from src.monitoring.metrics_collector import MetricsCollector
    from src.monitoring.alert_manager import AlertManager
    from src.execution.paper_trade_monitor import PaperTradeMonitor
    from src.execution.strategy_execution_manager import StrategyExecutionManager
    from config.logger_config import setup_logger
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("基本的なモニタリング機能のみで動作します")
    setup_logger = None

class PerformanceMonitor:
    """
    統合パフォーマンス監視システム
    - 戦略別詳細監視
    - 15分間隔定期更新
    - 既存ダッシュボード完全統合
    """
    
    def __init__(self, config_path: str = "config/performance_monitoring/monitoring_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # ログ設定
        if setup_logger:
            self.logger = setup_logger("performance_monitor", "logs/performance_monitor.log")
        else:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("performance_monitor")
        
        # 既存システム統合
        self._initialize_components()
        
        # 監視状態管理
        self.is_running = False
        self.monitoring_tasks = []
        self.performance_history = {}
        self.strategy_trackers = {}
        
        # 出力ディレクトリ作成
        self.output_dir = Path(self.config.get('output_settings', {}).get('file_output', {}).get('output_directory', 'logs/performance_monitoring'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict:
        """設定ファイル読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            "monitoring_settings": {
                "update_interval_seconds": 900,  # 15分間隔
                "history_retention_days": 30,
                "max_memory_usage_mb": 1024
            },
            "strategy_monitoring": {
                "monitor_all_strategies": True,
                "min_trades_for_analysis": 5
            },
            "metrics_configuration": {
                "basic_metrics": {"enabled": True},
                "risk_metrics": {"enabled": True},
                "performance_metrics": {"enabled": True}
            },
            "output_settings": {
                "console_output": {"enabled": True},
                "log_output": {"enabled": True},
                "file_output": {"enabled": True}
            },
            "alert_rules": {
                "portfolio_rules": {
                    "max_drawdown_threshold": 0.15,
                    "daily_loss_limit": 0.05
                }
            }
        }
    
    def _initialize_components(self):
        """既存システムコンポーネント初期化"""
        try:
            # 既存ダッシュボード統合
            self.dashboard = None
            try:
                self.dashboard = MonitoringDashboard()
                self.logger.info("既存ダッシュボード統合成功")
            except:
                self.logger.warning("ダッシュボード統合失敗、独立動作モード")
            
            # メトリクス収集システム
            self.metrics_collector = None
            try:
                self.metrics_collector = MetricsCollector()
                self.logger.info("メトリクス収集システム統合成功")
            except:
                self.logger.warning("メトリクス収集システム統合失敗")
            
            # アラート管理システム
            self.alert_manager = None
            try:
                self.alert_manager = AlertManager()
                self.logger.info("アラート管理システム統合成功")
            except:
                self.logger.warning("アラート管理システム統合失敗")
            
            # パフォーマンスアラート管理
            self.performance_alert_manager = PerformanceAlertManager()
            
            # ポートフォリオ分析器
            self.portfolio_analyzer = PortfolioPerformanceAnalyzer(self.config)
            
        except Exception as e:
            self.logger.error(f"コンポーネント初期化エラー: {e}")
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        self.logger.info(f"終了シグナル受信: {signum}")
        self.stop_monitoring()
        sys.exit(0)
    
    async def start_monitoring(self):
        """監視開始"""
        self.logger.info("=== パフォーマンス監視開始 ===")
        self.is_running = True
        
        try:
            # 既存データ読み込み
            await self._load_existing_data()
            
            # 監視タスク開始
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            data_collection_task = asyncio.create_task(self._data_collection_loop())
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.monitoring_tasks = [monitoring_task, data_collection_task, cleanup_task]
            
            # 全タスク完了まで待機
            await asyncio.gather(*self.monitoring_tasks)
            
        except Exception as e:
            self.logger.error(f"監視開始エラー: {e}")
            self.stop_monitoring()
    
    async def _monitoring_loop(self):
        """メイン監視ループ"""
        interval = self.config.get('monitoring_settings', {}).get('update_interval_seconds', 900)
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # パフォーマンス分析実行
                portfolio_analysis = await self._analyze_current_performance()
                
                # アラートチェック
                alerts = self.performance_alert_manager.check_performance_alerts(portfolio_analysis)
                
                # 結果出力
                await self._output_results(portfolio_analysis, alerts)
                
                # ダッシュボード更新
                if self.dashboard and self.config.get('dashboard_integration', {}).get('enabled', True):
                    await self._update_dashboard(portfolio_analysis)
                
                execution_time = time.time() - start_time
                self.logger.info(f"監視サイクル完了: {execution_time:.2f}秒")
                
                # 次回実行まで待機
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                await asyncio.sleep(60)  # エラー時は1分待機
    
    async def _data_collection_loop(self):
        """データ収集ループ"""
        polling_interval = self.config.get('data_sources', {}).get('paper_trade_runner', {}).get('polling_interval', 60)
        
        while self.is_running:
            try:
                # ペーパートレードデータ収集
                await self._collect_paper_trade_data()
                
                # 戦略実行データ収集
                await self._collect_strategy_execution_data()
                
                # システムメトリクス収集
                await self._collect_system_metrics()
                
                await asyncio.sleep(polling_interval)
                
            except Exception as e:
                self.logger.error(f"データ収集エラー: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self):
        """クリーンアップループ"""
        cleanup_interval = 3600  # 1時間間隔
        retention_days = self.config.get('monitoring_settings', {}).get('history_retention_days', 30)
        
        while self.is_running:
            try:
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                # 古いデータ削除
                self._cleanup_old_data(cutoff_date)
                
                # メモリ使用量チェック
                self._check_memory_usage()
                
                await asyncio.sleep(cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"クリーンアップエラー: {e}")
                await asyncio.sleep(1800)  # エラー時は30分待機
    
    async def _load_existing_data(self):
        """既存データ読み込み"""
        try:
            # ペーパートレード履歴読み込み
            paper_trade_logs = Path("logs/paper_trading")
            if paper_trade_logs.exists():
                self.logger.info("ペーパートレード履歴読み込み中...")
                # 実装: ログファイルから履歴データ復元
            
            # 戦略実行履歴読み込み
            strategy_logs = Path("logs/strategy_execution.log")
            if strategy_logs.exists():
                self.logger.info("戦略実行履歴読み込み中...")
                # 実装: 戦略実行ログから履歴データ復元
            
            self.logger.info("既存データ読み込み完了")
            
        except Exception as e:
            self.logger.error(f"既存データ読み込みエラー: {e}")
    
    async def _analyze_current_performance(self) -> Dict:
        """現在のパフォーマンス分析"""
        try:
            # 各戦略のパフォーマンス更新
            strategy_performances = {}
            for strategy_name, tracker in self.strategy_trackers.items():
                try:
                    performance = await tracker.get_current_performance()
                    strategy_performances[strategy_name] = performance
                except Exception as e:
                    self.logger.error(f"戦略パフォーマンス取得エラー [{strategy_name}]: {e}")
            
            # サンプル戦略データ（テスト用）
            if not strategy_performances:
                strategy_performances = self._generate_sample_strategy_data()
            
            # ポートフォリオレベル分析
            portfolio_analysis = self.portfolio_analyzer.analyze_portfolio_performance(strategy_performances)
            
            # 履歴に記録
            self.performance_history[datetime.now().isoformat()] = portfolio_analysis
            
            return portfolio_analysis
            
        except Exception as e:
            self.logger.error(f"パフォーマンス分析エラー: {e}")
            return {}
    
    def _generate_sample_strategy_data(self) -> Dict:
        """サンプル戦略データ生成（テスト用）"""
        import random
        
        sample_strategies = ["VWAP_Breakout", "Momentum_Investing", "Opening_Gap"]
        strategy_data = {}
        
        for strategy_name in sample_strategies:
            strategy_data[strategy_name] = {
                "basic_metrics": {
                    "total_trades": random.randint(10, 50),
                    "winning_trades": random.randint(5, 25),
                    "total_pnl": random.uniform(-1000, 2000),
                    "win_rate": random.uniform(0.3, 0.7)
                },
                "risk_metrics": {
                    "volatility": random.uniform(0.1, 0.4),
                    "max_drawdown": random.uniform(-0.2, -0.05),
                    "sharpe_ratio": random.uniform(-0.5, 2.0)
                },
                "performance_score": random.uniform(0.2, 0.9),
                "timestamp": datetime.now()
            }
        
    
    async def _collect_paper_trade_data(self):
        """ペーパートレードデータ収集"""
        try:
            data_path = self.config.get('data_sources', {}).get('paper_trade_runner', {}).get('data_path', 'logs/paper_trading')
            data_dir = Path(data_path)
            
            if data_dir.exists():
                # 最新のログファイルを読み込み
                log_files = list(data_dir.glob("*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    # ログファイル解析実装
                    
        except Exception as e:
            self.logger.error(f"ペーパートレードデータ収集エラー: {e}")
    
    async def _collect_strategy_execution_data(self):
        """戦略実行データ収集"""
        try:
            log_path = self.config.get('data_sources', {}).get('strategy_execution_logs', {}).get('log_path', 'logs/strategy_execution.log')
            log_file = Path(log_path)
            
            if log_file.exists():
                # ログファイル解析実装
                pass
                
        except Exception as e:
            self.logger.error(f"戦略実行データ収集エラー: {e}")
    
    async def _collect_system_metrics(self):
        """システムメトリクス収集"""
        try:
            if self.metrics_collector:
                # 既存メトリクス収集システム利用
                metrics = self.metrics_collector.get_current_metrics()
                # システムメトリクス処理実装
                
        except Exception as e:
            self.logger.error(f"システムメトリクス収集エラー: {e}")
    
    async def _output_results(self, portfolio_analysis: Dict, alerts: List[Dict]):
        """結果出力"""
        try:
            output_settings = self.config.get('output_settings', {})
            
            # コンソール出力
            if output_settings.get('console_output', {}).get('enabled', True):
                self._output_to_console(portfolio_analysis, alerts)
            
            # ファイル出力
            if output_settings.get('file_output', {}).get('enabled', True):
                await self._output_to_files(portfolio_analysis, alerts)
            
        except Exception as e:
            self.logger.error(f"結果出力エラー: {e}")
    
    def _output_to_console(self, portfolio_analysis: Dict, alerts: List[Dict]):
        """コンソール出力"""
        try:
            print(f"\n=== パフォーマンス監視レポート [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
            
            # ポートフォリオサマリー
            portfolio_metrics = portfolio_analysis.get('portfolio_metrics', {})
            if portfolio_metrics:
                print(f"📊 ポートフォリオサマリー:")
                print(f"  総PnL: {portfolio_metrics.get('total_pnl', 0):.2f}")
                print(f"  シャープレシオ: {portfolio_metrics.get('sharpe_ratio', 0):.3f}")
                print(f"  最大ドローダウン: {portfolio_metrics.get('max_drawdown', 0):.2%}")
                print(f"  勝率: {portfolio_metrics.get('win_rate', 0):.1%}")
            
            # 戦略別パフォーマンス
            strategy_performances = portfolio_analysis.get('strategy_performances', {})
            if strategy_performances:
                print(f"\n🔍 戦略別パフォーマンス:")
                for strategy_name, performance in strategy_performances.items():
                    basic_metrics = performance.get('basic_metrics', {})
                    print(f"  {strategy_name}:")
                    print(f"    取引数: {basic_metrics.get('total_trades', 0)}")
                    print(f"    勝率: {basic_metrics.get('win_rate', 0):.1%}")
                    print(f"    PnL: {basic_metrics.get('total_pnl', 0):.2f}")
            
            # アラート表示
            if alerts:
                print(f"\n🚨 アラート ({len(alerts)}件):")
                for alert in alerts:
                    severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(alert.get('severity', 'low'), "⚪")
                    print(f"  {severity_icon} {alert.get('message', 'Unknown alert')}")
            else:
                print(f"\n✅ アラートなし")
            
            print("=" * 80)
            
        except Exception as e:
            self.logger.error(f"コンソール出力エラー: {e}")
    
    async def _output_to_files(self, portfolio_analysis: Dict, alerts: List[Dict]):
        """ファイル出力"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSON出力
            json_file = self.output_dir / f"performance_analysis_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'portfolio_analysis': portfolio_analysis,
                    'alerts': alerts
                }, f, indent=2, ensure_ascii=False, default=str)
            
            # CSV出力（戦略別メトリクス）
            if portfolio_analysis.get('strategy_performances'):
                csv_file = self.output_dir / f"strategy_metrics_{timestamp}.csv"
                strategy_data = []
                for name, perf in portfolio_analysis.get('strategy_performances', {}).items():
                    row = {'strategy': name, 'timestamp': datetime.now().isoformat()}
                    row.update(perf.get('basic_metrics', {}))
                    row.update(perf.get('risk_metrics', {}))
                    strategy_data.append(row)
                
                if strategy_data:
                    pd.DataFrame(strategy_data).to_csv(csv_file, index=False)
            
        except Exception as e:
            self.logger.error(f"ファイル出力エラー: {e}")
    
    async def _update_dashboard(self, portfolio_analysis: Dict):
        """ダッシュボード更新"""
        try:
            if self.dashboard:
                # 既存ダッシュボードにデータ送信
                # Note: MonitoringDashboardのAPIに合わせて実装
                pass
            
        except Exception as e:
            self.logger.error(f"ダッシュボード更新エラー: {e}")
    
    def _cleanup_old_data(self, cutoff_date: datetime):
        """古いデータクリーンアップ"""
        try:
            # 履歴データクリーンアップ
            old_keys = [k for k, v in self.performance_history.items() 
                       if datetime.fromisoformat(k) < cutoff_date]
            for key in old_keys:
                del self.performance_history[key]
            
            # ファイルクリーンアップ
            for file_path in self.output_dir.glob("performance_analysis_*.json"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
            
            if old_keys:
                self.logger.info(f"古いデータクリーンアップ: {len(old_keys)}件")
                
        except Exception as e:
            self.logger.error(f"データクリーンアップエラー: {e}")
    
    def _check_memory_usage(self):
        """メモリ使用量チェック"""
        try:
            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            max_usage = self.config.get('monitoring_settings', {}).get('max_memory_usage_mb', 1024)
            
            if memory_usage > max_usage:
                self.logger.warning(f"メモリ使用量警告: {memory_usage:.1f}MB > {max_usage}MB")
                # メモリクリーンアップ実行
                self._force_cleanup()
                
        except ImportError:
            pass  # psutilが利用できない場合はスキップ
        except Exception as e:
            self.logger.error(f"メモリチェックエラー: {e}")
    
    def _force_cleanup(self):
        """強制クリーンアップ"""
        try:
            # 履歴データの半分を削除
            if len(self.performance_history) > 100:
                sorted_keys = sorted(self.performance_history.keys())
                keys_to_remove = sorted_keys[:len(sorted_keys)//2]
                for key in keys_to_remove:
                    del self.performance_history[key]
                self.logger.info(f"強制クリーンアップ: {len(keys_to_remove)}件削除")
                
        except Exception as e:
            self.logger.error(f"強制クリーンアップエラー: {e}")
    
    def stop_monitoring(self):
        """監視停止"""
        self.logger.info("監視停止中...")
        self.is_running = False
        
        # タスクキャンセル
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        self.logger.info("監視停止完了")
    
    def get_status(self) -> Dict:
        """監視状態取得"""
        return {
            "is_running": self.is_running,
            "strategy_count": len(self.strategy_trackers),
            "history_count": len(self.performance_history),
            "last_update": max(self.performance_history.keys()) if self.performance_history else None
        }


class StrategyPerformanceTracker:
    """戦略別パフォーマンス追跡・分析"""
    
    def __init__(self, strategy_name: str, config: Dict):
        self.strategy_name = strategy_name
        self.config = config
        self.performance_history = []
        self.current_metrics = {}
        
        # パフォーマンス追跡データ
        self.trades_history = []
        self.signals_history = []
        self.portfolio_values = []
        self.risk_metrics = {}
    
    async def get_current_performance(self) -> Dict:
        """現在のパフォーマンス取得"""
        try:
            # 基本メトリクス計算
            basic_metrics = self._calculate_basic_metrics()
            
            # リスクメトリクス計算
            risk_metrics = self._calculate_risk_metrics()
            
            # パフォーマンススコア計算
            performance_score = self._calculate_performance_score()
            
            performance_data = {
                "timestamp": datetime.now(),
                "strategy": self.strategy_name,
                "basic_metrics": basic_metrics,
                "risk_metrics": risk_metrics,
                "performance_score": performance_score
            }
            
            self.current_metrics = performance_data
            return performance_data
            
        except Exception as e:
            logging.error(f"パフォーマンス取得エラー [{self.strategy_name}]: {e}")
            return {}
    
    def _calculate_basic_metrics(self) -> Dict:
        """基本メトリクス計算"""
        if not self.trades_history:
            return {}
        
        trades_df = pd.DataFrame(self.trades_history)
        if trades_df.empty:
            return {}
        
        return {
            "total_trades": len(trades_df),
            "winning_trades": len(trades_df[trades_df.get('pnl', 0) > 0]),
            "losing_trades": len(trades_df[trades_df.get('pnl', 0) < 0]),
            "win_rate": len(trades_df[trades_df.get('pnl', 0) > 0]) / len(trades_df) if len(trades_df) > 0 else 0,
            "total_pnl": trades_df.get('pnl', pd.Series([0])).sum(),
            "average_pnl": trades_df.get('pnl', pd.Series([0])).mean(),
            "best_trade": trades_df.get('pnl', pd.Series([0])).max(),
            "worst_trade": trades_df.get('pnl', pd.Series([0])).min()
        }
    
    def _calculate_risk_metrics(self) -> Dict:
        """リスクメトリクス計算"""
        if not self.portfolio_values:
            return {}
        
        values = pd.Series(self.portfolio_values)
        returns = values.pct_change().dropna()
        
        if returns.empty:
            return {}
        
        return {
            "volatility": returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            "max_drawdown": self._calculate_max_drawdown(values),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns),
            "var_95": returns.quantile(0.05) if len(returns) > 0 else 0,
            "current_drawdown": self._calculate_current_drawdown(values)
        }
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """最大ドローダウン計算"""
        try:
            if values.empty:
                return 0
            peak = values.cummax()
            drawdown = (values - peak) / peak
            return drawdown.min()
        except:
            return 0
    
    def _calculate_current_drawdown(self, values: pd.Series) -> float:
        """現在のドローダウン計算"""
        try:
            if values.empty:
                return 0
            peak = values.max()
            current = values.iloc[-1]
            return (current - peak) / peak if peak > 0 else 0
        except:
            return 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """シャープレシオ計算"""
        try:
            if returns.empty or returns.std() == 0:
                return 0
            return returns.mean() / returns.std() * np.sqrt(252)
        except:
            return 0
    
    def _calculate_performance_score(self) -> float:
        """パフォーマンススコア計算"""
        try:
            basic = self._calculate_basic_metrics()
            risk = self._calculate_risk_metrics()
            
            if not basic or not risk:
                return 0
            
            # 簡易スコア計算
            win_rate_score = basic.get('win_rate', 0) * 0.3
            pnl_score = min(basic.get('total_pnl', 0) / 1000, 1) * 0.4  # 正規化
            sharpe_score = min(risk.get('sharpe_ratio', 0) / 2, 1) * 0.3  # 正規化
            
            return max(0, min(1, win_rate_score + pnl_score + sharpe_score))
            
        except:
            return 0


class PortfolioPerformanceAnalyzer:
    """ポートフォリオレベル統合パフォーマンス分析"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.portfolio_history = []
        self.correlation_matrix = None
    
    def analyze_portfolio_performance(self, strategy_performances: Dict) -> Dict:
        """ポートフォリオ統合分析"""
        try:
            if not strategy_performances:
                return {}
            
            # ポートフォリオレベル計算
            portfolio_metrics = self._calculate_portfolio_metrics(strategy_performances)
            
            # 戦略間相関分析
            correlation_analysis = self._analyze_strategy_correlations(strategy_performances)
            
            # リスク寄与分析
            risk_contribution = self._analyze_risk_contribution(strategy_performances)
            
            # ポートフォリオヘルススコア
            health_score = self._calculate_portfolio_health_score(portfolio_metrics)
            
            return {
                "timestamp": datetime.now(),
                "portfolio_metrics": portfolio_metrics,
                "strategy_performances": strategy_performances,
                "correlation_analysis": correlation_analysis,
                "risk_contribution": risk_contribution,
                "portfolio_health_score": health_score
            }
            
        except Exception as e:
            logging.error(f"ポートフォリオ分析エラー: {e}")
            return {}
    
    def _calculate_portfolio_metrics(self, strategy_performances: Dict) -> Dict:
        """ポートフォリオメトリクス計算"""
        try:
            if not strategy_performances:
                return {}
            
            total_pnl = 0
            total_trades = 0
            winning_trades = 0
            
            for strategy_name, performance in strategy_performances.items():
                basic_metrics = performance.get('basic_metrics', {})
                total_pnl += basic_metrics.get('total_pnl', 0)
                total_trades += basic_metrics.get('total_trades', 0)
                winning_trades += basic_metrics.get('winning_trades', 0)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # リスクメトリクス統合
            risk_metrics = []
            for performance in strategy_performances.values():
                risk_data = performance.get('risk_metrics', {})
                if risk_data:
                    risk_metrics.append(risk_data)
            
            avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in risk_metrics]) if risk_metrics else 0
            max_drawdown = min([r.get('max_drawdown', 0) for r in risk_metrics]) if risk_metrics else 0
            
            return {
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "sharpe_ratio": avg_sharpe,
                "max_drawdown": max_drawdown,
                "active_strategies": len(strategy_performances)
            }
            
        except Exception as e:
            logging.error(f"ポートフォリオメトリクス計算エラー: {e}")
            return {}
    
    def _analyze_strategy_correlations(self, strategy_performances: Dict) -> Dict:
        """戦略間相関分析"""
        try:
            if len(strategy_performances) < 2:
                return {}
            
            # 相関分析実装（簡易版）
            correlation_data = {}
            strategy_names = list(strategy_performances.keys())
            
            for i, strategy1 in enumerate(strategy_names):
                for j, strategy2 in enumerate(strategy_names[i+1:], i+1):
                    # 実装: 戦略間相関計算
                    correlation_data[f"{strategy1}_vs_{strategy2}"] = 0.0  # プレースホルダー
            
            return {
                "correlation_matrix": correlation_data,
                "avg_correlation": np.mean(list(correlation_data.values())) if correlation_data else 0
            }
            
        except Exception as e:
            logging.error(f"相関分析エラー: {e}")
            return {}
    
    def _analyze_risk_contribution(self, strategy_performances: Dict) -> Dict:
        """リスク寄与分析"""
        try:
            risk_contributions = {}
            total_risk = 0
            
            for strategy_name, performance in strategy_performances.items():
                risk_metrics = performance.get('risk_metrics', {})
                volatility = risk_metrics.get('volatility', 0)
                risk_contributions[strategy_name] = volatility
                total_risk += volatility
            
            # 正規化
            if total_risk > 0:
                for strategy_name in risk_contributions:
                    risk_contributions[strategy_name] /= total_risk
            
            return risk_contributions
            
        except Exception as e:
            logging.error(f"リスク寄与分析エラー: {e}")
            return {}
    
    def _calculate_portfolio_health_score(self, portfolio_metrics: Dict) -> float:
        """ポートフォリオヘルススコア計算"""
        try:
            if not portfolio_metrics:
                return 0
            
            # ヘルススコア計算
            win_rate_score = portfolio_metrics.get('win_rate', 0) * 0.3
            sharpe_score = min(portfolio_metrics.get('sharpe_ratio', 0) / 2, 1) * 0.4
            drawdown_score = max(0, 1 + portfolio_metrics.get('max_drawdown', 0)) * 0.3  # ドローダウンは負の値
            
            return max(0, min(1, win_rate_score + sharpe_score + drawdown_score))
            
        except:
            return 0


class PerformanceAlertManager:
    """パフォーマンス監視アラート管理"""
    
    def __init__(self, config_path: str = "config/performance_monitoring/alert_rules.json"):
        self.config_path = Path(config_path)
        self.alert_rules = self._load_alert_rules()
        self.active_alerts = {}
    
    def _load_alert_rules(self) -> Dict:
        """アラートルール読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._get_default_alert_rules()
        except:
            return self._get_default_alert_rules()
    
    def _get_default_alert_rules(self) -> Dict:
        """デフォルトアラートルール"""
        return {
            "portfolio_rules": {
                "max_drawdown_threshold": 0.15,
                "daily_loss_limit": 0.05,
                "volatility_threshold": 0.3
            },
            "strategy_rules": {
                "min_win_rate": 0.4,
                "max_consecutive_losses": 5
            }
        }
    
    def check_performance_alerts(self, portfolio_analysis: Dict) -> List[Dict]:
        """パフォーマンスアラートチェック"""
        alerts = []
        
        try:
            # ポートフォリオレベルアラート
            portfolio_alerts = self._check_portfolio_alerts(portfolio_analysis.get('portfolio_metrics', {}))
            alerts.extend(portfolio_alerts)
            
            # 戦略別アラート
            for strategy_name, performance in portfolio_analysis.get('strategy_performances', {}).items():
                strategy_alerts = self._check_strategy_alerts(strategy_name, performance)
                alerts.extend(strategy_alerts)
            
            # アラート処理
            for alert in alerts:
                self._process_alert(alert)
            
            return alerts
            
        except Exception as e:
            logging.error(f"アラートチェックエラー: {e}")
            return []
    
    def _check_portfolio_alerts(self, portfolio_metrics: Dict) -> List[Dict]:
        """ポートフォリオレベルアラートチェック"""
        alerts = []
        rules = self.alert_rules.get('portfolio_rules', {})
        
        try:
            # ドローダウンアラート
            max_drawdown = portfolio_metrics.get('max_drawdown', 0)
            if max_drawdown < -rules.get('max_drawdown_threshold', 0.15):
                alerts.append({
                    "type": "portfolio_drawdown",
                    "severity": "high",
                    "message": f"ポートフォリオドローダウン警告: {max_drawdown:.2%}",
                    "value": max_drawdown,
                    "threshold": -rules.get('max_drawdown_threshold'),
                    "timestamp": datetime.now()
                })
            
            # 勝率アラート
            win_rate = portfolio_metrics.get('win_rate', 0)
            if win_rate < rules.get('min_win_rate', 0.4):
                alerts.append({
                    "type": "portfolio_win_rate",
                    "severity": "medium",
                    "message": f"ポートフォリオ勝率低下: {win_rate:.1%}",
                    "value": win_rate,
                    "threshold": rules.get('min_win_rate'),
                    "timestamp": datetime.now()
                })
            
        except Exception as e:
            logging.error(f"ポートフォリオアラートチェックエラー: {e}")
        
        return alerts
    
    def _check_strategy_alerts(self, strategy_name: str, performance: Dict) -> List[Dict]:
        """戦略別アラートチェック"""
        alerts = []
        rules = self.alert_rules.get('strategy_rules', {})
        
        try:
            basic_metrics = performance.get('basic_metrics', {})
            risk_metrics = performance.get('risk_metrics', {})
            
            # 勝率アラート
            win_rate = basic_metrics.get('win_rate', 0)
            if win_rate < rules.get('min_win_rate', 0.4):
                alerts.append({
                    "type": "strategy_win_rate",
                    "severity": "medium",
                    "strategy": strategy_name,
                    "message": f"戦略勝率低下 [{strategy_name}]: {win_rate:.1%}",
                    "value": win_rate,
                    "threshold": rules.get('min_win_rate'),
                    "timestamp": datetime.now()
                })
            
            # ドローダウンアラート
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            if max_drawdown < -0.1:  # 10%以上のドローダウン
                alerts.append({
                    "type": "strategy_drawdown",
                    "severity": "high",
                    "strategy": strategy_name,
                    "message": f"戦略ドローダウン警告 [{strategy_name}]: {max_drawdown:.2%}",
                    "value": max_drawdown,
                    "threshold": -0.1,
                    "timestamp": datetime.now()
                })
            
        except Exception as e:
            logging.error(f"戦略アラートチェックエラー [{strategy_name}]: {e}")
        
        return alerts
    
    def _process_alert(self, alert: Dict):
        """アラート処理"""
        try:
            alert_key = f"{alert.get('type')}_{alert.get('strategy', 'portfolio')}"
            
            # 重複アラート抑制
            if alert_key in self.active_alerts:
                last_alert_time = self.active_alerts[alert_key]
                if datetime.now() - last_alert_time < timedelta(minutes=30):
                    return  # 30分以内の重複アラートは無視
            
            self.active_alerts[alert_key] = datetime.now()
            
            # アラート通知実装
            logging.warning(f"ALERT: {alert.get('message')}")
            
        except Exception as e:
            logging.error(f"アラート処理エラー: {e}")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="パフォーマンス監視スクリプト")
    parser.add_argument("--config", default="config/performance_monitoring/monitoring_config.json", 
                       help="設定ファイルパス")
    parser.add_argument("--interval", type=int, default=900, 
                       help="監視間隔（秒）")
    parser.add_argument("--daemon", action="store_true", 
                       help="デーモンモードで実行")
    
    args = parser.parse_args()
    
    # 監視システム初期化
    monitor = PerformanceMonitor(config_path=args.config)
    
    # 間隔設定上書き
    if args.interval != 900:
        monitor.config['monitoring_settings']['update_interval_seconds'] = args.interval
    
    print(f"🚀 パフォーマンス監視開始")
    print(f"   設定ファイル: {args.config}")
    print(f"   監視間隔: {args.interval}秒")
    print(f"   デーモンモード: {'有効' if args.daemon else '無効'}")
    print("   Ctrl+C で停止")
    
    try:
        # 非同期監視開始
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        print("\n👋 監視を停止しています...")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"❌ 監視エラー: {e}")
        monitor.stop_monitoring()
        sys.exit(1)


if __name__ == "__main__":
    main()
    """パフォーマンスメトリクス"""
    execution_time: float
    peak_memory_mb: float
    avg_cpu_percent: float
    memory_growth_mb: float
    gc_collections: int
    start_memory_mb: float
    end_memory_mb: float
    thread_count: int
    process_count: int

class PerformanceMonitor:
    """パフォーマンス監視システム"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory = 0.0
        self.cpu_samples: List[float] = []
        self.gc_start_count = 0
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self) -> str:
        """監視開始"""
        if self.monitoring:
            return "already_running"
            
        self.monitoring = True
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage_mb()
        self.peak_memory = self.start_memory
        self.cpu_samples = []
        self.gc_start_count = len(gc.get_stats())
        
        # 監視スレッド開始
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        monitor_id = f"monitor_{int(time.time())}"
        self.logger.info(f"Performance monitoring started: {monitor_id}")
        return monitor_id
        
    def stop_monitoring(self) -> PerformanceMetrics:
        """監視終了とメトリクス取得"""
        if not self.monitoring:
            raise RuntimeError("Monitoring not started")
            
        self.monitoring = False
        
        # スレッド終了を待機
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        end_time = time.time()
        end_memory = self._get_memory_usage_mb()
        gc_end_count = len(gc.get_stats())
        
        # None チェック
        start_time = self.start_time or 0.0
        start_memory = self.start_memory or 0.0
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            peak_memory_mb=self.peak_memory,
            avg_cpu_percent=sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0,
            memory_growth_mb=end_memory - start_memory,
            gc_collections=gc_end_count - self.gc_start_count,
            start_memory_mb=start_memory,
            end_memory_mb=end_memory,
            thread_count=threading.active_count(),
            process_count=self._get_process_count()
        )
        
        self.metrics_history.append(metrics)
        self.logger.info(f"Performance monitoring stopped. Execution time: {metrics.execution_time:.2f}s")
        return metrics
        
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                # メモリ使用量更新
                current_memory = self._get_memory_usage_mb()
                self.peak_memory = max(self.peak_memory, current_memory)
                
                # CPU使用率サンプリング（psutil利用可能時のみ）
                if PSUTIL_AVAILABLE and 'psutil' in globals():
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.cpu_samples.append(cpu_percent)
                
                # サンプリング間隔
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                break
                
    def _get_memory_usage_mb(self) -> float:
        """メモリ使用量取得（MB）"""
        try:
            if PSUTIL_AVAILABLE and 'psutil' in globals():
                process = psutil.Process()
                memory_info = process.memory_info()
                return memory_info.rss / 1024 / 1024  # MB変換
            else:
                # フォールバック（大まかな推定）
                import tracemalloc
                if tracemalloc.is_tracing():
                    _, peak = tracemalloc.get_traced_memory()
                    return peak / 1024 / 1024
                return 0.0
        except Exception:
            return 0.0
            
    def _get_process_count(self) -> int:
        """プロセス数取得"""
        try:
            if PSUTIL_AVAILABLE and 'psutil' in globals():
                return len(psutil.pids())
            else:
                return 1  # フォールバック
        except Exception:
            return 1
            
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        try:
            if PSUTIL_AVAILABLE and 'psutil' in globals():
                cpu_count = psutil.cpu_count()
                memory = psutil.virtual_memory()
                # Windowsでは '/' の代わりに 'C:\' を使用
                disk_path = 'C:\\' if os.name == 'nt' else '/'
                disk = psutil.disk_usage(disk_path)
                
                return {
                    "cpu_count": cpu_count,
                    "total_memory_gb": memory.total / (1024**3),
                    "available_memory_gb": memory.available / (1024**3),
                    "memory_percent": memory.percent,
                    "disk_total_gb": disk.total / (1024**3),
                    "disk_free_gb": disk.free / (1024**3),
                    "python_version": sys.version,
                    "platform": os.name
                }
            else:
                # psutil未利用時のフォールバック
                return {
                    "cpu_count": "unknown",
                    "total_memory_gb": "unknown",
                    "available_memory_gb": "unknown", 
                    "memory_percent": "unknown",
                    "disk_total_gb": "unknown",
                    "disk_free_gb": "unknown",
                    "python_version": sys.version,
                    "platform": os.name,
                    "note": "psutil not available - limited system info"
                }
        except Exception as e:
            self.logger.error(f"System info error: {e}")
            return {"error": str(e)}
            
    def export_metrics(self, filepath: str):
        """メトリクス履歴をJSONエクスポート"""
        try:
            export_data = {
                "system_info": self.get_system_info(),
                "metrics_history": [
                    {
                        "execution_time": m.execution_time,
                        "peak_memory_mb": m.peak_memory_mb,
                        "avg_cpu_percent": m.avg_cpu_percent,
                        "memory_growth_mb": m.memory_growth_mb,
                        "gc_collections": m.gc_collections,
                        "start_memory_mb": m.start_memory_mb,
                        "end_memory_mb": m.end_memory_mb,
                        "thread_count": m.thread_count,
                        "process_count": m.process_count
                    }
                    for m in self.metrics_history
                ],
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Metrics exported to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            
    def clear_history(self):
        """履歴クリア"""
        self.metrics_history.clear()
        self.logger.info("Metrics history cleared")

# テスト用の簡単な実行例
if __name__ == "__main__":
    import time
    
    # ロギング設定
    logging.basicConfig(level=logging.INFO)
    
    monitor = PerformanceMonitor()
    
    # 簡単な負荷テスト
    monitor_id = monitor.start_monitoring()
    
    # 重い処理をシミュレート
    data = [i**2 for i in range(100000)]
    time.sleep(2)
    
    metrics = monitor.stop_monitoring()
    
    print(f"実行時間: {metrics.execution_time:.2f}秒")
    print(f"ピークメモリ: {metrics.peak_memory_mb:.2f}MB")
    print(f"平均CPU使用率: {metrics.avg_cpu_percent:.2f}%")
    print(f"メモリ増加: {metrics.memory_growth_mb:.2f}MB")
    
    # システム情報表示
    system_info = monitor.get_system_info()
    print("\nシステム情報:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
