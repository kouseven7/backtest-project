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
    """統合パフォーマンス監視システム"""
    
    def __init__(self, config_path: str = "config/performance_monitoring/monitoring_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # ログ設定
        if setup_logger:
            self.logger = setup_logger("performance_monitor", logging.INFO, "logs/performance_monitor.log")
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
    
    def _load_config(self) -> dict:
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
    
    def _get_default_config(self) -> dict:
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
                "file_output": {"enabled": True, "output_directory": "logs/performance_monitoring"}
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
            # パフォーマンスアラート管理
            self.performance_alert_manager = PerformanceAlertManager()
            
            # ポートフォリオ分析器
            self.portfolio_analyzer = PortfolioPerformanceAnalyzer(self.config)
            
            self.logger.info("基本コンポーネント初期化完了")
            
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
            # 監視タスク開始
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.monitoring_tasks = [monitoring_task]
            
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
                
                execution_time = time.time() - start_time
                self.logger.info(f"監視サイクル完了: {execution_time:.2f}秒")
                
                # 次回実行まで待機
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                await asyncio.sleep(60)  # エラー時は1分待機
    
    async def _analyze_current_performance(self) -> dict:
        """現在のパフォーマンス分析"""
        try:
            # サンプル戦略データ生成（テスト用）
            strategy_performances = self._generate_sample_strategy_data()
            
            # ポートフォリオレベル分析
            portfolio_analysis = self.portfolio_analyzer.analyze_portfolio_performance(strategy_performances)
            
            # 履歴に記録
            self.performance_history[datetime.now().isoformat()] = portfolio_analysis
            
            return portfolio_analysis
            
        except Exception as e:
            self.logger.error(f"パフォーマンス分析エラー: {e}")
            return {}
    
    def _generate_sample_strategy_data(self) -> dict:
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
        
        return strategy_data
    
    async def _output_results(self, portfolio_analysis: dict, alerts: list):
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
    
    def _output_to_console(self, portfolio_analysis: dict, alerts: list):
        """コンソール出力"""
        try:
            print(f"\n=== パフォーマンス監視レポート [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
            
            # ポートフォリオサマリー
            portfolio_metrics = portfolio_analysis.get('portfolio_metrics', {})
            if portfolio_metrics:
                print(f"ポートフォリオサマリー:")
                print(f"  総PnL: {portfolio_metrics.get('total_pnl', 0):.2f}")
                print(f"  シャープレシオ: {portfolio_metrics.get('sharpe_ratio', 0):.3f}")
                print(f"  最大ドローダウン: {portfolio_metrics.get('max_drawdown', 0):.2%}")
                print(f"  勝率: {portfolio_metrics.get('win_rate', 0):.1%}")
            
            # 戦略別パフォーマンス
            strategy_performances = portfolio_analysis.get('strategy_performances', {})
            if strategy_performances:
                print(f"\n戦略別パフォーマンス:")
                for strategy_name, performance in strategy_performances.items():
                    basic_metrics = performance.get('basic_metrics', {})
                    print(f"  {strategy_name}:")
                    print(f"    取引数: {basic_metrics.get('total_trades', 0)}")
                    print(f"    勝率: {basic_metrics.get('win_rate', 0):.1%}")
                    print(f"    PnL: {basic_metrics.get('total_pnl', 0):.2f}")
            
            # アラート表示
            if alerts:
                print(f"\nアラート ({len(alerts)}件):")
                for alert in alerts:
                    severity_map = {"high": "[高]", "medium": "[中]", "low": "[低]"}
                    severity_label = severity_map.get(alert.get('severity', 'low'), "[?]")
                    print(f"  {severity_label} {alert.get('message', 'Unknown alert')}")
            else:
                print(f"\nアラートなし")
            
            print("=" * 80)
            
        except Exception as e:
            self.logger.error(f"コンソール出力エラー: {e}")
    
    async def _output_to_files(self, portfolio_analysis: dict, alerts: list):
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
            
        except Exception as e:
            self.logger.error(f"ファイル出力エラー: {e}")
    
    def stop_monitoring(self):
        """監視停止"""
        self.logger.info("監視停止中...")
        self.is_running = False
        
        # タスクキャンセル
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        self.logger.info("監視停止完了")
    
    def get_status(self) -> dict:
        """監視状態取得"""
        return {
            "is_running": self.is_running,
            "strategy_count": len(self.strategy_trackers),
            "history_count": len(self.performance_history),
            "last_update": max(self.performance_history.keys()) if self.performance_history else None
        }


class PortfolioPerformanceAnalyzer:
    """ポートフォリオレベル統合パフォーマンス分析"""
    
    def __init__(self, config: dict):
        self.config = config
        self.portfolio_history = []
    
    def analyze_portfolio_performance(self, strategy_performances: dict) -> dict:
        """ポートフォリオ統合分析"""
        try:
            if not strategy_performances:
                return {}
            
            # ポートフォリオレベル計算
            portfolio_metrics = self._calculate_portfolio_metrics(strategy_performances)
            
            return {
                "timestamp": datetime.now(),
                "portfolio_metrics": portfolio_metrics,
                "strategy_performances": strategy_performances,
                "portfolio_health_score": self._calculate_portfolio_health_score(portfolio_metrics)
            }
            
        except Exception as e:
            logging.error(f"ポートフォリオ分析エラー: {e}")
            return {}
    
    def _calculate_portfolio_metrics(self, strategy_performances: dict) -> dict:
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
    
    def _calculate_portfolio_health_score(self, portfolio_metrics: dict) -> float:
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
    
    def _load_alert_rules(self) -> dict:
        """アラートルール読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._get_default_alert_rules()
        except:
            return self._get_default_alert_rules()
    
    def _get_default_alert_rules(self) -> dict:
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
    
    def check_performance_alerts(self, portfolio_analysis: dict) -> list:
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
            
            return alerts
            
        except Exception as e:
            logging.error(f"アラートチェックエラー: {e}")
            return []
    
    def _check_portfolio_alerts(self, portfolio_metrics: dict) -> list:
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
            
        except Exception as e:
            logging.error(f"ポートフォリオアラートチェックエラー: {e}")
        
        return alerts
    
    def _check_strategy_alerts(self, strategy_name: str, performance: dict) -> list:
        """戦略別アラートチェック"""
        alerts = []
        rules = self.alert_rules.get('strategy_rules', {})
        
        try:
            basic_metrics = performance.get('basic_metrics', {})
            
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
            
        except Exception as e:
            logging.error(f"戦略アラートチェックエラー [{strategy_name}]: {e}")
        
        return alerts


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
    
    print(f"パフォーマンス監視開始")
    print(f"   設定ファイル: {args.config}")
    print(f"   監視間隔: {args.interval}秒")
    print(f"   デーモンモード: {'有効' if args.daemon else '無効'}")
    print("   Ctrl+C で停止")
    
    try:
        # 非同期監視開始
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        print("\n監視を停止しています...")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"監視エラー: {e}")
        monitor.stop_monitoring()
        sys.exit(1)


if __name__ == "__main__":
    main()
