"""
ペーパートレード実行スクリプト - フェーズ4A1
ハイブリッド型設計：シンプル戦略実行 + オプション統合機能

Usage:
    python paper_trade_runner.py --mode simple --strategy VWAP_Breakout
    python paper_trade_runner.py --mode integrated --interval 15
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import time
import signal
from typing import Dict, List, Any
import json

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 既存システムインポート
from config.logger_config import setup_logger
from src.execution.paper_trade_scheduler import PaperTradeScheduler
from src.execution.paper_trade_monitor import PaperTradeMonitor
from src.execution.strategy_execution_manager import StrategyExecutionManager

class PaperTradeRunner:
    """ペーパートレード実行メインクラス"""
    
    def __init__(self, config_path: str = "config/paper_trading/runner_config.json"):
        self.config_path = config_path
        self.logger = setup_logger("PaperTradeRunner", log_file="logs/paper_trade_runner.log")
        
        # コンポーネント初期化
        self.scheduler = None
        self.monitor = None
        self.strategy_manager = None
        self.running = False
        self.config = {}
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self, args: argparse.Namespace) -> bool:
        """システム初期化"""
        try:
            self.logger.info("=== ペーパートレードシステム初期化開始 ===")
            
            # 設定読み込み
            self.config = self._load_configuration(args)
            
            # コンポーネント初期化
            self.scheduler = PaperTradeScheduler(self.config.get('scheduler', {}))
            self.monitor = PaperTradeMonitor(self.config.get('monitor', {}))
            self.strategy_manager = StrategyExecutionManager(self.config.get('strategy', {}))
            
            # 初期化検証
            if not self._validate_initialization():
                return False
            
            self.logger.info("システム初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"初期化エラー: {e}")
            return False
    
    def _load_configuration(self, args: argparse.Namespace) -> Dict[str, Any]:
        """設定読み込み"""
        try:
            config_file = Path(args.config if hasattr(args, 'config') else self.config_path)
            
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"設定ファイル読み込み: {config_file}")
                return config
            else:
                self.logger.warning(f"設定ファイル不存在: {config_file}、デフォルト設定使用")
                return self._get_default_config()
                
        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定取得"""
        return {
            "scheduler": {"default_interval_minutes": 15, "market_hours_only": True},
            "monitor": {"performance_window_hours": 24, "alert_thresholds": {}},
            "strategy": {
                "execution_mode": "simple",
                "default_symbols": ["AAPL", "MSFT"],
                "lookback_periods": 100,
                "position_value": 10000
            }
        }
    
    def _validate_initialization(self) -> bool:
        """初期化検証"""
        if self.scheduler is None:
            self.logger.error("スケジューラー初期化失敗")
            return False
        
        if self.monitor is None:
            self.logger.error("モニター初期化失敗")
            return False
        
        if self.strategy_manager is None:
            self.logger.error("戦略管理初期化失敗")
            return False
        
        return True
    
    def run(self, args: argparse.Namespace) -> None:
        """メイン実行ループ"""
        try:
            if not self.initialize(args):
                self.logger.error("初期化失敗。終了します。")
                return
            
            self.running = True
            self.logger.info("=== ペーパートレード実行開始 ===")
            
            # ドライランモードチェック
            if hasattr(args, 'dry_run') and args.dry_run:
                self.logger.info("ドライランモード - 実際の取引は行いません")
                self._run_dry_mode(args)
                return
            
            if args.mode == "simple":
                self._run_simple_mode(args)
            elif args.mode == "integrated":
                self._run_integrated_mode(args)
            else:
                self.logger.error(f"未知の実行モード: {args.mode}")
                
        except Exception as e:
            self.logger.error(f"実行エラー: {e}")
        finally:
            self._cleanup()
    
    def _run_dry_mode(self, args: argparse.Namespace) -> None:
        """ドライランモード実行"""
        self.logger.info("ドライランモード開始")
        
        # システム状態チェック
        self.logger.info(f"スケジューラー状態: {self.scheduler.get_status()}")
        self.logger.info(f"モニター状態: {self.monitor.get_status()}")
        self.logger.info(f"戦略管理状態: {self.strategy_manager.get_execution_summary()}")
        
        # サンプル実行テスト
        try:
            if args.mode == "simple":
                result = self.strategy_manager.execute_strategy(args.strategy)
                self.monitor.record_execution(result)
                self.logger.info(f"サンプル戦略実行結果: {result.get('success', False)}")
            
            self.logger.info("ドライランモード完了")
        except Exception as e:
            self.logger.error(f"ドライランモードエラー: {e}")
    
    def _run_simple_mode(self, args: argparse.Namespace) -> None:
        """シンプルモード実行"""
        self.logger.info(f"シンプルモード開始 - 戦略: {args.strategy}")
        
        execution_count = 0
        max_executions = 100  # 安全制限
        
        while self.running and execution_count < max_executions:
            try:
                # スケジューラーチェック
                if self.scheduler.should_execute():
                    # 戦略実行
                    result = self.strategy_manager.execute_strategy(args.strategy)
                    
                    # 監視・ログ
                    self.monitor.record_execution(result)
                    
                    # 次回実行時刻設定
                    self.scheduler.schedule_next(args.interval)
                    
                    execution_count += 1
                    self.logger.info(f"実行完了 {execution_count}/{max_executions}")
                
                # 短時間待機
                time.sleep(10)  # 10秒間隔でチェック
                
            except Exception as e:
                self.logger.error(f"シンプル実行エラー: {e}")
                time.sleep(60)  # エラー時は1分待機
    
    def _run_integrated_mode(self, args: argparse.Namespace) -> None:
        """統合モード実行"""
        self.logger.info("統合モード開始")
        
        execution_count = 0
        max_executions = 50  # 統合モードは処理が重いため制限を小さく
        
        while self.running and execution_count < max_executions:
            try:
                if self.scheduler.should_execute():
                    # 統合戦略実行
                    result = self.strategy_manager.execute_integrated_strategies()
                    
                    # 監視・ログ
                    self.monitor.record_execution(result)
                    
                    # ポートフォリオ状態更新
                    self.monitor.update_portfolio_status()
                    
                    self.scheduler.schedule_next(args.interval)
                    
                    execution_count += 1
                    self.logger.info(f"統合実行完了 {execution_count}/{max_executions}")
                
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"統合実行エラー: {e}")
                time.sleep(60)
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー（Ctrl+C等）"""
        self.logger.info(f"シグナル {signum} を受信。安全に終了します...")
        self.running = False
    
    def _cleanup(self):
        """終了処理"""
        self.logger.info("=== ペーパートレード終了処理 ===")
        
        try:
            if self.monitor:
                self.monitor.generate_final_report()
            
            if self.strategy_manager:
                self.strategy_manager.cleanup()
            
            self.logger.info("終了処理完了")
        except Exception as e:
            self.logger.error(f"終了処理エラー: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ペーパートレード実行システム")
    parser.add_argument("--mode", choices=["simple", "integrated"], 
                       default="simple", help="実行モード")
    parser.add_argument("--strategy", default="VWAP_Breakout", 
                       help="実行戦略（シンプルモード用）")
    parser.add_argument("--interval", type=int, default=15, 
                       help="実行間隔（分）")
    parser.add_argument("--config", default="config/paper_trading/runner_config.json",
                       help="設定ファイルパス")
    parser.add_argument("--dry-run", action="store_true",
                       help="ドライランモード（実際の取引なし）")
    
    args = parser.parse_args()
    
    # ログディレクトリ作成
    Path("logs").mkdir(exist_ok=True)
    
    # 実行
    runner = PaperTradeRunner(args.config)
    runner.run(args)

if __name__ == "__main__":
    main()
