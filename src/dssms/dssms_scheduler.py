"""
DSSMS Phase 4 Task 4.2: DSSMS実行スケジューラー
前場後場スケジューリングシステム

ハイブリッドスケジューリング・kabu API統合・緊急切替対応
"""

import sys
import time
import schedule
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, date
import threading
import signal

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# DSSMS既存コンポーネントのインポート
from .market_time_manager import MarketTimeManager
from .emergency_detector import EmergencyDetector
from .execution_history import ExecutionHistory
from .kabu_integration_manager import KabuIntegrationManager, DSSMSKabuIntegrator

# 既存DSSMSコンポーネント
try:
    from .nikkei225_screener import Nikkei225Screener
    from .hierarchical_ranking_system import HierarchicalRankingSystem
    from .intelligent_switch_manager import IntelligentSwitchManager
    from .market_condition_monitor import MarketConditionMonitor
except ImportError as e:
    print(f"Warning: DSSMS components import error: {e}")

from config.logger_config import setup_logger

class DSSMSScheduler:
    """DSSMS前場後場スケジューリングシステム"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス（None時はデフォルト使用）
        """
        self.logger = setup_logger('dssms.scheduler')
        
        # スケジューリング制御
        self.market_time_manager = MarketTimeManager(config_path)
        self.emergency_detector = EmergencyDetector(config_path)
        self.execution_history = ExecutionHistory()
        
        # kabu API統合（段階的連携）
        try:
            self.kabu_integration = KabuIntegrationManager(config_path)
            self.kabu_integrator = DSSMSKabuIntegrator(config_path)
            self.integration_mode = 'phase4_only'  # 初期は独立
            self.logger.info("kabu API統合システム初期化成功")
        except Exception as e:
            self.logger.warning(f"kabu API統合システム初期化エラー: {e}")
            self.kabu_integration = None
            self.kabu_integrator = None
            self.integration_mode = 'disabled'
        
        # DSSMS コアエンジン統合
        try:
            self.nikkei225_screener = Nikkei225Screener()
            self.hierarchical_ranking = HierarchicalRankingSystem({'ranking_system': {}})
            self.intelligent_switch = IntelligentSwitchManager()
            self.market_monitor = MarketConditionMonitor()
            self.logger.info("DSSMSコアエンジン初期化成功")
        except Exception as e:
            self.logger.warning(f"DSSMSコアエンジン初期化エラー: {e}")
            self.nikkei225_screener = None
            self.hierarchical_ranking = None
            self.intelligent_switch = None
            self.market_monitor = None
        
        # スケジューラー状態
        self.is_running = False
        self.current_monitoring_symbol: Optional[str] = None
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # スケジュール設定
        self._setup_schedule()
        
        self.logger.info("DSSMSScheduler: 初期化完了")
    
    def _setup_schedule(self):
        """スケジュール設定"""
        try:
            # 前場スクリーニング (09:30)
            schedule.every().monday.at("09:30").do(self._run_morning_screening_job)
            schedule.every().tuesday.at("09:30").do(self._run_morning_screening_job)
            schedule.every().wednesday.at("09:30").do(self._run_morning_screening_job)
            schedule.every().thursday.at("09:30").do(self._run_morning_screening_job)
            schedule.every().friday.at("09:30").do(self._run_morning_screening_job)
            
            # 後場スクリーニング (12:30)
            schedule.every().monday.at("12:30").do(self._run_afternoon_screening_job)
            schedule.every().tuesday.at("12:30").do(self._run_afternoon_screening_job)
            schedule.every().wednesday.at("12:30").do(self._run_afternoon_screening_job)
            schedule.every().thursday.at("12:30").do(self._run_afternoon_screening_job)
            schedule.every().friday.at("12:30").do(self._run_afternoon_screening_job)
            
            # 緊急チェック（1分毎）
            schedule.every().minute.do(self._emergency_check_job)
            
            self.logger.info("スケジュール設定完了")
            
        except Exception as e:
            self.logger.error(f"スケジュール設定エラー: {e}")
    
    def run_morning_screening(self) -> str:
        """
        09:30前場スクリーニング実行
        
        Returns:
            str: 選択された銘柄コード（エラー時は空文字）
        """
        start_time = datetime.now()
        self.logger.info("=== 前場スクリーニング開始 ===")
        
        try:
            # 市場時間チェック
            if not self.market_time_manager.should_run_screening("morning"):
                self.logger.warning("前場スクリーニング実行条件未満足")
                return ""
            
            # スクリーニング実行
            screening_result = self._execute_screening("morning")
            
            # kabu API連携
            if self.kabu_integrator and screening_result.get("selected_symbol"):
                try:
                    sync_success = self.kabu_integrator.sync_screening_results_to_kabu(10000000.0)
                    self.logger.info(f"kabu API同期: {'成功' if sync_success else '失敗'}")
                    # BUYモック注文
                    symbol = screening_result["selected_symbol"]
                    order_result = self.kabu_integrator.kabu_manager.execute_dynamic_orders({
                        'symbol': symbol,
                        'side': '2',
                        'quantity': 100,
                        'price': 0
                    })
                    self.logger.info(f"前場BUY注文: {symbol} 結果={order_result}")
                except Exception as e:
                    self.logger.error(f"kabu API同期エラー: {e}")
            
            # 実行履歴記録
            duration = (datetime.now() - start_time).total_seconds()
            screening_result["duration"] = duration
            
            self.execution_history.record_screening_execution("morning", screening_result)
            
            selected_symbol = screening_result.get("selected_symbol", "")
            
            # 選択銘柄の監視開始
            if selected_symbol:
                self.start_selected_symbol_monitoring(selected_symbol)
            
            self.logger.info(f"=== 前場スクリーニング完了: {selected_symbol} ({duration:.2f}秒) ===")
            return selected_symbol
            
        except Exception as e:
            self.logger.error(f"前場スクリーニングエラー: {e}")
            # エラー記録
            error_result = {
                "status": "error",
                "error": str(e),
                "duration": (datetime.now() - start_time).total_seconds()
            }
            self.execution_history.record_screening_execution("morning", error_result)
            return ""
    
    def run_afternoon_screening(self) -> str:
        """
        12:30後場スクリーニング実行
        
        Returns:
            str: 選択された銘柄コード（エラー時は空文字）
        """
        start_time = datetime.now()
        self.logger.info("=== 後場スクリーニング開始 ===")
        
        try:
            # 市場時間チェック
            if not self.market_time_manager.should_run_screening("afternoon"):
                self.logger.warning("後場スクリーニング実行条件未満足")
                return ""
            
            # スクリーニング実行
            screening_result = self._execute_screening("afternoon")
            
            # kabu API連携
            if self.kabu_integrator and screening_result.get("selected_symbol"):
                try:
                    sync_success = self.kabu_integrator.sync_screening_results_to_kabu(10000000.0)
                    self.logger.info(f"kabu API同期: {'成功' if sync_success else '失敗'}")
                    # BUYモック注文
                    symbol = screening_result["selected_symbol"]
                    order_result = self.kabu_integrator.kabu_manager.execute_dynamic_orders({
                        'symbol': symbol,
                        'side': '2',
                        'quantity': 100,
                        'price': 0
                    })
                    self.logger.info(f"後場BUY注文: {symbol} 結果={order_result}")
                except Exception as e:
                    self.logger.error(f"kabu API同期エラー: {e}")
            
            # 実行履歴記録
            duration = (datetime.now() - start_time).total_seconds()
            screening_result["duration"] = duration
            
            self.execution_history.record_screening_execution("afternoon", screening_result)
            
            selected_symbol = screening_result.get("selected_symbol", "")
            
            # 選択銘柄の監視開始
            if selected_symbol:
                self.start_selected_symbol_monitoring(selected_symbol)
            
            self.logger.info(f"=== 後場スクリーニング完了: {selected_symbol} ({duration:.2f}秒) ===")
            return selected_symbol
            
        except Exception as e:
            self.logger.error(f"後場スクリーニングエラー: {e}")
            # エラー記録
            error_result = {
                "status": "error",
                "error": str(e),
                "duration": (datetime.now() - start_time).total_seconds()
            }
            self.execution_history.record_screening_execution("afternoon", error_result)
            return ""
    
    def start_selected_symbol_monitoring(self, symbol: str) -> None:
        """
        選択銘柄のリアルタイム監視開始
        
        Args:
            symbol: 監視対象銘柄コード
        """
        try:
            self.current_monitoring_symbol = symbol
            
            # kabu API監視登録
            if self.kabu_integration:
                try:
                    registration_success = self.kabu_integration.register_screening_symbols([symbol])
                    self.logger.info(f"kabu API監視登録: {symbol} - {'成功' if registration_success else '失敗'}")
                except Exception as e:
                    self.logger.error(f"kabu API監視登録エラー {symbol}: {e}")
            
            # 監視イベント記録
            monitoring_data = {
                "event_type": "monitoring_start",
                "symbol": symbol,
                "start_time": datetime.now().isoformat()
            }
            
            self.execution_history.record_monitoring_event(symbol, monitoring_data)
            
            self.logger.info(f"銘柄監視開始: {symbol}")
            
        except Exception as e:
            self.logger.error(f"銘柄監視開始エラー {symbol}: {e}")
    
    def handle_emergency_switch_check(self) -> None:
        """パーフェクトオーダー崩れ時の緊急判定処理"""
        try:
            # 現在監視中の銘柄がない場合はスキップ
            if not self.current_monitoring_symbol:
                return
            
            # 市場開場中のみ実行
            if not self.market_time_manager.is_market_open():
                return
            
            symbol = self.current_monitoring_symbol
            
            # 緊急事態判定
            emergency_result = self.emergency_detector.check_emergency_conditions(symbol)
            
            # 緊急判定記録
            self.execution_history.record_emergency_check(symbol, emergency_result)
            
            # 緊急事態の場合の処理
            if emergency_result.get("is_emergency"):
                self.logger.warning(f"緊急事態検出: {symbol} - Level {emergency_result.get('emergency_level')}")
                
                # 推奨アクションに基づく処理
                recommended_action = emergency_result.get("recommended_action")
                
                if recommended_action == "immediate_switch":
                    self._execute_emergency_switch(symbol, emergency_result)
                elif recommended_action == "prepare_switch":
                    self._prepare_emergency_switch(symbol, emergency_result)
                elif recommended_action == "close_monitoring":
                    self.logger.info(f"詳細監視開始: {symbol}")
            
        except Exception as e:
            self.logger.error(f"緊急切替チェックエラー: {e}")
    
    def _execute_screening(self, session: str) -> Dict[str, Any]:
        """スクリーニング実行"""
        try:
            result = {
                "status": "success",
                "session": session,
                "selected_symbol": None,
                "candidate_count": 0,
                "priority_distribution": {}
            }
            
            # Nikkei225スクリーニング
            if self.nikkei225_screener:
                try:
                    candidates = self.nikkei225_screener.get_filtered_symbols(
                        available_funds=10000000.0,
                        target_date=date.today()
                    )
                    result["candidate_count"] = len(candidates)
                    self.logger.info(f"スクリーニング候補取得: {len(candidates)}銘柄")
                except Exception as e:
                    self.logger.error(f"スクリーニング候補取得エラー: {e}")
                    candidates = []
            else:
                # フォールバック候補
                candidates = ["6758", "9433", "9984", "5401", "6645"]
                result["candidate_count"] = len(candidates)
            
            # 階層的ランキング
            if self.hierarchical_ranking and candidates:
                try:
                    # 上位候補から選択
                    top_candidates = self.hierarchical_ranking.get_backup_candidates(n=5)
                    if top_candidates:
                        result["selected_symbol"] = top_candidates[0]
                        result["priority_distribution"] = {"level1": 1, "level2": 2, "level3": 2}
                except Exception as e:
                    self.logger.error(f"階層ランキングエラー: {e}")
                    result["selected_symbol"] = candidates[0] if candidates else None
            else:
                # フォールバック選択
                result["selected_symbol"] = candidates[0] if candidates else None
            
            return result
            
        except Exception as e:
            self.logger.error(f"スクリーニング実行エラー {session}: {e}")
            return {
                "status": "error",
                "session": session,
                "error": str(e)
            }
    
    def _execute_emergency_switch(self, symbol: str, emergency_result: Dict[str, Any]) -> None:
        """緊急切替実行"""
        try:
            self.logger.warning(f"緊急切替実行: {symbol}")
            
            # インテリジェント切替実行
            if self.intelligent_switch:
                try:
                    # 切替候補取得
                    backup_candidates = self.hierarchical_ranking.get_backup_candidates(n=3) if self.hierarchical_ranking else ["6758", "9433", "9984"]
                    
                    if backup_candidates:
                        target_symbol = backup_candidates[0]
                        
                        # 切替実行
                        switch_success = self.intelligent_switch.execute_switch_with_risk_control(symbol, target_symbol)
                        
                        if switch_success:
                            self.current_monitoring_symbol = target_symbol
                            self.logger.info(f"緊急切替成功: {symbol} → {target_symbol}")

                            # SELL→BUYモック注文
                            if self.kabu_integration:
                                try:
                                    # 旧銘柄SELL
                                    sell_result = self.kabu_integration.execute_dynamic_orders({
                                        'symbol': symbol,
                                        'side': '1',
                                        'quantity': 100,
                                        'price': 0
                                    })
                                    self.logger.info(f"緊急SELL注文: {symbol} 結果={sell_result}")
                                    # 新銘柄BUY
                                    buy_result = self.kabu_integration.execute_dynamic_orders({
                                        'symbol': target_symbol,
                                        'side': '2',
                                        'quantity': 100,
                                        'price': 0
                                    })
                                    self.logger.info(f"緊急BUY注文: {target_symbol} 結果={buy_result}")
                                except Exception as e:
                                    self.logger.error(f"緊急注文エラー: {e}")

                            # 切替記録
                            switch_data = {
                                "from_symbol": symbol,
                                "to_symbol": target_symbol,
                                "reason": "emergency_switch",
                                "emergency_level": emergency_result.get("emergency_level"),
                                "conditions": emergency_result.get("trigger_conditions", []),
                                "execution_result": {"success": True}
                            }
                            self.execution_history.record_symbol_switch(switch_data)
                        else:
                            self.logger.error(f"緊急切替失敗: {symbol} → {target_symbol}")
                except Exception as e:
                    self.logger.error(f"インテリジェント切替エラー: {e}")
            
        except Exception as e:
            self.logger.error(f"緊急切替実行エラー: {e}")
    
    def _prepare_emergency_switch(self, symbol: str, emergency_result: Dict[str, Any]) -> None:
        """緊急切替準備"""
        try:
            self.logger.info(f"緊急切替準備: {symbol}")
            
            # 切替候補の事前準備
            if self.hierarchical_ranking:
                backup_candidates = self.hierarchical_ranking.get_backup_candidates(n=5)
                self.logger.info(f"切替候補準備完了: {backup_candidates}")
            
        except Exception as e:
            self.logger.error(f"緊急切替準備エラー: {e}")
    
    def _run_morning_screening_job(self):
        """前場スクリーニングジョブ（内部）"""
        try:
            result = self.run_morning_screening()
            self.logger.info(f"前場スクリーニングジョブ完了: {result}")
        except Exception as e:
            self.logger.error(f"前場スクリーニングジョブエラー: {e}")
    
    def _run_afternoon_screening_job(self):
        """後場スクリーニングジョブ（内部）"""
        try:
            result = self.run_afternoon_screening()
            self.logger.info(f"後場スクリーニングジョブ完了: {result}")
        except Exception as e:
            self.logger.error(f"後場スクリーニングジョブエラー: {e}")
    
    def _emergency_check_job(self):
        """緊急チェックジョブ（内部）"""
        try:
            self.handle_emergency_switch_check()
        except Exception as e:
            self.logger.error(f"緊急チェックジョブエラー: {e}")
    
    def start_scheduler(self):
        """スケジューラー開始"""
        if self.is_running:
            self.logger.warning("スケジューラーは既に実行中です")
            return
        
        self.is_running = True
        self.logger.info("DSSMSスケジューラー開始")
        
        def scheduler_loop():
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(30)  # 30秒間隔でチェック
                except Exception as e:
                    self.logger.error(f"スケジューラーループエラー: {e}")
                    time.sleep(60)  # エラー時は1分待機
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("DSSMSスケジューラースレッド開始")
    
    def stop_scheduler(self):
        """スケジューラー停止"""
        if not self.is_running:
            self.logger.warning("スケジューラーは実行中ではありません")
            return
        
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("DSSMSスケジューラー停止")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """スケジューラー状況取得"""
        try:
            return {
                "is_running": self.is_running,
                "current_monitoring_symbol": self.current_monitoring_symbol,
                "current_session": self.market_time_manager.get_current_session(),
                "next_screening_time": self.market_time_manager.get_next_screening_time(),
                "market_open": self.market_time_manager.is_market_open(),
                "integration_mode": self.integration_mode,
                "kabu_api_available": self.kabu_integration is not None,
                "dssms_components_available": {
                    "screener": self.nikkei225_screener is not None,
                    "ranking": self.hierarchical_ranking is not None,
                    "switch_manager": self.intelligent_switch is not None,
                    "market_monitor": self.market_monitor is not None
                }
            }
        except Exception as e:
            self.logger.error(f"スケジューラー状況取得エラー: {e}")
            return {"error": str(e)}


def main():
    """メイン実行関数"""
    scheduler = DSSMSScheduler()
    
    # シグナルハンドラー設定
    def signal_handler(signum, frame):
        print("\nシャットダウン信号を受信しました...")
        scheduler.stop_scheduler()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # スケジューラー開始
        scheduler.start_scheduler()
        
        # 状況表示
        status = scheduler.get_scheduler_status()
        print(f"DSSMSスケジューラー状況: {status}")
        
        # メインループ
        while True:
            time.sleep(60)  # 1分毎に状況チェック
            if not scheduler.is_running:
                break
                
    except KeyboardInterrupt:
        print("\nキーボード割り込みを受信しました...")
    except Exception as e:
        print(f"予期しないエラー: {e}")
    finally:
        scheduler.stop_scheduler()
        print("DSSMSスケジューラー終了")


if __name__ == "__main__":
    main()
