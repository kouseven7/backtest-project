"""
ペーパートレード監視・ログシステム
詳細モニタリング + パフォーマンス追跡
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path
import sys

# プロジェクトパス追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger

class PaperTradeMonitor:
    """ペーパートレード監視システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("PaperTradeMonitor", log_file="logs/paper_trade_monitor.log")
        
        # 実行履歴
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.alert_counters: Dict[str, int] = {}
        
        # ファイルパス
        self.logs_dir = Path("logs/paper_trading")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 監視設定
        self.performance_window = config.get('performance_window_hours', 24)
        self.alert_thresholds = config.get('alert_thresholds', {})
        
    def record_execution(self, execution_result: Dict[str, Any]) -> None:
        """実行結果記録"""
        timestamp = datetime.now()
        
        # 実行記録
        record = {
            "timestamp": timestamp.isoformat(),
            "execution_id": f"exec_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            "result": execution_result,
            "performance_snapshot": self._capture_performance_snapshot()
        }
        
        self.execution_history.append(record)
        
        # ログ出力
        self._log_execution_result(record)
        
        # アラートチェック
        self._check_alerts(record)
        
        # 履歴クリーンアップ
        self._cleanup_old_records()
    
    def update_portfolio_status(self) -> None:
        """ポートフォリオ状態更新"""
        try:
            # ポートフォリオデータ取得（既存システムから）
            portfolio_data = self._get_portfolio_data()
            
            # パフォーマンス計算
            performance = self._calculate_performance_metrics(portfolio_data)
            
            # 更新
            self.performance_metrics.update(performance)
            
            # ログ
            self.logger.info(f"ポートフォリオ更新: {performance}")
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ状態更新エラー: {e}")
    
    def generate_final_report(self) -> None:
        """最終レポート生成"""
        try:
            self.logger.info("最終レポート生成開始")
            
            # 実行サマリー
            summary = self._generate_execution_summary()
            
            # パフォーマンスレポート
            performance_report = self._generate_performance_report()
            
            # ファイル出力
            report_path = self.logs_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            final_report = {
                "generated_at": datetime.now().isoformat(),
                "execution_summary": summary,
                "performance_report": performance_report,
                "execution_history": self.execution_history[-100:]  # 直近100件
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"最終レポート保存: {report_path}")
            
        except Exception as e:
            self.logger.error(f"最終レポート生成エラー: {e}")
    
    def _capture_performance_snapshot(self) -> Dict[str, Any]:
        """パフォーマンススナップショット取得"""
        try:
            # 基本メトリクス
            return {
                "timestamp": datetime.now().isoformat(),
                "total_executions": len(self.execution_history),
                "recent_performance": self.performance_metrics.get("recent_return", 0.0),
                "portfolio_value": self.performance_metrics.get("portfolio_value", 100000.0),
                "active_positions": self.performance_metrics.get("active_positions", 0)
            }
        except Exception:
            return {"error": "snapshot_failed"}
    
    def _get_portfolio_data(self) -> Dict[str, Any]:
        """ポートフォリオデータ取得"""
        # 簡易実装：実際には既存のportfolio_trackerと連携
        return {
            "total_value": 100000.0,
            "positions": [],
            "cash": 100000.0,
            "daily_return": 0.0
        }
    
    def _calculate_performance_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """パフォーマンス計算"""
        return {
            "portfolio_value": portfolio_data.get("total_value", 100000.0),
            "recent_return": portfolio_data.get("daily_return", 0.0),
            "active_positions": len(portfolio_data.get("positions", [])),
            "cash_ratio": portfolio_data.get("cash", 0) / portfolio_data.get("total_value", 1)
        }
    
    def _log_execution_result(self, record: Dict[str, Any]) -> None:
        """実行結果ログ出力"""
        result = record["result"]
        
        if result.get("success", False):
            self.logger.info(f"実行成功: {record['execution_id']} - {result.get('summary', '')}")
        else:
            self.logger.warning(f"実行失敗: {record['execution_id']} - {result.get('error', '')}")
    
    def _check_alerts(self, record: Dict[str, Any]) -> None:
        """アラートチェック"""
        result = record["result"]
        
        # エラーアラート
        if not result.get("success", False):
            self._trigger_alert("execution_failure", f"実行失敗: {result.get('error', '')}")
        
        # パフォーマンスアラート
        recent_return = self.performance_metrics.get("recent_return", 0.0)
        if recent_return < self.alert_thresholds.get("min_return", -0.05):
            self._trigger_alert("poor_performance", f"低パフォーマンス: {recent_return:.2%}")
    
    def _trigger_alert(self, alert_type: str, message: str) -> None:
        """アラート発生"""
        self.alert_counters[alert_type] = self.alert_counters.get(alert_type, 0) + 1
        self.logger.warning(f"アラート[{alert_type}]: {message}")
        
        # アラート履歴保存
        alert_record = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message,
            "count": self.alert_counters[alert_type]
        }
        
        alerts_file = self.logs_dir / "alerts.jsonl"
        with open(alerts_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(alert_record, ensure_ascii=False) + '\n')
    
    def _cleanup_old_records(self) -> None:
        """古い記録のクリーンアップ"""
        # 24時間以上古いレコードを削除
        cutoff_time = datetime.now() - timedelta(hours=self.performance_window)
        
        self.execution_history = [
            record for record in self.execution_history
            if datetime.fromisoformat(record["timestamp"]) > cutoff_time
        ]
    
    def _generate_execution_summary(self) -> Dict[str, Any]:
        """実行サマリー生成"""
        total_executions = len(self.execution_history)
        successful_executions = len([r for r in self.execution_history if r["result"].get("success", False)])
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "alert_summary": self.alert_counters
        }
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート生成"""
        return {
            "current_metrics": self.performance_metrics,
            "performance_window_hours": self.performance_window,
            "monitoring_period": {
                "start": self.execution_history[0]["timestamp"] if self.execution_history else None,
                "end": datetime.now().isoformat()
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """監視システム状態取得"""
        return {
            "total_executions": len(self.execution_history),
            "recent_executions": len([r for r in self.execution_history 
                                    if datetime.fromisoformat(r["timestamp"]) > 
                                    datetime.now() - timedelta(hours=1)]),
            "performance_metrics": self.performance_metrics,
            "alert_counters": self.alert_counters,
            "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None
        }
