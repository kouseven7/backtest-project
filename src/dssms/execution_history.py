"""
DSSMS Phase 4 Task 4.2: ExecutionHistory
DSSMS実行履歴管理システム

スクリーニング・切替・監視イベントの履歴管理
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
import json
import threading
import shutil
import tempfile
import os

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

class ExecutionHistory:
    """DSSMS実行履歴管理システム"""
    
    def __init__(self, history_dir: Optional[str] = None, market_monitor=None):
        """
        初期化
        
        Args:
            history_dir: 履歴保存ディレクトリ（None時はデフォルト使用）
            market_monitor: MarketConditionMonitorインスタンス（市場分析用）
        """
        self.logger = setup_logger('dssms.execution_history')
        
        # 履歴保存ディレクトリ設定
        if history_dir is None:
            self.history_dir = Path(__file__).parent.parent.parent / "logs" / "dssms"
        else:
            self.history_dir = Path(history_dir)
        
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / "execution_history.json"
        
        # MarketConditionMonitor統合
        self.market_monitor = market_monitor
        
        # 現在セッションデータ
        self.current_session_data: Dict[str, Any] = {}
        
        # スレッドロック
        self._file_lock = threading.Lock()
        
        # 起動時JSON破損チェック
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                backup_file = self.history_file.with_suffix(
                    f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                )
                shutil.copy2(self.history_file, backup_file)
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                self.logger.warning(
                    f"履歴ファイル破損を検出しバックアップ保存: {backup_file.name} -> 空ファイルで初期化"
                )
        
        self.logger.info(f"ExecutionHistory: 初期化完了 (履歴保存先: {self.history_file})")
    
    def record_screening_execution(self, session: str, result: Dict[str, Any]) -> bool:
        """
        スクリーニング実行記録
        
        Args:
            session: セッション種別 ("morning" or "afternoon")
            result: スクリーニング結果
        
        Returns:
            bool: 記録成功の場合True
        """
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "screening",
                "session_type": session,
                "execution_status": result.get("status", "unknown"),
                "selected_symbol": result.get("selected_symbol"),
                "candidate_count": result.get("candidate_count", 0),
                "screening_duration_seconds": result.get("duration", 0),
                "priority_distribution": result.get("priority_distribution", {}),
                "market_conditions": self._capture_market_snapshot()
            }
            
            self._append_to_history(record)
            self.current_session_data[session] = record
            
            self.logger.info(f"スクリーニング実行記録: {session} - {result.get('selected_symbol', 'N/A')}")
            return True
            
        except Exception as e:
            self.logger.error(f"スクリーニング実行記録エラー: {e}")
            return False
    
    def record_symbol_switch(self, switch_data: Dict[str, Any]) -> bool:
        """
        銘柄切替記録
        
        Args:
            switch_data: 切替データ
        
        Returns:
            bool: 記録成功の場合True
        """
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "symbol_switch",
                "from_symbol": switch_data.get("from_symbol"),
                "to_symbol": switch_data.get("to_symbol"),
                "switch_reason": switch_data.get("reason"),
                "emergency_level": switch_data.get("emergency_level", 0),
                "switch_conditions": switch_data.get("conditions", []),
                "execution_result": switch_data.get("execution_result", {}),
                "market_conditions": self._capture_market_snapshot()
            }
            
            self._append_to_history(record)
            
            self.logger.info(f"銘柄切替記録: {switch_data.get('from_symbol')} → {switch_data.get('to_symbol')}")
            return True
            
        except Exception as e:
            self.logger.error(f"銘柄切替記録エラー: {e}")
            return False
    
    def record_monitoring_event(self, symbol: str, event_data: Dict[str, Any]) -> bool:
        """
        監視イベント記録
        
        Args:
            symbol: 監視対象銘柄
            event_data: イベントデータ
        
        Returns:
            bool: 記録成功の場合True
        """
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "monitoring",
                "symbol": symbol,
                "event_details": event_data,
                "market_conditions": self._capture_market_snapshot()
            }
            
            self._append_to_history(record)
            
            self.logger.debug(f"監視イベント記録: {symbol} - {event_data.get('event_type', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"監視イベント記録エラー: {e}")
            return False
    
    def record_emergency_check(self, symbol: str, emergency_result: Dict[str, Any]) -> bool:
        """
        緊急チェック記録
        
        Args:
            symbol: チェック対象銘柄
            emergency_result: 緊急判定結果
        
        Returns:
            bool: 記録成功の場合True
        """
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "emergency_check",
                "symbol": symbol,
                "is_emergency": emergency_result.get("is_emergency", False),
                "emergency_level": emergency_result.get("emergency_level", 0),
                "recommended_action": emergency_result.get("recommended_action", "hold"),
                "trigger_conditions": emergency_result.get("trigger_conditions", []),
                "emergency_score": emergency_result.get("emergency_score", 0),
                "market_conditions": self._capture_market_snapshot()
            }
            
            self._append_to_history(record)
            
            if emergency_result.get("is_emergency"):
                self.logger.warning(f"緊急事態記録: {symbol} - Level {emergency_result.get('emergency_level')}")
            else:
                self.logger.debug(f"緊急チェック記録: {symbol} - 正常")
            
            return True
            
        except Exception as e:
            self.logger.error(f"緊急チェック記録エラー: {e}")
            return False
    
    def _append_to_history(self, record: Dict[str, Any]) -> bool:
        """履歴ファイルに記録追加（スレッドセーフ・アトミック書き込み版）"""
        try:
            with self._file_lock:
                # 既存履歴読み込み
                if self.history_file.exists():
                    with open(self.history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                else:
                    history = []

                history.append(record)

                # アトミック書き込み（一時ファイル経由でrenameする）
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=str(self.history_file.parent),
                    suffix='.tmp',
                    prefix='execution_history_'
                )
                try:
                    with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                    # Windowsでは上書きrenameのためunlinkが必要
                    if self.history_file.exists():
                        self.history_file.unlink()
                    shutil.move(temp_path, str(self.history_file))
                except Exception:
                    Path(temp_path).unlink(missing_ok=True)
                    raise

                return True

        except Exception as e:
            self.logger.error(f"履歴ファイル記録エラー: {e}")
            return False
    
    def _capture_market_snapshot(self) -> Dict[str, Any]:
        """市場スナップショット取得"""
        try:
            # MarketConditionMonitorから日経225トレンドを取得
            nikkei225_trend = "unknown"
            if self.market_monitor is not None:
                try:
                    trend_analysis = self.market_monitor.analyze_nikkei225_trend()
                    nikkei225_trend = trend_analysis.get("trend_direction", "unknown")
                except Exception as e:
                    self.logger.warning(f"日経225トレンド取得エラー: {e}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "market_open": True,  # 簡略版
                "nikkei225_trend": nikkei225_trend,
                "volatility_level": "normal"
            }
        except Exception as e:
            self.logger.error(f"市場スナップショット取得エラー: {e}")
            return {}
    
    def get_daily_summary(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        日次サマリー取得
        
        Args:
            target_date: 対象日（None時は当日）
        
        Returns:
            Dict[str, Any]: 日次サマリー
        """
        if target_date is None:
            target_date = datetime.now().date()
        
        try:
            daily_records = self._get_records_by_date(target_date)
            
            summary = {
                "date": target_date.isoformat(),
                "total_events": len(daily_records),
                "total_screenings": len([r for r in daily_records if r.get("event_type") == "screening"]),
                "symbol_switches": len([r for r in daily_records if r.get("event_type") == "symbol_switch"]),
                "monitoring_events": len([r for r in daily_records if r.get("event_type") == "monitoring"]),
                "emergency_checks": len([r for r in daily_records if r.get("event_type") == "emergency_check"]),
                "emergency_alerts": len([r for r in daily_records if r.get("event_type") == "emergency_check" and r.get("is_emergency")]),
                "selected_symbols": list(set([r.get("selected_symbol") for r in daily_records if r.get("selected_symbol")])),
                "switch_reasons": [r.get("switch_reason") for r in daily_records if r.get("switch_reason")],
                "average_screening_duration": self._calculate_average_duration(daily_records)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"日次サマリー取得エラー: {e}")
            return {"error": str(e)}
    
    def _get_records_by_date(self, target_date: date) -> List[Dict[str, Any]]:
        """指定日の記録取得"""
        try:
            if not self.history_file.exists():
                return []
            
            with open(self.history_file, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
            
            daily_records = []
            target_date_str = target_date.isoformat()
            
            for record in all_records:
                record_date = record.get("timestamp", "")[:10]  # YYYY-MM-DD部分
                if record_date == target_date_str:
                    daily_records.append(record)
            
            return daily_records
            
        except Exception as e:
            self.logger.error(f"日別記録取得エラー: {e}")
            return []
    
    def _calculate_average_duration(self, records: List[Dict[str, Any]]) -> float:
        """平均実行時間計算"""
        try:
            screening_records = [r for r in records if r.get("event_type") == "screening"]
            durations = [r.get("screening_duration_seconds", 0) for r in screening_records]
            
            if durations:
                return sum(durations) / len(durations)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"平均実行時間計算エラー: {e}")
            return 0.0
    
    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        最近のイベント取得
        
        Args:
            limit: 取得件数制限
        
        Returns:
            List[Dict[str, Any]]: 最近のイベント
        """
        try:
            if not self.history_file.exists():
                return []
            
            with open(self.history_file, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
            
            # 最新順でソート
            sorted_records = sorted(all_records, key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return sorted_records[:limit]
            
        except Exception as e:
            self.logger.error(f"最近のイベント取得エラー: {e}")
            return []
    
    def cleanup_old_records(self, retention_days: int = 90) -> bool:
        """
        古い記録のクリーンアップ
        
        Args:
            retention_days: 保存期間（日）
        
        Returns:
            bool: クリーンアップ成功の場合True
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            cutoff_date_str = cutoff_date.date().isoformat()
            
            if not self.history_file.exists():
                return True
            
            with open(self.history_file, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
            
            # 保存期間内の記録のみ残す
            recent_records = []
            removed_count = 0
            
            for record in all_records:
                record_date = record.get("timestamp", "")[:10]
                if record_date >= cutoff_date_str:
                    recent_records.append(record)
                else:
                    removed_count += 1
            
            # ファイルに保存
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(recent_records, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"履歴クリーンアップ完了: {removed_count}件削除, {len(recent_records)}件保持")
            return True
            
        except Exception as e:
            self.logger.error(f"履歴クリーンアップエラー: {e}")
            return False
    
    def record_trade_buy(self, symbol: str, price: float, quantity: int, strategy: str) -> bool:
        """
        BUY取引を記録する
        
        Args:
            symbol: 銘柄コード
            price: 購入価格
            quantity: 購入株数
            strategy: 使用戦略名
        
        Returns:
            bool: 記録成功の場合True
        """
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "buy",
                "symbol": symbol,
                "price": price,
                "quantity": quantity,
                "strategy": strategy,
                "market_conditions": self._capture_market_snapshot()
            }
            self._append_to_history(record)
            self.logger.info(f"BUY取引記録: {symbol} - {quantity}株 @ {price}円 (戦略: {strategy})")
            return True
        except Exception as e:
            self.logger.error(f"BUY取引記録エラー: {e}")
            return False
    
    def record_trade_sell(self, symbol: str, sell_price: float, quantity: int,
                          entry_price: float, reason: str) -> bool:
        """
        SELL取引を記録する
        
        Args:
            symbol: 銘柄コード
            sell_price: 売却価格
            quantity: 売却株数
            entry_price: エントリー価格
            reason: 売却理由
        
        Returns:
            bool: 記録成功の場合True
        """
        try:
            profit_loss = (sell_price - entry_price) * quantity
            profit_loss_pct = (sell_price - entry_price) / entry_price * 100
            
            record = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "sell",
                "symbol": symbol,
                "sell_price": sell_price,
                "entry_price": entry_price,
                "quantity": quantity,
                "profit_loss": round(profit_loss, 0),
                "profit_loss_pct": round(profit_loss_pct, 2),
                "reason": reason,
                "market_conditions": self._capture_market_snapshot()
            }
            self._append_to_history(record)
            self.logger.info(
                f"SELL取引記録: {symbol} - {quantity}株 @ {sell_price}円 "
                f"(損益: {profit_loss:+,.0f}円 / {profit_loss_pct:+.2f}%, 理由: {reason})"
            )
            return True
        except Exception as e:
            self.logger.error(f"SELL取引記録エラー: {e}")
            return False
