"""
ペーパートレードスケジューラー
固定間隔実行 + 市場時間考慮
"""
from datetime import datetime, timedelta, time
from typing import Dict, Optional, Any
import json
from pathlib import Path

class PaperTradeScheduler:
    """ペーパートレード実行スケジューラー"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.next_execution_time: Optional[datetime] = None
        self.market_hours = self._load_market_hours()
        
        # デフォルト設定
        self.default_interval = config.get('default_interval_minutes', 15)
        self.market_hours_only = config.get('market_hours_only', True)
        
    def _load_market_hours(self) -> Dict[str, Any]:
        """市場時間設定読み込み"""
        try:
            market_hours_path = Path("config/paper_trading/market_hours.json")
            if market_hours_path.exists():
                with open(market_hours_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        
        # デフォルト市場時間
        return {
            "market_sessions": {
                "pre_market": {"open": "09:00", "close": "09:30"},
                "regular": {"open": "09:30", "close": "16:00"},
                "after_hours": {"open": "16:00", "close": "20:00"}
            },
            "active_sessions": ["regular"]
        }
    
    def should_execute(self) -> bool:
        """実行すべきかチェック"""
        now = datetime.now()
        
        # 初回実行
        if self.next_execution_time is None:
            if self._is_market_open(now):
                self.next_execution_time = now
                return True
            return False
        
        # 実行時刻チェック
        if now >= self.next_execution_time:
            if self._is_market_open(now):
                return True
            else:
                # 市場時間外の場合、次の市場開始時間に設定
                self._schedule_next_market_open()
        
        return False
    
    def schedule_next(self, interval_minutes: Optional[int] = None) -> None:
        """次回実行時刻設定"""
        interval = interval_minutes or self.default_interval
        self.next_execution_time = datetime.now() + timedelta(minutes=interval)
    
    def _is_market_open(self, dt: datetime) -> bool:
        """市場開始時間かチェック"""
        if not self.market_hours_only:
            return True
        
        current_time = dt.time()
        current_weekday = dt.weekday()
        
        # 土日チェック
        if current_weekday >= 5:  # 5=土曜, 6=日曜
            return False
        
        # アクティブセッションチェック
        market_sessions = self.market_hours.get("market_sessions", {})
        regular_session = market_sessions.get("regular", {})
        
        # 既存のmarket_hours.jsonの形式に合わせる
        open_time_str = regular_session.get("open", "09:30")
        close_time_str = regular_session.get("close", "16:00")
        
        try:
            open_time = time.fromisoformat(open_time_str)
            close_time = time.fromisoformat(close_time_str)
            
            if open_time <= current_time <= close_time:
                return True
        except ValueError:
            # パース失敗時はデフォルトで市場開始とみなす
            return True
        
        return False
    
    def _schedule_next_market_open(self) -> None:
        """次の市場開始時間に設定"""
        now = datetime.now()
        
        # 今日の市場開始時間をチェック
        market_start = self._get_next_market_start(now)
        self.next_execution_time = market_start
    
    def _get_next_market_start(self, from_dt: datetime) -> datetime:
        """次の市場開始時間取得"""
        # 簡易実装：平日9:30を返す
        next_day = from_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        
        if next_day <= from_dt:
            next_day += timedelta(days=1)
        
        # 土日スキップ
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        
        return next_day
    
    def get_status(self) -> Dict[str, Any]:
        """スケジューラー状態取得"""
        return {
            "next_execution": self.next_execution_time.isoformat() if self.next_execution_time else None,
            "market_open": self._is_market_open(datetime.now()),
            "default_interval": self.default_interval,
            "market_hours_only": self.market_hours_only
        }
