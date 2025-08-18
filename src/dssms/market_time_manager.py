"""
DSSMS Phase 4 Task 4.2: MarketTimeManager
日本市場時間管理システム

市場時間管理・祝日チェック・スケジューリング支援
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, time, date, timedelta
import pytz
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

class MarketTimeManager:
    """日本市場時間管理システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス（None時はデフォルト使用）
        """
        self.logger = setup_logger('dssms.market_time_manager')
        
        # 設定ファイル読み込み
        if config_path is None:
            config_path_obj = Path(__file__).parent.parent.parent / "config" / "dssms" / "scheduler_config.json"
        else:
            config_path_obj = Path(config_path)
        
        self.config = self._load_config(config_path_obj)
        self.timezone = pytz.timezone(self.config['market_hours']['timezone'])
        self.market_holidays = set(self.config.get('market_holidays', []))
        
        self.logger.info("MarketTimeManager: 初期化完了")
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"設定ファイル読み込み成功: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー {config_path}: {e}")
            # デフォルト設定を返す
            return {
                "market_hours": {
                    "timezone": "Asia/Tokyo",
                    "trading_sessions": {
                        "morning": {"start": "09:00", "end": "11:30"},
                        "afternoon": {"start": "12:30", "end": "15:00"}
                    }
                },
                "market_holidays": []
            }
    
    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """
        現在（または指定時刻）に市場が開場中かチェック
        
        Args:
            check_time: チェックする時刻（None時は現在時刻）
        
        Returns:
            bool: 市場開場中の場合True
        """
        if check_time is None:
            check_time = datetime.now(self.timezone)
        elif check_time.tzinfo is None:
            check_time = self.timezone.localize(check_time)
        else:
            check_time = check_time.astimezone(self.timezone)
        
        # 祝日チェック
        if self._is_market_holiday(check_time.date()):
            return False
        
        # 平日チェック (月-金: 0-4)
        if check_time.weekday() > 4:  # 土日
            return False
        
        # 時間チェック
        current_time = check_time.time()
        trading_sessions = self.config['market_hours']['trading_sessions']
        
        # 前場チェック
        morning_start = time.fromisoformat(trading_sessions['morning']['start'])
        morning_end = time.fromisoformat(trading_sessions['morning']['end'])
        
        # 後場チェック
        afternoon_start = time.fromisoformat(trading_sessions['afternoon']['start'])
        afternoon_end = time.fromisoformat(trading_sessions['afternoon']['end'])
        
        is_morning_session = morning_start <= current_time <= morning_end
        is_afternoon_session = afternoon_start <= current_time <= afternoon_end
        
        return is_morning_session or is_afternoon_session
    
    def _is_market_holiday(self, check_date: date) -> bool:
        """指定日が市場休日かチェック"""
        return check_date.isoformat() in self.market_holidays
    
    def get_next_screening_time(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """
        次回スクリーニング時刻取得
        
        Args:
            from_time: 基準時刻（None時は現在時刻）
        
        Returns:
            Optional[datetime]: 次回スクリーニング時刻
        """
        if from_time is None:
            from_time = datetime.now(self.timezone)
        elif from_time.tzinfo is None:
            from_time = self.timezone.localize(from_time)
        else:
            from_time = from_time.astimezone(self.timezone)
        
        # 当日のスクリーニング時刻
        morning_screening = from_time.replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        afternoon_screening = from_time.replace(
            hour=12, minute=30, second=0, microsecond=0
        )
        
        # 当日のスクリーニング機会をチェック
        if (from_time < morning_screening and 
            not self._is_market_holiday(from_time.date()) and
            from_time.weekday() <= 4):
            return morning_screening
        
        if (from_time < afternoon_screening and 
            not self._is_market_holiday(from_time.date()) and
            from_time.weekday() <= 4):
            return afternoon_screening
        
        # 翌営業日の朝に設定
        next_business_day = self._get_next_business_day(from_time.date())
        return next_business_day.replace(
            hour=9, minute=30, second=0, microsecond=0
        )
    
    def _get_next_business_day(self, from_date: date) -> datetime:
        """次の営業日を取得"""
        next_date = from_date + timedelta(days=1)
        
        # 最大10日先まで検索
        for _ in range(10):
            if (next_date.weekday() <= 4 and  # 平日
                not self._is_market_holiday(next_date)):
                return self.timezone.localize(
                    datetime.combine(next_date, time(9, 30))
                )
            next_date += timedelta(days=1)
        
        # 10日以内に営業日が見つからない場合は翌週月曜日
        while next_date.weekday() != 0:  # 月曜日まで進める
            next_date += timedelta(days=1)
        
        return self.timezone.localize(
            datetime.combine(next_date, time(9, 30))
        )
    
    def should_run_screening(self, session: str, tolerance_minutes: int = 5) -> bool:
        """
        スクリーニング実行タイミングかチェック
        
        Args:
            session: "morning" または "afternoon"
            tolerance_minutes: 許容時間（分）
        
        Returns:
            bool: 実行タイミングの場合True
        """
        now = datetime.now(self.timezone)
        
        if not self.is_market_open(now):
            return False
        
        if session == "morning":
            target_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        elif session == "afternoon":
            target_time = now.replace(hour=12, minute=30, second=0, microsecond=0)
        else:
            self.logger.error(f"無効なセッション: {session}")
            return False
        
        # 実行時刻の許容時間内をOKとする
        time_diff = abs((now - target_time).total_seconds())
        return time_diff <= (tolerance_minutes * 60)
    
    def get_current_session(self) -> Optional[str]:
        """
        現在のセッション取得
        
        Returns:
            Optional[str]: "morning", "afternoon", "closed"
        """
        now = datetime.now(self.timezone)
        
        if not self.is_market_open(now):
            return "closed"
        
        current_time = now.time()
        trading_sessions = self.config['market_hours']['trading_sessions']
        
        morning_start = time.fromisoformat(trading_sessions['morning']['start'])
        morning_end = time.fromisoformat(trading_sessions['morning']['end'])
        afternoon_start = time.fromisoformat(trading_sessions['afternoon']['start'])
        afternoon_end = time.fromisoformat(trading_sessions['afternoon']['end'])
        
        if morning_start <= current_time <= morning_end:
            return "morning"
        elif afternoon_start <= current_time <= afternoon_end:
            return "afternoon"
        else:
            return "closed"
    
    def get_time_until_next_session(self) -> Dict[str, Any]:
        """
        次セッションまでの時間取得
        
        Returns:
            Dict[str, Any]: 次セッション情報
        """
        now = datetime.now(self.timezone)
        current_session = self.get_current_session()
        
        result: Dict[str, Any] = {
            "current_session": current_session,
            "next_session": None,
            "time_until_next": None,
            "next_session_start": None
        }
        
        if current_session == "morning":
            # 前場中 → 後場開始まで
            afternoon_start = now.replace(hour=12, minute=30, second=0, microsecond=0)
            result["next_session"] = "afternoon"
            result["next_session_start"] = afternoon_start
            result["time_until_next"] = (afternoon_start - now).total_seconds()
        
        elif current_session == "afternoon":
            # 後場中 → 翌営業日前場まで
            next_business_day = self._get_next_business_day(now.date())
            result["next_session"] = "morning"
            result["next_session_start"] = next_business_day
            result["time_until_next"] = (next_business_day - now).total_seconds()
        
        else:  # closed
            next_screening = self.get_next_screening_time(now)
            if next_screening:
                if next_screening.hour == 9:
                    result["next_session"] = "morning"
                else:
                    result["next_session"] = "afternoon"
                result["next_session_start"] = next_screening
                result["time_until_next"] = (next_screening - now).total_seconds()
        
        return result
