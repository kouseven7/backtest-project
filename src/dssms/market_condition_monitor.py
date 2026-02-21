"""
DSSMS Market Condition Monitor
Phase 3 Task 3.1: 市場全体監視システム

日経225指数ベースの市場監視と売買停止判定
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存コンポーネントのインポート
from .dssms_data_manager import DSSMSDataManager
from .perfect_order_detector import PerfectOrderDetector
from .market_health_indicators import MarketHealthIndicators
from config.logger_config import setup_logger

class MarketConditionMonitor:
    """日経225指数ベースの市場監視システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス（None時はデフォルト使用）
        """
        self.logger = setup_logger('dssms.market_monitor')
        
        # 設定ファイル読み込み
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "dssms" / "market_monitoring_config.json"
        
        self.config = self._load_config(config_path)
        
        # 既存コンポーネント初期化
        try:
            self.data_manager = DSSMSDataManager()
            self.perfect_order_detector = PerfectOrderDetector()
            self.health_indicators = MarketHealthIndicators(self.config)
            self.logger.info("MarketConditionMonitor: 既存コンポーネント初期化成功")
        except Exception as e:
            self.logger.warning(f"既存コンポーネント初期化エラー: {e}")
            self.data_manager = None
            self.perfect_order_detector = None
            self.health_indicators = None
        
        # 監視状態
        self.last_check_time = None
        self.monitoring_cache = {}
        self.alert_history = []
        
        self.logger.info("MarketConditionMonitor initialized")
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"設定ファイル読み込み失敗: {e}. デフォルト設定使用")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "monitoring": {
                "check_interval_minutes": 15,
                "market_hours": {
                    "morning_start": "09:00",
                    "morning_end": "11:30",
                    "afternoon_start": "12:30",
                    "afternoon_end": "15:00"
                },
                "out_of_hours_behavior": "use_last_available"
            },
            "nikkei225_analysis": {
                "symbol": "^N225",
                "timeframes": {
                    "daily": {"short": 5, "medium": 25, "long": 75},
                    "weekly": {"short": 13, "medium": 26, "long": 52},
                    "monthly": {"short": 9, "medium": 24, "long": 60}
                }
            },
            "health_scoring": {
                "weights": {
                    "perfect_order_status": 0.40,
                    "volatility_level": 0.25,
                    "volume_profile": 0.20,
                    "trend_strength": 0.15
                }
            },
            "halt_conditions": {
                "levels": {
                    "normal": {"min_health_score": 0.7, "action": "continue"},
                    "warning": {"min_health_score": 0.5, "action": "monitor_closely"},
                    "caution": {"min_health_score": 0.3, "action": "reduce_exposure"},
                    "halt": {"min_health_score": 0.0, "action": "stop_trading"}
                },
                "emergency_conditions": {
                    "market_crash_threshold": -0.05,
                    "volatility_spike_threshold": 0.05,
                    "volume_dry_up_threshold": 0.3
                }
            }
        }
    
    def analyze_nikkei225_trend(self) -> Dict[str, Any]:
        """
        日経225指数のトレンド分析
        
        Returns:
            {
                "trend_direction": "up|down|sideways",
                "perfect_order_status": {"daily": bool, "weekly": bool, "monthly": bool},
                "strength_score": float,  # 0-1
                "volatility_level": "low|normal|high",
                "volume_profile": "increasing|decreasing|stable"
            }
        """
        try:
            # 日経225データ取得
            nikkei_data = self._get_nikkei225_data()
            if nikkei_data.empty:
                return {"error": "Failed to fetch Nikkei 225 data"}
            
            # トレンド方向判定
            trend_direction = self._analyze_trend_direction(nikkei_data)
            
            # パーフェクトオーダー状態
            perfect_order_status = self._check_perfect_order_status(nikkei_data)
            
            # 強度スコア計算
            strength_score = self._calculate_trend_strength(nikkei_data)
            
            # ボラティリティレベル
            volatility_level = self._analyze_volatility_level(nikkei_data)
            
            # 出来高プロファイル
            volume_profile = self._analyze_volume_profile(nikkei_data)
            
            result = {
                "symbol": "^N225",
                "trend_direction": trend_direction,
                "perfect_order_status": perfect_order_status,
                "strength_score": strength_score,
                "volatility_level": volatility_level,
                "volume_profile": volume_profile,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points": len(nikkei_data)
            }
            
            # キャッシュ保存
            self.monitoring_cache["trend_analysis"] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"日経225トレンド分析エラー: {e}")
            return {"error": str(e)}
    
    def check_market_perfect_order(self) -> bool:
        """
        市場全体のパーフェクトオーダー状態確認
        
        Returns:
            パーフェクトオーダー状態かどうか
        """
        try:
            nikkei_data = self._get_nikkei225_data()
            if nikkei_data.empty:
                return False
            
            # 日足パーフェクトオーダーチェック
            timeframes = self.config.get("nikkei225_analysis", {}).get("timeframes", {})
            daily_periods = timeframes.get("daily", {"short": 5, "medium": 25, "long": 75})
            
            if len(nikkei_data) < daily_periods["long"]:
                return False
            
            # 移動平均計算
            short_ma = nikkei_data['Close'].rolling(window=daily_periods["short"]).mean()
            medium_ma = nikkei_data['Close'].rolling(window=daily_periods["medium"]).mean()
            long_ma = nikkei_data['Close'].rolling(window=daily_periods["long"]).mean()
            
            current_price = nikkei_data['Close'].iloc[-1]
            current_short = short_ma.iloc[-1]
            current_medium = medium_ma.iloc[-1]
            current_long = long_ma.iloc[-1]
            
            # パーフェクトオーダー判定
            is_perfect = (
                current_price > current_short and
                current_short > current_medium and
                current_medium > current_long
            )
            
            self.logger.debug(f"Perfect Order Check: {is_perfect} (Price: {current_price:.2f}, "
                            f"MA5: {current_short:.2f}, MA25: {current_medium:.2f}, MA75: {current_long:.2f})")
            
            return is_perfect
            
        except Exception as e:
            self.logger.error(f"パーフェクトオーダーチェックエラー: {e}")
            return False
    
    def should_halt_trading(self) -> Tuple[bool, str]:
        """
        売買停止判定（段階的）
        
        Returns:
            (halt_flag, reason)
            - halt_flag: True=停止, False=継続
            - reason: "normal|warning|caution|halt"
        """
        try:
            # 市場ヘルススコア取得
            health_score = self.get_market_health_score()
            
            # 緊急条件チェック
            emergency_reason = self._check_emergency_conditions()
            if emergency_reason:
                self._log_alert("halt", f"Emergency condition: {emergency_reason}")
                return True, "halt"
            
            # 段階的判定
            halt_levels = self.config.get("halt_conditions", {}).get("levels", {})
            
            if health_score >= halt_levels.get("normal", {}).get("min_health_score", 0.7):
                return False, "normal"
            elif health_score >= halt_levels.get("warning", {}).get("min_health_score", 0.5):
                self._log_alert("warning", f"Health score: {health_score:.3f}")
                return False, "warning"
            elif health_score >= halt_levels.get("caution", {}).get("min_health_score", 0.3):
                self._log_alert("caution", f"Health score: {health_score:.3f}")
                return False, "caution"
            else:
                self._log_alert("halt", f"Low health score: {health_score:.3f}")
                return True, "halt"
            
        except Exception as e:
            self.logger.error(f"売買停止判定エラー: {e}")
            # エラー時は安全のため停止
            return True, "halt"
    
    def get_market_health_score(self) -> float:
        """
        市場ヘルススコア算出
        
        Returns:
            0.0-1.0のスコア
        """
        try:
            nikkei_data = self._get_nikkei225_data()
            if nikkei_data.empty:
                return 0.5  # デフォルトスコア
            
            if self.health_indicators:
                health_scores = self.health_indicators.get_composite_health_score(nikkei_data)
                composite_score = health_scores.get("composite", 0.5)
                
                self.logger.debug(f"Health scores breakdown: {health_scores}")
                return composite_score
            else:
                # フォールバック: 簡易計算
                return self._calculate_simple_health_score(nikkei_data)
                
        except Exception as e:
            self.logger.error(f"ヘルススコア計算エラー: {e}")
            return 0.5
    
    # ヘルパーメソッド
    def _get_nikkei225_data(self) -> pd.DataFrame:
        """日経225データ取得"""
        try:
            if self.data_manager and hasattr(self.data_manager, 'get_nikkei225_data'):
                return self.data_manager.get_nikkei225_data(period="1y")
            else:
                # フォールバック: 直接yfinance使用
                import yfinance as yf
                ticker = yf.Ticker("^N225")
                data = ticker.history(period="1y")
                return data.dropna() if not data.empty else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"日経225データ取得エラー: {e}")
            return pd.DataFrame()
    
    def _analyze_trend_direction(self, data: pd.DataFrame) -> str:
        """トレンド方向分析"""
        try:
            if len(data) < 25:
                return "unknown"
            
            # 短期・中期移動平均での判定
            ma_short = data['Close'].rolling(window=5).mean()
            ma_medium = data['Close'].rolling(window=25).mean()
            
            current_short = ma_short.iloc[-1]
            current_medium = ma_medium.iloc[-1]
            prev_short = ma_short.iloc[-5]
            prev_medium = ma_medium.iloc[-5]
            
            # トレンド判定
            if current_short > current_medium and current_short > prev_short:
                return "up"
            elif current_short < current_medium and current_short < prev_short:
                return "down"
            else:
                return "sideways"
                
        except Exception:
            return "unknown"
    
    def _check_perfect_order_status(self, data: pd.DataFrame) -> Dict[str, bool]:
        """パーフェクトオーダー状態チェック"""
        try:
            result = {"daily": False, "weekly": False, "monthly": False}
            
            # 日足チェック
            if len(data) >= 75:
                result["daily"] = self.check_market_perfect_order()
            
            # 週足・月足は簡易チェック（データが十分な場合のみ）
            if len(data) >= 252:  # 1年分のデータ
                # 週足相当チェック
                weekly_data = data.resample('W').last().dropna()
                if len(weekly_data) >= 52:
                    result["weekly"] = self._check_perfect_order_timeframe(weekly_data, 13, 26, 52)
                
                # 月足相当チェック
                monthly_data = data.resample('ME').last().dropna()
                if len(monthly_data) >= 60:
                    result["monthly"] = self._check_perfect_order_timeframe(monthly_data, 9, 24, 60)
            
            return result
            
        except Exception:
            return {"daily": False, "weekly": False, "monthly": False}
    
    def _check_perfect_order_timeframe(self, data: pd.DataFrame, short: int, medium: int, long: int) -> bool:
        """指定期間でのパーフェクトオーダーチェック"""
        try:
            if len(data) < long:
                return False
            
            short_ma = data['Close'].rolling(window=short).mean()
            medium_ma = data['Close'].rolling(window=medium).mean()
            long_ma = data['Close'].rolling(window=long).mean()
            
            current_price = data['Close'].iloc[-1]
            current_short = short_ma.iloc[-1]
            current_medium = medium_ma.iloc[-1]
            current_long = long_ma.iloc[-1]
            
            return (current_price > current_short and 
                   current_short > current_medium and 
                   current_medium > current_long)
                   
        except Exception:
            return False
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """トレンド強度計算"""
        try:
            if len(data) < 20:
                return 0.5
            
            # 価格変化率での強度評価
            returns = data['Close'].pct_change().dropna()
            recent_returns = returns.tail(10)
            
            # 一貫性スコア（同方向の動きが多いほど高い）
            positive_ratio = (recent_returns > 0).sum() / len(recent_returns)
            consistency = abs(positive_ratio - 0.5) * 2  # 0.5からの距離を2倍
            
            # ボラティリティ調整
            volatility = returns.std()
            volatility_score = min(volatility * 50, 1.0)  # 適度なボラティリティを評価
            
            strength = (consistency * 0.7 + volatility_score * 0.3)
            return min(max(strength, 0.0), 1.0)
            
        except Exception:
            return 0.5
    
    def _analyze_volatility_level(self, data: pd.DataFrame) -> str:
        """ボラティリティレベル分析"""
        try:
            if len(data) < 20:
                return "unknown"
            
            returns = data['Close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
            
            if volatility < 0.15:
                return "low"
            elif volatility > 0.30:
                return "high"
            else:
                return "normal"
                
        except Exception:
            return "unknown"
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> str:
        """出来高プロファイル分析"""
        try:
            if len(data) < 20:
                return "unknown"
            
            recent_volume = data['Volume'].tail(5).mean()
            historical_volume = data['Volume'].tail(20).head(15).mean()
            
            if historical_volume == 0:
                return "unknown"
            
            ratio = recent_volume / historical_volume
            
            if ratio > 1.2:
                return "increasing"
            elif ratio < 0.8:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "unknown"
    
    def _check_emergency_conditions(self) -> Optional[str]:
        """緊急条件チェック"""
        try:
            nikkei_data = self._get_nikkei225_data()
            if nikkei_data.empty or len(nikkei_data) < 2:
                return None
            
            emergency_config = self.config.get("halt_conditions", {}).get("emergency_conditions", {})
            
            # 市場暴落チェック
            latest_close = nikkei_data['Close'].iloc[-1]
            previous_close = nikkei_data['Close'].iloc[-2]
            daily_change = (latest_close - previous_close) / previous_close
            
            crash_threshold = emergency_config.get("market_crash_threshold", -0.05)
            if daily_change <= crash_threshold:
                return f"Market crash detected: {daily_change:.2%}"
            
            # ボラティリティ急騰チェック
            returns = nikkei_data['Close'].pct_change().dropna()
            current_vol = returns.rolling(window=5).std().iloc[-1]
            avg_vol = returns.rolling(window=20).std().iloc[-1]
            
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                vol_threshold = emergency_config.get("volatility_spike_threshold", 2.0)
                if vol_ratio >= vol_threshold:
                    return f"Volatility spike: {vol_ratio:.2f}x"
            
            # 出来高急減チェック
            recent_volume = nikkei_data['Volume'].tail(3).mean()
            avg_volume = nikkei_data['Volume'].rolling(window=20).mean().iloc[-1]
            
            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                volume_threshold = emergency_config.get("volume_dry_up_threshold", 0.3)
                if volume_ratio <= volume_threshold:
                    return f"Volume dry up: {volume_ratio:.2f}x"
            
            return None
            
        except Exception as e:
            self.logger.error(f"緊急条件チェックエラー: {e}")
            return None
    
    def _calculate_simple_health_score(self, data: pd.DataFrame) -> float:
        """簡易ヘルススコア計算（フォールバック）"""
        try:
            scores = []
            
            # パーフェクトオーダースコア
            po_score = 0.8 if self.check_market_perfect_order() else 0.3
            scores.append(po_score * 0.4)
            
            # ボラティリティスコア
            if len(data) >= 20:
                returns = data['Close'].pct_change().dropna()
                vol = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
                vol_score = 0.8 if 0.15 <= vol <= 0.30 else 0.4
                scores.append(vol_score * 0.25)
            
            # 出来高スコア
            if len(data) >= 20:
                volume_ratio = (data['Volume'].tail(5).mean() / 
                              data['Volume'].tail(20).head(15).mean())
                vol_score = 0.8 if 0.8 <= volume_ratio <= 1.5 else 0.4
                scores.append(vol_score * 0.2)
            
            # トレンド強度スコア
            trend_score = self._calculate_trend_strength(data)
            scores.append(trend_score * 0.15)
            
            return sum(scores) if scores else 0.5
            
        except Exception:
            return 0.5
    
    def _log_alert(self, level: str, message: str):
        """アラートログ"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.alert_history.append(alert)
        
        # 履歴制限（最新100件）
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"Market Alert [{level.upper()}]: {message}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視状況取得"""
        try:
            trend_analysis = self.analyze_nikkei225_trend()
            health_score = self.get_market_health_score()
            halt_flag, halt_reason = self.should_halt_trading()
            perfect_order = self.check_market_perfect_order()
            
            return {
                "status": "active",
                "last_check": datetime.now().isoformat(),
                "market_health_score": health_score,
                "halt_trading": halt_flag,
                "halt_reason": halt_reason,
                "perfect_order": perfect_order,
                "trend_analysis": trend_analysis,
                "recent_alerts": self.alert_history[-5:]  # 直近5件
            }
            
        except Exception as e:
            self.logger.error(f"監視状況取得エラー: {e}")
            return {"status": "error", "error": str(e)}


class DSSMSMarketMonitorIntegrator:
    """DSSMS市場監視統合インターフェース"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.monitor = MarketConditionMonitor(config_path)
        self.logger = setup_logger('dssms.market_monitor_integrator')
    
    def get_trading_permission(self) -> Dict[str, Any]:
        """取引許可状況取得"""
        try:
            halt_flag, reason = self.monitor.should_halt_trading()
            health_score = self.monitor.get_market_health_score()
            
            return {
                "trading_allowed": not halt_flag,
                "reason": reason,
                "health_score": health_score,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"取引許可確認エラー: {e}")
            return {
                "trading_allowed": False,
                "reason": "system_error",
                "health_score": 0.0,
                "error": str(e)
            }
    
    def get_market_summary(self) -> Dict[str, Any]:
        """市場サマリー取得"""
        return self.monitor.get_monitoring_status()
    
    def force_refresh(self):
        """強制リフレッシュ"""
        self.monitor.monitoring_cache.clear()
        self.logger.info("市場監視キャッシュをクリアしました")
