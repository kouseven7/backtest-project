"""
DSSMS Phase 4 Task 4.2: EmergencyDetector
階層的緊急事態判定システム

パーフェクトオーダー崩れ時の緊急切替判定
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .perfect_order_detector import PerfectOrderDetector
from .market_condition_monitor import MarketConditionMonitor
from config.logger_config import setup_logger

class EmergencyDetector:
    """階層的緊急事態判定システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス（None時はデフォルト使用）
        """
        self.logger = setup_logger('dssms.emergency_detector')
        
        # 設定ファイル読み込み
        if config_path is None:
            config_path_obj = Path(__file__).parent.parent.parent / "config" / "dssms" / "scheduler_config.json"
        else:
            config_path_obj = Path(config_path)
        
        self.config = self._load_config(config_path_obj)
        
        # 既存コンポーネント初期化
        try:
            self.perfect_order_detector = PerfectOrderDetector()
            self.market_monitor = MarketConditionMonitor()
            self.alert_thresholds = self.config.get('emergency_detection', {}).get('alert_thresholds', {})
            self.logger.info("EmergencyDetector: 初期化完了")
        except Exception as e:
            self.logger.error(f"EmergencyDetector初期化エラー: {e}")
            # フォールバック初期化
            self.perfect_order_detector = None
            self.market_monitor = None
            self.alert_thresholds = {
                "emergency_score_immediate": 40,
                "emergency_score_prepare": 25,
                "emergency_score_monitor": 15
            }
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"設定ファイル読み込み成功: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー {config_path}: {e}")
            return {}
    
    def check_emergency_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        階層的緊急判定（基本PO→詳細分析）
        
        Args:
            symbol: 監視対象銘柄コード
        
        Returns:
            Dict[str, Any]: 緊急判定結果
        """
        emergency_result: Dict[str, Any] = {
            "is_emergency": False,
            "emergency_level": 0,  # 0=正常, 1=注意, 2=警告, 3=緊急
            "trigger_conditions": [],
            "recommended_action": "hold",
            "analysis_details": {},
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Phase 1: 基本パーフェクトオーダーチェック（高速）
            po_status = self._check_perfect_order_status(symbol)
            
            if po_status.get("weekly_breakdown") or po_status.get("monthly_breakdown"):
                emergency_result["emergency_level"] = 2
                emergency_result["trigger_conditions"].append("perfect_order_breakdown")
                
                # Phase 2: 詳細分析（PO崩れ時のみ）
                detailed_analysis = self._perform_detailed_analysis(symbol)
                emergency_result["analysis_details"] = detailed_analysis
                
                # Phase 3: 最終判定
                final_decision = self._make_final_emergency_decision(po_status, detailed_analysis)
                emergency_result.update(final_decision)
            
            elif po_status.get("daily_breakdown"):
                emergency_result["emergency_level"] = 1
                emergency_result["trigger_conditions"].append("daily_perfect_order_breakdown")
                emergency_result["recommended_action"] = "observe"
            
            return emergency_result
            
        except Exception as e:
            self.logger.error(f"緊急判定エラー {symbol}: {e}")
            emergency_result["error"] = str(e)
            return emergency_result
    
    def _check_perfect_order_status(self, symbol: str) -> Dict[str, Any]:
        """パーフェクトオーダー状況チェック"""
        if self.perfect_order_detector is None:
            return {"error": "PerfectOrderDetector not available"}
        
        try:
            # 簡略版実装（実際の分析は他のコンポーネントのデータが必要）
            po_result = {"error": "Simplified implementation - requires actual data"}
            return po_result
        except Exception as e:
            self.logger.error(f"パーフェクトオーダーチェックエラー {symbol}: {e}")
            return {"error": str(e)}
    
    def _perform_detailed_analysis(self, symbol: str) -> Dict[str, Any]:
        """詳細分析実行"""
        analysis = {
            "volume_analysis": self._analyze_volume_pattern(symbol),
            "volatility_analysis": self._analyze_volatility_spike(symbol),
            "market_correlation": self._analyze_market_correlation(symbol),
            "technical_indicators": self._analyze_technical_breakdown(symbol)
        }
        return analysis
    
    def _analyze_volume_pattern(self, symbol: str) -> Dict[str, Any]:
        """出来高パターン分析"""
        try:
            # 簡略版実装（実際のデータ分析は省略）
            return {
                "abnormal_volume": False,
                "volume_spike_ratio": 1.0,
                "average_volume_comparison": "normal"
            }
        except Exception as e:
            self.logger.error(f"出来高分析エラー {symbol}: {e}")
            return {"error": str(e)}
    
    def _analyze_volatility_spike(self, symbol: str) -> Dict[str, Any]:
        """ボラティリティ急上昇分析"""
        try:
            # 簡略版実装
            return {
                "volatility_spike": False,
                "volatility_ratio": 1.0,
                "spike_threshold": 2.0
            }
        except Exception as e:
            self.logger.error(f"ボラティリティ分析エラー {symbol}: {e}")
            return {"error": str(e)}
    
    def _analyze_market_correlation(self, symbol: str) -> Dict[str, Any]:
        """市場相関分析"""
        try:
            if self.market_monitor is None:
                return {"error": "MarketConditionMonitor not available"}
            
            # 市場全体の健康度チェック
            market_health = self.market_monitor.get_market_health_score()
            
            return {
                "correlation_breakdown": market_health < 0.3,
                "market_health_score": market_health,
                "correlation_coefficient": 0.7  # 仮の値
            }
        except Exception as e:
            self.logger.error(f"市場相関分析エラー {symbol}: {e}")
            return {"error": str(e)}
    
    def _analyze_technical_breakdown(self, symbol: str) -> Dict[str, Any]:
        """テクニカル指標分析"""
        try:
            # 簡略版実装
            return {
                "rsi_oversold": False,
                "macd_divergence": False,
                "support_level_breakdown": False
            }
        except Exception as e:
            self.logger.error(f"テクニカル分析エラー {symbol}: {e}")
            return {"error": str(e)}
    
    def _make_final_emergency_decision(self, po_status: Dict[str, Any], detailed: Dict[str, Any]) -> Dict[str, Any]:
        """最終緊急判定決定"""
        decision: Dict[str, Any] = {
            "is_emergency": False,
            "recommended_action": "hold"
        }
        
        try:
            # 複合判定ロジック
            emergency_score = 0
            
            # パーフェクトオーダー崩れの重要度
            if po_status.get("monthly_breakdown"):
                emergency_score += 30
            if po_status.get("weekly_breakdown"):
                emergency_score += 20
            
            # 出来高異常
            if detailed.get("volume_analysis", {}).get("abnormal_volume"):
                emergency_score += 15
            
            # ボラティリティ急上昇
            if detailed.get("volatility_analysis", {}).get("volatility_spike"):
                emergency_score += 10
            
            # 市場全体との相関悪化
            if detailed.get("market_correlation", {}).get("correlation_breakdown"):
                emergency_score += 10
            
            # 判定基準
            if emergency_score >= self.alert_thresholds.get("emergency_score_immediate", 40):
                decision["is_emergency"] = True
                decision["recommended_action"] = "immediate_switch"
            elif emergency_score >= self.alert_thresholds.get("emergency_score_prepare", 25):
                decision["is_emergency"] = True
                decision["recommended_action"] = "prepare_switch"
            elif emergency_score >= self.alert_thresholds.get("emergency_score_monitor", 15):
                decision["recommended_action"] = "close_monitoring"
            
            decision["emergency_score"] = emergency_score
            return decision
            
        except Exception as e:
            self.logger.error(f"最終判定エラー: {e}")
            return decision
    
    def get_emergency_threshold_config(self) -> Dict[str, Any]:
        """緊急判定閾値設定取得"""
        return self.alert_thresholds.copy()
    
    def update_emergency_thresholds(self, new_thresholds: Dict[str, Any]) -> bool:
        """緊急判定閾値更新"""
        try:
            self.alert_thresholds.update(new_thresholds)
            self.logger.info(f"緊急判定閾値更新: {new_thresholds}")
            return True
        except Exception as e:
            self.logger.error(f"緊急判定閾値更新エラー: {e}")
            return False
