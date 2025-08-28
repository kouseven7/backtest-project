"""
DSSMS パフォーマンス目標管理システム
Task 3.4: パフォーマンス目標達成確認の中核コンポーネント
"""
import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class TargetPhase(Enum):
    EMERGENCY = "emergency"
    BASIC = "basic"
    OPTIMIZATION = "optimization"

class AchievementLevel(Enum):
    FAILED = "failed"
    MINIMUM = "minimum"
    TARGET = "target"
    STRETCH = "stretch"

@dataclass
class TargetResult:
    metric_name: str
    value: float
    target_value: float
    minimum_value: Optional[float]
    stretch_value: Optional[float]
    achievement_level: AchievementLevel
    phase: TargetPhase
    description: str

class PerformanceTargetManager:
    """パフォーマンス目標の管理と評価を行うクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config/dssms/performance_targets.json"
        self.targets = self._load_targets()
        self.current_phase = TargetPhase(self.targets["evaluation_settings"]["target_phases"]["current_phase"])
        
    def _load_targets(self) -> Dict[str, Any]:
        """目標設定を読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"目標設定ファイルが見つかりません: {self.config_path}")
            return self._get_default_targets()
        except Exception as e:
            self.logger.error(f"目標設定の読み込みエラー: {e}")
            return self._get_default_targets()
    
    def _get_default_targets(self) -> Dict[str, Any]:
        """デフォルト目標設定"""
        return {
            "emergency_targets": {
                "total_return": {"minimum": -50.0, "target": -20.0},
                "portfolio_value": {"minimum": 500000.0, "target": 800000.0},
                "switching_success_rate": {"minimum": 0.1, "target": 0.3},
                "max_drawdown": {"maximum": 80.0, "target": 60.0}
            },
            "evaluation_settings": {
                "target_phases": {"current_phase": "emergency"}
            }
        }
    
    def evaluate_metrics(self, performance_data: Dict[str, float]) -> List[TargetResult]:
        """パフォーマンス指標を目標と比較評価"""
        results = []
        phase_targets = self._get_current_phase_targets()
        
        for metric_name, metric_value in performance_data.items():
            if metric_name in phase_targets:
                result = self._evaluate_single_metric(
                    metric_name, metric_value, phase_targets[metric_name]
                )
                if result:
                    results.append(result)
        
        return results
    
    def _get_current_phase_targets(self) -> Dict[str, Any]:
        """現在のフェーズの目標を取得"""
        phase_key = f"{self.current_phase.value}_targets"
        return self.targets.get(phase_key, {})
    
    def _evaluate_single_metric(self, metric_name: str, value: float, target_config: Dict[str, Any]) -> Optional[TargetResult]:
        """単一指標の評価"""
        try:
            # 目標値の取得
            target_value = target_config.get("target")
            minimum_value = target_config.get("minimum")
            maximum_value = target_config.get("maximum")
            stretch_value = target_config.get("stretch")
            description = target_config.get("description", "")
            
            # 達成レベルの判定
            achievement_level = self._determine_achievement_level(
                value, target_value, minimum_value, maximum_value, stretch_value
            )
            
            return TargetResult(
                metric_name=metric_name,
                value=value,
                target_value=float(target_value or maximum_value or 0.0),
                minimum_value=float(minimum_value) if minimum_value is not None else None,
                stretch_value=float(stretch_value) if stretch_value is not None else None,
                achievement_level=achievement_level,
                phase=self.current_phase,
                description=description
            )
            
        except Exception as e:
            self.logger.error(f"指標評価エラー {metric_name}: {e}")
            return None
    
    def _determine_achievement_level(self, value: float, target: Optional[float], 
                                   minimum: Optional[float], maximum: Optional[float],
                                   stretch: Optional[float]) -> AchievementLevel:
        """達成レベルの判定ロジック"""
        
        # 最大値制約がある場合（ドローダウンなど、小さい方が良い指標）
        if maximum is not None:
            if stretch and value <= stretch:
                return AchievementLevel.STRETCH
            elif target and value <= target:
                return AchievementLevel.TARGET
            elif value <= maximum:
                return AchievementLevel.MINIMUM
            else:
                return AchievementLevel.FAILED
        
        # 最小値制約がある場合（リターンなど、大きい方が良い指標）
        else:
            if stretch and value >= stretch:
                return AchievementLevel.STRETCH
            elif target and value >= target:
                return AchievementLevel.TARGET
            elif minimum and value >= minimum:
                return AchievementLevel.MINIMUM
            else:
                return AchievementLevel.FAILED
    
    def get_phase_summary(self, results: List[TargetResult]) -> Dict:
        """フェーズ全体のサマリー生成"""
        if not results:
            return {"overall_status": "no_data", "progress": 0.0}
        
        # 達成レベル別の集計
        level_counts = {}
        for level in AchievementLevel:
            level_counts[level.value] = sum(1 for r in results if r.achievement_level == level)
        
        total_metrics = len(results)
        success_count = sum(1 for r in results if r.achievement_level != AchievementLevel.FAILED)
        progress = success_count / total_metrics if total_metrics > 0 else 0.0
        
        # 全体ステータスの判定
        if level_counts[AchievementLevel.FAILED.value] == 0:
            if level_counts[AchievementLevel.STRETCH.value] > total_metrics * 0.7:
                overall_status = "excellent"
            elif level_counts[AchievementLevel.TARGET.value] > total_metrics * 0.7:
                overall_status = "good"
            else:
                overall_status = "acceptable"
        elif level_counts[AchievementLevel.FAILED.value] < total_metrics * 0.3:
            overall_status = "needs_improvement"
        else:
            overall_status = "critical"
        
        return {
            "overall_status": overall_status,
            "progress": progress,
            "total_metrics": total_metrics,
            "success_count": success_count,
            "failed_count": level_counts[AchievementLevel.FAILED.value],
            "level_distribution": level_counts,
            "current_phase": self.current_phase.value
        }
    
    def suggest_next_phase(self, results: List[TargetResult]) -> Tuple[bool, str]:
        """次のフェーズへの移行提案"""
        summary = self.get_phase_summary(results)
        
        # 現在のフェーズでの成功率が80%以上なら次のフェーズを提案
        if summary["progress"] >= 0.8 and summary["overall_status"] in ["good", "excellent"]:
            progression = self.targets["evaluation_settings"]["target_phases"]["progression"]
            current_index = progression.index(self.current_phase.value)
            
            if current_index < len(progression) - 1:
                next_phase = progression[current_index + 1]
                return True, f"次のフェーズ '{next_phase}' への移行を推奨します"
            else:
                return False, "最終フェーズに到達しています"
        
        return False, f"現在のフェーズ '{self.current_phase.value}' での改善が必要です"
    
    def update_current_phase(self, new_phase: TargetPhase) -> bool:
        """現在のフェーズを更新"""
        try:
            self.current_phase = new_phase
            self.targets["evaluation_settings"]["target_phases"]["current_phase"] = new_phase.value
            
            # 設定ファイルの更新
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.targets, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"フェーズを '{new_phase.value}' に更新しました")
            return True
            
        except Exception as e:
            self.logger.error(f"フェーズ更新エラー: {e}")
            return False
