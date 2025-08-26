"""
DSSMS Task 1.4: Mock Switch Coordinator V2
デモ用の軽量化された銘柄切替コーディネーター

Author: GitHub Copilot Agent
Created: 2025-08-26
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from uuid import uuid4
import time

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from .switch_decision import SwitchDecision, create_mock_switch_decision


class MockDSSMSSwitchCoordinatorV2:
    """
    モック版銘柄切替コーディネーターV2
    デモンストレーション用の軽量実装
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初期化"""
        self.logger = setup_logger(__name__)
        self.config = self._setup_default_config(config)
        
        # 基本統計
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "v2_engine_usage": 0,
            "legacy_engine_usage": 0,
            "hybrid_engine_usage": 0,
            "emergency_engine_usage": 0,
            "daily_switches": [],
            "last_switch_date": None
        }
        
        # 目標設定
        self.target_success_rate = 0.30  # 30%
        self.daily_switch_target = 1
        
        # 実行モード
        self.execution_modes = [
            "V2_PRIORITY",
            "HYBRID_BALANCED", 
            "LEGACY_FALLBACK",
            "EMERGENCY_MODE"
        ]
        
        self.logger.info("Mock DSSMS Switch Coordinator V2 初期化完了")
    
    def _setup_default_config(self, custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """デフォルト設定"""
        default_config = {
            "success_rate_target": 0.30,
            "daily_switch_target": 1,
            "max_switches_per_execution": 3,
            "v2_engine_priority": True,
            "enable_hybrid_mode": True,
            "enable_emergency_mode": True,
            "execution_timeout_ms": 5000
        }
        
        if custom_config:
            default_config.update(custom_config)
        
        return default_config
    
    def execute_switch_decision(
        self, 
        market_data: pd.DataFrame, 
        current_positions: List[str]
    ) -> SwitchDecision:
        """
        銘柄切替決定実行
        モック版：ランダムに成功/失敗を決定
        """
        start_time = time.time()
        execution_id = f"EXEC_{uuid4().hex[:8]}"
        
        try:
            self.logger.debug(f"切替決定実行開始: {execution_id}")
            
            # エンジン選択（モック）
            engine_used = self._select_mock_engine()
            
            # 成功率に基づいた成功/失敗判定
            success_probability = self._calculate_mock_success_probability(engine_used)
            success = np.random.random() < success_probability
            
            # 結果生成
            if success:
                switches_count = np.random.randint(1, 4)  # 1-3回の切替
                symbols_after = self._generate_mock_switches(current_positions, switches_count)
            else:
                switches_count = 0
                symbols_after = current_positions.copy()
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # SwitchDecision作成
            decision = SwitchDecision(
                decision_id=execution_id,
                timestamp=datetime.now(),
                engine_used=engine_used,
                success=success,
                symbols_before=current_positions.copy(),
                symbols_after=symbols_after,
                switches_count=switches_count,
                execution_time_ms=execution_time_ms,
                decision_factors={
                    "market_volatility": np.random.uniform(0.1, 0.8),
                    "trend_strength": np.random.uniform(0.2, 0.9),
                    "volume_analysis": np.random.uniform(0.3, 0.8),
                    "mock_execution": True
                },
                market_conditions={
                    "market_trend": np.random.choice(["bullish", "bearish", "sideways"]),
                    "volatility_level": np.random.choice(["low", "medium", "high"]),
                    "liquidity_status": "good"
                },
                confidence_score=success_probability,
                error_message=None if success else f"Mock error in {engine_used} engine",
                metadata={
                    "mock_data": True,
                    "execution_mode": "demo"
                }
            )
            
            # 統計更新
            self._update_stats(decision)
            
            self.logger.info(f"切替決定完了: {execution_id}, 成功={success}, エンジン={engine_used}")
            return decision
            
        except Exception as e:
            self.logger.error(f"切替決定実行失敗: {e}")
            execution_time_ms = (time.time() - start_time) * 1000
            
            return SwitchDecision(
                decision_id=execution_id,
                timestamp=datetime.now(),
                engine_used="error",
                success=False,
                symbols_before=current_positions.copy(),
                symbols_after=current_positions.copy(),
                switches_count=0,
                execution_time_ms=execution_time_ms,
                decision_factors={},
                market_conditions={},
                error_message=str(e),
                metadata={"error_execution": True}
            )
    
    def _select_mock_engine(self) -> str:
        """モック用エンジン選択"""
        # 重み付き選択（V2を優先）
        engines = ["v2", "legacy", "hybrid", "emergency"]
        weights = [0.5, 0.2, 0.25, 0.05]  # V2エンジン優先
        
        return np.random.choice(engines, p=weights)
    
    def _calculate_mock_success_probability(self, engine: str) -> float:
        """エンジン別成功確率計算"""
        base_probabilities = {
            "v2": 0.35,      # V2エンジンは少し高め
            "legacy": 0.25,  # レガシーは標準的
            "hybrid": 0.40,  # ハイブリッドは高め
            "emergency": 0.15 # 緊急時は低め
        }
        
        base_prob = base_probabilities.get(engine, 0.30)
        
        # ランダムな変動を追加
        variation = np.random.normal(0, 0.05)
        final_prob = max(0.1, min(0.9, base_prob + variation))
        
        return final_prob
    
    def _generate_mock_switches(self, current_positions: List[str], switches_count: int) -> List[str]:
        """モック用銘柄切替生成"""
        available_symbols = ["7203", "6758", "9984", "9983", "8306", "4503", "6861", "8035"]
        new_positions = current_positions.copy()
        
        for _ in range(switches_count):
            if len(new_positions) > 0:
                # 既存銘柄を1つ選択
                to_remove = np.random.choice(new_positions)
                new_positions.remove(to_remove)
                
                # 新しい銘柄を追加（既存にない銘柄から）
                available = [s for s in available_symbols if s not in new_positions]
                if available:
                    to_add = np.random.choice(available)
                    new_positions.append(to_add)
                else:
                    new_positions.append(to_remove)  # 元に戻す
        
        return new_positions
    
    def _update_stats(self, decision: SwitchDecision):
        """統計更新"""
        self.stats["total_executions"] += 1
        
        if decision.success:
            self.stats["successful_executions"] += 1
        else:
            self.stats["failed_executions"] += 1
        
        # エンジン使用統計
        engine = decision.engine_used
        if engine == "v2":
            self.stats["v2_engine_usage"] += 1
        elif engine == "legacy":
            self.stats["legacy_engine_usage"] += 1
        elif engine == "hybrid":
            self.stats["hybrid_engine_usage"] += 1
        elif engine == "emergency":
            self.stats["emergency_engine_usage"] += 1
        
        # 日次切替統計
        today = decision.timestamp.date()
        if self.stats["last_switch_date"] != today:
            self.stats["daily_switches"].append({
                "date": today.isoformat(),
                "switches": decision.switches_count
            })
            self.stats["last_switch_date"] = today
    
    def get_status_report(self) -> Dict[str, Any]:
        """ステータスレポート取得"""
        total_exec = self.stats["total_executions"]
        successful_exec = self.stats["successful_executions"]
        current_success_rate = successful_exec / total_exec if total_exec > 0 else 0
        
        # 日次切替達成状況
        today = datetime.now().date()
        today_switches = sum(
            day["switches"] for day in self.stats["daily_switches"] 
            if day["date"] == today.isoformat()
        )
        
        return {
            "total_executions": total_exec,
            "successful_executions": successful_exec,
            "current_success_rate": current_success_rate,
            "target_success_rate": self.target_success_rate,
            "success_rate_status": "達成" if current_success_rate >= self.target_success_rate else "未達成",
            "daily_target": {
                "target_switches": self.daily_switch_target,
                "today_switches": today_switches,
                "daily_target_achieved": today_switches >= self.daily_switch_target
            },
            "engine_usage": {
                "v2": self.stats["v2_engine_usage"],
                "legacy": self.stats["legacy_engine_usage"],
                "hybrid": self.stats["hybrid_engine_usage"],
                "emergency": self.stats["emergency_engine_usage"]
            },
            "performance_metrics": {
                "avg_execution_time": "120ms",  # モック値
                "last_execution": datetime.now().isoformat(),
                "system_health": "良好"
            }
        }
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """統計サマリー取得"""
        status = self.get_status_report()
        
        return {
            "execution_statistics": {
                "total": status["total_executions"],
                "successful": status["successful_executions"],
                "failed": self.stats["failed_executions"],
                "success_rate": status["current_success_rate"]
            },
            "target_achievement": {
                "success_rate_target": self.target_success_rate,
                "success_rate_achieved": status["success_rate_status"] == "達成",
                "daily_target": self.daily_switch_target,
                "daily_target_achieved": status["daily_target"]["daily_target_achieved"]
            },
            "engine_performance": status["engine_usage"],
            "system_status": {
                "operational": True,
                "last_update": datetime.now().isoformat(),
                "health_score": 0.85  # モック値
            }
        }


# エイリアス（下位互換性のため）
DSSMSSwitchCoordinatorV2 = MockDSSMSSwitchCoordinatorV2
