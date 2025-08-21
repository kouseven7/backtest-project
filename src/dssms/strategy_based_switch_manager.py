"""
Module: DSSMS Strategy-Based Switch Manager
File: strategy_based_switch_manager.py
Description: 
  戦略固有の保有期間とスイッチング条件を管理するDSSMSスイッチマネージャーです。
  7つの戦略それぞれに最適化された保有期間を設定し、バランスの取れたアプローチで
  過度なスイッチングを防ぎ、取引コストを削減しながら利益機会を最大化します。

Author: GitHub Copilot
Created: 2025-01-23
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd

# プロジェクトのルートディレクトリを sys.path に追加
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.logger_config import setup_logger

# ロガーの設定
logger = setup_logger(__name__)

class StrategyBasedSwitchManager:
    """戦略固有の保有期間を考慮したスイッチング管理システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        戦略ベーススイッチマネージャーの初期化
        
        Parameters:
            config_path (str): 設定ファイルのパス
        """
        self.config_path = config_path or r"C:\Users\imega\Documents\my_backtest_project\config\dssms\intelligent_switch_config.json"
        self.config = self._load_config()
        
        # 現在の保有状況
        self.current_position = None
        self.position_entry_time = None
        self.position_strategy = None
        self.position_stock = None
        
        # スイッチング履歴
        self.switch_history: List[Dict[str, Any]] = []
        self.daily_switches = 0
        self.weekly_switches = 0
        self.last_switch_time = None
        
        # 緊急停止状態
        self.emergency_stop_active = False
        self.emergency_reason = None
        
        logger.info("戦略ベーススイッチマネージャーを初期化しました")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"設定ファイルの読み込みに失敗: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "switch_criteria": {
                "perfect_order_breakdown_threshold": 0.7,
                "score_difference_threshold": 0.25,
                "minimum_holding_period_hours": 24,
                "confidence_threshold": 0.75,
                "strategy_specific_holding_periods": {
                    "Opening_Gap": 4,
                    "VWAP_Breakout": 12,
                    "Breakout": 12,
                    "VWAP_Bounce": 6,
                    "Momentum_Investing": 48,
                    "Contrarian": 8,
                    "GC_Strategy": 24
                }
            },
            "risk_control": {
                "max_daily_switches": 2,
                "max_weekly_switches": 6,
                "transaction_cost_awareness": {
                    "cost_per_switch": 0.003,
                    "max_daily_cost_ratio": 0.01
                }
            }
        }
    
    def can_switch(self, current_time: datetime, new_strategy: str, current_strategy: Optional[str] = None) -> tuple[bool, str]:
        """
        スイッチング可能かを判定
        
        Parameters:
            current_time (datetime): 現在時刻
            new_strategy (str): 新戦略名
            current_strategy (str): 現在の戦略名
            
        Returns:
            tuple[bool, str]: (可能かどうか, 理由)
        """
        # 緊急停止中の場合
        if self.emergency_stop_active:
            return False, f"緊急停止中: {self.emergency_reason}"
        
        # ポジションがない場合は常にスイッチ可能
        if not self.current_position:
            return True, "新規ポジション"
        
        # 戦略固有の最小保有期間をチェック
        strategy_holding_periods = self.config["switch_criteria"]["strategy_specific_holding_periods"]
        min_holding_hours = strategy_holding_periods.get(current_strategy or self.position_strategy, 24)
        
        if self.position_entry_time:
            holding_duration = current_time - self.position_entry_time
            if holding_duration.total_seconds() / 3600 < min_holding_hours:
                return False, f"最小保有期間未達: {min_holding_hours}時間必要 (現在{holding_duration.total_seconds()/3600:.1f}時間)"
        
        # 1日のスイッチング回数制限
        max_daily = self.config["risk_control"]["max_daily_switches"]
        if self.daily_switches >= max_daily:
            return False, f"1日の最大スイッチング回数に達しました: {max_daily}回"
        
        # 1週間のスイッチング回数制限
        max_weekly = self.config["risk_control"]["max_weekly_switches"]
        if self.weekly_switches >= max_weekly:
            return False, f"1週間の最大スイッチング回数に達しました: {max_weekly}回"
        
        # クールダウン期間チェック
        cooldown_minutes = self.config["risk_control"].get("switch_frequency_cooldown_minutes", 60)
        if self.last_switch_time:
            time_since_last = current_time - self.last_switch_time
            if time_since_last.total_seconds() / 60 < cooldown_minutes:
                return False, f"クールダウン期間中: {cooldown_minutes}分待機必要"
        
        # 取引コスト考慮
        cost_config = self.config["risk_control"].get("transaction_cost_awareness", {})
        cost_per_switch = cost_config.get("cost_per_switch", 0.003)
        max_daily_cost_ratio = cost_config.get("max_daily_cost_ratio", 0.01)
        
        if self.daily_switches * cost_per_switch >= max_daily_cost_ratio:
            return False, f"1日の取引コスト上限に達しました: {max_daily_cost_ratio*100}%"
        
        return True, "スイッチング条件を満たしました"
    
    def should_emergency_exit(self, market_data: Dict[str, Any], position_data: Dict[str, Any]) -> tuple[bool, str]:
        """
        緊急退場が必要かを判定
        
        Parameters:
            market_data (dict): 市場データ
            position_data (dict): ポジションデータ
            
        Returns:
            tuple[bool, str]: (緊急退場が必要か, 理由)
        """
        # パーフェクトオーダー破綻チェック
        perfect_order_threshold = self.config["switch_criteria"]["perfect_order_breakdown_threshold"]
        if market_data.get("perfect_order_score", 1.0) < perfect_order_threshold:
            return True, "パーフェクトオーダー破綻"
        
        # ストップロス条件
        if position_data.get("unrealized_pnl_ratio", 0) < -0.05:  # 5%損失
            return True, "ストップロス発動"
        
        # 市場クラッシュ検知
        market_crash_threshold = self.config["risk_control"]["emergency_stop_conditions"]["market_crash_threshold"]
        if market_data.get("market_change_ratio", 0) < market_crash_threshold:
            return True, "市場クラッシュ検知"
        
        # ボラティリティスパイク
        volatility_multiplier = self.config["risk_control"]["emergency_stop_conditions"]["volatility_spike_multiplier"]
        if market_data.get("volatility_ratio", 1.0) > volatility_multiplier:
            return True, "ボラティリティスパイク"
        
        return False, ""
    
    def execute_switch(self, current_time: datetime, new_stock: str, new_strategy: str, 
                      switch_reason: str = "一般スイッチング") -> bool:
        """
        スイッチングを実行
        
        Parameters:
            current_time (datetime): 実行時刻
            new_stock (str): 新銘柄
            new_strategy (str): 新戦略
            switch_reason (str): スイッチング理由
            
        Returns:
            bool: 実行成功かどうか
        """
        try:
            # スイッチング可能性チェック
            can_switch, reason = self.can_switch(current_time, new_strategy, self.position_strategy)
            if not can_switch:
                logger.warning(f"スイッチング拒否: {reason}")
                return False
            
            # スイッチング実行
            old_position: Dict[str, Any] = {
                "stock": self.position_stock,
                "strategy": self.position_strategy,
                "entry_time": self.position_entry_time
            }
            
            # 新ポジション設定
            self.current_position = True
            self.position_entry_time = current_time
            self.position_strategy = new_strategy
            self.position_stock = new_stock
            
            # スイッチング履歴記録
            switch_record: Dict[str, Any] = {
                "timestamp": current_time,
                "from_stock": old_position["stock"],
                "from_strategy": old_position["strategy"],
                "to_stock": new_stock,
                "to_strategy": new_strategy,
                "reason": switch_reason
            }
            self.switch_history.append(switch_record)
            
            # カウンター更新
            self.daily_switches += 1
            self.weekly_switches += 1
            self.last_switch_time = current_time
            
            logger.info(f"スイッチング実行: {old_position['stock']}({old_position['strategy']}) → {new_stock}({new_strategy})")
            logger.info(f"理由: {switch_reason}")
            
            return True
            
        except Exception as e:
            logger.error(f"スイッチング実行エラー: {e}")
            return False
    
    def close_position(self, current_time: datetime, reason: str = "通常終了") -> bool:
        """
        ポジションを閉じる
        
        Parameters:
            current_time (datetime): 実行時刻
            reason (str): 終了理由
            
        Returns:
            bool: 実行成功かどうか
        """
        try:
            if not self.current_position:
                logger.warning("閉じるポジションがありません")
                return False
            
            # ポジション履歴記録
            position_record: Dict[str, Any] = {
                "timestamp": current_time,
                "stock": self.position_stock,
                "strategy": self.position_strategy,
                "entry_time": self.position_entry_time,
                "exit_time": current_time,
                "holding_duration": current_time - self.position_entry_time if self.position_entry_time else None,
                "exit_reason": reason
            }
            
            # ポジション状態リセット
            self.current_position = None
            self.position_entry_time = None
            self.position_strategy = None
            self.position_stock = None
            
            logger.info(f"ポジション終了: {position_record['stock']}({position_record['strategy']})")
            logger.info(f"保有期間: {position_record['holding_duration']}, 理由: {reason}")
            
            return True
            
        except Exception as e:
            logger.error(f"ポジション終了エラー: {e}")
            return False
    
    def reset_daily_counters(self):
        """1日のカウンターをリセット"""
        self.daily_switches = 0
        logger.debug("1日のスイッチングカウンターをリセットしました")
    
    def reset_weekly_counters(self):
        """1週間のカウンターをリセット"""
        self.weekly_switches = 0
        logger.debug("1週間のスイッチングカウンターをリセットしました")
    
    def activate_emergency_stop(self, reason: str):
        """緊急停止を有効化"""
        self.emergency_stop_active = True
        self.emergency_reason = reason
        logger.warning(f"緊急停止を有効化: {reason}")
    
    def deactivate_emergency_stop(self):
        """緊急停止を無効化"""
        self.emergency_stop_active = False
        self.emergency_reason = None
        logger.info("緊急停止を無効化しました")
    
    def get_current_status(self) -> Dict[str, Any]:
        """現在の状態を取得"""
        return {
            "has_position": bool(self.current_position),
            "current_stock": self.position_stock,
            "current_strategy": self.position_strategy,
            "position_entry_time": self.position_entry_time,
            "daily_switches": self.daily_switches,
            "weekly_switches": self.weekly_switches,
            "emergency_stop_active": self.emergency_stop_active,
            "emergency_reason": self.emergency_reason,
            "switch_history_count": len(self.switch_history)
        }
    
    def get_switch_statistics(self) -> Dict[str, Any]:
        """スイッチング統計を取得"""
        if not self.switch_history:
            return {"total_switches": 0}
        
        # 戦略別スイッチング回数
        strategy_switches = {}
        for switch in self.switch_history:
            from_strategy = switch["from_strategy"]
            to_strategy = switch["to_strategy"]
            
            if from_strategy:
                strategy_switches[from_strategy] = strategy_switches.get(from_strategy, 0) + 1
        
        # 最も多いスイッチング理由
        reasons = [switch["reason"] for switch in self.switch_history]
        reason_counts = {reason: reasons.count(reason) for reason in set(reasons)}
        
        return {
            "total_switches": len(self.switch_history),
            "daily_switches": self.daily_switches,
            "weekly_switches": self.weekly_switches,
            "strategy_switches": strategy_switches,
            "switch_reasons": reason_counts,
            "emergency_stops": sum(1 for switch in self.switch_history if "緊急" in switch["reason"])
        }

# テスト関数
def test_strategy_based_switch_manager():
    """戦略ベーススイッチマネージャーのテスト"""
    logger.info("戦略ベーススイッチマネージャーのテストを開始します")
    
    try:
        # インスタンス作成
        manager = StrategyBasedSwitchManager()
        
        # 現在時刻
        current_time = datetime.now()
        
        # 1. 新規ポジション（スイッチング可能）
        can_switch, reason = manager.can_switch(current_time, "Opening_Gap")
        logger.info(f"新規ポジション判定: {can_switch}, 理由: {reason}")
        
        # 2. ポジション開始
        success = manager.execute_switch(current_time, "7203", "Opening_Gap", "新規エントリー")
        logger.info(f"ポジション開始: {success}")
        
        # 3. すぐにスイッチング試行（Opening_Gapは4時間保有必要）
        can_switch, reason = manager.can_switch(current_time + timedelta(hours=2), "VWAP_Breakout", "Opening_Gap")
        logger.info(f"早期スイッチング判定: {can_switch}, 理由: {reason}")
        
        # 4. 4時間後にスイッチング試行
        can_switch, reason = manager.can_switch(current_time + timedelta(hours=5), "VWAP_Breakout", "Opening_Gap")
        logger.info(f"適切なタイミングでのスイッチング判定: {can_switch}, 理由: {reason}")
        
        # 5. 実際にスイッチング実行
        success = manager.execute_switch(current_time + timedelta(hours=5), "6758", "VWAP_Breakout", "より良い機会")
        logger.info(f"スイッチング実行: {success}")
        
        # 6. 緊急退場条件テスト
        market_data = {"perfect_order_score": 0.5, "market_change_ratio": -0.04}
        position_data = {"unrealized_pnl_ratio": -0.03}
        should_exit, exit_reason = manager.should_emergency_exit(market_data, position_data)
        logger.info(f"緊急退場判定: {should_exit}, 理由: {exit_reason}")
        
        # 7. 統計情報表示
        status = manager.get_current_status()
        statistics = manager.get_switch_statistics()
        logger.info(f"現在の状態: {status}")
        logger.info(f"スイッチング統計: {statistics}")
        
        logger.info("戦略ベーススイッチマネージャーのテストが完了しました")
        
    except Exception as e:
        logger.error(f"テスト中にエラーが発生: {e}")
        raise

if __name__ == "__main__":
    test_strategy_based_switch_manager()
