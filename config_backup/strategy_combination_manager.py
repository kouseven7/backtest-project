"""
Module: Strategy Combination Manager
File: strategy_combination_manager.py
Description: 
  4-2-2「複合戦略バックテスト機能実装」- Strategy Combination Manager
  戦略組み合わせの管理と最適化

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 戦略組み合わせの読み込み・管理
  - 動的重み調整
  - リスク制約の適用
  - パフォーマンス最適化
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ロガーの設定
logger = logging.getLogger(__name__)

class RebalancingFrequency(Enum):
    """リバランス頻度"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    TREND_CHANGE_BASED = "trend_change_based"
    MARKET_REGIME_BASED = "market_regime_based"
    PERFORMANCE_BASED = "performance_based"

@dataclass
class StrategyConfig:
    """戦略設定"""
    strategy_name: str
    weight: float
    conditions: Dict[str, Any] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # 重みの範囲チェック
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {self.weight}")

@dataclass
class CombinationConfig:
    """戦略組み合わせ設定"""
    combination_id: str
    name: str
    description: str = ""
    strategies: List[StrategyConfig] = field(default_factory=list)
    rebalancing_rules: Dict[str, Any] = field(default_factory=dict)
    risk_management: Dict[str, float] = field(default_factory=dict)
    expected_performance: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # 重みの合計チェック
        total_weight = sum(s.weight for s in self.strategies)
        if abs(total_weight - 1.0) > 0.01:  # 1%の許容範囲
            logger.warning(f"Total weight {total_weight:.3f} != 1.0 for combination {self.combination_id}")

@dataclass
class RebalancingEvent:
    """リバランス イベント"""
    event_id: str
    combination_id: str
    timestamp: datetime
    trigger_reason: str
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    market_conditions: Dict[str, Any]
    confidence_score: float

class StrategyCombinationManager:
    """戦略組み合わせマネージャー"""
    
    def __init__(self, config_path: Optional[str] = None):
        """マネージャーの初期化"""
        self.logger = logging.getLogger(__name__)
        
        # 設定の読み込み
        self.combinations_config = self._load_combinations_config(config_path)
        self.global_settings = self.combinations_config.get('global_settings', {})
        
        # 戦略組み合わせの管理
        self.active_combinations: Dict[str, CombinationConfig] = {}
        self.rebalancing_history: List[RebalancingEvent] = []
        
        # パフォーマンス監視
        self.performance_tracking: Dict[str, Dict[str, Any]] = {}
        
        # 戦略組み合わせの読み込み
        self._initialize_combinations()
        
        self.logger.info("StrategyCombinationManager initialized")
    
    def _load_combinations_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """戦略組み合わせ設定の読み込み"""
        
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "backtest", 
                "strategy_combinations.json"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Strategy combinations loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load combinations config: {e}")
            return self._get_default_combinations_config()
    
    def _get_default_combinations_config(self) -> Dict[str, Any]:
        """デフォルト戦略組み合わせ設定"""
        
        return {
            "combination_sets": [
                {
                    "combination_id": "default_balanced",
                    "name": "デフォルトバランス戦略",
                    "description": "バランスの取れたデフォルト戦略組み合わせ",
                    "strategies": [
                        {
                            "strategy_name": "VWAP_Breakout",
                            "weight": 0.5,
                            "conditions": {},
                            "risk_limits": {"max_position_size": 0.3}
                        },
                        {
                            "strategy_name": "Momentum_Investing", 
                            "weight": 0.5,
                            "conditions": {},
                            "risk_limits": {"max_position_size": 0.3}
                        }
                    ],
                    "rebalancing_rules": {
                        "frequency": "monthly"
                    }
                }
            ],
            "global_settings": {
                "max_concurrent_strategies": 5,
                "position_size_limits": {
                    "max_single_strategy_weight": 0.6,
                    "min_strategy_weight": 0.1
                }
            }
        }
    
    def _initialize_combinations(self):
        """戦略組み合わせの初期化"""
        
        combination_sets = self.combinations_config.get('combination_sets', [])
        
        for combo_data in combination_sets:
            try:
                # 戦略設定の変換
                strategies = []
                for strategy_data in combo_data.get('strategies', []):
                    strategy_config = StrategyConfig(
                        strategy_name=strategy_data['strategy_name'],
                        weight=strategy_data['weight'],
                        conditions=strategy_data.get('conditions', {}),
                        risk_limits=strategy_data.get('risk_limits', {}),
                        performance_targets=strategy_data.get('performance_targets', {})
                    )
                    strategies.append(strategy_config)
                
                # 組み合わせ設定の作成
                combination = CombinationConfig(
                    combination_id=combo_data['combination_id'],
                    name=combo_data['name'],
                    description=combo_data.get('description', ''),
                    strategies=strategies,
                    rebalancing_rules=combo_data.get('rebalancing_rules', {}),
                    risk_management=combo_data.get('risk_management', {}),
                    expected_performance=combo_data.get('expected_performance', {})
                )
                
                self.active_combinations[combination.combination_id] = combination
                
                # パフォーマンストラッキングの初期化
                self.performance_tracking[combination.combination_id] = {
                    "created_at": combination.created_at,
                    "total_returns": [],
                    "volatility_history": [],
                    "drawdown_history": [],
                    "rebalancing_count": 0,
                    "last_rebalance": None
                }
                
                self.logger.info(f"Initialized combination: {combination.combination_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize combination {combo_data.get('combination_id', 'unknown')}: {e}")
    
    def get_combination(self, combination_id: str) -> Optional[CombinationConfig]:
        """戦略組み合わせの取得"""
        
        return self.active_combinations.get(combination_id)
    
    def get_all_combinations(self) -> Dict[str, CombinationConfig]:
        """全戦略組み合わせの取得"""
        
        return self.active_combinations.copy()
    
    def get_available_strategies(self) -> List[str]:
        """利用可能な戦略の取得"""
        
        strategies = set()
        for combination in self.active_combinations.values():
            for strategy in combination.strategies:
                strategies.add(strategy.strategy_name)
        
        return sorted(list(strategies))
    
    def create_custom_combination(self, 
                                combination_id: str,
                                name: str,
                                strategy_weights: Dict[str, float],
                                rebalancing_rules: Optional[Dict[str, Any]] = None,
                                risk_limits: Optional[Dict[str, float]] = None) -> CombinationConfig:
        """カスタム戦略組み合わせの作成"""
        
        # 重みの正規化
        total_weight = sum(strategy_weights.values())
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        
        normalized_weights = {k: v/total_weight for k, v in strategy_weights.items()}
        
        # 戦略設定の作成
        strategies = []
        for strategy_name, weight in normalized_weights.items():
            if weight < 0.01:  # 最小重み制限
                continue
                
            strategy_config = StrategyConfig(
                strategy_name=strategy_name,
                weight=weight,
                risk_limits=risk_limits or {}
            )
            strategies.append(strategy_config)
        
        # 組み合わせ設定の作成
        combination = CombinationConfig(
            combination_id=combination_id,
            name=name,
            description=f"Custom combination created at {datetime.now()}",
            strategies=strategies,
            rebalancing_rules=rebalancing_rules or {"frequency": "monthly"}
        )
        
        # 登録
        self.active_combinations[combination_id] = combination
        self.performance_tracking[combination_id] = {
            "created_at": combination.created_at,
            "total_returns": [],
            "volatility_history": [],
            "drawdown_history": [],
            "rebalancing_count": 0,
            "last_rebalance": None
        }
        
        self.logger.info(f"Created custom combination: {combination_id}")
        
        return combination
    
    def optimize_combination_weights(self, 
                                   combination_id: str,
                                   historical_performance: Dict[str, pd.Series],
                                   optimization_method: str = "risk_parity") -> Dict[str, float]:
        """戦略組み合わせの重み最適化"""
        
        combination = self.get_combination(combination_id)
        if not combination:
            raise ValueError(f"Combination {combination_id} not found")
        
        strategy_names = [s.strategy_name for s in combination.strategies]
        
        # パフォーマンスデータの準備
        returns_data = {}
        for strategy_name in strategy_names:
            if strategy_name in historical_performance:
                returns_data[strategy_name] = historical_performance[strategy_name]
            else:
                # データがない場合はダミーデータを生成
                self.logger.warning(f"No performance data for {strategy_name}, using dummy data")
                returns_data[strategy_name] = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # 最適化手法の適用
        if optimization_method == "risk_parity":
            optimized_weights = self._risk_parity_optimization(returns_data)
        elif optimization_method == "mean_variance":
            optimized_weights = self._mean_variance_optimization(returns_data)
        elif optimization_method == "equal_weight":
            optimized_weights = self._equal_weight_optimization(strategy_names)
        else:
            self.logger.warning(f"Unknown optimization method {optimization_method}, using equal weight")
            optimized_weights = self._equal_weight_optimization(strategy_names)
        
        # 制約の適用
        optimized_weights = self._apply_weight_constraints(optimized_weights, combination)
        
        self.logger.info(f"Optimized weights for {combination_id}: {optimized_weights}")
        
        return optimized_weights
    
    def _risk_parity_optimization(self, returns_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """リスクパリティ最適化"""
        
        # 各戦略のボラティリティを計算
        volatilities = {}
        for strategy_name, returns in returns_data.items():
            vol = returns.std()
            volatilities[strategy_name] = max(vol, 0.001)  # 最小ボラティリティ設定
        
        # 逆ボラティリティ重み
        inv_vol_weights = {name: 1/vol for name, vol in volatilities.items()}
        
        # 正規化
        total_inv_vol = sum(inv_vol_weights.values())
        risk_parity_weights = {name: weight/total_inv_vol for name, weight in inv_vol_weights.items()}
        
        return risk_parity_weights
    
    def _mean_variance_optimization(self, returns_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """平均分散最適化（簡略版）"""
        
        strategy_names = list(returns_data.keys())
        n_strategies = len(strategy_names)
        
        # リターンと共分散行列の計算
        returns_matrix = pd.DataFrame(returns_data)
        mean_returns = returns_matrix.mean()
        cov_matrix = returns_matrix.cov()
        
        # 簡略化された最適化（逆共分散重み）
        try:
            inv_cov = np.linalg.inv(cov_matrix.values + np.eye(n_strategies) * 0.001)  # 正則化
            ones = np.ones(n_strategies)
            
            # 最小分散ポートフォリオ重み
            weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
            
            # 負の重みを制限
            weights = np.maximum(weights, 0.01)
            weights = weights / np.sum(weights)  # 正規化
            
            return dict(zip(strategy_names, weights))
            
        except Exception as e:
            self.logger.warning(f"Mean variance optimization failed: {e}, using equal weights")
            return self._equal_weight_optimization(strategy_names)
    
    def _equal_weight_optimization(self, strategy_names: List[str]) -> Dict[str, float]:
        """等重み最適化"""
        
        n_strategies = len(strategy_names)
        equal_weight = 1.0 / n_strategies
        
        return {name: equal_weight for name in strategy_names}
    
    def _apply_weight_constraints(self, 
                                weights: Dict[str, float], 
                                combination: CombinationConfig) -> Dict[str, float]:
        """重み制約の適用"""
        
        global_limits = self.global_settings.get('position_size_limits', {})
        max_single_weight = global_limits.get('max_single_strategy_weight', 0.6)
        min_single_weight = global_limits.get('min_strategy_weight', 0.05)
        
        # 重み制約の適用
        constrained_weights = {}
        for name, weight in weights.items():
            # 最大・最小制限
            weight = max(min_single_weight, min(weight, max_single_weight))
            constrained_weights[name] = weight
        
        # 正規化
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {name: w/total_weight for name, w in constrained_weights.items()}
        
        return constrained_weights
    
    def check_rebalancing_trigger(self, 
                                combination_id: str,
                                current_performance: Dict[str, Any],
                                market_conditions: Dict[str, Any]) -> Tuple[bool, str]:
        """リバランストリガーのチェック"""
        
        combination = self.get_combination(combination_id)
        if not combination:
            return False, "Combination not found"
        
        rebalancing_rules = combination.rebalancing_rules
        frequency = RebalancingFrequency(rebalancing_rules.get('frequency', 'monthly'))
        
        # 最後のリバランスからの経過時間
        tracking = self.performance_tracking.get(combination_id, {})
        last_rebalance = tracking.get('last_rebalance')
        
        current_time = datetime.now()
        
        # 頻度ベースのチェック
        if frequency == RebalancingFrequency.DAILY:
            if last_rebalance is None or (current_time - last_rebalance).days >= 1:
                return True, "Daily rebalancing schedule"
                
        elif frequency == RebalancingFrequency.WEEKLY:
            if last_rebalance is None or (current_time - last_rebalance).days >= 7:
                return True, "Weekly rebalancing schedule"
                
        elif frequency == RebalancingFrequency.MONTHLY:
            if last_rebalance is None or (current_time - last_rebalance).days >= 30:
                return True, "Monthly rebalancing schedule"
        
        # 条件ベースのチェック
        elif frequency == RebalancingFrequency.TREND_CHANGE_BASED:
            trigger_conditions = rebalancing_rules.get('trigger_conditions', {})
            
            # トレンド信頼度のチェック
            trend_confidence = market_conditions.get('trend_confidence', 0.5)
            confidence_threshold = trigger_conditions.get('trend_confidence_threshold', 0.7)
            
            if trend_confidence > confidence_threshold:
                return True, f"Trend confidence trigger: {trend_confidence:.2f} > {confidence_threshold:.2f}"
            
            # ボラティリティ変化のチェック
            volatility_change = market_conditions.get('volatility_change', 0.0)
            volatility_threshold = trigger_conditions.get('volatility_change_threshold', 0.15)
            
            if abs(volatility_change) > volatility_threshold:
                return True, f"Volatility change trigger: {abs(volatility_change):.2f} > {volatility_threshold:.2f}"
        
        elif frequency == RebalancingFrequency.PERFORMANCE_BASED:
            # パフォーマンスベースのトリガー
            current_return = current_performance.get('total_return', 0.0)
            expected_return = combination.expected_performance.get('annual_return', 0.1)
            
            deviation = abs(current_return - expected_return)
            if deviation > 0.05:  # 5%の偏差
                return True, f"Performance deviation trigger: {deviation:.2f} > 0.05"
        
        return False, "No rebalancing trigger met"
    
    def execute_rebalancing(self, 
                          combination_id: str,
                          new_weights: Dict[str, float],
                          trigger_reason: str,
                          market_conditions: Dict[str, Any],
                          confidence_score: float = 0.8) -> RebalancingEvent:
        """リバランスの実行"""
        
        combination = self.get_combination(combination_id)
        if not combination:
            raise ValueError(f"Combination {combination_id} not found")
        
        # 現在の重みを保存
        old_weights = {s.strategy_name: s.weight for s in combination.strategies}
        
        # 新しい重みを適用
        for strategy in combination.strategies:
            if strategy.strategy_name in new_weights:
                strategy.weight = new_weights[strategy.strategy_name]
        
        # リバランスイベントの記録
        event = RebalancingEvent(
            event_id=f"rebalance_{combination_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            combination_id=combination_id,
            timestamp=datetime.now(),
            trigger_reason=trigger_reason,
            old_weights=old_weights,
            new_weights=new_weights,
            market_conditions=market_conditions,
            confidence_score=confidence_score
        )
        
        self.rebalancing_history.append(event)
        
        # パフォーマンストラッキングの更新
        if combination_id in self.performance_tracking:
            self.performance_tracking[combination_id]['rebalancing_count'] += 1
            self.performance_tracking[combination_id]['last_rebalance'] = event.timestamp
        
        self.logger.info(f"Executed rebalancing for {combination_id}: {old_weights} -> {new_weights}")
        
        return event
    
    def get_performance_summary(self, combination_id: str) -> Dict[str, Any]:
        """パフォーマンスサマリーの取得"""
        
        if combination_id not in self.performance_tracking:
            return {"error": f"No tracking data for {combination_id}"}
        
        tracking = self.performance_tracking[combination_id]
        combination = self.get_combination(combination_id)
        
        summary = {
            "combination_id": combination_id,
            "combination_name": combination.name if combination else "Unknown",
            "created_at": tracking["created_at"],
            "rebalancing_count": tracking["rebalancing_count"],
            "last_rebalance": tracking["last_rebalance"],
            "total_returns_count": len(tracking["total_returns"]),
            "average_return": np.mean(tracking["total_returns"]) if tracking["total_returns"] else 0.0,
            "average_volatility": np.mean(tracking["volatility_history"]) if tracking["volatility_history"] else 0.0,
            "max_drawdown": min(tracking["drawdown_history"]) if tracking["drawdown_history"] else 0.0,
            "current_strategies": len(combination.strategies) if combination else 0
        }
        
        return summary
    
    def get_rebalancing_history(self, 
                              combination_id: Optional[str] = None,
                              limit: int = 50) -> List[RebalancingEvent]:
        """リバランス履歴の取得"""
        
        if combination_id:
            events = [e for e in self.rebalancing_history if e.combination_id == combination_id]
        else:
            events = self.rebalancing_history
        
        # 最新順にソート
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return events[:limit]
    
    def validate_combination(self, combination: CombinationConfig) -> List[str]:
        """戦略組み合わせの検証"""
        
        errors = []
        
        # 基本検証
        if not combination.strategies:
            errors.append("No strategies defined")
        
        # 重み検証
        total_weight = sum(s.weight for s in combination.strategies)
        if abs(total_weight - 1.0) > 0.02:  # 2%の許容範囲
            errors.append(f"Total weight {total_weight:.3f} is not close to 1.0")
        
        # 戦略重複チェック
        strategy_names = [s.strategy_name for s in combination.strategies]
        if len(strategy_names) != len(set(strategy_names)):
            errors.append("Duplicate strategies found")
        
        # グローバル制約チェック
        global_limits = self.global_settings.get('position_size_limits', {})
        max_strategies = self.global_settings.get('max_concurrent_strategies', 10)
        max_single_weight = global_limits.get('max_single_strategy_weight', 1.0)
        min_single_weight = global_limits.get('min_strategy_weight', 0.0)
        
        if len(combination.strategies) > max_strategies:
            errors.append(f"Too many strategies: {len(combination.strategies)} > {max_strategies}")
        
        for strategy in combination.strategies:
            if strategy.weight > max_single_weight:
                errors.append(f"Strategy {strategy.strategy_name} weight {strategy.weight:.3f} > {max_single_weight:.3f}")
            if strategy.weight < min_single_weight:
                errors.append(f"Strategy {strategy.strategy_name} weight {strategy.weight:.3f} < {min_single_weight:.3f}")
        
        return errors
    
    def export_combinations(self, output_path: str) -> bool:
        """戦略組み合わせのエクスポート"""
        
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "combinations": {},
                "performance_tracking": self.performance_tracking,
                "rebalancing_history": [asdict(event) for event in self.rebalancing_history]
            }
            
            # 戦略組み合わせのシリアライズ
            for combo_id, combination in self.active_combinations.items():
                export_data["combinations"][combo_id] = asdict(combination)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"Combinations exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export combinations: {e}")
            return False

# テスト関数
def test_strategy_combination_manager():
    """テスト関数"""
    logger.info("Testing StrategyCombinationManager")
    
    # マネージャーの初期化
    manager = StrategyCombinationManager()
    
    # 利用可能な戦略組み合わせの表示
    combinations = manager.get_all_combinations()
    logger.info(f"Available combinations: {list(combinations.keys())}")
    
    # カスタム組み合わせの作成
    custom_weights = {
        "VWAP_Breakout": 0.4,
        "Momentum_Investing": 0.3,
        "contrarian_strategy": 0.3
    }
    
    custom_combination = manager.create_custom_combination(
        combination_id="test_custom",
        name="テストカスタム戦略",
        strategy_weights=custom_weights
    )
    
    logger.info(f"Created custom combination: {custom_combination.combination_id}")
    
    # 重み最適化のテスト
    dummy_performance = {
        strategy.strategy_name: pd.Series(np.random.normal(0.001, 0.02, 252))
        for strategy in custom_combination.strategies
    }
    
    optimized_weights = manager.optimize_combination_weights(
        combination_id="test_custom",
        historical_performance=dummy_performance,
        optimization_method="risk_parity"
    )
    
    logger.info(f"Optimized weights: {optimized_weights}")
    
    # リバランストリガーのテスト
    market_conditions = {
        "trend_confidence": 0.8,
        "volatility_change": 0.2
    }
    
    should_rebalance, reason = manager.check_rebalancing_trigger(
        combination_id="test_custom",
        current_performance={"total_return": 0.05},
        market_conditions=market_conditions
    )
    
    logger.info(f"Rebalancing trigger: {should_rebalance}, reason: {reason}")
    
    # パフォーマンスサマリー
    summary = manager.get_performance_summary("test_custom")
    logger.info(f"Performance summary: {summary}")
    
    return manager

if __name__ == "__main__":
    # テスト実行
    test_manager = test_strategy_combination_manager()
    print("StrategyCombinationManager test completed")
