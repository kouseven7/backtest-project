"""
Module: Portfolio Weighting Agent
File: portfolio_weighting_agent.py
Description: 
  3-2-1「スコアベースの資金配分計算式設計」の一部
  4段階レベルの自動化エージェントシステム
  重み計算からリバランシングまでの包括的自動化

Author: imega
Created: 2025-07-13
Modified: 2025-07-13

Dependencies:
  - config.portfolio_weight_calculator
  - config.portfolio_weight_templates
  - config.strategy_scoring_model
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config.portfolio_weight_calculator import (
        PortfolioWeightCalculator, WeightAllocationConfig, AllocationResult,
        PortfolioConstraints, AllocationMethod
    )
    from config.portfolio_weight_templates import (
        WeightTemplateManager, WeightTemplate, MarketRegime, TemplateType
    )
    from config.strategy_scoring_model import StrategyScoreManager
    from config.strategy_selector import StrategySelector
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some functionality may be limited.")

# ロガーの設定
logger = logging.getLogger(__name__)

class AutomationLevel(Enum):
    """自動化レベル"""
    MANUAL = "manual"                    # 手動実行のみ
    SEMI_AUTOMATIC = "semi_automatic"    # 推奨提示 + 手動承認
    AUTOMATIC = "automatic"              # 自動実行 + 通知
    FULLY_AUTOMATIC = "fully_automatic"  # 完全自動実行

class TriggerCondition(Enum):
    """トリガー条件"""
    TIME_BASED = "time_based"           # 時間ベース
    SCORE_CHANGE = "score_change"       # スコア変化
    MARKET_CHANGE = "market_change"     # 市場変化
    WEIGHT_DRIFT = "weight_drift"       # 重みドリフト
    RISK_THRESHOLD = "risk_threshold"   # リスク閾値
    PERFORMANCE = "performance"         # パフォーマンス

class ActionType(Enum):
    """アクション種別"""
    CALCULATE_WEIGHTS = "calculate_weights"
    REBALANCE = "rebalance"
    UPDATE_TEMPLATE = "update_template"
    ALERT = "alert"
    REPORT = "report"

@dataclass
class AutomationRule:
    """自動化ルール"""
    name: str
    trigger_condition: TriggerCondition
    action_type: ActionType
    threshold_value: float
    automation_level: AutomationLevel
    enabled: bool = True
    cooldown_hours: int = 1
    priority: int = 1  # 1=highest, 5=lowest
    conditions: Dict[str, Any] = field(default_factory=dict)
    last_triggered: Optional[datetime] = None

@dataclass
class AgentDecision:
    """エージェント意思決定結果"""
    decision_type: ActionType
    recommended_action: str
    confidence_level: float
    reasoning: str
    required_approval: bool
    risk_level: str
    estimated_impact: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ExecutionResult:
    """実行結果"""
    action_type: ActionType
    success: bool
    execution_time: float
    details: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class PortfolioWeightingAgent:
    """
    ポートフォリオ重み付けエージェント
    
    機能:
    1. 4段階自動化レベル（手動～完全自動）
    2. 複数トリガー条件による自動実行
    3. リスク管理と承認フロー
    4. パフォーマンス監視とアラート
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 automation_level: AutomationLevel = AutomationLevel.SEMI_AUTOMATIC):
        """エージェントの初期化"""
        self.automation_level = automation_level
        self.base_dir = Path("config/portfolio_agent")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定の読み込み
        self.config = self._load_config(config_file)
        
        # コンポーネントの初期化
        self.weight_calculator = PortfolioWeightCalculator()
        self.template_manager = WeightTemplateManager()
        self.score_manager = StrategyScoreManager()
        self.strategy_selector = StrategySelector()
        
        # 自動化ルールの初期化
        self.automation_rules = self._initialize_automation_rules()
        
        # 状態管理
        self.current_weights = {}
        self.last_rebalance = None
        self.decision_history = []
        self.execution_history = []
        self.active_alerts = []
        
        # 承認待ちキュー
        self.pending_approvals = []
        
        logger.info(f"PortfolioWeightingAgent initialized with {automation_level.value} automation level")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # デフォルト設定
        return {
            "monitoring_interval_minutes": 30,
            "max_weight_drift_threshold": 0.05,
            "score_change_threshold": 0.1,
            "risk_threshold_multiplier": 1.5,
            "rebalance_cooldown_hours": 24,
            "alert_settings": {
                "email_enabled": False,
                "email_recipients": [],
                "slack_webhook": None
            },
            "approval_settings": {
                "require_approval_above_risk": "medium",
                "approval_timeout_hours": 24,
                "auto_approve_small_changes": True,
                "small_change_threshold": 0.02
            }
        }

    def _initialize_automation_rules(self) -> List[AutomationRule]:
        """自動化ルールの初期化"""
        rules = []
        
        # 1. 時間ベースリバランス
        rules.append(AutomationRule(
            name="Weekly Rebalance",
            trigger_condition=TriggerCondition.TIME_BASED,
            action_type=ActionType.REBALANCE,
            threshold_value=7*24,  # 7日 = 168時間
            automation_level=AutomationLevel.AUTOMATIC,
            priority=2,
            conditions={"weekday": 1}  # 月曜日
        ))
        
        # 2. スコア変化による重み再計算
        rules.append(AutomationRule(
            name="Score Change Trigger",
            trigger_condition=TriggerCondition.SCORE_CHANGE,
            action_type=ActionType.CALCULATE_WEIGHTS,
            threshold_value=self.config.get("score_change_threshold", 0.1),
            automation_level=AutomationLevel.SEMI_AUTOMATIC,
            priority=1,
            cooldown_hours=2
        ))
        
        # 3. 重みドリフト検出
        rules.append(AutomationRule(
            name="Weight Drift Rebalance",
            trigger_condition=TriggerCondition.WEIGHT_DRIFT,
            action_type=ActionType.REBALANCE,
            threshold_value=self.config.get("max_weight_drift_threshold", 0.05),
            automation_level=AutomationLevel.AUTOMATIC,
            priority=1,
            cooldown_hours=4
        ))
        
        # 4. リスク閾値アラート
        rules.append(AutomationRule(
            name="Risk Threshold Alert",
            trigger_condition=TriggerCondition.RISK_THRESHOLD,
            action_type=ActionType.ALERT,
            threshold_value=self.config.get("risk_threshold_multiplier", 1.5),
            automation_level=AutomationLevel.FULLY_AUTOMATIC,
            priority=1,
            cooldown_hours=1
        ))
        
        # 5. パフォーマンス監視
        rules.append(AutomationRule(
            name="Performance Monitor",
            trigger_condition=TriggerCondition.PERFORMANCE,
            action_type=ActionType.REPORT,
            threshold_value=-0.05,  # -5%のパフォーマンス
            automation_level=AutomationLevel.AUTOMATIC,
            priority=2,
            cooldown_hours=24
        ))
        
        return rules

    async def monitor_and_execute(self, 
                                ticker: str,
                                market_data: pd.DataFrame,
                                max_iterations: int = 1000):
        """
        監視・実行メインループ
        
        Parameters:
            ticker: 監視対象ティッカー
            market_data: 市場データ
            max_iterations: 最大反復回数
        """
        iteration = 0
        monitoring_interval = self.config.get("monitoring_interval_minutes", 30) * 60
        
        logger.info(f"Starting monitoring loop for {ticker} with {self.automation_level.value} automation")
        
        while iteration < max_iterations:
            try:
                # 各ルールのチェック
                triggered_rules = await self._check_trigger_conditions(ticker, market_data)
                
                # トリガーされたルールの処理
                for rule in triggered_rules:
                    decision = await self._make_decision(rule, ticker, market_data)
                    
                    if decision:
                        execution_result = await self._execute_decision(decision, ticker, market_data)
                        
                        # 実行結果の記録
                        self.execution_history.append(execution_result)
                        
                        # ログ出力
                        if execution_result.success:
                            logger.info(f"Successfully executed {decision.decision_type.value} for {ticker}")
                        else:
                            logger.error(f"Failed to execute {decision.decision_type.value} for {ticker}: {execution_result.errors}")
                
                # 承認待ちキューの処理
                await self._process_pending_approvals()
                
                # アラートの管理
                await self._manage_alerts()
                
                # 次の監視サイクルまで待機
                await asyncio.sleep(monitoring_interval)
                iteration += 1
                
            except KeyboardInterrupt:
                logger.info("Monitoring loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # エラー時は1分待機
                iteration += 1

    async def _check_trigger_conditions(self, 
                                      ticker: str,
                                      market_data: pd.DataFrame) -> List[AutomationRule]:
        """トリガー条件のチェック"""
        triggered_rules = []
        current_time = datetime.now()
        
        for rule in self.automation_rules:
            if not rule.enabled:
                continue
            
            # クールダウンチェック
            if (rule.last_triggered and 
                (current_time - rule.last_triggered).total_seconds() < rule.cooldown_hours * 3600):
                continue
            
            is_triggered = False
            
            # 条件別チェック
            if rule.trigger_condition == TriggerCondition.TIME_BASED:
                is_triggered = await self._check_time_based_trigger(rule)
            elif rule.trigger_condition == TriggerCondition.SCORE_CHANGE:
                is_triggered = await self._check_score_change_trigger(rule, ticker)
            elif rule.trigger_condition == TriggerCondition.WEIGHT_DRIFT:
                is_triggered = await self._check_weight_drift_trigger(rule, ticker)
            elif rule.trigger_condition == TriggerCondition.RISK_THRESHOLD:
                is_triggered = await self._check_risk_threshold_trigger(rule, ticker, market_data)
            elif rule.trigger_condition == TriggerCondition.PERFORMANCE:
                is_triggered = await self._check_performance_trigger(rule, ticker)
            
            if is_triggered:
                rule.last_triggered = current_time
                triggered_rules.append(rule)
        
        return triggered_rules

    async def _check_time_based_trigger(self, rule: AutomationRule) -> bool:
        """時間ベーストリガーのチェック"""
        current_time = datetime.now()
        
        if "weekday" in rule.conditions:
            target_weekday = rule.conditions["weekday"]
            if current_time.weekday() != target_weekday:
                return False
        
        # 最後のリバランスからの経過時間をチェック
        if self.last_rebalance:
            hours_since_last = (current_time - self.last_rebalance).total_seconds() / 3600
            return hours_since_last >= rule.threshold_value
        
        return True  # 初回実行

    async def _check_score_change_trigger(self, rule: AutomationRule, ticker: str) -> bool:
        """スコア変化トリガーのチェック"""
        try:
            # 現在のスコアを取得
            current_scores = self.score_manager.calculate_comprehensive_scores([ticker])
            
            if ticker not in current_scores:
                return False
            
            # 過去のスコアと比較（簡略化）
            # 実際の実装では履歴データベースからの取得が必要
            score_change = 0.05  # プレースホルダー
            
            return abs(score_change) >= rule.threshold_value
            
        except Exception as e:
            logger.error(f"Error checking score change trigger: {e}")
            return False

    async def _check_weight_drift_trigger(self, rule: AutomationRule, ticker: str) -> bool:
        """重みドリフトトリガーのチェック"""
        if not self.current_weights:
            return False
        
        try:
            # 現在の理想的な重みを計算
            result = self.weight_calculator.calculate_portfolio_weights(
                ticker=ticker,
                market_data=pd.DataFrame()  # 簡略化
            )
            
            if not result.strategy_weights:
                return False
            
            # ドリフトの計算
            max_drift = 0.0
            for strategy, current_weight in self.current_weights.items():
                ideal_weight = result.strategy_weights.get(strategy, 0.0)
                drift = abs(current_weight - ideal_weight)
                max_drift = max(max_drift, drift)
            
            return max_drift >= rule.threshold_value
            
        except Exception as e:
            logger.error(f"Error checking weight drift trigger: {e}")
            return False

    async def _check_risk_threshold_trigger(self, 
                                          rule: AutomationRule,
                                          ticker: str,
                                          market_data: pd.DataFrame) -> bool:
        """リスク閾値トリガーのチェック"""
        try:
            # 現在のリスク指標を計算
            if market_data.empty:
                return False
            
            returns = market_data['Adj Close'].pct_change().dropna()
            if len(returns) < 20:
                return False
            
            current_volatility = returns.std() * np.sqrt(252)
            
            # ベースラインリスクとの比較
            baseline_risk = 0.15  # プレースホルダー
            risk_ratio = current_volatility / baseline_risk
            
            return risk_ratio >= rule.threshold_value
            
        except Exception as e:
            logger.error(f"Error checking risk threshold trigger: {e}")
            return False

    async def _check_performance_trigger(self, rule: AutomationRule, ticker: str) -> bool:
        """パフォーマンストリガーのチェック"""
        try:
            # パフォーマンス計算の簡略化実装
            # 実際の実装では詳細なパフォーマンス追跡が必要
            performance = -0.02  # プレースホルダー
            
            return performance <= rule.threshold_value
            
        except Exception as e:
            logger.error(f"Error checking performance trigger: {e}")
            return False

    async def _make_decision(self, 
                           rule: AutomationRule,
                           ticker: str,
                           market_data: pd.DataFrame) -> Optional[AgentDecision]:
        """意思決定の実行"""
        try:
            # 自動化レベルに基づく決定
            if rule.automation_level == AutomationLevel.MANUAL:
                return None  # 手動実行のみ
            
            # 意思決定ロジック
            confidence_level = self._calculate_decision_confidence(rule, ticker, market_data)
            risk_level = self._assess_decision_risk(rule, ticker)
            
            # 承認要否の判定
            require_approval = (
                rule.automation_level == AutomationLevel.SEMI_AUTOMATIC or
                (risk_level in ["high", "very_high"]) or
                confidence_level < 0.7
            )
            
            # 推奨アクションの生成
            recommended_action = self._generate_recommended_action(rule, ticker)
            
            # 影響評価
            estimated_impact = self._estimate_impact(rule, ticker)
            
            decision = AgentDecision(
                decision_type=rule.action_type,
                recommended_action=recommended_action,
                confidence_level=confidence_level,
                reasoning=f"Triggered by {rule.name}: {rule.trigger_condition.value}",
                required_approval=require_approval,
                risk_level=risk_level,
                estimated_impact=estimated_impact
            )
            
            self.decision_history.append(decision)
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision for rule {rule.name}: {e}")
            return None

    def _calculate_decision_confidence(self, 
                                     rule: AutomationRule,
                                     ticker: str,
                                     market_data: pd.DataFrame) -> float:
        """意思決定信頼度の計算"""
        base_confidence = 0.5
        
        # データ品質による調整
        if not market_data.empty and len(market_data) >= 100:
            base_confidence += 0.2
        
        # ルール優先度による調整
        priority_bonus = (6 - rule.priority) * 0.05
        base_confidence += priority_bonus
        
        # 市場状況による調整
        # （簡略化実装）
        market_stability = 0.1
        base_confidence += market_stability
        
        return min(1.0, max(0.0, base_confidence))

    def _assess_decision_risk(self, rule: AutomationRule, ticker: str) -> str:
        """意思決定リスクの評価"""
        risk_factors = []
        
        # アクションタイプによるリスク
        if rule.action_type == ActionType.REBALANCE:
            risk_factors.append("medium")
        elif rule.action_type == ActionType.CALCULATE_WEIGHTS:
            risk_factors.append("low")
        elif rule.action_type == ActionType.ALERT:
            risk_factors.append("very_low")
        
        # 市場状況によるリスク
        # （簡略化実装）
        risk_factors.append("low")
        
        # 最高リスクレベルを返す
        risk_mapping = {"very_low": 1, "low": 2, "medium": 3, "high": 4, "very_high": 5}
        max_risk_value = max(risk_mapping.get(risk, 2) for risk in risk_factors)
        
        risk_reverse_mapping = {1: "very_low", 2: "low", 3: "medium", 4: "high", 5: "very_high"}
        return risk_reverse_mapping[max_risk_value]

    def _generate_recommended_action(self, rule: AutomationRule, ticker: str) -> str:
        """推奨アクションの生成"""
        if rule.action_type == ActionType.CALCULATE_WEIGHTS:
            return f"Recalculate portfolio weights for {ticker}"
        elif rule.action_type == ActionType.REBALANCE:
            return f"Rebalance portfolio for {ticker}"
        elif rule.action_type == ActionType.UPDATE_TEMPLATE:
            return f"Update weight template for {ticker}"
        elif rule.action_type == ActionType.ALERT:
            return f"Generate alert for {ticker}"
        elif rule.action_type == ActionType.REPORT:
            return f"Generate performance report for {ticker}"
        else:
            return f"Execute {rule.action_type.value} for {ticker}"

    def _estimate_impact(self, rule: AutomationRule, ticker: str) -> Dict[str, float]:
        """影響評価"""
        # 簡略化実装
        return {
            "estimated_return_impact": 0.001,
            "estimated_risk_impact": 0.005,
            "transaction_cost": 0.002,
            "execution_time_minutes": 5.0
        }

    async def _execute_decision(self, 
                              decision: AgentDecision,
                              ticker: str,
                              market_data: pd.DataFrame) -> ExecutionResult:
        """意思決定の実行"""
        start_time = datetime.now()
        
        try:
            # 承認要否チェック
            if decision.required_approval and self.automation_level != AutomationLevel.FULLY_AUTOMATIC:
                # 承認待ちキューに追加
                self.pending_approvals.append((decision, ticker, market_data))
                return ExecutionResult(
                    action_type=decision.decision_type,
                    success=True,
                    execution_time=0.0,
                    details={"status": "pending_approval"}
                )
            
            # アクション実行
            success = False
            details = {}
            errors = []
            
            if decision.decision_type == ActionType.CALCULATE_WEIGHTS:
                result = self.weight_calculator.calculate_portfolio_weights(ticker, market_data)
                success = bool(result.strategy_weights)
                details = {"weights": result.strategy_weights, "confidence": result.confidence_level}
                
            elif decision.decision_type == ActionType.REBALANCE:
                # リバランス実行（簡略化）
                success = True
                details = {"rebalanced_at": datetime.now().isoformat()}
                self.last_rebalance = datetime.now()
                
            elif decision.decision_type == ActionType.ALERT:
                # アラート生成
                alert_message = f"Alert for {ticker}: {decision.reasoning}"
                self.active_alerts.append(alert_message)
                success = True
                details = {"alert_message": alert_message}
                
            elif decision.decision_type == ActionType.REPORT:
                # レポート生成
                report_data = self._generate_performance_report(ticker)
                success = True
                details = {"report": report_data}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                action_type=decision.decision_type,
                success=success,
                execution_time=execution_time,
                details=details,
                errors=errors
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Execution error: {str(e)}"
            
            return ExecutionResult(
                action_type=decision.decision_type,
                success=False,
                execution_time=execution_time,
                details={},
                errors=[error_msg]
            )

    async def _process_pending_approvals(self):
        """承認待ちキューの処理"""
        # 自動承認ロジック（簡略化）
        auto_approve = self.config.get("approval_settings", {}).get("auto_approve_small_changes", True)
        
        if auto_approve:
            approved_items = []
            for item in self.pending_approvals:
                decision, ticker, market_data = item
                if decision.risk_level in ["very_low", "low"]:
                    # 低リスクのアイテムは自動承認
                    execution_result = await self._execute_decision(decision, ticker, market_data)
                    self.execution_history.append(execution_result)
                    approved_items.append(item)
            
            # 承認済みアイテムを削除
            for item in approved_items:
                self.pending_approvals.remove(item)

    async def _manage_alerts(self):
        """アラート管理"""
        # 古いアラートのクリーンアップ
        max_alerts = 10
        if len(self.active_alerts) > max_alerts:
            self.active_alerts = self.active_alerts[-max_alerts:]

    def _generate_performance_report(self, ticker: str) -> Dict[str, Any]:
        """パフォーマンスレポートの生成"""
        return {
            "ticker": ticker,
            "generated_at": datetime.now().isoformat(),
            "current_weights": self.current_weights,
            "recent_decisions": len(self.decision_history),
            "recent_executions": len(self.execution_history),
            "active_alerts": len(self.active_alerts)
        }

    def get_agent_status(self) -> Dict[str, Any]:
        """エージェント状態の取得"""
        return {
            "automation_level": self.automation_level.value,
            "active_rules": len([r for r in self.automation_rules if r.enabled]),
            "pending_approvals": len(self.pending_approvals),
            "active_alerts": len(self.active_alerts),
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "decision_history_count": len(self.decision_history),
            "execution_history_count": len(self.execution_history)
        }

    def approve_pending_decision(self, decision_index: int) -> bool:
        """承認待ち決定の手動承認"""
        try:
            if 0 <= decision_index < len(self.pending_approvals):
                decision, ticker, market_data = self.pending_approvals.pop(decision_index)
                
                # 即座に実行
                asyncio.create_task(self._execute_decision(decision, ticker, market_data))
                logger.info(f"Manually approved decision: {decision.decision_type.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error approving decision: {e}")
            return False

if __name__ == "__main__":
    # 簡単なテスト
    agent = PortfolioWeightingAgent(automation_level=AutomationLevel.SEMI_AUTOMATIC)
    print(f"Agent status: {agent.get_agent_status()}")
