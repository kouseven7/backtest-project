"""
DSSMS Phase 2 Task 2.3: Enhanced Risk Management System
Component: Automated Risk Actions

This module provides automated risk response actions based on risk events and thresholds.
Implements semi-automated approach with safety checks and human oversight capabilities.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import time

# Import types from unified risk monitor
from .unified_risk_monitor import RiskEvent


class ActionStatus(Enum):
    """Risk action execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_APPROVAL = "awaiting_approval"


class ActionType(Enum):
    """Types of automated risk actions"""
    POSITION_REDUCTION = "position_reduction"
    STOP_TRADING = "stop_trading"
    REBALANCE_PORTFOLIO = "rebalance_portfolio"
    HEDGE_POSITION = "hedge_position"
    ALERT_ONLY = "alert_only"
    EMERGENCY_LIQUIDATION = "emergency_liquidation"


@dataclass
class RiskAction:
    """Risk action definition and execution tracking"""
    action_id: str
    action_type: ActionType
    trigger_event: RiskEvent
    parameters: Dict[str, Any]
    status: ActionStatus
    created_time: datetime
    scheduled_time: Optional[datetime] = None
    executed_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approval_time: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class AutomatedRiskActions:
    """
    Automated risk action execution system with safety controls and human oversight.
    Implements semi-automated approach with configurable approval requirements.
    """
    
    def __init__(self, config_dir: str = "config/enhanced_risk_management/configs"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
        # Action tracking
        self.active_actions: Dict[str, RiskAction] = {}
        self.action_history: List[RiskAction] = []
        self.pending_approvals: Dict[str, RiskAction] = {}
        
        # Configuration
        self.action_config = self._load_action_config()
        self.safety_limits = self._load_safety_limits()
        
        # Control flags
        self.automation_enabled = True
        self.emergency_mode = False
        self.last_action_time = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Action executors (configurable)
        self.action_executors: Dict[ActionType, Callable] = {
            ActionType.POSITION_REDUCTION: self._execute_position_reduction,
            ActionType.STOP_TRADING: self._execute_stop_trading,
            ActionType.REBALANCE_PORTFOLIO: self._execute_rebalance,
            ActionType.HEDGE_POSITION: self._execute_hedge_position,
            ActionType.ALERT_ONLY: self._execute_alert_only,
            ActionType.EMERGENCY_LIQUIDATION: self._execute_emergency_liquidation
        }
        
        self.logger.info("AutomatedRiskActions initialized successfully")
    
    def _load_action_config(self) -> Dict[str, Any]:
        """Load action configuration from file"""
        config_file = self.config_dir / "action_config.json"
        try:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Default configuration
                default_config = {
                    "enabled_actions": [
                        "position_reduction",
                        "rebalance_portfolio",
                        "alert_only"
                    ],
                    "approval_required": {
                        "emergency_liquidation": True,
                        "stop_trading": True,
                        "position_reduction": False,
                        "rebalance_portfolio": False,
                        "hedge_position": True,
                        "alert_only": False
                    },
                    "cooldown_minutes": {
                        "position_reduction": 15,
                        "stop_trading": 60,
                        "rebalance_portfolio": 30,
                        "hedge_position": 45,
                        "emergency_liquidation": 120,
                        "alert_only": 5
                    },
                    "max_daily_actions": {
                        "position_reduction": 5,
                        "stop_trading": 2,
                        "rebalance_portfolio": 3,
                        "hedge_position": 4,
                        "emergency_liquidation": 1,
                        "alert_only": 20
                    }
                }
                # Save default configuration
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
        except Exception as e:
            self.logger.error(f"Failed to load action config: {e}")
            return {}
    
    def _load_safety_limits(self) -> Dict[str, Any]:
        """Load safety limits configuration"""
        config_file = self.config_dir / "safety_limits.json"
        try:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Default safety limits
                default_limits = {
                    "max_position_reduction_per_action": 0.20,  # 20% max reduction per action
                    "max_daily_portfolio_change": 0.50,  # 50% max portfolio change per day
                    "min_cash_reserve": 0.05,  # 5% minimum cash reserve
                    "max_leverage_after_action": 2.0,  # Maximum leverage after action
                    "emergency_liquidation_threshold": 0.15,  # 15% drawdown triggers emergency
                    "stop_loss_buffer": 0.02  # 2% buffer for stop loss actions
                }
                # Save default limits
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_limits, f, indent=2, ensure_ascii=False)
                return default_limits
        except Exception as e:
            self.logger.error(f"Failed to load safety limits: {e}")
            return {}
    
    def process_risk_event(self, risk_event: RiskEvent) -> Optional[RiskAction]:
        """
        Process a risk event and determine if automated action is required.
        
        Args:
            risk_event: Risk event to process
            
        Returns:
            RiskAction if action is required, None otherwise
        """
        try:
            if not self.automation_enabled:
                self.logger.info("Automation disabled, skipping action processing")
                return None
            
            # Determine action type based on risk event
            action_type = self._determine_action_type(risk_event)
            if not action_type:
                return None
            
            # Check if action is enabled
            if action_type.value not in self.action_config.get("enabled_actions", []):
                self.logger.info(f"Action type {action_type.value} is disabled")
                return None
            
            # Check cooldown period
            if not self._check_cooldown(action_type):
                self.logger.info(f"Action type {action_type.value} is in cooldown period")
                return None
            
            # Check daily limits
            if not self._check_daily_limits(action_type):
                self.logger.info(f"Daily limit reached for action type {action_type.value}")
                return None
            
            # Create action
            action = self._create_action(risk_event, action_type)
            
            # Check if approval is required
            approval_required = self.action_config.get("approval_required", {}).get(
                action_type.value, False
            )
            
            if approval_required:
                action.requires_approval = True
                action.status = ActionStatus.AWAITING_APPROVAL
                self.pending_approvals[action.action_id] = action
                self.logger.info(f"Action {action.action_id} requires approval")
            else:
                # Execute immediately
                self._execute_action(action)
            
            return action
            
        except Exception as e:
            self.logger.error(f"Failed to process risk event: {e}")
            return None
    
    def _determine_action_type(self, risk_event: RiskEvent) -> Optional[ActionType]:
        """Determine appropriate action type based on risk event"""
        risk_type = risk_event.event_type  # event_type instead of risk_type
        severity = risk_event.severity
        
        # Emergency conditions
        if severity == "critical":
            if risk_type == "drawdown" and risk_event.current_value > 0.15:
                return ActionType.EMERGENCY_LIQUIDATION
            elif risk_type in ["var_breach", "large_loss"]:
                return ActionType.POSITION_REDUCTION
            elif risk_type == "system_error":
                return ActionType.STOP_TRADING
        
        # High severity conditions
        elif severity == "high":
            if risk_type == "drawdown":
                return ActionType.POSITION_REDUCTION
            elif risk_type == "concentration_risk":
                return ActionType.REBALANCE_PORTFOLIO
            elif risk_type == "volatility":
                return ActionType.HEDGE_POSITION
            elif risk_type == "var_breach":
                return ActionType.POSITION_REDUCTION
        
        # Medium severity conditions
        elif severity == "medium":
            if risk_type in ["concentration_risk", "correlation_risk"]:
                return ActionType.REBALANCE_PORTFOLIO
            elif risk_type == "volatility":
                return ActionType.ALERT_ONLY
        
        # Default to alert only for low severity
        return ActionType.ALERT_ONLY
    
    def _check_cooldown(self, action_type: ActionType) -> bool:
        """Check if action type is in cooldown period"""
        cooldown_minutes = self.action_config.get("cooldown_minutes", {}).get(
            action_type.value, 30
        )
        
        last_time = self.last_action_time.get(action_type.value)
        if last_time:
            time_diff = datetime.now() - last_time
            if time_diff < timedelta(minutes=cooldown_minutes):
                return False
        
        return True
    
    def _check_daily_limits(self, action_type: ActionType) -> bool:
        """Check if daily action limit has been reached"""
        max_daily = self.action_config.get("max_daily_actions", {}).get(
            action_type.value, 10
        )
        
        today = datetime.now().date()
        daily_count = sum(
            1 for action in self.action_history
            if (action.created_time.date() == today and 
                action.action_type == action_type and
                action.status == ActionStatus.COMPLETED)
        )
        
        return daily_count < max_daily
    
    def _create_action(self, risk_event: RiskEvent, action_type: ActionType) -> RiskAction:
        """Create a risk action from risk event and action type"""
        action_id = f"{action_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate action parameters based on type and risk event
        parameters = self._generate_action_parameters(risk_event, action_type)
        
        action = RiskAction(
            action_id=action_id,
            action_type=action_type,
            trigger_event=risk_event,
            parameters=parameters,
            status=ActionStatus.PENDING,
            created_time=datetime.now()
        )
        
        with self._lock:
            self.active_actions[action_id] = action
        
        return action
    
    def _generate_action_parameters(self, risk_event: RiskEvent, action_type: ActionType) -> Dict[str, Any]:
        """Generate action parameters based on risk event and action type"""
        base_params = {
            "risk_type": risk_event.event_type,  # event_type instead of risk_type
            "severity": risk_event.severity,
            "current_value": getattr(risk_event, 'drawdown', 0.0),  # Use drawdown as current_value
            "threshold": 0.05  # Default threshold
        }
        
        if action_type == ActionType.POSITION_REDUCTION:
            # Calculate reduction percentage based on risk severity
            current_value = getattr(risk_event, 'drawdown', 0.05)
            if risk_event.severity == "critical":
                reduction_pct = min(0.50, current_value * 2)  # More aggressive
            elif risk_event.severity == "high":
                reduction_pct = min(0.30, current_value * 1.5)
            else:
                reduction_pct = min(0.20, current_value * 1.2)
            
            base_params.update({
                "reduction_percentage": reduction_pct,
                "target_positions": [],  # Default empty list
                "method": "proportional"
            })
        
        elif action_type == ActionType.REBALANCE_PORTFOLIO:
            base_params.update({
                "rebalance_method": "risk_parity",
                "target_allocation": {},  # Default empty dict
                "min_trade_size": 0.01
            })
        
        elif action_type == ActionType.HEDGE_POSITION:
            current_value = getattr(risk_event, 'drawdown', 0.05)
            base_params.update({
                "hedge_ratio": min(0.5, current_value),
                "hedge_instrument": "index_futures",
                "duration": "temporary"
            })
        
        elif action_type == ActionType.STOP_TRADING:
            base_params.update({
                "duration_minutes": 60,
                "strategies_to_stop": ["all"],  # Default to all strategies
                "resume_conditions": ["manual_approval", "risk_normalized"]
            })
        
        elif action_type == ActionType.EMERGENCY_LIQUIDATION:
            base_params.update({
                "liquidation_percentage": 1.0,  # Full liquidation
                "priority_order": "highest_risk_first",
                "market_order_allowed": True
            })
        
        return base_params
    
    def _execute_action(self, action: RiskAction) -> bool:
        """Execute a risk action"""
        try:
            action.status = ActionStatus.EXECUTING
            action.executed_time = datetime.now()
            
            # Get executor for action type
            executor = self.action_executors.get(action.action_type)
            if not executor:
                raise ValueError(f"No executor found for action type: {action.action_type}")
            
            # Execute the action
            result = executor(action)
            
            if result.get("success", False):
                action.status = ActionStatus.COMPLETED
                action.completion_time = datetime.now()
                action.execution_result = result
                
                # Update last action time
                self.last_action_time[action.action_type.value] = datetime.now()
                
                self.logger.info(f"Action {action.action_id} completed successfully")
                return True
            else:
                action.status = ActionStatus.FAILED
                action.error_message = result.get("error", "Unknown error")
                self.logger.error(f"Action {action.action_id} failed: {action.error_message}")
                return False
                
        except Exception as e:
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
            self.logger.error(f"Failed to execute action {action.action_id}: {e}")
            return False
        finally:
            # Move to history
            with self._lock:
                if action.action_id in self.active_actions:
                    del self.active_actions[action.action_id]
                self.action_history.append(action)
    
    def approve_action(self, action_id: str, approved_by: str) -> bool:
        """Approve a pending action"""
        try:
            if action_id not in self.pending_approvals:
                self.logger.error(f"Action {action_id} not found in pending approvals")
                return False
            
            action = self.pending_approvals[action_id]
            action.approved_by = approved_by
            action.approval_time = datetime.now()
            action.status = ActionStatus.PENDING
            
            # Move from pending to active
            with self._lock:
                del self.pending_approvals[action_id]
                self.active_actions[action_id] = action
            
            # Execute the approved action
            return self._execute_action(action)
            
        except Exception as e:
            self.logger.error(f"Failed to approve action {action_id}: {e}")
            return False
    
    def reject_action(self, action_id: str, rejected_by: str, reason: str = "") -> bool:
        """Reject a pending action"""
        try:
            if action_id not in self.pending_approvals:
                self.logger.error(f"Action {action_id} not found in pending approvals")
                return False
            
            action = self.pending_approvals[action_id]
            action.status = ActionStatus.CANCELLED
            action.error_message = f"Rejected by {rejected_by}: {reason}"
            
            # Move from pending to history
            with self._lock:
                del self.pending_approvals[action_id]
                self.action_history.append(action)
            
            self.logger.info(f"Action {action_id} rejected by {rejected_by}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reject action {action_id}: {e}")
            return False
    
    # Action executor methods (placeholder implementations)
    def _execute_position_reduction(self, action: RiskAction) -> Dict[str, Any]:
        """Execute position reduction action"""
        try:
            reduction_pct = action.parameters.get("reduction_percentage", 0.20)
            target_positions = action.parameters.get("target_positions", [])
            
            self.logger.info(f"Executing position reduction: {reduction_pct:.2%}")
            
            # Placeholder: Actual implementation would interface with trading system
            # For now, just simulate the action
            
            return {
                "success": True,
                "reduction_applied": reduction_pct,
                "positions_affected": len(target_positions),
                "execution_time": datetime.now().isoformat()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_stop_trading(self, action: RiskAction) -> Dict[str, Any]:
        """Execute stop trading action"""
        try:
            duration = action.parameters.get("duration_minutes", 60)
            strategies = action.parameters.get("strategies_to_stop", ["all"])
            
            self.logger.info(f"Stopping trading for {duration} minutes")
            
            # Placeholder implementation
            return {
                "success": True,
                "trading_stopped": True,
                "duration_minutes": duration,
                "strategies_affected": strategies
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_rebalance(self, action: RiskAction) -> Dict[str, Any]:
        """Execute portfolio rebalance action"""
        try:
            method = action.parameters.get("rebalance_method", "risk_parity")
            
            self.logger.info(f"Executing portfolio rebalance using {method}")
            
            # Placeholder implementation
            return {
                "success": True,
                "rebalance_method": method,
                "rebalance_completed": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_hedge_position(self, action: RiskAction) -> Dict[str, Any]:
        """Execute hedge position action"""
        try:
            hedge_ratio = action.parameters.get("hedge_ratio", 0.5)
            instrument = action.parameters.get("hedge_instrument", "index_futures")
            
            self.logger.info(f"Executing hedge with ratio {hedge_ratio}")
            
            # Placeholder implementation
            return {
                "success": True,
                "hedge_ratio": hedge_ratio,
                "hedge_instrument": instrument,
                "hedge_established": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_alert_only(self, action: RiskAction) -> Dict[str, Any]:
        """Execute alert-only action (no trading action required)"""
        try:
            self.logger.info("Alert-only action executed")
            return {
                "success": True,
                "alert_sent": True,
                "action_type": "alert_only"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_emergency_liquidation(self, action: RiskAction) -> Dict[str, Any]:
        """Execute emergency liquidation action"""
        try:
            liquidation_pct = action.parameters.get("liquidation_percentage", 1.0)
            
            self.logger.critical(f"Executing emergency liquidation: {liquidation_pct:.2%}")
            
            # Placeholder implementation - would interface with trading system
            return {
                "success": True,
                "liquidation_percentage": liquidation_pct,
                "emergency_liquidation_completed": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_action_status(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific action"""
        # Check active actions
        if action_id in self.active_actions:
            return asdict(self.active_actions[action_id])
        
        # Check pending approvals
        if action_id in self.pending_approvals:
            return asdict(self.pending_approvals[action_id])
        
        # Check history
        for action in self.action_history:
            if action.action_id == action_id:
                return asdict(action)
        
        return None
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all actions awaiting approval"""
        return [asdict(action) for action in self.pending_approvals.values()]
    
    def get_action_summary(self) -> Dict[str, Any]:
        """Get summary of all actions"""
        return {
            "active_actions": len(self.active_actions),
            "pending_approvals": len(self.pending_approvals),
            "completed_today": len([
                a for a in self.action_history 
                if (a.created_time.date() == datetime.now().date() and 
                    a.status == ActionStatus.COMPLETED)
            ]),
            "failed_today": len([
                a for a in self.action_history 
                if (a.created_time.date() == datetime.now().date() and 
                    a.status == ActionStatus.FAILED)
            ]),
            "automation_enabled": self.automation_enabled,
            "emergency_mode": self.emergency_mode
        }
    
    def set_automation_enabled(self, enabled: bool) -> None:
        """Enable or disable automation"""
        self.automation_enabled = enabled
        self.logger.info(f"Automation {'enabled' if enabled else 'disabled'}")
    
    def set_emergency_mode(self, enabled: bool) -> None:
        """Enable or disable emergency mode"""
        self.emergency_mode = enabled
        self.logger.warning(f"Emergency mode {'enabled' if enabled else 'disabled'}")
