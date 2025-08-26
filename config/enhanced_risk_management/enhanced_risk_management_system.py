"""
DSSMS Phase 2 Task 2.3: Enhanced Risk Management System
Main Integration System

This module provides the main integration point for the enhanced risk management system.
Coordinates all components and provides a unified interface for risk monitoring and control.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import pandas as pd

from .unified_risk_monitor import UnifiedRiskMonitor, RiskEvent
from .risk_alert_manager import RiskAlertManager
from .automated_risk_actions import AutomatedRiskActions, RiskAction
from .risk_metrics_calculator import RiskMetricsCalculator, RiskMetrics
from .risk_threshold_manager import RiskThresholdManager


class EnhancedRiskManagementSystem:
    """
    Main enhanced risk management system that coordinates all components.
    Provides unified interface for comprehensive risk monitoring and control.
    """
    
    def __init__(self, 
                 config_dir: str = "config/enhanced_risk_management/configs",
                 existing_portfolio_manager=None,
                 existing_drawdown_controller=None):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
        # Component initialization
        self.risk_monitor = UnifiedRiskMonitor(
            config_path=str(self.config_dir / "risk_thresholds.json")
        )
        
        self.alert_manager = RiskAlertManager(config_path=str(self.config_dir / "alert_rules.json"))
        self.action_manager = AutomatedRiskActions(config_dir=str(self.config_dir))
        self.metrics_calculator = RiskMetricsCalculator()
        self.threshold_manager = RiskThresholdManager(config_dir=str(self.config_dir))
        
        # System state
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.last_comprehensive_check = datetime.now()
        
        # Configuration
        self.config = self._load_system_config()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'risk_event': [],
            'action_completed': [],
            'threshold_adjusted': [],
            'system_error': []
        }
        
        # System metrics tracking
        self.system_metrics = {
            'total_risk_events': 0,
            'total_actions_taken': 0,
            'total_alerts_sent': 0,
            'system_uptime': datetime.now(),
            'last_error': None
        }
        
        # Connect alert manager to action manager
        self.alert_manager.add_event_handler(self._handle_alert_event)
        
        self.logger.info("EnhancedRiskManagementSystem initialized successfully")
    
    def _load_system_config(self) -> Dict[str, Any]:
        """Load system-wide configuration"""
        config_file = self.config_dir / "system_config.json"
        try:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Default system configuration
                default_config = {
                    "monitoring_interval_seconds": 10,
                    "comprehensive_check_interval_minutes": 5,
                    "enable_automated_actions": True,
                    "enable_threshold_adjustments": True,
                    "max_concurrent_actions": 3,
                    "system_health_check_interval_minutes": 1,
                    "data_retention_days": 30,
                    "log_level": "INFO",
                    "performance_monitoring": {
                        "track_execution_times": True,
                        "alert_on_slow_operations": True,
                        "max_operation_time_seconds": 30
                    }
                }
                # Save default configuration
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
        except Exception as e:
            self.logger.error(f"Failed to load system config: {e}")
            return {}
    
    def start_monitoring(self) -> bool:
        """Start the risk monitoring system"""
        try:
            if self.is_running:
                self.logger.warning("Risk monitoring system is already running")
                return False
            
            self.is_running = True
            self.system_metrics['system_uptime'] = datetime.now()
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="RiskMonitoring",
                daemon=True
            )
            self.monitoring_thread.start()
            
            # Start individual components
            self.risk_monitor.start_monitoring()
            
            self.logger.info("Enhanced Risk Management System monitoring started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self.is_running = False
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop the risk monitoring system"""
        try:
            if not self.is_running:
                self.logger.warning("Risk monitoring system is not running")
                return False
            
            self.is_running = False
            
            # Stop individual components
            self.risk_monitor.stop_monitoring()
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            self.logger.info("Enhanced Risk Management System monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that coordinates all components"""
        while self.is_running:
            try:
                start_time = datetime.now()
                
                # Check if comprehensive analysis is due
                if self._should_run_comprehensive_check():
                    self._run_comprehensive_risk_check()
                
                # System health check
                self._perform_system_health_check()
                
                # Process any pending approvals
                self._process_pending_approvals()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Calculate sleep time
                execution_time = (datetime.now() - start_time).total_seconds()
                sleep_time = max(0, self.config.get("monitoring_interval_seconds", 10) - execution_time)
                
                # Log performance if enabled
                if self.config.get("performance_monitoring", {}).get("track_execution_times", True):
                    if execution_time > self.config.get("performance_monitoring", {}).get("max_operation_time_seconds", 30):
                        self.logger.warning(f"Monitoring loop took {execution_time:.2f}s (longer than expected)")
                
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.system_metrics['last_error'] = {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
                time.sleep(5)  # Brief pause on error
    
    def _should_run_comprehensive_check(self) -> bool:
        """Check if comprehensive risk analysis should be run"""
        check_interval = self.config.get("comprehensive_check_interval_minutes", 5)
        time_since_last = datetime.now() - self.last_comprehensive_check
        return time_since_last >= timedelta(minutes=check_interval)
    
    def _run_comprehensive_risk_check(self) -> None:
        """Run comprehensive risk analysis and take appropriate actions"""
        try:
            self.logger.debug("Running comprehensive risk check")
            
            # Get current portfolio state (placeholder - would integrate with actual portfolio)
            current_portfolio_value = 1000000.0  # Placeholder
            current_positions = {"AAPL": 0.2, "GOOGL": 0.3, "MSFT": 0.3, "CASH": 0.2}  # Placeholder
            
            # Calculate comprehensive metrics
            risk_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                current_portfolio_value, current_positions
            )
            
            # Update threshold manager with current metrics
            self.threshold_manager.adjust_thresholds(risk_metrics)
            
            # Get current thresholds
            current_thresholds = self.threshold_manager.get_current_thresholds()
            
            # Check for risk threshold breaches
            risk_events = self._check_risk_thresholds(risk_metrics, current_thresholds)
            
            # Process any detected risk events
            for risk_event in risk_events:
                self._process_risk_event(risk_event)
            
            self.last_comprehensive_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to run comprehensive risk check: {e}")
    
    def _check_risk_thresholds(self, metrics: RiskMetrics, thresholds) -> List[RiskEvent]:
        """Check risk metrics against thresholds and generate events"""
        risk_events = []
        
        if not thresholds:
            return risk_events
        
        try:
            # Check drawdown threshold
            if metrics.current_drawdown > thresholds.max_drawdown:
                risk_events.append(RiskEvent(
                    event_id=f"drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    risk_type="drawdown",
                    severity="high" if metrics.current_drawdown > thresholds.max_drawdown * 1.5 else "medium",
                    current_value=metrics.current_drawdown,
                    threshold=thresholds.max_drawdown,
                    timestamp=datetime.now(),
                    description=f"Portfolio drawdown {metrics.current_drawdown:.2%} exceeds threshold {thresholds.max_drawdown:.2%}",
                    metadata={"portfolio_value": metrics.portfolio_value}
                ))
            
            # Check VaR threshold
            if metrics.var_95 > thresholds.var_95_threshold:
                risk_events.append(RiskEvent(
                    event_id=f"var_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    risk_type="var_breach",
                    severity="high",
                    current_value=metrics.var_95,
                    threshold=thresholds.var_95_threshold,
                    timestamp=datetime.now(),
                    description=f"VaR 95% {metrics.var_95:.2%} exceeds threshold {thresholds.var_95_threshold:.2%}",
                    metadata={"cvar_95": metrics.cvar_95}
                ))
            
            # Check volatility threshold
            if metrics.annualized_volatility > thresholds.volatility_threshold:
                risk_events.append(RiskEvent(
                    event_id=f"volatility_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    risk_type="volatility",
                    severity="medium",
                    current_value=metrics.annualized_volatility,
                    threshold=thresholds.volatility_threshold,
                    timestamp=datetime.now(),
                    description=f"Portfolio volatility {metrics.annualized_volatility:.2%} exceeds threshold {thresholds.volatility_threshold:.2%}",
                    metadata={"rolling_vol_30d": metrics.rolling_volatility_30d}
                ))
            
            # Check concentration risk
            if metrics.concentration_risk > thresholds.concentration_threshold:
                risk_events.append(RiskEvent(
                    event_id=f"concentration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    risk_type="concentration_risk",
                    severity="medium",
                    current_value=metrics.concentration_risk,
                    threshold=thresholds.concentration_threshold,
                    timestamp=datetime.now(),
                    description=f"Portfolio concentration {metrics.concentration_risk:.2%} exceeds threshold {thresholds.concentration_threshold:.2%}",
                    metadata={"max_position_weight": metrics.max_position_weight}
                ))
            
            # Check correlation risk
            if metrics.max_correlation > thresholds.correlation_threshold:
                risk_events.append(RiskEvent(
                    event_id=f"correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    risk_type="correlation_risk",
                    severity="medium",
                    current_value=metrics.max_correlation,
                    threshold=thresholds.correlation_threshold,
                    timestamp=datetime.now(),
                    description=f"Maximum correlation {metrics.max_correlation:.2%} exceeds threshold {thresholds.correlation_threshold:.2%}",
                    metadata={"average_correlation": metrics.average_correlation}
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to check risk thresholds: {e}")
        
        return risk_events
    
    def _process_risk_event(self, risk_event: RiskEvent) -> None:
        """Process a detected risk event"""
        try:
            self.system_metrics['total_risk_events'] += 1
            
            # Send to alert manager
            self.alert_manager.process_risk_event(risk_event)
            
            # Consider automated action if enabled
            if self.config.get("enable_automated_actions", True):
                action = self.action_manager.process_risk_event(risk_event)
                if action:
                    self.system_metrics['total_actions_taken'] += 1
                    self.logger.info(f"Automated action initiated: {action.action_id}")
            
            # Trigger event handlers
            self._trigger_event_handlers('risk_event', risk_event)
            
        except Exception as e:
            self.logger.error(f"Failed to process risk event: {e}")
    
    def _handle_alert_event(self, alert_data: Dict[str, Any]) -> None:
        """Handle alert events from alert manager"""
        try:
            self.system_metrics['total_alerts_sent'] += 1
            self.logger.debug(f"Alert processed: {alert_data.get('alert_type', 'unknown')}")
        except Exception as e:
            self.logger.error(f"Failed to handle alert event: {e}")
    
    def _perform_system_health_check(self) -> None:
        """Perform system health check"""
        try:
            # Check component health
            components_status = {
                'risk_monitor': self.risk_monitor.is_monitoring,
                'alert_manager': True,  # Placeholder
                'action_manager': True,  # Placeholder
                'metrics_calculator': True,  # Placeholder
                'threshold_manager': True  # Placeholder
            }
            
            # Log any issues
            for component, status in components_status.items():
                if not status:
                    self.logger.warning(f"Component {component} is not healthy")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def _process_pending_approvals(self) -> None:
        """Process any pending action approvals"""
        try:
            pending_approvals = self.action_manager.get_pending_approvals()
            if pending_approvals:
                self.logger.info(f"{len(pending_approvals)} actions awaiting approval")
        except Exception as e:
            self.logger.error(f"Failed to process pending approvals: {e}")
    
    def _update_system_metrics(self) -> None:
        """Update system performance metrics"""
        try:
            # Update uptime
            uptime = datetime.now() - self.system_metrics['system_uptime']
            self.system_metrics['current_uptime_hours'] = uptime.total_seconds() / 3600
        except Exception as e:
            self.logger.error(f"Failed to update system metrics: {e}")
    
    def _trigger_event_handlers(self, event_type: str, event_data: Any) -> None:
        """Trigger registered event handlers"""
        try:
            handlers = self.event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    handler(event_data)
                except Exception as e:
                    self.logger.error(f"Event handler failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to trigger event handlers: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """Remove an event handler"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get component statuses
            component_status = {
                'risk_monitor': {
                    'running': self.risk_monitor.is_monitoring,
                    'last_check': self.risk_monitor.last_check_time.isoformat() if hasattr(self.risk_monitor, 'last_check_time') else None
                },
                'alert_manager': {
                    'active_alerts': len(getattr(self.alert_manager, 'active_alerts', {})),
                    'total_alerts_today': self.system_metrics.get('total_alerts_sent', 0)
                },
                'action_manager': {
                    'pending_approvals': len(self.action_manager.get_pending_approvals()),
                    'active_actions': len(getattr(self.action_manager, 'active_actions', {})),
                    'automation_enabled': getattr(self.action_manager, 'automation_enabled', False)
                },
                'threshold_manager': {
                    'active_set': self.threshold_manager.active_threshold_id,
                    'auto_adjustment': self.threshold_manager.config.get('auto_adjustment_enabled', False),
                    'current_regime': self.threshold_manager.current_regime.value
                }
            }
            
            return {
                'system_running': self.is_running,
                'uptime_hours': self.system_metrics.get('current_uptime_hours', 0),
                'total_risk_events': self.system_metrics.get('total_risk_events', 0),
                'total_actions_taken': self.system_metrics.get('total_actions_taken', 0),
                'total_alerts_sent': self.system_metrics.get('total_alerts_sent', 0),
                'last_comprehensive_check': self.last_comprehensive_check.isoformat(),
                'components': component_status,
                'last_error': self.system_metrics.get('last_error'),
                'configuration': {
                    'monitoring_interval': self.config.get('monitoring_interval_seconds', 10),
                    'automated_actions_enabled': self.config.get('enable_automated_actions', True),
                    'threshold_adjustments_enabled': self.config.get('enable_threshold_adjustments', True)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def manual_risk_check(self) -> Dict[str, Any]:
        """Manually trigger a comprehensive risk check"""
        try:
            self.logger.info("Manual risk check triggered")
            self._run_comprehensive_risk_check()
            return {'success': True, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Manual risk check failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_portfolio_data(self, 
                            portfolio_values: pd.Series,
                            position_weights: pd.DataFrame,
                            price_data: Dict[str, pd.Series]) -> bool:
        """Update portfolio data for risk calculations"""
        try:
            # Update metrics calculator
            self.metrics_calculator.update_portfolio_values(portfolio_values)
            self.metrics_calculator.update_position_weights(position_weights)
            
            for symbol, prices in price_data.items():
                self.metrics_calculator.update_price_data(symbol, prices)
            
            # Update threshold manager with market data if available
            if 'market' in price_data:
                market_returns = price_data['market'].pct_change().dropna()
                portfolio_returns = portfolio_values.pct_change().dropna()
                self.threshold_manager.update_market_data(market_returns, portfolio_returns)
            
            self.logger.debug("Portfolio data updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio data: {e}")
            return False
    
    def set_risk_free_rate(self, rate: float) -> None:
        """Set risk-free rate for calculations"""
        self.metrics_calculator.risk_free_rate = rate
    
    def emergency_stop(self) -> bool:
        """Emergency stop of all automated actions"""
        try:
            self.action_manager.set_automation_enabled(False)
            self.action_manager.set_emergency_mode(True)
            self.logger.critical("Emergency stop activated - all automation disabled")
            return True
        except Exception as e:
            self.logger.error(f"Failed to execute emergency stop: {e}")
            return False
    
    def resume_automation(self) -> bool:
        """Resume automated operations after emergency stop"""
        try:
            self.action_manager.set_emergency_mode(False)
            self.action_manager.set_automation_enabled(True)
            self.logger.info("Automation resumed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to resume automation: {e}")
            return False
