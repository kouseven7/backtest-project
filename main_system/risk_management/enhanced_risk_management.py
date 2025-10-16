"""
DSSMS Phase 2 Task 2.3: Enhanced Risk Management System
Demo and Test Script

This script demonstrates the enhanced risk management system functionality
and provides comprehensive testing of all components.
"""

import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import the enhanced risk management system
from config.enhanced_risk_management import (
    UnifiedRiskMonitor,
    RiskAlertManager, 
    AutomatedRiskActions,
    RiskMetricsCalculator,
    RiskThresholdManager
)
from config.enhanced_risk_management.enhanced_risk_management_system import EnhancedRiskManagementSystem


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('demo_enhanced_risk_management.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def generate_sample_data():
    """Generate sample portfolio and market data for testing"""
    logger = logging.getLogger(__name__)
    logger.info("Generating sample data for testing...")
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate sample portfolio values (with some volatility and drawdowns)
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
    
    # Add some trend and volatility clustering
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Some autocorrelation
    
    # Create significant drawdown period
    drawdown_start = len(returns) // 3
    drawdown_end = drawdown_start + 30
    returns[drawdown_start:drawdown_end] = np.random.normal(-0.005, 0.03, drawdown_end - drawdown_start)
    
    # Calculate portfolio values
    portfolio_values = pd.Series(index=dates, data=100000.0)
    for i in range(1, len(dates)):
        portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + returns[i])
    
    # Generate sample position weights
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    position_data = {}
    for symbol in symbols:
        # Generate time-varying weights
        base_weight = 0.15 + np.random.normal(0, 0.05, len(dates))
        base_weight = np.clip(base_weight, 0.05, 0.35)
        position_data[symbol] = base_weight
    
    # Add cash position
    position_data['CASH'] = 1.0 - sum(position_data.values())
    
    position_weights = pd.DataFrame(position_data, index=dates)
    
    # Normalize weights to sum to 1
    position_weights = position_weights.div(position_weights.sum(axis=1), axis=0)
    
    # Generate sample price data for individual stocks
    price_data = {}
    for symbol in symbols:
        symbol_returns = np.random.normal(0.0005, 0.025, len(dates))
        symbol_prices = pd.Series(index=dates, data=100.0)
        for i in range(1, len(dates)):
            symbol_prices.iloc[i] = symbol_prices.iloc[i-1] * (1 + symbol_returns[i])
        price_data[symbol] = symbol_prices
    
    # Add market benchmark
    market_returns = np.random.normal(0.0006, 0.018, len(dates))
    market_prices = pd.Series(index=dates, data=1000.0)
    for i in range(1, len(dates)):
        market_prices.iloc[i] = market_prices.iloc[i-1] * (1 + market_returns[i])
    price_data['market'] = market_prices
    
    logger.info(f"Generated data for {len(dates)} days, {len(symbols)} positions")
    logger.info(f"Portfolio value range: ${portfolio_values.min():,.0f} - ${portfolio_values.max():,.0f}")
    
    return portfolio_values, position_weights, price_data


def test_risk_metrics_calculator(portfolio_values, position_weights, price_data):
    """Test the risk metrics calculator"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Risk Metrics Calculator...")
    
    try:
        calculator = RiskMetricsCalculator(lookback_days=252)
        
        # Update data
        calculator.update_portfolio_values(portfolio_values)
        calculator.update_position_weights(position_weights)
        
        for symbol, prices in price_data.items():
            calculator.update_price_data(symbol, prices)
        
        # Set benchmark
        calculator.set_benchmark_returns(price_data['market'].pct_change().dropna())
        
        # Calculate comprehensive metrics
        current_positions = position_weights.iloc[-1].to_dict()
        current_value = portfolio_values.iloc[-1]
        
        metrics = calculator.calculate_comprehensive_metrics(current_value, current_positions)
        
        # Display key metrics
        logger.info("=== Risk Metrics Results ===")
        logger.info(f"Portfolio Value: ${metrics.portfolio_value:,.0f}")
        logger.info(f"Daily Return: {metrics.daily_return:.4f}")
        logger.info(f"Annualized Volatility: {metrics.annualized_volatility:.2%}")
        logger.info(f"Current Drawdown: {metrics.current_drawdown:.2%}")
        logger.info(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"VaR 95%: {metrics.var_95:.2%}")
        logger.info(f"Concentration Risk: {metrics.concentration_risk:.2%}")
        logger.info(f"Max Correlation: {metrics.max_correlation:.2%}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Risk metrics calculation failed: {e}")
        return None


def test_threshold_manager(metrics):
    """Test the risk threshold manager"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Risk Threshold Manager...")
    
    try:
        threshold_manager = RiskThresholdManager()
        
        # Get current thresholds
        current_thresholds = threshold_manager.get_current_thresholds()
        if current_thresholds:
            logger.info("=== Current Risk Thresholds ===")
            logger.info(f"Max Drawdown: {current_thresholds.max_drawdown:.2%}")
            logger.info(f"VaR 95% Threshold: {current_thresholds.var_95_threshold:.2%}")
            logger.info(f"Volatility Threshold: {current_thresholds.volatility_threshold:.2%}")
            logger.info(f"Concentration Threshold: {current_thresholds.concentration_threshold:.2%}")
        
        # Test threshold adjustment
        if metrics:
            adjusted = threshold_manager.adjust_thresholds(metrics)
            if adjusted:
                logger.info("Thresholds were adjusted based on current metrics")
            else:
                logger.info("No threshold adjustments were needed")
        
        # Get threshold summary
        summary = threshold_manager.get_threshold_summary()
        logger.info(f"Active threshold set: {summary['active_threshold_set']}")
        logger.info(f"Current market regime: {summary['current_regime']}")
        
        return threshold_manager
        
    except Exception as e:
        logger.error(f"Threshold manager test failed: {e}")
        return None


def test_alert_manager():
    """Test the risk alert manager"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Risk Alert Manager...")
    
    try:
        alert_manager = RiskAlertManager(config_path="config/enhanced_risk_management/configs/alert_rules.json")
        
        # Import the enhanced risk management system
        from config.enhanced_risk_management.unified_risk_monitor import RiskEvent
        
        test_event = RiskEvent(
            timestamp=datetime.now(),
            event_type="drawdown",
            severity="high",
            description="Test drawdown event for demonstration",
            portfolio_value=950000,
            drawdown=0.12,
            var_breach=False,
            correlation_risk=0.0,
            recommendation="Consider position reduction",
            requires_action=True
        )
        
        # Process the test event
        notifications = alert_manager.process_alert(test_event)
        
        # Check alert status
        summary = alert_manager.get_alert_summary()
        logger.info("=== Alert Manager Status ===")
        logger.info(f"Active alerts: {summary['active_alerts']}")
        logger.info(f"Total alerts today: {summary['total_alerts_today']}")
        logger.info(f"Escalated alerts: {summary['escalated_alerts']}")
        
        return alert_manager
        
    except Exception as e:
        logger.error(f"Alert manager test failed: {e}")
        return None


def test_action_manager():
    """Test the automated risk actions manager"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Automated Risk Actions Manager...")
    
    try:
        action_manager = AutomatedRiskActions()
        
        # Create a test risk event that should trigger an action
        from config.enhanced_risk_management.unified_risk_monitor import RiskEvent
        
        test_event = RiskEvent(
            timestamp=datetime.now(),
            event_type="drawdown",
            severity="high", 
            description="Test drawdown for action trigger",
            portfolio_value=920000,
            drawdown=0.08,
            var_breach=False,
            correlation_risk=0.0,
            recommendation="Position reduction recommended",
            requires_action=True
        )
        
        # Process the test event
        action = action_manager.process_risk_event(test_event)
        
        if action:
            logger.info("=== Automated Action Created ===")
            logger.info(f"Action ID: {action.action_id}")
            logger.info(f"Action Type: {action.action_type.value}")
            logger.info(f"Status: {action.status.value}")
            logger.info(f"Requires Approval: {action.requires_approval}")
        
        # Get action summary
        summary = action_manager.get_action_summary()
        logger.info("=== Action Manager Status ===")
        logger.info(f"Active actions: {summary['active_actions']}")
        logger.info(f"Pending approvals: {summary['pending_approvals']}")
        logger.info(f"Completed today: {summary['completed_today']}")
        logger.info(f"Automation enabled: {summary['automation_enabled']}")
        
        return action_manager
        
    except Exception as e:
        logger.error(f"Action manager test failed: {e}")
        return None


def test_unified_system(portfolio_values, position_weights, price_data):
    """Test the complete enhanced risk management system"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Complete Enhanced Risk Management System...")
    
    try:
        # Initialize the complete system
        risk_system = EnhancedRiskManagementSystem(
            config_dir="config/enhanced_risk_management/configs"
        )
        
        # Update portfolio data
        risk_system.update_portfolio_data(portfolio_values, position_weights, price_data)
        
        # Start monitoring
        if risk_system.start_monitoring():
            logger.info("Risk monitoring system started successfully")
        
        # Run manual risk check
        check_result = risk_system.manual_risk_check()
        if check_result['success']:
            logger.info("Manual risk check completed successfully")
        
        # Get system status
        status = risk_system.get_system_status()
        logger.info("=== System Status ===")
        logger.info(f"System running: {status['system_running']}")
        logger.info(f"Uptime hours: {status['uptime_hours']:.2f}")
        logger.info(f"Total risk events: {status['total_risk_events']}")
        logger.info(f"Total actions taken: {status['total_actions_taken']}")
        logger.info(f"Total alerts sent: {status['total_alerts_sent']}")
        
        # Component status
        logger.info("=== Component Status ===")
        for component, comp_status in status['components'].items():
            logger.info(f"{component}: {comp_status}")
        
        # Stop monitoring
        if risk_system.stop_monitoring():
            logger.info("Risk monitoring system stopped successfully")
        
        return risk_system
        
    except Exception as e:
        logger.error(f"Unified system test failed: {e}")
        return None


def main():
    """Main demo function"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("DSSMS Phase 2 Task 2.3: Enhanced Risk Management System Demo")
    logger.info("=" * 60)
    
    try:
        # Generate sample data
        portfolio_values, position_weights, price_data = generate_sample_data()
        
        # Test individual components
        logger.info("\n" + "=" * 40)
        logger.info("COMPONENT TESTING")
        logger.info("=" * 40)
        
        # Test 1: Risk Metrics Calculator
        metrics = test_risk_metrics_calculator(portfolio_values, position_weights, price_data)
        
        # Test 2: Threshold Manager
        threshold_manager = test_threshold_manager(metrics)
        
        # Test 3: Alert Manager
        alert_manager = test_alert_manager()
        
        # Test 4: Action Manager
        action_manager = test_action_manager()
        
        # Test 5: Complete System Integration
        logger.info("\n" + "=" * 40)
        logger.info("SYSTEM INTEGRATION TESTING")
        logger.info("=" * 40)
        
        risk_system = test_unified_system(portfolio_values, position_weights, price_data)
        
        # Summary
        logger.info("\n" + "=" * 40)
        logger.info("DEMO SUMMARY")
        logger.info("=" * 40)
        
        components_tested = [
            ("Risk Metrics Calculator", metrics is not None),
            ("Risk Threshold Manager", threshold_manager is not None),
            ("Risk Alert Manager", alert_manager is not None),
            ("Automated Risk Actions", action_manager is not None),
            ("Complete System Integration", risk_system is not None)
        ]
        
        for component, success in components_tested:
            status = "✓ PASS" if success else "✗ FAIL"
            logger.info(f"{component}: {status}")
        
        all_passed = all(success for _, success in components_tested)
        
        if all_passed:
            logger.info("\n[SUCCESS] All tests passed! Enhanced Risk Management System is working correctly.")
        else:
            logger.warning("\n[WARNING]  Some tests failed. Please check the logs for details.")
        
        logger.info("Demo completed successfully!")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
