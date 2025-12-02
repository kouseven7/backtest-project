"""
DSSMS Phase 2 Task 2.3: Enhanced Risk Management System
Component: Risk Threshold Manager

This module manages dynamic risk thresholds with adaptive adjustment capabilities.
Provides intelligent threshold management based on market conditions and portfolio performance.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from .risk_metrics_calculator import RiskMetrics


class ThresholdAdjustmentMode(Enum):
    """Threshold adjustment modes"""
    STATIC = "static"
    ADAPTIVE = "adaptive"
    MARKET_REGIME = "market_regime"
    VOLATILITY_BASED = "volatility_based"


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class ThresholdSet:
    """Risk threshold configuration set"""
    threshold_id: str
    name: str
    description: str
    active: bool
    created_time: datetime
    last_updated: datetime
    
    # Portfolio-level thresholds
    max_drawdown: float
    var_95_threshold: float
    volatility_threshold: float
    concentration_threshold: float
    correlation_threshold: float
    
    # Position-level thresholds
    max_position_size: float
    stop_loss_threshold: float
    profit_target_threshold: Optional[float]
    
    # Performance thresholds
    min_sharpe_ratio: float
    max_tracking_error: float
    
    # Risk adjustment parameters
    adjustment_mode: ThresholdAdjustmentMode
    volatility_lookback: int
    adjustment_frequency: int  # days
    max_adjustment_factor: float
    
    # Market regime specific thresholds
    regime_specific: Dict[str, Dict[str, float]]


class RiskThresholdManager:
    """
    Intelligent risk threshold management system with adaptive capabilities.
    Manages multiple threshold sets and provides dynamic threshold adjustment.
    """
    
    def __init__(self, config_dir: str = "config/enhanced_risk_management/configs"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
        # Threshold storage
        self.threshold_sets: Dict[str, ThresholdSet] = {}
        self.active_threshold_id: Optional[str] = None
        self.threshold_history: List[Dict[str, Any]] = []
        
        # Market data for adaptive adjustments
        self.market_data: pd.DataFrame = pd.DataFrame()
        self.volatility_history: pd.Series = pd.Series(dtype=float)
        self.current_regime: MarketRegime = MarketRegime.SIDEWAYS
        
        # Configuration
        self.config = self._load_config()
        
        # Load existing threshold sets
        self._load_threshold_sets()
        
        # Adjustment tracking
        self.last_adjustment_time: Optional[datetime] = None
        self.adjustment_log: List[Dict[str, Any]] = []
        
        self.logger.info("RiskThresholdManager initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load threshold manager configuration"""
        config_file = self.config_dir / "threshold_config.json"
        try:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Default configuration
                default_config = {
                    "default_threshold_set": "balanced",
                    "auto_adjustment_enabled": True,
                    "adjustment_sensitivity": 0.1,
                    "min_adjustment_interval_hours": 6,
                    "regime_detection_enabled": True,
                    "volatility_window": 30,
                    "regime_lookback_days": 60,
                    "threshold_bounds": {
                        "max_drawdown": {"min": 0.02, "max": 0.25},
                        "var_95_threshold": {"min": 0.01, "max": 0.10},
                        "volatility_threshold": {"min": 0.10, "max": 0.50},
                        "concentration_threshold": {"min": 0.10, "max": 0.50}
                    }
                }
                # Save default configuration
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
        except Exception as e:
            self.logger.error(f"Failed to load threshold config: {e}")
            return {}
    
    def _load_threshold_sets(self) -> None:
        """Load existing threshold sets from storage"""
        try:
            threshold_file = self.config_dir / "threshold_sets.json"
            if threshold_file.exists():
                with open(threshold_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for set_data in data.get("threshold_sets", []):
                    threshold_set = self._dict_to_threshold_set(set_data)
                    self.threshold_sets[threshold_set.threshold_id] = threshold_set
                
                self.active_threshold_id = data.get("active_threshold_id")
                self.threshold_history = data.get("threshold_history", [])
                
                self.logger.info(f"Loaded {len(self.threshold_sets)} threshold sets")
            else:
                # Create default threshold sets
                self._create_default_threshold_sets()
                
        except Exception as e:
            self.logger.error(f"Failed to load threshold sets: {e}")
            self._create_default_threshold_sets()
    
    def _create_default_threshold_sets(self) -> None:
        """Create default threshold sets"""
        try:
            # Conservative threshold set
            conservative = ThresholdSet(
                threshold_id="conservative",
                name="Conservative Risk Profile",
                description="Low risk tolerance with tight controls",
                active=False,
                created_time=datetime.now(),
                last_updated=datetime.now(),
                max_drawdown=0.05,
                var_95_threshold=0.02,
                volatility_threshold=0.15,
                concentration_threshold=0.20,
                correlation_threshold=0.70,
                max_position_size=0.10,
                stop_loss_threshold=0.03,
                profit_target_threshold=0.05,
                min_sharpe_ratio=0.8,
                max_tracking_error=0.05,
                adjustment_mode=ThresholdAdjustmentMode.ADAPTIVE,
                volatility_lookback=30,
                adjustment_frequency=7,
                max_adjustment_factor=1.5,
                regime_specific={
                    "bull": {"max_drawdown": 0.07, "volatility_threshold": 0.18},
                    "bear": {"max_drawdown": 0.03, "volatility_threshold": 0.12},
                    "high_volatility": {"var_95_threshold": 0.015, "concentration_threshold": 0.15}
                }
            )
            
            # Balanced threshold set
            balanced = ThresholdSet(
                threshold_id="balanced",
                name="Balanced Risk Profile", 
                description="Moderate risk tolerance with balanced controls",
                active=True,
                created_time=datetime.now(),
                last_updated=datetime.now(),
                max_drawdown=0.10,
                var_95_threshold=0.03,
                volatility_threshold=0.25,
                concentration_threshold=0.30,
                correlation_threshold=0.80,
                max_position_size=0.15,
                stop_loss_threshold=0.05,
                profit_target_threshold=0.10,
                min_sharpe_ratio=0.5,
                max_tracking_error=0.08,
                adjustment_mode=ThresholdAdjustmentMode.MARKET_REGIME,
                volatility_lookback=30,
                adjustment_frequency=5,
                max_adjustment_factor=2.0,
                regime_specific={
                    "bull": {"max_drawdown": 0.12, "volatility_threshold": 0.30},
                    "bear": {"max_drawdown": 0.08, "volatility_threshold": 0.20},
                    "high_volatility": {"var_95_threshold": 0.025, "concentration_threshold": 0.25}
                }
            )
            
            # Aggressive threshold set
            aggressive = ThresholdSet(
                threshold_id="aggressive",
                name="Aggressive Risk Profile",
                description="High risk tolerance with looser controls",
                active=False,
                created_time=datetime.now(),
                last_updated=datetime.now(),
                max_drawdown=0.20,
                var_95_threshold=0.05,
                volatility_threshold=0.40,
                concentration_threshold=0.40,
                correlation_threshold=0.85,
                max_position_size=0.25,
                stop_loss_threshold=0.08,
                profit_target_threshold=0.15,
                min_sharpe_ratio=0.3,
                max_tracking_error=0.12,
                adjustment_mode=ThresholdAdjustmentMode.VOLATILITY_BASED,
                volatility_lookback=20,
                adjustment_frequency=3,
                max_adjustment_factor=2.5,
                regime_specific={
                    "bull": {"max_drawdown": 0.25, "volatility_threshold": 0.45},
                    "bear": {"max_drawdown": 0.15, "volatility_threshold": 0.35},
                    "high_volatility": {"var_95_threshold": 0.04, "concentration_threshold": 0.35}
                }
            )
            
            # Store threshold sets
            self.threshold_sets = {
                "conservative": conservative,
                "balanced": balanced,
                "aggressive": aggressive
            }
            
            self.active_threshold_id = "balanced"
            
            # Save to file
            self._save_threshold_sets()
            
            self.logger.info("Created default threshold sets")
            
        except Exception as e:
            self.logger.error(f"Failed to create default threshold sets: {e}")
    
    def get_current_thresholds(self) -> Optional[ThresholdSet]:
        """Get currently active threshold set"""
        if self.active_threshold_id and self.active_threshold_id in self.threshold_sets:
            return self.threshold_sets[self.active_threshold_id]
        return None
    
    def set_active_threshold_set(self, threshold_id: str) -> bool:
        """Set the active threshold set"""
        try:
            if threshold_id not in self.threshold_sets:
                self.logger.error(f"Threshold set {threshold_id} not found")
                return False
            
            # Deactivate current set
            if self.active_threshold_id:
                current_set = self.threshold_sets[self.active_threshold_id]
                current_set.active = False
            
            # Activate new set
            new_set = self.threshold_sets[threshold_id]
            new_set.active = True
            self.active_threshold_id = threshold_id
            
            # Log the change
            self.threshold_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "threshold_set_changed",
                "from_set": self.active_threshold_id,
                "to_set": threshold_id,
                "reason": "manual_selection"
            })
            
            # Save changes
            self._save_threshold_sets()
            
            self.logger.info(f"Activated threshold set: {threshold_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set active threshold set: {e}")
            return False
    
    def update_market_data(self, market_returns: pd.Series, portfolio_returns: pd.Series) -> None:
        """Update market data for regime detection and adaptive thresholds"""
        try:
            # Store market data
            aligned_market, aligned_portfolio = market_returns.align(portfolio_returns, join='inner')
            
            self.market_data = pd.DataFrame({
                'market_returns': aligned_market,
                'portfolio_returns': aligned_portfolio
            })
            
            # Update volatility history
            if len(aligned_market) > 1:
                volatility_window = self.config.get("volatility_window", 30)
                self.volatility_history = aligned_market.rolling(window=volatility_window).std()
            
            # Detect market regime
            self._detect_market_regime()
            
            self.logger.debug("Updated market data for threshold management")
            
        except Exception as e:
            self.logger.error(f"Failed to update market data: {e}")
    
    def _detect_market_regime(self) -> None:
        """Detect current market regime based on market data"""
        try:
            if len(self.market_data) < 60:  # Need minimum data
                return
            
            lookback_days = self.config.get("regime_lookback_days", 60)
            recent_data = self.market_data.tail(lookback_days)
            
            # Calculate regime indicators
            returns = recent_data['market_returns']
            volatility = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else returns.std()
            
            # Trend analysis
            cumulative_return = (1 + returns).prod() - 1
            trend_strength = returns.rolling(20).mean().iloc[-1] if len(returns) > 20 else returns.mean()
            
            # Volatility analysis
            vol_percentile = self._calculate_volatility_percentile(volatility)
            
            # Regime classification logic
            if vol_percentile > 0.8:
                self.current_regime = MarketRegime.HIGH_VOLATILITY
            elif vol_percentile < 0.2:
                self.current_regime = MarketRegime.LOW_VOLATILITY
            elif trend_strength > 0.001:  # Strong positive trend
                self.current_regime = MarketRegime.BULL
            elif trend_strength < -0.001:  # Strong negative trend
                self.current_regime = MarketRegime.BEAR
            else:
                self.current_regime = MarketRegime.SIDEWAYS
            
            self.logger.debug(f"Detected market regime: {self.current_regime.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to detect market regime: {e}")
    
    def _calculate_volatility_percentile(self, current_volatility: float) -> float:
        """Calculate volatility percentile based on historical data"""
        try:
            if len(self.volatility_history) < 100:
                return 0.5  # Default to median if insufficient data
            
            # Calculate percentile of current volatility
            percentile = (self.volatility_history < current_volatility).mean()
            return percentile
            
        except Exception as e:
            self.logger.error(f"Failed to calculate volatility percentile: {e}")
            return 0.5
    
    def adjust_thresholds(self, current_metrics: RiskMetrics) -> bool:
        """Adjust thresholds based on current market conditions and metrics"""
        try:
            if not self.config.get("auto_adjustment_enabled", True):
                return False
            
            current_set = self.get_current_thresholds()
            if not current_set:
                return False
            
            # Check if adjustment is due
            if not self._should_adjust_thresholds():
                return False
            
            # Perform adjustment based on mode
            if current_set.adjustment_mode == ThresholdAdjustmentMode.STATIC:
                return False  # No adjustment for static mode
            elif current_set.adjustment_mode == ThresholdAdjustmentMode.ADAPTIVE:
                return self._adaptive_adjustment(current_set, current_metrics)
            elif current_set.adjustment_mode == ThresholdAdjustmentMode.MARKET_REGIME:
                return self._regime_based_adjustment(current_set)
            elif current_set.adjustment_mode == ThresholdAdjustmentMode.VOLATILITY_BASED:
                return self._volatility_based_adjustment(current_set, current_metrics)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to adjust thresholds: {e}")
            return False
    
    def _should_adjust_thresholds(self) -> bool:
        """Check if threshold adjustment is due"""
        if not self.last_adjustment_time:
            return True
        
        min_interval = self.config.get("min_adjustment_interval_hours", 6)
        time_since_last = datetime.now() - self.last_adjustment_time
        
        return time_since_last >= timedelta(hours=min_interval)
    
    def _adaptive_adjustment(self, threshold_set: ThresholdSet, metrics: RiskMetrics) -> bool:
        """Perform adaptive threshold adjustment based on recent performance"""
        try:
            adjusted = False
            adjustment_factor = self.config.get("adjustment_sensitivity", 0.1)
            
            # Adjust based on recent drawdown performance
            if metrics.current_drawdown > threshold_set.max_drawdown * 0.8:
                # Tighten thresholds if approaching limits
                new_drawdown = threshold_set.max_drawdown * (1 - adjustment_factor)
                new_drawdown = max(new_drawdown, self.config["threshold_bounds"]["max_drawdown"]["min"])
                threshold_set.max_drawdown = new_drawdown
                adjusted = True
            
            # Adjust based on volatility
            if metrics.annualized_volatility > threshold_set.volatility_threshold * 0.9:
                new_vol_threshold = threshold_set.volatility_threshold * (1 + adjustment_factor)
                new_vol_threshold = min(new_vol_threshold, self.config["threshold_bounds"]["volatility_threshold"]["max"])
                threshold_set.volatility_threshold = new_vol_threshold
                adjusted = True
            
            # Adjust based on VaR
            if metrics.var_95 > threshold_set.var_95_threshold * 0.8:
                new_var_threshold = threshold_set.var_95_threshold * (1 + adjustment_factor)
                new_var_threshold = min(new_var_threshold, self.config["threshold_bounds"]["var_95_threshold"]["max"])
                threshold_set.var_95_threshold = new_var_threshold
                adjusted = True
            
            if adjusted:
                threshold_set.last_updated = datetime.now()
                self.last_adjustment_time = datetime.now()
                self._log_adjustment("adaptive", threshold_set.threshold_id, metrics)
                self._save_threshold_sets()
            
            return adjusted
            
        except Exception as e:
            self.logger.error(f"Failed to perform adaptive adjustment: {e}")
            return False
    
    def _regime_based_adjustment(self, threshold_set: ThresholdSet) -> bool:
        """Adjust thresholds based on detected market regime"""
        try:
            regime_thresholds = threshold_set.regime_specific.get(self.current_regime.value, {})
            if not regime_thresholds:
                return False
            
            adjusted = False
            
            # Apply regime-specific adjustments
            for threshold_name, value in regime_thresholds.items():
                if hasattr(threshold_set, threshold_name):
                    current_value = getattr(threshold_set, threshold_name)
                    if abs(current_value - value) > current_value * 0.05:  # 5% threshold for change
                        setattr(threshold_set, threshold_name, value)
                        adjusted = True
            
            if adjusted:
                threshold_set.last_updated = datetime.now()
                self.last_adjustment_time = datetime.now()
                self._log_adjustment("market_regime", threshold_set.threshold_id, None)
                self._save_threshold_sets()
            
            return adjusted
            
        except Exception as e:
            self.logger.error(f"Failed to perform regime-based adjustment: {e}")
            return False
    
    def _volatility_based_adjustment(self, threshold_set: ThresholdSet, metrics: RiskMetrics) -> bool:
        """Adjust thresholds based on volatility levels"""
        try:
            if len(self.volatility_history) < 30:
                return False
            
            # Calculate volatility adjustment factor
            recent_vol = self.volatility_history.tail(20).mean()
            historical_vol = self.volatility_history.mean()
            
            if historical_vol <= 0:
                return False
            
            vol_ratio = recent_vol / historical_vol
            adjustment_factor = min(vol_ratio, threshold_set.max_adjustment_factor)
            
            adjusted = False
            
            # Adjust volatility-related thresholds
            new_vol_threshold = threshold_set.volatility_threshold * adjustment_factor
            vol_bounds = self.config["threshold_bounds"]["volatility_threshold"]
            new_vol_threshold = max(min(new_vol_threshold, vol_bounds["max"]), vol_bounds["min"])
            
            if abs(new_vol_threshold - threshold_set.volatility_threshold) > threshold_set.volatility_threshold * 0.05:
                threshold_set.volatility_threshold = new_vol_threshold
                adjusted = True
            
            # Adjust VaR threshold
            new_var_threshold = threshold_set.var_95_threshold * adjustment_factor
            var_bounds = self.config["threshold_bounds"]["var_95_threshold"]
            new_var_threshold = max(min(new_var_threshold, var_bounds["max"]), var_bounds["min"])
            
            if abs(new_var_threshold - threshold_set.var_95_threshold) > threshold_set.var_95_threshold * 0.05:
                threshold_set.var_95_threshold = new_var_threshold
                adjusted = True
            
            if adjusted:
                threshold_set.last_updated = datetime.now()
                self.last_adjustment_time = datetime.now()
                self._log_adjustment("volatility_based", threshold_set.threshold_id, metrics)
                self._save_threshold_sets()
            
            return adjusted
            
        except Exception as e:
            self.logger.error(f"Failed to perform volatility-based adjustment: {e}")
            return False
    
    def _log_adjustment(self, adjustment_type: str, threshold_id: str, metrics: Optional[RiskMetrics]) -> None:
        """Log threshold adjustments"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "adjustment_type": adjustment_type,
            "threshold_set_id": threshold_id,
            "market_regime": self.current_regime.value,
            "metrics": asdict(metrics) if metrics else None
        }
        
        self.adjustment_log.append(log_entry)
        
        # Keep only recent log entries
        if len(self.adjustment_log) > 1000:
            self.adjustment_log = self.adjustment_log[-500:]
    
    def _save_threshold_sets(self) -> None:
        """Save threshold sets to file"""
        try:
            data = {
                "threshold_sets": [self._threshold_set_to_dict(ts) for ts in self.threshold_sets.values()],
                "active_threshold_id": self.active_threshold_id,
                "threshold_history": self.threshold_history,
                "last_saved": datetime.now().isoformat()
            }
            
            threshold_file = self.config_dir / "threshold_sets.json"
            threshold_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(threshold_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
            self.logger.debug("Saved threshold sets to file")
            
        except Exception as e:
            self.logger.error(f"Failed to save threshold sets: {e}")
    
    def _threshold_set_to_dict(self, threshold_set: ThresholdSet) -> Dict[str, Any]:
        """Convert ThresholdSet to dictionary"""
        data = asdict(threshold_set)
        # Convert datetime objects to ISO format
        data['created_time'] = threshold_set.created_time.isoformat()
        data['last_updated'] = threshold_set.last_updated.isoformat()
        data['adjustment_mode'] = threshold_set.adjustment_mode.value
        return data
    
    def _dict_to_threshold_set(self, data: Dict[str, Any]) -> ThresholdSet:
        """Convert dictionary to ThresholdSet"""
        # Convert ISO format to datetime
        data['created_time'] = datetime.fromisoformat(data['created_time'])
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        data['adjustment_mode'] = ThresholdAdjustmentMode(data['adjustment_mode'])
        
        return ThresholdSet(**data)
    
    def create_threshold_set(self, threshold_data: Dict[str, Any]) -> bool:
        """Create a new threshold set"""
        try:
            threshold_set = ThresholdSet(
                created_time=datetime.now(),
                last_updated=datetime.now(),
                **threshold_data
            )
            
            self.threshold_sets[threshold_set.threshold_id] = threshold_set
            self._save_threshold_sets()
            
            self.logger.info(f"Created new threshold set: {threshold_set.threshold_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create threshold set: {e}")
            return False
    
    def get_threshold_summary(self) -> Dict[str, Any]:
        """Get summary of threshold management status"""
        current_set = self.get_current_thresholds()
        
        return {
            "active_threshold_set": self.active_threshold_id,
            "current_regime": self.current_regime.value,
            "total_threshold_sets": len(self.threshold_sets),
            "auto_adjustment_enabled": self.config.get("auto_adjustment_enabled", True),
            "last_adjustment": self.last_adjustment_time.isoformat() if self.last_adjustment_time else None,
            "current_thresholds": {
                "max_drawdown": current_set.max_drawdown if current_set else None,
                "var_95_threshold": current_set.var_95_threshold if current_set else None,
                "volatility_threshold": current_set.volatility_threshold if current_set else None,
                "concentration_threshold": current_set.concentration_threshold if current_set else None
            } if current_set else None
        }
