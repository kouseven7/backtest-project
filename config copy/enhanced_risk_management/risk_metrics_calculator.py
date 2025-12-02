"""
DSSMS Phase 2 Task 2.3: Enhanced Risk Management System
Component: Risk Metrics Calculator

This module provides comprehensive risk metric calculations for portfolio and position analysis.
Integrates with existing PortfolioRiskManager and provides enhanced risk calculations.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics container"""
    timestamp: datetime
    portfolio_value: float
    
    # Return metrics
    daily_return: float
    monthly_return: float
    ytd_return: float
    annualized_return: float
    
    # Volatility metrics
    daily_volatility: float
    monthly_volatility: float
    annualized_volatility: float
    rolling_volatility_30d: float
    
    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    max_drawdown_duration: int
    recovery_time: Optional[int]
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # VaR and ES metrics
    var_95: float
    var_99: float
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float
    
    # Concentration and correlation metrics
    concentration_risk: float
    max_position_weight: float
    effective_positions: float
    average_correlation: float
    max_correlation: float
    
    # Additional risk metrics
    beta: float
    tracking_error: float
    downside_deviation: float
    tail_ratio: float
    skewness: float
    kurtosis: float


class RiskMetricsCalculator:
    """
    Advanced risk metrics calculator with comprehensive portfolio analysis.
    Provides enhanced risk calculations beyond basic PortfolioRiskManager.
    """
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.price_data: Dict[str, pd.Series] = {}
        self.portfolio_values: pd.Series = pd.Series(dtype=float)
        self.position_weights: pd.DataFrame = pd.DataFrame()
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)
        
        # Risk-free rate (can be updated)
        self.risk_free_rate = 0.02  # 2% annual
        
        self.logger.info(f"RiskMetricsCalculator initialized with {lookback_days} days lookback")
    
    def update_price_data(self, symbol: str, price_series: pd.Series) -> None:
        """Update price data for a symbol"""
        try:
            # Ensure proper datetime index
            if not isinstance(price_series.index, pd.DatetimeIndex):
                price_series.index = pd.to_datetime(price_series.index)
            
            # Keep only recent data
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days * 2)
            price_series = price_series[price_series.index >= cutoff_date]
            
            self.price_data[symbol] = price_series
            self.logger.debug(f"Updated price data for {symbol}: {len(price_series)} points")
            
        except Exception as e:
            self.logger.error(f"Failed to update price data for {symbol}: {e}")
    
    def update_portfolio_values(self, portfolio_series: pd.Series) -> None:
        """Update portfolio value time series"""
        try:
            # Ensure proper datetime index
            if not isinstance(portfolio_series.index, pd.DatetimeIndex):
                portfolio_series.index = pd.to_datetime(portfolio_series.index)
            
            # Keep only recent data
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days * 2)
            portfolio_series = portfolio_series[portfolio_series.index >= cutoff_date]
            
            self.portfolio_values = portfolio_series
            self.logger.debug(f"Updated portfolio values: {len(portfolio_series)} points")
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio values: {e}")
    
    def update_position_weights(self, weights_df: pd.DataFrame) -> None:
        """Update position weights over time"""
        try:
            # Ensure proper datetime index
            if not isinstance(weights_df.index, pd.DatetimeIndex):
                weights_df.index = pd.to_datetime(weights_df.index)
            
            # Keep only recent data
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days * 2)
            weights_df = weights_df[weights_df.index >= cutoff_date]
            
            self.position_weights = weights_df
            self.logger.debug(f"Updated position weights: {weights_df.shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to update position weights: {e}")
    
    def set_benchmark_returns(self, benchmark_series: pd.Series) -> None:
        """Set benchmark returns for relative risk metrics"""
        try:
            # Ensure proper datetime index
            if not isinstance(benchmark_series.index, pd.DatetimeIndex):
                benchmark_series.index = pd.to_datetime(benchmark_series.index)
            
            # Keep only recent data
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days * 2)
            benchmark_series = benchmark_series[benchmark_series.index >= cutoff_date]
            
            self.benchmark_returns = benchmark_series
            self.logger.debug(f"Updated benchmark returns: {len(benchmark_series)} points")
            
        except Exception as e:
            self.logger.error(f"Failed to update benchmark returns: {e}")
    
    def calculate_comprehensive_metrics(self, 
                                      current_portfolio_value: float,
                                      current_positions: Dict[str, float]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for current portfolio state.
        
        Args:
            current_portfolio_value: Current total portfolio value
            current_positions: Current position weights {symbol: weight}
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        try:
            timestamp = datetime.now()
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns()
            
            # Return metrics
            return_metrics = self._calculate_return_metrics(portfolio_returns)
            
            # Volatility metrics  
            volatility_metrics = self._calculate_volatility_metrics(portfolio_returns)
            
            # Drawdown metrics
            drawdown_metrics = self._calculate_drawdown_metrics()
            
            # Risk-adjusted metrics
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(portfolio_returns)
            
            # VaR and ES metrics
            var_metrics = self._calculate_var_metrics(portfolio_returns)
            
            # Concentration and correlation metrics
            concentration_metrics = self._calculate_concentration_metrics(current_positions)
            
            # Additional risk metrics
            additional_metrics = self._calculate_additional_metrics(portfolio_returns)
            
            # Create comprehensive metrics object
            metrics = RiskMetrics(
                timestamp=timestamp,
                portfolio_value=current_portfolio_value,
                **return_metrics,
                **volatility_metrics,
                **drawdown_metrics,
                **risk_adjusted_metrics,
                **var_metrics,
                **concentration_metrics,
                **additional_metrics
            )
            
            self.logger.info(f"Calculated comprehensive risk metrics at {timestamp}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate comprehensive metrics: {e}")
            # Return default metrics on error
            return self._get_default_metrics(current_portfolio_value)
    
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns from value series"""
        if len(self.portfolio_values) < 2:
            return pd.Series(dtype=float)
        
        returns = self.portfolio_values.pct_change().dropna()
        return returns
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return-based metrics"""
        if len(returns) == 0:
            return {
                'daily_return': 0.0,
                'monthly_return': 0.0,
                'ytd_return': 0.0,
                'annualized_return': 0.0
            }
        
        # Daily return (most recent)
        daily_return = returns.iloc[-1] if len(returns) > 0 else 0.0
        
        # Monthly return (last 30 days)
        monthly_returns = returns.tail(30)
        monthly_return = (1 + monthly_returns).prod() - 1 if len(monthly_returns) > 0 else 0.0
        
        # YTD return
        current_year = datetime.now().year
        ytd_returns = returns[returns.index.year == current_year]
        ytd_return = (1 + ytd_returns).prod() - 1 if len(ytd_returns) > 0 else 0.0
        
        # Annualized return
        if len(returns) > 252:
            annualized_return = (1 + returns.mean()) ** 252 - 1
        else:
            annualized_return = returns.mean() * 252 if len(returns) > 0 else 0.0
        
        return {
            'daily_return': daily_return,
            'monthly_return': monthly_return,
            'ytd_return': ytd_return,
            'annualized_return': annualized_return
        }
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility-based metrics"""
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'monthly_volatility': 0.0,
                'annualized_volatility': 0.0,
                'rolling_volatility_30d': 0.0
            }
        
        # Daily volatility (standard deviation)
        daily_volatility = returns.std() if len(returns) > 1 else 0.0
        
        # Monthly volatility
        monthly_volatility = daily_volatility * np.sqrt(30)
        
        # Annualized volatility
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # 30-day rolling volatility
        if len(returns) >= 30:
            rolling_vol_30d = returns.tail(30).std()
        else:
            rolling_vol_30d = daily_volatility
        
        return {
            'daily_volatility': daily_volatility,
            'monthly_volatility': monthly_volatility,
            'annualized_volatility': annualized_volatility,
            'rolling_volatility_30d': rolling_vol_30d
        }
    
    def _calculate_drawdown_metrics(self) -> Dict[str, float]:
        """Calculate drawdown-based metrics"""
        if len(self.portfolio_values) < 2:
            return {
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'recovery_time': None
            }
        
        # Calculate running maximum
        running_max = self.portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (self.portfolio_values - running_max) / running_max
        
        # Current drawdown
        current_drawdown = abs(drawdown.iloc[-1])
        
        # Maximum drawdown
        max_drawdown = abs(drawdown.min())
        
        # Drawdown duration calculation
        is_drawdown = drawdown < 0
        drawdown_periods = self._calculate_drawdown_periods(is_drawdown)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Recovery time (if currently in drawdown)
        recovery_time = None
        if current_drawdown > 0:
            # Find the start of current drawdown period
            current_dd_start = None
            for i in range(len(is_drawdown) - 1, -1, -1):
                if not is_drawdown.iloc[i]:
                    current_dd_start = i + 1
                    break
            
            if current_dd_start is not None:
                recovery_time = len(is_drawdown) - current_dd_start
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'recovery_time': recovery_time
        }
    
    def _calculate_drawdown_periods(self, is_drawdown: pd.Series) -> List[int]:
        """Calculate the length of each drawdown period"""
        periods = []
        current_period = 0
        
        for in_dd in is_drawdown:
            if in_dd:
                current_period += 1
            else:
                if current_period > 0:
                    periods.append(current_period)
                    current_period = 0
        
        # Add the last period if still in drawdown
        if current_period > 0:
            periods.append(current_period)
        
        return periods
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        if len(returns) < 2:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'information_ratio': 0.0
            }
        
        # Annualized metrics
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_volatility
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
        
        # Calmar ratio (return / max drawdown)
        drawdown_metrics = self._calculate_drawdown_metrics()
        max_drawdown = drawdown_metrics['max_drawdown']
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Information ratio (if benchmark available)
        information_ratio = 0.0
        if len(self.benchmark_returns) > 0:
            # Align returns with benchmark
            aligned_returns, aligned_benchmark = returns.align(self.benchmark_returns, join='inner')
            if len(aligned_returns) > 1:
                excess_returns = aligned_returns - aligned_benchmark
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }
    
    def _calculate_var_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk and Expected Shortfall metrics"""
        if len(returns) < 30:  # Need minimum data for reliable VaR
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0
            }
        
        # Sort returns for quantile calculation
        sorted_returns = returns.sort_values()
        
        # VaR calculation (negative of quantile)
        var_95 = -sorted_returns.quantile(0.05)
        var_99 = -sorted_returns.quantile(0.01)
        
        # Conditional VaR (Expected Shortfall)
        # Average of returns below VaR threshold
        returns_below_var95 = sorted_returns[sorted_returns <= -var_95]
        returns_below_var99 = sorted_returns[sorted_returns <= -var_99]
        
        cvar_95 = -returns_below_var95.mean() if len(returns_below_var95) > 0 else var_95
        cvar_99 = -returns_below_var99.mean() if len(returns_below_var99) > 0 else var_99
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }
    
    def _calculate_concentration_metrics(self, current_positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate concentration and correlation risk metrics"""
        if not current_positions:
            return {
                'concentration_risk': 0.0,
                'max_position_weight': 0.0,
                'effective_positions': 0.0,
                'average_correlation': 0.0,
                'max_correlation': 0.0
            }
        
        # Position weights (absolute values)
        weights = np.array([abs(w) for w in current_positions.values()])
        
        # Concentration risk (Herfindahl index)
        concentration_risk = np.sum(weights ** 2)
        
        # Maximum position weight
        max_position_weight = np.max(weights)
        
        # Effective number of positions (inverse of Herfindahl index)
        effective_positions = 1.0 / concentration_risk if concentration_risk > 0 else 0.0
        
        # Correlation metrics (if price data available)
        average_correlation = 0.0
        max_correlation = 0.0
        
        symbols = list(current_positions.keys())
        if len(symbols) > 1 and len(self.price_data) >= 2:
            correlations = self._calculate_position_correlations(symbols)
            if correlations is not None:
                # Extract upper triangle (excluding diagonal)
                mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
                corr_values = correlations[mask]
                
                if len(corr_values) > 0:
                    average_correlation = np.mean(corr_values)
                    max_correlation = np.max(corr_values)
        
        return {
            'concentration_risk': concentration_risk,
            'max_position_weight': max_position_weight,
            'effective_positions': effective_positions,
            'average_correlation': average_correlation,
            'max_correlation': max_correlation
        }
    
    def _calculate_position_correlations(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Calculate correlation matrix for position symbols"""
        try:
            # Get price data for symbols
            price_data = {}
            for symbol in symbols:
                if symbol in self.price_data and len(self.price_data[symbol]) > 30:
                    price_data[symbol] = self.price_data[symbol]
            
            if len(price_data) < 2:
                return None
            
            # Create price DataFrame
            price_df = pd.DataFrame(price_data)
            
            # Calculate returns
            returns_df = price_df.pct_change().dropna()
            
            if len(returns_df) < 30:  # Need minimum data
                return None
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr().values
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to calculate correlations: {e}")
            return None
    
    def _calculate_additional_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate additional risk metrics"""
        if len(returns) < 2:
            return {
                'beta': 0.0,
                'tracking_error': 0.0,
                'downside_deviation': 0.0,
                'tail_ratio': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
        
        # Beta calculation (if benchmark available)
        beta = 0.0
        tracking_error = 0.0
        if len(self.benchmark_returns) > 0:
            aligned_returns, aligned_benchmark = returns.align(self.benchmark_returns, join='inner')
            if len(aligned_returns) > 30:
                # Calculate beta using linear regression
                try:
                    slope, _, _, _, _ = stats.linregress(aligned_benchmark, aligned_returns)
                    beta = slope
                    
                    # Tracking error
                    excess_returns = aligned_returns - aligned_benchmark
                    tracking_error = excess_returns.std() * np.sqrt(252)
                except:
                    beta = 0.0
                    tracking_error = 0.0
        
        # Downside deviation
        downside_returns = returns[returns < returns.mean()]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        
        # Tail ratio (95th percentile / 5th percentile)
        try:
            p95 = returns.quantile(0.95)
            p5 = returns.quantile(0.05)
            tail_ratio = abs(p95 / p5) if p5 != 0 else 0.0
        except:
            tail_ratio = 0.0
        
        # Skewness and kurtosis
        try:
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
        except:
            skewness = 0.0
            kurtosis = 0.0
        
        return {
            'beta': beta,
            'tracking_error': tracking_error,
            'downside_deviation': downside_deviation,
            'tail_ratio': tail_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _get_default_metrics(self, portfolio_value: float) -> RiskMetrics:
        """Return default metrics when calculation fails"""
        return RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            daily_return=0.0,
            monthly_return=0.0,
            ytd_return=0.0,
            annualized_return=0.0,
            daily_volatility=0.0,
            monthly_volatility=0.0,
            annualized_volatility=0.0,
            rolling_volatility_30d=0.0,
            current_drawdown=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            recovery_time=None,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            concentration_risk=0.0,
            max_position_weight=0.0,
            effective_positions=0.0,
            average_correlation=0.0,
            max_correlation=0.0,
            beta=0.0,
            tracking_error=0.0,
            downside_deviation=0.0,
            tail_ratio=0.0,
            skewness=0.0,
            kurtosis=0.0
        )
    
    def get_risk_summary(self, metrics: RiskMetrics) -> Dict[str, Any]:
        """Generate a risk summary from metrics"""
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "portfolio_value": metrics.portfolio_value,
            "key_metrics": {
                "daily_return": f"{metrics.daily_return:.4f}",
                "annualized_volatility": f"{metrics.annualized_volatility:.4f}",
                "current_drawdown": f"{metrics.current_drawdown:.4f}",
                "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                "var_95": f"{metrics.var_95:.4f}",
                "concentration_risk": f"{metrics.concentration_risk:.4f}"
            },
            "risk_flags": {
                "high_volatility": metrics.annualized_volatility > 0.25,
                "significant_drawdown": metrics.current_drawdown > 0.05,
                "poor_sharpe": metrics.sharpe_ratio < 0.5,
                "high_var": metrics.var_95 > 0.03,
                "concentration_risk": metrics.concentration_risk > 0.25,
                "correlation_risk": metrics.max_correlation > 0.80
            }
        }
