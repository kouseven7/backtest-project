# filepath: c:\Users\imega\Documents\my_backtest_project\time_decay.py

import numpy as np
import pandas as pd

def apply_time_decay(data: pd.DataFrame, decay_rate: float) -> pd.Series:
    """
    Apply a time decay factor to the data.
    
    Parameters:
    - data: pd.DataFrame containing the historical data.
    - decay_rate: float, the rate at which older data loses weight.
    
    Returns:
    - pd.Series with the time-decayed values.
    """
    # Calculate the time decay weights
    time_weights = np.exp(-decay_rate * (data.index - data.index[-1]).days)
    time_weights /= time_weights.sum()  # Normalize weights
    
    # Apply weights to the data
    return (data * time_weights).sum(axis=0)