# filepath: c:\Users\imega\Documents\my_backtest_project\time_decay.py

import pandas as pd
import numpy as np

def apply_time_decay(data: pd.DataFrame, decay_rate: float) -> pd.Series:
    """
    Apply a time decay factor to the data.
    
    Parameters:
    - data: pd.DataFrame - Historical data with a datetime index.
    - decay_rate: float - The rate at which older data is discounted.
    
    Returns:
    - pd.Series - Time-decayed values.
    """
    # Calculate the time decay weights
    time_weights = np.exp(-decay_rate * (data.index - data.index[-1]).days)
    time_weights /= time_weights.sum()  # Normalize weights

    # Apply weights to the data
    decayed_data = (data * time_weights).sum(axis=0)
    return decayed_data