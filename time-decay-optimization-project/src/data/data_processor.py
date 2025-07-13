# utils/time_decay.py
import pandas as pd
import numpy as np

def apply_time_decay(data: pd.DataFrame, decay_rate: float) -> pd.Series:
    """
    Apply a time decay factor to the data.
    
    Parameters:
        data (pd.DataFrame): The input data.
        decay_rate (float): The decay rate (0 < decay_rate < 1).
        
    Returns:
        pd.Series: The time-decayed values.
    """
    weights = np.exp(-decay_rate * np.arange(len(data)))
    weighted_data = data * weights
    return weighted_data / weights.sum()