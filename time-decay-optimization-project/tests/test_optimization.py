import pandas as pd
from datetime import datetime, timedelta

def apply_time_decay(data: pd.DataFrame, decay_rate: float) -> pd.DataFrame:
    """
    Apply a time decay factor to the data.
    
    Parameters:
    - data: DataFrame containing historical data with a 'Date' column.
    - decay_rate: The rate at which older data loses significance.
    
    Returns:
    - DataFrame with adjusted values based on time decay.
    """
    current_date = datetime.now()
    data['Weight'] = (1 - decay_rate) ** ((current_date - data['Date']).dt.days)
    data['Adjusted_Value'] = data['Value'] * data['Weight']
    return data