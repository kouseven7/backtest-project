# data_fetcher.py
import pandas as pd
from datetime import datetime, timedelta

def fetch_data_with_decay(ticker: str, start_date: str, end_date: str, decay_factor: float) -> pd.DataFrame:
    """Fetch data and apply time decay to prioritize recent data."""
    # Fetch data (this is a placeholder, replace with actual data fetching logic)
    data = fetch_data(ticker, start_date, end_date)
    
    # Apply time decay
    current_date = datetime.now()
    data['decay_weight'] = (current_date - data.index).days * decay_factor
    data['weighted_value'] = data['value'] * (1 / (1 + data['decay_weight']))
    
    return data