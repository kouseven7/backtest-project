# data_fetcher.py
import pandas as pd
from datetime import datetime, timedelta

def fetch_data_with_decay(ticker: str, start_date: str, end_date: str, decay_factor: float) -> pd.DataFrame:
    """Fetch stock data and apply time decay."""
    # Fetch data (this is a placeholder for actual data fetching logic)
    data = fetch_data(ticker, start_date, end_date)
    
    # Apply time decay
    current_time = datetime.now()
    data['decayed_value'] = data.apply(lambda row: row['value'] * (1 - decay_factor) ** ((current_time - row['date']).days), axis=1)
    
    return data