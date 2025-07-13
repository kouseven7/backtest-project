# data_fetcher.py
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Mock data fetching
    dates = pd.date_range(start=start_date, end=end_date)
    data = pd.DataFrame({
        'Date': dates,
        'Close': [100 + i * 0.5 for i in range(len(dates))]  # Mock closing prices
    })
    return data.set_index('Date')