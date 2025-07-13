# data_fetcher.py
import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data