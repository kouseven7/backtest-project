"""
Debug why Breakout strategy receives incorrect column structure from test

This script replicates the exact data flow from test_breakout_8306T.py
to identify where the 'High' column goes missing.
"""

import pandas as pd
import yfinance as yf
import sys
sys.path.append('.')

from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from strategies.Breakout import BreakoutStrategy

print("=" * 80)
print("STEP 1: Fetch data using YFinanceDataFeed (as test does)")
print("=" * 80)

data_feed = YFinanceDataFeed()
stock_data = data_feed.get_stock_data(
    ticker="9101.T",
    start_date="2024-01-01",
    end_date="2024-12-31"
)

print(f"Data shape: {stock_data.shape}")
print(f"Data columns: {stock_data.columns.tolist()}")
print(f"Column types: {stock_data.columns}")
print(f"Is MultiIndex: {isinstance(stock_data.columns, pd.MultiIndex)}")
print("\nFirst row:")
print(stock_data.head(1))

print("\n" + "=" * 80)
print("STEP 2: Check if 'High' column exists")
print("=" * 80)

print(f"'High' in columns: {'High' in stock_data.columns}")
print(f"Columns contain 'High': {[col for col in stock_data.columns if 'High' in str(col)]}")

print("\n" + "=" * 80)
print("STEP 3: Initialize BreakoutStrategy with this data (as test does)")
print("=" * 80)

# Copy data as test does
data_copy = stock_data.copy()
print(f"Copied data columns: {data_copy.columns.tolist()}")

# Initialize strategy
strategy = BreakoutStrategy(
    data=data_copy,
    params=None,
    price_column="Adj Close",
    volume_column="Volume"
)

print(f"Strategy.data columns: {strategy.data.columns.tolist()}")
print(f"'High' in strategy.data: {'High' in strategy.data.columns}")

print("\n" + "=" * 80)
print("STEP 4: Try to access 'High' column as generate_entry_signal does")
print("=" * 80)

try:
    idx = 10  # Test at index 10
    look_back = 1
    
    if 'High' not in strategy.data.columns:
        print("[ERROR] 'High' column not found in strategy.data!")
        print(f"Available columns: {strategy.data.columns.tolist()}")
    else:
        previous_high = strategy.data['High'].iloc[idx - look_back]
        print(f"[OK] Successfully accessed 'High' column: previous_high={previous_high}")
        
except Exception as e:
    print(f"[ERROR] Exception when accessing 'High': {e}")

print("\n" + "=" * 80)
print("STEP 5: Run backtest and check for signals")
print("=" * 80)

result = strategy.backtest()
print(f"Result shape: {result.shape}")
print(f"Result columns: {result.columns.tolist()}")

entry_count = (result['Entry_Signal'] == 1).sum()
exit_count = (result['Exit_Signal'] == -1).sum()

print(f"\nEntry signals: {entry_count}")
print(f"Exit signals: {exit_count}")

if entry_count > 0:
    print("\n[OK] Signals were generated!")
    entry_dates = result[result['Entry_Signal'] == 1].index.tolist()
    print(f"First 5 entry dates: {entry_dates[:5]}")
else:
    print("\n[ERROR] No signals generated!")
    
print("\n" + "=" * 80)
print("STEP 6: Direct calculation to verify expected signals")
print("=" * 80)

# Calculate signals manually
look_back = 1
breakout_buffer = 0.01
volume_threshold = 1.2

test_data = result.copy()
test_data['Prev_High'] = test_data['High'].shift(look_back)
test_data['Prev_Volume'] = test_data['Volume'].shift(look_back)

price_breakout = test_data['Adj Close'] > test_data['Prev_High'] * (1 + breakout_buffer)
volume_increase = test_data['Volume'] > test_data['Prev_Volume'] * volume_threshold
both_conditions = price_breakout & volume_increase

expected_signals = both_conditions.sum()
print(f"Expected entry signals (manual calculation): {expected_signals}")
print(f"Actual entry signals (from backtest): {entry_count}")

if expected_signals != entry_count:
    print(f"\n[MISMATCH] Expected {expected_signals} but got {entry_count}")
    print("Checking first expected signal date...")
    first_expected = test_data[both_conditions].head(1)
    if len(first_expected) > 0:
        date = first_expected.index[0]
        print(f"First expected signal: {date}")
        print(f"Close: {first_expected['Adj Close'].iloc[0]:.2f}")
        print(f"Prev High: {first_expected['Prev_High'].iloc[0]:.2f}")
        print(f"Volume: {first_expected['Volume'].iloc[0]:.0f}")
        print(f"Prev Volume: {first_expected['Prev_Volume'].iloc[0]:.0f}")
else:
    print("\n[OK] Signal count matches!")
