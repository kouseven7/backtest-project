"""
Analyze why 9101.T generated no breakout signals in 2024

This script examines price breakouts and volume conditions for 9101.T
to understand why the Breakout strategy did not generate any signals.
"""

import pandas as pd
import yfinance as yf

# Download 9101.T data for 2024
print("Downloading 9101.T data for 2024...")
data = yf.download('9101.T', start='2024-01-01', end='2024-12-31', progress=False)
print(f"Data shape: {data.shape}")

# Flatten multi-index columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

# Breakout parameters (from default params)
look_back = 1
breakout_buffer = 0.01  # 1%
volume_threshold = 1.2  # 20% increase

# Calculate breakout conditions
data['Previous_High'] = data['High'].shift(look_back)
data['Previous_Volume'] = data['Volume'].shift(look_back)

# Price breakout condition
data['Price_Breakout'] = data['Close'] > data['Previous_High'] * (1 + breakout_buffer)

# Volume increase condition
data['Volume_Increase'] = data['Volume'] > data['Previous_Volume'] * volume_threshold

# Both conditions met
data['Both_Conditions'] = data['Price_Breakout'] & data['Volume_Increase']

# Count signals
price_breakouts = data['Price_Breakout'].sum()
volume_increases = data['Volume_Increase'].sum()
both_conditions = data['Both_Conditions'].sum()

print("\n" + "=" * 80)
print("BREAKOUT SIGNAL ANALYSIS - 9101.T (2024)")
print("=" * 80)
print(f"Total days analyzed: {len(data)}")
print(f"\nPrice breakouts (Close > Previous High * 1.01): {price_breakouts}")
print(f"Volume increases (Volume > Previous Volume * 1.2): {volume_increases}")
print(f"BOTH conditions met (Entry signals): {both_conditions}")

# Show days where price breakout occurred but volume didn't increase
print("\n" + "=" * 80)
print("MISSED OPPORTUNITIES (Price breakout but insufficient volume)")
print("=" * 80)
missed = data[data['Price_Breakout'] & ~data['Volume_Increase']].copy()
if len(missed) > 0:
    print(f"Total: {len(missed)} days")
    print("\nFirst 10 examples:")
    for i, (date, row) in enumerate(missed.head(10).iterrows()):
        volume_ratio = row['Volume'] / row['Previous_Volume']
        price_change = (row['Close'] / row['Previous_High'] - 1) * 100
        print(f"{i+1}. {date.strftime('%Y-%m-%d')}: Close={row['Close']:.2f} (+{price_change:.2f}%), "
              f"Volume ratio={volume_ratio:.2f}x (need 1.2x)")
else:
    print("None")

# Show days where volume increased but price didn't break out
print("\n" + "=" * 80)
print("HIGH VOLUME DAYS (Volume increased but no price breakout)")
print("=" * 80)
high_volume = data[~data['Price_Breakout'] & data['Volume_Increase']].copy()
if len(high_volume) > 0:
    print(f"Total: {len(high_volume)} days")
    print("\nFirst 10 examples:")
    for i, (date, row) in enumerate(high_volume.head(10).iterrows()):
        volume_ratio = row['Volume'] / row['Previous_Volume']
        price_vs_high = (row['Close'] / row['Previous_High'] - 1) * 100
        print(f"{i+1}. {date.strftime('%Y-%m-%d')}: Close={row['Close']:.2f} "
              f"({price_vs_high:+.2f}% vs prev high), Volume ratio={volume_ratio:.2f}x")
else:
    print("None")

# Show statistics for how close we got to both conditions
print("\n" + "=" * 80)
print("STATISTICS - How close did we get?")
print("=" * 80)

# For price breakouts: show distribution
price_vs_threshold = (data['Close'] / data['Previous_High'] - 1) * 100
print(f"\nPrice change vs Previous High:")
print(f"  Mean: {price_vs_threshold.mean():.2f}%")
print(f"  Median: {price_vs_threshold.median():.2f}%")
print(f"  Max: {price_vs_threshold.max():.2f}%")
print(f"  Days above +1% threshold: {(price_vs_threshold > 1.0).sum()}")

# For volume: show distribution
volume_ratios = data['Volume'] / data['Previous_Volume']
print(f"\nVolume ratio vs Previous Day:")
print(f"  Mean: {volume_ratios.mean():.2f}x")
print(f"  Median: {volume_ratios.median():.2f}x")
print(f"  Max: {volume_ratios.max():.2f}x")
print(f"  Days above 1.2x threshold: {(volume_ratios > 1.2).sum()}")

# Suggest parameter adjustments
print("\n" + "=" * 80)
print("PARAMETER SENSITIVITY ANALYSIS")
print("=" * 80)

# Test with looser breakout_buffer
for buffer in [0.005, 0.0075, 0.01, 0.015]:
    for vol_thresh in [1.1, 1.15, 1.2, 1.25]:
        test_price = data['Close'] > data['Previous_High'] * (1 + buffer)
        test_vol = data['Volume'] > data['Previous_Volume'] * vol_thresh
        signals = (test_price & test_vol).sum()
        if signals > 0:
            print(f"breakout_buffer={buffer:.4f}, volume_threshold={vol_thresh:.2f}: {signals} signals")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("The Breakout strategy requires BOTH conditions to be met simultaneously:")
print("1. Price must exceed previous high by 1%")
print("2. Volume must be 20% higher than previous day")
print("\nThis is a very strict filter. Consider:")
print("- Lowering breakout_buffer to 0.5-0.75% for more price breakouts")
print("- Lowering volume_threshold to 1.1-1.15x for more volume signals")
