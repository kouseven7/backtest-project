"""
Compare Close vs Adj Close for 9101.T to understand signal discrepancy

The analyze_9101T_breakout.py script used 'Close' and found 18 signals.
The BreakoutStrategy uses 'Adj Close' (price_column parameter) and found 0 signals.
"""

import pandas as pd
import yfinance as yf

print("Downloading 9101.T data for 2024...")
data = yf.download('9101.T', start='2024-01-01', end='2024-12-31', progress=False)

# Flatten columns if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")

print("\n" + "=" * 80)
print("COMPARISON: Close vs Adj Close")
print("=" * 80)

# Calculate statistics
close_mean = data['Close'].mean()
adj_close_mean = data['Adj Close'].mean()
diff_pct = ((adj_close_mean - close_mean) / close_mean) * 100

print(f"Close mean: {close_mean:.2f}")
print(f"Adj Close mean: {adj_close_mean:.2f}")
print(f"Difference: {diff_pct:.2f}%")

print("\nFirst 10 rows:")
comparison = pd.DataFrame({
    'Close': data['Close'].head(10),
    'Adj Close': data['Adj Close'].head(10),
    'Diff (%)': ((data['Adj Close'] - data['Close']) / data['Close'] * 100).head(10)
})
print(comparison)

print("\n" + "=" * 80)
print("SIGNAL ANALYSIS WITH BOTH COLUMNS")
print("=" * 80)

# Test with Close
look_back = 1
breakout_buffer = 0.01
volume_threshold = 1.2

# Using Close
data['Prev_High_Close'] = data['High'].shift(look_back)
data['Prev_Volume'] = data['Volume'].shift(look_back)

price_breakout_close = data['Close'] > data['Prev_High_Close'] * (1 + breakout_buffer)
volume_increase = data['Volume'] > data['Prev_Volume'] * volume_threshold
signals_close = (price_breakout_close & volume_increase).sum()

print(f"Signals using Close: {signals_close}")

# Using Adj Close
price_breakout_adj = data['Adj Close'] > data['Prev_High_Close'] * (1 + breakout_buffer)
signals_adj = (price_breakout_adj & volume_increase).sum()

print(f"Signals using Adj Close: {signals_adj}")

print("\n" + "=" * 80)
print("ROOT CAUSE: Comparing Close/Adj Close to High")
print("=" * 80)

print("\nProblem: We're comparing Adj Close (adjusted) to High (unadjusted)!")
print("Solution options:")
print("1. Use Close instead of Adj Close for breakout comparison")
print("2. Calculate adjusted High values")
print("3. Document that Adj Close may not work well with raw High")

# Show specific examples
print("\nExamples where Close triggers but Adj Close doesn't:")
close_triggers = price_breakout_close & ~price_breakout_adj & volume_increase
if close_triggers.sum() > 0:
    examples = data[close_triggers].head(5)[['Close', 'Adj Close', 'High', 'Volume']]
    for date, row in examples.iterrows():
        prev_high = data.loc[:date].shift(1).loc[date, 'High'] if date in data.index else None
        if prev_high and not pd.isna(prev_high):
            close_vs_high = (row['Close'] / prev_high - 1) * 100
            adj_vs_high = (row['Adj Close'] / prev_high - 1) * 100
            print(f"\n{date.strftime('%Y-%m-%d')}:")
            print(f"  High (prev): {prev_high:.2f}")
            print(f"  Close: {row['Close']:.2f} (+{close_vs_high:.2f}% vs prev high) ✓ TRIGGERS")
            print(f"  Adj Close: {row['Adj Close']:.2f} (+{adj_vs_high:.2f}% vs prev high) ✗ NO TRIGGER")
