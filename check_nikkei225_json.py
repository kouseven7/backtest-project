import json
from pathlib import Path

json_path = Path("config/dssms/nikkei225_components.json")

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

symbols = data["symbols"]
unique_symbols = set(symbols)

print(f"Total symbols: {len(symbols)}")
print(f"Unique symbols: {len(unique_symbols)}")
print(f"Duplicates: {len(symbols) - len(unique_symbols)}")
print(f"Last updated: {data['metadata']['last_updated']}")
print(f"\nFirst 10 symbols: {symbols[:10]}")
print(f"Last 10 symbols: {symbols[-10:]}")

# 重複銘柄を確認
from collections import Counter
counter = Counter(symbols)
duplicates = {k: v for k, v in counter.items() if v > 1}
if duplicates:
    print(f"\nDuplicate symbols found: {len(duplicates)}")
    for symbol, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {symbol}: {count} times")
