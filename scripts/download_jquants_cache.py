"""
download_jquants_cache - Bulk download Nikkei 225 price history via J-Quants.

This script downloads daily OHLCV history for all Nikkei 225 symbols from
J-Quants through the existing Ticker wrapper and stores each symbol as a
Parquet cache file.

Main features:
- Load symbols from config/dssms/nikkei225_components.json
- Download from 2016-03-24 to today
- Save per-symbol cache under data/jquants_cache/{symbol}.parquet
- Skip already cached symbols to support resume
- Continue on per-symbol errors and print a failure summary

Integrated components:
- src.utils.yfinance_lazy_wrapper.Ticker: API data access abstraction

Safety notes:
- Adds 1.0 second delay between symbol requests to reduce 429 risk
- Keeps processing even when a symbol fails

Author: Backtest Project Team
Created: 2026-03-24
Last Modified: 2026-03-24
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.yfinance_lazy_wrapper import Ticker


START_DATE = "2016-03-24"
SLEEP_SECONDS = 1.0
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_symbols(config_path: Path) -> List[str]:
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list):
        raise ValueError("Invalid config format: 'symbols' must be a list")

    return [str(symbol).strip() for symbol in symbols if str(symbol).strip()]


def _to_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("empty dataframe")

    out_df = df.copy()

    if "Date" not in out_df.columns:
        out_df = out_df.reset_index()
        if "Date" not in out_df.columns:
            out_df = out_df.rename(columns={out_df.columns[0]: "Date"})

    missing = [col for col in REQUIRED_COLUMNS if col not in out_df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    return out_df[REQUIRED_COLUMNS]


def main() -> None:
    root = _project_root()
    symbols_path = root / "config" / "dssms" / "nikkei225_components.json"
    cache_dir = root / "data" / "jquants_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    symbols = _load_symbols(symbols_path)
    total = len(symbols)

    end_date = date.today().isoformat()
    failed_symbols: List[str] = []

    print(f"Start download: symbols={total}, start={START_DATE}, end={end_date}")

    for idx, symbol in enumerate(symbols, start=1):
        out_path = cache_dir / f"{symbol}.csv"

        if out_path.exists():
            print(f"[{idx}/{total}] {symbol} skipped (cache exists)")
            continue

        try:
            ticker = Ticker(symbol)
            df = ticker.history(start=START_DATE, end=end_date, interval="1d")
            output_df = _to_output_dataframe(df)
            output_df.to_csv(out_path, index=True)
            print(f"[{idx}/{total}] {symbol} done")
        except Exception as exc:
            failed_symbols.append(symbol)
            print(f"[{idx}/{total}] {symbol} error: {exc}")
        finally:
            time.sleep(SLEEP_SECONDS)

    success_count = total - len(failed_symbols)
    print(f"Finished: success={success_count}, failed={len(failed_symbols)}")

    if failed_symbols:
        print("Failed symbols:")
        for symbol in failed_symbols:
            print(symbol)


if __name__ == "__main__":
    main()
