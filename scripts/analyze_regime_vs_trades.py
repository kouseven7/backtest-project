"""
analyze_regime_vs_trades - Analyze GCStrategy trades by market regime.

This script parses market regime records from a market analysis log and maps
them to trading dates, then joins the regime data to GCStrategy trades from
all_transactions.csv for summary analysis.

Main features:
- Extract daily regime sequence from market analysis log
- Build trading date list from a J-Quants cache CSV file
- Join regime to GCStrategy entry dates and compute summary tables

Integrated components:
- output/dssms_integration/* logs and CSVs: source of regime and trade records
- data/jquants_cache/*.csv: source of common trading calendar dates

Safety notes:
- Uses strict input validation and clear error messages
- Handles length mismatch between regime logs and trading dates safely

Author: Backtest Project Team
Created: 2026-04-19
Last Modified: 2026-04-19
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_LOG_PATH = Path(
    "output/dssms_integration/backtest_20260418_000548/market_analysis_20260418_000548.log"
)
TRADES_CSV_PATH = Path(
    "output/dssms_integration/dssms_20260418_022803/all_transactions.csv"
)
JQUANTS_CACHE_DIR = Path("data/jquants_cache")
OUTPUT_PATH = Path("output/regime_analysis_result.csv")

DATE_START = pd.Timestamp("2016-09-01")
DATE_END = pd.Timestamp("2025-12-31")

REGIME_PATTERN = re.compile(
    r"\[MARKET_ANALYSIS\] Market analysis completed - Regime: "
    r"(strong_uptrend|uptrend|weak_uptrend|sideways|weak_downtrend|downtrend)"
)
VALID_REGIMES = {
    "strong_uptrend",
    "uptrend",
    "weak_uptrend",
    "sideways",
    "weak_downtrend",
    "downtrend",
}


def _resolve_path(path: Path) -> Path:
    return PROJECT_ROOT / path


def load_regime_list(log_path: Path) -> list[str]:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    regimes: list[str] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = REGIME_PATTERN.search(line)
            if match:
                regime = match.group(1)
                if regime in VALID_REGIMES:
                    regimes.append(regime)

    if not regimes:
        raise ValueError("No regime records were extracted from market analysis log")

    return regimes


def load_trading_dates(cache_dir: Path) -> pd.DatetimeIndex:
    if not cache_dir.exists() or not cache_dir.is_dir():
        raise FileNotFoundError(f"J-Quants cache directory not found: {cache_dir}")

    cache_files = sorted(cache_dir.glob("*.csv"))
    if not cache_files:
        raise FileNotFoundError(f"No CSV file found in cache directory: {cache_dir}")

    first_csv = cache_files[0]
    df = pd.read_csv(first_csv)

    if "Date" not in df.columns:
        raise ValueError(f"Date column not found in cache CSV: {first_csv}")

    dates = pd.to_datetime(df["Date"], errors="coerce")
    dates = dates.dropna().dt.tz_localize(None).dt.normalize()
    dates = dates[(dates >= DATE_START) & (dates <= DATE_END)]

    if dates.empty:
        raise ValueError(
            f"No trading dates in range {DATE_START.date()} to {DATE_END.date()} from {first_csv}"
        )

    return pd.DatetimeIndex(sorted(dates.unique()))


def load_gc_trades(trades_csv_path: Path) -> pd.DataFrame:
    if not trades_csv_path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_csv_path}")

    df = pd.read_csv(trades_csv_path)
    required_cols = {
        "entry_date",
        "exit_date",
        "pnl",
        "holding_period_days",
        "strategy_name",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in trades CSV: {sorted(missing)}")

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.normalize()
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["entry_date", "exit_date", "pnl"])  # defensive cleanup

    gc_df = df[df["strategy_name"] == "GCStrategy"].copy()
    if gc_df.empty:
        raise ValueError("No GCStrategy trades found in all_transactions.csv")

    gc_df["entry_year"] = gc_df["entry_date"].dt.year
    return gc_df


def build_regime_calendar(trading_dates: pd.DatetimeIndex, regimes: list[str]) -> pd.DataFrame:
    if len(trading_dates) == 0:
        raise ValueError("Trading date list is empty")
    if len(regimes) == 0:
        raise ValueError("Regime list is empty")

    if len(regimes) != len(trading_dates):
        usable_len = min(len(regimes), len(trading_dates))
        print(
            "[WARN] regime count and trading date count differ: "
            f"regimes={len(regimes)}, trading_dates={len(trading_dates)}, using first {usable_len}",
            file=sys.stderr,
        )
    else:
        usable_len = len(regimes)

    return pd.DataFrame(
        {
            "date": trading_dates[:usable_len],
            "regime": regimes[:usable_len],
        }
    )


def print_main_summary(joined_df: pd.DataFrame) -> None:
    print("")
    print("[Regime-wise trade distribution and pnl]")

    summary = (
        joined_df.groupby("regime", dropna=False)
        .agg(
            trades=("pnl", "size"),
            win_rate=("pnl", lambda s: (s > 0).mean() * 100 if len(s) else 0.0),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
        )
        .reset_index()
        .sort_values("trades", ascending=False)
    )

    print("regime | trades | win_rate_pct | total_pnl | avg_pnl")
    for _, row in summary.iterrows():
        regime = str(row["regime"])
        print(
            f"{regime} | {int(row['trades'])} | {row['win_rate']:.2f} | "
            f"{row['total_pnl']:.2f} | {row['avg_pnl']:.2f}"
        )


def print_short_term_summary(joined_df: pd.DataFrame) -> None:
    print("")
    print("[Regime-wise short trades (holding 1-5 days)]")

    short_df = joined_df[
        joined_df["holding_period_days"].notna()
        & (joined_df["holding_period_days"] >= 1)
        & (joined_df["holding_period_days"] <= 5)
    ].copy()

    if short_df.empty:
        print("No short trades (1-5 days) found")
        return

    summary = (
        short_df.groupby("regime", dropna=False)
        .agg(
            short_trades=("pnl", "size"),
            short_win_rate=("pnl", lambda s: (s > 0).mean() * 100 if len(s) else 0.0),
            short_total_pnl=("pnl", "sum"),
        )
        .reset_index()
        .sort_values("short_trades", ascending=False)
    )

    print("regime | short_trades | short_win_rate_pct | short_total_pnl")
    for _, row in summary.iterrows():
        regime = str(row["regime"])
        print(
            f"{regime} | {int(row['short_trades'])} | {row['short_win_rate']:.2f} | "
            f"{row['short_total_pnl']:.2f}"
        )


def print_loss_year_summary(joined_df: pd.DataFrame) -> None:
    print("")
    print("[Trades in loss years (2021, 2022, 2023) by regime]")

    target_years = [2021, 2022, 2023]
    year_df = joined_df[joined_df["entry_year"].isin(target_years)].copy()

    if year_df.empty:
        print("No trades found for years 2021-2023")
        return

    summary = (
        year_df.groupby(["entry_year", "regime"], dropna=False)
        .agg(trades=("pnl", "size"), total_pnl=("pnl", "sum"))
        .reset_index()
        .sort_values(["entry_year", "trades"], ascending=[True, False])
    )

    print("year | regime | trades | total_pnl")
    for _, row in summary.iterrows():
        regime = str(row["regime"])
        print(f"{int(row['entry_year'])} | {regime} | {int(row['trades'])} | {row['total_pnl']:.2f}")


def main() -> int:
    try:
        log_path = _resolve_path(MARKET_LOG_PATH)
        trades_path = _resolve_path(TRADES_CSV_PATH)
        cache_dir = _resolve_path(JQUANTS_CACHE_DIR)
        output_path = _resolve_path(OUTPUT_PATH)

        regimes = load_regime_list(log_path)
        trading_dates = load_trading_dates(cache_dir)
        regime_calendar = build_regime_calendar(trading_dates, regimes)

        gc_trades = load_gc_trades(trades_path)
        joined = gc_trades.merge(
            regime_calendar,
            how="left",
            left_on="entry_date",
            right_on="date",
        ).drop(columns=["date"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(output_path, index=False)

        print("[INFO] Input files loaded successfully")
        print(f"[INFO] Regime records: {len(regimes)}")
        print(f"[INFO] Trading dates in range: {len(trading_dates)}")
        print(f"[INFO] GCStrategy trades: {len(gc_trades)}")
        print(f"[INFO] Joined trades with regime: {len(joined)}")
        print(f"[INFO] Output saved: {output_path.relative_to(PROJECT_ROOT)}")

        print_main_summary(joined)
        print_short_term_summary(joined)
        print_loss_year_summary(joined)

        missing_regime_count = int(joined["regime"].isna().sum())
        print("")
        print(f"[INFO] Trades with missing regime after join: {missing_regime_count}")
        return 0

    except Exception as e:  # top-level guard for robust CLI behavior
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
