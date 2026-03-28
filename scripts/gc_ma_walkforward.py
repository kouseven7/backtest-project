"""
gc_ma_walkforward - Walk-forward evaluation for GCStrategy MA candidates.

This script runs a lightweight walk-forward test over all CSV files under
/data/jquants_cache using GCStrategy directly (without DSSMS full startup).
It evaluates fixed MA candidates in in-sample, out-of-sample, and full periods.

Main features:
- Candidate-based walk-forward evaluation for MA windows
- In-sample (2016-09-01 to 2022-12-31) and out-of-sample (2023-01-01 to 2025-12-31)
- Full-period reference run (2016-09-01 to 2025-12-31)
- Per-ticker memory cleanup with gc.collect()

Integrated components:
- strategies.gc_strategy_signal.GCStrategy: direct backtest execution
- data/jquants_cache/*.csv: ticker price data source

Safety notes:
- Existing files are not modified
- CSV load starts from 2016-01-01 to secure MA warmup
- Adj Close is always filled from Close after loading
- Ticker-level failures are logged and skipped

Author: Backtest Project Team
Created: 2026-03-25
Last Modified: 2026-03-25
"""

from __future__ import annotations

import gc as _gc
from datetime import datetime
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.gc_strategy_signal import GCStrategy  # noqa: E402


JQUANTS_DIR: Path = PROJECT_ROOT / "data" / "jquants_cache"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

DATA_LOAD_START = pd.Timestamp("2016-01-01")
IS_START = pd.Timestamp("2016-09-01")
IS_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")
OOS_END = pd.Timestamp("2025-12-31")
FULL_START = pd.Timestamp("2016-09-01")
FULL_END = pd.Timestamp("2025-12-31")

STOP_LOSS: float = 0.03
MIN_ROWS: int = 150
SHARES_PER_TRADE: int = 100

CANDIDATES: List[Dict[str, int]] = [
    {"short_window": 25, "long_window": 100},
    {"short_window": 5, "long_window": 75},
    {"short_window": 5, "long_window": 25},
]


def setup_logger() -> logging.Logger:
    """Create the terminal logger for progress and summary output."""
    logger = logging.getLogger("gc_ma_walkforward")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result_path = OUTPUT_DIR / (
            f"gc_ma_walkforward_result_{datetime.now().strftime('%Y%m%d')}.txt"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(str(result_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def suppress_strategy_logging() -> None:
    """Reduce GCStrategy/BaseStrategy log volume while running the script."""
    for name in ("GCStrategy", "BaseStrategy"):
        logging.getLogger(name).setLevel(logging.WARNING)


def load_ticker_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load one CSV ticker file and normalize columns for GCStrategy."""
    try:
        df = pd.read_csv(str(csv_path), index_col=0)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df.index = pd.DatetimeIndex(df["Date"])
        if df.index.tz is not None:
            # Align with naive trading_start/end timestamps used by backtest.
            df.index = df.index.tz_localize(None)
        df = df.drop(columns=["Date"], errors="ignore")
        df = df.sort_index()

        for column in ("Open", "High", "Low", "Close", "Volume"):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        if "Close" not in df.columns or "Open" not in df.columns:
            return None

        df = df[df.index >= DATA_LOAD_START]
        df = df.dropna(subset=["Open", "Close"])

        # Required by user request: always complement Adj Close from Close.
        df["Adj Close"] = df["Close"]

        if len(df) < MIN_ROWS:
            return None

        return df
    except Exception:
        return None


def extract_trades(result_df: pd.DataFrame) -> List[dict]:
    """Extract completed trades from one GCStrategy backtest result."""
    entries = result_df[result_df["Entry_Signal"] == 1].copy()
    exits = result_df[result_df["Exit_Signal"] == -1].copy()

    if entries.empty or exits.empty:
        return []

    entries = entries[["Trade_ID"]].copy()
    entries["entry_date"] = entries.index.astype(str)

    exits = exits[["Trade_ID", "Profit_Loss"]].copy()
    exits["exit_date"] = exits.index.astype(str)

    merged = exits.merge(entries, on="Trade_ID", how="left")

    trades: List[dict] = []
    for _, row in merged.iterrows():
        trades.append(
            {
                "entry_date": str(row.get("entry_date", "")),
                "exit_date": str(row.get("exit_date", "")),
                "profit_loss_per_share": float(row.get("Profit_Loss", 0.0)),
            }
        )
    return trades


def aggregate_metrics(trades: List[dict]) -> Dict[str, float]:
    """Aggregate trade list into PF/Win/Trades/NetProfit/MaxDD metrics."""
    empty = {
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "total_trades": 0.0,
        "net_profit": 0.0,
        "max_drawdown": 0.0,
    }

    if not trades:
        return empty

    trades_df = pd.DataFrame(trades)
    trades_df["profit_loss_per_share"] = pd.to_numeric(
        trades_df["profit_loss_per_share"], errors="coerce"
    )
    trades_df["exit_date"] = pd.to_datetime(
        trades_df["exit_date"], errors="coerce", utc=True
    )
    trades_df = trades_df.dropna(subset=["profit_loss_per_share", "exit_date"])

    if trades_df.empty:
        return empty

    trades_df = trades_df.sort_values("exit_date")
    pnl_per_share = trades_df["profit_loss_per_share"].astype(float)

    total_trades = int(len(pnl_per_share))
    wins = int((pnl_per_share > 0).sum())
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = float(pnl_per_share[pnl_per_share > 0].sum())
    gross_loss = float(pnl_per_share[pnl_per_share < 0].sum())
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    net_profit = float(pnl_per_share.sum()) * SHARES_PER_TRADE

    cumulative = (pnl_per_share * SHARES_PER_TRADE).cumsum().tolist()
    peak = cumulative[0]
    max_drawdown = 0.0
    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_drawdown:
            max_drawdown = float(drawdown)

    return {
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_trades": float(total_trades),
        "net_profit": net_profit,
        "max_drawdown": max_drawdown,
    }


def format_pf(value: float) -> str:
    """Format PF with special handling for infinity."""
    return "inf" if value == float("inf") else f"{value:.2f}"


def format_rate(value: float) -> str:
    """Format a percentage value."""
    return f"{value:.1f}%"


def format_int(value: float) -> str:
    """Format a number with comma separators."""
    return f"{int(round(value)):,}"


def format_signed_jpy(value: float) -> str:
    """Format JPY with explicit sign and comma separators."""
    return f"{value:+,.0f}"


def run_one_backtest(
    stock_df: pd.DataFrame,
    ticker_code: str,
    short_window: int,
    long_window: int,
    trading_start_date: pd.Timestamp,
    trading_end_date: pd.Timestamp,
) -> List[dict]:
    """Run one GCStrategy backtest and return extracted completed trades."""
    strategy = GCStrategy(
        data=stock_df,
        params={
            "stop_loss": STOP_LOSS,
            "short_window": short_window,
            "long_window": long_window,
        },
        price_column="Adj Close",
        ticker=ticker_code,
    )
    results = strategy.backtest(
        trading_start_date=trading_start_date,
        trading_end_date=trading_end_date,
    )
    return extract_trades(results)


def evaluate_candidate(
    ticker_files: List[Path],
    short_window: int,
    long_window: int,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    """Evaluate one MA candidate over all tickers for IS/OOS/FULL periods."""
    trades_is: List[dict] = []
    trades_oos: List[dict] = []
    trades_full: List[dict] = []

    total_tickers = len(ticker_files)
    for ticker_index, ticker_path in enumerate(ticker_files, start=1):
        ticker = ticker_path.stem
        ticker_code = f"{ticker}.T"
        logger.info(
            f"  [{ticker_index}/{total_tickers}] {ticker_code} processing"
        )

        stock_df = load_ticker_data(ticker_path)
        if stock_df is None:
            logger.warning(f"    [SKIP] {ticker_code}: invalid or insufficient data")
            continue

        try:
            candidate_df = stock_df.copy()

            ticker_is = run_one_backtest(
                stock_df=candidate_df,
                ticker_code=ticker_code,
                short_window=short_window,
                long_window=long_window,
                trading_start_date=IS_START,
                trading_end_date=IS_END,
            )
            ticker_oos = run_one_backtest(
                stock_df=candidate_df,
                ticker_code=ticker_code,
                short_window=short_window,
                long_window=long_window,
                trading_start_date=OOS_START,
                trading_end_date=OOS_END,
            )
            ticker_full = run_one_backtest(
                stock_df=candidate_df,
                ticker_code=ticker_code,
                short_window=short_window,
                long_window=long_window,
                trading_start_date=FULL_START,
                trading_end_date=FULL_END,
            )

            trades_is.extend(ticker_is)
            trades_oos.extend(ticker_oos)
            trades_full.extend(ticker_full)

            logger.info(
                f"    trades IS={len(ticker_is):,}, OOS={len(ticker_oos):,}, FULL={len(ticker_full):,}"
            )
        except Exception as exc:
            logger.error(f"    [ERROR] {ticker_code}: {exc}")
            logger.debug(traceback.format_exc())
        finally:
            del stock_df
            _gc.collect()

    return {
        "is": aggregate_metrics(trades_is),
        "oos": aggregate_metrics(trades_oos),
        "full": aggregate_metrics(trades_full),
    }


def print_candidate_report(
    short_window: int,
    long_window: int,
    metrics_by_period: Dict[str, Dict[str, float]],
    logger: logging.Logger,
) -> None:
    """Print one candidate block in the requested walk-forward format."""
    is_metrics = metrics_by_period["is"]
    oos_metrics = metrics_by_period["oos"]
    full_metrics = metrics_by_period["full"]

    is_pf = float(is_metrics["profit_factor"])
    oos_pf = float(oos_metrics["profit_factor"])
    if is_pf > 0 and is_pf != float("inf"):
        ratio = oos_pf / is_pf
    elif is_pf == float("inf") and oos_pf == float("inf"):
        ratio = 1.0
    else:
        ratio = 0.0

    logger.info("")
    logger.info(f"[{short_window}x{long_window}]")
    logger.info(
        "                   PF      Win%   Trades   NetProfit(JPY)   MaxDD(JPY)"
    )
    logger.info(
        "in-sample        "
        f"{format_pf(is_metrics['profit_factor']):>5}    "
        f"{format_rate(is_metrics['win_rate']):>6}   "
        f"{format_int(is_metrics['total_trades']):>7}    "
        f"{format_signed_jpy(is_metrics['net_profit']):>13}      "
        f"{format_int(is_metrics['max_drawdown']):>11}"
    )
    logger.info(
        "out-of-sample    "
        f"{format_pf(oos_metrics['profit_factor']):>5}    "
        f"{format_rate(oos_metrics['win_rate']):>6}   "
        f"{format_int(oos_metrics['total_trades']):>7}    "
        f"{format_signed_jpy(oos_metrics['net_profit']):>13}      "
        f"{format_int(oos_metrics['max_drawdown']):>11}"
    )
    logger.info(
        "full period      "
        f"{format_pf(full_metrics['profit_factor']):>5}    "
        f"{format_rate(full_metrics['win_rate']):>6}   "
        f"{format_int(full_metrics['total_trades']):>7}    "
        f"{format_signed_jpy(full_metrics['net_profit']):>13}      "
        f"{format_int(full_metrics['max_drawdown']):>11}"
    )
    logger.info(
        f"OOS/IS PF ratio: {ratio:.2f}  (1.0以上で過学習なし)"
    )


def main() -> None:
    """Run walk-forward tests for configured MA candidates."""
    logger = setup_logger()
    suppress_strategy_logging()

    ticker_files = sorted(JQUANTS_DIR.glob("*.csv"))
    if not ticker_files:
        logger.error(f"[ERROR] No J-Quants cache files found: {JQUANTS_DIR}")
        return

    logger.info(f"J-Quants cache files: {len(ticker_files)}")
    logger.info(f"Data load start: {DATA_LOAD_START.date()} (warmup secured)")
    logger.info(f"In-sample period: {IS_START.date()} to {IS_END.date()}")
    logger.info(f"Out-of-sample period: {OOS_START.date()} to {OOS_END.date()}")
    logger.info(f"Full period: {FULL_START.date()} to {FULL_END.date()}")
    logger.info(f"Fixed stop_loss: {STOP_LOSS:.2%}")

    logger.info("")
    logger.info("=== Walk-Forward Test Results ===")

    for idx, candidate in enumerate(CANDIDATES, start=1):
        sw = int(candidate["short_window"])
        lw = int(candidate["long_window"])

        logger.info("")
        logger.info(f"[{idx}/{len(CANDIDATES)}] Candidate {sw}x{lw} started")

        metrics_by_period = evaluate_candidate(
            ticker_files=ticker_files,
            short_window=sw,
            long_window=lw,
            logger=logger,
        )
        print_candidate_report(
            short_window=sw,
            long_window=lw,
            metrics_by_period=metrics_by_period,
            logger=logger,
        )


if __name__ == "__main__":
    main()
