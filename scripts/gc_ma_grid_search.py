"""
gc_ma_grid_search - Grid search for GCStrategy moving average windows.

This script runs GCStrategy directly against all CSV files under
data/jquants_cache and evaluates multiple short_window / long_window
combinations without starting the full DSSMS pipeline.

Main features:
- Grid search for 9 moving average combinations
- JSON checkpoints per combination for restart-safe execution
- Per-ticker memory cleanup with gc.collect()
- Final CSV output with aggregate metrics and yearly PnL

Integrated components:
- strategies.gc_strategy_signal.GCStrategy: direct backtest execution
- data/jquants_cache/*.csv: ticker price data source

Safety notes:
- Existing files are not modified
- If Adj Close is missing, Close is copied into Adj Close
- Checkpoints are written only after a combination completes
- Ticker-level errors are logged and skipped so the full run can continue

Author: Backtest Project Team
Created: 2026-03-24
Last Modified: 2026-03-24
"""

from __future__ import annotations

import gc as _gc
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.gc_strategy_signal import GCStrategy  # noqa: E402


JQUANTS_DIR: Path = PROJECT_ROOT / "data" / "jquants_cache"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "output" / "gc_ma_checkpoints"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

TRADING_START = pd.Timestamp("2016-09-01")
TRADING_END = pd.Timestamp("2025-12-31")

STOP_LOSS: float = 0.03
MAX_POSITIONS: int = 3
MIN_ROWS: int = 150
SHARES_PER_TRADE: int = 100

GRID: List[Tuple[int, int]] = [
    (5, 25),
    (5, 50),
    (5, 75),
    (10, 25),
    (10, 50),
    (10, 75),
    (25, 50),
    (25, 75),
    (25, 100),
]


def setup_main_logger() -> logging.Logger:
    """Create the progress logger for terminal output."""
    logger = logging.getLogger("gc_ma_grid_search")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def setup_error_logger(error_log_path: Path) -> logging.Logger:
    """Create the error logger used for per-ticker failures."""
    err_logger = logging.getLogger("gc_ma_grid_search_errors")
    if not err_logger.handlers:
        err_logger.setLevel(logging.ERROR)
        handler = logging.FileHandler(str(error_log_path), encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        err_logger.addHandler(handler)
    return err_logger


def suppress_strategy_logging() -> None:
    """Reduce GCStrategy and BaseStrategy log volume during the grid search."""
    for name in ("GCStrategy", "BaseStrategy"):
        logging.getLogger(name).setLevel(logging.WARNING)


def checkpoint_path(short_window: int, long_window: int) -> Path:
    """Return the JSON checkpoint path for a grid combination."""
    return CHECKPOINT_DIR / f"checkpoint_{short_window}_{long_window}.json"


def load_checkpoint(short_window: int, long_window: int) -> Optional[dict]:
    """Load a checkpoint if the combination has already completed."""
    path = checkpoint_path(short_window, long_window)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "completed":
        return None
    return payload


def save_checkpoint(summary_row: dict) -> None:
    """Persist a completed combination summary to JSON."""
    path = checkpoint_path(summary_row["short_window"], summary_row["long_window"])
    payload = {
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary_row,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_ticker_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load one J-Quants CSV file into the format expected by GCStrategy."""
    try:
        df = pd.read_csv(str(csv_path), index_col=0)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df.index = pd.DatetimeIndex(df["Date"])
        df = df.drop(columns=["Date"], errors="ignore")
        df = df.sort_index()

        for column in ("Open", "High", "Low", "Close", "Volume"):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        if "Close" not in df.columns or "Open" not in df.columns:
            return None

        df = df.dropna(subset=["Open", "Close"])
        df["Adj Close"] = df["Close"]

        if len(df) < MIN_ROWS:
            return None

        return df
    except Exception:
        return None


def extract_trades(result_df: pd.DataFrame, ticker: str) -> List[dict]:
    """Extract completed trades from a GCStrategy backtest result."""
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
                "ticker": ticker,
                "entry_date": str(row.get("entry_date", "")),
                "exit_date": str(row.get("exit_date", "")),
                "profit_loss_per_share": float(row.get("Profit_Loss", 0.0)),
            }
        )
    return trades


def run_single_ticker(
    ticker: str,
    stock_df: pd.DataFrame,
    short_window: int,
    long_window: int,
    logger: logging.Logger,
    err_logger: logging.Logger,
) -> Optional[List[dict]]:
    """Run GCStrategy for one ticker and return extracted trade rows."""
    try:
        strategy = GCStrategy(
            data=stock_df.copy(),
            params={
                "stop_loss": STOP_LOSS,
                "short_window": short_window,
                "long_window": long_window,
            },
            price_column="Adj Close",
            ticker=f"{ticker}.T",
        )
        result = strategy.backtest(
            trading_start_date=TRADING_START,
            trading_end_date=TRADING_END,
        )
        trades = extract_trades(result, ticker)
        logger.info(f"      [GCStrategy] {ticker}.T: {len(trades)} trades")
        return trades
    except Exception as exc:
        err_logger.error(
            f"[ERROR] sw={short_window}, lw={long_window}, ticker={ticker}: {exc}\n"
            f"{traceback.format_exc()}"
        )
        return None


def aggregate_metrics(trades: List[dict]) -> Dict[str, float]:
    """Aggregate trade rows into the final summary metrics."""
    year_columns = [f"yearly_pnl_{year}" for year in range(2016, 2026)]
    empty_metrics: Dict[str, float] = {
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "total_trades": 0.0,
        "net_profit": 0.0,
        "max_drawdown": 0.0,
        **{column: 0.0 for column in year_columns},
    }

    if not trades:
        return empty_metrics

    trades_df = pd.DataFrame(trades)
    trades_df["profit_loss_per_share"] = pd.to_numeric(
        trades_df["profit_loss_per_share"], errors="coerce"
    )
    trades_df["exit_date"] = pd.to_datetime(
        trades_df["exit_date"], errors="coerce", utc=True
    )
    trades_df = trades_df.dropna(subset=["profit_loss_per_share", "exit_date"])

    if trades_df.empty:
        return empty_metrics

    trades_df = trades_df.sort_values("exit_date")
    profit_loss = trades_df["profit_loss_per_share"].astype(float)

    total_trades = int(len(profit_loss))
    wins = int((profit_loss > 0).sum())
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = float(profit_loss[profit_loss > 0].sum())
    gross_loss = float(profit_loss[profit_loss < 0].sum())
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    net_profit = float(profit_loss.sum()) * SHARES_PER_TRADE

    cumulative = (profit_loss * SHARES_PER_TRADE).cumsum().tolist()
    peak = cumulative[0]
    max_drawdown = 0.0
    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_drawdown:
            max_drawdown = float(drawdown)

    yearly_pnl: Dict[str, float] = {column: 0.0 for column in year_columns}
    yearly_grouped = (
        trades_df.groupby(trades_df["exit_date"].dt.year)["profit_loss_per_share"].sum()
        * SHARES_PER_TRADE
    )
    for year in range(2016, 2026):
        yearly_pnl[f"yearly_pnl_{year}"] = float(yearly_grouped.get(year, 0.0))

    return {
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_trades": float(total_trades),
        "net_profit": net_profit,
        "max_drawdown": max_drawdown,
        **yearly_pnl,
    }


def build_summary_row(short_window: int, long_window: int, trades: List[dict]) -> dict:
    """Build one output row for a completed combination."""
    metrics = aggregate_metrics(trades)
    row = {
        "short_window": short_window,
        "long_window": long_window,
        "profit_factor": metrics["profit_factor"],
        "win_rate": metrics["win_rate"],
        "total_trades": int(metrics["total_trades"]),
        "net_profit": metrics["net_profit"],
        "max_drawdown": metrics["max_drawdown"],
    }
    for year in range(2016, 2026):
        key = f"yearly_pnl_{year}"
        row[key] = metrics[key]
    return row


def load_summary_from_checkpoint(payload: dict) -> Optional[dict]:
    """Extract a summary row from a valid checkpoint payload."""
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    return summary


def print_summary(summary_rows: List[dict], logger: logging.Logger) -> None:
    """Print a concise table of the finished grid search."""
    header = (
        f"{'short':>5} | {'long':>5} | {'PF':>8} | {'Win%':>6} | "
        f"{'Trades':>7} | {'Net Profit(JPY)':>16}"
    )
    separator = "-" * len(header)
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    for row in summary_rows:
        pf_value = row["profit_factor"]
        pf_text = "inf" if pf_value == float("inf") else f"{pf_value:.2f}"
        logger.info(
            f"{row['short_window']:>5} | {row['long_window']:>5} | {pf_text:>8} | "
            f"{row['win_rate']:>5.1f}% | {row['total_trades']:>7,} | "
            f"{row['net_profit']:>+16,.0f}"
        )
    logger.info(separator)


def save_final_csv(summary_rows: List[dict], logger: logging.Logger) -> Path:
    """Write the final grid search summary CSV."""
    today = datetime.now().strftime("%Y%m%d")
    output_path = OUTPUT_DIR / f"gc_ma_grid_result_{today}.csv"

    ordered_columns = [
        "short_window",
        "long_window",
        "profit_factor",
        "win_rate",
        "total_trades",
        "net_profit",
        "max_drawdown",
    ] + [f"yearly_pnl_{year}" for year in range(2016, 2026)]

    rows_for_csv: List[dict] = []
    for row in summary_rows:
        csv_row = {column: row.get(column, 0.0) for column in ordered_columns}
        if csv_row["profit_factor"] == float("inf"):
            csv_row["profit_factor"] = 999999.0
        rows_for_csv.append(csv_row)

    result_df = pd.DataFrame(rows_for_csv, columns=ordered_columns)
    result_df.to_csv(str(output_path), index=False, encoding="utf-8-sig")
    logger.info(f"Final CSV saved: {output_path}")
    return output_path


def main() -> None:
    """Run the moving-average grid search across all cached tickers."""
    logger = setup_main_logger()
    suppress_strategy_logging()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    error_log_path = CHECKPOINT_DIR / "error_log.txt"
    err_logger = setup_error_logger(error_log_path)

    ticker_files = sorted(JQUANTS_DIR.glob("*.csv"))
    if not ticker_files:
        logger.info(f"[ERROR] No J-Quants cache files found: {JQUANTS_DIR}")
        return

    logger.info(f"J-Quants cache files: {len(ticker_files)}")
    logger.info(f"Trading period: {TRADING_START.date()} to {TRADING_END.date()}")
    logger.info(f"Fixed stop_loss: {STOP_LOSS:.2%}")
    logger.info(f"Fixed max_positions metadata: {MAX_POSITIONS}")

    total_combinations = len(GRID)
    summary_rows: List[dict] = []

    for combination_index, (short_window, long_window) in enumerate(GRID, start=1):
        checkpoint = load_checkpoint(short_window, long_window)
        if checkpoint is not None:
            summary = load_summary_from_checkpoint(checkpoint)
            if summary is not None:
                logger.info(
                    f"[{combination_index}/{total_combinations}] "
                    f"sw={short_window}, lw={long_window} skipped by checkpoint"
                )
                summary_rows.append(summary)
                continue

        logger.info(
            f"[{combination_index}/{total_combinations}] "
            f"sw={short_window}, lw={long_window} started"
        )

        combination_trades: List[dict] = []
        total_tickers = len(ticker_files)

        for ticker_index, ticker_path in enumerate(ticker_files, start=1):
            ticker = ticker_path.stem
            logger.info(
                f"    [{ticker_index}/{total_tickers}] processing {ticker}.csv"
            )

            stock_df = load_ticker_data(ticker_path)
            if stock_df is None:
                err_logger.error(
                    f"[SKIP] sw={short_window}, lw={long_window}, ticker={ticker}: "
                    "load failed or insufficient rows"
                )
                logger.info(f"      [SKIP] {ticker}.T: invalid or insufficient data")
                continue

            trades = run_single_ticker(
                ticker=ticker,
                stock_df=stock_df,
                short_window=short_window,
                long_window=long_window,
                logger=logger,
                err_logger=err_logger,
            )

            del stock_df
            _gc.collect()

            if trades is None:
                logger.info(f"      [ERROR] {ticker}.T: backtest failed")
                continue

            combination_trades.extend(trades)

        summary_row = build_summary_row(
            short_window=short_window,
            long_window=long_window,
            trades=combination_trades,
        )
        save_checkpoint(summary_row)
        summary_rows.append(summary_row)

        pf_value = summary_row["profit_factor"]
        pf_text = "inf" if pf_value == float("inf") else f"{pf_value:.2f}"
        logger.info(
            f"  -> PF={pf_text}, Win={summary_row['win_rate']:.1f}%, "
            f"Trades={summary_row['total_trades']:,}, "
            f"Net={summary_row['net_profit']:+,.0f} JPY"
        )

    summary_rows = sorted(summary_rows, key=lambda row: (row["short_window"], row["long_window"]))

    logger.info("")
    logger.info("=== GC MA grid search completed ===")
    print_summary(summary_rows, logger)
    save_final_csv(summary_rows, logger)


if __name__ == "__main__":
    main()