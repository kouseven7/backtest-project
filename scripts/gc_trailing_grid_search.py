"""
gc_trailing_grid_search - Grid search for GCStrategy trailing_stop_pct.

This script runs GCStrategy directly against all CSV files under
data/jquants_cache and evaluates multiple trailing_stop_pct values
without starting the full DSSMS pipeline.

Design pattern follows scripts/gc_ma_grid_search.py exactly.

Main features:
- Grid search for trailing_stop_pct candidates (including None)
- JSON checkpoints per candidate for restart-safe execution
- Walk-forward analysis (IS: 2016-2022 / OOS: 2023-2025)
- Per-ticker memory cleanup with gc.collect()
- Final CSV output with aggregate metrics and yearly PnL

Integrated components:
- strategies.gc_strategy_signal.GCStrategy: direct backtest execution
- data/jquants_cache/*.csv: ticker price data source

Safety notes:
- Existing files are not modified
- If Adj Close is missing, Close is copied into Adj Close
- Checkpoints are written only after a candidate completes
- Ticker-level errors are logged and skipped so the full run can continue
- trailing_stop_pct=None is converted to 0.999 (effectively disabled)
  because GCStrategy does not handle None internally

Fixed parameters (DO NOT CHANGE):
- stop_loss = 0.03
- short_window = 5
- long_window = 75
- max_positions = 3 (metadata only, not enforced per-ticker)

Author: Backtest Project Team
Created: 2026-03-26
Last Modified: 2026-03-26
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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JQUANTS_DIR: Path = PROJECT_ROOT / "data" / "jquants_cache"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "output" / "gc_trailing_checkpoints"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Trading period (full grid search)
# ---------------------------------------------------------------------------
TRADING_START = pd.Timestamp("2016-09-01")
TRADING_END = pd.Timestamp("2025-12-31")

# Walk-forward split
WF_IS_START = pd.Timestamp("2016-09-01")
WF_IS_END = pd.Timestamp("2022-12-31")
WF_OOS_START = pd.Timestamp("2023-01-01")
WF_OOS_END = pd.Timestamp("2025-12-31")

# ---------------------------------------------------------------------------
# Fixed parameters (DO NOT CHANGE)
# ---------------------------------------------------------------------------
STOP_LOSS: float = 0.03
SHORT_WINDOW: int = 5
LONG_WINDOW: int = 75
MAX_POSITIONS: int = 3
MIN_ROWS: int = 150
SHARES_PER_TRADE: int = 100

# ---------------------------------------------------------------------------
# Candidate values
# None = trailing stop disabled (converted to 0.999 internally)
# ---------------------------------------------------------------------------
CANDIDATES: List[Optional[float]] = [ None]

# Sentinel value passed to GCStrategy when trailing_stop_pct=None
# 0.999 means the trailing stop triggers only when price drops 99.9% from peak,
# which is effectively disabled in practice.
_NONE_SENTINEL: float = 0.999


def _candidate_label(value: Optional[float]) -> str:
    """Human-readable label for a candidate value (used in logs / CSV)."""
    return "None" if value is None else f"{value:.3f}"


def _candidate_to_param(value: Optional[float]) -> float:
    """Convert a candidate value to the float passed to GCStrategy."""
    return _NONE_SENTINEL if value is None else value


def _checkpoint_key(value: Optional[float]) -> str:
    """Unique string key used for checkpoint file naming."""
    return "None" if value is None else f"{value:.4f}"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_main_logger() -> logging.Logger:
    """Create the progress logger for terminal output."""
    logger = logging.getLogger("gc_trailing_grid_search")
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
    err_logger = logging.getLogger("gc_trailing_grid_search_errors")
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


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_path(candidate: Optional[float]) -> Path:
    """Return the JSON checkpoint path for a candidate value."""
    return CHECKPOINT_DIR / f"checkpoint_trailing_{_checkpoint_key(candidate)}.json"


def load_checkpoint(candidate: Optional[float]) -> Optional[dict]:
    """Load a checkpoint if the candidate has already completed."""
    path = checkpoint_path(candidate)
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
    """Persist a completed candidate summary to JSON."""
    # Use label to reconstruct candidate key for the path
    label = summary_row["trailing_stop_pct"]
    candidate = None if label == "None" else float(label)
    path = checkpoint_path(candidate)

    payload = {
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary_row,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_summary_from_checkpoint(payload: dict) -> Optional[dict]:
    """Extract a summary row from a valid checkpoint payload."""
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    return summary


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-ticker backtest
# ---------------------------------------------------------------------------

def run_single_ticker(
    ticker: str,
    stock_df: pd.DataFrame,
    trailing_stop_param: float,
    trading_start: pd.Timestamp,
    trading_end: pd.Timestamp,
    logger: logging.Logger,
    err_logger: logging.Logger,
    candidate_label: str,
) -> Optional[List[dict]]:
    """Run GCStrategy for one ticker and return extracted trade rows."""
    try:
        strategy = GCStrategy(
            data=stock_df.copy(),
            params={
                "stop_loss": STOP_LOSS,
                "short_window": SHORT_WINDOW,
                "long_window": LONG_WINDOW,
                "trailing_stop_pct": trailing_stop_param,
            },
            price_column="Adj Close",
            ticker=f"{ticker}.T",
        )
        result = strategy.backtest(
            trading_start_date=trading_start,
            trading_end_date=trading_end,
        )
        trades = extract_trades(result, ticker)
        logger.info(
            f"      [GCStrategy trailing={candidate_label}] {ticker}.T: {len(trades)} trades"
        )
        return trades
    except Exception as exc:
        err_logger.error(
            f"[ERROR] trailing={candidate_label}, ticker={ticker}: {exc}\n"
            f"{traceback.format_exc()}"
        )
        return None


# ---------------------------------------------------------------------------
# Trade extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(
    trades: List[dict],
    start_year: int = 2016,
    end_year: int = 2025,
) -> Dict[str, float]:
    """Aggregate trade rows into the final summary metrics."""
    year_columns = [f"yearly_pnl_{year}" for year in range(start_year, end_year + 1)]
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
    for year in range(start_year, end_year + 1):
        yearly_pnl[f"yearly_pnl_{year}"] = float(yearly_grouped.get(year, 0.0))

    return {
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_trades": float(total_trades),
        "net_profit": net_profit,
        "max_drawdown": max_drawdown,
        **yearly_pnl,
    }


# ---------------------------------------------------------------------------
# Summary row builder
# ---------------------------------------------------------------------------

def build_summary_row(candidate: Optional[float], trades: List[dict]) -> dict:
    """Build one output row for a completed candidate."""
    metrics = aggregate_metrics(trades)
    row = {
        "trailing_stop_pct": _candidate_label(candidate),
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


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary(summary_rows: List[dict], logger: logging.Logger) -> None:
    """Print a concise table of the finished grid search."""
    header = (
        f"{'trailing':>9} | {'PF':>8} | {'Win%':>6} | "
        f"{'Trades':>7} | {'Net Profit(JPY)':>16} | {'MaxDD(JPY)':>12}"
    )
    separator = "-" * len(header)
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    for row in summary_rows:
        pf_value = row["profit_factor"]
        pf_text = "inf" if pf_value == float("inf") else f"{pf_value:.2f}"
        logger.info(
            f"{row['trailing_stop_pct']:>9} | {pf_text:>8} | "
            f"{row['win_rate']:>5.1f}% | {row['total_trades']:>7,} | "
            f"{row['net_profit']:>+16,.0f} | {row['max_drawdown']:>12,.0f}"
        )
    logger.info(separator)


def save_final_csv(summary_rows: List[dict], logger: logging.Logger) -> Path:
    """Write the final grid search summary CSV."""
    today = datetime.now().strftime("%Y%m%d")
    output_path = OUTPUT_DIR / f"gc_trailing_grid_result_{today}.csv"

    ordered_columns = [
        "trailing_stop_pct",
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


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------

def run_walkforward(
    ticker_files: List[Path],
    logger: logging.Logger,
    err_logger: logging.Logger,
) -> None:
    """
    Run walk-forward analysis for all candidates.

    IS  period: WF_IS_START  ~ WF_IS_END
    OOS period: WF_OOS_START ~ WF_OOS_END
    """
    logger.info("")
    logger.info("=== Walk-Forward Analysis ===")
    logger.info(
        f"IS : {WF_IS_START.date()} to {WF_IS_END.date()}  "
        f"| OOS: {WF_OOS_START.date()} to {WF_OOS_END.date()}"
    )

    wf_results: List[dict] = []

    for candidate in CANDIDATES:
        label = _candidate_label(candidate)
        trailing_param = _candidate_to_param(candidate)
        logger.info(f"  [WF] trailing={label} ...")

        is_trades: List[dict] = []
        oos_trades: List[dict] = []

        for ticker_path in ticker_files:
            ticker = ticker_path.stem

            stock_df = load_ticker_data(ticker_path)
            if stock_df is None:
                continue

            # IS
            is_result = run_single_ticker(
                ticker=ticker,
                stock_df=stock_df,
                trailing_stop_param=trailing_param,
                trading_start=WF_IS_START,
                trading_end=WF_IS_END,
                logger=logging.getLogger("gc_trailing_wf_dummy"),  # suppress ticker logs
                err_logger=err_logger,
                candidate_label=label,
            )
            if is_result:
                is_trades.extend(is_result)

            # OOS
            oos_result = run_single_ticker(
                ticker=ticker,
                stock_df=stock_df,
                trailing_stop_param=trailing_param,
                trading_start=WF_OOS_START,
                trading_end=WF_OOS_END,
                logger=logging.getLogger("gc_trailing_wf_dummy"),
                err_logger=err_logger,
                candidate_label=label,
            )
            if oos_result:
                oos_trades.extend(oos_result)

            del stock_df
            _gc.collect()

        is_metrics = aggregate_metrics(is_trades, start_year=2016, end_year=2022)
        oos_metrics = aggregate_metrics(oos_trades, start_year=2023, end_year=2025)

        is_pf = is_metrics["profit_factor"]
        oos_pf = oos_metrics["profit_factor"]
        oos_is_ratio = (oos_pf / is_pf) if (is_pf > 0 and is_pf != float("inf")) else float("nan")

        wf_results.append(
            {
                "trailing_stop_pct": label,
                "is_pf": is_pf,
                "oos_pf": oos_pf,
                "oos_is_ratio": oos_is_ratio,
                "oos_net_profit": oos_metrics["net_profit"],
                "oos_trades": int(oos_metrics["total_trades"]),
            }
        )

    # Print walk-forward table
    header = (
        f"{'trailing':>9} | {'IS PF':>7} | {'OOS PF':>7} | "
        f"{'OOS/IS':>7} | {'OOS Net(JPY)':>14} | {'OOS Trades':>10}"
    )
    separator = "-" * len(header)
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    for row in wf_results:
        def _fmt_pf(v: float) -> str:
            if v == float("inf"):
                return "    inf"
            if v != v:  # nan
                return "    N/A"
            return f"{v:>7.2f}"

        ratio_str = "    N/A" if row["oos_is_ratio"] != row["oos_is_ratio"] else f"{row['oos_is_ratio']:>7.2f}"
        logger.info(
            f"{row['trailing_stop_pct']:>9} | {_fmt_pf(row['is_pf'])} | {_fmt_pf(row['oos_pf'])} | "
            f"{ratio_str} | {row['oos_net_profit']:>+14,.0f} | {row['oos_trades']:>10,}"
        )
    logger.info(separator)
    logger.info("Judgment criteria:")
    logger.info("  - OOS/IS ratio >= 1.0  (no overfitting)")
    logger.info("  - OOS trades   >= 100  (sufficient sample)")
    logger.info("  - Profit hill shape confirmed in grid search CSV")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the trailing_stop_pct grid search across all cached tickers."""
    logger = setup_main_logger()
    suppress_strategy_logging()

    # Suppress walk-forward ticker-level log (we use a dummy logger there)
    logging.getLogger("gc_trailing_wf_dummy").setLevel(logging.CRITICAL)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    error_log_path = CHECKPOINT_DIR / "error_log.txt"
    err_logger = setup_error_logger(error_log_path)

    ticker_files = sorted(JQUANTS_DIR.glob("*.csv"))
    if not ticker_files:
        logger.info(f"[ERROR] No J-Quants cache files found: {JQUANTS_DIR}")
        return

    logger.info(f"J-Quants cache files : {len(ticker_files)}")
    logger.info(f"Trading period       : {TRADING_START.date()} to {TRADING_END.date()}")
    logger.info(f"Fixed stop_loss      : {STOP_LOSS:.2%}")
    logger.info(f"Fixed short_window   : {SHORT_WINDOW}")
    logger.info(f"Fixed long_window    : {LONG_WINDOW}")
    logger.info(f"Fixed max_positions  : {MAX_POSITIONS} (metadata only)")
    logger.info(f"Candidates           : {[_candidate_label(c) for c in CANDIDATES]}")
    logger.info(
        f"Note: trailing_stop_pct=None is passed as {_NONE_SENTINEL} "
        f"(effectively disabled) to GCStrategy"
    )

    total_candidates = len(CANDIDATES)
    summary_rows: List[dict] = []

    # -----------------------------------------------------------------------
    # Grid search
    # -----------------------------------------------------------------------
    for candidate_index, candidate in enumerate(CANDIDATES, start=1):
        label = _candidate_label(candidate)

        checkpoint = load_checkpoint(candidate)
        if checkpoint is not None:
            summary = load_summary_from_checkpoint(checkpoint)
            if summary is not None:
                logger.info(
                    f"[{candidate_index}/{total_candidates}] "
                    f"trailing={label} skipped by checkpoint"
                )
                summary_rows.append(summary)
                continue

        logger.info(
            f"[{candidate_index}/{total_candidates}] "
            f"trailing={label} started"
        )

        trailing_param = _candidate_to_param(candidate)
        candidate_trades: List[dict] = []
        total_tickers = len(ticker_files)

        for ticker_index, ticker_path in enumerate(ticker_files, start=1):
            ticker = ticker_path.stem
            logger.info(
                f"    [{ticker_index}/{total_tickers}] processing {ticker}.csv"
            )

            stock_df = load_ticker_data(ticker_path)
            if stock_df is None:
                err_logger.error(
                    f"[SKIP] trailing={label}, ticker={ticker}: "
                    "load failed or insufficient rows"
                )
                logger.info(f"      [SKIP] {ticker}.T: invalid or insufficient data")
                continue

            trades = run_single_ticker(
                ticker=ticker,
                stock_df=stock_df,
                trailing_stop_param=trailing_param,
                trading_start=TRADING_START,
                trading_end=TRADING_END,
                logger=logger,
                err_logger=err_logger,
                candidate_label=label,
            )

            del stock_df
            _gc.collect()

            if trades is None:
                logger.info(f"      [ERROR] {ticker}.T: backtest failed")
                continue

            candidate_trades.extend(trades)

        summary_row = build_summary_row(candidate, candidate_trades)
        save_checkpoint(summary_row)
        summary_rows.append(summary_row)

        pf_value = summary_row["profit_factor"]
        pf_text = "inf" if pf_value == float("inf") else f"{pf_value:.2f}"
        logger.info(
            f"  -> PF={pf_text}, Win={summary_row['win_rate']:.1f}%, "
            f"Trades={summary_row['total_trades']:,}, "
            f"Net={summary_row['net_profit']:+,.0f} JPY, "
            f"MaxDD={summary_row['max_drawdown']:,.0f} JPY"
        )

    # -----------------------------------------------------------------------
    # Print full-period summary and save CSV
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=== GC trailing_stop_pct grid search completed ===")
    print_summary(summary_rows, logger)
    save_final_csv(summary_rows, logger)

    # -----------------------------------------------------------------------
    # Walk-forward analysis
    # -----------------------------------------------------------------------
    run_walkforward(ticker_files, logger, err_logger)


if __name__ == "__main__":
    main()
