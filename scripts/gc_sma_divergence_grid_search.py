"""
gc_sma_divergence_grid_search - Grid search and walk-forward for GCStrategy SMA divergence.

This script runs GCStrategy across all J-Quants cache CSV files and optimizes
sma_divergence_threshold with fixed core parameters. It records full-period
metrics, yearly PF, and walk-forward IS/OOS validation metrics from a single
full-period backtest per ticker.

Main features:
- Grid search for sma_divergence_threshold candidates
- Checkpoint resume by candidate using pickle files
- Full-period aggregation (2016-01-01 to 2025-12-31)
- Walk-forward aggregation from one backtest result per ticker
- Per-ticker error skip and consolidated error log output

Integrated components:
- strategies.gc_strategy_signal.GCStrategy: strategy backtest engine
- data/jquants_cache/*.csv: input price universe

Safety notes:
- Invalid ticker data is skipped with error logging
- Adj Close is filled from Close for compatibility
- Checkpoint is written only after candidate completion
- PF=inf is converted to 999.99 only at CSV export time

Author: Backtest Project Team
Created: 2026-03-28
Last Modified: 2026-03-28
"""

from __future__ import annotations

import gc as _gc
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.gc_strategy_signal import GCStrategy  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JQUANTS_DIR: Path = PROJECT_ROOT / "data" / "jquants_cache"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
CHECKPOINT_DIR: Path = OUTPUT_DIR / "gc_sma_divergence_checkpoints"


# ---------------------------------------------------------------------------
# Fixed parameters (DO NOT CHANGE)
# ---------------------------------------------------------------------------
SHORT_WINDOW: int = 5
LONG_WINDOW: int = 75
STOP_LOSS: float = 0.03
TRAILING_STOP_PCT: float = 0.999
TREND_STRENGTH_PERCENTILE: int = 60
USE_ENTRY_FILTER: bool = True
TREND_STRENGTH_ENABLED: bool = True
SMA_DIVERGENCE_ENABLED: bool = True
MAX_POSITIONS: int = 3

SHARES_PER_TRADE: int = 100
MIN_ROWS: int = 150


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
CANDIDATES: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------
FULL_START = pd.Timestamp("2016-01-01")
FULL_END = pd.Timestamp("2025-12-31")

IS_START = pd.Timestamp("2016-01-01")
IS_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")
OOS_END = pd.Timestamp("2025-12-31")


def checkpoint_path(sma_divergence_threshold: float) -> Path:
    return CHECKPOINT_DIR / f"checkpoint_{sma_divergence_threshold:.1f}.pkl"


def suppress_strategy_logging() -> None:
    logging.disable(logging.WARNING)


def load_checkpoint(sma_divergence_threshold: float) -> Optional[dict]:
    path = checkpoint_path(sma_divergence_threshold)
    if not path.exists():
        return None

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "completed":
        return None

    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else None


def save_checkpoint(sma_divergence_threshold: float, summary_row: dict) -> None:
    path = checkpoint_path(sma_divergence_threshold)
    payload = {
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary_row,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_ticker_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
    if "Date" not in df.columns:
        raise ValueError("Date列がありません")

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"必須列不足: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).copy()
    if df.empty:
        raise ValueError("Dateの有効行がありません")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < MIN_ROWS:
        raise ValueError(f"行数不足: {len(df)} < {MIN_ROWS}")

    df = df.sort_values("Date")
    df = df.set_index("Date")
    df.index = df.index.tz_localize(None)

    df["Adj Close"] = df["Close"]
    return df


def extract_trades(result_df: pd.DataFrame, ticker: str) -> List[dict]:
    if result_df is None or result_df.empty:
        return []
    if "Entry_Signal" not in result_df.columns:
        return []
    if "Exit_Signal" not in result_df.columns:
        return []
    if "Trade_ID" not in result_df.columns:
        return []
    if "Profit_Loss" not in result_df.columns:
        return []

    entries = result_df[result_df["Entry_Signal"] == 1][["Trade_ID"]].copy()
    exits = result_df[result_df["Exit_Signal"] == -1][["Trade_ID", "Profit_Loss"]].copy()

    if entries.empty or exits.empty:
        return []

    entries["entry_date"] = entries.index
    exits["exit_date"] = exits.index

    merged = exits.merge(entries, on="Trade_ID", how="left")
    trades: List[dict] = []
    for _, row in merged.iterrows():
        trades.append(
            {
                "ticker": ticker,
                "entry_date": pd.to_datetime(row.get("entry_date"), errors="coerce"),
                "exit_date": pd.to_datetime(row.get("exit_date"), errors="coerce"),
                "profit_loss_per_share": float(row.get("Profit_Loss", 0.0)),
            }
        )
    return trades


def run_ticker_backtest(
    ticker: str,
    stock_df: pd.DataFrame,
    sma_divergence_threshold: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> List[dict]:
    strategy = GCStrategy(
        data=stock_df.copy(),
        params={
            "short_window": SHORT_WINDOW,
            "long_window": LONG_WINDOW,
            "stop_loss": STOP_LOSS,
            "trailing_stop_pct": TRAILING_STOP_PCT,
            "trend_strength_percentile": TREND_STRENGTH_PERCENTILE,
            "use_entry_filter": USE_ENTRY_FILTER,
            "trend_strength_enabled": TREND_STRENGTH_ENABLED,
            "sma_divergence_enabled": SMA_DIVERGENCE_ENABLED,
            "sma_divergence_threshold": sma_divergence_threshold,
            "max_positions": MAX_POSITIONS,
        },
        price_column="Adj Close",
        ticker=f"{ticker}.T",
    )
    result_df = strategy.backtest(
        trading_start_date=start_date,
        trading_end_date=end_date,
    )
    return extract_trades(result_df, ticker)


def filter_trades_by_exit_period(
    trades: List[dict],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> List[dict]:
    filtered: List[dict] = []
    for trade in trades:
        exit_date = pd.to_datetime(trade.get("exit_date"), errors="coerce")
        if pd.isna(exit_date):
            continue
        if start_date <= exit_date <= end_date:
            filtered.append(trade)
    return filtered


def _profit_factor(profit_loss_series: pd.Series) -> float:
    gross_profit = float(profit_loss_series[profit_loss_series > 0].sum())
    gross_loss = float(profit_loss_series[profit_loss_series < 0].sum())
    if gross_loss < 0:
        return gross_profit / abs(gross_loss)
    if gross_profit > 0:
        return float("inf")
    return float("nan")


def aggregate_metrics(
    trades: List[dict],
    yearly_pf_start: int = 2016,
    yearly_pf_end: int = 2025,
    include_yearly_pf: bool = True,
) -> Dict[str, float]:
    base = {
        "pf": float("nan"),
        "win_rate_pct": 0.0,
        "trades": 0,
        "net_profit_man": 0.0,
        "max_dd_man": 0.0,
    }
    for year in range(yearly_pf_start, yearly_pf_end + 1):
        base[f"pf_{year}"] = float("nan")

    if not trades:
        return base

    df = pd.DataFrame(trades)
    if df.empty:
        return base

    df["profit_loss_per_share"] = pd.to_numeric(df["profit_loss_per_share"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df = df.dropna(subset=["profit_loss_per_share", "exit_date"]).copy()
    if df.empty:
        return base

    df = df.sort_values("exit_date")
    pl_per_share = df["profit_loss_per_share"].astype(float)
    pl_yen = pl_per_share * SHARES_PER_TRADE

    trades_count = int(len(df))
    wins = int((pl_per_share > 0).sum())
    win_rate = (wins / trades_count * 100.0) if trades_count > 0 else 0.0

    pf = _profit_factor(pl_per_share)
    net_profit_man = float(pl_yen.sum()) / 10000.0

    cumulative = pl_yen.cumsum()
    running_peak = cumulative.cummax()
    drawdown = running_peak - cumulative
    max_dd_man = float(drawdown.max()) / 10000.0 if len(drawdown) else 0.0

    result = {
        "pf": pf,
        "win_rate_pct": win_rate,
        "trades": trades_count,
        "net_profit_man": net_profit_man,
        "max_dd_man": max_dd_man,
    }

    if include_yearly_pf:
        for year in range(yearly_pf_start, yearly_pf_end + 1):
            yearly = df[df["exit_date"].dt.year == year]
            if yearly.empty:
                result[f"pf_{year}"] = float("nan")
                continue
            result[f"pf_{year}"] = _profit_factor(yearly["profit_loss_per_share"].astype(float))

    return result


def build_summary_row(
    sma_divergence_threshold: float,
    full_trades: List[dict],
    is_trades: List[dict],
    oos_trades: List[dict],
) -> dict:
    full_metrics = aggregate_metrics(full_trades, 2016, 2025, include_yearly_pf=True)
    is_metrics = aggregate_metrics(is_trades, 2016, 2022, include_yearly_pf=False)
    oos_metrics = aggregate_metrics(oos_trades, 2023, 2025, include_yearly_pf=False)

    is_pf = is_metrics["pf"]
    oos_pf = oos_metrics["pf"]
    oos_is_ratio = float("nan")
    if pd.notna(is_pf) and pd.notna(oos_pf) and is_pf not in (0.0, float("inf")):
        oos_is_ratio = oos_pf / is_pf

    row = {
        "sma_divergence_threshold": sma_divergence_threshold,
        "pf": full_metrics["pf"],
        "win_rate_pct": full_metrics["win_rate_pct"],
        "trades": int(full_metrics["trades"]),
        "net_profit_man": full_metrics["net_profit_man"],
        "max_dd_man": full_metrics["max_dd_man"],
        "is_pf": is_pf,
        "oos_pf": oos_pf,
        "oos_is_ratio": oos_is_ratio,
        "oos_net_profit_man": oos_metrics["net_profit_man"],
        "max_positions": MAX_POSITIONS,
    }

    for year in range(2016, 2026):
        row[f"pf_{year}"] = full_metrics.get(f"pf_{year}", float("nan"))

    return row


def save_result_csv(rows: List[dict]) -> Path:
    today = datetime.now().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"gc_sma_divergence_grid_result_{today}.csv"

    columns = [
        "sma_divergence_threshold",
        "pf",
        "win_rate_pct",
        "trades",
        "net_profit_man",
        "max_dd_man",
    ] + [f"pf_{year}" for year in range(2016, 2026)] + [
        "is_pf",
        "oos_pf",
        "oos_is_ratio",
        "oos_net_profit_man",
        "max_positions",
    ]

    csv_rows: List[dict] = []
    for row in rows:
        csv_row: Dict[str, object] = {}
        for column in columns:
            value = row.get(column)
            if isinstance(value, float) and value == float("inf"):
                value = 999.99
            csv_row[column] = value
        csv_rows.append(csv_row)

    out_df = pd.DataFrame(csv_rows, columns=columns)
    out_df.to_csv(str(out_path), index=False, encoding="utf-8-sig")
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    suppress_strategy_logging()

    ticker_files = sorted(JQUANTS_DIR.glob("*.csv"))
    if not ticker_files:
        print(f"[ERROR] データが見つかりません: {JQUANTS_DIR}")
        return

    total_tickers = len(ticker_files)
    total_candidates = len(CANDIDATES)
    summary_rows: List[dict] = []
    error_log: List[str] = []

    for candidate_idx, candidate in enumerate(CANDIDATES, start=1):
        cp = load_checkpoint(candidate)
        if cp is not None:
            print(f"[{candidate_idx}/{total_candidates}] threshold={candidate:.1f} チェックポイントあり -> スキップ")
            summary_rows.append(cp)
            continue

        print(f"[{candidate_idx}/{total_candidates}] threshold={candidate:.1f} 処理開始")

        full_trades: List[dict] = []
        is_trades: List[dict] = []
        oos_trades: List[dict] = []

        for ticker_idx, csv_path in enumerate(ticker_files, start=1):
            if ticker_idx == 1 or ticker_idx % 50 == 0 or ticker_idx == total_tickers:
                print(
                    f"[{candidate_idx}/{total_candidates}] threshold={candidate:.1f} 処理中... "
                    f"銘柄 {ticker_idx}/{total_tickers}"
                )

            ticker = csv_path.stem
            try:
                stock_df = load_ticker_data(csv_path)
            except Exception as exc:
                error_log.append(f"threshold={candidate:.1f} {ticker}: 読込失敗 - {exc}")
                continue

            try:
                ticker_trades = run_ticker_backtest(
                    ticker=ticker,
                    stock_df=stock_df,
                    sma_divergence_threshold=candidate,
                    start_date=FULL_START,
                    end_date=FULL_END,
                )

                full_trades.extend(ticker_trades)
                is_trades.extend(filter_trades_by_exit_period(ticker_trades, IS_START, IS_END))
                oos_trades.extend(filter_trades_by_exit_period(ticker_trades, OOS_START, OOS_END))
            except Exception as exc:
                error_log.append(f"threshold={candidate:.1f} {ticker}: バックテスト失敗 - {exc}")
            finally:
                del stock_df
                _gc.collect()

        summary_row = build_summary_row(candidate, full_trades, is_trades, oos_trades)
        save_checkpoint(candidate, summary_row)
        summary_rows.append(summary_row)

        pf_value = summary_row["pf"]
        pf_text = "inf" if pf_value == float("inf") else ("NaN" if pd.isna(pf_value) else f"{pf_value:.2f}")
        print(
            f"[{candidate_idx}/{total_candidates}] threshold={candidate:.1f} 完了: "
            f"PF={pf_text}, 取引数={summary_row['trades']}, 純利益={summary_row['net_profit_man']:.2f}万円"
        )

    output_path = save_result_csv(summary_rows)
    print("グリッドサーチ完了。結果を output/ に保存しました。")
    print(f"出力CSV: {output_path}")

    if error_log:
        print("\n=== スキップ銘柄エラーログ ===")
        for message in error_log:
            print(message)


if __name__ == "__main__":
    main()