"""
gc_trend_strength_grid_search - Grid search and walk-forward for GCStrategy trend strength.

This script runs GCStrategy across all J-Quants cache CSV files and optimizes
trend_strength_percentile with fixed core parameters. It records full-period
metrics, yearly PF, and walk-forward IS/OOS validation metrics.

Main features:
- Grid search for trend_strength_percentile candidates
- Checkpoint resume by candidate
- Full-period aggregation (2016-01-01 to 2025-12-31)
- Walk-forward aggregation (IS: 2016-2022, OOS: 2023-2025)
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
import json
import logging
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
CHECKPOINT_DIR: Path = OUTPUT_DIR / "gc_trend_strength_checkpoints"


# ---------------------------------------------------------------------------
# Fixed parameters (DO NOT CHANGE)
# ---------------------------------------------------------------------------
SHORT_WINDOW: int = 5
LONG_WINDOW: int = 75
STOP_LOSS: float = 0.03
TRAILING_STOP_PCT: Optional[float] = None
MAX_POSITIONS: int = 3  # metadata only

# GCStrategy は trailing_stop_pct=None を内部処理できないため、
# 事実上無効となる sentinel 値へ変換して渡す。
TRAILING_STOP_NONE_SENTINEL: float = 0.999

SHARES_PER_TRADE: int = 100
MIN_ROWS: int = 150


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
CANDIDATES: List[int] = [30, 40, 50, 55, 60, 67, 70, 75, 80, 85, 90]


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------
FULL_START = pd.Timestamp("2016-01-01")
FULL_END = pd.Timestamp("2025-12-31")

IS_START = pd.Timestamp("2016-01-01")
IS_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")
OOS_END = pd.Timestamp("2025-12-31")


# 採用基準:
# 1. プロフィットヒル形成（隣接候補より明確にPFが高い山型）
# 2. ウォークフォワード OOS/IS比 >= 1.0（過学習なし）
# 3. 取引数 100件以上（統計的有意性）


def checkpoint_path(trend_strength_percentile: int) -> Path:
    return CHECKPOINT_DIR / f"checkpoint_{trend_strength_percentile}.json"


def trailing_stop_param_value(value: Optional[float]) -> float:
    return TRAILING_STOP_NONE_SENTINEL if value is None else value


def suppress_strategy_logging() -> None:
    logging.disable(logging.WARNING)


def load_checkpoint(trend_strength_percentile: int) -> Optional[dict]:
    path = checkpoint_path(trend_strength_percentile)
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
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else None


def save_checkpoint(trend_strength_percentile: int, summary_row: dict) -> None:
    path = checkpoint_path(trend_strength_percentile)
    payload = {
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary_row,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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
    # tz-aware/naive比較の不整合防止
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
    trend_strength_percentile: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> List[dict]:
    trailing_stop_pct = trailing_stop_param_value(TRAILING_STOP_PCT)
    strategy = GCStrategy(
        data=stock_df.copy(),
        params={
            "short_window": SHORT_WINDOW,
            "long_window": LONG_WINDOW,
            "stop_loss": STOP_LOSS,
            "trailing_stop_pct": trailing_stop_pct,
            "trend_strength_percentile": trend_strength_percentile,
        },
        price_column="Adj Close",
        ticker=f"{ticker}.T",
    )
    result_df = strategy.backtest(
        trading_start_date=start_date,
        trading_end_date=end_date,
    )
    return extract_trades(result_df, ticker)


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
    for y in range(yearly_pf_start, yearly_pf_end + 1):
        base[f"pf_{y}"] = float("nan")

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
        for y in range(yearly_pf_start, yearly_pf_end + 1):
            yearly = df[df["exit_date"].dt.year == y]
            if yearly.empty:
                result[f"pf_{y}"] = float("nan")
                continue
            result[f"pf_{y}"] = _profit_factor(yearly["profit_loss_per_share"].astype(float))

    return result


def build_summary_row(
    trend_strength_percentile: int,
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
        "trend_strength_percentile": trend_strength_percentile,
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

    for y in range(2016, 2026):
        row[f"pf_{y}"] = full_metrics.get(f"pf_{y}", float("nan"))

    return row


def save_result_csv(rows: List[dict]) -> Path:
    today = datetime.now().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"gc_trend_strength_grid_result_{today}.csv"

    columns = [
        "trend_strength_percentile",
        "pf",
        "win_rate_pct",
        "trades",
        "net_profit_man",
        "max_dd_man",
    ] + [f"pf_{y}" for y in range(2016, 2026)] + [
        "is_pf",
        "oos_pf",
        "oos_is_ratio",
        "oos_net_profit_man",
        "max_positions",
    ]

    csv_rows: List[dict] = []
    for row in rows:
        csv_row: Dict[str, object] = {}
        for c in columns:
            v = row.get(c)
            if isinstance(v, float) and v == float("inf"):
                v = 999.99
            csv_row[c] = v
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
    summary_rows: List[dict] = []
    error_log: List[str] = []

    for candidate in CANDIDATES:
        cp = load_checkpoint(candidate)
        if cp is not None:
            print(f"候補{candidate}: チェックポイントあり → スキップ")
            summary_rows.append(cp)
            continue

        print(f"=== trend_strength_percentile={candidate} 開始 ===")

        full_trades: List[dict] = []
        is_trades: List[dict] = []
        oos_trades: List[dict] = []

        for idx, csv_path in enumerate(ticker_files, start=1):
            if idx % 100 == 0:
                print(f"{idx}/{total_tickers} 銘柄処理中...")

            ticker = csv_path.stem
            try:
                stock_df = load_ticker_data(csv_path)
            except Exception as exc:
                error_log.append(f"{ticker}: 読込失敗 - {exc}")
                continue

            try:
                full_trades.extend(
                    run_ticker_backtest(
                        ticker=ticker,
                        stock_df=stock_df,
                        trend_strength_percentile=candidate,
                        start_date=FULL_START,
                        end_date=FULL_END,
                    )
                )

                is_trades.extend(
                    run_ticker_backtest(
                        ticker=ticker,
                        stock_df=stock_df,
                        trend_strength_percentile=candidate,
                        start_date=IS_START,
                        end_date=IS_END,
                    )
                )

                oos_trades.extend(
                    run_ticker_backtest(
                        ticker=ticker,
                        stock_df=stock_df,
                        trend_strength_percentile=candidate,
                        start_date=OOS_START,
                        end_date=OOS_END,
                    )
                )
            except Exception as exc:
                error_log.append(f"{ticker}: バックテスト失敗 - {exc}")
            finally:
                del stock_df
                _gc.collect()

        summary_row = build_summary_row(candidate, full_trades, is_trades, oos_trades)
        save_checkpoint(candidate, summary_row)
        summary_rows.append(summary_row)

        pf_value = summary_row["pf"]
        pf_text = "inf" if pf_value == float("inf") else ("NaN" if pd.isna(pf_value) else f"{pf_value:.2f}")
        print(
            f"候補{candidate} 完了: PF={pf_text}, 取引数={summary_row['trades']}, "
            f"純利益={summary_row['net_profit_man']:.2f}万円"
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
