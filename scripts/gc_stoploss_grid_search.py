"""
gc_stoploss_grid_search - Grid search for GCStrategy stop_loss in DSSMS.

This script runs DSSMS integrated backtests for multiple GCStrategy stop_loss values
without modifying existing project files. It injects GCStrategy params at runtime via
monkey patch and exports a single summary CSV.

Main features:
- Runs grid search for stop_loss candidates including None (disabled stop-loss)
- Forces GCStrategy selection and injects params dynamically
- Collects PF, win rate, total trades, net profit, max drawdown, yearly PnL

Integrated components:
- src.dssms.dssms_integrated_main.DSSMSIntegratedBacktester: Backtest engine
- strategies.gc_strategy_signal.GCStrategy: Target strategy for stop_loss injection

Safety notes:
- Existing files are not modified
- DSSMS bulk output generation is disabled at runtime to keep output minimal
- None stop_loss is mapped to a very large value (9999.0)

Author: Backtest Project Team
Created: 2026-03-24
Last Modified: 2026-03-24
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
from strategies.gc_strategy_signal import GCStrategy


START_DATE = "2016-09-01"
END_DATE = "2025-12-31"
STOP_LOSS_CANDIDATES: List[Optional[float]] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, None]
NO_STOP_LOSS_SENTINEL = 9999.0
MAX_POSITIONS = 3


def build_config() -> Dict[str, Any]:
    return {
        "initial_capital": 1000000,
        "symbol_switch": {
            "switch_management": {
                "min_holding_days": 5,
                "max_switches_per_month": 5,
                "switch_cost_rate": 0.001,
            }
        },
    }


def format_stop_loss_label(stop_loss: Optional[float]) -> str:
    if stop_loss is None:
        return "None"
    return f"{stop_loss:.2%}"


def compute_max_drawdown_pct(daily_results: List[Dict[str, Any]]) -> float:
    portfolio_values: List[float] = []
    for daily in daily_results:
        value = daily.get("portfolio_value_end")
        if value is None:
            value = daily.get("portfolio_value")
        if value is None:
            continue
        try:
            portfolio_values.append(float(value))
        except (TypeError, ValueError):
            continue

    if not portfolio_values:
        return 0.0

    peak = portfolio_values[0]
    max_dd = 0.0
    for value in portfolio_values:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100.0


def extract_gc_trades(backtester: DSSMSIntegratedBacktester, results: Dict[str, Any]) -> pd.DataFrame:
    execution_details: List[Dict[str, Any]] = []
    for daily in results.get("daily_results", []):
        execution_details.extend(daily.get("execution_details", []))

    trades = backtester._convert_execution_details_to_trades(execution_details)
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol",
                "entry_date",
                "entry_price",
                "exit_date",
                "exit_price",
                "shares",
                "pnl",
                "return_pct",
                "holding_period_days",
                "strategy_name",
                "position_value",
                "is_forced_exit",
            ]
        )

    trades_df = pd.DataFrame(trades)
    if "strategy_name" in trades_df.columns:
        trades_df = trades_df[trades_df["strategy_name"] == "GCStrategy"].copy()

    for col in ["pnl", "return_pct", "entry_price", "exit_price", "position_value"]:
        if col in trades_df.columns:
            trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")

    if "exit_date" in trades_df.columns:
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"], errors="coerce")

    return trades_df


def calculate_metrics(trades_df: pd.DataFrame, daily_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_trades = int(len(trades_df))

    if total_trades == 0:
        yearly = {f"year_{year}": 0.0 for year in range(2016, 2026)}
        return {
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "net_profit_jpy": 0.0,
            "max_drawdown_pct": compute_max_drawdown_pct(daily_results),
            **yearly,
        }

    pnl = trades_df["pnl"].fillna(0.0)
    wins = int((pnl > 0).sum())
    win_rate = (wins / total_trades) * 100.0

    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(pnl[pnl < 0].sum())

    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    net_profit = float(pnl.sum())
    max_drawdown_pct = compute_max_drawdown_pct(daily_results)

    yearly = {f"year_{year}": 0.0 for year in range(2016, 2026)}
    if "exit_date" in trades_df.columns:
        valid = trades_df.dropna(subset=["exit_date"]).copy()
        if not valid.empty:
            valid["year"] = valid["exit_date"].dt.year
            yearly_pnl = valid.groupby("year")["pnl"].sum()
            for year in range(2016, 2026):
                yearly[f"year_{year}"] = float(yearly_pnl.get(year, 0.0))

    return {
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "net_profit_jpy": net_profit,
        "max_drawdown_pct": max_drawdown_pct,
        **yearly,
    }


def run_single_backtest(stop_loss: Optional[float]) -> Dict[str, Any]:
    effective_stop_loss = NO_STOP_LOSS_SENTINEL if stop_loss is None else stop_loss

    backtester = DSSMSIntegratedBacktester(config=build_config(), force_strategy="GCStrategy")
    backtester.max_positions = MAX_POSITIONS

    # Suppress bulk DSSMS output files; keep only grid-search summary CSV.
    backtester._generate_outputs = lambda _final_results: None

    original_create_strategy_instance = backtester._create_strategy_instance

    def patched_create_strategy_instance(strategy_name: str, data: pd.DataFrame):
        if strategy_name == "GCStrategy":
            return GCStrategy(data, params={"stop_loss": effective_stop_loss})
        return original_create_strategy_instance(strategy_name, data)

    backtester._create_strategy_instance = patched_create_strategy_instance

    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

    results = backtester.run_dynamic_backtest(start_date, end_date, target_symbols=None)
    if "error" in results:
        raise RuntimeError(str(results["error"]))

    trades_df = extract_gc_trades(backtester, results)
    metrics = calculate_metrics(trades_df, results.get("daily_results", []))
    return metrics


def main() -> None:
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"gc_stoploss_grid_search_{today_str}.csv"

    records: List[Dict[str, Any]] = []
    total = len(STOP_LOSS_CANDIDATES)

    print("GCStrategy stop_loss grid search started")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Candidates: {STOP_LOSS_CANDIDATES}")

    for idx, stop_loss in enumerate(STOP_LOSS_CANDIDATES, start=1):
        label = format_stop_loss_label(stop_loss)
        print(f"[{idx}/{total}] stop_loss={label} running...")

        row: Dict[str, Any] = {
            "stop_loss": "None" if stop_loss is None else stop_loss,
            "stop_loss_effective": NO_STOP_LOSS_SENTINEL if stop_loss is None else stop_loss,
            "max_positions": MAX_POSITIONS,
            "auto_adjust": False,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "error": "",
        }

        try:
            metrics = run_single_backtest(stop_loss)
            row.update(metrics)
            pf = metrics["profit_factor"]
            pf_display = "inf" if pf == float("inf") else f"{pf:.4f}"
            print(
                "  done: "
                f"PF={pf_display}, "
                f"WinRate={metrics['win_rate']:.2f}%, "
                f"Trades={metrics['total_trades']}, "
                f"NetProfit={metrics['net_profit_jpy']:.0f} JPY, "
                f"MaxDD={metrics['max_drawdown_pct']:.2f}%"
            )
        except Exception as exc:
            row.update(
                {
                    "profit_factor": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "net_profit_jpy": 0.0,
                    "max_drawdown_pct": 0.0,
                    **{f"year_{year}": 0.0 for year in range(2016, 2026)},
                }
            )
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  error: {row['error']}")

        records.append(row)

    result_df = pd.DataFrame(records)

    preferred_cols = [
        "stop_loss",
        "stop_loss_effective",
        "profit_factor",
        "win_rate",
        "total_trades",
        "net_profit_jpy",
        "max_drawdown_pct",
    ] + [f"year_{year}" for year in range(2016, 2026)] + [
        "max_positions",
        "auto_adjust",
        "start_date",
        "end_date",
        "error",
    ]

    cols = [c for c in preferred_cols if c in result_df.columns]
    result_df = result_df[cols]
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Grid search completed: {output_path}")


if __name__ == "__main__":
    main()
