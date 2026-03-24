"""
gc_stoploss_grid_search_v2 - GCStrategy stop_loss グリッドサーチ（銘柄別直接バックテスト方式）

J-Quantsキャッシュ（data/jquants_cache/ 内の全 CSV）に対して GCStrategy を
直接実行し、stop_loss 候補ごとの PF・勝率・取引数・純利益を集計する。
DSSMSIntegratedBacktester を経由しないため、銘柄単位での細かい制御が可能。

主な機能:
- 8 stop_loss 候補 x 224 銘柄のグリッドサーチ
- チェックポイント CSV による途中再開（候補単位 + 銘柄単位）
- 銘柄ごとのメモリ解放（del + gc.collect）
- コンソール進捗表示 + 最終 CSV 出力（年次 PnL 含む）

統合コンポーネント:
- strategies.gc_strategy_signal.GCStrategy: 直接バックテスト実行
- data/jquants_cache/{code}.csv: 銘柄データ（列: Date, Open, High, Low, Close, Volume）

セーフティ機能/注意事項:
- 個別銘柄エラーはスキップして error_log.txt に記録（処理継続）
- データ 150 行未満はウォームアップ不足としてスキップ
- GCStrategy / BaseStrategy の INFO ログはグリッドサーチ中に抑制
- J-Quants データには Adj Close がないため Close を Adj Close として代用
- None stop_loss は params={"stop_loss": 99.0} で実質無効化

Author: Backtest Project Team
Created: 2026-03-24
Last Modified: 2026-03-24
"""

from __future__ import annotations

import gc as _gc
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# プロジェクトルート設定
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.gc_strategy_signal import GCStrategy  # noqa: E402

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
JQUANTS_DIR: Path = PROJECT_ROOT / "data" / "jquants_cache"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "output" / "gc_grid_checkpoints"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# バックテスト期間
TRADING_START = pd.Timestamp("2016-09-01")
TRADING_END = pd.Timestamp("2025-12-31")

# stop_loss 候補（None = 実質無効: 99.0 を渡す）
STOP_LOSS_CANDIDATES: List[Optional[float]] = [
    0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, None
]

# ウォームアップ最小行数
MIN_ROWS: int = 150

# 1 トレートあたり株数（日本株標準単元株）
SHARES_PER_TRADE: int = 100

# チェックポイント CSV の列定義
CHKPT_COLS: List[str] = ["ticker", "entry_date", "exit_date", "profit_loss_per_share"]

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def sl_label(sl: Optional[float]) -> str:
    """コンソール表示用ラベル。例: 0.03 -> '-3%', None -> 'None'"""
    if sl is None:
        return "None"
    return f"-{sl * 100:.0f}%"


def sl_effective(sl: Optional[float]) -> float:
    """None を実質無効値 99.0 に変換"""
    return 99.0 if sl is None else sl


def sl_checkpoint_path(sl: Optional[float]) -> Path:
    """チェックポイント CSV のパスを返す"""
    name = "checkpoint_slNone.csv" if sl is None else f"checkpoint_sl{sl}.csv"
    return CHECKPOINT_DIR / name


def setup_main_logger() -> logging.Logger:
    """グリッドサーチの進捗表示用ロガー"""
    logger = logging.getLogger("gc_grid_v2")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def setup_error_logger(error_log_path: Path) -> logging.Logger:
    """銘柄エラーをファイルに記録するロガー"""
    err_logger = logging.getLogger("gc_grid_v2_errors")
    if not err_logger.handlers:
        err_logger.setLevel(logging.ERROR)
        fh = logging.FileHandler(str(error_log_path), encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        err_logger.addHandler(fh)
    return err_logger


def suppress_strategy_logging() -> None:
    """GCStrategy / BaseStrategy の INFO ログをグリッドサーチ中に抑制"""
    for name in ("GCStrategy", "BaseStrategy"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------

def load_ticker_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    J-Quantsキャッシュ CSV を読み込み GCStrategy 用 DataFrame を返す。

    - 列: Date, Open, High, Low, Close, Volume（Adj Close は Close で代用）
    - インデックスを tz-aware な DatetimeIndex に設定
    - MIN_ROWS 未満の場合は None を返す
    """
    try:
        # 先頭列（行番号）を index_col=0 で読み飛ばし
        df = pd.read_csv(str(csv_path), index_col=0)

        # Date 列を tz-aware DatetimeIndex に変換
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df.index = pd.DatetimeIndex(df["Date"])
        df = df.drop(columns=["Date"], errors="ignore")
        df = df.sort_index()

        # 数値型変換（型不整合対策）
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 必須列チェック + NaN 除去
        required = [c for c in ("Open", "Close") if c in df.columns]
        if required:
            df = df.dropna(subset=required)

        # Adj Close は Close で代用
        # (base_strategy.py の最終日フォールバック `result['Adj Close']` に対応)
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        if len(df) < MIN_ROWS:
            return None

        return df

    except Exception:
        return None


# ---------------------------------------------------------------------------
# チェックポイント管理
# ---------------------------------------------------------------------------

def load_checkpoint(sl: Optional[float]) -> pd.DataFrame:
    """
    チェックポイント CSV を読み込む。
    存在しない・壊れている場合は空 DataFrame を返す。
    """
    path = sl_checkpoint_path(sl)
    if not path.exists():
        return pd.DataFrame(columns=CHKPT_COLS)
    try:
        df = pd.read_csv(str(path))
        # 列が揃っているか確認
        for col in CHKPT_COLS:
            if col not in df.columns:
                return pd.DataFrame(columns=CHKPT_COLS)
        return df
    except Exception:
        return pd.DataFrame(columns=CHKPT_COLS)


def get_processed_tickers(chkpt_df: pd.DataFrame) -> set:
    """チェックポイントから処理済み銘柄コードのセットを返す"""
    if chkpt_df.empty:
        return set()
    return set(chkpt_df["ticker"].unique())


def append_to_checkpoint(sl: Optional[float], new_rows: List[dict],
                         chkpt_df: pd.DataFrame) -> pd.DataFrame:
    """
    新しい行をチェックポイント DataFrame に追加して CSV に保存する。
    更新後の DataFrame を返す。
    """
    if not new_rows:
        return chkpt_df

    new_df = pd.DataFrame(new_rows, columns=CHKPT_COLS)
    updated = pd.concat([chkpt_df, new_df], ignore_index=True)
    updated.to_csv(str(sl_checkpoint_path(sl)), index=False)
    return updated


# ---------------------------------------------------------------------------
# バックテスト実行
# ---------------------------------------------------------------------------

def extract_trades(result_df: pd.DataFrame, ticker: str) -> List[dict]:
    """
    backtest() 結果 DataFrame からトレード一覧を抽出する。

    - Entry_Signal == 1 の行の Trade_ID と entry_date を取得
    - Exit_Signal == -1 の行の Profit_Loss と exit_date を取得
    - Trade_ID で結合してトレードごとの損益を確定

    Returns:
        list of dicts with keys: ticker, entry_date, exit_date, profit_loss_per_share
    """
    entries = result_df[result_df["Entry_Signal"] == 1].copy()
    exits = result_df[result_df["Exit_Signal"] == -1].copy()

    if entries.empty or exits.empty:
        return []

    # インデックスを文字列形式の日付として保持
    entries = entries[["Trade_ID"]].copy()
    entries["entry_date"] = entries.index.astype(str)

    exits = exits[["Trade_ID", "Profit_Loss"]].copy()
    exits["exit_date"] = exits.index.astype(str)

    merged = exits.merge(entries, on="Trade_ID", how="left")

    rows: List[dict] = []
    for _, row in merged.iterrows():
        rows.append({
            "ticker": ticker,
            "entry_date": str(row.get("entry_date", "")),
            "exit_date": str(row.get("exit_date", "")),
            "profit_loss_per_share": float(row.get("Profit_Loss", 0.0)),
        })
    return rows


def run_single_ticker(
    ticker: str,
    stock_df: pd.DataFrame,
    sl: Optional[float],
    logger: logging.Logger,
    err_logger: logging.Logger,
) -> Optional[List[dict]]:
    """
    1 銘柄のバックテストを実行してトレード行リストを返す。

    Returns:
        list of trade dicts（成功・0 件含む）、エラー時は None
    """
    try:
        effective_sl = sl_effective(sl)
        strategy = GCStrategy(
            data=stock_df.copy(),
            params={"stop_loss": effective_sl},
            price_column="Adj Close",
            ticker=ticker,
        )
        result = strategy.backtest(
            trading_start_date=TRADING_START,
            trading_end_date=TRADING_END,
        )
        trades = extract_trades(result, ticker)
        logger.info(f"    [GCStrategy] {ticker}.T: {len(trades)}件の取引")
        return trades
    except Exception as exc:
        err_logger.error(f"[ERROR] {ticker}: {exc}\n{traceback.format_exc()}")
        return None


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------

def aggregate_metrics(chkpt_df: pd.DataFrame) -> Dict:
    """
    チェックポイント DataFrame から指標を集計する。

    センチネル行（entry_date が空文字）はスキップ銘柄として除外する。

    Returns dict with keys:
        pf, win_rate, total_trades, net_profit, max_drawdown,
        y2016, y2017, ..., y2025
    """
    yearly_keys = [f"y{y}" for y in range(2016, 2026)]
    empty = {"pf": 0.0, "win_rate": 0.0, "total_trades": 0,
             "net_profit": 0.0, "max_drawdown": 0.0,
             **{k: 0.0 for k in yearly_keys}}

    if chkpt_df is None or chkpt_df.empty:
        return empty

    # センチネル行（スキップ銘柄）を除外して実トレードのみ対象
    trades_df = chkpt_df[chkpt_df["entry_date"].astype(str).str.strip() != ""].copy()

    if trades_df.empty:
        return empty

    pl = trades_df["profit_loss_per_share"].astype(float)
    total_trades = int(len(pl))
    wins = int((pl > 0).sum())
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = float(pl[pl > 0].sum())
    gross_loss = float(pl[pl < 0].sum())

    if gross_loss < 0:
        pf = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0

    net_profit = float(pl.sum()) * SHARES_PER_TRADE

    # 最大ドローダウン（累積損益曲線から計算）
    max_drawdown = 0.0
    try:
        df_sorted = trades_df.copy()
        df_sorted["exit_date"] = pd.to_datetime(
            df_sorted["exit_date"], errors="coerce", utc=True
        )
        df_sorted = df_sorted.dropna(subset=["exit_date"]).sort_values("exit_date")
        if not df_sorted.empty:
            cumul = (df_sorted["profit_loss_per_share"].astype(float) * SHARES_PER_TRADE
                     ).cumsum().values
            peak = cumul[0]
            max_dd = 0.0
            for v in cumul:
                if v > peak:
                    peak = v
                dd = peak - v
                if dd > max_dd:
                    max_dd = dd
            max_drawdown = float(max_dd)
    except Exception:
        pass

    # 年次 PnL
    yearly: Dict[str, float] = {k: 0.0 for k in yearly_keys}
    try:
        df_y = trades_df.copy()
        df_y["exit_date"] = pd.to_datetime(
            df_y["exit_date"], errors="coerce", utc=True
        )
        df_y = df_y.dropna(subset=["exit_date"])
        df_y["year"] = df_y["exit_date"].dt.year
        by_year = df_y.groupby("year")["profit_loss_per_share"].sum() * SHARES_PER_TRADE
        for y in range(2016, 2026):
            if y in by_year.index:
                yearly[f"y{y}"] = float(by_year[y])
    except Exception:
        pass

    return {
        "pf": pf,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "net_profit": net_profit,
        "max_drawdown": max_drawdown,
        **yearly,
    }


# ---------------------------------------------------------------------------
# 結果表示・保存
# ---------------------------------------------------------------------------

def print_summary(summary_rows: List[dict], logger: logging.Logger) -> None:
    """コンソールにサマリー表を出力する"""
    header = (
        f"{'stop_loss':>10} | {'PF':>6} | {'Win%':>6} | "
        f"{'Trades':>7} | {'Net Profit(JPY)':>16}"
    )
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for row in summary_rows:
        pf_str = f"{row['pf']:.2f}" if row['pf'] != float("inf") else "  inf"
        logger.info(
            f"{row['stop_loss']:>10} | {pf_str:>6} | {row['win_rate']:>5.1f}% | "
            f"{row['total_trades']:>7,} | {row['net_profit']:>+16,.0f}"
        )
    logger.info(sep)


def save_final_csv(summary_rows: List[dict], logger: logging.Logger) -> None:
    """最終集計 CSV を output/ に保存する"""
    today = datetime.now().strftime("%Y%m%d")
    output_path = OUTPUT_DIR / f"gc_stoploss_grid_result_{today}.csv"

    year_cols = [f"y{y}" for y in range(2016, 2026)]
    rows_for_csv = []
    for row in summary_rows:
        r = {
            "stop_loss":    row["stop_loss"],
            "pf":           row["pf"] if row["pf"] != float("inf") else 999999.0,
            "win_rate":     row["win_rate"],
            "total_trades": row["total_trades"],
            "net_profit":   row["net_profit"],
            "max_drawdown": row["max_drawdown"],
        }
        for yc in year_cols:
            r[yc] = row.get(yc, 0.0)
        rows_for_csv.append(r)

    result_df = pd.DataFrame(rows_for_csv)
    result_df.to_csv(str(output_path), index=False, encoding="utf-8-sig")
    logger.info(f"最終結果を保存: {output_path}")


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def main() -> None:
    logger = setup_main_logger()
    suppress_strategy_logging()

    # 出力フォルダ準備
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    error_log_path = CHECKPOINT_DIR / "error_log.txt"
    err_logger = setup_error_logger(error_log_path)

    # 銘柄ファイル一覧（ファイル名 = 銘柄コード, 例: 7203.csv）
    ticker_files = sorted(JQUANTS_DIR.glob("*.csv"))
    if not ticker_files:
        logger.info(f"[ERROR] J-Quantsキャッシュが見つかりません: {JQUANTS_DIR}")
        return

    logger.info(f"J-Quantsキャッシュ: {len(ticker_files)} 銘柄")
    logger.info(f"stop_loss 候補: {[sl_label(sl) for sl in STOP_LOSS_CANDIDATES]}")
    logger.info(f"取引期間: {TRADING_START.date()} ~ {TRADING_END.date()}")

    total_candidates = len(STOP_LOSS_CANDIDATES)
    all_tickers = {p.stem for p in ticker_files}
    summary_rows: List[dict] = []

    for sl_idx, sl in enumerate(STOP_LOSS_CANDIDATES, start=1):
        label = sl_label(sl)

        # ---- チェックポイント確認 ----
        chkpt_df = load_checkpoint(sl)
        processed = get_processed_tickers(chkpt_df)
        remaining = all_tickers - processed

        if not remaining:
            # 全銘柄処理済み → スキップ
            logger.info(f"[SKIP] stop_loss={label} は処理済みです")
            metrics = aggregate_metrics(chkpt_df)
            summary_rows.append({"stop_loss": label, **metrics})
            continue

        logger.info(
            f"[{sl_idx}/{total_candidates}] stop_loss={label} 開始... "
            f"(未処理: {len(remaining)}/{len(all_tickers)} 銘柄)"
        )

        # ---- 銘柄ループ ----
        for ticker_path in ticker_files:
            ticker = ticker_path.stem  # 例: "7203"

            if ticker in processed:
                continue

            # データ読み込み
            stock_df = load_ticker_data(ticker_path)

            if stock_df is None:
                logger.info(f"    [SKIP] {ticker}.T: データ不足のためスキップ")
                err_logger.error(f"[SKIP] {ticker}: データ 150 行未満またはロード失敗")
                # センチネル行として記録（再実行時にスキップ済みと認識）
                sentinel = [{"ticker": ticker, "entry_date": "",
                             "exit_date": "", "profit_loss_per_share": 0.0}]
                chkpt_df = append_to_checkpoint(sl, sentinel, chkpt_df)
                processed.add(ticker)
                continue

            # バックテスト実行
            trades = run_single_ticker(ticker, stock_df, sl, logger, err_logger)

            # メモリ解放
            del stock_df
            _gc.collect()

            if trades is None:
                # エラー時もセンチネル行で記録
                sentinel = [{"ticker": ticker, "entry_date": "",
                             "exit_date": "", "profit_loss_per_share": 0.0}]
                chkpt_df = append_to_checkpoint(sl, sentinel, chkpt_df)
            elif not trades:
                # 取引 0 件もセンチネル行で記録（再実行しないよう）
                sentinel = [{"ticker": ticker, "entry_date": "",
                             "exit_date": "", "profit_loss_per_share": 0.0}]
                chkpt_df = append_to_checkpoint(sl, sentinel, chkpt_df)
            else:
                chkpt_df = append_to_checkpoint(sl, trades, chkpt_df)

            processed.add(ticker)

        # ---- stop_loss 候補完了後に集計 ----
        chkpt_df = load_checkpoint(sl)  # 最新データを再読み込み
        metrics = aggregate_metrics(chkpt_df)
        summary_rows.append({"stop_loss": label, **metrics})

        pf_display = f"{metrics['pf']:.2f}" if metrics["pf"] != float("inf") else "inf"
        logger.info(
            f"  -> PF={pf_display}, Win={metrics['win_rate']:.1f}%, "
            f"Trades={metrics['total_trades']:,}, "
            f"Net={metrics['net_profit']:+,.0f} JPY"
        )

    # ---- 最終サマリー ----
    logger.info("")
    logger.info("=== グリッドサーチ完了 ===")
    print_summary(summary_rows, logger)
    save_final_csv(summary_rows, logger)


if __name__ == "__main__":
    main()
