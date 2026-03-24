#!/usr/bin/env python3
"""
yfinance/J-Quants data-source wrapper.

This module keeps lazy import behavior for yfinance and adds a source switch
that can be controlled from config/config.yaml with one key:
- data_source: yfinance | jquants
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# SystemFallbackPolicy integration
try:
    from src.config.system_modes import SystemFallbackPolicy, ComponentType

    _fallback_policy = SystemFallbackPolicy()
except ImportError:
    _fallback_policy = None
    print("[WARNING] SystemFallbackPolicy not available")


_CONFIG_CACHE: Dict[str, Any] = {"mtime": None, "data": {}}
_JQUANTS_CLIENT: Optional[Any] = None


def _handle_yfinance_error(error: Exception, operation: str):
    """yfinance error handling."""
    if _fallback_policy:
        return _fallback_policy.handle_component_failure(
            component_type=ComponentType.DATA_FETCHER,
            component_name="yfinance_lazy_wrapper",
            error=error,
            fallback_func=lambda: None,
        )

    print(f"[ERROR] yfinance error in {operation}: {error}")
    raise error


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_path() -> Path:
    return _project_root() / "config" / "config.yaml"


def _load_runtime_config() -> Dict[str, Any]:
    """Load config/config.yaml with a tiny fallback parser if PyYAML is missing."""
    path = _config_path()
    if not path.exists():
        return {}

    mtime = path.stat().st_mtime
    if _CONFIG_CACHE["mtime"] == mtime:
        return _CONFIG_CACHE["data"]

    text = path.read_text(encoding="utf-8")

    # Preferred parser.
    data: Dict[str, Any] = {}
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        # Minimal parser for keys used by this module.
        current_section: Optional[str] = None
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.endswith(":") and not line.startswith(" "):
                current_section = line[:-1].strip()
                data.setdefault(current_section, {})
                continue
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if raw_line.startswith("  ") and current_section and isinstance(data.get(current_section), dict):
                data[current_section][key] = value
            else:
                data[key] = value

    _CONFIG_CACHE["mtime"] = mtime
    _CONFIG_CACHE["data"] = data
    return data


def _get_data_source() -> str:
    cfg = _load_runtime_config()
    source = str(cfg.get("data_source", "yfinance")).strip().lower()
    return source if source in {"yfinance", "jquants"} else "yfinance"


def _get_jquants_api_key() -> str:
    cfg = _load_runtime_config()
    j_cfg = cfg.get("jquants", {})
    if isinstance(j_cfg, dict):
        return str(j_cfg.get("api_key", "")).strip()
    return ""


class YfinanceLazyWrapper:
    """Lazy yfinance loader with import timing stats."""

    def __init__(self):
        self._yfinance = None
        self._import_time = None
        self._first_access = True

    def _import_yfinance(self) -> Any:
        if self._yfinance is None:
            start_time = time.perf_counter()
            try:
                import yfinance as yf

                self._yfinance = yf
                self._import_time = (time.perf_counter() - start_time) * 1000
                if self._first_access:
                    print(f"[INFO] yfinance lazy import: {self._import_time:.1f}ms")
                    self._first_access = False
            except ImportError as e:
                _handle_yfinance_error(e, "_import_yfinance")

        return self._yfinance

    def __getattr__(self, name: str) -> Any:
        yf = self._import_yfinance()
        return getattr(yf, name)

    def download(self, *args, **kwargs):
        yf = self._import_yfinance()
        return yf.download(*args, **kwargs)

    def make_ticker(self, *args, **kwargs):
        yf = self._import_yfinance()
        return yf.Ticker(*args, **kwargs)

    def get_import_stats(self) -> Dict[str, Any]:
        return {
            "imported": self._yfinance is not None,
            "import_time_ms": self._import_time,
            "first_access_completed": not self._first_access,
        }


_lazy_yfinance = YfinanceLazyWrapper()


def _yfinance_download(
    ticker: str,
    start: Any = None,
    end: Any = None,
    period: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = False,
):
    kwargs: Dict[str, Any] = {
        "interval": interval,
        "auto_adjust": auto_adjust,
    }
    if start is not None:
        kwargs["start"] = start
    if end is not None:
        kwargs["end"] = end
    if period is not None and start is None and end is None:
        kwargs["period"] = period
    return _lazy_yfinance.download(ticker, **kwargs)


def _yfinance_history(
    ticker: str,
    start: Any = None,
    end: Any = None,
    period: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = False,
):
    kwargs: Dict[str, Any] = {
        "interval": interval,
        "auto_adjust": auto_adjust,
    }
    if start is not None:
        kwargs["start"] = start
    if end is not None:
        kwargs["end"] = end
    if period is not None and start is None and end is None:
        kwargs["period"] = period

    ticker_obj = _lazy_yfinance.make_ticker(ticker)
    return ticker_obj.history(**kwargs)


def _to_jquants_code(ticker: str) -> str:
    symbol = str(ticker).strip()
    if symbol.endswith(".T"):
        symbol = symbol[:-2]
    return symbol


def _is_nikkei_symbol(ticker: str) -> bool:
    symbol = str(ticker).strip().upper()
    return symbol in {"^N225", "N225", "NIKKEI", "NI225"}


def _to_yyyymmdd(value: Any) -> str:
    if value is None:
        return ""
    try:
        return pd.Timestamp(value).strftime("%Y%m%d")
    except Exception:
        return str(value).replace("-", "")


def _period_to_start(period: Optional[str]) -> str:
    if not period:
        return ""

    p = str(period).strip().lower()
    now = pd.Timestamp(datetime.now().date())
    if p == "max":
        return ""

    mapping = {
        "1d": pd.Timedelta(days=1),
        "5d": pd.Timedelta(days=5),
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
        "6mo": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "5y": pd.DateOffset(years=5),
        "10y": pd.DateOffset(years=10),
        "ytd": None,
    }

    delta = mapping.get(p)
    if p == "ytd":
        return pd.Timestamp(year=now.year, month=1, day=1).strftime("%Y%m%d")
    if delta is None:
        return ""

    start = now - delta
    return start.strftime("%Y%m%d")


def _normalize_jquants_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"])

    df = raw_df.copy()
    
    # Step 1: カラム名を小文字に統一（大文字/混在対応）
    df.columns = df.columns.str.lower()
    
    # Step 2: J-Quants API V2 カラムマッピング（調整済み列を使用）
    col_map = {
        "adjo":  "Open",
        "adjh":  "High",
        "adjl":  "Low",
        "adjc":  "Close",
        "adjvo": "Volume",
        # V1互換
        "adjustmentopen":   "Open",
        "adjustmenthigh":   "High",
        "adjustmentlow":    "Low",
        "adjustmentclose":  "Close",
        "adjustmentvolume": "Volume",
        # 日付カラム（インデックス設定に必要）
        "date": "Date",
        "datetime": "Date",
    }

    # Step 3: マッピングを適用（存在するカラムのみ）
    rename_map = {original: mapped for original, mapped in col_map.items() if original in df.columns}

    if rename_map:
        df = df.rename(columns=rename_map)

    df["Adj Close"] = df["Close"]  # Close と Adj Close を同値に設定

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    # インデックスを yfinance と同じ形式（Asia/Tokyo）に統一
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Tokyo")
    else:
        df.index = df.index.tz_convert("Asia/Tokyo")

    output_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    df = df[[c for c in output_cols if c in df.columns]]
    return df.sort_index()


def _get_jquants_client() -> Any:
    global _JQUANTS_CLIENT

    if _JQUANTS_CLIENT is not None:
        return _JQUANTS_CLIENT

    api_key = _get_jquants_api_key()
    if not api_key:
        raise ValueError("jquants.api_key is empty in config/config.yaml")

    try:
        import jquantsapi

        # J-Quants API V2 (ClientV2) を使用
        _JQUANTS_CLIENT = jquantsapi.ClientV2(api_key=api_key)
        return _JQUANTS_CLIENT
    except Exception as exc:
        print(f"[ERROR] J-Quants client initialization failed: {exc}")
        raise


def _jquants_download(ticker: str, start: Any = None, end: Any = None, interval: str = "1d") -> pd.DataFrame:
    if _is_nikkei_symbol(ticker):
        print("[INFO] ^N225 is not available via J-Quants; fallback to yfinance download")
        return _yfinance_download(ticker, start=start, end=end, interval=interval, auto_adjust=False)

    code = _to_jquants_code(ticker)
    cache_path = _project_root() / "data" / "jquants_cache" / f"{code}.csv"
    if cache_path.exists():
        try:
            cache_df = pd.read_csv(cache_path)
            cache_df["Date"] = pd.to_datetime(cache_df["Date"], errors="coerce")
            cache_df = cache_df.dropna(subset=["Date"]).set_index("Date")
            if cache_df.index.tz is not None:
                cache_df.index = cache_df.index.tz_localize(None)

            if start is not None:
                start_ts = pd.Timestamp(start)
                cache_df = cache_df[cache_df.index >= start_ts]
            if end is not None:
                end_ts = pd.Timestamp(end)
                cache_df = cache_df[cache_df.index <= end_ts]

            return cache_df
        except Exception as exc:
            print(f"[WARNING] Failed to read J-Quants cache {cache_path}: {exc}; fallback to API")

    from_yyyymmdd = _to_yyyymmdd(start)
    to_yyyymmdd = _to_yyyymmdd(end)

    if str(interval).lower() != "1d":
        print(f"[INFO] interval={interval} is not supported by J-Quants daily endpoint; fallback to yfinance")
        return _yfinance_download(ticker, start=start, end=end, interval=interval, auto_adjust=False)

    client = _get_jquants_client()

    raw_df = client.get_eq_bars_daily(
        code=code,
        from_yyyymmdd=from_yyyymmdd,
        to_yyyymmdd=to_yyyymmdd,
    )
    return _normalize_jquants_dataframe(raw_df)


def _jquants_history(ticker: str, start: Any = None, end: Any = None, interval: str = "1d") -> pd.DataFrame:
    return _jquants_download(ticker=ticker, start=start, end=end, interval=interval)


def download(
    ticker: str,
    start: Any = None,
    end: Any = None,
    period: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = False,
):
    source = _get_data_source()
    if source == "jquants":
        if start is None and end is None and period:
            start = _period_to_start(period)
        return _jquants_download(ticker, start=start, end=end, interval=interval)

    return _yfinance_download(
        ticker=ticker,
        start=start,
        end=end,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
    )


class Ticker:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self._source = _get_data_source()
        self._yfinance_ticker = None

        if self._source != "jquants":
            self._yfinance_ticker = _lazy_yfinance.make_ticker(symbol)

    def history(
        self,
        start: Any = None,
        end: Any = None,
        period: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = False,
    ):
        if self._source == "jquants":
            if start is None and end is None and period:
                start = _period_to_start(period)
            return _jquants_history(self.symbol, start=start, end=end, interval=interval)

        return _yfinance_history(
            ticker=self.symbol,
            start=start,
            end=end,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
        )

    def __getattr__(self, name: str) -> Any:
        if self._source == "jquants":
            raise AttributeError(f"Ticker.{name} is only available for yfinance source")
        return getattr(self._yfinance_ticker, name)


# Stats export for existing callers.
def get_yfinance_import_stats():
    return _lazy_yfinance.get_import_stats()


# Keep module-level attribute passthrough for yfinance access compatibility.
def __getattr__(name: str):
    return getattr(_lazy_yfinance, name)
