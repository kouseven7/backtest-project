"""
DSSMS Phase 4 Task 4.2: DSSMS実行スケジューラー
前場後場スケジューリングシステム

ハイブリッドスケジューリング・kabu API統合・緊急切替対応
"""

import sys
import time
import schedule
import json
import os
import math
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, date
import threading
import signal
import pandas as pd

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# DSSMS既存コンポーネントのインポート
from src.dssms.market_time_manager import MarketTimeManager
from src.dssms.emergency_detector import EmergencyDetector
from src.dssms.execution_history import ExecutionHistory
from src.dssms.kabu_integration_manager import KabuIntegrationManager, DSSMSKabuIntegrator
from src.dssms.email_notifier import EmailNotifier

# データ取得・戦略クラス
from data_fetcher import get_parameters_and_data
from strategies.gc_strategy_signal import GCStrategy

# 既存DSSMSコンポーネント
try:
    from src.dssms.nikkei225_screener import Nikkei225Screener
    from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem
    from src.dssms.intelligent_switch_manager import IntelligentSwitchManager
    from src.dssms.market_condition_monitor import MarketConditionMonitor
except ImportError as e:
    print(f"Warning: DSSMS components import error: {e}")

from config.logger_config import setup_logger

# ============================================================
# ペーパートレード残高管理
# ============================================================

class PaperBalance:
    """
    ペーパートレード用の仮想残高を管理するクラス。
    残高を logs/dssms/paper_balance.json に永続化する。

    初期残高: 1,000,000円
    BUY時: 残高 -= 株価 × 株数
    SELL時: 残高 += 株価 × 株数
    """
    INITIAL_BALANCE = 1_000_000  # 100万円スタート
    BALANCE_FILE = "logs/dssms/paper_balance.json"

    def __init__(self):
        self._balance = self._load()

    def _load(self) -> float:
        """残高ファイルから読み込む。存在しない場合は初期残高を返す。"""
        try:
            if os.path.exists(self.BALANCE_FILE):
                with open(self.BALANCE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return float(data.get("balance", self.INITIAL_BALANCE))
        except Exception:
            pass
        return float(self.INITIAL_BALANCE)

    def _save(self) -> None:
        """残高をファイルに保存する。"""
        try:
            os.makedirs(os.path.dirname(self.BALANCE_FILE), exist_ok=True)
            with open(self.BALANCE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {"balance": self._balance, "updated_at": datetime.now().isoformat()},
                    f, ensure_ascii=False, indent=2
                )
        except Exception:
            pass  # 保存失敗は無視（残高計算には影響させない）

    @property
    def balance(self) -> float:
        return self._balance

    def can_afford(self, price: float, quantity: int) -> bool:
        """指定の株数を購入できる残高があるか確認する。"""
        return self._balance >= price * quantity

    def deduct(self, price: float, quantity: int) -> float:
        """BUY時: 残高から購入金額を減算して保存する。残高を返す。"""
        cost = price * quantity
        self._balance -= cost
        self._save()
        return self._balance

    def add(self, price: float, quantity: int) -> float:
        """SELL時: 残高に売却金額を加算して保存する。残高を返す。"""
        proceeds = price * quantity
        self._balance += proceeds
        self._save()
        return self._balance

    def calc_quantity(self, price: float, max_positions: int) -> int:
        """
        残高と最大ポジション数から購入可能な株数（100株単位）を計算する。

        Args:
            price: 株価
            max_positions: 最大ポジション数（通常3）

        Returns:
            int: 購入株数（100株単位、0以上）
        """
        if price <= 0 or max_positions <= 0:
            return 0
        allocated = self._balance / max_positions          # 割当資金
        raw_qty = allocated / price                        # 理論上の株数
        units = int(math.floor(raw_qty / 100))            # 100株単位に切り捨て
        return units * 100                                 # 0, 100, 200, ...

class DSSMSScheduler:
    """DSSMS前場後場スケジューリングシステム"""

    PAPER_TRADE_DIR = Path("logs/paper_trade")
    DAILY_SNAPSHOT_FILE = PAPER_TRADE_DIR / "daily_snapshot.csv"
    CLOSED_TRADES_FILE = PAPER_TRADE_DIR / "closed_trades.csv"
    LAST_RUN_FILE = Path("logs/paper_trade/last_run.json")
    EXECUTION_HISTORY_FILE = Path("logs/dssms/execution_history.json")

    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス（None時はデフォルト使用）
        """
        self.logger = setup_logger('dssms.scheduler')
        
        # スケジューリング制御
        self.market_time_manager = MarketTimeManager(config_path)
        
        # kabu API統合（段階的連携）
        try:
            self.kabu_integration = KabuIntegrationManager(config_path)
            self.kabu_integrator = DSSMSKabuIntegrator(config_path)
            self.integration_mode = 'phase4_only'  # 初期は独立
            self.logger.info("kabu API統合システム初期化成功")
        except Exception as e:
            self.logger.warning(f"kabu API統合システム初期化エラー: {e}")
            self.kabu_integration = None
            self.kabu_integrator = None
            self.integration_mode = 'disabled'
        
        # EmergencyDetectorにkabu統合を渡す
        self.emergency_detector = EmergencyDetector(
            config_path=config_path,
            kabu_integration=self.kabu_integration
        )
        
        # DSSMS コアエンジン統合
        try:
            self.nikkei225_screener = Nikkei225Screener()
            self.hierarchical_ranking = HierarchicalRankingSystem({'ranking_system': {}})
            self.intelligent_switch = IntelligentSwitchManager()
            self.market_monitor = MarketConditionMonitor()
            self.logger.info("DSSMSコアエンジン初期化成功")
        except Exception as e:
            self.logger.warning(f"DSSMSコアエンジン初期化エラー: {e}")
            self.nikkei225_screener = None
            self.hierarchical_ranking = None
            self.intelligent_switch = None
            self.market_monitor = None
        
        # ExecutionHistory初期化（MarketConditionMonitor連携）
        self.execution_history = ExecutionHistory(market_monitor=self.market_monitor)
        
        # スケジューラー状態
        self.is_running = False
        self.current_monitoring_symbol: Optional[str] = None
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # ポジション管理
        self.positions_file = Path("logs/dssms/positions.json")
        self.max_positions = 3
        self._load_positions()
        
        # スケジュール設定
        self._setup_schedule()
        
        # ペーパートレード残高管理
        self.paper_balance = PaperBalance()
        self.logger.info(
            f"[PAPER_BALANCE] 初期化完了: 残高={self.paper_balance.balance:,.0f}円"
        )

        # ペーパートレード専用ディレクトリ・ファイル初期化
        self.PAPER_TRADE_DIR.mkdir(parents=True, exist_ok=True)
        if not self.DAILY_SNAPSHOT_FILE.exists():
            with open(self.DAILY_SNAPSHOT_FILE, "w", encoding="utf-8") as f:
                f.write("date,cash,unrealized_pnl,total_assets,daily_pnl,cumulative_realized_pnl,open_positions,total_trades,price_fetch_ok\n")
        if not self.CLOSED_TRADES_FILE.exists():
            with open(self.CLOSED_TRADES_FILE, "w", encoding="utf-8") as f:
                f.write("trade_id,symbol,strategy,entry_date,entry_price,shares,exit_date,exit_price,exit_reason,holding_days,pnl,pnl_pct,win\n")
        self.logger.info(f"[PAPER_TRADE_DIR] 初期化完了: {self.PAPER_TRADE_DIR}")
        
        # メール通知システム初期化
        config_file_path = config_path or (Path(__file__).parent.parent.parent / "config" / "kabu_api" / "kabu_connection_config.json")
        self.email_notifier = EmailNotifier(str(config_file_path))
        
        self.logger.info("DSSMSScheduler: 初期化完了")
    
    def _load_positions(self) -> dict:
        """positions.jsonからポジション情報を読み込む"""
        try:
            if self.positions_file.exists():
                with open(self.positions_file, 'r', encoding='utf-8') as f:
                    self.positions = json.load(f)
                self.logger.info(f"ポジション情報読み込み: {len(self.positions)}件")
            else:
                self.positions = {}
                self.logger.info("positions.json未存在 - 空のポジション辞書で初期化")
        except Exception as e:
            self.logger.error(f"ポジション情報読み込みエラー: {e}")
            self.positions = {}
        return self.positions
    
    def _save_positions(self) -> None:
        """positions.jsonにポジション情報を保存"""
        try:
            self.positions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.positions_file, 'w', encoding='utf-8') as f:
                json.dump(self.positions, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"ポジション情報保存: {len(self.positions)}件")
        except Exception as e:
            self.logger.error(f"ポジション情報保存エラー: {e}")

    def _load_balance(self) -> float:
        """paper_balance から現金残高を取得する。"""
        return self.paper_balance.balance

    def _write_daily_snapshot(self, current_prices: dict) -> None:
        """
        日次スナップショットを daily_snapshot.csv に1行追記する。

        Args:
            current_prices: {symbol: price} の辞書。価格取得失敗の銘柄は含まれない。
                            空辞書の場合は price_fetch_ok=False として記録する。
        """
        import csv
        from datetime import date as _date

        today = _date.today().isoformat()
        price_fetch_ok = len(current_prices) > 0

        positions = self._load_positions()
        cash = self._load_balance()

        # 含み損益の計算（価格取得成功時のみ）
        unrealized_pnl = None
        total_assets = None
        if price_fetch_ok:
            unrealized_pnl = 0.0
            for symbol, pos in positions.items():
                if symbol in current_prices:
                    entry_price = pos.get("entry_price", 0)
                    shares = pos.get("quantity", pos.get("shares", 0))
                    unrealized_pnl += (current_prices[symbol] - entry_price) * shares
            total_assets = cash + unrealized_pnl

        # 前日のスナップショットから cumulative_realized_pnl と total_trades を引き継ぐ
        cumulative_realized_pnl = 0.0
        total_trades = 0
        prev_total_assets = None

        if self.DAILY_SNAPSHOT_FILE.exists():
            with open(self.DAILY_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    last = rows[-1]
                    cumulative_realized_pnl = float(last.get("cumulative_realized_pnl") or 0)
                    total_trades = int(last.get("total_trades") or 0)
                    prev_total_assets_str = last.get("total_assets")
                    if prev_total_assets_str:
                        prev_total_assets = float(prev_total_assets_str)

        # 当日損益（total_assets が両日とも取得できた場合のみ計算）
        daily_pnl = None
        if total_assets is not None and prev_total_assets is not None:
            daily_pnl = total_assets - prev_total_assets

        row = {
            "date": today,
            "cash": round(cash, 2),
            "unrealized_pnl": round(unrealized_pnl, 2) if unrealized_pnl is not None else "",
            "total_assets": round(total_assets, 2) if total_assets is not None else "",
            "daily_pnl": round(daily_pnl, 2) if daily_pnl is not None else "",
            "cumulative_realized_pnl": round(cumulative_realized_pnl, 2),
            "open_positions": len(positions),
            "total_trades": total_trades,
            "price_fetch_ok": price_fetch_ok,
        }

        with open(self.DAILY_SNAPSHOT_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        status = "OK" if price_fetch_ok else "価格取得失敗（残高のみ記録）"
        self.logger.info(f"[DAILY_SNAPSHOT] {today} 記録完了 ({status})")

    def _write_last_run(
        self,
        status: str,
        price_fetch_ok: bool,
        positions_held: int,
        signals_today: dict,
        error_message: str | None = None,
    ) -> None:
        """スケジューラーの最終実行状態を last_run.json に書き込む。
        ダッシュボードのヘルスチェック用。24時間以上更新されない場合は異常とみなす。
        """
        import json
        from datetime import datetime

        data = {
            "last_run_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "status": status,
            "positions_held": positions_held,
            "signals_today": signals_today,
            "price_fetch_ok": price_fetch_ok,
            "error_message": error_message,
            "scheduler_version": "1.0",
        }

        try:
            self.LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.LAST_RUN_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[LAST_RUN] last_run.json を書き込みました: status={status}")
        except Exception as e:
            self.logger.warning(f"[LAST_RUN] last_run.json の書き込みに失敗: {e}")

    def _record_opportunity_skipped(
        self,
        date: str,
        symbol: str,
        strategy: str,
        positions_at_time: int,
    ) -> None:
        """max_positions到達によりBUYをスキップしたイベントをexecution_history.jsonに追記する。"""
        import json
        from datetime import datetime

        event = {
            "event_type": "opportunity_skipped",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "date": date,
            "symbol": symbol,
            "strategy": strategy,
            "reason": "max_positions_reached",
            "positions_at_time": positions_at_time,
            "max_positions": self.max_positions,
        }

        try:
            self.EXECUTION_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

            existing = []
            if self.EXECUTION_HISTORY_FILE.exists():
                try:
                    with open(self.EXECUTION_HISTORY_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            existing = data
                        elif isinstance(data, dict):
                            existing = data.get("events", [])
                except (json.JSONDecodeError, KeyError):
                    existing = []

            existing.append(event)

            with open(self.EXECUTION_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

            self.logger.info(
                f"[OPPORTUNITY_SKIPPED] 記録: {symbol} ({strategy}), "
                f"ポジション={positions_at_time}/{self.max_positions}"
            )
        except Exception as e:
            self.logger.warning(f"[OPPORTUNITY_SKIPPED] 記録失敗（処理は継続）: {e}")

    def _collect_portfolio_summary(self, current_prices: dict) -> Dict[str, Any]:
        """朝サマリー用の軽量なポートフォリオ情報を収集する。"""
        import csv

        positions = []
        total_unrealized_pnl = 0.0
        has_price_data = bool(current_prices)

        for symbol, pos in self.positions.items():
            entry_price = float(pos.get("entry_price", 0) or 0)
            shares = int(pos.get("quantity", pos.get("shares", 0)) or 0)
            price = current_prices.get(symbol, entry_price)
            position_unrealized_pnl = (price - entry_price) * shares if has_price_data else 0.0
            positions.append({
                "symbol": symbol,
                "price": float(price or 0),
                "shares": shares,
                "unrealized_pnl": float(position_unrealized_pnl),
            })
            total_unrealized_pnl += position_unrealized_pnl

        cash_balance = float(getattr(self.paper_balance, "balance", 0) or 0)
        unrealized_pnl = total_unrealized_pnl if has_price_data else None
        total_assets = cash_balance + total_unrealized_pnl if has_price_data else 0.0
        daily_pnl = 0.0

        if self.DAILY_SNAPSHOT_FILE.exists():
            with open(self.DAILY_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            today = date.today().isoformat()
            previous_total_assets = None
            for row in reversed(rows):
                row_date = (row.get("date") or "").strip()
                row_total_assets = (row.get("total_assets") or "").strip()
                if row_date and row_date != today and row_total_assets:
                    previous_total_assets = float(row_total_assets)
                    break

            if unrealized_pnl is not None and previous_total_assets is not None:
                daily_pnl = total_assets - previous_total_assets

        return {
            "cash_balance": cash_balance,
            "unrealized_pnl": unrealized_pnl,
            "total_assets": total_assets,
            "daily_pnl": daily_pnl,
            "positions": positions,
        }

    def _get_hours_since_last_screening_run(self, session: str, reference_time: datetime) -> float:
        """ExecutionHistory から直近実行時刻との差分時間を取得する。"""
        try:
            if not self.execution_history.history_file.exists():
                return 0.0

            with open(self.execution_history.history_file, "r", encoding="utf-8") as f:
                history = json.load(f)

            for record in reversed(history):
                if record.get("event_type") != "screening":
                    continue
                if record.get("session_type") != session:
                    continue

                timestamp_str = record.get("timestamp")
                if not timestamp_str:
                    continue

                last_run_time = datetime.fromisoformat(timestamp_str)
                if last_run_time.tzinfo is not None:
                    last_run_time = last_run_time.replace(tzinfo=None)
                if reference_time.tzinfo is not None:
                    reference_time = reference_time.replace(tzinfo=None)

                if last_run_time >= reference_time:
                    continue

                return (reference_time - last_run_time).total_seconds() / 3600.0

        except Exception as e:
            self.logger.warning(f"[MORNING_SUMMARY] 前回実行時刻取得失敗: {e}")

        return 0.0

    def _write_closed_trade(
        self,
        symbol: str,
        strategy: str,
        entry_date: str,
        entry_price: float,
        shares: int,
        exit_date: str,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """
        SELL 実行時にクローズド取引を closed_trades.csv に1行追記する。
        あわせて daily_snapshot.csv の最終行の
        cumulative_realized_pnl と total_trades を更新する。

        Args:
            symbol      : 銘柄コード（例: "3103.T"）
            strategy    : 戦略名（例: "GCStrategy"）
            entry_date  : エントリー日（YYYY-MM-DD）
            entry_price : エントリー価格
            shares      : 株数
            exit_date   : エグジット日（YYYY-MM-DD）
            exit_price  : エグジット価格
            exit_reason : "death_cross" / "stop_loss" / "forced_close" のいずれか
        """
        import csv
        from datetime import datetime

        pnl = round((exit_price - entry_price) * shares, 2)
        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4) if entry_price > 0 else 0.0
        win = 1 if pnl > 0 else 0

        try:
            entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
            exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
            holding_days = (exit_dt - entry_dt).days
        except Exception:
            holding_days = 0

        trade_id = 1
        if self.CLOSED_TRADES_FILE.exists():
            with open(self.CLOSED_TRADES_FILE, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                trade_id = len(rows) + 1

        row = {
            "trade_id": trade_id,
            "symbol": symbol,
            "strategy": strategy,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "shares": shares,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "holding_days": holding_days,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "win": win,
        }

        with open(self.CLOSED_TRADES_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        self.logger.info(
            f"[CLOSED_TRADE] {symbol} 記録完了 "
            f"PnL={pnl:+,.0f}円({pnl_pct:+.2f}%) "
            f"理由={exit_reason} 保有={holding_days}日"
        )

        self._update_snapshot_on_close(pnl)

    def _update_snapshot_on_close(self, pnl: float) -> None:
        """
        SELL 実行時に daily_snapshot.csv の最終行の
        cumulative_realized_pnl と total_trades を更新する。
        当日のスナップショットがまだない場合は何もしない。
        """
        import csv

        if not self.DAILY_SNAPSHOT_FILE.exists():
            return

        with open(self.DAILY_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            return

        last = rows[-1]
        last["cumulative_realized_pnl"] = round(
            float(last.get("cumulative_realized_pnl") or 0) + pnl, 2
        )
        last["total_trades"] = int(last.get("total_trades") or 0) + 1

        fieldnames = list(rows[0].keys())
        with open(self.DAILY_SNAPSHOT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _add_position(self, symbol: str, entry_price: float = 0.0,
                      strategy: str = "GCStrategy", quantity: int = 100) -> None:
        """
        ポジションを追加
        
        Args:
            symbol: 銘柄コード
            entry_price: エントリー価格（order_resultのexecuted_priceから取得）
            strategy: 使用戦略名（デフォルト: GCStrategy）
            quantity: 取引株数（デフォルト: 100株）
        """
        self.positions[symbol] = {
            "symbol": symbol,
            "entry_time": datetime.now().isoformat(),
            "entry_price": entry_price,
            "strategy": strategy,
            "quantity": quantity,
            "status": "open"
        }
        self._save_positions()
        self.logger.info(
            f"ポジション追加: {symbol} (戦略={strategy}, entry_price={entry_price:.2f}円, "
            f"quantity={quantity}株, {len(self.positions)}/{self.max_positions})"
        )
    
    def _execute_with_retry(self, execute_func, order_params: dict, caller: str) -> dict:
        """
        kabu API発注をリトライ付きで実行する。
        
        Args:
            execute_func: 発注を実行する関数（callable）
            order_params: 発注パラメータdict
            caller: ログ識別用の呼び出し元名（例: "SL_SELL", "前場BUY"）
        
        Returns:
            dict: success=True/False、order_id or error
        """
        max_retries = 3
        retry_interval = 5  # 秒

        for attempt in range(1, max_retries + 1):
            try:
                result = execute_func(order_params)
                if result.get("success"):
                    if attempt > 1:
                        self.logger.info(
                            f"[RETRY_OK] {caller}: {attempt}回目で発注成功"
                        )
                    return result
                else:
                    error = result.get("error", "unknown")
                    self.logger.warning(
                        f"[RETRY] {caller}: 発注失敗({attempt}/{max_retries}) error={error}"
                    )
            except Exception as e:
                self.logger.warning(
                    f"[RETRY] {caller}: 発注例外({attempt}/{max_retries}) {type(e).__name__}: {e}"
                )

            if attempt < max_retries:
                import time
                time.sleep(retry_interval)

        self.logger.error(
            f"[ORDER_FAILED] {caller}: {max_retries}回リトライ全失敗"
        )
        
        # メール通知送信
        self.email_notifier.send_order_failed(
            symbol=order_params.get("Symbol", "unknown"),
            order_type=order_params.get("Side", "unknown"),
            order_params=order_params,
            retry_count=max_retries
        )
        
        return {"success": False, "error": f"max_retries({max_retries})exceeded"}
    
    def _execute_sl_sell(self, symbol: str, quantity: int) -> dict:
        """
        ストップロス発動時のSELL発注を実行する。

        Args:
            symbol: 銘柄コード（例: "8031"）
            quantity: 売却株数

        Returns:
            dict:
                success=True  → 発注成功
                success=False → debug_modeスキップ または 発注失敗
                error         → エラー内容（debug_modeスキップ時は "debug_mode=True"）
        """
        # debug_modeチェック
        debug_mode = False
        try:
            debug_mode = self.kabu_integration.config.get(
                "development_settings", {}
            ).get("debug_mode", True)
        except Exception:
            debug_mode = True  # 取得失敗時は安全側（発注しない）

        if debug_mode:
            self.logger.info(
                f"[DEBUG_SELL] {symbol}: SL SELL発注スキップ "
                f"(debug_mode=True, qty={quantity})"
            )
            return {"success": False, "error": "debug_mode=True"}

        # kabu_integrationが使用不能な場合はスキップ
        if self.kabu_integration is None:
            self.logger.warning(
                f"[SL_SELL] {symbol}: kabu_integration未初期化、発注スキップ"
            )
            return {"success": False, "error": "kabu_integration=None"}

        try:
            self.logger.warning(
                f"[SL_SELL] {symbol}: SELL発注送信 (qty={quantity})"
            )

            sell_result = self._execute_with_retry(
                execute_func=self.kabu_integration.execute_dynamic_orders,
                order_params={
                    "symbol": symbol,
                    "side": "1",        # 1=SELL（2=BUY）
                    "quantity": quantity,
                    "price": 0,         # 成行
                    "exchange": 1       # 東証
                },
                caller=f"SL_SELL_{symbol}"
            )

            if sell_result.get("success"):
                return {
                    "success": True,
                    "order_id": sell_result.get("order_id", "unknown")
                }
            else:
                return {
                    "success": False,
                    "error": sell_result.get("error", "unknown error")
                }

        except Exception as e:
            self.logger.error(
                f"[SL_SELL] {symbol}: 発注例外 ({type(e).__name__}: {e})"
            )
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
    
    def _remove_position(self, symbol: str) -> None:
        """ポジションを削除"""
        if symbol in self.positions:
            del self.positions[symbol]
            self._save_positions()
            self.logger.info(f"ポジション削除: {symbol} ({len(self.positions)}/{self.max_positions})")
        else:
            self.logger.warning(f"ポジション削除失敗: {symbol} が存在しません")
    
    def _is_position_full(self) -> bool:
        """ポジションが満枠かどうか"""
        return len(self.positions) >= self.max_positions    
    def _check_exit_for_positions(self) -> List[str]:
        """
        保有ポジションのエグジット判定
        
        各保有ポジションについて:
        1. data_fetcherでstock_dataを取得（warmup_days=150, auto_adjust=False）
        2. position['strategy']に応じた戦略クラスをインスタンス化（まずGCStrategyのみ対応）
        3. strategy.backtest_daily()を呼び出してエグジット判定
        4. result['action']=='exit'ならSELL発注を実行
        
        Returns:
            List[str]: SELL実行した銘柄リスト
        """
        self.logger.info(f"[EXIT_CHECK] ポジション監視開始: {len(self.positions)}銘柄")
        sold_symbols = []
        
        for symbol, position in list(self.positions.items()):
            try:
                self.logger.info(f"[EXIT_CHECK] {symbol}: strategy={position.get('strategy', 'unknown')}, entry_price={position.get('entry_price', 0)}, entry_time={position.get('entry_time', 'unknown')}")
                
                # Step 1: stock_data取得（warmup_days=150）
                try:
                    ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data(
                        ticker=symbol,
                        start_date="2023-01-01",  # ウォームアップ期間を含めた開始日
                        end_date=date.today().strftime('%Y-%m-%d'),
                        warmup_days=150
                    )
                    
                    if stock_data is None or stock_data.empty:
                        self.logger.warning(f"[EXIT_CHECK] {symbol}: データ取得失敗（空データ）→スキップ")
                        continue
                    
                    self.logger.info(f"[EXIT_CHECK] {symbol}: データ取得成功 ({len(stock_data)}行, {stock_data.index[0]} - {stock_data.index[-1]})")
                    
                except Exception as e:
                    self.logger.warning(f"[EXIT_CHECK] {symbol}: データ取得エラー → スキップ: {e}")
                    continue
                
                # Step 2: 戦略クラスのインスタンス化（まずGCStrategyのみ対応、他はフォールバック）
                strategy_name = position.get('strategy', 'GCStrategy')
                if strategy_name != 'GCStrategy':
                    self.logger.info(f"[EXIT_CHECK] {symbol}: 戦略'{strategy_name}'はGCStrategyにフォールバック")
                    strategy_name = 'GCStrategy'
                
                try:
                    strategy = GCStrategy(data=stock_data, ticker=symbol)
                    strategy.initialize_strategy()  # 戦略の初期化
                    self.logger.info(f"[EXIT_CHECK] {symbol}: {strategy_name}インスタンス化成功")
                except Exception as e:
                    self.logger.warning(f"[EXIT_CHECK] {symbol}: 戦略インスタンス化失敗 → スキップ: {e}")
                    continue
                
                # Step 3: existing_position構築（entry_timeをentry_dateに変換）
                try:
                    entry_time_str = position.get('entry_time', '')
                    if isinstance(entry_time_str, str):
                        entry_date = pd.Timestamp(entry_time_str)
                    else:
                        entry_date = pd.Timestamp(entry_time_str)
                    
                    existing_position = {
                        'symbol': symbol,
                        'entry_price': position.get('entry_price', 0.0),
                        'entry_date': entry_date,
                        'quantity': position.get('quantity', 100),
                        'force_close': False  # 通常エグジット判定
                    }
                    
                    self.logger.info(f"[EXIT_CHECK] {symbol}: existing_position構築成功 (entry_price={existing_position['entry_price']}, entry_date={existing_position['entry_date'].strftime('%Y-%m-%d')})")
                    
                except Exception as e:
                    self.logger.warning(f"[EXIT_CHECK] {symbol}: existing_position構築エラー → スキップ: {e}")
                    continue
                
                # Step 4: backtest_daily()呼び出し
                try:
                    result = strategy.backtest_daily(
                        current_date=date.today(),
                        stock_data=stock_data,
                        existing_position=existing_position
                    )
                    
                    self.logger.info(f"[EXIT_CHECK] {symbol}: backtest_daily結果 action={result.get('action')}, signal={result.get('signal')}, reason={result.get('reason', 'N/A')}")
                    
                    # Step 5: exitアクション時のSELL発注
                    if result.get('action') == 'exit':
                        exit_price = result.get('price', 0.0)
                        exit_shares = result.get('shares', existing_position['quantity'])
                        exit_reason = result.get('reason', 'Exit signal')
                        today = date.today().isoformat()

                        strategy_signal = str(result.get('signal', '')).lower()
                        close_exit_reason = "death_cross" if strategy_signal == "sell" else "forced_close"

                        entry_date_raw = position.get('entry_time') or position.get('entry_date') or today
                        try:
                            entry_date_for_csv = pd.Timestamp(entry_date_raw).strftime('%Y-%m-%d')
                        except Exception:
                            entry_date_for_csv = today
                        
                        self.logger.info(f"[EXIT_SIGNAL] {symbol}: SELL発注 (price={exit_price}, shares={exit_shares}, reason={exit_reason})")
                        
                        # SELL発注実行（_execute_sl_sell()を使用）
                        sell_result = self._execute_sl_sell(symbol=symbol, quantity=exit_shares)
                        if sell_result.get("success"):
                            self.logger.info(
                                f"[EXIT_SELL_OK] {symbol}: SELL発注成功 "
                                f"(order_id={sell_result.get('order_id')})"
                            )
                        else:
                            self.logger.info(
                                f"[EXIT_SELL_NG] {symbol}: SELL発注失敗またはスキップ "
                                f"({sell_result.get('error', 'debug_mode')})"
                            )
                        
                        # SELL時の残高加算
                        if exit_price > 0 and exit_shares > 0:
                            new_balance = self.paper_balance.add(exit_price, exit_shares)
                            self.logger.info(
                                f"[BALANCE_ADD] {symbol} (EXIT): "
                                f"+{exit_price * exit_shares:,.0f}円 -> 残高={new_balance:,.0f}円"
                            )
                        
                        # ポジション削除
                        self._remove_position(symbol)

                        self._write_closed_trade(
                            symbol=symbol,
                            strategy=strategy_name,
                            entry_date=entry_date_for_csv,
                            entry_price=float(existing_position.get('entry_price', 0.0) or 0.0),
                            shares=int(exit_shares or 0),
                            exit_date=today,
                            exit_price=float(exit_price or 0.0),
                            exit_reason=close_exit_reason,
                        )

                        sold_symbols.append(symbol)
                        
                        # 実行履歴記録
                        if hasattr(self, 'execution_history'):
                            self.execution_history.record_trade_sell(
                                symbol=symbol,
                                sell_price=exit_price,
                                quantity=exit_shares,
                                entry_price=existing_position['entry_price'],
                                reason=exit_reason
                            )
                    else:
                        self.logger.info(f"[EXIT_CHECK] {symbol}: ホールド継続 (action={result.get('action')})")
                    
                except Exception as e:
                    self.logger.warning(f"[EXIT_CHECK] {symbol}: backtest_daily実行エラー → スキップ: {e}")
                    continue
                
            except Exception as e:
                self.logger.error(f"[EXIT_CHECK] {symbol}: 予期しないエラー → スキップ: {e}")
                continue
        
        self.logger.info(f"[EXIT_CHECK] 監視完了: {len(sold_symbols)}銘柄をSELL ({sold_symbols})")
        return sold_symbols    
    def _setup_schedule(self):
        """スケジュール設定"""
        try:
            # 前場スクリーニング (09:30)
            schedule.every().monday.at("09:30").do(self._run_morning_screening_job)
            schedule.every().tuesday.at("09:30").do(self._run_morning_screening_job)
            schedule.every().wednesday.at("09:30").do(self._run_morning_screening_job)
            schedule.every().thursday.at("09:30").do(self._run_morning_screening_job)
            schedule.every().friday.at("09:30").do(self._run_morning_screening_job)
            
            # 後場スクリーニング (12:30)
            schedule.every().monday.at("12:30").do(self._run_afternoon_screening_job)
            schedule.every().tuesday.at("12:30").do(self._run_afternoon_screening_job)
            schedule.every().wednesday.at("12:30").do(self._run_afternoon_screening_job)
            schedule.every().thursday.at("12:30").do(self._run_afternoon_screening_job)
            schedule.every().friday.at("12:30").do(self._run_afternoon_screening_job)
            
            # 緊急チェック（1分毎）
            schedule.every().minute.do(self._emergency_check_job)
            
            self.logger.info("スケジュール設定完了")
            
        except Exception as e:
            self.logger.error(f"スケジュール設定エラー: {e}")
    
    def run_morning_screening(self) -> str:
        """
        09:30前場スクリーニング実行
        
        Returns:
            str: 選択された銘柄コード（エラー時は空文字）
        """
        start_time = datetime.now()
        self.logger.info("=== 前場スクリーニング開始 ===")
        
        selected_symbol = ""
        error_occurred = False
        error_message = ""
        screening_result = {}
        current_prices = {}
        
        try:
            # 市場時間チェック
            if not self.market_time_manager.should_run_screening("morning"):
                self.logger.warning("前場スクリーニング実行条件未満足")
                return ""
            
            # --- エグジット判定（追加） ---
            exited_symbols = self._check_exit_for_positions()
            if exited_symbols:
                self.logger.info(f"[EXIT] {len(exited_symbols)}銘柄をSELL: {exited_symbols}")
            # --- ここまで追加 ---

            # --- 保有銘柄の現在価格取得 + 日次スナップショット記録 ---
            current_prices = {}
            for sym in list(self.positions.keys()):
                try:
                    price = None
                    if self.kabu_integration is not None:
                        try:
                            price = self.kabu_integration.get_current_price(sym)
                        except Exception:
                            pass
                    if price is None or price <= 0:
                        from datetime import timedelta
                        try:
                            _, _, _, sym_data, _ = get_parameters_and_data(
                                ticker=sym,
                                start_date=(date.today() - timedelta(days=5)).strftime('%Y-%m-%d'),
                                end_date=date.today().strftime('%Y-%m-%d'),
                                warmup_days=0
                            )
                            if sym_data is not None and not sym_data.empty:
                                price = float(sym_data['Adj Close'].iloc[-1])
                        except Exception as e:
                            self.logger.warning(f"[PRICE_FETCH] {sym}: yfinance価格取得失敗: {e}")
                    if price is not None and price > 0:
                        current_prices[sym] = price
                        self.logger.info(f"[PRICE_FETCH] {sym}: {price:.2f}円")
                    else:
                        self.logger.warning(f"[PRICE_FETCH] {sym}: 価格取得失敗")
                except Exception as e:
                    self.logger.warning(f"[PRICE_FETCH] {sym}: エラー: {e}")
            self._write_daily_snapshot(current_prices)
            # --- ここまで ---

            # スクリーニング実行
            screening_result = self._execute_screening("morning")
            
            # kabu API連携
            if self.kabu_integrator and screening_result.get("selected_symbol"):
                try:
                    sync_success = self.kabu_integrator.sync_screening_results_to_kabu(10000000.0)
                    self.logger.info(f"kabu API同期: {'成功' if sync_success else '失敗'}")
                    
                    symbol = screening_result["selected_symbol"]
                    
                    # 満枠チェック
                    if self._is_position_full():
                        self.logger.info(f"ポジション満枠({len(self.positions)}/{self.max_positions})のためBUYスキップ: {symbol}")
                        self._record_opportunity_skipped(
                            date=str(date.today()),
                            symbol=symbol,
                            strategy=screening_result.get("selected_strategy", "Unknown"),
                            positions_at_time=len(self.positions),
                        )
                    else:
                        # 株価取得（BUY前に株数計算のため）
                        # kabu STATIONからリアルタイム価格取得
                        kabu_price = None
                        try:
                            if self.kabu_integration is not None:
                                kabu_price = self.kabu_integration.get_current_price(symbol)
                        except Exception as e:
                            self.logger.warning(f"[PRICE_FETCH] {symbol}: kabu API価格取得失敗 ({e})")

                        if kabu_price is not None and kabu_price > 0:
                            current_price = kabu_price
                            self.logger.info(f"[PRICE_FETCH] {symbol}: kabu API価格={current_price:.2f}円")
                        else:
                            # フォールバック: yfinance
                            try:
                                from datetime import timedelta
                                ticker, start, end, stock_data, _ = get_parameters_and_data(
                                    ticker=symbol,
                                    start_date=(date.today() - timedelta(days=5)).strftime('%Y-%m-%d'),
                                    end_date=date.today().strftime('%Y-%m-%d'),
                                    warmup_days=0
                                )
                                current_price = stock_data['Adj Close'].iloc[-1] if not stock_data.empty else 1000.0
                                self.logger.info(f"[PRICE_FETCH] {symbol}: yfinance価格={current_price:.2f}円（フォールバック）")
                            except Exception as e:
                                self.logger.warning(f"[PRICE_FETCH] {symbol}: 価格取得失敗、仮株価1000円を使用: {e}")
                                current_price = 1000.0
                        
                        # C-2: 動的株数計算
                        quantity = self.paper_balance.calc_quantity(
                            price=current_price,
                            max_positions=self.max_positions
                        )
                        
                        # C-3: 資金残高チェック
                        if quantity == 0:
                            self.logger.warning(
                                f"[BALANCE_NG] {symbol}: 残高不足または株価高すぎで発注スキップ "
                                f"(残高={self.paper_balance.balance:,.0f}円, 株価={current_price:.2f}円)"
                            )
                        elif not self.paper_balance.can_afford(current_price, quantity):
                            self.logger.warning(
                                f"[BALANCE_NG] {symbol}: 残高不足 "
                                f"(必要={current_price * quantity:,.0f}円, 残高={self.paper_balance.balance:,.0f}円)"
                            )
                        else:
                            # BUYモック注文
                            order_result = self._execute_with_retry(
                                execute_func=self.kabu_integrator.kabu_manager.execute_dynamic_orders,
                                order_params={
                                    'symbol': symbol,
                                    'side': '2',
                                    'quantity': quantity,
                                    'price': 0
                                },
                                caller=f"前場BUY_{symbol}"
                            )
                            self.logger.info(f"前場BUY注文: {symbol} qty={quantity}株 結果={order_result}")
                            if order_result.get("success"):
                                # executed_priceを取得（0の場合はcurrent_priceをフォールバック）
                                executed_price = order_result.get("executed_price") or current_price
                                if executed_price == 0 or executed_price is None:
                                    executed_price = current_price
                                self.logger.info(f"[BUY] {symbol}: executed_price={executed_price}円 (current_price={current_price}円)")
                                
                                # 残高から減算（BUY確定）
                                new_balance = self.paper_balance.deduct(executed_price, quantity)
                                self.logger.info(
                                    f"[BALANCE_DEDUCT] {symbol}: "
                                    f"-{executed_price * quantity:,.0f}円 -> 残高={new_balance:,.0f}円"
                                )
                                # ポジション追加
                                self._add_position(
                                    symbol=symbol,
                                    entry_price=executed_price,
                                    strategy="GCStrategy",  # TODO: DynamicStrategySelectorで動的選択
                                    quantity=quantity
                                )
                                # 実行履歴記録（BUY）
                                if hasattr(self, 'execution_history'):
                                    self.execution_history.record_trade_buy(
                                        symbol=symbol,
                                        price=executed_price,
                                        quantity=quantity,
                                        strategy="GCStrategy"
                                    )
                except Exception as e:
                    self.logger.error(f"kabu API同期エラー: {e}")
            
            selected_symbol = screening_result.get("selected_symbol", "")
            
            # 選択銘柄の監視開始
            if selected_symbol:
                self.start_selected_symbol_monitoring(selected_symbol)
            
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            self.logger.error(f"前場スクリーニングエラー: {e}")
        
        # try-exceptの外で1回だけ記録
        duration = (datetime.now() - start_time).total_seconds()
        if error_occurred:
            record_result = {
                "status": "error",
                "error": error_message,
                "duration": duration
            }
        else:
            screening_result["duration"] = duration
            record_result = screening_result
        
        self.execution_history.record_screening_execution("morning", record_result)
        
        if not error_occurred:
            self.logger.info(f"=== 前場スクリーニング完了: {selected_symbol} ({duration:.2f}秒) ===")

        try:
            portfolio_summary = self._collect_portfolio_summary(current_prices)
            summary_data = {
                "execution_time": datetime.now().strftime("%H:%M"),
                "status": "異常" if error_occurred else "正常",
                "error_message": error_message,
                "cash_balance": portfolio_summary.get("cash_balance", 0.0),
                "unrealized_pnl": portfolio_summary.get("unrealized_pnl"),
                "total_assets": portfolio_summary.get("total_assets", 0.0),
                "daily_pnl": portfolio_summary.get("daily_pnl", 0.0),
                "positions": portfolio_summary.get("positions", []),
                "screened_symbols": screening_result.get("screened_symbols") or ([selected_symbol] if selected_symbol else []),
                "hours_since_last_run": self._get_hours_since_last_screening_run("morning", start_time),
            }
            self.email_notifier.send_morning_summary(summary_data)
        except Exception as e:
            self.logger.error(f"[MORNING_SUMMARY] 朝サマリー送信処理エラー: {e}")
        
        return selected_symbol
    
    def run_afternoon_screening(self) -> str:
        """
        12:30後場スクリーニング実行
        
        Returns:
            str: 選択された銘柄コード（エラー時は空文字）
        """
        start_time = datetime.now()
        self.logger.info("=== 後場スクリーニング開始 ===")
        
        selected_symbol = ""
        error_occurred = False
        error_message = ""
        screening_result = {}
        
        try:
            # 市場時間チェック
            if not self.market_time_manager.should_run_screening("afternoon"):
                self.logger.warning("後場スクリーニング実行条件未満足")
                return ""
            
            # --- エグジット判定（追加） ---
            exited_symbols = self._check_exit_for_positions()
            if exited_symbols:
                self.logger.info(f"[EXIT] {len(exited_symbols)}銘柄をSELL: {exited_symbols}")
            # --- ここまで追加 ---
            
            # スクリーニング実行
            screening_result = self._execute_screening("afternoon")
            
            # kabu API連携
            if self.kabu_integrator and screening_result.get("selected_symbol"):
                try:
                    sync_success = self.kabu_integrator.sync_screening_results_to_kabu(10000000.0)
                    self.logger.info(f"kabu API同期: {'成功' if sync_success else '失敗'}")
                    
                    symbol = screening_result["selected_symbol"]
                    
                    # 満枠チェック
                    if self._is_position_full():
                        self.logger.info(f"ポジション満枠({len(self.positions)}/{self.max_positions})のためBUYスキップ: {symbol}")
                        self._record_opportunity_skipped(
                            date=str(date.today()),
                            symbol=symbol,
                            strategy=screening_result.get("selected_strategy", "Unknown"),
                            positions_at_time=len(self.positions),
                        )
                    else:
                        # 株価取得（BUY前に株数計算のため）
                        # kabu STATIONからリアルタイム価格取得
                        kabu_price = None
                        try:
                            if self.kabu_integration is not None:
                                kabu_price = self.kabu_integration.get_current_price(symbol)
                        except Exception as e:
                            self.logger.warning(f"[PRICE_FETCH] {symbol}: kabu API価格取得失敗 ({e})")

                        if kabu_price is not None and kabu_price > 0:
                            current_price = kabu_price
                            self.logger.info(f"[PRICE_FETCH] {symbol}: kabu API価格={current_price:.2f}円")
                        else:
                            # フォールバック: yfinance
                            try:
                                from datetime import timedelta
                                ticker, start, end, stock_data, _ = get_parameters_and_data(
                                    ticker=symbol,
                                    start_date=(date.today() - timedelta(days=5)).strftime('%Y-%m-%d'),
                                    end_date=date.today().strftime('%Y-%m-%d'),
                                    warmup_days=0
                                )
                                current_price = stock_data['Adj Close'].iloc[-1] if not stock_data.empty else 1000.0
                                self.logger.info(f"[PRICE_FETCH] {symbol}: yfinance価格={current_price:.2f}円（フォールバック）")
                            except Exception as e:
                                self.logger.warning(f"[PRICE_FETCH] {symbol}: 価格取得失敗、仮株価1000円を使用: {e}")
                                current_price = 1000.0
                        
                        # C-2: 動的株数計算
                        quantity = self.paper_balance.calc_quantity(
                            price=current_price,
                            max_positions=self.max_positions
                        )
                        
                        # C-3: 資金残高チェック
                        if quantity == 0:
                            self.logger.warning(
                                f"[BALANCE_NG] {symbol}: 残高不足または株価高すぎで発注スキップ "
                                f"(残高={self.paper_balance.balance:,.0f}円, 株価={current_price:.2f}円)"
                            )
                        elif not self.paper_balance.can_afford(current_price, quantity):
                            self.logger.warning(
                                f"[BALANCE_NG] {symbol}: 残高不足 "
                                f"(必要={current_price * quantity:,.0f}円, 残高={self.paper_balance.balance:,.0f}円)"
                            )
                        else:
                            # BUYモック注文
                            order_result = self._execute_with_retry(
                                execute_func=self.kabu_integrator.kabu_manager.execute_dynamic_orders,
                                order_params={
                                    'symbol': symbol,
                                    'side': '2',
                                    'quantity': quantity,
                                    'price': 0
                                },
                                caller=f"後場BUY_{symbol}"
                            )
                            self.logger.info(f"後場BUY注文: {symbol} qty={quantity}株 結果={order_result}")
                            if order_result.get("success"):
                                # executed_priceを取得（0の場合はcurrent_priceをフォールバック）
                                executed_price = order_result.get("executed_price") or current_price
                                if executed_price == 0 or executed_price is None:
                                    executed_price = current_price
                                self.logger.info(f"[BUY] {symbol}: executed_price={executed_price}円 (current_price={current_price}円)")
                                
                                # 残高から減算（BUY確定）
                                new_balance = self.paper_balance.deduct(executed_price, quantity)
                                self.logger.info(
                                    f"[BALANCE_DEDUCT] {symbol}: "
                                    f"-{executed_price * quantity:,.0f}円 -> 残高={new_balance:,.0f}円"
                                )
                                # ポジション追加
                                self._add_position(
                                    symbol=symbol,
                                    entry_price=executed_price,
                                    strategy="GCStrategy",  # TODO: DynamicStrategySelectorで動的選択
                                    quantity=quantity
                                )
                                # 実行履歴記録（BUY）
                                if hasattr(self, 'execution_history'):
                                    self.execution_history.record_trade_buy(
                                        symbol=symbol,
                                        price=executed_price,
                                        quantity=quantity,
                                        strategy="GCStrategy"
                                    )
                except Exception as e:
                    self.logger.error(f"kabu API同期エラー: {e}")
            
            selected_symbol = screening_result.get("selected_symbol", "")
            
            # 選択銘柄の監視開始
            if selected_symbol:
                self.start_selected_symbol_monitoring(selected_symbol)
            
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            self.logger.error(f"後場スクリーニングエラー: {e}")
        
        # try-exceptの外で1回だけ記録
        duration = (datetime.now() - start_time).total_seconds()
        if error_occurred:
            record_result = {
                "status": "error",
                "error": error_message,
                "duration": duration
            }
        else:
            screening_result["duration"] = duration
            record_result = screening_result
        
        self.execution_history.record_screening_execution("afternoon", record_result)
        
        if not error_occurred:
            self.logger.info(f"=== 後場スクリーニング完了: {selected_symbol} ({duration:.2f}秒) ===")
        
        return selected_symbol
    
    def start_selected_symbol_monitoring(self, symbol: str) -> None:
        """
        選択銘柄のリアルタイム監視開始
        
        Args:
            symbol: 監視対象銘柄コード
        """
        try:
            self.current_monitoring_symbol = symbol
            
            # kabu API監視登録
            if self.kabu_integration:
                try:
                    registration_success = self.kabu_integration.register_screening_symbols([symbol])
                    self.logger.info(f"kabu API監視登録: {symbol} - {'成功' if registration_success else '失敗'}")
                except Exception as e:
                    self.logger.error(f"kabu API監視登録エラー {symbol}: {e}")
            
            # 監視イベント記録
            monitoring_data = {
                "event_type": "monitoring_start",
                "symbol": symbol,
                "start_time": datetime.now().isoformat()
            }
            
            self.execution_history.record_monitoring_event(symbol, monitoring_data)
            
            self.logger.info(f"銘柄監視開始: {symbol}")
            
        except Exception as e:
            self.logger.error(f"銘柄監視開始エラー {symbol}: {e}")
    
    def handle_emergency_switch_check(self) -> None:
        """パーフェクトオーダー崩れ時の緊急判定処理"""
        try:
            # 現在監視中の銘柄がない場合はスキップ
            if not self.current_monitoring_symbol:
                return
            
            # 市場開場中のみ実行
            if not self.market_time_manager.is_market_open():
                return
            
            symbol = self.current_monitoring_symbol
            
            # 緊急事態判定
            emergency_result = self.emergency_detector.check_emergency_conditions(symbol)
            
            # 緊急判定記録
            self.execution_history.record_emergency_check(symbol, emergency_result)
            
            # 緊急事態の場合の処理
            if emergency_result.get("is_emergency"):
                self.logger.warning(f"緊急事態検出: {symbol} - Level {emergency_result.get('emergency_level')}")
                
                # 推奨アクションに基づく処理
                recommended_action = emergency_result.get("recommended_action")
                
                if recommended_action == "immediate_exit":
                    # ストップロス発動時の処理
                    position_info = self.positions.get(symbol, {})
                    stop_loss_details = emergency_result.get("stop_loss_details", {})
                    entry_price = stop_loss_details.get("entry_price", "N/A")
                    current_price = stop_loss_details.get("current_price", "N/A")
                    loss_percentage = stop_loss_details.get("loss_percentage", 0.0)
                    quantity = position_info.get("quantity", 100)

                    today = date.today().isoformat()
                    strategy_name = position_info.get("strategy", "GCStrategy")
                    entry_date_raw = position_info.get("entry_time") or position_info.get("entry_date") or today
                    try:
                        entry_date_for_csv = pd.Timestamp(entry_date_raw).strftime('%Y-%m-%d')
                    except Exception:
                        entry_date_for_csv = today
                    entry_price_for_csv = float(position_info.get("entry_price", 0.0) or 0.0)

                    self.logger.warning(
                        f"[SL_TRIGGERED] {symbol}: "
                        f"entry={entry_price}, current={current_price:.2f}, "
                        f"loss={loss_percentage:.2%}, qty={quantity}"
                    )

                    # --- C-1: 実SELL発注 ---
                    sell_result = self._execute_sl_sell(symbol=symbol, quantity=quantity)
                    if sell_result.get("success"):
                        self.logger.warning(
                            f"[SL_SELL_OK] {symbol}: SELL発注成功 "
                            f"(order_id={sell_result.get('order_id')})"
                        )
                    else:
                        self.logger.warning(
                            f"[SL_SELL_NG] {symbol}: SELL発注失敗またはスキップ "
                            f"({sell_result.get('error', 'debug_mode')})"
                        )
                    # --- C-1 ここまで ---
                    
                    # SELL時の残高加算（約定価格はcurrent_priceを使用、取得できない場合はentry_priceをフォールバック）
                    sell_price = current_price if isinstance(current_price, (int, float)) and current_price > 0 else self.positions.get(symbol, {}).get("entry_price", 0)
                    if sell_price > 0 and quantity > 0:
                        new_balance = self.paper_balance.add(sell_price, quantity)
                        self.logger.info(
                            f"[BALANCE_ADD] {symbol} (SL): "
                            f"+{sell_price * quantity:,.0f}円 -> 残高={new_balance:,.0f}円"
                        )

                    # ポジション削除（発注成否に関わらず必ず実行）
                    self._remove_position(symbol)
                    self.logger.warning(f"[SL_EXECUTED] {symbol}: ポジション削除完了")

                    if position_info:
                        self._write_closed_trade(
                            symbol=symbol,
                            strategy=strategy_name,
                            entry_date=entry_date_for_csv,
                            entry_price=entry_price_for_csv,
                            shares=int(quantity or 0),
                            exit_date=today,
                            exit_price=float(sell_price or 0.0),
                            exit_reason="stop_loss",
                        )
                    
                elif recommended_action == "immediate_switch":
                    self._execute_emergency_switch(symbol, emergency_result)
                elif recommended_action == "prepare_switch":
                    self._prepare_emergency_switch(symbol, emergency_result)
                elif recommended_action == "close_monitoring":
                    self.logger.info(f"詳細監視開始: {symbol}")
            
        except Exception as e:
            self.logger.error(f"緊急切替チェックエラー: {e}")
    
    def _execute_screening(self, session: str) -> Dict[str, Any]:
        """
        スクリーニング実行
        
        Returns:
            Dict[str, Any]: スクリーニング結果
                - status: 'success' | 'error'
                - session: 'morning' | 'afternoon'
                - selected_symbol: 選択された銘柄コード
                - selected_strategy: 使用戦略名 (TODO: DynamicStrategySelector統合)
                - candidate_count: 候補銘柄数
                - priority_distribution: 優先度分布
        """
        try:
            result = {
                "status": "success",
                "session": session,
                "selected_symbol": None,
                "selected_strategy": "GCStrategy",  # TODO: DynamicStrategySelectorで動的選択
                "candidate_count": 0,
                "priority_distribution": {}
            }
            
            # Nikkei225スクリーニング
            if self.nikkei225_screener:
                try:
                    candidates = self.nikkei225_screener.get_filtered_symbols(
                        available_funds=10000000.0,
                        target_date=date.today()
                    )
                    result["candidate_count"] = len(candidates)
                    self.logger.info(f"スクリーニング候補取得: {len(candidates)}銘柄")
                except Exception as e:
                    self.logger.error(f"スクリーニング候補取得エラー: {e}")
                    candidates = []
            else:
                # フォールバック候補
                candidates = ["6758", "9433", "9984", "5401", "6645"]
                result["candidate_count"] = len(candidates)
            
            # 階層的ランキング
            if self.hierarchical_ranking and candidates:
                try:
                    # 上位候補から選択
                    top_candidates = self.hierarchical_ranking.get_backup_candidates(n=5)
                    if top_candidates:
                        result["selected_symbol"] = top_candidates[0]
                        result["priority_distribution"] = {"level1": 1, "level2": 2, "level3": 2}
                except Exception as e:
                    self.logger.error(f"階層ランキングエラー: {e}")
                    result["selected_symbol"] = candidates[0] if candidates else None
            else:
                # フォールバック選択
                result["selected_symbol"] = candidates[0] if candidates else None
            
            return result
            
        except Exception as e:
            self.logger.error(f"スクリーニング実行エラー {session}: {e}")
            return {
                "status": "error",
                "session": session,
                "selected_strategy": "GCStrategy",  # エラー時もデフォルト戦略を返す
                "error": str(e)
            }
    
    def _execute_emergency_switch(self, symbol: str, emergency_result: Dict[str, Any]) -> None:
        """緊急切替実行"""
        try:
            self.logger.warning(f"緊急切替実行: {symbol}")
            
            # インテリジェント切替実行
            if self.intelligent_switch:
                try:
                    # 切替候補取得
                    backup_candidates = self.hierarchical_ranking.get_backup_candidates(n=3) if self.hierarchical_ranking else ["6758", "9433", "9984"]
                    
                    if backup_candidates:
                        target_symbol = backup_candidates[0]
                        
                        # 切替実行
                        switch_success = self.intelligent_switch.execute_switch_with_risk_control(symbol, target_symbol)
                        
                        if switch_success:
                            self.current_monitoring_symbol = target_symbol
                            self.logger.info(f"緊急切替成功: {symbol} → {target_symbol}")

                            # SELL→BUYモック注文
                            if self.kabu_integration:
                                try:
                                    position_info = self.positions.get(symbol, {})
                                    today = date.today().isoformat()
                                    strategy_name = position_info.get("strategy", "GCStrategy")
                                    entry_date_raw = position_info.get("entry_time") or position_info.get("entry_date") or today
                                    try:
                                        entry_date_for_csv = pd.Timestamp(entry_date_raw).strftime('%Y-%m-%d')
                                    except Exception:
                                        entry_date_for_csv = today
                                    entry_price_for_csv = float(position_info.get("entry_price", 0.0) or 0.0)

                                    # 旧銘柄SELL
                                    sell_result = self.kabu_integration.execute_dynamic_orders({
                                        'symbol': symbol,
                                        'side': '1',
                                        'quantity': 100,
                                        'price': 0
                                    })
                                    self.logger.info(f"緊急SELL注文: {symbol} 結果={sell_result}")

                                    if position_info:
                                        forced_exit_price = sell_result.get("executed_price")
                                        if forced_exit_price is None or forced_exit_price == 0:
                                            forced_exit_price = entry_price_for_csv
                                        forced_shares = int(position_info.get("quantity", 100) or 100)
                                        self._write_closed_trade(
                                            symbol=symbol,
                                            strategy=strategy_name,
                                            entry_date=entry_date_for_csv,
                                            entry_price=entry_price_for_csv,
                                            shares=forced_shares,
                                            exit_date=today,
                                            exit_price=float(forced_exit_price or 0.0),
                                            exit_reason="forced_close",
                                        )

                                    # 新銘柄BUY
                                    buy_result = self.kabu_integration.execute_dynamic_orders({
                                        'symbol': target_symbol,
                                        'side': '2',
                                        'quantity': 100,
                                        'price': 0
                                    })
                                    self.logger.info(f"緊急BUY注文: {target_symbol} 結果={buy_result}")
                                except Exception as e:
                                    self.logger.error(f"緊急注文エラー: {e}")

                            # 切替記録
                            switch_data = {
                                "from_symbol": symbol,
                                "to_symbol": target_symbol,
                                "reason": "emergency_switch",
                                "emergency_level": emergency_result.get("emergency_level"),
                                "conditions": emergency_result.get("trigger_conditions", []),
                                "execution_result": {"success": True}
                            }
                            self.execution_history.record_symbol_switch(switch_data)
                        else:
                            self.logger.error(f"緊急切替失敗: {symbol} → {target_symbol}")
                except Exception as e:
                    self.logger.error(f"インテリジェント切替エラー: {e}")
            
        except Exception as e:
            self.logger.error(f"緊急切替実行エラー: {e}")
    
    def _prepare_emergency_switch(self, symbol: str, emergency_result: Dict[str, Any]) -> None:
        """緊急切替準備"""
        try:
            self.logger.info(f"緊急切替準備: {symbol}")
            
            # 切替候補の事前準備
            if self.hierarchical_ranking:
                backup_candidates = self.hierarchical_ranking.get_backup_candidates(n=5)
                self.logger.info(f"切替候補準備完了: {backup_candidates}")
            
        except Exception as e:
            self.logger.error(f"緊急切替準備エラー: {e}")
    
    def _run_morning_screening_job(self):
        """前場スクリーニングジョブ（内部）"""
        try:
            result = self.run_morning_screening()
            self.logger.info(f"前場スクリーニングジョブ完了: {result}")
            self._write_last_run(
                status="success",
                price_fetch_ok=True,
                positions_held=len(self.positions),
                signals_today={"buy": 0, "sell": 0, "hold": 0},
                error_message=None,
            )
        except Exception as e:
            self.logger.error(f"前場スクリーニングジョブエラー: {e}")
    
    def _run_afternoon_screening_job(self):
        """後場スクリーニングジョブ（内部）"""
        try:
            result = self.run_afternoon_screening()
            self.logger.info(f"後場スクリーニングジョブ完了: {result}")
        except Exception as e:
            self.logger.error(f"後場スクリーニングジョブエラー: {e}")
    
    def _emergency_check_job(self):
        """緊急チェックジョブ（内部）"""
        try:
            self.handle_emergency_switch_check()
        except Exception as e:
            self.logger.error(f"緊急チェックジョブエラー: {e}")
    
    def start_scheduler(self):
        """スケジューラー開始"""
        if self.is_running:
            self.logger.warning("スケジューラーは既に実行中です")
            return
        
        self.is_running = True
        self.logger.info("DSSMSスケジューラー開始")
        
        def scheduler_loop():
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(30)  # 30秒間隔でチェック
                except Exception as e:
                    self.logger.error(f"スケジューラーループエラー: {e}")
                    time.sleep(60)  # エラー時は1分待機
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("DSSMSスケジューラースレッド開始")
    
    def stop_scheduler(self):
        """スケジューラー停止"""
        if not self.is_running:
            self.logger.warning("スケジューラーは実行中ではありません")
            return
        
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("DSSMSスケジューラー停止")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """スケジューラー状況取得"""
        try:
            return {
                "is_running": self.is_running,
                "current_monitoring_symbol": self.current_monitoring_symbol,
                "current_session": self.market_time_manager.get_current_session(),
                "next_screening_time": self.market_time_manager.get_next_screening_time(),
                "market_open": self.market_time_manager.is_market_open(),
                "integration_mode": self.integration_mode,
                "kabu_api_available": self.kabu_integration is not None,
                "dssms_components_available": {
                    "screener": self.nikkei225_screener is not None,
                    "ranking": self.hierarchical_ranking is not None,
                    "switch_manager": self.intelligent_switch is not None,
                    "market_monitor": self.market_monitor is not None
                }
            }
        except Exception as e:
            self.logger.error(f"スケジューラー状況取得エラー: {e}")
            return {"error": str(e)}


def main():
    """メイン実行関数"""
    scheduler = DSSMSScheduler()
    
    # シグナルハンドラー設定
    def signal_handler(signum, frame):
        print("\nシャットダウン信号を受信しました...")
        scheduler.stop_scheduler()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # スケジューラー開始
        scheduler.start_scheduler()
        
        # 状況表示
        status = scheduler.get_scheduler_status()
        print(f"DSSMSスケジューラー状況: {status}")
        
        # メインループ
        while True:
            time.sleep(60)  # 1分毎に状況チェック
            if not scheduler.is_running:
                break
                
    except KeyboardInterrupt:
        print("\nキーボード割り込みを受信しました...")
    except Exception as e:
        print(f"予期しないエラー: {e}")
        try:
            scheduler._write_last_run(
                status="error",
                price_fetch_ok=False,
                positions_held=0,
                signals_today={"buy": 0, "sell": 0, "hold": 0},
                error_message=str(e)[:500],
            )
        except Exception:
            pass  # last_run.json の書き込み失敗はスケジューラーの停止に影響させない
    finally:
        scheduler.stop_scheduler()
        print("DSSMSスケジューラー終了")


if __name__ == "__main__":
    main()
