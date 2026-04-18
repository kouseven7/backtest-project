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
    LAST_RUN_FILE = Path("logs/paper_trade/last_run.json")
    DAILY_SNAPSHOT_FILE = Path("logs/dssms/daily_snapshot.csv")
    GUARD_FILE = Path("logs/dssms/session_guard.json")
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス（None時はデフォルト使用）
        """
        log_dir = Path("logs/dssms")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_filename = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"

        self.logger = setup_logger(
            'dssms.scheduler',
            log_file=str(log_filename)
        )

        self.config = {}
        
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
        self.positions_lock = threading.RLock()
        
        # ポジション管理
        self.positions_file = Path(__file__).parent.parent.parent / "logs" / "dssms" / "positions.json"
        self.max_positions = 3
        self._load_positions()
        self._startup_safety_check()
        
        # スケジュール設定
        self._setup_schedule()
        
        # ペーパートレード残高管理
        self.paper_balance = PaperBalance()
        self.logger.info(
            f"[PAPER_BALANCE] 初期化完了: 残高={self.paper_balance.balance:,.0f}円"
        )
        
        # メール通知システム初期化
        config_file_path = config_path or (Path(__file__).parent.parent.parent / "config" / "kabu_api" / "kabu_connection_config.json")
        self.email_notifier = EmailNotifier(str(config_file_path))
        
        self.logger.info("DSSMSScheduler: 初期化完了")
    
    def _load_positions(self) -> None:
        """positions.jsonからポジション情報を読み込む"""
        try:
            if self.positions_file.exists():
                with open(self.positions_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)

                # --- ゴーストガード ---
                valid = {}
                ghost_found = False
                for sym, pos in raw.items():
                    entry_time = str(pos.get("entry_time", ""))
                    status = pos.get("status")  # 正規ポジションは必ず "open"

                    is_ghost = (
                        entry_time.startswith("2026-04-01")  # 固定日付（テストデータの特徴）
                        or status is None                     # statusフィールドなし
                    )
                    if is_ghost:
                        self.logger.error(
                            f"[GHOST_BLOCKED] {sym}: entry_time={entry_time}, "
                            f"status={status} -> ロードをブロック"
                        )
                        ghost_found = True
                    else:
                        valid[sym] = pos

                self.positions = valid

                # ゴーストが混入していた場合はファイルも即時クリーン化
                # EmergencyDetector が次に読んだ時点でゴーストなしになる
                if ghost_found:
                    self._save_positions()
                    self.logger.error(
                        "[GHOST_PURGED] positions.json からゴーストを除去して上書き保存しました"
                    )
                # --- ゴーストガード終わり ---

                self.logger.info(f"ポジション情報読み込み: {len(self.positions)}件")
            else:
                self.positions = {}
                self.logger.info("positions.json未存在 - 空のポジション辞書で初期化")
        except Exception as e:
            self.logger.error(f"ポジション情報読み込みエラー: {e}")
            self.positions = {}

    def _startup_safety_check(self) -> None:
        """
        スケジューラー起動時の安全確認。
        「本日morning実行済み（session_guard記録あり）なのに
        self.positionsが空」という矛盾を検出して警告する。
        """
        if self.positions:
            # ポジションが存在する場合は正常
            self.logger.info(
                f"[STARTUP_CHECK] positions読み込み完了: {len(self.positions)}件 "
                f"({list(self.positions.keys())})"
            )
            return

        # self.positions が空の場合、session_guard を確認
        morning_executed = self._already_executed_today("morning")
        afternoon_executed = self._already_executed_today("afternoon")

        if morning_executed or afternoon_executed:
            executed_sessions = []
            if morning_executed:
                executed_sessions.append("morning")
            if afternoon_executed:
                executed_sessions.append("afternoon")

            self.logger.warning(
                f"[STARTUP_CHECK] 警告: 本日 {executed_sessions} の screening 実行済み記録が"
                f"ありますが、positions.json が空です。"
                f"ポジションが消失している可能性があります。"
                f"positions.json を確認してください: {self.positions_file}"
            )
        else:
            # session_guard も空 → 正常な初回起動
            self.logger.info("[STARTUP_CHECK] 初回起動または前日以前のセッション。positions空は正常です。")

    def run_manual_screening(self, session: str = "morning") -> None:
        """
        手動screening実行用の安全ラッパー。
        実行前にpositions状態・残高・session_guard状態を表示して確認を求める。

        使用方法（前場）:
            .\.venv-3\Scripts\python.exe -c "
              from src.dssms.dssms_scheduler import DSSMSScheduler
              s = DSSMSScheduler()
              s.run_manual_screening('morning')
            "

        使用方法（後場）:
            .\.venv-3\Scripts\python.exe -c "
              from src.dssms.dssms_scheduler import DSSMSScheduler
              s = DSSMSScheduler()
              s.run_manual_screening('afternoon')
            "
        """
        import json

        if session not in ("morning", "afternoon"):
            print(f"[エラー] 不明なsession: '{session}'。morning または afternoon を指定してください。")
            return

        # 現在の状態を表示
        print("=" * 60)
        print(f"[手動screening実行前確認]  session: {session}")
        print("-" * 60)
        print("【現在のpositions】")
        if self.positions:
            print(json.dumps(self.positions, ensure_ascii=False, indent=2))
        else:
            print("  {} (空) ← ポジションなしと判断されます")
        print(f"【paper_balance】{self.paper_balance.balance:,.0f}円")

        # session_guard の状態を表示
        already = self._already_executed_today(session)
        print(f"【session_guard】本日の {session} 実行済み: {already}")
        if already:
            print(f"  [WARNING] 本日すでに {session} screeningを実行済みです。")
            print("           重複BUYが発生する可能性があります。")
        print("=" * 60)

        # 確認プロンプト
        ans = input(f"この状態で {session} screeningを実行しますか？ (yes/no): ").strip().lower()
        if ans != "yes":
            print("キャンセルしました。")
            return

        # 実行
        if session == "morning":
            self.run_morning_screening(force=True)
        else:
            self.run_afternoon_screening(force=True)

        print(f"[run_manual_screening] {session} screening 完了。")

    def _save_positions(self) -> None:
        """positions.jsonにポジション情報を保存"""
        try:
            self.positions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.positions_file, 'w', encoding='utf-8') as f:
                json.dump(self.positions, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"ポジション情報保存: {len(self.positions)}件")
        except Exception as e:
            self.logger.error(f"ポジション情報保存エラー: {e}")

    def _write_last_run(
        self,
        status: str,
        price_fetch_ok: bool,
        positions_held: int,
        signals_today: dict,
        error_message=None,
    ) -> None:
        """スケジューラーの最終実行状態を last_run.json に書き込む。"""
        import json
        from datetime import datetime as _dt
        data = {
            "last_run_at": _dt.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "status": status,
            "positions_held": positions_held,
            "signals_today": signals_today,
            "price_fetch_ok": price_fetch_ok,
            "error_message": error_message,
            "scheduler_version": "1.0",
        }
        try:
            self.LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
            from pathlib import Path as _Path
            _Path(self.LAST_RUN_FILE).write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            self.logger.info(
                f"[LAST_RUN] last_run.json 書き込み完了: status={status}"
            )
        except Exception as e:
            self.logger.warning(f"[LAST_RUN] 書き込み失敗（処理は継続）: {e}")

    def _write_daily_snapshot(self, price_fetch_ok: bool = True) -> None:
        """daily_snapshot.csv に本日の資産状況を1行記録する（同日はupsert）。"""
        import csv
        from datetime import date as _date
        try:
            # --- R-6: 実値計算ブロック ---
            # execution_history から全レコードを取得
            # ※ 既存の self.execution_history インスタンスを使うこと
            all_records = self.execution_history.get_recent_events(limit=100000)  # 実質全件取得

            # SELLレコードのみ抽出
            sell_records = [r for r in all_records if r.get("event_type") == "sell"]

            # cumulative_realized_pnl: 全SELLの profit_loss 合計
            cumulative_realized_pnl = sum(r.get("profit_loss", 0) or 0 for r in sell_records)

            # daily_pnl: 当日日付のSELLの profit_loss 合計
            today_str = datetime.now().strftime("%Y-%m-%d")
            daily_pnl = sum(
                r.get("profit_loss", 0) or 0
                for r in sell_records
                if str(r.get("timestamp", ""))[:10] == today_str
            )

            # total_trades: 全SELL件数
            total_trades = len(sell_records)

            # unrealized_pnl: 保有ポジション × (現在値 - entry_price) × quantity
            unrealized_pnl = 0.0
            for symbol, pos in self.positions.items():
                try:
                    current_price = self._get_current_price(symbol)
                    entry_price = float(pos.get("entry_price", 0) or 0)
                    quantity = int(pos.get("quantity", 0) or 0)
                    if current_price is not None and current_price > 0:
                        unrealized_pnl += (current_price - entry_price) * quantity
                    else:
                        self.logger.warning(
                            f"[SNAPSHOT] {symbol}: 現在値取得失敗のため unrealized_pnl をスキップ"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"[SNAPSHOT] {symbol}: 現在値取得例外 ({e})、unrealized_pnl=0 として処理"
                    )
            # --- R-6 計算ブロック終わり ---

            self.DAILY_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
            today = str(_date.today())
            headers = [
                'date', 'cash', 'unrealized_pnl', 'total_assets',
                'daily_pnl', 'cumulative_realized_pnl',
                'open_positions', 'total_trades', 'price_fetch_ok'
            ]

            cash = self.paper_balance.balance
            total_assets = cash + unrealized_pnl
            new_row = {
                'date': today,
                'cash': cash,
                'unrealized_pnl': unrealized_pnl,
                'total_assets': total_assets,
                'daily_pnl': daily_pnl,
                'cumulative_realized_pnl': cumulative_realized_pnl,
                'open_positions': len(self.positions),
                'total_trades': total_trades,
                'price_fetch_ok': price_fetch_ok,
            }

            existing_rows = []
            if self.DAILY_SNAPSHOT_FILE.exists():
                with open(self.DAILY_SNAPSHOT_FILE, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_date = str(row.get('date', ''))
                        if row_date != today:
                            existing_rows.append(row)

            existing_rows.append(new_row)
            with open(self.DAILY_SNAPSHOT_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(existing_rows)

            self.logger.info(
                f"[SNAPSHOT] {today} 記録完了: daily_pnl={daily_pnl:,.0f}, "
                f"cumulative_realized_pnl={cumulative_realized_pnl:,.0f}, "
                f"total_trades={total_trades}, unrealized_pnl={unrealized_pnl:,.0f}"
            )
        except Exception as e:
            self.logger.warning(f"[SNAPSHOT] 書き込み失敗: {e}")

    def _write_closed_trade(self, symbol: str, exit_price: float,
                             exit_reason: str) -> None:
        """SELL成功時にclosed_trades.csvへ1行追記する"""
        from datetime import date, datetime
        import csv
        from pathlib import Path
        try:
            position = self.positions.get(symbol)
            if position is None:
                self.logger.warning(
                    f"[CLOSED_TRADE] {symbol}のpositionデータが見つかりません"
                )
                return

            entry_date_str = position.get('entry_date', position.get('entry_time', ''))
            entry_price = position.get('entry_price', 0)
            shares = position.get('quantity', position.get('shares', 0))
            strategy = position.get('strategy', '')
            exit_date_str = str(date.today())

            # holding_days計算
            try:
                entry_dt = datetime.strptime(entry_date_str[:10], '%Y-%m-%d').date()
                holding_days = (date.today() - entry_dt).days
            except Exception:
                holding_days = 0

            # 損益計算
            pnl = (exit_price - entry_price) * shares
            pnl_pct = ((exit_price - entry_price) / entry_price * 100
                       if entry_price > 0 else 0.0)
            win = 1 if pnl > 0 else 0

            # trade_id：既存行数+1
            closed_trades_file = Path('logs/paper_trade/closed_trades.csv')
            closed_trades_file.parent.mkdir(parents=True, exist_ok=True)

            write_header = not closed_trades_file.exists()
            trade_id = 1
            if not write_header:
                with open(closed_trades_file, 'r', encoding='utf-8') as f:
                    trade_id = sum(1 for _ in f)  # ヘッダー含む行数=次のtrade_id

            with open(closed_trades_file, 'a', newline='',
                      encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'trade_id', 'symbol', 'strategy',
                        'entry_date', 'entry_price', 'shares',
                        'exit_date', 'exit_price', 'exit_reason',
                        'holding_days', 'pnl', 'pnl_pct', 'win'
                    ])
                    trade_id = 1
                writer.writerow([
                    trade_id, symbol, strategy,
                    entry_date_str, entry_price, shares,
                    exit_date_str, exit_price, exit_reason,
                    holding_days, round(pnl, 2),
                    round(pnl_pct, 4), win
                ])

            self.logger.info(
                f"[CLOSED_TRADE] 記録完了: {symbol} "
                f"exit={exit_price} pnl={pnl:+,.0f}円 reason={exit_reason}"
            )

        except Exception as e:
            self.logger.warning(f"[CLOSED_TRADE] 書き込み失敗: {e}")

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """銘柄の現在値を取得する。取得不可時はNoneを返す。"""
        # 1) kabu API
        try:
            if self.kabu_integration is not None:
                kabu_price = self.kabu_integration.get_current_price(symbol)
                if kabu_price is not None and kabu_price > 0:
                    return float(kabu_price)
        except Exception as e:
            self.logger.warning(f"[SNAPSHOT] {symbol}: kabu API価格取得失敗 ({e})")

        # 2) yfinance via data_fetcher
        try:
            from datetime import timedelta
            _, _, _, stock_data, _ = get_parameters_and_data(
                ticker=symbol,
                start_date=(date.today() - timedelta(days=5)).strftime('%Y-%m-%d'),
                end_date=date.today().strftime('%Y-%m-%d'),
                warmup_days=0
            )
            if stock_data is not None and not stock_data.empty:
                yf_price = None
                if 'Adj Close' in stock_data.columns:
                    yf_price = float(stock_data['Adj Close'].iloc[-1])
                elif 'Close' in stock_data.columns:
                    yf_price = float(stock_data['Close'].iloc[-1])
                if yf_price is not None and yf_price > 0:
                    return yf_price
                else:
                    self.logger.warning(
                        f"[SNAPSHOT] {symbol}: yfinance価格が0または無効 ({yf_price})"
                    )
        except Exception as e:
            self.logger.warning(f"[SNAPSHOT] {symbol}: yfinance価格取得失敗 ({e})")

        return None

    def _already_executed_today(self, session: str) -> bool:
        """当日の指定セッションが実行済みかどうかを返す。"""
        # 設計方針:
        # - force=True で手動実行した場合もガードを通す
        # - 意図的な再実行が必要な場合は session_guard.json を手動削除して対応
        # - キーに日付を含むため翌日は自動的に無効になる
        today = datetime.now().strftime("%Y-%m-%d")
        key = f"{today}:{session}"
        try:
            if self.GUARD_FILE.exists():
                with open(self.GUARD_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get(key, False)
        except Exception as e:
            self.logger.warning(f"[SESSION_GUARD] 読み込み失敗: {e}")
        return False

    def _mark_executed_today(self, session: str) -> None:
        """当日の指定セッションを実行済みとして記録する。"""
        today = datetime.now().strftime("%Y-%m-%d")
        key = f"{today}:{session}"
        try:
            self.GUARD_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if self.GUARD_FILE.exists():
                with open(self.GUARD_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data[key] = True
            with open(self.GUARD_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[SESSION_GUARD] 記録: {key}")
        except Exception as e:
            self.logger.error(f"[SESSION_GUARD] 書き込み失敗: {e}")
    
    def _add_position(self, symbol: str, entry_price: float = 0.0,
                      strategy: str = "GCStrategy", quantity: int = 100,
                      entry_idx: Optional[int] = None) -> None:
        """
        ポジションを追加
        
        Args:
            symbol: 銘柄コード
            entry_price: エントリー価格（order_resultのexecuted_priceから取得）
            strategy: 使用戦略名（デフォルト: GCStrategy）
            quantity: 取引株数（デフォルト: 100株）
        """
        if symbol in self.positions:
            self.logger.warning(
                f"[ADD_POSITION_SKIP] {symbol} は既にポジション保有中のためBUYをスキップします "
                f"(既存entry_price={self.positions[symbol].get('entry_price', 'N/A')}円)"
            )
            return
        with self.positions_lock:
            self.positions[symbol] = {
                "symbol": symbol,
                "entry_time": datetime.now().isoformat(),
                "entry_price": entry_price,
                "entry_idx": entry_idx,
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
            symbol=order_params.get("symbol", "unknown"),
            order_type=order_params.get("side", "unknown"),
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
            current_price = self._get_current_price(symbol)
            if current_price is None or current_price <= 0:
                current_price = float(
                    self.positions.get(symbol, {}).get("entry_price", 0) or 0
                )
            if current_price <= 0:
                current_price = 1000.0
            self.logger.info(
                f"[PAPER_SELL] {symbol}: SL ペーパー約定 "
                f"(debug_mode=True, price={current_price:.0f}円, qty={quantity}株)"
            )
            return {
                "success": True,
                "executed_price": current_price,
                "order_id": f"PAPER_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "paper": True
            }

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
            count = len(self.positions)
            max_pos = self.max_positions
            self.logger.info(f"ポジション削除: {symbol} ({count}/{max_pos})")
    
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

                if position.get('entry_idx') is None:
                    self.logger.warning(
                        f"[R5_FALLBACK] {symbol}: entry_idx が未保存（旧形式 or 取得失敗）。"
                        f"days_held 計算は GCStrategy 側フォールバックに委ねます。"
                        f"次回エントリーから自動修正されます。"
                    )
                
                try:
                    strategy = GCStrategy(
                        data=stock_data,
                        ticker=symbol,
                        params={
                            "short_window": 5,
                            "long_window": 75,
                            "stop_loss": 0.03,
                            "trailing_stop_pct": None,
                            "trend_strength_percentile": 60,
                            "sma_divergence_threshold": 3.0,
                        }
                    )
                    strategy.initialize_strategy()  # 戦略の初期化
                    self.logger.info(f"[EXIT_CHECK] {symbol}: {strategy_name}インスタンス化成功")
                except Exception as e:
                    self.logger.warning(f"[EXIT_CHECK] {symbol}: 戦略インスタンス化失敗 → スキップ: {e}")
                    continue
                
                # Step 3: existing_position構築（entry_timeをentry_dateに変換）
                try:
                    entry_time_str = position.get('entry_date', position.get('entry_time', ''))
                    if isinstance(entry_time_str, str) and entry_time_str:
                        entry_date = pd.Timestamp(entry_time_str)
                    else:
                        entry_date = pd.Timestamp(entry_time_str) if entry_time_str else pd.Timestamp('today')

                    if pd.isna(entry_date):
                        self.logger.warning(f"[EXIT_CHECK] {symbol}: entry_dateがNaT → 本日日付で代替")
                        entry_date = pd.Timestamp('today')
                    
                    existing_position = {
                        'symbol': symbol,
                        'entry_price': position.get('entry_price', 0.0),
                        'entry_date': entry_date,
                        'entry_idx': position.get('entry_idx'),
                        'quantity': position.get('quantity', 100),
                        'force_close': False  # 通常エグジット判定
                    }
                    
                    entry_date_str = existing_position['entry_date'].strftime('%Y-%m-%d') if not pd.isna(existing_position['entry_date']) else 'unknown'
                    self.logger.info(f"[EXIT_CHECK] {symbol}: existing_position構築成功 (entry_price={existing_position['entry_price']}, entry_date={entry_date_str})")
                    
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
                        
                        self.logger.info(f"[EXIT_SIGNAL] {symbol}: SELL発注 (price={exit_price}, shares={exit_shares}, reason={exit_reason})")
                        
                        # SELL発注実行（_execute_sl_sell()を使用）
                        sell_result = self._execute_sl_sell(symbol=symbol, quantity=exit_shares)
                        if sell_result.get("success"):
                            self.logger.info(
                                f"[EXIT_SELL_OK] {symbol}: SELL発注成功 "
                                f"(order_id={sell_result.get('order_id')})"
                            )
                            # SELL時の残高加算
                            if exit_price > 0 and exit_shares > 0:
                                new_balance = self.paper_balance.add(exit_price, exit_shares)
                                self.logger.info(
                                    f"[BALANCE_ADD] {symbol} (EXIT): "
                                    f"+{exit_price * exit_shares:,.0f}円 -> 残高={new_balance:,.0f}円"
                                )

                            # ポジション削除
                            self._write_closed_trade(symbol, exit_price, exit_reason=exit_reason)
                            self._remove_position(symbol)
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
                            self.logger.error(
                                f"[EXIT_SELL_NG] {symbol}: SELL発注失敗 - "
                                f"ポジション維持・残高変更なし。"
                                f"error={sell_result.get('error', 'unknown')}"
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
            # 前回インスタンスのジョブ蓄積を防ぐためクリア
            schedule.clear()
            self.logger.info("[SCHEDULE_CLEAR] グローバルscheduleをクリアしました")

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
    
    def run_morning_screening(self, force: bool = False) -> str:
        """
        09:30前場スクリーニング実行
        
        Returns:
            str: 選択された銘柄コード（エラー時は空文字）
        """
        start_time = datetime.now()
        self.logger.info("=== 前場スクリーニング開始 ===")

        # --- session guard ---
        if self._already_executed_today("morning"):
            self.logger.warning("[SESSION_GUARD] morning already executed today. skip.")
            return ""
        # --- /session guard ---
        
        selected_symbol = ""
        error_occurred = False
        error_message = ""
        price_fetch_ok = True
        screening_result = {}
        
        try:
            # 市場時間チェック
            if not force and not self.market_time_manager.should_run_screening("morning"):
                self.logger.warning("前場スクリーニング実行条件未満足")
                return ""
            
            # --- エグジット判定（追加） ---
            exited_symbols = self._check_exit_for_positions()
            if exited_symbols:
                self.logger.info(f"[EXIT] {len(exited_symbols)}銘柄をSELL: {exited_symbols}")
            # --- ここまで追加 ---
            
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
                    else:
                        stock_data = None
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
                                if stock_data.empty:
                                    self.logger.error(
                                        f"[PRICE_ERROR] {symbol}: kabu API・yfinanceともに価格取得失敗"
                                        f"(yfinanceが空データを返しました)。前場BUYをスキップします。"
                                    )
                                    return
                                current_price = stock_data['Adj Close'].iloc[-1]
                                self.logger.info(f"[PRICE_FETCH] {symbol}: yfinance価格={current_price:.2f}円（フォールバック）")
                            except Exception as e:
                                self.logger.error(
                                    f"[PRICE_ERROR] {symbol}: kabu API・yfinanceともに価格取得失敗。"
                                    f"前場BUYをスキップします。理由: {e}"
                                )
                                return

                        # バックテスト側の shares を100株単位に安全丸め（base_strategy暫定実装の補正）
                        raw_shares = screening_result.get('shares', None)
                        if raw_shares is not None:
                            lot_shares = (int(raw_shares) // 100) * 100
                            if lot_shares == 0:
                                self.logger.warning(
                                    f"[LOT_GUARD] 単元未満のためエントリースキップ: "
                                    f"銘柄={symbol}, 株価={current_price:.0f}円, "
                                    f"算出株数={raw_shares}株 → 100株単位={lot_shares}株"
                                )
                            else:
                                screening_result['shares'] = lot_shares
                        
                        
                        # C-2: 動的株数計算
                        if raw_shares is not None and screening_result.get('shares', 0) == 0:
                            quantity = 0
                        else:
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
                            # 重複BUYチェック（注文前に確認）
                            if symbol in self.positions:
                                self.logger.warning(f"[DUPLICATE_BUY_GUARD] {symbol} は既にポジションあり。BUYをスキップします。")
                            else:
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
                                    entry_idx = None
                                    entry_date = pd.Timestamp(date.today())
                                    if stock_data is not None and not stock_data.empty:
                                        try:
                                            entry_idx = stock_data.index.get_loc(entry_date)
                                            if isinstance(entry_idx, slice):
                                                entry_idx = entry_idx.start
                                            elif hasattr(entry_idx, '__iter__'):
                                                entry_idx = int(list(entry_idx).index(True))
                                        except (KeyError, TypeError, ValueError):
                                            entry_idx = None
                                            self.logger.warning(
                                                f"[R5] {symbol}: entry_idx 取得失敗、None で保存します"
                                            )
                                    else:
                                        self.logger.warning(
                                            f"[R5] {symbol}: stock_data 未参照のため entry_idx=None で保存します"
                                        )
                                    # ポジション追加
                                    self._add_position(
                                        symbol=symbol,
                                        entry_price=executed_price,
                                        strategy="GCStrategy",  # TODO: DynamicStrategySelectorで動的選択
                                        quantity=quantity,
                                        entry_idx=entry_idx
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

        # 朝サマリーメール送信
        try:
            from datetime import datetime as _dt
            hours_since_last_run = 0.0
            try:
                if self.LAST_RUN_FILE.exists():
                    import json as _json
                    last_run_data = _json.loads(
                        self.LAST_RUN_FILE.read_text(encoding='utf-8-sig')
                    )
                    last_run_at_str = last_run_data.get('last_run_at', '')
                    if last_run_at_str:
                        from datetime import datetime as _dt2
                        last_run_at = _dt2.fromisoformat(last_run_at_str)
                        hours_since_last_run = (
                            _dt2.now() - last_run_at
                        ).total_seconds() / 3600
                        self.logger.info(
                            f"[HEALTH_CHECK] 前回実行からの経過時間: "
                            f"{hours_since_last_run:.1f}時間"
                        )
                else:
                    self.logger.info(
                        "[HEALTH_CHECK] last_run.jsonが存在しないため経過時間=0.0として処理"
                    )
            except Exception as e:
                self.logger.warning(f"[HEALTH_CHECK] 経過時間計算失敗: {e}")
                hours_since_last_run = 0.0

            # 含み損益をリアルタイム計算
            unrealized_pnl_calc = 0.0
            positions_for_email = []
            for sym, pos in self.positions.items():
                try:
                    cur_price = self._get_current_price(sym)
                    ep = float(pos.get("entry_price", 0) or 0)
                    qty = int(pos.get("quantity", 0) or 0)
                    pos_unrealized = (cur_price - ep) * qty if cur_price and ep and qty else 0
                    unrealized_pnl_calc += pos_unrealized
                    positions_for_email.append({
                        'symbol': sym,
                        'price': cur_price or ep,
                        'shares': qty,
                        'unrealized_pnl': pos_unrealized,
                    })
                except Exception:
                    ep = float(pos.get("entry_price", 0) or 0)
                    qty = int(pos.get("quantity", 0) or 0)
                    positions_for_email.append({
                        'symbol': sym,
                        'price': ep,
                        'shares': qty,
                        'unrealized_pnl': 0,
                    })

            summary_data = {
                'execution_time': _dt.now().strftime('%H:%M'),
                'status': '異常' if error_occurred else '正常',
                'error_message': error_message if error_occurred else '',
                'cash_balance': self.paper_balance.balance,
                'unrealized_pnl': unrealized_pnl_calc,
                'total_assets': self.paper_balance.balance,
                'daily_pnl': 0,
                'positions': positions_for_email,
                'screened_symbols': [selected_symbol] if selected_symbol else [],
                'hours_since_last_run': hours_since_last_run
            }
            self.email_notifier.send_morning_summary(summary_data)
        except Exception as e:
            self.logger.warning(f"[EMAIL_SKIP] 朝サマリーメール送信失敗: {e}")

        # daily_snapshot.csv に記録
        self._write_daily_snapshot(price_fetch_ok=True)

        # last_run.json に記録
        self._write_last_run(
            status="success" if not error_occurred else "error",
            price_fetch_ok=price_fetch_ok,
            positions_held=len(self.positions),
            signals_today={"buy": 0, "sell": 0, "hold": 0},
            error_message=error_message if error_occurred else None,
        )

        if not error_occurred:
            self._mark_executed_today("morning")
        
        return selected_symbol
    
    def run_afternoon_screening(self, force: bool = False) -> str:
        """
        12:30後場スクリーニング実行
        
        Returns:
            str: 選択された銘柄コード（エラー時は空文字）
        """
        start_time = datetime.now()
        self.logger.info("=== 後場スクリーニング開始 ===")
        screening_start_time = datetime.now()

        # HEALTH_CHECK: 前回実行からの経過時間を先行計算（_write_last_run より前に取得）
        hours_since_last_run_af = 0.0
        try:
            if self.LAST_RUN_FILE.exists():
                import json as _json_af_pre
                _last_run_data_af = _json_af_pre.loads(
                    self.LAST_RUN_FILE.read_text(encoding='utf-8-sig')
                )
                _last_run_at_str_af = _last_run_data_af.get('last_run_at', '')
                if _last_run_at_str_af:
                    from datetime import datetime as _dt_af_pre
                    _last_run_at_af = _dt_af_pre.fromisoformat(_last_run_at_str_af)
                    hours_since_last_run_af = (
                        _dt_af_pre.now() - _last_run_at_af
                    ).total_seconds() / 3600
                    self.logger.info(
                        f"[HEALTH_CHECK] 前回実行からの経過時間: "
                        f"{hours_since_last_run_af:.1f}時間"
                    )
            else:
                self.logger.info(
                    "[HEALTH_CHECK] last_run.jsonが存在しないため経過時間=0.0として処理"
                )
        except Exception as _e_af:
            self.logger.warning(f"[HEALTH_CHECK] 経過時間計算失敗: {_e_af}")
            hours_since_last_run_af = 0.0

        # --- session guard ---
        if self._already_executed_today("afternoon"):
            self.logger.warning("[SESSION_GUARD] afternoon already executed today. skip.")
            return ""
        # --- /session guard ---
        
        selected_symbol = ""
        error_occurred = False
        error_message = ""
        price_fetch_ok = True
        screening_result = {}
        
        try:
            # 市場時間チェック
            if not force and not self.market_time_manager.should_run_screening("afternoon"):
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
                    else:
                        stock_data = None
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
                                if stock_data.empty:
                                    self.logger.error(
                                        f"[PRICE_ERROR] {symbol}: kabu API・yfinanceともに価格取得失敗"
                                        f"(yfinanceが空データを返しました)。後場BUYをスキップします。"
                                    )
                                    return
                                current_price = stock_data['Adj Close'].iloc[-1]
                                self.logger.info(f"[PRICE_FETCH] {symbol}: yfinance価格={current_price:.2f}円（フォールバック）")
                            except Exception as e:
                                self.logger.error(
                                    f"[PRICE_ERROR] {symbol}: kabu API・yfinanceともに価格取得失敗。"
                                    f"後場BUYをスキップします。理由: {e}"
                                )
                                return

                        # バックテスト側の shares を100株単位に安全丸め（base_strategy暫定実装の補正）
                        raw_shares = screening_result.get('shares', None)
                        if raw_shares is not None:
                            lot_shares = (int(raw_shares) // 100) * 100
                            if lot_shares == 0:
                                self.logger.warning(
                                    f"[LOT_GUARD] 単元未満のためエントリースキップ: "
                                    f"銘柄={symbol}, 株価={current_price:.0f}円, "
                                    f"算出株数={raw_shares}株 → 100株単位={lot_shares}株"
                                )
                            else:
                                screening_result['shares'] = lot_shares
                        
                        # C-2: 動的株数計算
                        if raw_shares is not None and screening_result.get('shares', 0) == 0:
                            quantity = 0
                        else:
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
                            # 重複BUYチェック（注文前に確認）
                            if symbol in self.positions:
                                self.logger.warning(f"[DUPLICATE_BUY_GUARD] {symbol} は既にポジションあり。BUYをスキップします。")
                            else:
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
                                    entry_idx = None
                                    entry_date = pd.Timestamp(date.today())
                                    if stock_data is not None and not stock_data.empty:
                                        try:
                                            entry_idx = stock_data.index.get_loc(entry_date)
                                            if isinstance(entry_idx, slice):
                                                entry_idx = entry_idx.start
                                            elif hasattr(entry_idx, '__iter__'):
                                                entry_idx = int(list(entry_idx).index(True))
                                        except (KeyError, TypeError, ValueError):
                                            entry_idx = None
                                            self.logger.warning(
                                                f"[R5] {symbol}: entry_idx 取得失敗、None で保存します"
                                            )
                                    else:
                                        self.logger.warning(
                                            f"[R5] {symbol}: stock_data 未参照のため entry_idx=None で保存します"
                                        )
                                    # ポジション追加
                                    self._add_position(
                                        symbol=symbol,
                                        entry_price=executed_price,
                                        strategy="GCStrategy",  # TODO: DynamicStrategySelectorで動的選択
                                        quantity=quantity,
                                        entry_idx=entry_idx
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
        finally:
            # daily_snapshot.csv に記録（後場完了時）
            self._write_daily_snapshot(price_fetch_ok=price_fetch_ok)
            self._write_last_run(
                status="success" if not error_occurred else "error",
                price_fetch_ok=price_fetch_ok,
                positions_held=len(self.positions),
                signals_today={"buy": 0, "sell": 0, "hold": 0},
                error_message=error_message if error_occurred else None,
            )
        
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
            self._mark_executed_today("afternoon")

        # 後場サマリーメール送信
        try:
            # HEALTH_CHECK の経過時間はメソッド先頭で取得済みの値を使用
            # （_write_last_run より前に読み込んだ値のため正確）

            # 含み損益をリアルタイム計算
            unrealized_pnl_calc = 0.0
            positions_for_email = []
            for sym, pos in self.positions.items():
                try:
                    cur_price = self._get_current_price(sym)
                    ep = float(pos.get("entry_price", 0) or 0)
                    qty = int(pos.get("quantity", 0) or 0)
                    pos_unrealized = (cur_price - ep) * qty if cur_price and ep and qty else 0
                    unrealized_pnl_calc += pos_unrealized
                    positions_for_email.append({
                        'symbol': sym,
                        'price': cur_price or ep,
                        'shares': qty,
                        'unrealized_pnl': pos_unrealized,
                    })
                except Exception:
                    ep = float(pos.get("entry_price", 0) or 0)
                    qty = int(pos.get("quantity", 0) or 0)
                    positions_for_email.append({
                        'symbol': sym,
                        'price': ep,
                        'shares': qty,
                        'unrealized_pnl': 0,
                    })

            summary_data = {
                'execution_time': datetime.now().strftime('%H:%M'),
                'status': '異常' if error_occurred else '正常',
                'error_message': error_message if error_occurred else '',
                'cash_balance': self.paper_balance.balance,
                'unrealized_pnl': unrealized_pnl_calc,
                'total_assets': self.paper_balance.balance,
                'daily_pnl': 0,
                'positions': positions_for_email,
                'screened_symbols': [selected_symbol] if selected_symbol else [],
                'hours_since_last_run': hours_since_last_run_af
            }
            self.email_notifier.send_afternoon_summary(summary_data)
        except Exception as e:
            self.logger.warning(f"[EMAIL_SKIP] 後場サマリーメール送信失敗: {e}")
        
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
            # 最新のpositions.jsonを再読み込み（ゴーストガード込み）
            self._load_positions()

            # 保有ポジションがない場合はスキップ
            if not self.positions:
                self.logger.debug("[EMERGENCY_CHECK] 保有ポジションなし、スキップ")
                return
            
            # 市場開場中のみ実行
            if not self.market_time_manager.is_market_open():
                return
            
            symbols_to_check = list(self.positions.keys())
            self.logger.info(
                f"[EMERGENCY_CHECK] 監視銘柄数: {len(symbols_to_check)}, "
                f"銘柄: {symbols_to_check}"
            )

            # 監視対象は positions.json 由来の現在保有ポジションのみとする
            for symbol in symbols_to_check:
                # ループ中に削除済みの銘柄はSLチェック対象外
                if symbol not in self.positions:
                    self.logger.info(
                        f"[SL_SKIP] {symbol}: positions削除済みのためSLチェックをスキップ"
                        f"（通常EXITと競合の可能性）"
                    )
                    continue

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
                        stop_loss_details = emergency_result.get("stop_loss_details", {})
                        entry_price = stop_loss_details.get("entry_price", "N/A")
                        current_price = stop_loss_details.get("current_price", "N/A")
                        loss_percentage = stop_loss_details.get("loss_percentage", 0.0)
                        quantity = self.positions.get(symbol, {}).get("quantity", 100)

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
                            # SELL時の残高加算（約定価格はcurrent_priceを使用、取得できない場合はentry_priceをフォールバック）
                            sell_price = current_price if isinstance(current_price, (int, float)) and current_price > 0 else self.positions.get(symbol, {}).get("entry_price", 0)
                            if sell_price > 0 and quantity > 0:
                                new_balance = self.paper_balance.add(sell_price, quantity)
                                self.logger.info(
                                    f"[BALANCE_ADD] {symbol} (SL): "
                                    f"+{sell_price * quantity:,.0f}円 -> 残高={new_balance:,.0f}円"
                                )

                            # ポジション削除
                            self._write_closed_trade(symbol, sell_price, exit_reason='stop_loss')
                            self._remove_position(symbol)
                            self.logger.warning(f"[SL_EXECUTED] {symbol}: ポジション削除完了")
                        else:
                            self.logger.error(
                                f"[SL_SELL_NG] {symbol}: SELL発注失敗 - "
                                f"ポジション維持・残高変更なし。"
                                f"error={sell_result.get('error', 'unknown')}"
                            )
                        # --- C-1 ここまで ---
                        
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
                candidates = []
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
        # DISABLED: ペーパートレード段階では緊急切替を無効化
        # SL発動後の新銘柄BUYがゴーストポジション問題を引き起こすため
        # 次回スクリーニングサイクルで新銘柄を選択する運用とする
        self.logger.info(f"[EMERGENCY_SWITCH_DISABLED] {symbol}: 緊急切替は無効化中")
        return

        try:
            self.logger.warning(f"緊急切替実行: {symbol}")
            
            # インテリジェント切替実行
            if self.intelligent_switch:
                try:
                    # 切替候補取得
                    backup_candidates = self.hierarchical_ranking.get_backup_candidates(n=3) if self.hierarchical_ranking else []
                    
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
                                    # 旧銘柄SELL
                                    sell_result = self.kabu_integration.execute_dynamic_orders({
                                        'symbol': symbol,
                                        'side': '1',
                                        'quantity': 100,
                                        'price': 0
                                    })
                                    self.logger.info(f"緊急SELL注文: {symbol} 結果={sell_result}")
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
            result = self.run_morning_screening(force=True)
            self.logger.info(f"前場スクリーニングジョブ完了: {result}")
        except Exception as e:
            self.logger.error(f"前場スクリーニングジョブエラー: {e}")
    
    def _run_afternoon_screening_job(self):
        """後場スクリーニングジョブ（内部）"""
        try:
            result = self.run_afternoon_screening(force=True)
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
    finally:
        scheduler.stop_scheduler()
        print("DSSMSスケジューラー終了")


if __name__ == "__main__":
    main()
