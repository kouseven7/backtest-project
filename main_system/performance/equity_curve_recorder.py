"""
ポートフォリオ損益推移記録システム - Phase 5-B-5

リスク管理デバッグ用に、ポートフォリオの時系列推移を詳細に記録し、
CSV形式で出力してドローダウン計算の妥当性検証を容易にする。

主な機能:
- 取引実行時・日次終値時のスナップショット記録（Q1: C案）
- リスク状態の追跡（DrawdownControllerから取得、Q2: 案1）
- 当日ブロック取引数の累積記録（Q3: A案）
- CSV出力（Excel禁止、copilot-instructions.md準拠）
- 実データのみ記録（モック/ダミーデータ禁止）

統合コンポーネント:
- PaperBroker: ポートフォリオ価値・現金残高・ポジション情報取得
- DrawdownController: ドローダウン計算・リスク状態取得
- StrategyExecutionManager: 取引実行時のスナップショット記録
- IntegratedExecutionManager: リスクブロック時の記録更新
- ComprehensiveReporter: CSV出力連携

セーフティ機能/注意事項:
- データ取得失敗時はエラーログ、フォールバック禁止
- 同一タイムスタンプの重複記録を防止
- CSV出力時の型変換エラー処理
- copilot-instructions.md準拠: 実データのみ記録

Author: Backtest Project Team
Created: 2025-11-07
Last Modified: 2025-11-07
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class EquityCurveRecorder:
    """
    ポートフォリオ損益推移記録クラス
    
    Phase 5-B-5仕様:
    - Q1: C案（取引時+日次終値の両方）
    - Q2: 案1（DrawdownControllerから取得）
    - Q3: A案（当日累積）
    """
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # スナップショット記録
        self.snapshots: List[Dict[str, Any]] = []
        
        # 重複記録防止用
        self.last_snapshot_time: Optional[datetime] = None
        
        # 当日のブロック取引数追跡
        self.current_date: Optional[str] = None
        self.daily_blocked_count: int = 0
        
        self.logger.info("[EQUITY_CURVE] EquityCurveRecorder initialized")
    
    def record_snapshot(
        self,
        date: datetime,
        portfolio_value: float,
        cash_balance: float,
        position_value: float,
        peak_value: float,
        drawdown_pct: float,
        cumulative_pnl: float,
        daily_pnl: float,
        total_trades: int,
        active_positions: int,
        risk_status: str,
        blocked_trades: int = 0,
        risk_action: str = "NONE",
        snapshot_type: str = "TRADE"  # TRADE or DAILY
    ) -> None:
        """
        ポートフォリオスナップショット記録
        
        copilot-instructions.md準拠:
        - 実データのみ記録（モック/ダミー禁止）
        - データ検証実施
        
        Args:
            date: 記録日時
            portfolio_value: ポートフォリオ総額（現金+ポジション時価）
            cash_balance: 現金残高
            position_value: 保有ポジション時価評価額
            peak_value: これまでの最高資産額
            drawdown_pct: 現在のドローダウン（0.0-1.0）
            cumulative_pnl: 累積損益
            daily_pnl: 当日損益
            total_trades: 累積取引数
            active_positions: 現在の保有ポジション数
            risk_status: リスク状態（DrawdownControllerから取得）
            blocked_trades: 当日ブロックされた取引数（累積）
            risk_action: 実行されたリスク管理アクション
            snapshot_type: スナップショットタイプ（TRADE/DAILY）
        
        Raises:
            ValueError: データ検証失敗時
        """
        try:
            # データ検証（copilot-instructions.md: 実データ確認）
            if portfolio_value < 0:
                raise ValueError(f"Invalid portfolio_value: {portfolio_value}")
            
            if peak_value < 0:
                raise ValueError(f"Invalid peak_value: {peak_value}")
            
            if not 0.0 <= drawdown_pct <= 1.0:
                self.logger.warning(f"Unusual drawdown_pct: {drawdown_pct:.2%} (expected 0-100%)")
            
            # 重複記録防止（同一タイムスタンプ）
            if self.last_snapshot_time == date:
                self.logger.debug(f"[EQUITY_CURVE] Skipping duplicate snapshot: {date}")
                return
            
            # 日付変更時にブロック数リセット
            date_str = date.strftime('%Y-%m-%d')
            if self.current_date != date_str:
                self.current_date = date_str
                self.daily_blocked_count = 0
            
            # ブロック数更新（累積）
            self.daily_blocked_count += blocked_trades
            
            # スナップショット作成
            snapshot = {
                'date': date,
                'portfolio_value': float(portfolio_value),
                'cash_balance': float(cash_balance),
                'position_value': float(position_value),
                'peak_value': float(peak_value),
                'drawdown_pct': float(drawdown_pct),
                'cumulative_pnl': float(cumulative_pnl),
                'daily_pnl': float(daily_pnl),
                'total_trades': int(total_trades),
                'active_positions': int(active_positions),
                'risk_status': str(risk_status),
                'blocked_trades': int(self.daily_blocked_count),
                'risk_action': str(risk_action),
                'snapshot_type': str(snapshot_type)
            }
            
            self.snapshots.append(snapshot)
            self.last_snapshot_time = date
            
            self.logger.debug(
                f"[EQUITY_CURVE] Snapshot recorded: {date} | "
                f"Portfolio={portfolio_value:,.0f} | DD={drawdown_pct:.2%} | "
                f"Risk={risk_status} | Type={snapshot_type}"
            )
            
        except Exception as e:
            self.logger.error(f"[EQUITY_CURVE] Failed to record snapshot: {e}")
            # copilot-instructions.md準拠: フォールバック禁止、エラー時は記録スキップ
            raise
    
    def update_blocked_trades(self, date: datetime, count: int = 1) -> None:
        """
        ブロック取引数を更新（Q3: A案 - 当日累積）
        
        IntegratedExecutionManagerからリスクブロック時に呼び出される
        
        Args:
            date: 記録日時
            count: 追加するブロック数（デフォルト1）
        """
        try:
            date_str = date.strftime('%Y-%m-%d')
            
            # 日付変更時にリセット
            if self.current_date != date_str:
                self.current_date = date_str
                self.daily_blocked_count = 0
            
            self.daily_blocked_count += count
            
            self.logger.debug(
                f"[EQUITY_CURVE] Blocked trades updated: {date_str} | "
                f"Count={self.daily_blocked_count}"
            )
            
        except Exception as e:
            self.logger.error(f"[EQUITY_CURVE] Failed to update blocked trades: {e}")
    
    def export_to_csv(self, output_path: str) -> str:
        """
        CSV出力（copilot-instructions.md準拠: Excel禁止）
        
        Args:
            output_path: 出力ファイルパス
        
        Returns:
            str: 出力ファイルパス
        
        Raises:
            ValueError: スナップショットが空の場合
        """
        try:
            if not self.snapshots:
                raise ValueError("No snapshots to export")
            
            # DataFrameに変換
            df = pd.DataFrame(self.snapshots)
            
            # 日付フォーマット
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 列順序を明示的に指定
            columns_order = [
                'date',
                'portfolio_value',
                'cash_balance',
                'position_value',
                'peak_value',
                'drawdown_pct',
                'cumulative_pnl',
                'daily_pnl',
                'total_trades',
                'active_positions',
                'risk_status',
                'blocked_trades',
                'risk_action',
                'snapshot_type'
            ]
            
            df = df[columns_order]
            
            # CSV出力（UTF-8 BOM付き、Excel対応）
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            self.logger.info(
                f"[EQUITY_CURVE] CSV exported: {output_path} | "
                f"Snapshots={len(df)} | "
                f"Date range={df['date'].min()} to {df['date'].max()}"
            )
            
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"[EQUITY_CURVE] CSV export failed: {e}")
            raise
    
    def get_snapshot_count(self) -> int:
        """記録済みスナップショット数を取得"""
        return len(self.snapshots)
    
    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """最新スナップショットを取得"""
        return self.snapshots[-1] if self.snapshots else None
    
    def clear(self) -> None:
        """記録をクリア"""
        self.snapshots.clear()
        self.last_snapshot_time = None
        self.current_date = None
        self.daily_blocked_count = 0
        self.logger.info("[EQUITY_CURVE] Snapshots cleared")
