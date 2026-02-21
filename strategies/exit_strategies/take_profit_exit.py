"""
TakeProfitExit - 固定利確エグジット戦略

エントリー価格から指定利益率に達したら決済する単純なエグジット戦略。

主な機能:
- 固定利益率到達での決済（例: 10%, 15%, 20%）
- エントリー価格基準の利益判定
- 日次バックテスト対応
- ルックアヘッドバイアス防止（idx日目判断→idx+1日目執行）

統合コンポーネント:
- BaseExitStrategy継承
- GCStrategyWithExit等のエントリー戦略と組み合わせ
- CompositeExit経由での組み合わせ戦略対応

セーフティ機能/注意事項:
- パラメータは利益率のみ（過学習回避）
- 市場トレンド非考慮（Phase 3で対応予定）
- エグジット価格はidx+1日目始値
- フォールバック禁止（copilot-instructions.md準拠）

Author: Backtest Project Team
Created: 2026-01-22
Last Modified: 2026-01-22
"""
from typing import Dict, Tuple
import pandas as pd
from .base_exit_strategy import BaseExitStrategy


class TakeProfitExit(BaseExitStrategy):
    """固定利確エグジット戦略"""
    
    def __init__(self, take_profit_pct: float = 0.10):
        """
        初期化
        
        Args:
            take_profit_pct: 利益確定率（0.10 = 10%利益）
                推奨範囲: 0.05 ~ 0.30
        
        Note:
            - デフォルト10%はPhase 2検証用（10%, 15%, 20%で比較）
            - 過度に小さい値（<5%）は手数料負け・オーバートレードのリスク
            - 過度に大きい値（>30%）は利確機会喪失のリスク
        """
        if not 0.01 <= take_profit_pct <= 1.0:
            raise ValueError(
                f"take_profit_pct must be between 0.01 and 1.0, got {take_profit_pct}"
            )
        
        super().__init__(params={"take_profit_pct": take_profit_pct})
        self.take_profit_pct = take_profit_pct
        
        self.logger.info(
            f"[TAKE_PROFIT_INIT] Initialized with take_profit_pct={take_profit_pct:.1%}"
        )
    
    def should_exit(
        self,
        position: Dict,
        current_idx: int,
        data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        利益確定判定
        
        Args:
            position: ポジション情報（entry_price必須）
            current_idx: 現在のインデックス
            data: 株価データ
        
        Returns:
            (should_exit, reason)
            
        Raises:
            ValueError: position検証失敗時
        """
        # ポジション情報検証
        self.validate_position(position)
        
        # 現在価格取得（idx日目終値で判定）
        current_price = data['Close'].iloc[current_idx]
        entry_price = position['entry_price']
        
        # 利益率計算
        profit_pct = (current_price - entry_price) / entry_price
        target_profit = self.take_profit_pct
        
        # 利確判定
        if profit_pct >= target_profit:
            reason = (
                f"Take profit triggered: {profit_pct:.2%} >= {target_profit:.2%} "
                f"(entry={entry_price:.2f}, current={current_price:.2f})"
            )
            self.logger.info(f"[TAKE_PROFIT_EXIT] {reason}")
            return (True, reason)
        
        return (False, "")
    
    def update_position_state(
        self,
        position: Dict,
        current_idx: int,
        data: pd.DataFrame
    ) -> None:
        """
        ポジション状態更新
        
        Note:
            TakeProfitExitは状態管理不要（エントリー価格のみ参照）
            TrailingStopExitとの組み合わせ時に備えてメソッドのみ実装
        """
        pass
