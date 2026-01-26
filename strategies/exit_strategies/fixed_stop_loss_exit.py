"""
FixedStopLossExit - 固定損切エグジット戦略

エントリー価格から指定損失率に達したら決済する損切戦略。

主な機能:
- 固定損失率到達での決済（例: 2%, 3%, 5%）
- エントリー価格基準の損失判定
- 日次バックテスト対応
- ルックアヘッドバイアス防止（idx日目判断→idx+1日目執行）

統合コンポーネント:
- BaseExitStrategy継承
- GCStrategyWithExit等のエントリー戦略と組み合わせ
- CompositeExit経由での組み合わせ戦略対応（TrailingStop + StopLoss）

セーフティ機能/注意事項:
- パラメータは損失率のみ（過学習回避）
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


class FixedStopLossExit(BaseExitStrategy):
    """固定損切エグジット戦略"""
    
    def __init__(self, stop_loss_pct: float = 0.03):
        """
        初期化
        
        Args:
            stop_loss_pct: 損切率（0.03 = 3%損失）
                推奨範囲: 0.01 ~ 0.10
        
        Note:
            - デフォルト3%はPhase 2検証用（2%, 3%, 5%で比較）
            - 過度に小さい値（<1%）はノイズでの誤決済リスク
            - 過度に大きい値（>10%）は破産リスク
        """
        if not 0.005 <= stop_loss_pct <= 0.20:
            raise ValueError(
                f"stop_loss_pct must be between 0.005 and 0.20, got {stop_loss_pct}"
            )
        
        super().__init__(params={"stop_loss_pct": stop_loss_pct})
        self.stop_loss_pct = stop_loss_pct
        
        self.logger.info(
            f"[STOP_LOSS_INIT] Initialized with stop_loss_pct={stop_loss_pct:.1%}"
        )
    
    def should_exit(
        self,
        position: Dict,
        current_idx: int,
        data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        損切判定
        
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
        
        # 損失率計算
        loss_pct = (entry_price - current_price) / entry_price
        stop_loss_threshold = self.stop_loss_pct
        
        # 損切判定
        if loss_pct >= stop_loss_threshold:
            reason = (
                f"Stop loss triggered: {loss_pct:.2%} >= {stop_loss_threshold:.2%} "
                f"(entry={entry_price:.2f}, current={current_price:.2f})"
            )
            self.logger.info(f"[STOP_LOSS_EXIT] {reason}")
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
            FixedStopLossExitは状態管理不要（エントリー価格のみ参照）
            TrailingStopExitとの組み合わせ時に備えてメソッドのみ実装
        """
        pass
