"""
TrailingStopExit - トレーリングストップエグジット戦略

高値からの下落率でエグジット判定を行うシンプルな戦略。

主な機能:
- 保有期間中の最高価格追跡
- 下落率ベースのトレーリングストップ
- 固定損切り併用（オプション）
- ルックアヘッドバイアス防止

統合コンポーネント:
- BaseExitStrategy基底クラス継承
- GCStrategyWithExit経由で実行
- PaperBroker統合（決済実行）

セーフティ機能/注意事項:
- パラメータはtrailing_stop_pctのみ（過学習回避）
- 市場トレンド未考慮（シンプル設計）
- position['highest_price']はインプレース更新
- フォールバック禁止（copilot-instructions.md準拠）

Author: Backtest Project Team
Created: 2026-01-22
Last Modified: 2026-01-22
"""
from strategies.exit_strategies.base_exit_strategy import BaseExitStrategy
from typing import Dict, Tuple
import pandas as pd


class TrailingStopExit(BaseExitStrategy):
    """トレーリングストップエグジット戦略"""
    
    def __init__(self, trailing_stop_pct: float = 0.05):
        """
        初期化
        
        Args:
            trailing_stop_pct: トレーリングストップ率（デフォルト5%）
                例: 0.05 = 最高価格から5%下落でエグジット
        
        Note:
            - パラメータは1つのみ（過学習回避）
            - 推奨範囲: 0.03 ~ 0.10（3% ~ 10%）
        """
        super().__init__(params={'trailing_stop_pct': trailing_stop_pct})
        self.trailing_stop_pct = trailing_stop_pct
        
        self.logger.info(
            f"[TRAILING_INIT] TrailingStopExit initialized: "
            f"trailing_stop_pct={trailing_stop_pct:.1%}"
        )
    
    def should_exit(
        self,
        position: Dict,
        current_idx: int,
        data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        トレーリングストップ判定
        
        Args:
            position: ポジション情報
            current_idx: 現在のインデックス
            data: 株価データ
        
        Returns:
            (should_exit, reason)
            - should_exit: True=エグジット実行、False=保持継続
            - reason: エグジット理由
        
        Note:
            - current_idx日目の終値で判断
            - エグジット実行はcurrent_idx+1日目の始値
        
        Raises:
            ValueError: position情報が不正な場合
        """
        # ポジション情報検証
        self.validate_position(position)
        
        # 現在価格取得（idx日目終値）
        current_price = data['Close'].iloc[current_idx]
        
        # ポジション状態更新（最高価格追跡）
        self.update_position_state(position, current_idx, data)
        
        # トレーリングストップ計算
        highest_price = position['highest_price']
        trailing_stop = highest_price * (1 - self.trailing_stop_pct)
        
        self.logger.debug(
            f"[TRAILING_CHECK] idx={current_idx}, "
            f"current_price={current_price:.2f}, "
            f"highest_price={highest_price:.2f}, "
            f"trailing_stop={trailing_stop:.2f} "
            f"({self.trailing_stop_pct:.1%})"
        )
        
        # トレーリングストップ判定
        if current_price < trailing_stop:
            reason = (
                f"Trailing stop triggered: "
                f"{current_price:.2f} < {trailing_stop:.2f} "
                f"(highest={highest_price:.2f}, "
                f"stop_pct={self.trailing_stop_pct:.1%})"
            )
            self.logger.info(f"[TRAILING_EXIT] {reason}")
            return (True, reason)
        
        # 保持継続
        return (False, "")
    
    def __repr__(self) -> str:
        """文字列表現"""
        return f"TrailingStopExit(trailing_stop_pct={self.trailing_stop_pct:.1%})"
