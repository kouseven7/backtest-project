"""
CompositeExit - 複数エグジット戦略組み合わせ

複数のエグジット戦略を組み合わせ、いずれか1つが発火したら決済する複合戦略。

主な機能:
- 複数BaseExitStrategy統合（OR条件）
- TrailingStop + TakeProfit組み合わせ
- TrailingStop + StopLoss組み合わせ
- 発火理由の追跡（どの戦略が決済トリガーか）

統合コンポーネント:
- BaseExitStrategy継承
- TrailingStopExit、TakeProfitExit、FixedStopLossExit統合
- GCStrategyWithExit等のエントリー戦略と組み合わせ

セーフティ機能/注意事項:
- 組み合わせ戦略は最大3つまで推奨（過学習回避）
- update_position_state()は全戦略に伝播
- ルックアヘッドバイアス防止（各戦略が個別遵守）
- フォールバック禁止（copilot-instructions.md準拠）

Author: Backtest Project Team
Created: 2026-01-22
Last Modified: 2026-01-22
"""
from typing import Dict, List, Tuple
import pandas as pd
from .base_exit_strategy import BaseExitStrategy


class CompositeExit(BaseExitStrategy):
    """複数エグジット戦略組み合わせ"""
    
    def __init__(
        self,
        strategies: List[BaseExitStrategy],
        name: str = "CompositeExit"
    ):
        """
        初期化
        
        Args:
            strategies: 組み合わせるエグジット戦略リスト
                例: [TrailingStopExit(0.05), TakeProfitExit(0.15)]
            name: 戦略名（レポート用）
        
        Raises:
            ValueError: strategiesが空またはBaseExitStrategy以外を含む場合
        
        Note:
            - 組み合わせは最大3つまで推奨（過学習回避）
            - 各戦略のパラメータは独立して管理
        """
        if not strategies:
            raise ValueError("strategies must contain at least one exit strategy")
        
        if not all(isinstance(s, BaseExitStrategy) for s in strategies):
            raise ValueError("All strategies must be instances of BaseExitStrategy")
        
        # パラメータ統合（各戦略のパラメータをマージ）
        merged_params = {}
        for i, strategy in enumerate(strategies):
            strategy_params = {
                f"{strategy.name}_{k}": v 
                for k, v in strategy.params.items()
            }
            merged_params.update(strategy_params)
        
        super().__init__(params=merged_params)
        self.strategies = strategies
        self.name = name
        
        strategy_names = [s.name for s in strategies]
        self.logger.info(
            f"[COMPOSITE_INIT] {name} initialized with {len(strategies)} strategies: "
            f"{strategy_names}"
        )
    
    def should_exit(
        self,
        position: Dict,
        current_idx: int,
        data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        複合エグジット判定（OR条件）
        
        Args:
            position: ポジション情報
            current_idx: 現在のインデックス
            data: 株価データ
        
        Returns:
            (should_exit, reason)
            - いずれか1つの戦略がTrue → 即座にTrue返却
            - 複数戦略発火時は最初の発火理由を返却
            
        Raises:
            ValueError: position検証失敗時
        """
        # ポジション情報検証
        self.validate_position(position)
        
        # 各戦略を順次チェック（OR条件）
        triggered_reasons = []
        
        for strategy in self.strategies:
            should_exit, reason = strategy.should_exit(position, current_idx, data)
            
            if should_exit:
                triggered_reasons.append(f"{strategy.name}: {reason}")
        
        # いずれか発火していればエグジット
        if triggered_reasons:
            combined_reason = " | ".join(triggered_reasons)
            self.logger.info(f"[COMPOSITE_EXIT] {combined_reason}")
            return (True, combined_reason)
        
        return (False, "")
    
    def update_position_state(
        self,
        position: Dict,
        current_idx: int,
        data: pd.DataFrame
    ) -> None:
        """
        ポジション状態更新（全戦略に伝播）
        
        Note:
            - TrailingStopExitはhighest_priceを更新
            - TakeProfitExit、FixedStopLossExitは状態更新なし
            - 各戦略のupdate_position_state()を順次呼び出し
        """
        for strategy in self.strategies:
            strategy.update_position_state(position, current_idx, data)
