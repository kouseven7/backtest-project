"""
Exit Strategies Package

エグジット戦略パッケージ。エントリー戦略から独立したエグジット判定を提供。

主な機能:
- BaseExitStrategy: エグジット戦略基底クラス
- TrailingStopExit: トレーリングストップエグジット
- TakeProfitExit: 固定利確エグジット（Phase 2完了）
- FixedStopLossExit: 固定損切エグジット（Phase 2完了）
- CompositeExit: 複数エグジット戦略組み合わせ（Phase 2完了）
- TrendFollowingExit: トレンドフォロー型エグジット（Phase 3完了）

統合コンポーネント:
- GCStrategyWithExit: GC戦略統合版
- BreakoutStrategyWithExit: Breakout戦略統合版（Phase 4実装予定）

セーフティ機能/注意事項:
- ルックアヘッドバイアス防止（idx日目のデータで判断、idx+1日目で執行）
- フォールバック禁止（copilot-instructions.md準拠）
- position情報の完全性チェック必須

Author: Backtest Project Team
Created: 2026-01-22
Last Modified: 2026-01-22
"""

from .base_exit_strategy import BaseExitStrategy
from .trailing_stop_exit import TrailingStopExit
from .take_profit_exit import TakeProfitExit
from .fixed_stop_loss_exit import FixedStopLossExit
from .composite_exit import CompositeExit
from .trend_following_exit import TrendFollowingExit

__all__ = [
    'BaseExitStrategy',
    'TrailingStopExit',
    'TakeProfitExit',
    'FixedStopLossExit',
    'CompositeExit',
    'TrendFollowingExit',
]
