"""
BaseExitStrategy - エグジット戦略基底クラス

エントリー戦略と独立したエグジット判定を提供。
position情報のみを受け取り、決済判断を返す。

主な機能:
- ポジション状態に基づくエグジット判定
- 市場トレンド・価格変動考慮
- 過学習回避（シンプルなルールベース推奨）
- ルックアヘッドバイアス防止

統合コンポーネント:
- GCStrategy等のエントリー戦略と組み合わせ
- backtest_daily()経由で日次実行
- PaperBroker統合（決済実行）

セーフティ機能/注意事項:
- ルックアヘッドバイアス防止（idx日目のデータのみ使用）
- position情報の完全性チェック必須
- エグジット価格はidx+1日目始値想定
- フォールバック禁止（copilot-instructions.md準拠）

Author: Backtest Project Team
Created: 2026-01-22
Last Modified: 2026-01-22
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import pandas as pd
from datetime import datetime
import logging


class BaseExitStrategy(ABC):
    """エグジット戦略基底クラス"""
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初期化
        
        Args:
            params: エグジット戦略パラメータ
                例: {"trailing_stop_pct": 0.05, "take_profit": 0.15}
        
        Note:
            - paramsはエグジット戦略固有のパラメータ
            - 過学習回避のため、パラメータ数は最小限に抑える
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(self.name)
        
        self.logger.info(
            f"[EXIT_INIT] {self.name} initialized with params={self.params}"
        )
    
    @abstractmethod
    def should_exit(
        self, 
        position: Dict,
        current_idx: int,
        data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        エグジット判定（日次対応）
        
        Args:
            position: ポジション情報
                {
                    'entry_date': pd.Timestamp,     # エントリー日
                    'entry_price': float,           # エントリー価格
                    'entry_idx': int,               # エントリー時のインデックス
                    'quantity': int,                # 保有株数
                    'symbol': str,                  # 銘柄コード
                    'highest_price': float          # 保有期間中の最高価格（エグジット戦略が更新）
                }
            current_idx: 現在のインデックス（判定日）
            data: 株価データ（current_idxまでのデータ）
        
        Returns:
            (should_exit: bool, reason: str)
            - should_exit: True=エグジット実行、False=保持継続
            - reason: エグジット理由（例: "Trailing stop triggered"）
            
        Note:
            - data.iloc[:current_idx+1]のみ使用（未来情報禁止）
            - 判定はdata['Close'].iloc[current_idx]で実施
            - エグジット実行はcurrent_idx+1日目の始値想定
            
        Raises:
            ValueError: position情報が不正な場合
        """
        pass
    
    def calculate_exit_price(
        self,
        current_idx: int,
        data: pd.DataFrame
    ) -> float:
        """
        エグジット価格計算（翌日始値想定）
        
        Args:
            current_idx: 判定日のインデックス
            data: 株価データ
        
        Returns:
            float: エグジット実行価格（current_idx+1日目の始値）
        
        Note:
            - copilot-instructions.md準拠（ルックアヘッドバイアス防止）
            - idx日目終値で判断 → idx+1日目始値で執行
            - 最終日フォールバック: 終値を返す（例外処理のみ）
        """
        if current_idx + 1 >= len(data):
            # 最終日フォールバック（終値使用）
            fallback_price = data['Close'].iloc[current_idx]
            self.logger.warning(
                f"[EXIT_PRICE] Last day fallback: "
                f"using Close={fallback_price:.2f} (idx={current_idx})"
            )
            return fallback_price
        
        # 翌日始値取得（標準）
        exit_price = data['Open'].iloc[current_idx + 1]
        self.logger.debug(
            f"[EXIT_PRICE] Next day Open: {exit_price:.2f} "
            f"(current_idx={current_idx}, next_idx={current_idx + 1})"
        )
        return exit_price
    
    def update_position_state(
        self,
        position: Dict,
        current_idx: int,
        data: pd.DataFrame
    ) -> None:
        """
        ポジション状態更新（トレーリングストップ用の最高価格など）
        
        Args:
            position: ポジション情報（インプレース更新）
            current_idx: 現在のインデックス
            data: 株価データ
        
        Note:
            - トレーリングストップ用の最高価格を追跡
            - positionはインプレース更新（戻り値なし）
            - current_idx日目の終値を使用
        """
        current_price = data['Close'].iloc[current_idx]
        
        # highest_price初期化（初回のみ）
        if 'highest_price' not in position:
            position['highest_price'] = position.get('entry_price', current_price)
            self.logger.debug(
                f"[POSITION_UPDATE] highest_price initialized: "
                f"{position['highest_price']:.2f}"
            )
        
        # 最高価格更新
        old_highest = position['highest_price']
        position['highest_price'] = max(
            position['highest_price'],
            current_price
        )
        
        if position['highest_price'] > old_highest:
            self.logger.debug(
                f"[POSITION_UPDATE] highest_price updated: "
                f"{old_highest:.2f} -> {position['highest_price']:.2f}"
            )
    
    def validate_position(self, position: Dict) -> None:
        """
        ポジション情報の妥当性検証
        
        Args:
            position: ポジション情報
        
        Raises:
            ValueError: position情報が不正な場合
        
        Note:
            - copilot-instructions.md準拠（フォールバック禁止）
            - 不正なpositionはエラーとして処理（補完禁止）
        """
        required_keys = ['entry_price', 'entry_date', 'quantity']
        missing_keys = [key for key in required_keys if key not in position]
        
        if missing_keys:
            error_msg = (
                f"[EXIT_ERROR] Invalid position: missing keys={missing_keys}, "
                f"position={position}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # entry_priceがNoneまたは0の場合もエラー
        if position['entry_price'] is None or position['entry_price'] <= 0:
            error_msg = (
                f"[EXIT_ERROR] Invalid entry_price: "
                f"{position['entry_price']}, position={position}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
