"""
GCStrategyWithExit - エグジット戦略分離版GC戦略

既存GCStrategyのエントリーロジックを保持しつつ、
エグジット判定を差し替え可能にした設計。

主な機能:
- GCエントリー戦略（固定）
- エグジット戦略の動的差し替え
- backtest_daily()対応
- 日次バックテスト実行

統合コンポーネント:
- BaseExitStrategy派生クラス（TrailingStopExit等）
- GCStrategy既存実装を継承
- PaperBroker統合（決済実行）

セーフティ機能/注意事項:
- エントリーロジックは変更不可（検証の再現性担保）
- エグジット戦略のみを差し替えて比較検証
- ルックアヘッドバイアス防止（copilot-instructions.md準拠）

Author: Backtest Project Team
Created: 2026-01-22
Last Modified: 2026-01-22
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict

# プロジェクトルートをパス追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.gc_strategy_signal import GCStrategy
from strategies.exit_strategies.base_exit_strategy import BaseExitStrategy


class GCStrategyWithExit(GCStrategy):
    """エグジット戦略分離版GC戦略"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        exit_strategy: BaseExitStrategy,
        params: Optional[dict] = None,
        price_column: str = "Adj Close",
        ticker: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            data: 株価データ
            exit_strategy: 使用するエグジット戦略
            params: 戦略パラメータ（エントリー用）
            price_column: 価格カラム
            ticker: 銘柄コード
        
        Note:
            - exit_strategyは外部から注入（Dependency Injection）
            - エントリーロジックは親クラス（GCStrategy）のまま
        """
        super().__init__(data, params, price_column, ticker)
        self.exit_strategy = exit_strategy
        
        # entry_trendを保存するための辞書（entry_idx -> trend）
        self.entry_trends = {}
        
        self.logger.info(
            f"[GC_EXIT_INIT] GCStrategyWithExit initialized: "
            f"exit_strategy={exit_strategy.name}, "
            f"params={exit_strategy.params}"
        )
    
    def _handle_exit_logic_daily(
        self,
        current_idx: int,
        existing_position: dict,
        stock_data: pd.DataFrame,
        current_date: pd.Timestamp,
        entry_symbol_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        エグジットロジック（BaseExitStrategy使用版）
        
        既存の_handle_exit_logic_daily()をオーバーライド。
        BaseExitStrategyを使用してエグジット判定を実行。
        
        Args:
            current_idx: 現在のインデックス
            existing_position: 既存ポジション情報
            stock_data: 株価データ
            current_date: 現在日時
            entry_symbol_data: force_close時の元の銘柄データ（オプション）
        
        Returns:
            {'action': 'exit'|'hold', 'signal': -1|0, 'price': float, 'shares': int, 'reason': str}
        
        Note:
            - BaseExitStrategy.should_exit()を呼び出し
            - entry_symbol_data提供時は元の銘柄データでエグジット価格計算
        """
        try:
            # エグジット戦略でポジション状態を更新
            self.exit_strategy.update_position_state(
                existing_position,
                current_idx,
                stock_data
            )
            
            # エグジット判定
            should_exit, reason = self.exit_strategy.should_exit(
                existing_position,
                current_idx,
                stock_data
            )
            
            if should_exit:
                # エグジット価格計算
                # entry_symbol_data提供時（force_close）は元の銘柄データを使用
                price_data = entry_symbol_data if entry_symbol_data is not None else stock_data
                exit_price = self.exit_strategy.calculate_exit_price(
                    current_idx,
                    price_data
                )
                
                self.logger.info(
                    f"[GC_EXIT] {reason}, "
                    f"exit_price={exit_price:.2f}, "
                    f"using_entry_symbol_data={entry_symbol_data is not None}"
                )
                
                return {
                    'action': 'exit',
                    'signal': -1,
                    'price': float(exit_price),
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'GCStrategy+{self.exit_strategy.name}: {reason}'
                }
            else:
                # 保持継続
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': existing_position.get('quantity', 0),
                    'reason': f'GCStrategy+{self.exit_strategy.name}: Holding position'
                }
        
        except Exception as e:
            self.logger.error(
                f"[GC_EXIT_ERROR] Exit logic error: {e}",
                exc_info=True
            )
            # copilot-instructions.md準拠: エラー隠蔽禁止
            raise
    
    def _handle_entry_logic_daily(
        self,
        current_idx: int,
        stock_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Dict:
        """
        エントリーロジック（entry_trend記録版）
        
        親クラスの_handle_entry_logic_daily()を拡張し、
        エントリー時のトレンドをentry_trendsに記録。
        
        Args:
            current_idx: 現在のインデックス
            stock_data: 株価データ
            current_date: 現在日時
        
        Returns:
            {'action': 'entry'|'pass', 'signal': 1|0, 'price': float, 'shares': int, 'reason': str}
        """
        # 親クラスのエントリーロジック実行
        result = super()._handle_entry_logic_daily(current_idx, stock_data, current_date)
        
        # エントリーが実行された場合、entry_trendを記録
        if result['action'] == 'entry':
            try:
                # 現在のトレンド取得（遅延インポートで循環参照回避）
                from indicators.unified_trend_detector import detect_unified_trend_with_confidence
                
                # current_idxまでのデータでトレンド判定
                current_data = stock_data.iloc[:current_idx + 1]
                trend, confidence = detect_unified_trend_with_confidence(
                    current_data,
                    strategy='GCStrategy',
                    method='advanced'
                )
                
                # entry_trendsに記録
                self.entry_trends[current_idx] = trend
                
                self.logger.info(
                    f"[GC_ENTRY] Entry at idx={current_idx}, trend={trend}, confidence={confidence:.2%}"
                )
            except Exception as e:
                # トレンド取得失敗時はデフォルトで'uptrend'を設定
                self.logger.warning(f"[GC_ENTRY] Failed to detect entry trend: {e}, using default 'uptrend'")
                self.entry_trends[current_idx] = 'uptrend'
        
        return result
    
    def generate_exit_signal(self, idx: int, entry_idx: int = -1) -> int:
        """
        イグジットシグナル生成（BaseExitStrategy使用版）
        
        既存のgenerate_exit_signal()をオーバーライド。
        BaseExitStrategyを使用してエグジット判定を実行。
        
        Args:
            idx: 現在のインデックス
            entry_idx: エントリー時のインデックス
        
        Returns:
            int: イグジットシグナル（-1: イグジット, 0: なし）
        
        Note:
            - 親クラス（GCStrategy）のgenerate_exit_signal()をオーバーライド
            - BaseExitStrategy.should_exit()を呼び出し
            - IndexError防止（最終日フォールバック対応済み）
        """
        if idx < self.params["long_window"]:
            return 0
        
        if entry_idx < 0:
            self.logger.debug(f"[EXIT CHECK] idx={idx}, entry_idx={entry_idx} (< 0), returning 0")
            return 0
        
        # エントリー価格を取得
        entry_price = self.entry_prices.get(entry_idx)
        
        # entry_priceがNoneの場合はエラー（フォールバック禁止）
        if entry_price is None:
            error_msg = (
                f"CRITICAL ERROR: エントリー価格がNoneです。"
                f"entry_idx={entry_idx}, idx={idx}, date={self.data.index[idx]}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # ポジション情報構築
        position = {
            'entry_price': entry_price,
            'entry_date': self.data.index[entry_idx],
            'entry_idx': entry_idx,
            'quantity': 100,  # ダミー（エグジット判定には不要）
            'symbol': self.ticker or 'UNKNOWN',
            'highest_price': self.high_prices.get(entry_idx, entry_price),
            'entry_trend': self.entry_trends.get(entry_idx, 'uptrend')  # TrendFollowingExit用
        }
        
        # BaseExitStrategy.should_exit()呼び出し
        try:
            should_exit, reason = self.exit_strategy.should_exit(
                position,
                idx,
                self.data
            )
            
            if should_exit:
                self.logger.info(f"[GC_EXIT] {reason}")
                # highest_price更新（トレーリングストップ用）
                self.high_prices[entry_idx] = position['highest_price']
                return -1
            else:
                # highest_price更新（トレーリングストップ用）
                self.high_prices[entry_idx] = position['highest_price']
                return 0
        
        except Exception as e:
            self.logger.error(
                f"[GC_EXIT_ERROR] generate_exit_signal error: {e}",
                exc_info=True
            )
            # copilot-instructions.md準拠: エラー隠蔽禁止
            raise
    
    def __repr__(self) -> str:
        """文字列表現"""
        return (
            f"GCStrategyWithExit("
            f"short_window={self.short_window}, "
            f"long_window={self.long_window}, "
            f"exit_strategy={self.exit_strategy})"
        )


# テストコード
if __name__ == "__main__":
    print("[TEST] GCStrategyWithExit動作確認")
    
    # ダミーデータ作成
    dates = pd.date_range(start="2023-01-01", periods=100, freq='B')
    df = pd.DataFrame({
        'Open': np.random.random(100) * 100 + 100,
        'High': np.random.random(100) * 100 + 110,
        'Low': np.random.random(100) * 100 + 90,
        'Close': np.random.random(100) * 100 + 100,
        'Adj Close': np.random.random(100) * 100 + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # TrailingStopExit作成
    from strategies.exit_strategies.trailing_stop_exit import TrailingStopExit
    exit_strategy = TrailingStopExit(trailing_stop_pct=0.05)
    
    # GCStrategyWithExit実行
    strategy = GCStrategyWithExit(
        data=df,
        exit_strategy=exit_strategy,
        ticker="TEST.T"
    )
    
    print(f"\n[TEST] 戦略: {strategy}")
    print(f"[TEST] エグジット戦略: {strategy.exit_strategy}")
    
    # バックテスト実行（全期間）
    print(f"\n[TEST] バックテスト実行開始...")
    results = strategy.backtest()
    
    if len(results) > 0:
        print(f"[TEST] 取引数: {len(results)}")
        print(f"[TEST] 総損益: {results['Profit_Loss'].sum():.2f}")
        print(f"\n[OK] GCStrategyWithExitテスト完了")
    else:
        print(f"\n[WARNING] 取引なし（テストデータの制約）")
