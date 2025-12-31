"""
backtest_daily()実装テンプレート（Phase 3-B Step B2）

このテンプレートは他戦略への展開のためのパターン確立を目的として作成されました。
VWAPBreakout.backtest_daily()の実装をベースに、汎用的なパターンを抽出しています。

使用手順:
1. existing_positionハンドリングコピー
2. インジケーター計算部分のshift(1)適用
3. エントリー価格設定部分のOpen価格使用
4. リターン形式の統一

Author: Backtest Project Team
Created: 2025-12-31 (Phase 3-B Step B2)
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# ログ設定
logger = logging.getLogger(__name__)

def backtest_daily_template(self, current_date, stock_data: pd.DataFrame, existing_position: Optional[Dict] = None) -> Dict[str, Any]:
    """
    backtest_daily()実装テンプレート
    
    【重要】各戦略での実装時の注意事項：
    - copilot-instructions.md準拠必須
    - ルックアヘッドバイアス防止3原則の徹底
    - 実データのみ使用（フォールバック禁止）
    - バックテスト実行必須
    
    Parameters:
        current_date: 判定対象日（datetime）
        stock_data: current_dateまでのデータ（ウォームアップ含む）
        existing_position: 既存のポジション情報（銘柄切替時に使用）
            {
                'symbol': str,           # 保有銘柄コード
                'quantity': int,         # 保有株数
                'entry_price': float,    # エントリー価格
                'entry_date': datetime,  # エントリー日
                'entry_idx': int         # エントリー時のインデックス（オプション）
            }
    
    Returns:
        {
            'action': 'entry'|'exit'|'hold',  # 実行アクション
            'signal': 1|-1|0,                 # シグナル値（1:買い、-1:売り、0:何もしない）
            'price': float,                   # 実行価格（翌日始値想定）
            'shares': int,                    # 取引株数
            'reason': str                     # 判定理由
        }
    """
    
    # Phase 1: current_dateの型変換・検証
    if isinstance(current_date, str):
        current_date = pd.Timestamp(current_date)
    elif not isinstance(current_date, pd.Timestamp):
        current_date = pd.Timestamp(current_date)
    
    # Phase 2: データ整合性チェック
    if current_date not in stock_data.index:
        return {
            'action': 'hold',
            'signal': 0,
            'price': 0.0,
            'shares': 0,
            'reason': f'{self.__class__.__name__}: No data available for {current_date.strftime("%Y-%m-%d")}'
        }
    
    # Phase 3: ウォームアップ期間考慮（150日推奨）
    current_idx = stock_data.index.get_loc(current_date)
    warmup_period = 150  # copilot-instructions.mdで推奨される値
    
    if current_idx < warmup_period:
        return {
            'action': 'hold',
            'signal': 0,
            'price': 0.0,
            'shares': 0,
            'reason': f'{self.__class__.__name__}: Insufficient warmup data. Required: {warmup_period}, Available: {current_idx}'
        }
    
    # Phase 4: データ更新（Option B方式を活用）
    # 【注意】戦略固有のself.dataとの整合性を保つ
    original_data = self.data.copy()  # 一時保存
    
    try:
        # BaseStrategy.backtest_daily()の Option B ロジックを活用
        basic_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        updated_columns = []
        
        for col in basic_columns:
            if col in stock_data.columns and col in self.data.columns:
                # インデックスが一致する部分のみ安全に更新
                common_index = self.data.index.intersection(stock_data.index)
                if len(common_index) > 0:
                    self.data.loc[common_index, col] = stock_data.loc[common_index, col]
                    updated_columns.append(col)
        
        logger.debug(f"[{self.__class__.__name__}.backtest_daily] Data updated: {updated_columns}")
        
        # Phase 5: 既存ポジション処理分岐
        if existing_position is not None:
            # 【既存ポジションあり: エグジット判定】
            return self._handle_exit_logic(current_idx, existing_position, stock_data, current_date)
        else:
            # 【既存ポジションなし: エントリー判定】
            return self._handle_entry_logic(current_idx, stock_data, current_date)
    
    finally:
        # データの復元（元の状態に戻す）
        self.data = original_data


def _handle_exit_logic(self, current_idx: int, existing_position: Dict, stock_data: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, Any]:
    """
    エグジット判定ロジック（テンプレート）
    
    【実装ガイド】：
    - 戦略固有のgenerate_exit_signal()を使用
    - ルックアヘッドバイアス防止: 前日データで判定
    - 翌日始値でエグジット価格設定
    """
    try:
        # 戦略固有のエグジット判定（各戦略で実装）
        entry_idx = existing_position.get('entry_idx', current_idx)
        exit_signal = self.generate_exit_signal(current_idx, entry_idx)
        
        if exit_signal == -1:
            # エグジットシグナル発生
            # 【重要】翌日始値でエグジット（ルックアヘッドバイアス防止）
            if current_idx + 1 < len(stock_data):
                exit_price = stock_data.iloc[current_idx + 1]['Open']
            else:
                # 最終日の場合（限定的フォールバック）
                exit_price = stock_data.iloc[current_idx]['Close']
                logger.warning(f"[{self.__class__.__name__}] Using Close price fallback for final day: {current_date}")
            
            return {
                'action': 'exit',
                'signal': -1,
                'price': float(exit_price),
                'shares': existing_position.get('quantity', 0),
                'reason': f'{self.__class__.__name__}: Exit signal detected on {current_date.strftime("%Y-%m-%d")}'
            }
        else:
            # エグジットシグナルなし: ホールド
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': existing_position.get('quantity', 0),
                'reason': f'{self.__class__.__name__}: Holding position from {current_date.strftime("%Y-%m-%d")}'
            }
    
    except Exception as e:
        logger.error(f"[{self.__class__.__name__}] Exit logic error: {e}")
        return {
            'action': 'hold',
            'signal': 0,
            'price': 0.0,
            'shares': 0,
            'reason': f'{self.__class__.__name__}: Exit logic error: {str(e)}'
        }


def _handle_entry_logic(self, current_idx: int, stock_data: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, Any]:
    """
    エントリー判定ロジック（テンプレート）
    
    【実装ガイド】：
    - 戦略固有のgenerate_entry_signal()を使用
    - ルックアヘッドバイアス防止: 前日データで判定
    - 翌日始値 + スリッページでエントリー価格設定
    """
    try:
        # 戦略固有のエントリー判定（各戦略で実装）
        entry_signal = self.generate_entry_signal(current_idx)
        
        if entry_signal == 1:
            # エントリーシグナル発生
            # 【重要】翌日始値でエントリー + スリッページ（ルックアヘッドバイアス防止）
            if current_idx + 1 < len(stock_data):
                entry_price = stock_data.iloc[current_idx + 1]['Open']
                
                # スリッページ・取引コスト適用（copilot-instructions.md推奨0.1%）
                slippage = self.params.get("slippage", 0.001)
                transaction_cost = self.params.get("transaction_cost", 0.0)
                entry_price = entry_price * (1 + slippage + transaction_cost)
                
                # 標準的な取引株数計算（戦略固有パラメータに応じて調整）
                shares = self._calculate_position_size(entry_price)
                
                return {
                    'action': 'entry',
                    'signal': 1,
                    'price': float(entry_price),
                    'shares': shares,
                    'reason': f'{self.__class__.__name__}: Entry signal detected on {current_date.strftime("%Y-%m-%d")}'
                }
            else:
                # 最終日の場合（エントリー不可）
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': 0,
                    'reason': f'{self.__class__.__name__}: Cannot enter on final day: {current_date.strftime("%Y-%m-%d")}'
                }
        else:
            # エントリーシグナルなし: ホールド
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'{self.__class__.__name__}: No entry signal on {current_date.strftime("%Y-%m-%d")}'
            }
    
    except Exception as e:
        logger.error(f"[{self.__class__.__name__}] Entry logic error: {e}")
        return {
            'action': 'hold',
            'signal': 0,
            'price': 0.0,
            'shares': 0,
            'reason': f'{self.__class__.__name__}: Entry logic error: {str(e)}'
        }


def _calculate_position_size(self, entry_price: float) -> int:
    """
    ポジションサイズ計算（テンプレート）
    
    【実装ガイド】：
    - 戦略固有の資金管理ルールを適用
    - リスク管理パラメータ考慮
    - 最小単位（通常100株）を考慮
    """
    # デフォルト実装: 固定金額方式
    target_amount = self.params.get("position_amount", 100000)  # 10万円相当
    
    if entry_price > 0:
        shares = int(target_amount / entry_price)
        # 最小単位調整（100株単位）
        shares = max(100, shares // 100 * 100)
        return shares
    else:
        return 0


# ===============================
# 戦略別実装例（参考）
# ===============================

class StrategyImplementationExample:
    """
    backtest_daily()実装例
    
    【Phase 3-B Step B2】他戦略実装時の参考コード
    """
    
    def backtest_daily(self, current_date, stock_data, existing_position=None):
        """
        実装例: テンプレートの活用パターン
        """
        # テンプレート関数を呼び出し
        return backtest_daily_template(self, current_date, stock_data, existing_position)
    
    def _handle_exit_logic(self, current_idx, existing_position, stock_data, current_date):
        """戦略固有のエグジット処理"""
        return _handle_exit_logic(self, current_idx, existing_position, stock_data, current_date)
    
    def _handle_entry_logic(self, current_idx, stock_data, current_date):
        """戦略固有のエントリー処理"""
        return _handle_entry_logic(self, current_idx, stock_data, current_date)
    
    def _calculate_position_size(self, entry_price):
        """戦略固有のポジションサイズ計算"""
        return _calculate_position_size(self, entry_price)


# ===============================
# 品質保証チェックリスト
# ===============================

"""
【Phase 3-B Step B2】実装品質チェックリスト:

□ ルックアヘッドバイアス防止:
  □ インジケーター計算でshift(1)適用済み
  □ エントリー価格でdata['Open'].iloc[idx + 1]使用
  □ 前日データのみでシグナル判定

□ copilot-instructions.md遵守:
  □ バックテスト実行必須（strategy.backtest_daily()呼び出し）
  □ フォールバック禁止（実データのみ使用）
  □ 検証なしの報告禁止（実際の数値確認）

□ エラーハンドリング:
  □ データ不足時の適切なメッセージ
  □ 例外発生時のログ記録
  □ リターン形式の統一

□ パフォーマンス:
  □ データコピーの最小化
  □ 不要な計算の削減
  □ メモリリークの防止

□ 決定論保証:
  □ 同じ入力で同じ出力
  □ ランダム性の排除
  □ 再現可能な結果

□ 銘柄切替対応:
  □ existing_positionハンドリング
  □ ポジション継続性の保持
  □ データ整合性の確保
"""