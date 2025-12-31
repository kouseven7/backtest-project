"""
MomentumInvestingStrategy backtest_daily()実装（Phase 3-B Step B3）

このファイルは Momentum_Investing.py へのbacktest_daily()追加実装です。
Phase 3-B Step B3: 2つ目の戦略実装として、テンプレート活用パターンの有効性を検証します。

実装方針:
- templates/backtest_daily_template.py のパターンを活用
- Momentum戦略固有のインジケーター（SMA, RSI, MACD, ATR, Volume）を考慮
- 既存のgenerate_entry_signal()/generate_exit_signal()を活用
- ルックアヘッドバイアス防止3原則の徹底

Author: Backtest Project Team
Created: 2025-12-31 (Phase 3-B Step B3)
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def backtest_daily(self, current_date, stock_data: pd.DataFrame, existing_position: Optional[Dict] = None) -> Dict[str, Any]:
    """
    MomentumInvestingStrategy 日次バックテスト実行
    
    Phase 3-B Step B3実装: Momentum戦略での実証実装
    
    Parameters:
        current_date (datetime): 判定対象日
        stock_data (pd.DataFrame): 最新の株価データ
        existing_position (dict, optional): 既存ポジション情報
            {
                'symbol': str,           # 保有銘柄コード
                'quantity': int,         # 保有株数
                'entry_price': float,    # エントリー価格
                'entry_date': datetime,  # エントリー日
                'entry_idx': int         # エントリー時のインデックス（オプション）
            }
            
    Returns:
        dict: {
            'action': 'entry'|'exit'|'hold',
            'signal': 1|-1|0,
            'price': float,
            'shares': int,
            'reason': str
        }
        
    実装内容:
    1. current_dateのindexを特定
    2. ウォームアップ期間考慮（150日）
    3. 前日データのみでMomentumインジケーター計算（shift(1)適用済み）
    4. エントリー/エグジット判定（ルックアヘッドバイアス防止）
    5. 翌日始値エントリー/エグジット価格設定
    
    copilot-instructions.md遵守:
    - バックテスト実行必須
    - フォールバック禁止（実データのみ使用）
    - ルックアヘッドバイアス防止3原則
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
            'reason': f'MomentumInvesting: No data available for {current_date.strftime("%Y-%m-%d")}'
        }
        
    # Phase 3: ウォームアップ期間考慮（150日推奨）
    current_idx = stock_data.index.get_loc(current_date)
    warmup_period = 150  # copilot-instructions.mdで推奨される値
    
    # Momentum戦略の最小要求期間も考慮
    min_required = max(warmup_period, self.params.get("sma_long", 50))
    
    if current_idx < min_required:
        return {
            'action': 'hold',
            'signal': 0,
            'price': 0.0,
            'shares': 0,
            'reason': f'MomentumInvesting: Insufficient warmup data. Required: {min_required}, Available: {current_idx}'
        }
    
    # Phase 4: データ更新（Option B方式を活用）
    # 既存のself.dataを一時保存
    original_data = self.data.copy()
    
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
        
        logger.debug(f"[MomentumInvesting.backtest_daily] Data updated: {updated_columns}")
        
        # Phase 5: 既存ポジション処理分岐
        if existing_position is not None:
            # 【既存ポジションあり: エグジット判定】
            return _handle_exit_logic(self, current_idx, existing_position, stock_data, current_date)
        else:
            # 【既存ポジションなし: エントリー判定】
            return _handle_entry_logic(self, current_idx, stock_data, current_date)
    
    finally:
        # データの復元（元の状態に戻す）
        self.data = original_data


def _handle_exit_logic(self, current_idx: int, existing_position: Dict, stock_data: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, Any]:
    """
    エグジット判定ロジック（Momentum戦略固有）
    
    実装ガイド:
    - Momentum戦略のgenerate_exit_signal()を使用
    - ルックアヘッドバイアス防止: 前日データで判定
    - 翌日始値でエグジット価格設定
    - 既存のATRストップロス、モメンタム失速、最大保有期間等を考慮
    """
    try:
        # エントリーインデックスの取得（existing_positionから）
        entry_idx = existing_position.get('entry_idx', current_idx)
        
        # 【重要】entry_pricesに記録（generate_exit_signal()が参照する）
        if entry_idx not in self.entry_prices:
            self.entry_prices[entry_idx] = existing_position.get('entry_price', 0.0)
        
        # Momentum戦略固有のエグジット判定（既存メソッドを活用）
        # 注意: idx+1アクセスが必要なため、最終日チェック
        if current_idx + 1 >= len(stock_data):
            # 最終日の場合は当日終値でエグジット（境界条件）
            exit_price = stock_data.iloc[current_idx]['Close']
            logger.warning(f"[MomentumInvesting.exit] Final day exit: {current_date}")
            
            return {
                'action': 'exit',
                'signal': -1,
                'price': float(exit_price),
                'shares': existing_position.get('quantity', 0),
                'reason': f'MomentumInvesting: Final day exit on {current_date.strftime("%Y-%m-%d")}'
            }
        
        exit_signal = self.generate_exit_signal(current_idx)
        
        if exit_signal == -1:
            # エグジットシグナル発生
            # 【重要】翌日始値でエグジット（ルックアヘッドバイアス防止）
            exit_price = stock_data.iloc[current_idx + 1]['Open']
            
            # Series型のままの場合はスカラー化
            if isinstance(exit_price, pd.Series):
                exit_price = exit_price.values[0]
            
            # スリッページ・取引コスト適用（Phase 2: 2025-12-23）
            slippage = self.params.get("slippage", 0.001)
            transaction_cost = self.params.get("transaction_cost", 0.0)
            exit_price = exit_price * (1 - slippage - transaction_cost)  # 売りはマイナス
            
            return {
                'action': 'exit',
                'signal': -1,
                'price': float(exit_price),
                'shares': existing_position.get('quantity', 0),
                'reason': f'MomentumInvesting: Exit signal detected on {current_date.strftime("%Y-%m-%d")}'
            }
        else:
            # エグジットシグナルなし: ホールド
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': existing_position.get('quantity', 0),
                'reason': f'MomentumInvesting: Holding position from {current_date.strftime("%Y-%m-%d")}'
            }
    
    except Exception as e:
        logger.error(f"[MomentumInvesting.exit] Exit logic error: {e}", exc_info=True)
        return {
            'action': 'hold',
            'signal': 0,
            'price': 0.0,
            'shares': 0,
            'reason': f'MomentumInvesting: Exit logic error: {str(e)}'
        }


def _handle_entry_logic(self, current_idx: int, stock_data: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, Any]:
    """
    エントリー判定ロジック（Momentum戦略固有）
    
    実装ガイド:
    - Momentum戦略のgenerate_entry_signal()を使用
    - ルックアヘッドバイアス防止: 前日データで判定
    - 翌日始値 + スリッページでエントリー価格設定
    - 既存のSMA/RSI/MACD/Volume条件を考慮
    """
    try:
        # Momentum戦略固有のエントリー判定（既存メソッドを活用）
        entry_signal = self.generate_entry_signal(current_idx)
        
        if entry_signal == 1:
            # エントリーシグナル発生
            # 【重要】翌日始値でエントリー + スリッページ（ルックアヘッドバイアス防止）
            if current_idx + 1 < len(stock_data):
                entry_price = stock_data.iloc[current_idx + 1]['Open']
                
                # Series型のままの場合はスカラー化
                if isinstance(entry_price, pd.Series):
                    entry_price = entry_price.values[0]
                
                # スリッページ・取引コスト適用（Phase 2: 2025-12-23）
                slippage = self.params.get("slippage", 0.001)
                transaction_cost = self.params.get("transaction_cost", 0.0)
                entry_price = entry_price * (1 + slippage + transaction_cost)
                
                # 標準的な取引株数計算（戦略固有パラメータに応じて調整）
                shares = _calculate_position_size(self, entry_price)
                
                return {
                    'action': 'entry',
                    'signal': 1,
                    'price': float(entry_price),
                    'shares': shares,
                    'reason': f'MomentumInvesting: Entry signal detected on {current_date.strftime("%Y-%m-%d")}'
                }
            else:
                # 最終日の場合（エントリー不可）
                return {
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': 0,
                    'reason': f'MomentumInvesting: Cannot enter on final day: {current_date.strftime("%Y-%m-%d")}'
                }
        else:
            # エントリーシグナルなし: ホールド
            return {
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'MomentumInvesting: No entry signal on {current_date.strftime("%Y-%m-%d")}'
            }
    
    except Exception as e:
        logger.error(f"[MomentumInvesting.entry] Entry logic error: {e}", exc_info=True)
        return {
            'action': 'hold',
            'signal': 0,
            'price': 0.0,
            'shares': 0,
            'reason': f'MomentumInvesting: Entry logic error: {str(e)}'
        }


def _calculate_position_size(self, entry_price: float) -> int:
    """
    ポジションサイズ計算（Momentum戦略固有）
    
    実装ガイド:
    - Momentum戦略のリスク管理ルールを適用
    - ATRベースの position sizing（オプション）
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
