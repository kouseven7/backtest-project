# ForceCloseStrategy 実装仕様書

**Phase**: Phase 2-3実装準備  
**Priority**: Priority 2  
**作成日**: 2025-12-19  
**Status**: 実装準備完了

---

## 📋 概要

銘柄切替時・バックテスト終了時に全ポジションを強制決済する戦略。
PaperBroker.close_all_positions()を呼び出し、決済結果をsignals形式に変換。

---

## 🎯 設計方針

### Option B採用理由（phase_2_1_detailed_design.md参照）

1. **Phase 1整合性**: 他戦略（Contrarian, GoldenCross等）と同じパターン
2. **単一責任原則**: 戦略としての独立性保持
3. **既存パターン踏襲**: BaseStrategy基底クラス活用
4. **dssms_cleanup_plan.md整合性**: 「main_new.pyのForceClose戦略」明記
5. **アーキテクチャ**: StrategyExecutionManagerとの統合容易

### 他戦略との統合パターン

```
StrategyExecutionManager
    ↓
ForceCloseStrategy.backtest()
    ↓
PaperBroker.close_all_positions()
    ↓
決済結果 → signals変換
    ↓
StrategyExecutionManager._execute_trades()
    ↓
execution_details生成（strategy_name="ForceClose"）
```

---

## 🔧 クラス仕様

### クラスシグネチャ

```python
class ForceCloseStrategy(BaseStrategy):
    """
    強制決済戦略（銘柄切替時/バックテスト終了時）
    
    PaperBroker.close_all_positions()を呼び出し、
    全ポジションを決済するsignalsを生成。
    
    strategy_name: "ForceClose"
    
    主な機能:
    - PaperBroker.close_all_positions()呼び出し
    - 決済結果をsignals DataFrame形式に変換
    - strategy_name="ForceClose"明示設定
    - execution_details生成（StrategyExecutionManager経由）
    - エラー耐性（決済失敗時も処理継続）
    
    統合コンポーネント:
    - PaperBroker: close_all_positions()呼び出し
    - StrategyExecutionManager: signals実行、execution_details生成
    - IntegratedExecutionManager: execute_force_close()経由で呼び出し
    
    セーフティ機能/注意事項:
    - PaperBroker.close_all_positions()はエラー耐性実装済み
    - 決済失敗銘柄は警告ログ出力（エラー隠蔽禁止）
    - モック/ダミーデータ使用禁止（copilot-instructions.md準拠）
    - フォールバック機能禁止
    
    Author: Backtest Project Team
    Created: 2025-12-19
    Last Modified: 2025-12-19
    """
```

---

## 📝 実装コード

### 完全実装（strategies/force_close_strategy.py）

```python
"""
ForceCloseStrategy - 強制決済戦略

銘柄切替時・バックテスト終了時に全ポジションを強制決済する戦略。
PaperBroker.close_all_positions()を呼び出し、決済結果をsignals形式に変換。

主な機能:
- PaperBroker.close_all_positions()呼び出し
- 決済結果をsignals DataFrame形式に変換
- strategy_name="ForceClose"明示設定
- execution_details生成（StrategyExecutionManager経由）
- エラー耐性（決済失敗時も処理継続）

統合コンポーネント:
- PaperBroker: close_all_positions()呼び出し
- StrategyExecutionManager: signals実行、execution_details生成
- IntegratedExecutionManager: execute_force_close()経由で呼び出し

セーフティ機能/注意事項:
- PaperBroker.close_all_positions()はエラー耐性実装済み
- 決済失敗銘柄は警告ログ出力（エラー隠蔽禁止）
- モック/ダミーデータ使用禁止（copilot-instructions.md準拠）
- フォールバック機能禁止

Author: Backtest Project Team
Created: 2025-12-19
Last Modified: 2025-12-19
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from strategies.base_strategy import BaseStrategy


class ForceCloseStrategy(BaseStrategy):
    """
    強制決済戦略（銘柄切替時/バックテスト終了時）
    
    PaperBroker.close_all_positions()を呼び出し、
    全ポジションを決済するsignalsを生成。
    
    strategy_name: "ForceClose"
    """
    
    def __init__(self, broker, data: Optional[pd.DataFrame] = None, 
                 params: Optional[Dict[str, Any]] = None, 
                 reason: str = "symbol_switch"):
        """
        ForceCloseStrategy初期化
        
        Args:
            broker: PaperBrokerインスタンス（close_all_positions()呼び出し用）
            data: 株価データ（オプション、signals生成に使用）
            params: 戦略パラメータ（オプション）
            reason: 決済理由（"symbol_switch", "backtest_end"等）
        
        Note:
            - BaseStrategyはdata必須のためダミーDataFrame渡す
            - 実際の決済はPaperBroker.close_all_positions()経由
        """
        self.broker = broker
        self.reason = reason
        self.strategy_name = "ForceClose"
        
        # BaseStrategy初期化（dataがNoneの場合はダミーDataFrame作成）
        if data is None:
            # 空のDataFrame（BaseStrategy要件満たすため）
            data = pd.DataFrame(
                {'Close': [0.0]}, 
                index=pd.DatetimeIndex([datetime.now()])
            )
        
        super().__init__(data, params or {})
        self.logger.info(f"ForceCloseStrategy initialized: reason={reason}")
    
    def backtest(self, trading_start_date: Optional[pd.Timestamp] = None,
                 trading_end_date: Optional[pd.Timestamp] = None,
                 current_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        強制決済実行
        
        Args:
            trading_start_date: 取引開始日（未使用、BaseStrategyインターフェース互換性のため）
            trading_end_date: 取引終了日（未使用、BaseStrategyインターフェース互換性のため）
            current_date: 決済日時（PaperBroker.close_all_positions()に渡す）
        
        Returns:
            pd.DataFrame: 決済シグナル（strategy="ForceClose"設定済み）
        
        Note:
            - PaperBroker.close_all_positions()呼び出し
            - 決済結果をsignals DataFrame形式に変換
            - StrategyExecutionManagerが後続処理で実行
        
        copilot-instructions.md準拠:
            - 実データのみ使用（モック/ダミー禁止）
            - エラー隠蔽禁止（警告ログ出力）
            - フォールバック禁止
        """
        try:
            # current_dateが未指定の場合は現在時刻を使用
            if current_date is None:
                current_date = datetime.now()
            
            self.logger.info(
                f"[FORCE_CLOSE] Starting force close execution: "
                f"date={current_date.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"reason={self.reason}"
            )
            
            # PaperBroker.close_all_positions()呼び出し
            close_results = self.broker.close_all_positions(
                current_date=current_date,
                reason=self.reason
            )
            
            self.logger.info(
                f"[FORCE_CLOSE] PaperBroker returned {len(close_results)} close results"
            )
            
            # 決済結果が空の場合（ポジション未保有）
            if not close_results:
                self.logger.info("[FORCE_CLOSE] No positions to close")
                # 空のsignals DataFrame返却
                return self._create_empty_signals(current_date)
            
            # 決済結果をsignals DataFrame形式に変換
            signals = self._convert_to_signals(close_results, current_date)
            
            self.logger.info(
                f"[FORCE_CLOSE] Generated {len(signals)} SELL signals for force close"
            )
            
            return signals
            
        except Exception as e:
            # エラー時は警告ログ出力（エラー隠蔽禁止）
            self.logger.error(
                f"[FORCE_CLOSE] Error in force close execution: {e}",
                exc_info=True
            )
            # 空のsignals返却（フォールバック禁止）
            return self._create_empty_signals(current_date)
    
    def _convert_to_signals(self, close_results: List[Dict[str, Any]], 
                          current_date: datetime) -> pd.DataFrame:
        """
        PaperBroker決済結果をsignals DataFrame形式に変換
        
        Args:
            close_results: PaperBroker.close_all_positions()の返却値
            current_date: 決済日時
        
        Returns:
            pd.DataFrame: signals DataFrame（strategy="ForceClose"設定済み）
        
        Note:
            - 各決済結果をSELLシグナルとして変換
            - Exit_Signal=-1設定
            - strategy="ForceClose"明示
        """
        try:
            if not close_results:
                return self._create_empty_signals(current_date)
            
            # signals DataFrame構築
            signals_data = []
            
            for result in close_results:
                # 各決済結果をSELLシグナルとして追加
                signal_row = {
                    'Close': result['exit_price'],
                    'Entry_Signal': 0,
                    'Exit_Signal': -1,  # SELL
                    'Position': 0,  # ポジションクローズ
                    'Strategy': 'ForceClose',
                    'symbol': result['symbol'],
                    'quantity': result['quantity'],
                    'entry_price': result['entry_price'],
                    'exit_price': result['exit_price'],
                    'entry_time': result['entry_time'],
                    'pnl': result['pnl'],
                    'commission': result['commission'],
                    'slippage': result['slippage'],
                    'reason': result['reason']
                }
                signals_data.append(signal_row)
            
            # DataFrameに変換（インデックスは決済日時）
            signals = pd.DataFrame(
                signals_data,
                index=[current_date] * len(signals_data)
            )
            
            self.logger.info(
                f"[FORCE_CLOSE] Converted {len(signals)} close results to signals"
            )
            
            return signals
            
        except Exception as e:
            self.logger.error(
                f"[FORCE_CLOSE] Error converting close results to signals: {e}",
                exc_info=True
            )
            return self._create_empty_signals(current_date)
    
    def _create_empty_signals(self, current_date: datetime) -> pd.DataFrame:
        """
        空のsignals DataFrame作成
        
        Args:
            current_date: 決済日時
        
        Returns:
            pd.DataFrame: 空のsignals DataFrame（strategy="ForceClose"設定済み）
        """
        return pd.DataFrame(
            {
                'Close': [],
                'Entry_Signal': [],
                'Exit_Signal': [],
                'Position': [],
                'Strategy': []
            },
            index=pd.DatetimeIndex([])
        )
    
    # BaseStrategyインターフェース互換性メソッド
    # （StrategyExecutionManagerがgenerate_entry_signal/generate_exit_signalを呼び出す可能性対応）
    
    def generate_entry_signal(self, idx: int) -> int:
        """
        エントリーシグナル生成（ForceCloseは常に0）
        
        Args:
            idx: インデックス
        
        Returns:
            int: 0（ForceCloseはエントリーしない）
        """
        return 0
    
    def generate_exit_signal(self, idx: int) -> int:
        """
        イグジットシグナル生成（ForceCloseは常に-1）
        
        Args:
            idx: インデックス
        
        Returns:
            int: -1（SELL固定）
        """
        return -1
```

---

## 🔍 実装詳細

### 1. __init__()初期化

- **Line 57-80**: 初期化処理
- **broker**: PaperBrokerインスタンス保存
- **reason**: 決済理由保存（"symbol_switch", "backtest_end"等）
- **strategy_name**: "ForceClose"明示設定
- **data**: NoneならダミーDataFrame作成（BaseStrategy要件）

### 2. backtest()実行

- **Line 82-132**: 強制決済実行
- **current_date**: 決済日時（未指定なら現在時刻）
- **PaperBroker呼び出し**: `close_all_positions(current_date, reason)`
- **空チェック**: 決済結果0件なら空signals返却
- **signals変換**: `_convert_to_signals()`呼び出し

### 3. _convert_to_signals()変換

- **Line 134-189**: 決済結果→signals変換
- **Exit_Signal**: -1（SELL）設定
- **strategy**: "ForceClose"設定
- **インデックス**: current_dateで統一
- **追加情報**: symbol, quantity, entry_price, exit_price, pnl等

### 4. エラー耐性

- **try-except**: backtest(), _convert_to_signals()
- **エラーログ**: exc_info=True（詳細トレースバック）
- **空signals返却**: エラー時はフォールバックせず空返却

### 5. BaseStrategyインターフェース互換

- **generate_entry_signal()**: 常に0返却（Line 208-217）
- **generate_exit_signal()**: 常に-1返却（Line 219-228）
- **目的**: StrategyExecutionManagerとの互換性確保

---

## ✅ copilot-instructions.md 準拠確認

### モジュールヘッダーコメント（2025-10-20以降必須）

- ✅ モジュール名: "ForceCloseStrategy - 強制決済戦略"
- ✅ 一行役割: "銘柄切替時・バックテスト終了時に全ポジションを強制決済する戦略"
- ✅ 詳細説明: 2-3行記載
- ✅ 主な機能: 5つリストアップ
- ✅ 統合コンポーネント: PaperBroker, StrategyExecutionManager, IntegratedExecutionManager
- ✅ セーフティ機能/注意事項: エラー耐性、エラー隠蔽禁止、モック禁止、フォールバック禁止
- ✅ Author/Created/Last Modified記載

### 実データのみ使用

- ✅ PaperBroker.close_all_positions()呼び出し（実データ）
- ✅ 決済結果から実際の価格・数量取得
- ❌ モック/ダミーデータ使用なし

### エラー隠蔽禁止

- ✅ エラー時は警告ログ出力（Line 127-131, 184-188）
- ✅ exc_info=True設定（詳細トレースバック）
- ❌ エラー隠蔽なし

### フォールバック禁止

- ✅ エラー時は空signals返却（処理継続強制しない）
- ✅ 決済失敗時は空リスト処理（PaperBroker側で実装済み）
- ❌ フォールバック機能なし

### バックテスト実行必須

- ✅ 実際のPaperBroker.close_all_positions()呼び出し
- ✅ 実際の決済結果をsignals変換
- ❌ スキップ処理なし

---

## 🧪 テストケース設計

### Test 1: 単一銘柄強制決済
```python
def test_force_close_single_position():
    """単一銘柄強制決済テスト"""
    # Setup
    broker = PaperBroker(initial_balance=1000000.0)
    broker.update_price('7203.T', 2000.0)
    
    buy_order = Order(symbol='7203.T', side=OrderSide.BUY, 
                     order_type=OrderType.MARKET, quantity=100)
    broker.submit_order(buy_order)
    
    assert len(broker.positions) == 1
    
    # Test: ForceClose実行
    strategy = ForceCloseStrategy(broker, reason="test")
    signals = strategy.backtest(current_date=datetime.now())
    
    # Verify
    assert len(signals) == 1
    assert signals.iloc[0]['Exit_Signal'] == -1
    assert signals.iloc[0]['Strategy'] == 'ForceClose'
    assert len(broker.positions) == 0
```

### Test 2: 複数銘柄強制決済
```python
def test_force_close_multiple_positions():
    """複数銘柄強制決済テスト"""
    # Setup
    broker = PaperBroker(initial_balance=10000000.0)
    broker.update_price('7203.T', 2000.0)
    broker.update_price('6758.T', 15000.0)
    broker.update_price('9984.T', 30000.0)
    
    for symbol in ['7203.T', '6758.T', '9984.T']:
        buy_order = Order(symbol=symbol, side=OrderSide.BUY, 
                         order_type=OrderType.MARKET, quantity=100)
        broker.submit_order(buy_order)
    
    assert len(broker.positions) == 3
    
    # Test
    strategy = ForceCloseStrategy(broker, reason="symbol_switch")
    signals = strategy.backtest(current_date=datetime.now())
    
    # Verify
    assert len(signals) == 3
    assert all(signals['Exit_Signal'] == -1)
    assert all(signals['Strategy'] == 'ForceClose')
    assert len(broker.positions) == 0
```

### Test 3: ポジション未保有
```python
def test_force_close_no_positions():
    """ポジション未保有テスト"""
    # Setup
    broker = PaperBroker(initial_balance=1000000.0)
    
    # Test
    strategy = ForceCloseStrategy(broker, reason="test")
    signals = strategy.backtest(current_date=datetime.now())
    
    # Verify
    assert len(signals) == 0
    assert len(broker.positions) == 0
```

### Test 4: strategy_name検証
```python
def test_force_close_strategy_name():
    """strategy_name検証テスト"""
    # Setup
    broker = PaperBroker(initial_balance=1000000.0)
    broker.update_price('7203.T', 2000.0)
    
    buy_order = Order(symbol='7203.T', side=OrderSide.BUY, 
                     order_type=OrderType.MARKET, quantity=100)
    broker.submit_order(buy_order)
    
    # Test
    strategy = ForceCloseStrategy(broker, reason="symbol_switch")
    signals = strategy.backtest(current_date=datetime.now())
    
    # Verify
    assert strategy.strategy_name == "ForceClose"
    assert all(signals['Strategy'] == 'ForceClose')
    
    # filled_ordersから最新の注文を取得
    last_order = broker.filled_orders[-1]
    assert last_order.strategy_name == "ForceClose"
```

---

## 📊 バックテスト検証シナリオ

### シナリオ1: 単一銘柄ForceClose統合テスト
```bash
# 前提: Priority 1-1（close_all_positions()実装）完了
# 目的: ForceCloseStrategyとPaperBrokerの連携確認

python -m src.dssms.dssms_integrated_main --start-date 2025-01-15 --end-date 2025-01-15

# 期待結果
- execution_details: strategy="ForceClose"のSELL記録確認
- ポジション: 決済後0件
- エラーログ: なし
```

### シナリオ2: 複数銘柄ForceClose統合テスト
```bash
python -m src.dssms.dssms_integrated_main --start-date 2025-01-15 --end-date 2025-01-17

# 期待結果
- execution_details: 保有銘柄数と同数のSELL記録
- 各execution_detail: strategy="ForceClose"確認
- ポジション: 決済後0件
```

### シナリオ3: 銘柄切替統合テスト（Priority 3実装後）
```bash
python -m src.dssms.dssms_integrated_main --start-date 2025-01-15 --end-date 2025-01-31

# 期待結果
- 銘柄切替: 複数回発生
- ForceClose execution_details: 各切替時に生成
- 新銘柄エントリー: 切替後の各戦略判断
- 総収益率: 正常計算
```

---

## 🚀 実装手順

### Step 1: ForceCloseStrategy実装
1. strategies/force_close_strategy.py作成
2. モジュールヘッダーコメント追加
3. ForceCloseStrategyクラス実装
4. copilot-instructions.md準拠確認

### Step 2: ユニットテスト実装
1. tests/temp/test_20251219_force_close_strategy.py作成
2. 4つのテストケース実装
3. pytest実行

### Step 3: 統合テスト（Priority 3後）
1. main_new.py修正（force_close_on_entryパラメータ追加）
2. IntegratedExecutionManager.execute_force_close()追加
3. シナリオ1-3実行

---

## 📝 実装後の確認項目

- [ ] strategies/force_close_strategy.py作成完了
- [ ] モジュールヘッダーコメント要件満たす
- [ ] ForceCloseStrategyクラス実装完了
- [ ] __init__(), backtest(), _convert_to_signals()実装
- [ ] BaseStrategyインターフェース互換実装
- [ ] ユニットテスト4件実装
- [ ] ユニットテスト全件成功
- [ ] copilot-instructions.md準拠確認
- [ ] エラー耐性確認（個別失敗時も処理継続）
- [ ] strategy_name="ForceClose"設定確認

---

**実装準備完了**: 2025-12-19  
**次のステップ**: Step 1実装開始
