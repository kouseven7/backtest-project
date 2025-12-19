# PaperBroker.close_all_positions() 実装仕様書

**Phase**: Phase 2-2実装準備  
**Priority**: Priority 1  
**作成日**: 2025-12-19  
**Status**: 実装準備完了

---

## 📋 概要

PaperBrokerに全ポジション強制決済機能を追加。
銘柄切替時やバックテスト終了時に既存ポジションを一括決済する。

---

## 🎯 メソッド仕様

### メソッドシグネチャ

```python
def close_all_positions(
    self, 
    current_date: datetime,
    reason: str = "symbol_switch"
) -> List[Dict[str, Any]]:
    """
    全ポジション強制決済
    
    Args:
        current_date (datetime): 決済日時
        reason (str): 決済理由
            - "symbol_switch": 銘柄切替時（デフォルト）
            - "backtest_end": バックテスト終了時
            - その他カスタム理由
    
    Returns:
        List[Dict[str, Any]]: 決済結果のリスト
            各要素の構造:
            {
                'symbol': str,              # 銘柄コード
                'action': str,              # "SELL"固定
                'quantity': int,            # 決済数量
                'entry_price': float,       # エントリー価格
                'exit_price': float,        # イグジット価格
                'entry_time': datetime,     # エントリー日時
                'exit_time': datetime,      # イグジット日時（current_date）
                'pnl': float,               # 損益
                'commission': float,        # 手数料
                'slippage': float,          # スリッページコスト
                'order_id': str,            # 注文ID
                'reason': str               # 決済理由
            }
    
    Raises:
        なし（個別エラーは警告ログ出力、処理継続）
    
    Note:
        - 個別銘柄の決済失敗時も他の銘柄の決済を継続
        - エラー時は警告ログ出力
        - 成功した決済結果のみ返却
        - ポジション未保有時は空リスト返却
    
    copilot-instructions.md準拠:
        - 実データのみ使用（モック/ダミー禁止）
        - エラー隠蔽禁止（警告ログ出力）
        - フォールバック禁止
    """
```

---

## 🔧 実装コード

### 完全実装（paper_broker.py Line 611以降に追加）

```python
def close_all_positions(
    self, 
    current_date: datetime,
    reason: str = "symbol_switch"
) -> List[Dict[str, Any]]:
    """
    全ポジション強制決済
    
    Args:
        current_date: 決済日時
        reason: 決済理由（"symbol_switch", "backtest_end"等）
    
    Returns:
        List[Dict]: 決済結果のリスト
    
    copilot-instructions.md準拠:
        - 実データのみ使用（モック/ダミー禁止）
        - エラー隠蔽禁止（警告ログ出力）
        - フォールバック禁止
    """
    results = []
    
    # ポジション未保有チェック
    if not self.positions:
        self.logger.info(f"[FORCE_CLOSE] No positions to close. Reason: {reason}")
        return results
    
    # ポジションコピー（ループ中変更回避）
    positions_to_close = list(self.positions.items())
    
    self.logger.info(
        f"[FORCE_CLOSE] Closing all positions: {len(positions_to_close)} positions | "
        f"Reason: {reason} | Date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    # 個別決済実行
    success_count = 0
    failed_symbols = []
    
    for symbol, position_data in positions_to_close:
        try:
            # エントリー情報取得
            entry_price = position_data['entry_price']
            entry_time = position_data['entry_time']
            quantity = position_data['quantity']
            
            # イグジット価格取得
            exit_price = self.get_current_price(symbol)
            
            # SELL注文作成
            sell_order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                status=OrderStatus.PENDING,
                created_at=current_date,
                strategy_name="ForceClose"  # 明示的に設定
            )
            
            # 注文実行
            success = self.submit_order(sell_order)
            
            if success and sell_order.status == OrderStatus.FILLED:
                # PnL計算
                pnl = (sell_order.filled_price - entry_price) * quantity
                
                # 決済結果記録
                result = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'exit_price': sell_order.filled_price,
                    'entry_time': entry_time,
                    'exit_time': current_date,
                    'pnl': pnl,
                    'commission': sell_order.commission,
                    'slippage': sell_order.slippage,
                    'order_id': sell_order.id,
                    'reason': reason
                }
                results.append(result)
                success_count += 1
                
                self.logger.info(
                    f"[FORCE_CLOSE] Success: {symbol} | "
                    f"Qty: {quantity} | "
                    f"Entry: {entry_price:.2f} | "
                    f"Exit: {sell_order.filled_price:.2f} | "
                    f"PnL: {pnl:+.2f}"
                )
            else:
                # 決済失敗
                failed_symbols.append(symbol)
                self.logger.warning(
                    f"[FORCE_CLOSE] Failed to close position: {symbol} | "
                    f"Order status: {sell_order.status.value} | "
                    f"Quantity: {quantity}"
                )
                
        except Exception as e:
            # 個別エラー処理（処理継続）
            failed_symbols.append(symbol)
            self.logger.error(
                f"[FORCE_CLOSE] Error closing position: {symbol} | "
                f"Error: {e} | "
                f"Continuing with other positions..."
            )
            continue
    
    # 最終サマリーログ
    self.logger.info(
        f"[FORCE_CLOSE] Completed: "
        f"Success: {success_count}/{len(positions_to_close)} | "
        f"Failed: {len(failed_symbols)} | "
        f"Reason: {reason}"
    )
    
    if failed_symbols:
        self.logger.warning(
            f"[FORCE_CLOSE] Failed symbols: {', '.join(failed_symbols)}"
        )
    
    return results
```

---

## 🔍 実装詳細

### 1. ポジション未保有チェック
- Line 25-27: ポジション未保有時は空リスト返却
- 早期リターンでパフォーマンス最適化

### 2. ポジションコピー（ループ中変更回避）
- Line 30: `list(self.positions.items())`でコピー作成
- ループ内でpositions辞書が変更されても安全

### 3. 個別SELL注文実行
- Line 43-72: 各ポジションに対してSELL注文生成
- Line 58-63: Order作成（strategy_name="ForceClose"明示）
- Line 66: submit_order()呼び出し

### 4. エラー耐性
- Line 69-82: 個別失敗時も継続（`continue`）
- try-exceptブロックで個別エラー捕捉
- 失敗銘柄をfailed_symbolsに記録

### 5. 詳細ログ出力
- Line 32-35: 開始ログ（ポジション数、理由、日時）
- Line 83-94: 成功時ログ（銘柄、数量、価格、PnL）
- Line 97-102: 失敗時警告
- Line 107-113: 最終サマリーログ

### 6. 決済結果返却
- Line 74-87: 結果辞書作成
- 12フィールド含む詳細情報
- 成功した決済のみ返却

---

## ✅ copilot-instructions.md 準拠確認

### 実データのみ使用
- ✅ get_current_price()で実際の価格取得（Line 56）
- ✅ positionsから実際のポジションデータ取得（Line 47-49）
- ✅ submit_order()で実際の注文実行（Line 66）
- ❌ モック/ダミーデータ使用なし

### エラー隠蔽禁止
- ✅ 個別失敗時は警告ログ出力（Line 97-102）
- ✅ 例外時はエラーログ出力（Line 107-113）
- ✅ 失敗銘柄を明示（Line 115-118）
- ❌ エラー隠蔽なし

### フォールバック禁止
- ✅ 決済失敗時に代替処理なし
- ✅ テスト継続のための強制成功なし
- ✅ 実際の実行結果のみ返却
- ❌ フォールバック機能なし

### バックテスト実行必須
- ✅ 実際のsubmit_order()呼び出し
- ✅ 実際のポジション操作
- ❌ スキップ処理なし

---

## 🧪 テストケース設計

### Test 1: 全ポジション決済成功
```python
def test_close_all_positions_success():
    """全ポジション決済成功テスト"""
    # Setup
    broker = PaperBroker(initial_balance=1000000.0)
    broker.update_price('7203.T', 2000.0)
    broker.update_price('6758.T', 15000.0)
    broker.update_price('9984.T', 30000.0)
    
    # 3銘柄購入
    for symbol in ['7203.T', '6758.T', '9984.T']:
        buy_order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        broker.submit_order(buy_order)
    
    assert len(broker.positions) == 3
    
    # Test: 全決済
    results = broker.close_all_positions(
        current_date=datetime.now(),
        reason="test"
    )
    
    # Verify
    assert len(results) == 3
    assert len(broker.positions) == 0
    for result in results:
        assert result['action'] == 'SELL'
        assert result['reason'] == 'test'
        assert result['quantity'] == 100
```

### Test 2: 一部決済失敗
```python
def test_close_all_positions_partial_failure():
    """一部決済失敗テスト"""
    # Setup
    broker = PaperBroker(initial_balance=1000000.0)
    broker.update_price('7203.T', 2000.0)
    broker.update_price('6758.T', 15000.0)
    
    # 2銘柄購入
    for symbol in ['7203.T', '6758.T']:
        buy_order = Order(symbol=symbol, side=OrderSide.BUY, 
                         order_type=OrderType.MARKET, quantity=100)
        broker.submit_order(buy_order)
    
    # 1銘柄の価格データ削除（決済失敗をシミュレート）
    del broker.current_prices['6758.T']
    
    # Test
    results = broker.close_all_positions(
        current_date=datetime.now(),
        reason="test"
    )
    
    # Verify: 1銘柄のみ決済成功
    assert len(results) >= 1  # 少なくとも1銘柄成功
    assert any(r['symbol'] == '7203.T' for r in results)
```

### Test 3: ポジション未保有
```python
def test_close_all_positions_empty():
    """ポジション未保有テスト"""
    # Setup
    broker = PaperBroker(initial_balance=1000000.0)
    
    # Test
    results = broker.close_all_positions(
        current_date=datetime.now(),
        reason="test"
    )
    
    # Verify
    assert len(results) == 0
    assert len(broker.positions) == 0
```

### Test 4: strategy_name検証
```python
def test_close_all_positions_strategy_name():
    """strategy_name検証テスト"""
    # Setup
    broker = PaperBroker(initial_balance=1000000.0)
    broker.update_price('7203.T', 2000.0)
    
    buy_order = Order(symbol='7203.T', side=OrderSide.BUY, 
                     order_type=OrderType.MARKET, quantity=100)
    broker.submit_order(buy_order)
    
    # Test
    results = broker.close_all_positions(
        current_date=datetime.now(),
        reason="symbol_switch"
    )
    
    # Verify
    assert len(results) == 1
    # filled_ordersから最新の注文を取得
    last_order = broker.filled_orders[-1]
    assert last_order.strategy_name == "ForceClose"
```

---

## 📊 バックテスト検証シナリオ

### シナリオ1: 単一銘柄ForceClose
```bash
# コマンド
python -m src.dssms.dssms_integrated_main --start-date 2025-01-15 --end-date 2025-01-15

# 期待結果
- 銘柄: 1件保有時にclose_all_positions()実行
- execution_details: strategy="ForceClose"のSELL記録1件
- ポジション: 決済後0件
- 総収益率: PnL反映
```

### シナリオ2: 複数銘柄ForceClose
```bash
# コマンド
python -m src.dssms.dssms_integrated_main --start-date 2025-01-15 --end-date 2025-01-17

# 期待結果
- 銘柄: 複数保有時にclose_all_positions()実行
- execution_details: 保有銘柄数と同数のSELL記録
- ポジション: 決済後0件
- エラーログ: なし
```

### シナリオ3: 銘柄切替統合テスト
```bash
# コマンド
python -m src.dssms.dssms_integrated_main --start-date 2025-01-15 --end-date 2025-01-31

# 期待結果
- 銘柄切替: 複数回発生
- ForceClose execution_details: 各切替時に生成
- 新銘柄エントリー: 切替後の各戦略判断
- 総収益率: 正常計算
- エラー: なし
```

---

## 🚀 実装手順

### Step 1: paper_broker.pyに実装追加
1. Line 611以降にclose_all_positions()メソッド追加
2. uuidインポート確認（Line 1-10）
3. datetimeインポート確認（Line 1-10）

### Step 2: ユニットテスト実装
1. tests/temp/test_20251219_close_all_positions.py作成
2. 4つのテストケース実装
3. pytest実行

### Step 3: バックテスト検証
1. シナリオ1実行（単一銘柄）
2. execution_details確認
3. ポジション状態確認

### Step 4: 統合テスト
1. シナリオ3実行（銘柄切替）
2. 総収益率検証
3. エラーログ確認

---

## 📝 実装後の確認項目

- [ ] close_all_positions()メソッド追加完了
- [ ] uuidインポート確認
- [ ] datetimeインポート確認
- [ ] ユニットテスト4件実装
- [ ] ユニットテスト全件成功
- [ ] バックテスト検証（シナリオ1）成功
- [ ] execution_details生成確認
- [ ] strategy_name="ForceClose"確認
- [ ] ポジション決済確認
- [ ] エラーログ確認（警告のみ、エラーなし）
- [ ] copilot-instructions.md準拠確認

---

**実装準備完了**: 2025-12-19  
**次のステップ**: Step 1実装開始
