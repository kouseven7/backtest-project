# Cycle 10-9: 銘柄切替時の強制決済処理調査報告

**調査日**: 2026-02-07  
**調査者**: Sprint 1 Task 1-12  
**関連Issue**: Cycle 10-8修正後もエグジット条件が発動しない問題

---

## 📋 調査サマリー

**結論**: 銘柄切替時の強制決済処理が**完全に削除**されており、ポジションが決済されない。

---

## 🔍 調査結果詳細

### 1. 銘柄切替の処理フロー

#### **ファイル**: `src/dssms/dssms_integrated_main.py`

#### **Line 895-970: `_process_daily_trading()`**

```python
# Line 895: 銘柄切替の評価と実行
switch_result = self._evaluate_and_execute_switch(selected_symbol, target_date)

# Line 897-902: switch_history への記録
if switch_result.get('switch_executed', False):
    daily_result['switch_executed'] = True
    daily_result['symbol'] = self.current_symbol  # 切替後の銘柄
    self.switch_history.append(switch_result)
    
    # Line 903-912: execution_details の収集（削除済みのため0件）
    if 'execution_details' in switch_result:
        # この条件は常にFalse（後述）
        daily_result['execution_details'].extend(switch_result['execution_details'])

# Line 915-931: マルチ戦略実行
switch_executed = switch_result.get('switch_executed', False)

if switch_executed:
    # Line 929-931: force_closeフラグ設定
    if switch_executed and self.current_position:
        self.current_position['force_close'] = True
    
    # Line 933-937: 日次取引モード実行
    strategy_result = self._execute_multi_strategies_daily(
        target_date,
        self.current_symbol,  # ← 切替後の銘柄（6645）
        stock_data
    )
```

**問題箇所**:
- ❌ Line 931で`force_close=True`を設定するだけ
- ❌ 実際の決済処理（SELL注文）が実行されない
- ❌ `_execute_multi_strategies_daily()`は切替後の銘柄（6645）のデータで判定

---

### 2. `_evaluate_and_execute_switch()`の実装

#### **Line 1863-1994: 銘柄切替の評価と実行**

```python
def _evaluate_and_execute_switch(self, selected_symbol: str, 
                               target_date: datetime) -> Dict[str, Any]:
    # ... ポジション保護ロジック（Cycle 4-A、Cycle 10-8修正済み）...
    
    # Line 1966-1978: 削除コメント
    if should_switch:
        # 削除: 銘柄切替時の取引実行処理(_close_position, _open_position呼び出し)
        # 削除: execution_type='switch'設定
        # 理由: DSSMSは銘柄選択のみ担当,取引実行はmain_new.py(PaperBroker)が担当
        # 影響: switch関連のexecution_detailsが0件になる(意図通り)
        
        # 削除理由: DSSMSが取引実行しないため,コスト計算は無意味
        # 影響: portfolio_value更新なし,switch_result簡略化
        
        switch_result.update({
            'switch_executed': True,
            'reason': switch_evaluation.get('reason', 'dss_optimization'),
            'executed_date': target_date
        })
        
        # 現在銘柄更新
        self.current_symbol = selected_symbol
        
        # 切替履歴記録
        self.switch_manager.record_switch_executed(switch_result)
        
        self.logger.info(f"銘柄切替実行: {switch_result['from_symbol']} -> {selected_symbol}")
    
    return switch_result  # ← execution_details なし
```

**問題**:
- ❌ **Line 1966-1978**: 決済処理が完全に削除されている
- ❌ `switch_result`に`execution_details`が含まれない
- ❌ `self.current_symbol`を更新するだけで終了

---

### 3. `_execute_multi_strategies_daily()`の動作

#### **Line 2380-2420: ポジション状態判定**

```python
# Line 2391-2394: 銘柄切替時のforce_close設定
if self.current_position['symbol'] != symbol:
    existing_position = {
        'entry_idx': self.current_position.get('entry_idx', 0),
        'quantity': self.current_position.get('shares', 0),
        'entry_price': self.current_position.get('entry_price', 0.0),
        'entry_date': self.current_position.get('entry_date', None),
        'strategy': self.current_position.get('strategy', best_strategy_name),
        'force_close': True,  # 銘柄切替フラグ
        'entry_symbol': self.current_position.get('symbol', '')  # エントリー銘柄
    }
    self.logger.warning(f"[PHASE3-C-B1] 銘柄切替: {self.current_position['symbol']} → {symbol}, force_close=True")

# Line 2401-2420: entry_symbol_data取得（Cycle 7実装）
entry_symbol_data = None
if existing_position and existing_position.get('force_close', False):
    entry_symbol = existing_position.get('entry_symbol', '')
    if entry_symbol:
        entry_symbol_data, _ = self._get_symbol_data(entry_symbol, adjusted_target_date)
```

**期待される動作**:
- `force_close=True`で`entry_symbol_data`（3105のデータ）を取得
- 戦略が`entry_symbol_data`で決済処理を実行

**実際の動作**:
- 戦略は`stock_data`（6645のデータ）でエントリー判定
- エントリー条件を満たさない → `action='hold'`
- 3105のポジションが残り続ける

---

### 4. 戦略結果の処理

#### **Line 2483-2488: SELLアクション処理**

```python
elif result['action'] == 'sell':
    # エグジット: current_positionクリア
    self.logger.info(f"[PHASE3-C-B1] ポジション更新（sell）: current_position={self.current_position} → None")
    self.current_position = None
    self.last_entry_price = None
```

**問題**:
- ✅ SELL処理自体は実装されている
- ❌ しかし、戦略が`action='sell'`を返していない

---

## 🧩 データフロー図

```
2024-01-04: 3105でエントリー
    ↓
    current_position = {'symbol': '3105', 'entry_price': 1156.15, ...}
    ↓
2024-01-16: 6645への切替を決定
    ↓
    _evaluate_and_execute_switch('6645', 2024-01-16)
    ↓
    should_switch = True
    ↓
    self.current_symbol = '6645'  ← 更新
    switch_history.append({...})  ← 記録
    ❌ 決済処理なし（削除済み）
    ↓
    return {'switch_executed': True, ...}  ← execution_details なし
    ↓
    _process_daily_trading()
    ↓
    self.current_position['force_close'] = True
    ↓
    _execute_multi_strategies_daily(target_date, '6645', stock_data)
    ↓
    existing_position = {
        'symbol': '3105',
        'force_close': True,
        'entry_symbol': '3105'
    }
    entry_symbol_data = データ取得('3105')  ← Cycle 7実装
    ↓
    strategy.backtest_daily(..., existing_position=existing_position, entry_symbol_data=entry_symbol_data)
    ↓
    戦略の動作:
    - force_close フラグを認識
    - entry_symbol_data で決済価格を取得すべき
    - しかし、実際は stock_data（6645）でエントリー判定
    ↓
    result = {'action': 'hold', ...}  ← エントリー条件を満たさない
    ↓
    current_position = {'symbol': '3105', ...}  ← 残り続ける（❌）
    ↓
2024-12-31: 期間終了の強制決済
    ↓
    backtest終了処理（Line 670-850付近）
    ↓
    action = 'sell', is_forced_exit = True
```

---

## 🚨 問題の整理

### パターンC: 決済処理が存在しない（該当）

```python
# _evaluate_and_execute_switch() Line 1966-1978
if should_switch:
    # ❌ 決済処理が削除されている
    # self._close_position(...)  ← 削除済み
    # self._open_position(...)   ← 削除済み
    
    # ✅ switch_historyに記録するだけで終了
    self.current_symbol = selected_symbol
    self.switch_manager.record_switch_executed(switch_result)
    
    return switch_result  # ← execution_details なし
```

### 期待されるフロー vs 実際のフロー

| 期待 | 実際 |
|------|------|
| 1. 銘柄切替判定（3105 → 6645） | ✅ Line 1951: should_switch = True |
| 2. 3105のポジション決済（SELL） | ❌ 処理なし（削除済み） |
| 3. execution_details に SELL 記録 | ❌ 記録なし |
| 4. 6645で新規エントリー判定 | ⚠️ エントリー条件を満たさず HOLD |
| 5. all_transactions.csv に記録 | ❌ SELLレコードなし |

---

## 🔧 修正が必要な箇所

### 修正対象: `_evaluate_and_execute_switch()`

**ファイル**: `src/dssms/dssms_integrated_main.py`  
**Line**: 1966-1978  
**問題**: 決済処理が削除されている

### 修正案

#### Option A: 強制決済処理を復元

```python
if should_switch:
    execution_details = []
    
    # 既存ポジションの強制決済
    if self.current_position is not None:
        sell_price = ...  # 元の銘柄（3105）の当日終値
        sell_shares = self.current_position.get('shares', 0)
        
        # SELL execution_detail 生成
        sell_detail = {
            'timestamp': target_date.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': self.current_position['symbol'],  # 3105
            'action': 'SELL',
            'price': sell_price,
            'shares': sell_shares,
            'strategy': self.current_position.get('strategy', 'DSSMS'),
            'reason': 'symbol_switch_forced_exit',
            'status': 'executed'
        }
        execution_details.append(sell_detail)
        
        # 現金残高更新
        self.cash_balance += sell_price * sell_shares
        
        # ポジションクリア
        self.current_position = None
        self.last_entry_price = None
    
    switch_result.update({
        'switch_executed': True,
        'reason': switch_evaluation.get('reason', 'dss_optimization'),
        'executed_date': target_date,
        'execution_details': execution_details  # ← 追加
    })
    
    # 現在銘柄更新
    self.current_symbol = selected_symbol
    
    # 切替履歴記録
    self.switch_manager.record_switch_executed(switch_result)
```

---

## 📊 影響範囲

### 現在の動作（修正前）
- 取引数: **1件**（期間終了の強制決済のみ）
- 銘柄切替: 32回記録されるが、決済処理なし
- ポジション: 3105を1年間保有（-22%損失）

### 期待される動作（修正後）
- 取引数: **30〜60件**（銘柄切替ごとに決済＋新規エントリー）
- 銘柄切替: 32回 → 各回でSELLアクション実行
- ポジション: 平均保有期間 10〜15日

---

## ✅ 次のステップ

1. **Option A実装**: `_evaluate_and_execute_switch()`に強制決済処理を追加
2. **テスト実行**: 1週間テスト（2024-01-17 to 2024-01-24）
3. **検証**: 銘柄切替時のSELLアクションが記録されるか確認
4. **完全テスト**: 2024年全体（262日）で動作確認

---

## 📝 備考

- Cycle 7実装により、`entry_symbol_data`取得は完了している
- しかし、戦略が`force_close`時に決済判定を返していない可能性あり
- 根本的には、DSSMS側で決済処理を実行すべき設計

---

**報告者**: GitHub Copilot  
**承認**: ユーザー確認後
